"""RRG chart for stocks vs benchmark (selectable universe via CLI/env).

Universe: universes/*.py (quality, n500, bse_largemidcap, nifty_largemidcap) via stock_rrg_universe.py.
Analysis: default 3-month lookback (13 weekly points, 10w rolling window); optional --period 6m.
Side panel: top 10 (Was vs Now) + swing trading cheat sheet.
Downloads extra history for RRG warmup (~22w for 3m / ~30w for 6m) — not plotted on the slider.
Not a ranker: interactive quadrant plot with tail/date sliders and sector table.

Examples:
    python momentum/stock/RRGIndicatorStocks3m.py
    python momentum/stock/RRGIndicatorStocks.py
    python momentum/stock/RRGIndicatorStocks.py --period 6m
    python momentum/stock/RRGIndicatorStocks.py --universe quality --period 3m --window 10
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.rrg_app import RrgAppConfig, run_rrg_app
from momentum.rrg_core import RRG_WINDOW_DEFAULT, RRG_WINDOW_ETF
from momentum.rrg_swing_cheat_sheet import STOCK_SWING_CHEAT_SHEET
from momentum.stock.stock_rrg_universe import (
    RRG_BENCHMARK_NSE,
    RRG_DEFAULT_VISIBLE_IDS,
    RRG_LOAD_NSE_INDEX_NAMES,
    RRG_LOAD_NSE_STOCK_SYMBOLS,
    RRG_ROW_BY_ID,
    RRG_ROWS,
    RRG_STOCK_ROW_IDS,
    active_universe_key,
    active_universe_module,
    row_display_label,
    row_kind,
    row_ref_label,
    use_universe_key,
)
from momentum.stock.universes import BY_KEY, DEFAULT_KEY, ENV_UNIVERSE_KEY
from utils.nse_bhavcopy import (
    load_nse_cm_histories_range,
    load_nse_equity_weekly_histories,
    load_nse_index_weekly_histories,
    load_nse_index_weekly_histories_range,
)

ENV_STOCK_PERIOD = "RRG_STOCK_PERIOD"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock RRG vs benchmark")
    parser.add_argument(
        "--universe",
        "-u",
        choices=sorted(BY_KEY),
        help=f"Universe module key (default: env {ENV_UNIVERSE_KEY} or {DEFAULT_KEY})",
    )
    parser.add_argument(
        "--period",
        "-p",
        choices=("3m", "6m"),
        default=os.environ.get(ENV_STOCK_PERIOD, "3m"),
        help="Analysis lookback on Date slider (default: 3m for tactical stock rotation)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        choices=(10, 14),
        default=None,
        help="RRG rolling window in weeks (default: 10 for 3m, 14 for 6m)",
    )
    return parser.parse_args()


def _resolve_rrg_window(period: str, window_arg: int | None) -> int:
    if window_arg is not None:
        return window_arg
    return RRG_WINDOW_ETF if period == "3m" else RRG_WINDOW_DEFAULT


def _resolve_row_id(requested: str) -> str | None:
    text = requested.strip().upper().replace(".NS", "")
    if not text:
        return None
    if text in RRG_ROW_BY_ID:
        return text
    for row_id, row in RRG_ROW_BY_ID.items():
        if text == row_id.upper() or requested.strip().lower() == row.label.lower():
            return row_id
    return None


def _load_all_histories(
    period: str, min_weekly_points: int, rrg_window: int, freq: str = "week"
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    print(f"Loading NSE index EOD (ind_close_all) for RRG benchmark ({freq})...")
    index_batch = load_nse_index_weekly_histories(
        RRG_LOAD_NSE_INDEX_NAMES,
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
        freq=freq,
    )
    out[RRG_BENCHMARK_NSE] = index_batch.get(RRG_BENCHMARK_NSE, pd.Series(dtype=float))

    print("Loading NSE stock EOD (CM bhavcopy) for RRG...")
    stock_batch = load_nse_equity_weekly_histories(
        RRG_LOAD_NSE_STOCK_SYMBOLS,
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
        freq=freq,
    )
    for sym in RRG_STOCK_ROW_IDS:
        out[sym] = stock_batch.get(sym, pd.Series(dtype=float))
    return out


def _load_all_histories_range(
    start_date: date,
    end_date: date,
    min_weekly_points: int,
    rrg_window: int,
    freq: str = "week",
) -> dict[str, pd.Series]:
    """Load RRG histories for an explicit calendar range (backtests)."""
    out: dict[str, pd.Series] = {}
    print(
        f"Loading NSE index EOD (ind_close_all) for RRG ({freq}) "
        f"{start_date:%Y-%m-%d} .. {end_date:%Y-%m-%d}..."
    )
    index_batch = load_nse_index_weekly_histories_range(
        RRG_LOAD_NSE_INDEX_NAMES,
        start_date,
        end_date,
        min_points=min_weekly_points,
        freq=freq,
    )
    out[RRG_BENCHMARK_NSE] = index_batch.get(RRG_BENCHMARK_NSE, pd.Series(dtype=float))

    if RRG_LOAD_NSE_STOCK_SYMBOLS:
        print(
            f"Loading NSE stock EOD (CM bhavcopy) for RRG "
            f"{start_date:%Y-%m-%d} .. {end_date:%Y-%m-%d}..."
        )
        stock_batch = load_nse_cm_histories_range(
            RRG_LOAD_NSE_STOCK_SYMBOLS,
            start_date,
            end_date,
            min_points=min_weekly_points,
            quiet=True,
            asset_label="equity symbol",
            freq=freq,
        )
        for sym in RRG_LOAD_NSE_STOCK_SYMBOLS:
            out[sym] = stock_batch.get(sym, pd.Series(dtype=float))
    return out


def _load_row_history(
    row_id: str,
    kind: str,
    period: str,
    min_weekly_points: int,
    rrg_window: int,
    freq: str = "week",
) -> pd.Series:
    if kind == "index":
        return load_nse_index_weekly_histories(
            [row_id],
            period=period,
            min_points=min_weekly_points,
            rrg_window=rrg_window,
            freq=freq,
        ).get(row_id, pd.Series(dtype=float))
    return load_nse_equity_weekly_histories(
        [row_id],
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
        freq=freq,
    ).get(row_id, pd.Series(dtype=float))


def _count_summary(kind_list: list[str]) -> str:
    n_stock = sum(1 for k in kind_list if k == "stock")
    return f"{n_stock} stocks"


def _build_config(analysis_period: str, rrg_window: int) -> RrgAppConfig:
    mod = active_universe_module()
    return RrgAppConfig(
        window_title=f"RRG — {mod.LABEL} ({analysis_period} EOD)",
        benchmark_nse=RRG_BENCHMARK_NSE,
        rows=RRG_ROWS,
        row_by_id=RRG_ROW_BY_ID,
        default_visible_ids=RRG_DEFAULT_VISIBLE_IDS,
        ref_column_header="Industry",
        name_column_header="Symbol",
        defaults_checkbox_text="Default stocks",
        hover_ref_prefix="Industry",
        universe_summary=(
            f"RRG universe: {active_universe_key()} ({mod.LABEL}, "
            f"{len(RRG_LOAD_NSE_STOCK_SYMBOLS)} names) — {mod.DESCRIPTION}; "
            f"benchmark {RRG_BENCHMARK_NSE}; analysis {analysis_period}"
        ),
        row_ref_label=row_ref_label,
        row_display_label=row_display_label,
        row_kind=row_kind,
        resolve_row_id=_resolve_row_id,
        load_all_histories=_load_all_histories,
        load_row_history=_load_row_history,
        count_summary=_count_summary,
        analysis_period=analysis_period,
        rrg_window=rrg_window,
        default_tail=1,
        top_movers_panel=True,
        top_movers_count=10,
        top_movers_title="Portfolio — Was vs Now",
        side_cheat_sheet=STOCK_SWING_CHEAT_SHEET,
        etf_table_extras=True,
        preview_today_picks=True,
        etf_recommend_profile="stock",
        pick_strategy="leading_improved",
        etf_recommend_count=7,
        backtest_enabled=True,
        backtest_profile="stock",
        backtest_universe_key=active_universe_key(),
    )


def main() -> None:
    args = _parse_args()
    key = args.universe or os.environ.get(ENV_UNIVERSE_KEY, DEFAULT_KEY)
    use_universe_key(key)
    rrg_window = _resolve_rrg_window(args.period, args.window)
    run_rrg_app(_build_config(args.period, rrg_window))


if __name__ == "__main__":
    main()
