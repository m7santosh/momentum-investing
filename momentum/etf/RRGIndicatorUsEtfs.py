"""RRG chart for US ETFs vs S&P 500 (JdK RS ratio vs momentum).

Universe: universes/us.py (via us_rrg_universe.py) — Yahoo Finance weekly EOD.
Analysis: default 3-month lookback (13 weekly points, 10w rolling window); optional --period 6m.
Side panel: top 10 (Was vs Now) + swing trading cheat sheet.
Downloads extra history for RRG warmup (~22w for 3m / ~30w for 6m) — not plotted on the slider.
Not a ranker: interactive quadrant plot with tail/date sliders and sector table.

vs momentum ETF rankers (Excel output):
    momentum_us_etfs.py — abs return ranks only.
    momentum_us_rs_etfs.py — abs + RS vs ^GSPC blended ranks (swing).
    momentum_us_rs_etfs_adaptive.py — RS-only, short horizons.

vs India ETF RRG:
    RRGIndicatorEtfs.py — NSE indices + ETFs (bhavcopy / ind_close_all).

Examples:
    python momentum/etf/RRGIndicatorUsEtfs3m.py
    python momentum/etf/RRGIndicatorUsEtfs.py
    python momentum/etf/RRGIndicatorUsEtfs.py --period 6m
    python momentum/etf/RRGIndicatorUsEtfs.py --period 3m --window 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.us_rrg_universe import (  # noqa: E402
    RRG_BENCHMARK_YAHOO,
    RRG_DEFAULT_VISIBLE_IDS,
    RRG_ETF_LABELS,
    RRG_ETF_ROW_IDS,
    RRG_LOAD_YAHOO_TICKERS,
    RRG_ROW_BY_ID,
    RRG_ROWS,
    row_display_label,
    row_kind,
    row_ref_label,
)
from momentum.rrg_app import RrgAppConfig, run_rrg_app  # noqa: E402
from momentum.rrg_core import RRG_WINDOW_DEFAULT, RRG_WINDOW_ETF  # noqa: E402
from momentum.rrg_swing_cheat_sheet import ETF_SWING_CHEAT_SHEET  # noqa: E402
from utils.yahoo_weekly import load_yahoo_histories  # noqa: E402

ENV_US_ETF_PERIOD = "RRG_US_ETF_PERIOD"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRG for US ETFs vs S&P 500")
    parser.add_argument(
        "--period",
        "-p",
        choices=("3m", "6m"),
        default=os.environ.get(ENV_US_ETF_PERIOD, "3m"),
        help="Analysis lookback on Date slider (default: 3m for tactical ETF rotation)",
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


def _resolve_etf_ticker(requested: str) -> str | None:
    text = requested.strip().upper()
    if not text:
        return None
    if text in RRG_ETF_ROW_IDS:
        return text
    for sym, label in RRG_ETF_LABELS.items():
        if text == sym.upper() or requested.strip().lower() == label.lower():
            return sym
    return None


def _resolve_row_id(requested: str) -> str | None:
    text = requested.strip()
    if not text:
        return None
    if text in RRG_ROW_BY_ID:
        return text
    for row_id, row in RRG_ROW_BY_ID.items():
        if text.lower() == row.label.lower():
            return row_id
    return _resolve_etf_ticker(text)


def _load_all_histories(
    period: str, min_weekly_points: int, rrg_window: int, freq: str = "week"
) -> dict[str, pd.Series]:
    print(f"Loading US ETF EOD (Yahoo Finance) for RRG ({freq})...")
    batch = load_yahoo_histories(
        RRG_LOAD_YAHOO_TICKERS,
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
        freq=freq,
    )
    out: dict[str, pd.Series] = {}
    for ticker in RRG_ETF_ROW_IDS:
        out[ticker] = batch.get(ticker, pd.Series(dtype=float))
    out[RRG_BENCHMARK_YAHOO] = batch.get(RRG_BENCHMARK_YAHOO, pd.Series(dtype=float))
    return out


def _load_row_history(
    row_id: str,
    kind: str,
    period: str,
    min_weekly_points: int,
    rrg_window: int,
    freq: str = "week",
) -> pd.Series:
    return load_yahoo_histories(
        [row_id],
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
        freq=freq,
    ).get(row_id, pd.Series(dtype=float))


def _count_summary(kind_list: list[str]) -> str:
    n_etf = sum(1 for k in kind_list if k == "etf")
    return f"{n_etf} US ETFs"


def _build_config(analysis_period: str, rrg_window: int) -> RrgAppConfig:
    return RrgAppConfig(
        window_title=f"RRG — US ETFs vs S&P 500 ({analysis_period} Yahoo)",
        benchmark_nse=RRG_BENCHMARK_YAHOO,
        rows=RRG_ROWS,
        row_by_id=RRG_ROW_BY_ID,
        default_visible_ids=RRG_DEFAULT_VISIBLE_IDS,
        ref_column_header="Ticker",
        name_column_header="Name",
        defaults_checkbox_text="Default ETFs",
        hover_ref_prefix="ticker",
        universe_summary=(
            f"RRG universe: {len(RRG_ETF_ROW_IDS)} US ETFs "
            f"(benchmark {RRG_BENCHMARK_YAHOO}; edit momentum/etf/universes/us.py)"
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
        top_movers_title="Top 10 — Was vs Now",
        side_cheat_sheet=ETF_SWING_CHEAT_SHEET,
        etf_table_extras=True,
        etf_recommend_profile="us",
        etf_recommend_count=7,
        backtest_enabled=True,
        backtest_profile="us",
        backtest_universe_mode="core",
    )


def main() -> None:
    args = _parse_args()
    rrg_window = _resolve_rrg_window(args.period, args.window)
    run_rrg_app(_build_config(args.period, rrg_window))


if __name__ == "__main__":
    main()
