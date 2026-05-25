"""RRG chart for stocks vs Nifty 500 (selectable universe via CLI/env).

Universe: universes/*.py (quality, n500, bse_largemidcap, nifty_largemidcap) via stock_rrg_universe.py.
Analysis: 6-month lookback (26 weekly points on Date slider); downloads extra ~30w for RRG warmup.
Not a ranker: interactive quadrant plot with tail/date sliders.

vs stock momentum rankers (Excel output):
- momentum_stocks.py — BSE LargeMidcap; abs return ranks (3M/6M/9M).
- momentum_rs_stocks.py — Nifty LargeMidcap; abs + RS vs LM250.
- quality_momentum_rs.py — Quality ~130; abs + RS vs ^CRSLDX; daily scan.
- quality_momentum_rs_lv.py — Quality ~130; + low-vol; rebalance + state.
- quality_momentum_rs_no_lv.py — Quality ~130; rebalance/state; no low-vol.
- quality_momentum_rs_lv_list.py — Quality ~130; + low-vol; list-only.
- momentum_rs_lv_n500.py — Nifty 500; + low-vol; rebalance + state.
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

from momentum.rrg_app import RrgAppConfig, run_rrg_app
from momentum.rrg_core import RRG_WINDOW_DEFAULT
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
    load_nse_equity_weekly_histories,
    load_nse_index_weekly_histories,
)

# Stock RRG: 6-month analysis (26 weekly points), 14-week rolling window.
STOCK_RRG_PERIOD = "6m"
STOCK_RRG_WINDOW = RRG_WINDOW_DEFAULT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock RRG vs Nifty 500")
    parser.add_argument(
        "--universe",
        "-u",
        choices=sorted(BY_KEY),
        help=f"Universe module key (default: env {ENV_UNIVERSE_KEY} or {DEFAULT_KEY})",
    )
    return parser.parse_args()


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
    period: str, min_weekly_points: int, rrg_window: int = 14
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    print("Loading NSE index EOD (ind_close_all) for RRG benchmark...")
    index_batch = load_nse_index_weekly_histories(
        RRG_LOAD_NSE_INDEX_NAMES,
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
    )
    out[RRG_BENCHMARK_NSE] = index_batch.get(RRG_BENCHMARK_NSE, pd.Series(dtype=float))

    print("Loading NSE stock EOD (CM bhavcopy) for RRG...")
    stock_batch = load_nse_equity_weekly_histories(
        RRG_LOAD_NSE_STOCK_SYMBOLS,
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
    )
    for sym in RRG_STOCK_ROW_IDS:
        out[sym] = stock_batch.get(sym, pd.Series(dtype=float))
    return out


def _load_row_history(
    row_id: str, kind: str, period: str, min_weekly_points: int, rrg_window: int = 14
) -> pd.Series:
    if kind == "index":
        return load_nse_index_weekly_histories(
            [row_id],
            period=period,
            min_points=min_weekly_points,
            rrg_window=rrg_window,
        ).get(row_id, pd.Series(dtype=float))
    return load_nse_equity_weekly_histories(
        [row_id],
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
    ).get(row_id, pd.Series(dtype=float))


def _count_summary(kind_list: list[str]) -> str:
    n_stock = sum(1 for k in kind_list if k == "stock")
    return f"{n_stock} stocks"


def _build_config() -> RrgAppConfig:
    mod = active_universe_module()
    return RrgAppConfig(
        window_title=f"RRG — {mod.LABEL} ({STOCK_RRG_PERIOD} EOD)",
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
            f"benchmark {RRG_BENCHMARK_NSE}; analysis {STOCK_RRG_PERIOD}"
        ),
        row_ref_label=row_ref_label,
        row_display_label=row_display_label,
        row_kind=row_kind,
        resolve_row_id=_resolve_row_id,
        load_all_histories=_load_all_histories,
        load_row_history=_load_row_history,
        count_summary=_count_summary,
        analysis_period=STOCK_RRG_PERIOD,
        rrg_window=STOCK_RRG_WINDOW,
    )


def main() -> None:
    args = _parse_args()
    key = args.universe or os.environ.get(ENV_UNIVERSE_KEY, DEFAULT_KEY)
    use_universe_key(key)
    run_rrg_app(_build_config())


if __name__ == "__main__":
    main()
