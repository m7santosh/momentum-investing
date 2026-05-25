"""RRG chart for NSE indices and ETFs (JdK RS ratio vs momentum).

Universe: universes/india.py (via etf_rrg_universe.py) — index EOD + ETF bhavcopy.
Analysis: default 3-month lookback (13 weekly points, 10w rolling window); optional --period 6m.
Downloads extra history for RRG warmup (~22w for 3m / ~30w for 6m) — not plotted on the slider.
Not a ranker: interactive quadrant plot with tail/date sliders and sector table.

vs momentum ETF rankers (Excel output):
- momentum_etfs.py / momentum_us_etfs.py — abs return ranks only.
- momentum_rs_etfs.py / momentum_us_rs_etfs.py — abs + RS blended ranks (swing).
- momentum_rs_etfs_adaptive.py / momentum_us_rs_etfs_adaptive.py — RS-only, short horizons.

Examples:
    python momentum/etf/RRGIndicatorEtfs.py
    python momentum/etf/RRGIndicatorEtfs.py --period 6m
    python momentum/etf/RRGIndicatorEtfs.py --period 3m --window 10
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.etf_rrg_universe import (
    RRG_BENCHMARK_NSE,
    RRG_DEFAULT_VISIBLE_IDS,
    RRG_ETF_LABELS,
    RRG_ETF_ROW_IDS,
    RRG_INDEX_ROW_IDS,
    RRG_LOAD_ETF_NSE_SYMBOLS,
    RRG_LOAD_NSE_INDEX_NAMES,
    RRG_ROW_BY_ID,
    RRG_ROWS,
    index_ref_etf_label,
    row_display_label,
    row_kind,
)
from momentum.rrg_app import RrgAppConfig, run_rrg_app  # noqa: E402
from momentum.rrg_core import RRG_WINDOW_DEFAULT, RRG_WINDOW_ETF  # noqa: E402
from utils.nse_bhavcopy import (  # noqa: E402
    fetch_index_close_all,
    load_nse_etf_weekly_histories,
    load_nse_index_weekly_histories,
    resolve_index_name,
    today_ist,
)

ENV_ETF_PERIOD = "RRG_ETF_PERIOD"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRG for NSE indices and ETFs")
    parser.add_argument(
        "--period",
        "-p",
        choices=("3m", "6m"),
        default=os.environ.get(ENV_ETF_PERIOD, "3m"),
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


def _resolve_nse_index_name(requested: str) -> str | None:
    text = requested.strip()
    if not text:
        return None
    if text in RRG_INDEX_ROW_IDS:
        return text
    d = today_ist()
    for _ in range(12):
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        day_map = fetch_index_close_all(d, quiet=True)
        if day_map:
            canonical = resolve_index_name(text, day_map)
            if canonical:
                return canonical
        d -= timedelta(days=1)
    return None


def _resolve_etf_symbol(requested: str) -> str | None:
    text = requested.strip().upper().replace(".NS", "")
    if not text:
        return None
    if text in RRG_ETF_ROW_IDS:
        return text
    for sym, label in RRG_ETF_LABELS.items():
        bare = sym.replace(".NS", "")
        if text == bare.upper() or requested.strip().lower() == label.lower():
            return bare
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
    idx = _resolve_nse_index_name(text)
    if idx:
        return idx
    return _resolve_etf_symbol(text)


def _load_all_histories(
    period: str, min_weekly_points: int, rrg_window: int
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    print("Loading NSE index EOD (ind_close_all) for RRG...")
    index_batch = load_nse_index_weekly_histories(
        RRG_LOAD_NSE_INDEX_NAMES,
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
    )
    for name in RRG_INDEX_ROW_IDS:
        out[name] = index_batch.get(name, pd.Series(dtype=float))
    out[RRG_BENCHMARK_NSE] = index_batch.get(RRG_BENCHMARK_NSE, pd.Series(dtype=float))

    if RRG_LOAD_ETF_NSE_SYMBOLS:
        print("Loading NSE ETF EOD (CM bhavcopy) for RRG...")
        etf_batch = load_nse_etf_weekly_histories(
            RRG_LOAD_ETF_NSE_SYMBOLS,
            period=period,
            min_points=min_weekly_points,
            rrg_window=rrg_window,
        )
        for sym in RRG_LOAD_ETF_NSE_SYMBOLS:
            out[sym] = etf_batch.get(sym, pd.Series(dtype=float))
    return out


def _load_row_history(
    row_id: str, kind: str, period: str, min_weekly_points: int, rrg_window: int
) -> pd.Series:
    if kind == "index":
        return load_nse_index_weekly_histories(
            [row_id],
            period=period,
            min_points=min_weekly_points,
            rrg_window=rrg_window,
        ).get(row_id, pd.Series(dtype=float))
    return load_nse_etf_weekly_histories(
        [row_id],
        period=period,
        min_points=min_weekly_points,
        rrg_window=rrg_window,
    ).get(row_id, pd.Series(dtype=float))


def _count_summary(kind_list: list[str]) -> str:
    n_index = sum(1 for k in kind_list if k == "index")
    n_etf = sum(1 for k in kind_list if k == "etf")
    return f"{n_index} indices + {n_etf} ETFs"


def _build_config(analysis_period: str, rrg_window: int) -> RrgAppConfig:
    return RrgAppConfig(
        window_title=f"RRG — NSE Indices & ETFs ({analysis_period} EOD)",
        benchmark_nse=RRG_BENCHMARK_NSE,
        rows=RRG_ROWS,
        row_by_id=RRG_ROW_BY_ID,
        default_visible_ids=RRG_DEFAULT_VISIBLE_IDS,
        ref_column_header="Ref ETF",
        name_column_header="Index",
        defaults_checkbox_text="Default indices",
        hover_ref_prefix="ref ETF",
        universe_summary=(
            f"RRG universe: {len(RRG_LOAD_NSE_INDEX_NAMES)} NSE indices "
            f"+ {len(RRG_LOAD_ETF_NSE_SYMBOLS)} ETFs (hardcoded in momentum/etf/universes/india.py)"
        ),
        row_ref_label=index_ref_etf_label,
        row_display_label=row_display_label,
        row_kind=row_kind,
        resolve_row_id=_resolve_row_id,
        load_all_histories=_load_all_histories,
        load_row_history=_load_row_history,
        count_summary=_count_summary,
        analysis_period=analysis_period,
        rrg_window=rrg_window,
    )


def main() -> None:
    args = _parse_args()
    rrg_window = _resolve_rrg_window(args.period, args.window)
    run_rrg_app(_build_config(args.period, rrg_window))


if __name__ == "__main__":
    main()
