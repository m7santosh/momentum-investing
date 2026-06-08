"""RRG for US ETFs vs S&P 500 — fixed 3-month analysis (swing / tactical).

Universe: canonical us.py ETF list (no core/expanded switch in UI).

Examples:
    python momentum/etf/RRGIndicatorUsEtfs3m.py
    python momentum/etf/RRGIndicatorUsEtfs3m.py --universe expanded
    python momentum/etf/RRGIndicatorUsEtfs3m.py --window 10

For 6-month analysis use RRGIndicatorUsEtfs.py --period 6m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.us_rrg_universe_modes import (  # noqa: E402
    US_UNIVERSE_CORE,
    US_UNIVERSE_EXPANDED,
    build_us_rrg_config,
)
from momentum.rrg_app import run_rrg_app  # noqa: E402
from momentum.rrg_core import RRG_WINDOW_ETF  # noqa: E402

US_ETF_RRG_PERIOD = "3m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRG for US ETFs vs S&P 500 (3-month analysis only)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        choices=(10, 14),
        default=RRG_WINDOW_ETF,
        help="RRG rolling window in weeks (default: 10)",
    )
    parser.add_argument(
        "--universe",
        "-u",
        choices=(US_UNIVERSE_CORE, US_UNIVERSE_EXPANDED),
        default=US_UNIVERSE_CORE,
        help="Kept for CLI compat; core and expanded both load us.py (default: core)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = build_us_rrg_config(
        args.universe,
        period=US_ETF_RRG_PERIOD,
        rrg_window=args.window,
    )
    cfg.window_title = (
        f"RRG — US ETFs vs S&P 500 (3-month · {args.window}w window · Yahoo)"
    )
    cfg.universe_summary = (
        f"{cfg.universe_summary} | analysis: 3-month lookback, {args.window}w RRG window"
    )
    run_rrg_app(cfg)


if __name__ == "__main__":
    main()
