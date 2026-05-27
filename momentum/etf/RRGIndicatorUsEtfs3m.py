"""RRG for US ETFs vs S&P 500 — fixed 3-month analysis (swing / tactical).

Same UI as RRGIndicatorUsEtfs.py but locked to:
  - period: 3m (13 weekly points on the Date slider)
  - rolling window: 10 weeks (override with --window)

Universe and data: us_rrg_universe.py / Yahoo Finance.

Examples:
    python momentum/etf/RRGIndicatorUsEtfs3m.py
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

from momentum.etf.RRGIndicatorUsEtfs import _build_config  # noqa: E402
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _build_config(US_ETF_RRG_PERIOD, args.window)
    cfg.window_title = (
        f"RRG — US ETFs vs S&P 500 (3-month · {args.window}w window · Yahoo)"
    )
    cfg.universe_summary = (
        f"{cfg.universe_summary} | analysis: 3-month lookback, {args.window}w RRG window"
    )
    run_rrg_app(cfg)


if __name__ == "__main__":
    main()
