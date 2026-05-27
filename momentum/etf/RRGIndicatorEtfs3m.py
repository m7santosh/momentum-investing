"""RRG for NSE indices & ETFs — fixed 3-month analysis (swing / tactical).

Same UI as RRGIndicatorEtfs.py but locked to:
  - period: 3m (13 weekly points on the Date slider)
  - rolling window: 10 weeks (override with --window)

Universe and data: etf_rrg_universe.py / NSE ind_close_all + CM bhavcopy.

Examples:
    python momentum/etf/RRGIndicatorEtfs3m.py
    python momentum/etf/RRGIndicatorEtfs3m.py --window 10

For 6-month analysis use RRGIndicatorEtfs.py --period 6m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.RRGIndicatorEtfs import _build_config  # noqa: E402
from momentum.rrg_app import run_rrg_app  # noqa: E402
from momentum.rrg_core import RRG_WINDOW_ETF  # noqa: E402

ETF_RRG_PERIOD = "3m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRG for NSE indices and ETFs (3-month analysis only)",
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
    cfg = _build_config(ETF_RRG_PERIOD, args.window)
    cfg.window_title = (
        f"RRG — NSE Indices & ETFs (3-month · {args.window}w window · EOD)"
    )
    cfg.universe_summary = (
        f"{cfg.universe_summary} | analysis: 3-month lookback, {args.window}w RRG window"
    )
    run_rrg_app(cfg)


if __name__ == "__main__":
    main()
