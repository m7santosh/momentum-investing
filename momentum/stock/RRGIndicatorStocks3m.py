"""RRG for NSE stocks — fixed 3-month analysis (swing / tactical).

Same UI as RRGIndicatorStocks.py but locked to:
  - period: 3m (13 weekly points on the Date slider)
  - rolling window: 10 weeks (override with --window)

Universe and data: stock_rrg_universe.py / NSE ind_close_all + CM bhavcopy.

Examples:
    python momentum/stock/RRGIndicatorStocks3m.py
    python momentum/stock/RRGIndicatorStocks3m.py --universe quality

For 6-month analysis use RRGIndicatorStocks.py --period 6m
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.stock.RRGIndicatorStocks import _build_config
from momentum.stock.stock_rrg_universe import active_universe_module, use_universe_key
from momentum.stock.universes import BY_KEY, DEFAULT_KEY, ENV_UNIVERSE_KEY
from momentum.rrg_app import run_rrg_app  # noqa: E402
from momentum.rrg_core import RRG_WINDOW_ETF  # noqa: E402

STOCK_RRG_PERIOD = "3m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRG for NSE stocks (3-month analysis only)",
    )
    parser.add_argument(
        "--universe",
        "-u",
        choices=sorted(BY_KEY),
        help=f"Universe module key (default: env {ENV_UNIVERSE_KEY} or {DEFAULT_KEY})",
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
    key = args.universe or os.environ.get(ENV_UNIVERSE_KEY, DEFAULT_KEY)
    use_universe_key(key)
    mod = active_universe_module()
    cfg = _build_config(STOCK_RRG_PERIOD, args.window)
    cfg.window_title = (
        f"RRG — {mod.LABEL} (3-month · {args.window}w window · EOD)"
    )
    cfg.universe_summary = (
        f"{cfg.universe_summary} | analysis: 3-month lookback, {args.window}w RRG window"
    )
    run_rrg_app(cfg)


if __name__ == "__main__":
    main()
