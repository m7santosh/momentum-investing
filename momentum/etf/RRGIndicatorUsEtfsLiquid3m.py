"""RRG for US ETFs — expanded universe (core us.py + ADV$ discoveries).

Keeps every ticker from ``universes/us.py`` (~60) and adds liquid sector /
country / thematic names that pass your ADV$ floor. Vol filter is off by default.

Universe: ``us_liquid_candidates.py`` + screen in ``us_liquid_screener.py``

Examples:
    python momentum/etf/RRGIndicatorUsEtfsLiquid3m.py
    python momentum/etf/RRGIndicatorUsEtfsLiquid3m.py --screen-only
    python momentum/etf/RRGIndicatorUsEtfsLiquid3m.py --min-adv 5000000
    python momentum/etf/RRGIndicatorUsEtfsLiquid3m.py --vol-percentile 70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.us_liquid_rrg_config import (  # noqa: E402
    DEFAULT_MIN_ADV,
    DEFAULT_VOL_PERCENTILE,
    US_ETF_LIQUID_RRG_PERIOD,
    build_liquid_rrg_config,
    run_screen,
)
from momentum.etf.us_liquid_rrg_universe import build_universe  # noqa: E402
from momentum.etf.us_liquid_screener import format_screen_table  # noqa: E402
from momentum.rrg_app import run_rrg_app  # noqa: E402
from momentum.rrg_core import RRG_WINDOW_ETF  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRG — core us.py ETFs plus ADV$-screened discoveries (3m)",
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
        "--min-adv",
        type=float,
        default=DEFAULT_MIN_ADV,
        metavar="USD",
        help=f"Min ADV$ for *new* discoveries (core us.py always kept; default: {DEFAULT_MIN_ADV:,.0f})",
    )
    parser.add_argument(
        "--vol-percentile",
        type=float,
        default=DEFAULT_VOL_PERCENTILE,
        help=(
            "Optional vol cap for discoveries only (100=off, default). "
            "Core us.py ignores vol filter."
        ),
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=["sector", "country", "thematic", "core", "all"],
        default=["all"],
        help="Categories for discovery scan (default: all; core us.py always included)",
    )
    parser.add_argument(
        "--adv-days",
        type=int,
        default=20,
        help="Trading days for ADV$ (default: 20)",
    )
    parser.add_argument(
        "--vol-days",
        type=int,
        default=63,
        help="Trading days for volatility (default: 63)",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Print universe table and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    screened = run_screen(args)
    print()
    print(format_screen_table(screened))
    print()

    if args.screen_only:
        return

    if not screened:
        raise SystemExit(
            "Empty universe. Check Yahoo data or lower --min-adv for discoveries."
        )

    universe = build_universe(screened)
    cfg = build_liquid_rrg_config(
        universe,
        len(screened),
        period=US_ETF_LIQUID_RRG_PERIOD,
        rrg_window=args.window,
        min_adv=args.min_adv,
        vol_percentile=args.vol_percentile,
        categories=args.categories,
    )
    run_rrg_app(cfg)


if __name__ == "__main__":
    main()
