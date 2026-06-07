"""Tradable ref-ETF / ticker price checks for RRG picks (India bhavcopy, US Yahoo)."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

import pandas as pd

from momentum.rrg_ema_exit import _weekly_price_series
from momentum.rrg_ranking import series_at

T = TypeVar("T")


def has_ref_weekly_price(
    price_weekly: dict[str, pd.Series],
    ticker: str,
    as_of: pd.Timestamp,
) -> bool:
    """True when a positive weekly close exists on or before ``as_of``."""
    series = _weekly_price_series(price_weekly, ticker)
    if series is None or series.empty:
        return False
    try:
        return float(series_at(series, as_of)) > 0
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def filter_tickers_with_ref_price(
    tickers: list[str],
    price_weekly: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> list[str]:
    """Keep tickers that have tradable weekly history at rebalance."""
    return [
        t
        for t in tickers
        if t and has_ref_weekly_price(price_weekly, t, as_of)
    ]


def weekly_map_from_daily(
    daily_close: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """W-FRI weekly closes from daily CM/Yahoo series (India/US live RRG)."""
    out: dict[str, pd.Series] = {}
    for sym, daily in daily_close.items():
        if daily is not None and len(daily):
            out[sym] = daily.sort_index().resample("W-FRI").last().dropna()
    return out


def filter_picks_with_ref_price(
    picks: list[T],
    price_weekly: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> list[T]:
    """Drop recommendations with no ref price; renumber ``pick_rank`` when present."""
    if not picks:
        return []
    kept = [
        p
        for p in picks
        if has_ref_weekly_price(price_weekly, getattr(p, "ticker", ""), as_of)
    ]
    if not kept:
        return []
    if not hasattr(kept[0], "pick_rank"):
        return kept
    return [replace(p, pick_rank=i) for i, p in enumerate(kept, 1)]
