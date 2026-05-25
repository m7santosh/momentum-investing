"""Shared RRG (Relative Rotation Graph) indicator math and row metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RrgRow:
    """One RRG universe line."""

    row_id: str
    label: str
    ref_label: str
    kind: str  # "index" | "etf" | "stock"


def compute_rrg_indicators(index_series, benchmark_series, window=14):
    rs = 100 * (index_series / benchmark_series)
    rsr = (
        100
        + (rs - rs.rolling(window=window).mean())
        / rs.rolling(window=window).std(ddof=0)
    ).dropna()
    if len(rsr) < 2:
        return None, None, None
    rsr_roc = 100 * ((rsr / rsr.iloc[0]) - 1)
    rsm = (
        101
        + (
            (rsr_roc - rsr_roc.rolling(window=window).mean())
            / rsr_roc.rolling(window=window).std(ddof=0)
        )
    ).dropna()
    rsr = rsr[rsr.index.isin(rsm.index)]
    rsm = rsm[rsm.index.isin(rsr.index)]
    if len(rsr) < 2:
        return None, None, None
    return rsr, rsr_roc, rsm


def get_status(x, y):
    if x < 100 and y < 100:
        return "lagging"
    if x > 100 and y > 100:
        return "leading"
    if x < 100 and y > 100:
        return "improving"
    if x > 100 and y < 100:
        return "weakening"
    return None


def get_color(x, y):
    status = get_status(x, y)
    if status == "lagging":
        return "red"
    if status == "leading":
        return "green"
    if status == "improving":
        return "blue"
    if status == "weakening":
        return "yellow"
    return "gray"


TAIL_MARKER_SIZE = 22
HEAD_ARROW_SCALE = 14
HOVER_PIXEL_RADIUS = 14
# Navigable weekly points on the Date slider (~calendar months of analysis).
RRG_NAV_WEEKS_3M = 13
RRG_NAV_WEEKS_6M = 26
RRG_NAV_WEEKS = RRG_NAV_WEEKS_6M  # default / stock RRG

# Rolling window (weeks) for JdK RS-ratio and momentum z-scores.
RRG_WINDOW_DEFAULT = 14
RRG_WINDOW_ETF = 10  # faster response for tactical ETF rotation


def rrg_nav_weeks(period: str) -> int:
    """Analysis window length (Date slider) for ``period``."""
    p = period.lower()
    if p in ("3m", "3mo"):
        return RRG_NAV_WEEKS_3M
    if p in ("6m", "6mo"):
        return RRG_NAV_WEEKS_6M
    return RRG_NAV_WEEKS_6M


def rrg_warmup_weeks(window: int) -> int:
    """Weeks of history before the first valid RRG point (``window * 2 + 2``)."""
    return window * 2 + 2


def rrg_fetch_calendar_days(period: str, window: int = RRG_WINDOW_DEFAULT) -> int:
    """Calendar days to download: warmup + analysis window + small buffer."""
    total_weeks = rrg_warmup_weeks(window) + rrg_nav_weeks(period)
    return total_weeks * 7 + 14


def rrg_period_display(period: str) -> str:
    """Short label for logs and window title."""
    p = period.lower()
    if p in ("3m", "3mo"):
        return "3-month"
    if p in ("6m", "6mo"):
        return "6-month"
    if p == "1y":
        return "1-year"
    if p == "2y":
        return "2-year"
    return period


def rrg_period_label(period: str) -> str:
    """Human label for the RRG analysis window (what the chart navigates)."""
    return f"{rrg_period_display(period)} lookback ({rrg_nav_weeks(period)} weekly points)"
