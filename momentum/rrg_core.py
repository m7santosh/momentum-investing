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


# Light quadrant fills (table rows + RRG trail); readable with black text.
RRG_COLOR_LEADING = "#C8E6C9"
RRG_COLOR_IMPROVING = "#BBDEFB"
RRG_COLOR_WEAKENING = "#FFF9C4"
RRG_COLOR_LAGGING = "#FFCDD2"
RRG_COLOR_NA = "#E8E8E8"

# Slightly richer tints for the RRG chart quadrant background.
RRG_CHART_COLOR_LEADING = "#A5D6A7"
RRG_CHART_COLOR_IMPROVING = "#90CAF9"
RRG_CHART_COLOR_WEAKENING = "#FFF59D"
RRG_CHART_COLOR_LAGGING = "#EF9A9A"


def get_color(x, y):
    status = get_status(x, y)
    if status == "lagging":
        return RRG_COLOR_LAGGING
    if status == "leading":
        return RRG_COLOR_LEADING
    if status == "improving":
        return RRG_COLOR_IMPROVING
    if status == "weakening":
        return RRG_COLOR_WEAKENING
    return RRG_COLOR_NA


def rrg_row_fg_color(bg_color: str) -> str:
    """Foreground text on RRG row backgrounds (light fills → dark text)."""
    return "black"


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

RRG_DEFAULT_TAIL = 5
RRG_MAX_TAIL = 10  # tail slider max; extra index weeks reserved for fetch/slider
RRG_TRADING_DAYS_PER_WEEK = 5
RRG_BAR_UNITS = ("week", "day")


def rrg_nav_weeks(period: str) -> int:
    """Analysis window length (Date slider) for ``period``."""
    p = period.lower()
    if p in ("3m", "3mo"):
        return RRG_NAV_WEEKS_3M
    if p in ("6m", "6mo"):
        return RRG_NAV_WEEKS_6M
    return RRG_NAV_WEEKS_6M


def rrg_normalize_bar_unit(unit: str) -> str:
    """``week`` or ``day`` (default ``week``)."""
    u = (unit or "week").strip().lower()
    return u if u in RRG_BAR_UNITS else "week"


def rrg_nav_bars(period: str, unit: str = "week") -> int:
    """Navigable bars on the Date slider for ``period`` at ``unit`` frequency."""
    weeks = rrg_nav_weeks(period)
    if rrg_normalize_bar_unit(unit) == "day":
        return weeks * RRG_TRADING_DAYS_PER_WEEK
    return weeks


def rrg_effective_window(window: int, unit: str = "week") -> int:
    """Rolling-window length in bars (weeks or trading days)."""
    if rrg_normalize_bar_unit(unit) == "day":
        return window * RRG_TRADING_DAYS_PER_WEEK
    return window


def rrg_min_history_bars(window: int, unit: str = "week") -> int:
    return rrg_effective_window(window, unit) + 2


def rrg_warmup_weeks(window: int) -> int:
    """Weeks of history before the first valid RRG point (``window * 2 + 2``)."""
    return window * 2 + 2


def rrg_warmup_bars(window: int, unit: str = "week") -> int:
    """Bars of history before the first valid RRG point at ``unit`` frequency."""
    return rrg_warmup_weeks(rrg_effective_window(window, unit))


def rrg_slider_index_weeks(period: str, *, tail: int = RRG_MAX_TAIL) -> int:
    """Weekly points on the Date index: analysis window + room for tail at early ends."""
    return rrg_nav_weeks(period) + tail


def rrg_slider_index_bars(
    period: str, *, tail: int = RRG_MAX_TAIL, unit: str = "week"
) -> int:
    """Date-index length: analysis window + tail buffer at ``unit`` frequency."""
    return rrg_nav_bars(period, unit) + tail


def rrg_fetch_calendar_days(
    period: str,
    window: int = RRG_WINDOW_DEFAULT,
    *,
    tail: int = RRG_MAX_TAIL,
    unit: str = "week",
) -> int:
    """Calendar days to download: warmup + analysis window + tail buffer + small pad."""
    eff = rrg_effective_window(window, unit)
    total_bars = rrg_warmup_bars(window, unit) + rrg_slider_index_bars(
        period, tail=tail, unit=unit
    )
    if rrg_normalize_bar_unit(unit) == "day":
        return int(total_bars * 7 / RRG_TRADING_DAYS_PER_WEEK) + 21
    return total_bars * 7 + 14


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


def rrg_period_label(period: str, unit: str = "week") -> str:
    """Human label for the RRG analysis window (what the chart navigates)."""
    bars = rrg_nav_bars(period, unit)
    bar_word = "daily" if rrg_normalize_bar_unit(unit) == "day" else "weekly"
    return f"{rrg_period_display(period)} lookback ({bars} {bar_word} points)"
