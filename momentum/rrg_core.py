"""Shared RRG (Relative Rotation Graph) indicator math and row metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

RRG_DISPLAY_DATE_FMT = "%d-%m-%Y"


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


# Dark strokes for RRG chart trails, markers, and labels (table keeps light fills).
RRG_PLOT_COLOR_LEADING = "#2E7D32"
RRG_PLOT_COLOR_IMPROVING = "#1565C0"
RRG_PLOT_COLOR_WEAKENING = "#E65100"
RRG_PLOT_COLOR_LAGGING = "#C62828"
RRG_PLOT_COLOR_NA = "#424242"


def get_chart_color(x, y):
    """Quadrant-matched dark color for RRG graph overlays."""
    status = get_status(x, y)
    if status == "lagging":
        return RRG_PLOT_COLOR_LAGGING
    if status == "leading":
        return RRG_PLOT_COLOR_LEADING
    if status == "improving":
        return RRG_PLOT_COLOR_IMPROVING
    if status == "weakening":
        return RRG_PLOT_COLOR_WEAKENING
    return RRG_PLOT_COLOR_NA


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


def rrg_coerce_date(value) -> pd.Timestamp:
    """Parse RRG date inputs: ``date``, Timestamp, YYYY-MM-DD, or DD-MM-YYYY."""
    if value is None or value == "":
        raise ValueError("Date is required.")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("Date is required.")
        parts = raw.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts) and len(parts[2]) == 4:
            if len(parts[0]) == 4:
                ts = pd.Timestamp(datetime(int(parts[0]), int(parts[1]), int(parts[2])))
            else:
                ts = rrg_parse_user_date(raw)
        else:
            ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value!r}")
    return ts


def rrg_format_date(value) -> str:
    """Format a timestamp for RRG UI labels (DD-MM-YYYY)."""
    if value is None or value == "":
        return ""
    try:
        ts = rrg_coerce_date(value)
    except ValueError:
        return ""
    return ts.strftime(RRG_DISPLAY_DATE_FMT)


def rrg_parse_user_date(text: str) -> pd.Timestamp:
    """Parse RRG date entry fields — strict DD-MM-YYYY only."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Date is required (DD-MM-YYYY).")
    parts = raw.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"Use DD-MM-YYYY only (e.g. 03-06-2026). Got: {raw!r}"
        )
    day, month, year = (int(parts[0]), int(parts[1]), int(parts[2]))
    if len(parts[2]) != 4:
        raise ValueError(
            f"Use DD-MM-YYYY only (4-digit year). Got: {raw!r}"
        )
    try:
        ts = pd.Timestamp(datetime(year, month, day))
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date: {raw!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {raw!r}")
    if ts.strftime(RRG_DISPLAY_DATE_FMT) != raw:
        raise ValueError(f"Use DD-MM-YYYY only (e.g. 03-06-2026). Got: {raw!r}")
    return ts


def rrg_config_date_str(text: str) -> str:
    """Normalize DD-MM-YYYY entry to YYYY-MM-DD for engines and data loads."""
    return rrg_parse_user_date(text).strftime("%Y-%m-%d")


def panel_rebal_bar_index(
    weekly_index: pd.DatetimeIndex,
    as_of_ts: pd.Timestamp,
    tail_bars: int,
) -> int:
    """
    Weekly bar index for the portfolio panel rebalance (same rule as main RRG).

    When ``as_of`` falls on a weekly bar that starts a new hold week, that bar is
    the rebalance date (e.g. slider on 08-05 → rebalance 08-05, not prior 01-05).

    Mid-week ``as_of`` (between bars) maps to the hold week that contains it.
    The latest available weekly bar is a rebalance when ``as_of`` is on that date
    (or later with no following bar yet).
    """
    wi = pd.DatetimeIndex(weekly_index).sort_values()
    if not len(wi):
        return 0
    tail_n = max(1, int(tail_bars))
    as_of = pd.Timestamp(as_of_ts)
    end_i = int(wi.get_indexer([as_of], method="ffill")[0])
    if end_i < 0:
        return 0
    if end_i <= tail_n:
        return end_i
    end_ts_local = pd.Timestamp(wi[end_i]).normalize()
    as_of_day = as_of.normalize()

    if as_of_day == end_ts_local:
        return end_i

    for k in range(end_i, tail_n - 1, -1):
        if k + 1 < len(wi):
            week_start = pd.Timestamp(wi[k])
            week_end = pd.Timestamp(wi[k + 1])
            if week_start <= as_of <= week_end:
                return k

    if end_i == len(wi) - 1 and as_of_day >= end_ts_local:
        return end_i

    return max(tail_n, end_i - 1)


def rrg_build_slider_date_index(
    bench: pd.Series,
    *,
    analysis_period: str,
    window: int,
    unit: str = "week",
    daily_sources: list[pd.Series] | None = None,
) -> pd.DatetimeIndex:
    """
    Bar dates for the RRG Date slider (shared by main app and backtest panel).

    Trims warmup, then caps to analysis window + tail buffer — same rule everywhere.
    """
    warmup = rrg_warmup_bars(window, unit)
    slider_bars = rrg_slider_index_bars(
        analysis_period, tail=RRG_MAX_TAIL, unit=unit
    )

    def _cap(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
        cal = pd.DatetimeIndex(cal).sort_values()
        if len(cal) > warmup:
            cal = cal[warmup:]
        if len(cal) > slider_bars:
            cal = cal[-slider_bars:]
        return cal

    if rrg_normalize_bar_unit(unit) == "day" and daily_sources:
        best: pd.DatetimeIndex | None = None
        for daily in daily_sources:
            if daily is None or not len(daily):
                continue
            sub = _cap(daily.dropna().sort_index().index)
            if len(sub) and (best is None or len(sub) > len(best)):
                best = sub
        if best is not None and len(best):
            return best

    bench = bench.dropna().sort_index()
    if len(bench.index):
        return _cap(bench.index)
    if daily_sources:
        candidates = [
            s.dropna().sort_index()
            for s in daily_sources
            if s is not None and len(s)
        ]
        if candidates:
            longest = max(candidates, key=len)
            return _cap(longest.index)
    return pd.DatetimeIndex([])
