"""9 EMA exit guard for RRG ETF portfolios (mid-week exit, no refill until rebalance)."""

from __future__ import annotations

import pandas as pd

ETF_EMA_SPAN = 9


def close_below_9ema(close: pd.Series, as_of: pd.Timestamp) -> bool:
    """True if last regular close on/before ``as_of`` is below its 9 EMA of close."""
    sliced = close.loc[:as_of]
    if len(sliced) < ETF_EMA_SPAN:
        return False
    ema9 = float(sliced.ewm(span=ETF_EMA_SPAN, adjust=False).mean().iloc[-1])
    return float(sliced.iloc[-1]) < ema9


def filter_holdings_below_9ema(
    holdings: list[str],
    daily_close: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> list[str]:
    """Drop symbols already below 9 EMA at rebalance (immediate exit)."""
    kept: list[str] = []
    for sym in holdings:
        close = daily_close.get(sym)
        if close is None or close.empty:
            kept.append(sym)
            continue
        if not close_below_9ema(close, as_of):
            kept.append(sym)
    return kept


def first_9ema_exit_day(
    close: pd.Series,
    after: pd.Timestamp,
    through: pd.Timestamp,
) -> pd.Timestamp | None:
    """First session after ``after`` through ``through`` where close < 9 EMA."""
    if close.empty:
        return None
    idx = close.index
    mask = (idx > after) & (idx <= through)
    days = idx[mask]
    for day in days:
        cs = close.loc[:day]
        if len(cs) < ETF_EMA_SPAN:
            continue
        ema9 = float(cs.ewm(span=ETF_EMA_SPAN, adjust=False).mean().iloc[-1])
        if float(cs.iloc[-1]) < ema9:
            return pd.Timestamp(day)
    return None


def simulate_week_with_9ema_exits(
    holdings: list[str],
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
    daily_close: dict[str, pd.Series],
    price_weekly: dict[str, pd.Series],
    top_n: int,
) -> tuple[list[float], list[str], int]:
    """
    Equal-weight slot returns for one rebalance week with 9 EMA exits.

    Exited slots earn return through exit day; empty slots (no refill) earn 0%.
    Returns (slot_returns length top_n, end_holdings, mid_week_exit_count).
    """
    if not holdings:
        return [0.0] * top_n, [], 0

    all_days: list[pd.Timestamp] = []
    for sym in holdings:
        close = daily_close.get(sym)
        if close is not None and len(close):
            all_days.extend(close.index.tolist())
    if not all_days:
        return _weekly_returns_only(holdings, decision_date, next_date, price_weekly, top_n)

    intraweek = pd.DatetimeIndex(sorted(set(all_days)))
    intraweek = intraweek[(intraweek > decision_date) & (intraweek <= next_date)]

    per_slot: list[dict] = []
    mid_week_exits = 0

    for sym in holdings:
        weekly = price_weekly.get(sym)
        daily = daily_close.get(sym, pd.Series(dtype=float))
        if weekly is None or weekly.empty:
            per_slot.append({"sym": sym, "ret": 0.0})
            continue
        entry_slice = weekly.loc[:decision_date]
        if len(entry_slice) == 0:
            per_slot.append({"sym": sym, "ret": 0.0})
            continue
        p_entry = float(entry_slice.iloc[-1])
        if p_entry <= 0:
            per_slot.append({"sym": sym, "ret": 0.0})
            continue

        exit_day = first_9ema_exit_day(daily, decision_date, next_date)
        if exit_day is not None:
            exit_slice = weekly.loc[:exit_day]
            if len(exit_slice) == 0:
                exit_slice = daily.loc[:exit_day]
            p_exit = float(exit_slice.iloc[-1]) if len(exit_slice) else p_entry
            per_slot.append(
                {
                    "sym": sym,
                    "ret": (p_exit / p_entry - 1.0),
                    "exited": True,
                }
            )
            mid_week_exits += 1
            continue

        exit_slice = weekly.loc[:next_date]
        p_exit = float(exit_slice.iloc[-1]) if len(exit_slice) else p_entry
        per_slot.append({"sym": sym, "ret": (p_exit / p_entry - 1.0), "exited": False})

    end_holdings = [s["sym"] for s in per_slot if not s.get("exited")]
    slot_returns = [s["ret"] for s in per_slot]
    n_empty = max(top_n - len(slot_returns), 0)
    slot_returns.extend([0.0] * n_empty)
    return slot_returns, end_holdings, mid_week_exits


def _weekly_returns_only(
    holdings: list[str],
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
    price_weekly: dict[str, pd.Series],
    top_n: int,
) -> tuple[list[float], list[str], int]:
    week_rets: list[float] = []
    for sym in holdings:
        series = price_weekly.get(sym)
        if series is None or series.empty:
            week_rets.append(0.0)
            continue
        s_from = series.loc[:decision_date]
        s_to = series.loc[:next_date]
        if len(s_from) == 0 or len(s_to) == 0:
            week_rets.append(0.0)
            continue
        p0 = float(s_from.iloc[-1])
        p1 = float(s_to.iloc[-1])
        week_rets.append((p1 / p0 - 1) if p0 > 0 else 0.0)
    n_empty = max(top_n - len(week_rets), 0)
    week_rets.extend([0.0] * n_empty)
    return week_rets, list(holdings), 0
