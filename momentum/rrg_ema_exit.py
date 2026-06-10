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
    kept, _ = split_holdings_below_9ema(holdings, daily_close, as_of)
    return kept


def _bare_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".NS", "")


def _daily_close_series(
    daily_close: dict[str, pd.Series], sym: str
) -> pd.Series | None:
    """Lookup daily close by symbol or bare ticker (never use ``or`` on Series)."""
    for key in (sym, _bare_symbol(sym)):
        if not key:
            continue
        close = daily_close.get(key)
        if close is not None and not close.empty:
            return close
    return None


def _weekly_price_series(
    price_weekly: dict[str, pd.Series], sym: str
) -> pd.Series | None:
    """Lookup weekly prices by symbol or bare ticker."""
    for key in (sym, _bare_symbol(sym)):
        if not key:
            continue
        series = price_weekly.get(key)
        if series is not None and not series.empty:
            return series
    return None


def _entry_price_for_stop(
    sym: str,
    p_rebal: float,
    entry_prices: dict[str, float] | None,
) -> float:
    if not entry_prices:
        return p_rebal
    bare = _bare_symbol(sym)
    for key in (sym, bare):
        px = entry_prices.get(key)
        if px is not None and px > 0:
            return float(px)
    return p_rebal


def split_holdings_at_stop_loss(
    holdings: list[str],
    mark_series: dict[str, pd.Series],
    entry_prices: dict[str, float],
    as_of: pd.Timestamp,
    stop_loss_pct: float,
) -> tuple[list[str], list[str]]:
    """Drop holdings at/through ``as_of`` that are at or below stop from cost basis."""
    if stop_loss_pct <= 0 or not holdings:
        return list(holdings), []
    stop_mult = 1.0 - stop_loss_pct / 100.0
    kept: list[str] = []
    dropped: list[str] = []
    for sym in holdings:
        p_stop = _entry_price_for_stop(sym, 0.0, entry_prices)
        if p_stop <= 0:
            kept.append(sym)
            continue
        series = _daily_close_series(mark_series, sym)
        if series is None:
            kept.append(sym)
            continue
        sliced = series.loc[:as_of]
        if sliced.empty:
            kept.append(sym)
            continue
        mark = float(sliced.iloc[-1])
        if mark <= p_stop * stop_mult:
            dropped.append(sym)
        else:
            kept.append(sym)
    return kept, dropped


def split_holdings_below_9ema(
    holdings: list[str],
    daily_close: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) at rebalance for symbols below 9 EMA."""
    kept: list[str] = []
    dropped: list[str] = []
    for sym in holdings:
        close = _daily_close_series(daily_close, sym)
        if close is None:
            kept.append(sym)
            continue
        if close_below_9ema(close, as_of):
            dropped.append(sym)
        else:
            kept.append(sym)
    return kept, dropped


def apply_9ema_rebalance_slots(
    rebalance_holdings: list[str],
    daily_close: dict[str, pd.Series],
    as_of: pd.Timestamp,
    *,
    enabled: bool,
) -> tuple[list[str], list[str]]:
    """
    Top N in table order: exclude picks below 9 EMA at rebalance (empty slot).
    Returns (slot list with "" for excluded, dropped pick tickers).
    """
    if not enabled or not rebalance_holdings:
        return list(rebalance_holdings), []
    slots: list[str] = []
    dropped: list[str] = []
    for sym in rebalance_holdings:
        close = _daily_close_series(daily_close, sym)
        if close is not None and close_below_9ema(close, as_of):
            dropped.append(sym)
            slots.append("")
        else:
            slots.append(sym)
    return slots, dropped


def rebalance_holdings_entered(rebalance_slots: list[str]) -> list[str]:
    """Non-empty tickers from slot-aligned rebalance list."""
    return [sym for sym in rebalance_slots if sym]


def rebalance_9ema_dropped(
    rebalance_holdings: list[str],
    prior_holdings: list[str] | None,
    daily_close: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    """
    9 EMA drops at rebalance: new picks below 9 EMA plus prior-week names
    rotated out that are still below 9 EMA (so they get a 9 EMA reason, not only strategy).
    """
    kept, dropped = split_holdings_below_9ema(rebalance_holdings, daily_close, as_of)
    if not prior_holdings:
        return kept, dropped
    kept_set = {_bare_symbol(s) for s in kept}
    dropped_set = {_bare_symbol(s) for s in dropped}
    for sym in prior_holdings:
        bare = _bare_symbol(sym)
        if not bare or bare in kept_set or bare in dropped_set:
            continue
        close = _daily_close_series(daily_close, sym)
        if close is None:
            continue
        if close_below_9ema(close, as_of):
            dropped.append(sym)
            dropped_set.add(bare)
    return kept, dropped


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


def _close_on_date(
    daily: pd.Series,
    weekly: pd.Series | None,
    as_of: pd.Timestamp,
) -> float | None:
    """Mark on ``as_of``: daily close first (mid-week exits), else weekly W-FRI."""
    as_of = pd.Timestamp(as_of)
    if daily is not None and not daily.empty:
        sliced = daily.loc[:as_of]
        if len(sliced):
            px = float(sliced.iloc[-1])
            if px > 0:
                return px
    if weekly is not None and not weekly.empty:
        sliced = weekly.loc[:as_of]
        if len(sliced):
            px = float(sliced.iloc[-1])
            if px > 0:
                return px
    return None


def first_stop_loss_exit_day(
    close: pd.Series,
    entry_price: float,
    after: pd.Timestamp,
    through: pd.Timestamp,
    stop_loss_pct: float,
) -> pd.Timestamp | None:
    """First session after ``after`` where close is at/below entry stop (``stop_loss_pct`` % loss)."""
    if close.empty or entry_price <= 0 or stop_loss_pct <= 0:
        return None
    stop_px = entry_price * (1.0 - stop_loss_pct / 100.0)
    idx = close.index
    mask = (idx > after) & (idx <= through)
    for day in idx[mask]:
        sliced = close.loc[:day]
        if len(sliced) == 0:
            continue
        if float(sliced.iloc[-1]) <= stop_px:
            return pd.Timestamp(day)
    return None


def _first_intraweek_exit_day(
    daily_9ema: pd.Series,
    daily_stop: pd.Series,
    p_stop_entry: float,
    decision_date: pd.Timestamp,
    period_end: pd.Timestamp,
    *,
    exit_below_9ema: bool,
    exit_stop_loss: bool,
    stop_loss_pct: float,
) -> tuple[pd.Timestamp | None, str | None]:
    """Earliest mid-week exit day and rule tag (``9ema`` | ``stop_loss``)."""
    candidates: list[tuple[pd.Timestamp, str]] = []
    if exit_below_9ema:
        ema_day = first_9ema_exit_day(daily_9ema, decision_date, period_end)
        if ema_day is not None:
            candidates.append((ema_day, "9ema"))
    if exit_stop_loss:
        sl_day = first_stop_loss_exit_day(
            daily_stop, p_stop_entry, decision_date, period_end, stop_loss_pct
        )
        if sl_day is not None:
            candidates.append((sl_day, "stop_loss"))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def simulate_week_with_exits(
    holdings: list[str],
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
    daily_close: dict[str, pd.Series],
    price_weekly: dict[str, pd.Series],
    top_n: int,
    *,
    exit_below_9ema: bool = False,
    exit_stop_loss: bool = False,
    stop_loss_pct: float = 5.0,
    through_date: pd.Timestamp | None = None,
    entry_prices: dict[str, float] | None = None,
    daily_adj: dict[str, pd.Series] | None = None,
) -> tuple[
    list[float],
    list[str],
    list[tuple[str, pd.Timestamp]],
    list[tuple[str, pd.Timestamp]],
]:
    """
    Equal-weight slot returns for one rebalance week with optional 9 EMA / stop-loss exits.

    When both are enabled, the earlier trigger wins. Returns
    (slot_returns, end_holdings, mid_week_9ema, mid_week_stop_loss).
    """
    if not holdings:
        return [0.0] * top_n, [], [], []

    if not exit_below_9ema and not exit_stop_loss:
        rets, end_h, mid = _weekly_returns_only(
            holdings, decision_date, next_date, price_weekly, top_n
        )
        return rets, end_h, mid, []

    period_end = pd.Timestamp(next_date)
    if through_date is not None:
        period_end = min(period_end, pd.Timestamp(through_date))

    all_days: list[pd.Timestamp] = []
    for sym in holdings:
        close = _daily_close_series(daily_close, sym)
        if close is not None:
            all_days.extend(close.index.tolist())
    if not all_days:
        rets, end_h, mid = _weekly_returns_only(
            holdings, decision_date, period_end, price_weekly, top_n
        )
        return rets, end_h, mid, []

    per_slot: list[dict] = []
    mid_week_9ema: list[tuple[str, pd.Timestamp]] = []
    mid_week_stop_loss: list[tuple[str, pd.Timestamp]] = []

    for sym in holdings:
        weekly = _weekly_price_series(price_weekly, sym)
        daily = _daily_close_series(daily_close, sym)
        if daily is None:
            daily = pd.Series(dtype=float)
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

        p_stop = _entry_price_for_stop(sym, p_entry, entry_prices)
        daily_stop = _daily_close_series(daily_adj, sym) if daily_adj else daily
        if daily_stop is None or daily_stop.empty:
            daily_stop = daily

        exit_day, rule = _first_intraweek_exit_day(
            daily,
            daily_stop,
            p_stop,
            decision_date,
            period_end,
            exit_below_9ema=exit_below_9ema,
            exit_stop_loss=exit_stop_loss,
            stop_loss_pct=stop_loss_pct,
        )
        if exit_day is not None:
            p_exit = _close_on_date(daily, weekly, exit_day) or p_entry
            per_slot.append(
                {
                    "sym": sym,
                    "ret": (p_exit / p_entry - 1.0),
                    "exited": True,
                }
            )
            if rule == "9ema":
                mid_week_9ema.append((sym, exit_day))
            elif rule == "stop_loss":
                mid_week_stop_loss.append((sym, exit_day))
            continue

        exit_slice = weekly.loc[:period_end]
        p_exit = float(exit_slice.iloc[-1]) if len(exit_slice) else p_entry
        per_slot.append({"sym": sym, "ret": (p_exit / p_entry - 1.0), "exited": False})

    end_holdings = [s["sym"] for s in per_slot if not s.get("exited")]
    slot_returns = [s["ret"] for s in per_slot]
    n_empty = max(top_n - len(slot_returns), 0)
    slot_returns.extend([0.0] * n_empty)
    return slot_returns, end_holdings, mid_week_9ema, mid_week_stop_loss


def simulate_week_with_9ema_exits(
    holdings: list[str],
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
    daily_close: dict[str, pd.Series],
    price_weekly: dict[str, pd.Series],
    top_n: int,
    *,
    through_date: pd.Timestamp | None = None,
) -> tuple[list[float], list[str], list[tuple[str, pd.Timestamp]]]:
    """Backward-compatible wrapper — 9 EMA exits only."""
    rets, end_h, mid_9, _mid_sl = simulate_week_with_exits(
        holdings,
        decision_date,
        next_date,
        daily_close,
        price_weekly,
        top_n,
        exit_below_9ema=True,
        exit_stop_loss=False,
        through_date=through_date,
    )
    return rets, end_h, mid_9


def midweek_9ema_exit_count(
    mid_week_exits: list[tuple[str, pd.Timestamp]],
) -> int:
    return len(mid_week_exits)


def _weekly_returns_only(
    holdings: list[str],
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
    price_weekly: dict[str, pd.Series],
    top_n: int,
) -> tuple[list[float], list[str], list[tuple[str, pd.Timestamp]]]:
    week_rets: list[float] = []
    for sym in holdings:
        series = _weekly_price_series(price_weekly, sym)
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
    return week_rets, list(holdings), []
