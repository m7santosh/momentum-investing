"""Per-ETF entry / exit / running prices and P/L for RRG backtest weeks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from momentum.rrg_ema_exit import (
    _bare_symbol,
    _daily_close_series,
    _weekly_price_series,
)
from momentum.rrg_portfolio_exits import PortfolioExit
from momentum.rrg_core import rrg_format_date
from momentum.rrg_ranking import series_at


def _norm(sym: str) -> str:
    return _bare_symbol(sym)


def _weekly_price(
    price_weekly: dict[str, pd.Series], sym: str, as_of: pd.Timestamp
) -> float | None:
    series = _weekly_price_series(price_weekly, sym)
    if series is None or series.empty:
        return None
    try:
        return round(float(series_at(series, as_of)), 2)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _daily_price(
    daily_close: dict[str, pd.Series], sym: str, as_of: pd.Timestamp
) -> float | None:
    series = _daily_close_series(daily_close, sym)
    if series is None or series.empty:
        return None
    try:
        sliced = series.loc[:as_of]
        if sliced.empty:
            return None
        return round(float(sliced.iloc[-1]), 2)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _pl_pct(entry: float | None, mark: float | None) -> float | None:
    if entry is None or mark is None or entry <= 0:
        return None
    return round((mark / entry - 1.0) * 100.0, 2)


def _row(
    ticker: str,
    status: str,
    entry: float | None,
    exit_price: float | None,
    running: float | None,
    exit_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    mark = exit_price if exit_price is not None else running
    return {
        "ticker": ticker,
        "status": status,
        "entry": entry,
        "exit": exit_price,
        "running": running,
        "exit_date": exit_date,
        "pl_pct": _pl_pct(entry, mark),
    }


def build_week_position_rows(
    *,
    held_at_rebal: list[str],
    end_holdings: list[str],
    mid_week_9ema: list[tuple[str, pd.Timestamp]],
    week_exits: list[PortfolioExit],
    entry_prices: dict[str, float],
    decision_date: pd.Timestamp,
    end_date: pd.Timestamp,
    price_weekly: dict[str, pd.Series],
    daily_close: dict[str, pd.Series],
    strategy_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build per-ETF price/P/L rows for names held at this week's rebalance only."""
    mid_by_ticker = {_norm(t): pd.Timestamp(d) for t, d in mid_week_9ema}
    end_set = {_norm(t) for t in end_holdings}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sym in held_at_rebal:
        if not sym:
            continue
        bare = _norm(sym)
        if bare in seen:
            continue
        seen.add(bare)
        entry = entry_prices.get(bare) or _weekly_price(price_weekly, sym, decision_date)
        if bare in mid_by_ticker:
            exit_day = mid_by_ticker[bare]
            exit_p = _daily_price(daily_close, sym, exit_day) or _weekly_price(
                price_weekly, sym, exit_day
            )
            rows.append(
                _row(
                    sym,
                    "Exited mid-week (9 EMA)",
                    entry,
                    exit_p,
                    None,
                    exit_date=exit_day,
                )
            )
        elif bare in end_set:
            running = _weekly_price(price_weekly, sym, end_date)
            rows.append(_row(sym, "Held", entry, None, running))
        else:
            exit_p = _weekly_price(price_weekly, sym, decision_date)
            rows.append(
                _row(
                    sym,
                    "Exited @ rebalance",
                    entry,
                    exit_p,
                    None,
                    exit_date=decision_date,
                )
            )

    return sort_position_rows(rows, strategy_order)


def sort_position_rows(
    rows: list[dict[str, Any]],
    strategy_order: list[str] | None,
) -> list[dict[str, Any]]:
    """Order rows to match strategy Top N table (same as Selected week picks)."""
    if not strategy_order:
        return rows
    rank = {_norm(t): i for i, t in enumerate(strategy_order) if t}
    extra_base = len(rank)

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        bare = _norm(row.get("ticker") or "")
        return (rank.get(bare, extra_base), str(row.get("ticker") or ""))

    return sorted(rows, key=sort_key)


def register_new_week_entries(
    entry_prices: dict[str, float],
    *,
    held_at_rebal: list[str],
    prev_holdings: list[str],
    decision_date: pd.Timestamp,
    price_weekly: dict[str, pd.Series],
) -> None:
    """Record entry prices for names bought at this week's rebalance."""
    prev_set = {_norm(t) for t in prev_holdings if t}
    for sym in held_at_rebal:
        if not sym:
            continue
        bare = _norm(sym)
        if bare in prev_set:
            continue
        px = _weekly_price(price_weekly, sym, decision_date)
        if px is not None:
            entry_prices[bare] = px


def update_entry_prices_after_week(
    entry_prices: dict[str, float],
    *,
    end_holdings: list[str],
    mid_week_9ema: list[tuple[str, pd.Timestamp]],
    week_exits: list[PortfolioExit],
) -> None:
    """Drop entry prices for names exited this week; keep still-held positions."""
    exited = {_norm(t) for t, _ in mid_week_9ema}
    exited.update(_norm(ex.ticker) for ex in week_exits)
    end_set = {_norm(t) for t in end_holdings}
    for bare in list(entry_prices):
        if bare in exited or bare not in end_set:
            entry_prices.pop(bare, None)


def format_exit_date(value) -> str:
    if value is None or value == "":
        return "—"
    return rrg_format_date(value)


def format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}"


def format_pl_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%"
