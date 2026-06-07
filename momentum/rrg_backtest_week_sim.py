"""Shared intraweek exit simulation helpers for RRG backtests."""

from __future__ import annotations

from momentum.rrg_ema_exit import simulate_week_with_exits
from momentum.rrg_portfolio_fill import equal_weight_port_return

import numpy as np


def intraweek_exits_enabled(cfg) -> bool:
    return bool(getattr(cfg, "exit_below_9ema", False)) or bool(
        getattr(cfg, "exit_stop_loss", False)
    )


def simulate_backtest_week(
    cfg,
    holdings: list[str],
    decision_date,
    next_date,
    daily_close: dict,
    price_weekly: dict,
    held_count: int,
) -> tuple[float, list[str], list, list]:
    """
    Weekly portfolio return and end holdings.

    Returns (port_ret, end_holdings, mid_week_9ema, mid_week_stop_loss).
    """
    slot_n = max(int(cfg.top_n), len(holdings))
    if not holdings:
        return 0.0, [], [], []

    if intraweek_exits_enabled(cfg):
        week_rets, end_holdings, mid_week_9ema, mid_week_stop_loss = (
            simulate_week_with_exits(
                holdings,
                decision_date,
                next_date,
                daily_close,
                price_weekly,
                slot_n,
                exit_below_9ema=bool(cfg.exit_below_9ema),
                exit_stop_loss=bool(cfg.exit_stop_loss),
                stop_loss_pct=float(getattr(cfg, "stop_loss_pct", 5.0) or 5.0),
            )
        )
        port_ret = equal_weight_port_return(week_rets, held_count)
        return port_ret, end_holdings, mid_week_9ema, mid_week_stop_loss

    week_rets: list[float] = []
    for ref in holdings:
        series = price_weekly.get(ref)
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
    port_ret = float(np.mean(week_rets)) if week_rets else 0.0
    return port_ret, list(holdings), [], []
