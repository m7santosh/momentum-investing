"""Portfolio exit events (rank vs 9 EMA) for RRG ETF swing picks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from momentum.rrg_core import rrg_format_date, rrg_parse_user_date

RULE_RANK = "Rank"
RULE_9EMA = "9 EMA"
RULE_STOP_LOSS = "Stop loss"
RULE_STRATEGY = "Strategy"

TIMING_REBALANCE = "Rebalance"
TIMING_MIDWEEK = "Mid-week"


@dataclass(frozen=True)
class PortfolioExit:
    ticker: str
    rule: str
    timing: str
    detail: str


@dataclass(frozen=True)
class PortfolioPanelTotals:
    """Equal-weight portfolio returns (%%), capital split across N slots."""

    was_ew_pct: float | None
    was_slots: int
    pick_ew_pct: float | None
    pick_slots: int


def rank_overlay_exits(
    prev_holdings: list[str],
    curr_ranks: dict[int, int],
    ref_to_j: dict[str, int],
    *,
    max_hold_rank: int,
    enabled: bool,
) -> list[PortfolioExit]:
    """Prior holdings dropped because momentum rank exceeded the hold threshold."""
    if not enabled or not prev_holdings:
        return []
    n_rows = max(curr_ranks.values(), default=0) + 1
    n_rows = max(n_rows, len(ref_to_j) + 1)
    out: list[PortfolioExit] = []
    for ref in prev_holdings:
        bare = ref.strip().upper().replace(".NS", "")
        j = ref_to_j.get(bare)
        if j is None:
            continue
        rank = curr_ranks.get(j, n_rows)
        if rank > max_hold_rank:
            out.append(
                PortfolioExit(
                    ticker=bare,
                    rule=RULE_RANK,
                    timing=TIMING_REBALANCE,
                    detail=f"rank {rank} > max hold {max_hold_rank}",
                )
            )
    return out


def exits_9ema_at_rebalance(
    dropped: list[str], as_of: pd.Timestamp
) -> list[PortfolioExit]:
    date_s = rrg_format_date(as_of)
    return [
        PortfolioExit(
            ticker=t,
            rule=RULE_9EMA,
            timing=TIMING_REBALANCE,
            detail=f"close below 9 EMA at rebalance ({date_s})",
        )
        for t in dropped
    ]


def exits_9ema_midweek(
    exited: list[tuple[str, pd.Timestamp]],
) -> list[PortfolioExit]:
    out: list[PortfolioExit] = []
    for ticker, day in exited:
        out.append(
            PortfolioExit(
                ticker=ticker,
                rule=RULE_9EMA,
                timing=TIMING_MIDWEEK,
                detail=f"close below 9 EMA on {rrg_format_date(day)}",
            )
        )
    return out


def exits_stop_loss_midweek(
    exited: list[tuple[str, pd.Timestamp]],
    *,
    stop_loss_pct: float,
) -> list[PortfolioExit]:
    out: list[PortfolioExit] = []
    pct = float(stop_loss_pct)
    for ticker, day in exited:
        out.append(
            PortfolioExit(
                ticker=ticker,
                rule=RULE_STOP_LOSS,
                timing=TIMING_MIDWEEK,
                detail=f"hit {pct:g}% stop loss on {rrg_format_date(day)}",
            )
        )
    return out


def strategy_rebalance_exits(
    prev_holdings: list[str],
    rebalance_holdings: list[str],
    *,
    already_exited: set[str],
) -> list[PortfolioExit]:
    """Dropped at rebalance by base pick rotation (not rank / 9 EMA)."""
    rebal_set = {t.strip().upper().replace(".NS", "") for t in rebalance_holdings}
    out: list[PortfolioExit] = []
    for ref in prev_holdings:
        bare = ref.strip().upper().replace(".NS", "")
        if not bare or bare in already_exited or bare in rebal_set:
            continue
        out.append(
            PortfolioExit(
                ticker=bare,
                rule=RULE_STRATEGY,
                timing=TIMING_REBALANCE,
                detail="not in rebalance picks",
            )
        )
    return out


def build_week_exits(
    *,
    prev_holdings: list[str],
    rebalance_holdings: list[str],
    hold_until_rank_exit: bool,
    curr_ranks: dict[int, int],
    ref_to_j: dict[str, int],
    max_hold_rank: int,
    exit_below_9ema: bool,
    dropped_9ema_rebal: list[str],
    mid_week_9ema: list[tuple[str, pd.Timestamp]],
    decision_date: pd.Timestamp,
    include_strategy_exits: bool = True,
    exit_stop_loss: bool = False,
    mid_week_stop_loss: list[tuple[str, pd.Timestamp]] | None = None,
    stop_loss_pct: float = 5.0,
) -> list[PortfolioExit]:
    rank_ex = rank_overlay_exits(
        prev_holdings,
        curr_ranks,
        ref_to_j,
        max_hold_rank=max_hold_rank,
        enabled=hold_until_rank_exit,
    )
    ema_rebal = (
        exits_9ema_at_rebalance(dropped_9ema_rebal, decision_date)
        if exit_below_9ema
        else []
    )
    ema_mid = exits_9ema_midweek(mid_week_9ema) if exit_below_9ema else []
    sl_mid = (
        exits_stop_loss_midweek(
            mid_week_stop_loss or [],
            stop_loss_pct=stop_loss_pct,
        )
        if exit_stop_loss
        else []
    )
    already = {ex.ticker for ex in rank_ex + ema_rebal + ema_mid + sl_mid}
    strat: list[PortfolioExit] = []
    if include_strategy_exits:
        strat = strategy_rebalance_exits(
            prev_holdings, rebalance_holdings, already_exited=already
        )
    return merge_week_exits(rank_ex, ema_rebal, ema_mid, sl_mid, strat)


def merge_week_exits(*groups: list[PortfolioExit]) -> list[PortfolioExit]:
    """De-duplicate by ticker (first reason wins)."""
    seen: set[str] = set()
    merged: list[PortfolioExit] = []
    for group in groups:
        for ex in group:
            key = ex.ticker.strip().upper().replace(".NS", "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(ex)
    return merged


def filter_exits_to_top_n_picks(
    exits: list[PortfolioExit],
    rebalance_tickers: list[str] | None,
) -> list[PortfolioExit]:
    """Log/picks-only: drop exits for prior Was holdings not in this week's Top N."""
    pick_set = _top_n_pick_set(rebalance_tickers)
    if pick_set is None:
        return exits
    return [
        ex
        for ex in exits
        if ex.ticker.strip().upper().replace(".NS", "") in pick_set
    ]


def format_exit_summary(
    exits: list[PortfolioExit],
    *,
    max_items: int = 8,
    rebalance_tickers: list[str] | None = None,
) -> str:
    if rebalance_tickers is not None:
        exits = filter_exits_to_top_n_picks(exits, rebalance_tickers)
    if not exits:
        return ""
    parts = [format_exit_line(ex) for ex in exits[:max_items]]
    if len(exits) > max_items:
        parts.append(f"+{len(exits) - max_items} more")
    return " | ".join(parts)


def exit_display_parts(
    ex: PortfolioExit, exit_ts: pd.Timestamp
) -> tuple[str, str]:
    """Headline and detail for UI (one date in the headline, not repeated in detail)."""
    when_s = rrg_format_date(exit_ts)
    if ex.rule == RULE_9EMA:
        if ex.timing == TIMING_MIDWEEK:
            return f"9 EMA @ {when_s}", "close below 9 EMA"
        return f"9 EMA @ {when_s}", "close below 9 EMA at rebalance"
    if ex.rule == RULE_STRATEGY:
        return "Strategy @ rebalance", ex.detail
    if ex.rule == RULE_RANK:
        return f"Rank @ {when_s}", ex.detail
    return ex.rule, ex.detail


def format_exit_line(ex: PortfolioExit) -> str:
    when = ex.timing
    if ex.timing == TIMING_MIDWEEK and " on " in ex.detail:
        when = ex.detail.split(" on ", 1)[-1]
    elif ex.timing == TIMING_REBALANCE:
        when = "rebal"
    return f"{ex.ticker}: {ex.rule} ({when})"


def _top_n_pick_set(rebalance_tickers: list[str] | None) -> set[str] | None:
    if not rebalance_tickers:
        return None
    from momentum.rrg_portfolio_panel import norm_ticker

    return {norm_ticker(t) for t in rebalance_tickers if t}


def mid_week_9ema_label(
    mid_week_exits: list[tuple[str, pd.Timestamp]],
    *,
    rebalance_tickers: list[str] | None = None,
) -> str:
    """Mid-week 9 EMA exits for this week's Top N picks only (not prior Was holdings)."""
    if not mid_week_exits:
        return ""
    pick_set = _top_n_pick_set(rebalance_tickers)
    parts: list[str] = []
    for ticker, day in mid_week_exits:
        bare = ticker.strip().upper().replace(".NS", "")
        if not bare or (pick_set is not None and bare not in pick_set):
            continue
        parts.append(f"{bare}@{rrg_format_date(day)}")
    return ", ".join(parts)


def mid_week_stop_loss_label(
    mid_week_exits: list[tuple[str, pd.Timestamp]],
    *,
    rebalance_tickers: list[str] | None = None,
) -> str:
    """Mid-week stop-loss exits for this week's Top N picks only."""
    if not mid_week_exits:
        return ""
    pick_set = _top_n_pick_set(rebalance_tickers)
    parts: list[str] = []
    for ticker, day in mid_week_exits:
        bare = ticker.strip().upper().replace(".NS", "")
        if not bare or (pick_set is not None and bare not in pick_set):
            continue
        parts.append(f"{bare}@{rrg_format_date(day)}")
    return ", ".join(parts)


def rebal_9ema_label(
    dropped_at_rebal: list[str],
    rebalance_ts: pd.Timestamp,
    *,
    rebalance_tickers: list[str] | None = None,
) -> str:
    """Top N picks skipped at rebalance (close < 9 EMA); excludes rotated-out Was names."""
    if not dropped_at_rebal:
        return ""
    pick_set = _top_n_pick_set(rebalance_tickers)
    rebal_d = rrg_format_date(rebalance_ts)
    parts: list[str] = []
    seen: set[str] = set()
    for ticker in dropped_at_rebal:
        bare = ticker.strip().upper().replace(".NS", "")
        if not bare or bare in seen:
            continue
        if pick_set is not None and bare not in pick_set:
            continue
        seen.add(bare)
        parts.append(f"{bare}@{rebal_d}")
    return ", ".join(parts)


def nine_ema_out_label(
    dropped_at_rebal: list[str],
    mid_week_exits: list[tuple[str, pd.Timestamp]],
    rebalance_ts: pd.Timestamp,
) -> str:
    """Combined 9 EMA exits (rebal + mid); prefer separate log columns when possible."""
    rebal = rebal_9ema_label(dropped_at_rebal, rebalance_ts)
    mid = mid_week_9ema_label(mid_week_exits)
    if rebal and mid:
        return f"{rebal} | mid: {mid}"
    return rebal or mid


def filter_exits_through(
    exits: list[PortfolioExit],
    *,
    rebalance_ts: pd.Timestamp,
    through_ts: pd.Timestamp,
) -> list[PortfolioExit]:
    """Only exits on or before ``through_ts`` (slider as-of date)."""
    through = pd.Timestamp(through_ts)
    return [
        ex
        for ex in exits
        if exit_event_timestamp(ex, rebalance_ts) <= through
    ]


def exits_as_of_through_date(
    week_slices: list[tuple[pd.Timestamp, list[PortfolioExit]]],
    through_ts: pd.Timestamp,
) -> list[PortfolioExit]:
    """Merge rebalance-week exit lists; keep first reason per ticker (mid-week before rebal)."""
    through = pd.Timestamp(through_ts)
    groups: list[list[PortfolioExit]] = []
    for rebalance_ts, exits in week_slices:
        groups.append(
            filter_exits_through(exits, rebalance_ts=rebalance_ts, through_ts=through)
        )
    return merge_week_exits(*groups) if groups else []


def _bare_ticker(sym: str) -> str:
    return sym.strip().upper().replace(".NS", "")


def panel_was_out_exits(
    week_slices: list[tuple[pd.Timestamp, list[PortfolioExit]]],
    through_ts: pd.Timestamp,
    *,
    prev_rebal_ts: pd.Timestamp | None,
    panel_rebal_ts: pd.Timestamp,
    prev_holdings: list[str],
    rebalance_holdings: list[str],
    exit_below_9ema: bool,
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
) -> list[PortfolioExit]:
    """
    Was OUT rows: rebalance-week exits plus mid 9 EMA recomputed over the Was
    hold window (prev rebalance → current rebalance). Avoids stale cached mid
    dates from merged prior-week exit lists.
    """
    through = pd.Timestamp(through_ts)
    rebal_ts = pd.Timestamp(panel_rebal_ts)
    cur_exits: list[PortfolioExit] = []
    if week_slices:
        cur_rebal, cur_list = week_slices[-1]
        cur_exits = filter_exits_through(
            cur_list, rebalance_ts=cur_rebal, through_ts=through
        )
    base = filter_exits_portfolio_panel(
        cur_exits,
        prev_holdings=prev_holdings,
        rebalance_holdings=rebalance_holdings,
    )
    if not exit_below_9ema or prev_rebal_ts is None or daily_for_ticker is None:
        return base

    prev_ts = pd.Timestamp(prev_rebal_ts)
    rebal_set = {_bare_ticker(t) for t in rebalance_holdings if t}
    out_order = [
        _bare_ticker(t)
        for t in prev_holdings
        if t and _bare_ticker(t) not in rebal_set
    ]
    if not out_order:
        return base

    from momentum.rrg_ema_exit import first_9ema_exit_day

    by_ticker = {_bare_ticker(ex.ticker): ex for ex in base}
    resolved: list[PortfolioExit] = []
    for key in out_order:
        daily = daily_for_ticker(key)
        exit_day = None
        if daily is not None and len(daily):
            exit_day = first_9ema_exit_day(daily, prev_ts, rebal_ts)
        if exit_day is not None:
            resolved.append(
                PortfolioExit(
                    ticker=key,
                    rule=RULE_9EMA,
                    timing=TIMING_MIDWEEK,
                    detail=f"close below 9 EMA on {rrg_format_date(exit_day)}",
                )
            )
        elif key in by_ticker:
            resolved.append(by_ticker[key])
        else:
            resolved.append(
                PortfolioExit(
                    ticker=key,
                    rule=RULE_STRATEGY,
                    timing=TIMING_REBALANCE,
                    detail="not in rebalance picks",
                )
            )
    return resolved


def filter_exits_portfolio_panel(
    exits: list[PortfolioExit],
    *,
    prev_holdings: list[str],
    rebalance_holdings: list[str],
) -> list[PortfolioExit]:
    """
    Exits list for the portfolio panel: only names in Was (prior Top N) that
  are not in this week's Top N @ rebalance (same set as Move = OUT).
    """
    rebal_set = {_bare_ticker(t) for t in rebalance_holdings if t}
    out_order = [
        _bare_ticker(t) for t in prev_holdings if t and _bare_ticker(t) not in rebal_set
    ]
    if not out_order:
        return []
    out_set = set(out_order)
    by_ticker: dict[str, PortfolioExit] = {}
    for ex in exits:
        key = _bare_ticker(ex.ticker)
        if key in out_set and key not in by_ticker:
            by_ticker[key] = ex
    out: list[PortfolioExit] = []
    for key in out_order:
        if key in by_ticker:
            out.append(by_ticker[key])
        else:
            out.append(
                PortfolioExit(
                    ticker=key,
                    rule=RULE_STRATEGY,
                    timing=TIMING_REBALANCE,
                    detail="not in rebalance picks",
                )
            )
    return out


def format_exits_multiline(
    exits: list[PortfolioExit], *, rebalance_ts: pd.Timestamp
) -> str:
    if not exits:
        return "No exits this week."
    rebal = pd.Timestamp(rebalance_ts)
    lines: list[str] = []
    for ex in exits:
        exit_ts = exit_event_timestamp(ex, rebal)
        head, detail = exit_display_parts(ex, exit_ts)
        lines.append(f"{ex.ticker}  —  {head}  —  {detail}")
    return "\n".join(lines)


def exit_event_timestamp(
    ex: PortfolioExit, rebalance_ts: pd.Timestamp
) -> pd.Timestamp:
    """When the exit took effect (mid-week date for 9 EMA, else rebalance)."""
    if ex.rule == RULE_9EMA and ex.timing == TIMING_MIDWEEK and " on " in ex.detail:
        raw = ex.detail.split(" on ", 1)[-1].strip()
        try:
            return rrg_parse_user_date(raw)
        except (TypeError, ValueError):
            pass
    return pd.Timestamp(rebalance_ts)


def equal_weight_portfolio_pct(
    returns_pct: list[float | None],
    *,
    slots: int,
) -> float | None:
    """Arithmetic mean; empty Top N slots count as 0%% (cash)."""
    if slots <= 0:
        return None
    total = 0.0
    for i in range(slots):
        r = returns_pct[i] if i < len(returns_pct) else None
        total += 0.0 if r is None else float(r)
    return total / slots


def _exit_ts_by_ticker(
    exits: list[PortfolioExit],
    rebalance_ts: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    rebal = pd.Timestamp(rebalance_ts)
    return {
        _bare_ticker(ex.ticker): exit_event_timestamp(ex, rebal) for ex in exits
    }


def was_portfolio_returns_pct(
    was_list: list[str],
    *,
    rebal_holdings: set[str],
    panel_exits: list[PortfolioExit],
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    as_of_ts: pd.Timestamp,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> list[float | None]:
    """Per-name %% return for prior-week portfolio (OUT at exit, HOLD through as-of)."""
    entry_ts = (
        pd.Timestamp(prev_rebalance_ts)
        if prev_rebalance_ts is not None
        else pd.Timestamp(rebalance_ts)
    )
    as_of = pd.Timestamp(as_of_ts)
    rebal = pd.Timestamp(rebalance_ts)
    exit_ts = _exit_ts_by_ticker(panel_exits, rebal)
    out: list[float | None] = []
    for sym in was_list:
        bare = _bare_ticker(sym)
        if not bare:
            continue
        mark_ts = as_of if bare in rebal_holdings else exit_ts.get(bare, rebal)
        weekly = weekly_for_ticker(sym)
        daily = daily_for_ticker(sym) if daily_for_ticker else None
        daily = _daily_for_pick_pnl(
            daily, bare, mark_ts, live_close_by_ticker=live_close_by_ticker
        )
        out.append(holding_return_pct(weekly, daily, entry_ts, mark_ts))
    return out


def pick_portfolio_returns_pct(
    rebal_slot_tickers: list[str],
    *,
    prev_portfolio: list[str],
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    as_of_ts: pd.Timestamp,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    mid_week_exit_ts: dict[str, pd.Timestamp] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> list[float | None]:
    """Per-slot %% for this week's Top N (empty slot → None → 0 in EW mean)."""
    rebal = pd.Timestamp(rebalance_ts)
    as_of = pd.Timestamp(as_of_ts)
    mid_exit = mid_week_exit_ts or {}
    prev_set = {_bare_ticker(t) for t in prev_portfolio if t}
    out: list[float | None] = []
    for sym in rebal_slot_tickers:
        if not sym:
            out.append(None)
            continue
        bare = _bare_ticker(sym)
        if prev_rebalance_ts is not None and bare in prev_set:
            entry_ts = pd.Timestamp(prev_rebalance_ts)
        else:
            entry_ts = rebal
        exit_ts = mid_exit.get(bare)
        mark_ts = exit_ts if exit_ts is not None and exit_ts <= as_of else as_of
        weekly = weekly_for_ticker(sym)
        daily = daily_for_ticker(sym) if daily_for_ticker else None
        daily = _daily_for_pick_pnl(
            daily, bare, mark_ts, live_close_by_ticker=live_close_by_ticker
        )
        out.append(holding_return_pct(weekly, daily, entry_ts, mark_ts))
    return out


def compute_portfolio_panel_totals(
    *,
    was_list: list[str],
    rebal_slot_tickers: list[str],
    prev_portfolio: list[str],
    panel_exits: list[PortfolioExit],
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    as_of_ts: pd.Timestamp,
    portfolio_slots: int,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    mid_week_exit_ts: dict[str, pd.Timestamp] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> PortfolioPanelTotals:
    """Equal-weight totals for Was portfolio and Top N pick portfolio."""
    was_names = [t for t in was_list if t]
    rebal_holdings = {_bare_ticker(t) for t in rebal_slot_tickers if t}
    was_slots = len(was_names)
    pick_slots = max(int(portfolio_slots), len(rebal_slot_tickers), 1)

    was_rets = was_portfolio_returns_pct(
        was_names,
        rebal_holdings=rebal_holdings,
        panel_exits=panel_exits,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        as_of_ts=as_of_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        live_close_by_ticker=live_close_by_ticker,
    )
    pick_rets = pick_portfolio_returns_pct(
        rebal_slot_tickers,
        prev_portfolio=prev_portfolio,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        as_of_ts=as_of_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        mid_week_exit_ts=mid_week_exit_ts,
        live_close_by_ticker=live_close_by_ticker,
    )
    return PortfolioPanelTotals(
        was_ew_pct=equal_weight_portfolio_pct(was_rets, slots=was_slots)
        if was_slots
        else None,
        was_slots=was_slots,
        pick_ew_pct=equal_weight_portfolio_pct(pick_rets, slots=pick_slots),
        pick_slots=pick_slots,
    )


def format_ew_total_pct(pct: float | None) -> str:
    return f"{pct:+.2f}%" if pct is not None else "—"


def holding_return_pct(
    weekly: pd.Series,
    daily: pd.Series | None,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
) -> float | None:
    """Total-return % from entry to exit using one price source (daily CM close preferred)."""
    try:
        from momentum.rrg_ranking import series_at

        # Same series for entry and exit — mixing index EOD (weekly) with ETF bhavcopy
        # (daily) produced bogus ~-99% prints on India rows that map index → ref ETF.
        if daily is not None and len(daily):
            p0 = float(series_at(daily, entry_ts))
            p1 = float(series_at(daily, exit_ts))
        elif weekly is not None and len(weekly):
            p0 = float(series_at(weekly, entry_ts))
            p1 = float(series_at(weekly, exit_ts))
        else:
            return None
        if p0 <= 0:
            return None
        return (p1 / p0 - 1.0) * 100.0
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def exit_detail_and_pnl_maps(
    exits: list[PortfolioExit],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Per-ticker P/L labels for portfolio panel OUT rows."""
    rebal = pd.Timestamp(rebalance_ts)
    entry_ts = (
        pd.Timestamp(prev_rebalance_ts)
        if prev_rebalance_ts is not None
        else rebal
    )
    detail_by: dict[str, str] = {}
    pnl_by: dict[str, str] = {}
    for ex in exits:
        bare = _bare_ticker(ex.ticker)
        exit_ts = exit_event_timestamp(ex, rebal)
        head, detail = exit_display_parts(ex, exit_ts)
        detail_by[bare] = f"{head} — {detail}"
        weekly = weekly_for_ticker(ex.ticker)
        daily = daily_for_ticker(ex.ticker) if daily_for_ticker else None
        pnl = holding_return_pct(weekly, daily, entry_ts, exit_ts)
        pnl_by[bare] = f"{pnl:+.2f}%" if pnl is not None else "—"
    return detail_by, pnl_by


def mid_week_exit_ts_map(
    mid_week_exits: list[tuple[str, pd.Timestamp]] | list,
    *,
    rebalance_tickers: list[str] | None = None,
) -> dict[str, pd.Timestamp]:
    """Per Top N pick: mid-week 9 EMA exit timestamp (if any)."""
    pick_set = _top_n_pick_set(rebalance_tickers)
    out: dict[str, pd.Timestamp] = {}
    for item in mid_week_exits:
        if not item:
            continue
        ticker, day = item[0], item[1]
        bare = _bare_ticker(ticker)
        if not bare or (pick_set is not None and bare not in pick_set):
            continue
        out[bare] = pd.Timestamp(day)
    return out


def format_close_price(px: float | None) -> str:
    return f"{px:,.2f}" if px is not None else "—"


def close_price_at(
    weekly: pd.Series,
    daily: pd.Series | None,
    ts: pd.Timestamp,
    *,
    bare: str,
    live_close_by_ticker: dict[str, float] | None = None,
) -> float | None:
    """ETF/index close on ``ts`` (daily CM preferred; live overlay when applicable)."""
    from momentum.rrg_ranking import series_at

    mark = pd.Timestamp(ts)
    daily = _daily_for_pick_pnl(
        daily, bare, mark, live_close_by_ticker=live_close_by_ticker
    )
    try:
        if daily is not None and len(daily):
            return float(series_at(daily, mark))
        if weekly is not None and len(weekly):
            return float(series_at(weekly, mark))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return None


def was_close_map(
    was_list: list[str],
    *,
    rebal_holdings: set[str],
    panel_exits: list[PortfolioExit],
    rebalance_ts: pd.Timestamp,
    as_of_ts: pd.Timestamp,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> dict[str, str]:
    """Per Was name: close at exit (OUT) or as-of (HOLD)."""
    rebal = pd.Timestamp(rebalance_ts)
    as_of = pd.Timestamp(as_of_ts)
    exit_ts = _exit_ts_by_ticker(panel_exits, rebal)
    out: dict[str, str] = {}
    for sym in was_list:
        bare = _bare_ticker(sym)
        if not bare:
            continue
        mark = as_of if bare in rebal_holdings else exit_ts.get(bare, rebal)
        weekly = weekly_for_ticker(sym)
        daily = daily_for_ticker(sym) if daily_for_ticker else None
        px = close_price_at(
            weekly,
            daily,
            mark,
            bare=bare,
            live_close_by_ticker=live_close_by_ticker,
        )
        out[bare] = format_close_price(px)
    return out


def was_entry_close_map(
    was_list: list[str],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
) -> dict[str, str]:
    """Per Was name: close when the position was entered (prior rebalance)."""
    entry_ts = (
        pd.Timestamp(prev_rebalance_ts)
        if prev_rebalance_ts is not None
        else pd.Timestamp(rebalance_ts)
    )
    out: dict[str, str] = {}
    for sym in was_list:
        bare = _bare_ticker(sym)
        if not bare:
            continue
        weekly = weekly_for_ticker(sym)
        daily = daily_for_ticker(sym) if daily_for_ticker else None
        px = close_price_at(weekly, daily, entry_ts, bare=bare)
        out[bare] = format_close_price(px)
    return out


def pick_entry_close_map(
    rows: list[dict[str, str]],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
) -> dict[str, str]:
    """Per Top N pick: close when entered (NEW at rebalance, KEEP at prior rebalance)."""
    rebal = pd.Timestamp(rebalance_ts)
    out: dict[str, str] = {}
    for row in rows:
        ticker = row.get("rebal", "")
        if not ticker:
            continue
        bare = _bare_ticker(ticker)
        if not bare:
            continue
        if row.get("pick") == "KEEP" and prev_rebalance_ts is not None:
            entry_ts = pd.Timestamp(prev_rebalance_ts)
        else:
            entry_ts = rebal
        weekly = weekly_for_ticker(ticker)
        daily = daily_for_ticker(ticker) if daily_for_ticker else None
        px = close_price_at(weekly, daily, entry_ts, bare=bare)
        out[bare] = format_close_price(px)
    return out


def pick_close_map(
    rows: list[dict[str, str]],
    *,
    rebalance_ts: pd.Timestamp,
    as_of_ts: pd.Timestamp,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    mid_week_exit_ts: dict[str, pd.Timestamp] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> dict[str, str]:
    """Per Top N pick: close at 9 EMA exit or as-of (same date as P/L pick)."""
    rebal = pd.Timestamp(rebalance_ts)
    as_of = pd.Timestamp(as_of_ts)
    mid_exit = mid_week_exit_ts or {}
    out: dict[str, str] = {}
    for row in rows:
        ticker = row.get("rebal", "")
        if not ticker:
            continue
        bare = _bare_ticker(ticker)
        if not bare:
            continue
        exit_ts = mid_exit.get(bare)
        mark = exit_ts if exit_ts is not None and exit_ts <= as_of else as_of
        weekly = weekly_for_ticker(ticker)
        daily = daily_for_ticker(ticker) if daily_for_ticker else None
        px = close_price_at(
            weekly,
            daily,
            mark,
            bare=bare,
            live_close_by_ticker=live_close_by_ticker,
        )
        out[bare] = format_close_price(px)
    return out


def _daily_for_pick_pnl(
    daily: pd.Series | None,
    bare: str,
    as_of_ts: pd.Timestamp,
    *,
    live_close_by_ticker: dict[str, float] | None,
) -> pd.Series | None:
    """Optional live close on ``as_of`` when marking picks to latest session."""
    if daily is None or not len(daily) or not live_close_by_ticker:
        return daily
    px = live_close_by_ticker.get(bare)
    if px is None or px <= 0:
        return daily
    as_of = pd.Timestamp(as_of_ts).normalize()
    last_bar = pd.Timestamp(daily.index[-1]).normalize()
    if as_of <= last_bar:
        return daily
    s = daily.copy()
    s.loc[as_of] = float(px)
    return s.sort_index()


def pick_pnl_map(
    rows: list[dict[str, str]],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    as_of_ts: pd.Timestamp,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    mid_week_exit_ts: dict[str, pd.Timestamp] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> dict[str, str]:
    """
    Mark-to-market % for Top N picks from entry through ``as_of_ts`` (date slider).

    NEW: from this rebalance; KEEP: from prior rebalance when available.
    Mid-week 9 EMA exit caps the exit date when it occurs on or before ``as_of_ts``.
    """
    rebal = pd.Timestamp(rebalance_ts)
    as_of = pd.Timestamp(as_of_ts)
    mid_exit = mid_week_exit_ts or {}
    out: dict[str, str] = {}
    for row in rows:
        ticker = row.get("rebal", "")
        if not ticker:
            continue
        bare = _bare_ticker(ticker)
        if not bare:
            continue
        pick_tag = row.get("pick", "")
        if pick_tag == "KEEP" and prev_rebalance_ts is not None:
            entry_ts = pd.Timestamp(prev_rebalance_ts)
        else:
            entry_ts = rebal
        exit_ts = mid_exit.get(bare)
        mark_ts = (
            exit_ts
            if exit_ts is not None and exit_ts <= as_of
            else as_of
        )
        weekly = weekly_for_ticker(ticker)
        daily = daily_for_ticker(ticker) if daily_for_ticker else None
        daily = _daily_for_pick_pnl(
            daily, bare, mark_ts, live_close_by_ticker=live_close_by_ticker
        )
        pnl = holding_return_pct(weekly, daily, entry_ts, mark_ts)
        out[bare] = f"{pnl:+.2f}%" if pnl is not None else "—"
    return out


def holding_pnl_map(
    tickers: list[str],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    as_of_ts: pd.Timestamp | None = None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
) -> dict[str, str]:
    """Return % on Was HOLD from prior rebalance through ``as_of_ts`` (date slider)."""
    entry_ts = (
        pd.Timestamp(prev_rebalance_ts)
        if prev_rebalance_ts is not None
        else pd.Timestamp(rebalance_ts)
    )
    mark_ts = (
        pd.Timestamp(as_of_ts) if as_of_ts is not None else pd.Timestamp(rebalance_ts)
    )
    out: dict[str, str] = {}
    for sym in tickers:
        bare = _bare_ticker(sym)
        if not bare:
            continue
        weekly = weekly_for_ticker(sym)
        daily = daily_for_ticker(sym) if daily_for_ticker else None
        daily = _daily_for_pick_pnl(
            daily, bare, mark_ts, live_close_by_ticker=live_close_by_ticker
        )
        pnl = holding_return_pct(weekly, daily, entry_ts, mark_ts)
        out[bare] = f"{pnl:+.2f}%" if pnl is not None else "—"
    return out


def format_exits_multiline_with_pnl(
    exits: list[PortfolioExit],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
) -> str:
    if not exits:
        return "No exits this week."
    lines: list[str] = []
    entry_ts = (
        pd.Timestamp(prev_rebalance_ts)
        if prev_rebalance_ts is not None
        else pd.Timestamp(rebalance_ts)
    )
    for ex in exits:
        exit_ts = exit_event_timestamp(ex, rebalance_ts)
        weekly = weekly_for_ticker(ex.ticker)
        daily = daily_for_ticker(ex.ticker) if daily_for_ticker else None
        pnl = holding_return_pct(weekly, daily, entry_ts, exit_ts)
        pnl_s = f"{pnl:+.2f}%" if pnl is not None else "—"
        head, detail = exit_display_parts(ex, exit_ts)
        lines.append(f"{ex.ticker}  —  {head}  —  {detail}  ·  P&L {pnl_s}")
    return "\n".join(lines)
