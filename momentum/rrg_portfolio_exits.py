"""Portfolio exit events (rank vs 9 EMA) for RRG ETF swing picks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

RULE_RANK = "Rank"
RULE_9EMA = "9 EMA"
RULE_STRATEGY = "Strategy"

TIMING_REBALANCE = "Rebalance"
TIMING_MIDWEEK = "Mid-week"


@dataclass(frozen=True)
class PortfolioExit:
    ticker: str
    rule: str
    timing: str
    detail: str


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
    date_s = pd.Timestamp(as_of).strftime("%Y-%m-%d")
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
                detail=f"close below 9 EMA on {pd.Timestamp(day).strftime('%Y-%m-%d')}",
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
    already = {ex.ticker for ex in rank_ex + ema_rebal + ema_mid}
    strat: list[PortfolioExit] = []
    if include_strategy_exits:
        strat = strategy_rebalance_exits(
            prev_holdings, rebalance_holdings, already_exited=already
        )
    return merge_week_exits(rank_ex, ema_rebal, ema_mid, strat)


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
    when_s = pd.Timestamp(exit_ts).strftime("%Y-%m-%d")
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
        parts.append(f"{bare}@{pd.Timestamp(day).strftime('%Y-%m-%d')}")
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
    rebal_d = pd.Timestamp(rebalance_ts).strftime("%Y-%m-%d")
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
            return pd.Timestamp(raw)
        except (TypeError, ValueError):
            pass
    return pd.Timestamp(rebalance_ts)


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


def holding_pnl_map(
    tickers: list[str],
    *,
    rebalance_ts: pd.Timestamp,
    prev_rebalance_ts: pd.Timestamp | None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None = None,
) -> dict[str, str]:
    """Return % on Was (held) from prior rebalance through this rebalance bar (still held)."""
    rebal = pd.Timestamp(rebalance_ts)
    entry_ts = (
        pd.Timestamp(prev_rebalance_ts)
        if prev_rebalance_ts is not None
        else rebal
    )
    out: dict[str, str] = {}
    for sym in tickers:
        bare = _bare_ticker(sym)
        if not bare:
            continue
        weekly = weekly_for_ticker(sym)
        daily = daily_for_ticker(sym) if daily_for_ticker else None
        pnl = holding_return_pct(weekly, daily, entry_ts, rebal)
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
