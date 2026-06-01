"""Portfolio Was / Now / Top N panel rows (shared by main RRG UI and backtest)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

PORTFOLIO_PANEL_HEADERS: tuple[tuple[str, str, str, int], ...] = (
    ("rank", "#", "e", 28),
    ("was", "Was (held)", "w", 110),
    ("now", "Now (held)", "w", 168),
    ("tag", "Move", "w", 44),
    ("pnl", "P/L", "e", 56),
    ("rebal", "Top N pick", "w", 118),
    ("pick_tag", "Pick", "w", 40),
    ("mid_9ema", "9EMA mid", "w", 72),
)

PORTFOLIO_PANEL_GRID_KEYS: tuple[str, ...] = tuple(k for k, *_ in PORTFOLIO_PANEL_HEADERS)


def norm_ticker(sym: str) -> str:
    return (sym or "").strip().upper().replace(".NS", "")


def compact_tickers(tickers: list[str]) -> list[str]:
    """Holdings in order, omitting empty rebalance slots."""
    return [t for t in tickers if t]


def format_portfolio_cell(ticker: str, rank: int | None = None) -> str:
    if not ticker:
        return ""
    if rank is not None:
        return f"{ticker} (rk {rank})"
    return ticker


def format_rebal_pick_cell(
    ticker: str,
    rank: int | None,
    *,
    held_at_rebal: set[str] | None,
) -> str:
    """Top N pick cell (9 EMA exclusions use ``format_rebal_skip_cell``)."""
    return format_portfolio_cell(ticker, rank)


def format_rebal_skip_cell(
    ticker: str,
    rank: int | None,
    *,
    rebalance_ts,
    reason: str = "9 EMA",
) -> str:
    """Strategy pick skipped at rebalance (e.g. below 9 EMA)."""
    reb_d = pd.Timestamp(rebalance_ts).strftime("%Y-%m-%d")
    name = format_portfolio_cell(ticker, rank)
    return f"{name} · {reason} @{reb_d}"


def build_rebal_display_rows(
    strategy_tickers: list[str],
    rebal_slots: list[str],
    *,
    rebalance_ts,
    rank_for_ticker: Callable[[str], int | None] | None = None,
    exit_below_9ema: bool = False,
) -> list[tuple[str, str]]:
    """
    Top N column rows in strategy order: entered tickers and skipped-pick labels.
    Returns list of (kind, value) where kind is ``ticker`` or ``skip``.
    """
    rank_fn = rank_for_ticker or (lambda _t: None)
    entered = compact_tickers(rebal_slots)
    if not exit_below_9ema:
        source = entered if entered else compact_tickers(strategy_tickers)
        return [("ticker", t) for t in source]

    n = max(len(strategy_tickers), len(rebal_slots))
    out: list[tuple[str, str]] = []
    for slot in range(n):
        strat = strategy_tickers[slot] if slot < len(strategy_tickers) else ""
        picked = rebal_slots[slot] if slot < len(rebal_slots) else ""
        if picked:
            out.append(("ticker", picked))
        elif strat:
            out.append(
                (
                    "skip",
                    format_rebal_skip_cell(
                        strat, rank_fn(strat), rebalance_ts=rebalance_ts
                    ),
                )
            )
    return out


def portfolio_panel_rows(
    prev_portfolio: list[str],
    rebal_display: list[tuple[str, str]],
    *,
    rebal_entered: list[str],
    end_prev_week_holdings: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Simple paired rows: last week's holdings (Was/Now/Move) beside this rebalance (Top N).

    Was / Now: prior-week holdings only (no blank rows for empty last-week slots).
    Move: HOLD or OUT for Was names only (never NEW).
    Top N: entered picks + skipped-pick labels; Pick NEW/KEEP vs last week.
    """
    was_list = compact_tickers(prev_portfolio)
    rebal_set = {norm_ticker(t) for t in rebal_entered}
    prev_set = {norm_ticker(t) for t in was_list}
    if end_prev_week_holdings is not None:
        end_set = {norm_ticker(t) for t in end_prev_week_holdings if t}
    else:
        end_set = {norm_ticker(t) for t in was_list}

    n_rows = max(len(was_list), len(rebal_display), 1)
    rows: list[dict[str, str]] = []
    for i in range(n_rows):
        was_t = was_list[i] if i < len(was_list) else ""
        rebal_t = ""
        rebal_note = ""
        if i < len(rebal_display):
            kind, val = rebal_display[i]
            if kind == "ticker":
                rebal_t = val
            else:
                rebal_note = val

        if was_t:
            now_t = was_t if norm_ticker(was_t) in end_set else ""
            move = "HOLD" if norm_ticker(was_t) in rebal_set else "OUT"
        else:
            now_t = ""
            move = ""

        if rebal_t:
            pick = "NEW" if norm_ticker(rebal_t) not in prev_set else "KEEP"
        else:
            pick = ""

        rows.append(
            {
                "was": was_t,
                "now": now_t,
                "move": move,
                "rebal": rebal_t,
                "rebal_note": rebal_note,
                "pick": pick,
                "pnl": "",
                "mid_9ema": "",
            }
        )
    return rows


def mid_week_9ema_cell_map(
    mid_week_exits: list[tuple[str, pd.Timestamp]] | list,
    *,
    rebalance_tickers: list[str] | None = None,
) -> dict[str, str]:
    """Per Top N pick: mid-week 9 EMA exit date for panel column."""
    pick_set = {norm_ticker(t) for t in rebalance_tickers if t} if rebalance_tickers else None
    out: dict[str, str] = {}
    for item in mid_week_exits:
        if not item:
            continue
        ticker, day = item[0], item[1]
        bare = norm_ticker(ticker)
        if not bare or (pick_set is not None and bare not in pick_set):
            continue
        out[bare] = f"@{pd.Timestamp(day).strftime('%Y-%m-%d')}"
    return out


def enrich_portfolio_panel_rows(
    rows: list[dict[str, str]],
    *,
    pnl_by_ticker: dict[str, str],
    hold_pnl_by_ticker: dict[str, str] | None = None,
    mid_9ema_by_ticker: dict[str, str] | None = None,
    exit_detail_by_ticker: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    P/L on Was (held): OUT through exit; HOLD through rebalance bar.
    Now (held): exit reason for OUT (9 EMA mid-week or strategy @ rebalance).
    """
    hold_pnl = hold_pnl_by_ticker or {}
    mid_9ema = mid_9ema_by_ticker or {}
    exit_detail = exit_detail_by_ticker or {}
    for row in rows:
        move = row["move"]
        was_t = row["was"]
        rebal_t = row.get("rebal", "")
        if rebal_t:
            row["mid_9ema"] = mid_9ema.get(norm_ticker(rebal_t), "")
        else:
            row["mid_9ema"] = ""
        if move == "HOLD" and was_t:
            row["pnl"] = hold_pnl.get(norm_ticker(was_t), "—")
            row["now_exit"] = ""
        elif move == "OUT" and was_t:
            bare = norm_ticker(was_t)
            row["pnl"] = pnl_by_ticker.get(bare, "—")
            row["now_exit"] = exit_detail.get(
                bare, "Strategy @ rebalance — not in rebalance picks"
            )
        else:
            row["pnl"] = ""
            row["now_exit"] = ""
    return rows


def apply_portfolio_panel_display(
    rows: list[dict[str, str]],
    *,
    was_rank_for_ticker: Callable[[str], int | None],
    curr_rank_for_ticker: Callable[[str], int | None],
) -> list[dict[str, str]]:
    """Add was_text / now_text / rebal_text and fg colors for UI binding."""
    for row in rows:
        was_t = row["was"]
        now_t = row["now"]
        move = row["move"]
        rebal_t = row["rebal"]
        now_exit = row.get("now_exit", "")
        rebal_note = row.get("rebal_note", "")
        row["was_text"] = (
            format_portfolio_cell(was_t, was_rank_for_ticker(was_t)) if was_t else ""
        )
        if now_exit:
            row["now_text"] = now_exit
        elif now_t:
            row["now_text"] = format_portfolio_cell(
                now_t, was_rank_for_ticker(now_t)
            )
        else:
            row["now_text"] = ""
        if rebal_t:
            row["rebal_text"] = format_portfolio_cell(
                rebal_t, curr_rank_for_ticker(rebal_t)
            )
        elif rebal_note:
            row["rebal_text"] = rebal_note
        else:
            row["rebal_text"] = ""
        row["now_fg"] = "#5d4037" if move == "OUT" else "black"
        row["rebal_fg"] = "#5d4037" if rebal_note else "black"
        row["mid_fg"] = "#5d4037" if row.get("mid_9ema") else "black"
    return rows


def build_portfolio_panel(
    *,
    prev_portfolio: list[str],
    rebal_strategy: list[str],
    rebal_tickers: list[str],
    end_prev_week_holdings: list[str] | None,
    panel_exits: list,
    rebalance_ts,
    prev_rebalance_ts: pd.Timestamp | None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None,
    was_rank_for_ticker: Callable[[str], int | None],
    curr_rank_for_ticker: Callable[[str], int | None],
    exit_below_9ema: bool,
    mid_week_9ema: list | None = None,
) -> list[dict[str, str]]:
    """Full portfolio panel rows (main RRG UI and backtest Current week)."""
    from momentum.rrg_portfolio_exits import (
        exit_detail_and_pnl_maps,
        holding_pnl_map,
    )

    was_list = compact_tickers(prev_portfolio)
    rebal_entered = compact_tickers(rebal_tickers)
    rebal_display = build_rebal_display_rows(
        rebal_strategy,
        rebal_tickers,
        rebalance_ts=rebalance_ts,
        rank_for_ticker=curr_rank_for_ticker,
        exit_below_9ema=exit_below_9ema,
    )
    rows = portfolio_panel_rows(
        prev_portfolio,
        rebal_display,
        rebal_entered=rebal_entered,
        end_prev_week_holdings=end_prev_week_holdings,
    )
    exit_detail_by, pnl_by = exit_detail_and_pnl_maps(
        panel_exits,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
    )
    rebal_set = {norm_ticker(t) for t in rebal_entered}
    hold_pnl_by = holding_pnl_map(
        [t for t in was_list if norm_ticker(t) in rebal_set],
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
    )
    mid_9ema_by: dict[str, str] = {}
    if exit_below_9ema:
        mid_9ema_by = mid_week_9ema_cell_map(
            mid_week_9ema or [],
            rebalance_tickers=rebal_entered,
        )
    rows = enrich_portfolio_panel_rows(
        rows,
        pnl_by_ticker=pnl_by,
        hold_pnl_by_ticker=hold_pnl_by,
        mid_9ema_by_ticker=mid_9ema_by,
        exit_detail_by_ticker=exit_detail_by,
    )
    return apply_portfolio_panel_display(
        rows,
        was_rank_for_ticker=was_rank_for_ticker,
        curr_rank_for_ticker=curr_rank_for_ticker,
    )


def portfolio_panel_dates_line(
    *,
    rebalance_label: str,
    was_n: int,
    was_label: str,
    rebal_n: int,
    pick_shortfall: str = "",
    exit_below_9ema: bool = False,
    subtitle: str = "",
    exits_through_label: str | None = None,
) -> str:
    if exit_below_9ema:
        through = exits_through_label or rebalance_label
        exits_note = f" · Exits through {through}"
    else:
        exits_note = ""
    pick_part = pick_shortfall or f"Top N {rebal_n} this week"
    return (
        f"Rebalance {rebalance_label}  ·  Was {was_n} from {was_label}  ·  "
        f"{pick_part} (★ order){exits_note}  ·  {subtitle}"
    )
