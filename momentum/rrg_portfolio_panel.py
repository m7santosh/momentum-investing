"""Portfolio Was / Now / Top N panel rows (shared by main RRG UI and backtest)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from momentum.rrg_core import rrg_format_date
from momentum.rrg_portfolio_exits import PortfolioPanelTotals

PORTFOLIO_PANEL_WAS_ROW: tuple[tuple[str, str, str, int], ...] = (
    ("rank", "#", "e", 28),
    ("was", "Was", "w", 140),
    ("now", "Now (held)", "w", 200),
    ("tag", "Move", "w", 44),
    ("was_pnl", "P/L Was", "e", 56),
    ("was_entry", "Was entry", "e", 68),
    ("was_close", "Was close", "e", 72),
)

PORTFOLIO_PANEL_PICK_ROW: tuple[tuple[str, str, str, int], ...] = (
    ("rebal", "Top N", "w", 200),
    ("pick_tag", "Pick", "w", 40),
    ("pick_pnl", "P/L pick", "e", 56),
    ("pick_entry", "Pick entry", "e", 68),
    ("pick_close", "Pick close", "e", 72),
    ("mid_9ema", "9EMA mid", "w", 72),
)

PORTFOLIO_PANEL_PICK_GRID_ROW: tuple[tuple[str, str, str, int], ...] = (
    PORTFOLIO_PANEL_WAS_ROW[0],
    *PORTFOLIO_PANEL_PICK_ROW,
)

PORTFOLIO_PANEL_NUM_COLS = len(PORTFOLIO_PANEL_WAS_ROW)

# Backward-compatible flat header list (Was row + Pick row keys).
PORTFOLIO_PANEL_HEADERS: tuple[tuple[str, str, str, int], ...] = (
    *PORTFOLIO_PANEL_WAS_ROW,
    *PORTFOLIO_PANEL_PICK_ROW,
)

PORTFOLIO_PANEL_GRID_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        k for k, *_ in (*PORTFOLIO_PANEL_WAS_ROW, *PORTFOLIO_PANEL_PICK_ROW)
    )
)

PORTFOLIO_PANEL_WAS_COL = 1
PORTFOLIO_PANEL_REBAL_COL = 1

PORTFOLIO_PANEL_SMALL_KEYS = frozenset(
    {
        "tag",
        "pick_tag",
        "was_pnl",
        "was_entry",
        "was_close",
        "pick_pnl",
        "pick_entry",
        "pick_close",
        "mid_9ema",
    }
)

PORTFOLIO_PANEL_TAG_KEYS = frozenset({"tag", "pick_tag"})


def portfolio_panel_col_minsize(col: int) -> int:
    if col <= 0 or col >= PORTFOLIO_PANEL_NUM_COLS:
        return PORTFOLIO_PANEL_WAS_ROW[0][3]
    return max(
        PORTFOLIO_PANEL_WAS_ROW[col][3],
        PORTFOLIO_PANEL_PICK_ROW[col - 1][3],
    )


def portfolio_panel_col_weight(col: int) -> int:
    if col <= 0:
        return 0
    keys = (PORTFOLIO_PANEL_WAS_ROW[col][0], PORTFOLIO_PANEL_PICK_ROW[col - 1][0])
    return 1 if any(k in ("was", "now", "rebal") for k in keys) else 0


def configure_portfolio_panel_table_columns(table: tk.Misc) -> None:
    """Shared column widths so Was and Top N grids align vertically."""
    for col in range(PORTFOLIO_PANEL_NUM_COLS):
        table.columnconfigure(
            col,
            minsize=portfolio_panel_col_minsize(col),
            weight=portfolio_panel_col_weight(col),
        )


def portfolio_panel_was_header(was_label: str | None = None) -> str:
    """Column title for prior-week holdings (includes rebalance date when known)."""
    label = (was_label or "").strip()
    if label and label != "—":
        return f"Was - {label}"
    return "Was"


def portfolio_panel_pick_header(pick_label: str | None = None) -> str:
    """Column title for this week's Top N picks (includes rebalance date when known)."""
    label = (pick_label or "").strip()
    if label and label != "—":
        return f"Top N - {label}"
    return "Top N"


def norm_ticker(sym: str) -> str:
    return (sym or "").strip().upper().replace(".NS", "")


def live_close_for_panel(
    profile: str,
    *,
    etf_daily_close: dict[str, pd.Series] | None = None,
    at_latest_bar: bool = False,
) -> dict[str, float] | None:
    """
    Latest marks for panel P/L when the Date slider is on the last bar.

    India: NSE live LTP. US: last loaded Yahoo daily close per ticker.
    """
    if not at_latest_bar:
        return None
    if profile == "india":
        from utils.nse_bhavcopy import fetch_nse_live_quotes

        quotes = fetch_nse_live_quotes()
        if not quotes:
            return None
        return {
            sym: float(q["close"])
            for sym, q in quotes.items()
            if q.get("close")
        }
    if profile == "stock":
        from utils.nse_bhavcopy import fetch_nse_live_quotes

        quotes = fetch_nse_live_quotes()
        if not quotes:
            return None
        return {
            sym: float(q["close"])
            for sym, q in quotes.items()
            if q.get("close")
        }
    if profile == "us" and etf_daily_close:
        out: dict[str, float] = {}
        for sym, series in etf_daily_close.items():
            if series is not None and len(series):
                px = float(series.iloc[-1])
                if px > 0:
                    out[sym] = px
        return out or None
    return None


def pad_rebal_slots(tickers: list[str], portfolio_slots: int) -> list[str]:
    """Top N slot list with empty strings for skipped 9 EMA slots."""
    slots = list(tickers)
    n = max(int(portfolio_slots), len(slots))
    while len(slots) < n:
        slots.append("")
    return slots


def trading_days_for_asof(
    tickers: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    daily_for_ticker: Callable[[str], pd.Series | None],
) -> list[pd.Timestamp]:
    """Trading days between rebalance and week-end (for backtest as-of slider)."""
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    if end <= start:
        return [start]
    best: pd.DatetimeIndex | None = None
    for sym in tickers:
        if not sym:
            continue
        daily = daily_for_ticker(sym)
        if daily is None or not len(daily):
            continue
        sub = daily.loc[start:end].index
        if len(sub) and (best is None or len(sub) > len(best)):
            best = sub
    if best is None or not len(best):
        return [start, end]
    days = sorted(pd.Timestamp(t) for t in best.unique())
    if days[0] != start and start not in days:
        days = [start, *days]
    if days[-1] != end:
        days.append(end)
    return days


def compact_tickers(tickers: list[str]) -> list[str]:
    """Holdings in order, omitting empty rebalance slots."""
    return [t for t in tickers if t]


def format_portfolio_cell(
    ticker: str,
    rank: int | None = None,
    *,
    detail: str | None = None,
) -> str:
    if not ticker:
        return ""
    text = ticker
    extra = (detail or "").strip()
    if extra and extra.upper() != ticker.strip().upper():
        text = f"{ticker} — {extra}"
    if rank is not None:
        return f"{text} (rk {rank})"
    return text


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
    detail: str | None = None,
) -> str:
    """Strategy pick skipped at rebalance (e.g. below 9 EMA)."""
    reb_d = rrg_format_date(rebalance_ts)
    name = format_portfolio_cell(ticker, rank, detail=detail)
    return f"{name} · {reason} @{reb_d}"


def build_rebal_display_rows(
    strategy_tickers: list[str],
    rebal_slots: list[str],
    *,
    rebalance_ts,
    rank_for_ticker: Callable[[str], int | None] | None = None,
    detail_for_ticker: Callable[[str], str] | None = None,
    exit_below_9ema: bool = False,
) -> list[tuple[str, str]]:
    """
    Top N column rows in strategy order: entered tickers and skipped-pick labels.
    Returns list of (kind, value) where kind is ``ticker`` or ``skip``.
    """
    rank_fn = rank_for_ticker or (lambda _t: None)

    def _detail(sym: str) -> str | None:
        if detail_for_ticker is None:
            return None
        text = (detail_for_ticker(sym) or "").strip()
        return text or None

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
                        strat,
                        rank_fn(strat),
                        rebalance_ts=rebalance_ts,
                        detail=_detail(strat),
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
            move = "HOLD" if norm_ticker(was_t) in rebal_set else "OUT"
            if move == "HOLD" and norm_ticker(was_t) in end_set:
                now_t = was_t
            else:
                now_t = ""
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
                "was_pnl": "",
                "was_entry": "",
                "was_close": "",
                "pick_pnl": "",
                "pick_entry": "",
                "pick_close": "",
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
        out[bare] = f"@{rrg_format_date(day)}"
    return out


def enrich_portfolio_panel_rows(
    rows: list[dict[str, str]],
    *,
    pnl_by_ticker: dict[str, str],
    hold_pnl_by_ticker: dict[str, str] | None = None,
    pick_pnl_by_ticker: dict[str, str] | None = None,
    was_close_by_ticker: dict[str, str] | None = None,
    was_entry_by_ticker: dict[str, str] | None = None,
    pick_close_by_ticker: dict[str, str] | None = None,
    pick_entry_by_ticker: dict[str, str] | None = None,
    mid_9ema_by_ticker: dict[str, str] | None = None,
    exit_detail_by_ticker: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    P/L Was: prior-week OUT at exit; HOLD carried forward through as-of.
    P/L pick: Top N NEW/KEEP through as-of, or through 9 EMA mid-week exit.
    Now (held): exit reason text only (no P/L).
    """
    hold_pnl = hold_pnl_by_ticker or {}
    mid_9ema = mid_9ema_by_ticker or {}
    exit_detail = exit_detail_by_ticker or {}
    pick_pnl = pick_pnl_by_ticker or {}
    was_close = was_close_by_ticker or {}
    was_entry = was_entry_by_ticker or {}
    pick_close = pick_close_by_ticker or {}
    pick_entry = pick_entry_by_ticker or {}
    for row in rows:
        move = row["move"]
        was_t = row["was"]
        rebal_t = row.get("rebal", "")
        row["was_pnl"] = ""
        row["was_entry"] = ""
        row["was_close"] = ""
        row["pick_pnl"] = ""
        row["pick_entry"] = ""
        row["pick_close"] = ""
        row["mid_9ema"] = ""

        if move == "OUT" and was_t:
            bare = norm_ticker(was_t)
            row["was_pnl"] = pnl_by_ticker.get(bare, "—")
            row["was_entry"] = was_entry.get(bare, "—")
            row["was_close"] = was_close.get(bare, "—")
            row["now_exit"] = exit_detail.get(
                bare, "Strategy @ rebalance — not in rebalance picks"
            )
        elif move == "HOLD" and was_t:
            bare = norm_ticker(was_t)
            row["now_exit"] = ""
            row["was_pnl"] = hold_pnl.get(bare, "—")
            row["was_entry"] = was_entry.get(bare, "—")
            row["was_close"] = was_close.get(bare, "—")
        else:
            row["now_exit"] = ""

        if rebal_t:
            bare = norm_ticker(rebal_t)
            row["pick_pnl"] = pick_pnl.get(bare, "—")
            row["pick_entry"] = pick_entry.get(bare, "—")
            row["pick_close"] = pick_close.get(bare, "—")
            row["mid_9ema"] = mid_9ema.get(bare, "")
    return rows


def apply_portfolio_panel_display(
    rows: list[dict[str, str]],
    *,
    was_rank_for_ticker: Callable[[str], int | None],
    curr_rank_for_ticker: Callable[[str], int | None],
    rebal_detail_for_ticker: Callable[[str], str] | None = None,
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
            detail = (
                (rebal_detail_for_ticker(rebal_t) or "").strip()
                if rebal_detail_for_ticker is not None
                else None
            )
            row["rebal_text"] = format_portfolio_cell(
                rebal_t,
                curr_rank_for_ticker(rebal_t),
                detail=detail or None,
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
    as_of_ts: pd.Timestamp | None = None,
    weekly_for_ticker: Callable[[str], pd.Series],
    daily_for_ticker: Callable[[str], pd.Series | None] | None,
    was_rank_for_ticker: Callable[[str], int | None],
    curr_rank_for_ticker: Callable[[str], int | None],
    exit_below_9ema: bool,
    mid_week_9ema: list | None = None,
    live_close_by_ticker: dict[str, float] | None = None,
    portfolio_slots: int | None = None,
    rebal_detail_for_ticker: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, str]], PortfolioPanelTotals]:
    """Full portfolio panel rows and equal-weight Was / pick totals."""
    from momentum.rrg_portfolio_exits import (
        compute_portfolio_panel_totals,
        exit_detail_and_pnl_maps,
        holding_pnl_map,
        mid_week_exit_ts_map,
        pick_close_map,
        pick_entry_close_map,
        pick_pnl_map,
        was_close_map,
        was_entry_close_map,
    )

    was_list = compact_tickers(prev_portfolio)
    rebal_entered = compact_tickers(rebal_tickers)
    rebal_display = build_rebal_display_rows(
        rebal_strategy,
        rebal_tickers,
        rebalance_ts=rebalance_ts,
        rank_for_ticker=curr_rank_for_ticker,
        detail_for_ticker=rebal_detail_for_ticker,
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
    mark_ts = (
        pd.Timestamp(as_of_ts) if as_of_ts is not None else pd.Timestamp(rebalance_ts)
    )
    rebal_set = {norm_ticker(t) for t in rebal_entered}
    hold_pnl_by = holding_pnl_map(
        [t for t in was_list if norm_ticker(t) in rebal_set],
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        as_of_ts=mark_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        live_close_by_ticker=live_close_by_ticker,
    )
    mid_9ema_by: dict[str, str] = {}
    mid_exit_ts: dict[str, pd.Timestamp] = {}
    if exit_below_9ema:
        mid_9ema_by = mid_week_9ema_cell_map(
            mid_week_9ema or [],
            rebalance_tickers=rebal_entered,
        )
        mid_exit_ts = mid_week_exit_ts_map(
            mid_week_9ema or [],
            rebalance_tickers=rebal_entered,
        )
    pick_pnl_by = pick_pnl_map(
        rows,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        as_of_ts=mark_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        mid_week_exit_ts=mid_exit_ts,
        live_close_by_ticker=live_close_by_ticker,
    )
    was_close_by = was_close_map(
        was_list,
        rebal_holdings=rebal_set,
        panel_exits=panel_exits,
        rebalance_ts=rebalance_ts,
        as_of_ts=mark_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        live_close_by_ticker=live_close_by_ticker,
    )
    was_entry_by = was_entry_close_map(
        was_list,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
    )
    pick_close_by = pick_close_map(
        rows,
        rebalance_ts=rebalance_ts,
        as_of_ts=mark_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        mid_week_exit_ts=mid_exit_ts,
        live_close_by_ticker=live_close_by_ticker,
    )
    pick_entry_by = pick_entry_close_map(
        rows,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
    )
    rows = enrich_portfolio_panel_rows(
        rows,
        pnl_by_ticker=pnl_by,
        hold_pnl_by_ticker=hold_pnl_by,
        pick_pnl_by_ticker=pick_pnl_by,
        was_close_by_ticker=was_close_by,
        was_entry_by_ticker=was_entry_by,
        pick_close_by_ticker=pick_close_by,
        pick_entry_by_ticker=pick_entry_by,
        mid_9ema_by_ticker=mid_9ema_by,
        exit_detail_by_ticker=exit_detail_by,
    )
    rows = apply_portfolio_panel_display(
        rows,
        was_rank_for_ticker=was_rank_for_ticker,
        curr_rank_for_ticker=curr_rank_for_ticker,
        rebal_detail_for_ticker=rebal_detail_for_ticker,
    )
    slots = portfolio_slots if portfolio_slots is not None else max(len(rebal_tickers), 1)
    totals = compute_portfolio_panel_totals(
        was_list=prev_portfolio,
        rebal_slot_tickers=rebal_tickers,
        prev_portfolio=prev_portfolio,
        panel_exits=panel_exits,
        rebalance_ts=rebalance_ts,
        prev_rebalance_ts=prev_rebalance_ts,
        as_of_ts=mark_ts,
        portfolio_slots=slots,
        weekly_for_ticker=weekly_for_ticker,
        daily_for_ticker=daily_for_ticker,
        mid_week_exit_ts=mid_exit_ts,
        live_close_by_ticker=live_close_by_ticker,
    )
    return rows, totals


def portfolio_panel_totals_line(
    totals: PortfolioPanelTotals | None,
    *,
    live_pick: bool = False,
    mode: str = "week",
) -> str:
    """Equal-weight P/L for Was holdings and current Top N picks."""
    from momentum.rrg_portfolio_exits import format_ew_total_pct

    if totals is None:
        return ""
    live_tag = " live" if live_pick else ""
    was_hdr = "Prior day P/L" if mode == "day" else "Last week P/L"
    pick_hdr = "Day P/L" if mode == "day" else "Current week P/L"
    if totals.was_slots:
        was_part = (
            f"{was_hdr} ({totals.was_slots}): "
            f"{format_ew_total_pct(totals.was_ew_pct)}"
        )
    else:
        was_part = f"{was_hdr}: —"
    if totals.pick_slots:
        pick_part = (
            f"{pick_hdr} ({totals.pick_slots}){live_tag}: "
            f"{format_ew_total_pct(totals.pick_ew_pct)}"
        )
    else:
        pick_part = f"{pick_hdr}{live_tag}: —"
    return f"{was_part}  ·  {pick_part}"


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
    mode: str = "week",
    rebalance_preview: bool = False,
) -> str:
    if exit_below_9ema:
        through = exits_through_label or rebalance_label
        exits_note = f" · Exits through {through}"
    else:
        exits_note = ""
    if mode == "day":
        pick_part = pick_shortfall or f"Top N {rebal_n} today"
        return (
            f"As of {rebalance_label}  ·  Was {was_n} from {was_label}  ·  "
            f"{pick_part} (★ order){exits_note}  ·  {subtitle}"
        )
    pick_part = pick_shortfall or f"Top N {rebal_n} this week"
    preview_note = " · weekly @ latest Fri" if rebalance_preview else ""
    return (
        f"Rebalance {rebalance_label}{preview_note}  ·  Was {was_n} from {was_label}  ·  "
        f"{pick_part} (★ order){exits_note}  ·  {subtitle}"
    )
