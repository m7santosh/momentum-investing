"""US RRG portfolio pick rules (base strategies + optional rank-hold overlay)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from momentum.etf.india_rrg_pick_strategies import (
    BASE_PICK_STRATEGIES,
    PICK_STRATEGIES,
    pick_strategy_label,
    pick_strategy_subtitle,
)
from momentum.etf.us_rrg_recommendations import (
    UsEtfRecommendation,
    parse_rank_delta,
    recommend_us_etfs,
)
from momentum.rrg_core import get_status

__all__ = [
    "BASE_PICK_STRATEGIES",
    "PICK_STRATEGIES",
    "UsPickContext",
    "count_strategy_eligible",
    "pick_shortfall_hint",
    "pick_strategy_label",
    "pick_strategy_subtitle",
    "pick_us_portfolio",
]


@dataclass(frozen=True)
class UsPickContext:
    """Weekly ranking + RRG state for US pick strategies."""

    ranked_row_indices: list[int]
    indices: list[str]
    display_labels: list[str]
    vol_by_ticker: dict[str, float]
    end_ts: object
    rsr_series_by_row: list
    rsm_series_by_row: list
    rank_delta_by_row: dict[int, str]
    change_pct_fn: Callable[[int], float]
    series_at_fn: Callable
    curr_ranks: dict[int, int]
    prev_ranks: dict[int, int]
    top_n: int
    prev_holdings: list[str]
    hold_until_rank_exit: bool = False
    max_hold_rank: int = 10


def order_picks_by_table_rank(
    picks: list[UsEtfRecommendation],
    ranked_row_indices: list[int],
) -> list[UsEtfRecommendation]:
    """Match main RRG table / ★ order (not swing-score pick_rank)."""
    if not picks:
        return picks
    by_row = {p.row_idx: p for p in picks}
    ordered: list[UsEtfRecommendation] = []
    for j in ranked_row_indices:
        if j in by_row:
            ordered.append(by_row[j])
    for i, p in enumerate(ordered, start=1):
        ordered[i - 1] = replace(p, pick_rank=i)
    return ordered


def _ticker(ctx: UsPickContext, row_j: int) -> str:
    return ctx.indices[row_j].strip().upper()


def _quadrant_at(ctx: UsPickContext, row_j: int) -> str:
    try:
        rsr = float(ctx.series_at_fn(ctx.rsr_series_by_row[row_j], ctx.end_ts))
        rsm = float(ctx.series_at_fn(ctx.rsm_series_by_row[row_j], ctx.end_ts))
    except (KeyError, TypeError, ValueError, IndexError):
        return "—"
    return get_status(rsr, rsm).capitalize()


def _row_indices_to_picks(
    ctx: UsPickContext, row_indices: list[int]
) -> list[UsEtfRecommendation]:
    picks: list[UsEtfRecommendation] = []
    for pick_rank, j in enumerate(row_indices, start=1):
        ticker = _ticker(ctx, j)
        if not ticker:
            continue
        chg = ctx.change_pct_fn(j)
        vol = ctx.vol_by_ticker.get(ticker, 0.0)
        delta_text = ctx.rank_delta_by_row.get(j, "—")
        picks.append(
            UsEtfRecommendation(
                pick_rank=pick_rank,
                row_idx=j,
                ticker=ticker,
                name=ctx.display_labels[j],
                change_pct=chg if chg != float("-inf") else 0.0,
                rank_delta=delta_text,
                vol_pct=vol,
                quadrant=_quadrant_at(ctx, j),
                size_hint="",
                score=0.0,
                reason="",
            )
        )
    return picks


def _eligible_by_momentum(ctx: UsPickContext) -> list[int]:
    out: list[int] = []
    for j in ctx.ranked_row_indices:
        ticker = _ticker(ctx, j)
        if not ticker:
            continue
        if ctx.change_pct_fn(j) == float("-inf"):
            continue
        out.append(j)
    return out


def pick_leading_only(ctx: UsPickContext) -> list[UsEtfRecommendation]:
    eligible: list[int] = []
    for j in ctx.ranked_row_indices:
        ticker = _ticker(ctx, j)
        if not ticker:
            continue
        try:
            rsr = float(ctx.series_at_fn(ctx.rsr_series_by_row[j], ctx.end_ts))
            rsm = float(ctx.series_at_fn(ctx.rsm_series_by_row[j], ctx.end_ts))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if get_status(rsr, rsm) != "leading":
            continue
        if ctx.change_pct_fn(j) == float("-inf"):
            continue
        eligible.append(j)
        if len(eligible) >= ctx.top_n:
            break
    return _row_indices_to_picks(ctx, eligible)


def pick_leading_improved(ctx: UsPickContext) -> list[UsEtfRecommendation]:
    eligible: list[int] = []
    for j in ctx.ranked_row_indices:
        ticker = _ticker(ctx, j)
        if not ticker:
            continue
        delta_val = parse_rank_delta(ctx.rank_delta_by_row.get(j, "—"))
        if delta_val is None or delta_val <= 0:
            continue
        try:
            rsr = float(ctx.series_at_fn(ctx.rsr_series_by_row[j], ctx.end_ts))
            rsm = float(ctx.series_at_fn(ctx.rsm_series_by_row[j], ctx.end_ts))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if get_status(rsr, rsm) != "leading":
            continue
        if ctx.change_pct_fn(j) == float("-inf"):
            continue
        eligible.append(j)
        if len(eligible) >= ctx.top_n:
            break
    return _row_indices_to_picks(ctx, eligible)


def pick_top_n_plain(ctx: UsPickContext) -> list[UsEtfRecommendation]:
    eligible = _eligible_by_momentum(ctx)[: ctx.top_n]
    return _row_indices_to_picks(ctx, eligible)


def _pick_base(strategy: str, ctx: UsPickContext) -> list[UsEtfRecommendation]:
    key = (strategy or "recommend").strip().lower()
    if key == "recommend":
        return recommend_us_etfs(
            ranked_row_indices=ctx.ranked_row_indices,
            indices=ctx.indices,
            display_labels=ctx.display_labels,
            vol_by_ticker=ctx.vol_by_ticker,
            end_ts=ctx.end_ts,
            rsr_series_by_row=ctx.rsr_series_by_row,
            rsm_series_by_row=ctx.rsm_series_by_row,
            rank_delta_by_row=ctx.rank_delta_by_row,
            change_pct_fn=ctx.change_pct_fn,
            series_at_fn=ctx.series_at_fn,
            curr_ranks=ctx.curr_ranks,
            prev_ranks=ctx.prev_ranks,
            limit=ctx.top_n,
        )
    if key == "leading_only":
        return pick_leading_only(ctx)
    if key == "leading_improved":
        return pick_leading_improved(ctx)
    if key == "top_n":
        return pick_top_n_plain(ctx)
    if key == "top_n_rank_exit":
        return pick_top_n_plain(ctx)
    raise ValueError(
        f"Unknown base pick strategy {strategy!r}. "
        f"Choose from: {', '.join(BASE_PICK_STRATEGIES)}"
    )


def apply_rank_exit_overlay(
    ctx: UsPickContext,
    base_picks: list[UsEtfRecommendation],
) -> list[UsEtfRecommendation]:
    n = len(ctx.indices)
    ticker_to_j: dict[str, int] = {}
    for j in range(n):
        ticker = _ticker(ctx, j)
        if ticker and ticker not in ticker_to_j:
            ticker_to_j[ticker] = j

    pick_by_ticker = {p.ticker: p for p in base_picks}

    held: list[UsEtfRecommendation] = []
    for ticker in ctx.prev_holdings:
        bare = ticker.strip().upper()
        j = ticker_to_j.get(bare)
        if j is None:
            continue
        rank = ctx.curr_ranks.get(j, n + 1)
        if rank <= ctx.max_hold_rank:
            held.append(pick_by_ticker.get(bare) or _row_indices_to_picks(ctx, [j])[0])

    held_tickers = {p.ticker for p in held}
    merged: list[UsEtfRecommendation] = list(held)
    for p in base_picks:
        if len(merged) >= ctx.top_n:
            break
        if p.ticker in held_tickers:
            continue
        merged.append(p)
        held_tickers.add(p.ticker)

    for i, p in enumerate(merged[: ctx.top_n], start=1):
        merged[i - 1] = UsEtfRecommendation(
            pick_rank=i,
            row_idx=p.row_idx,
            ticker=p.ticker,
            name=p.name,
            change_pct=p.change_pct,
            rank_delta=p.rank_delta,
            vol_pct=p.vol_pct,
            quadrant=p.quadrant,
            size_hint=p.size_hint,
            score=p.score,
            reason=p.reason,
        )
    return merged[: ctx.top_n]


def count_strategy_eligible(strategy: str, ctx: UsPickContext) -> int:
    """Universe rows passing this strategy's entry filters (before overlap dedupe)."""
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
    n = 0
    for j in ctx.ranked_row_indices:
        if not _ticker(ctx, j):
            continue
        if ctx.change_pct_fn(j) == float("-inf"):
            continue
        if key == "top_n":
            n += 1
            continue
        try:
            rsr = float(ctx.series_at_fn(ctx.rsr_series_by_row[j], ctx.end_ts))
            rsm = float(ctx.series_at_fn(ctx.rsm_series_by_row[j], ctx.end_ts))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        status = get_status(rsr, rsm)
        if key == "leading_only":
            if status == "leading":
                n += 1
        elif key == "leading_improved":
            delta_val = parse_rank_delta(ctx.rank_delta_by_row.get(j, "—"))
            if status == "leading" and delta_val is not None and delta_val > 0:
                n += 1
        else:
            delta_val = parse_rank_delta(ctx.rank_delta_by_row.get(j, "—"))
            if (
                delta_val is not None
                and delta_val > 0
                and status in ("leading", "improving")
            ):
                n += 1
    return n


def pick_shortfall_hint(strategy: str, ctx: UsPickContext, picked_n: int) -> str:
    """Why pick count can be below Portfolio N despite many ETFs in the table."""
    if picked_n >= ctx.top_n:
        return ""
    eligible = count_strategy_eligible(strategy, ctx)
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
    if key == "recommend" and eligible > picked_n:
        return (
            f"{picked_n}/{ctx.top_n} picks — {eligible} passed Leading/Improving "
            f"+ RankΔ>0; overlap buckets skipped the rest"
        )
    if key == "leading_only":
        return (
            f"{picked_n}/{ctx.top_n} picks — only {eligible} ETFs in Leading "
            f"quadrant this week"
        )
    if key == "leading_improved":
        return (
            f"{picked_n}/{ctx.top_n} picks — only {eligible} in Leading with "
            f"Rank Δ>0 this week"
        )
    if key == "top_n":
        return f"{picked_n}/{ctx.top_n} picks — {eligible} with valid momentum data"
    return f"{picked_n}/{ctx.top_n} picks — {eligible} passed strategy filters"


def pick_us_portfolio(strategy: str, ctx: UsPickContext) -> list[UsEtfRecommendation]:
    key = (strategy or "recommend").strip().lower()
    hold = ctx.hold_until_rank_exit or key == "top_n_rank_exit"
    base_key = "top_n" if key == "top_n_rank_exit" else key
    base = _pick_base(base_key, ctx)
    if not hold:
        return base
    return apply_rank_exit_overlay(ctx, base)
