"""US RRG backtest portfolio pick rules (alternatives to swing recommend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from momentum.etf.india_rrg_pick_strategies import PICK_STRATEGIES, pick_strategy_label
from momentum.etf.us_rrg_recommendations import (
    UsEtfRecommendation,
    parse_rank_delta,
    recommend_us_etfs,
)
from momentum.rrg_core import get_status

__all__ = [
    "PICK_STRATEGIES",
    "UsPickContext",
    "pick_strategy_label",
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
    max_hold_rank: int = 20


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


def pick_leading_improved(ctx: UsPickContext) -> list[UsEtfRecommendation]:
    """Top N in Leading quadrant with positive rank delta, best momentum first."""
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
    """Top N by tail-window momentum rank — no quadrant or rank-delta filter."""
    eligible = _eligible_by_momentum(ctx)[: ctx.top_n]
    return _row_indices_to_picks(ctx, eligible)


def pick_top_n_rank_exit(ctx: UsPickContext) -> list[UsEtfRecommendation]:
    """Keep holdings while rank <= threshold; refill to N from momentum rank."""
    n = len(ctx.indices)
    ticker_to_j: dict[str, int] = {}
    for j in range(n):
        ticker = _ticker(ctx, j)
        if ticker and ticker not in ticker_to_j:
            ticker_to_j[ticker] = j

    held: list[str] = []
    for ticker in ctx.prev_holdings:
        bare = ticker.strip().upper()
        j = ticker_to_j.get(bare)
        if j is None:
            continue
        rank = ctx.curr_ranks.get(j, n + 1)
        if rank <= ctx.max_hold_rank:
            held.append(bare)

    held_set = set(held)
    for j in _eligible_by_momentum(ctx):
        if len(held) >= ctx.top_n:
            break
        ticker = _ticker(ctx, j)
        if ticker in held_set:
            continue
        held.append(ticker)
        held_set.add(ticker)

    row_indices = [ticker_to_j[t] for t in held if t in ticker_to_j]
    return _row_indices_to_picks(ctx, row_indices[: ctx.top_n])


def pick_us_portfolio(strategy: str, ctx: UsPickContext) -> list[UsEtfRecommendation]:
    """Dispatch to the configured US backtest pick strategy."""
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
    if key == "leading_improved":
        return pick_leading_improved(ctx)
    if key == "top_n":
        return pick_top_n_plain(ctx)
    if key == "top_n_rank_exit":
        return pick_top_n_rank_exit(ctx)
    raise ValueError(
        f"Unknown pick strategy {strategy!r}. "
        f"Choose from: {', '.join(PICK_STRATEGIES)}"
    )
