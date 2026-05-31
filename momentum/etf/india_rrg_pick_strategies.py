"""India RRG backtest portfolio pick rules (alternatives to swing recommend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from momentum.etf.india_rrg_recommendations import (
    EXCLUDE_REF_ETFS,
    IndiaEtfRecommendation,
    recommend_india_etfs,
)
from momentum.etf.us_rrg_recommendations import parse_rank_delta
from momentum.rrg_core import get_status

PICK_STRATEGIES: dict[str, str] = {
    "recommend": "Recommended Top N (RRG swing score)",
    "leading_improved": "Top N — Leading quadrant, rank improved",
    "top_n": "Top N — momentum rank only (no filter)",
    "top_n_rank_exit": "Top N — hold until rank worse than threshold",
}


@dataclass(frozen=True)
class IndiaPickContext:
    """Weekly ranking + RRG state for pick strategies."""

    ranked_row_indices: list[int]
    indices: list[str]
    ref_labels: list[str]
    display_labels: list[str]
    vol_by_ref: dict[str, float]
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


def _bare_ref(ctx: IndiaPickContext, row_j: int) -> str:
    ref = (ctx.ref_labels[row_j] or ctx.indices[row_j]).strip().upper().replace(
        ".NS", ""
    )
    return ref


def _quadrant_at(ctx: IndiaPickContext, row_j: int) -> str:
    try:
        rsr = float(ctx.series_at_fn(ctx.rsr_series_by_row[row_j], ctx.end_ts))
        rsm = float(ctx.series_at_fn(ctx.rsm_series_by_row[row_j], ctx.end_ts))
    except (KeyError, TypeError, ValueError, IndexError):
        return "—"
    return get_status(rsr, rsm).capitalize()


def _row_indices_to_picks(
    ctx: IndiaPickContext, row_indices: list[int]
) -> list[IndiaEtfRecommendation]:
    picks: list[IndiaEtfRecommendation] = []
    for pick_rank, j in enumerate(row_indices, start=1):
        ref = _bare_ref(ctx, j)
        if not ref:
            continue
        chg = ctx.change_pct_fn(j)
        vol = ctx.vol_by_ref.get(ref, 0.0)
        delta_text = ctx.rank_delta_by_row.get(j, "—")
        picks.append(
            IndiaEtfRecommendation(
                pick_rank=pick_rank,
                row_idx=j,
                ticker=ref,
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


def _eligible_by_momentum(ctx: IndiaPickContext) -> list[int]:
    out: list[int] = []
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref or ref in EXCLUDE_REF_ETFS:
            continue
        if ctx.change_pct_fn(j) == float("-inf"):
            continue
        out.append(j)
    return out


def pick_leading_improved(ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
    """Top N in Leading quadrant with positive rank delta, best momentum first."""
    eligible: list[int] = []
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref or ref in EXCLUDE_REF_ETFS:
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


def pick_top_n_plain(ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
    """Top N by tail-window momentum rank — no quadrant or rank-delta filter."""
    eligible = _eligible_by_momentum(ctx)[: ctx.top_n]
    return _row_indices_to_picks(ctx, eligible)


def pick_top_n_rank_exit(ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
    """Keep holdings while rank <= threshold; refill to N from momentum rank."""
    n = len(ctx.indices)
    ref_to_j: dict[str, int] = {}
    for j in range(n):
        ref = _bare_ref(ctx, j)
        if ref and ref not in ref_to_j:
            ref_to_j[ref] = j

    held_refs: list[str] = []
    for ref in ctx.prev_holdings:
        bare = ref.strip().upper().replace(".NS", "")
        j = ref_to_j.get(bare)
        if j is None:
            continue
        rank = ctx.curr_ranks.get(j, n + 1)
        if rank <= ctx.max_hold_rank:
            held_refs.append(bare)

    held_set = set(held_refs)
    for j in _eligible_by_momentum(ctx):
        if len(held_refs) >= ctx.top_n:
            break
        ref = _bare_ref(ctx, j)
        if ref in held_set:
            continue
        held_refs.append(ref)
        held_set.add(ref)

    row_indices = [ref_to_j[r] for r in held_refs if r in ref_to_j]
    return _row_indices_to_picks(ctx, row_indices[: ctx.top_n])


def pick_india_portfolio(strategy: str, ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
    """Dispatch to the configured India backtest pick strategy."""
    key = (strategy or "recommend").strip().lower()
    if key == "recommend":
        return recommend_india_etfs(
            ranked_row_indices=ctx.ranked_row_indices,
            indices=ctx.indices,
            ref_labels=ctx.ref_labels,
            display_labels=ctx.display_labels,
            vol_by_ref=ctx.vol_by_ref,
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


def pick_strategy_label(strategy: str) -> str:
    return PICK_STRATEGIES.get((strategy or "recommend").strip().lower(), strategy)


def pick_strategy_subtitle(strategy: str, *, max_hold_rank: int = 20) -> str:
    key = (strategy or "recommend").strip().lower()
    if key == "leading_improved":
        return "Leading quadrant · rank improved · best momentum first"
    if key == "top_n":
        return "Top N by tail momentum rank (no quadrant filter)"
    if key == "top_n_rank_exit":
        return (
            f"Top N · keep while rank ≤ {max_hold_rank} · refill on rebalance week"
        )
    return "Leading/Improving · Rank Δ>0 · momentum+reliability score"
