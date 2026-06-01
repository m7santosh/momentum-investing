"""India RRG portfolio pick rules (base strategies + optional rank-hold overlay)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from momentum.etf.india_rrg_recommendations import (
    EXCLUDE_REF_ETFS,
    IndiaEtfRecommendation,
    recommend_india_etfs,
)
from momentum.etf.us_rrg_recommendations import parse_rank_delta
from momentum.rrg_core import get_status

# Base strategies only — rank-hold is a separate overlay (checkbox), not a 5th mode.
BASE_PICK_STRATEGIES: dict[str, str] = {
    "recommend": "Recommended Top N (RRG swing score)",
    "leading_only": "Top N — Leading quadrant only",
    "leading_improved": "Top N — Leading quadrant + momentum rank ↑",
    "top_n": "Top N — momentum rank only (no filter)",
}

PICK_STRATEGIES = BASE_PICK_STRATEGIES


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
    hold_until_rank_exit: bool = False
    max_hold_rank: int = 10


def ref_to_row_index(
    indices: list[str],
    ref_labels: list[str],
) -> dict[str, int]:
    ref_to_j: dict[str, int] = {}
    for j in range(len(indices)):
        ref = (ref_labels[j] if j < len(ref_labels) else indices[j]).strip().upper()
        ref = ref.replace(".NS", "")
        if ref and ref not in ref_to_j:
            ref_to_j[ref] = j
    return ref_to_j


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


def order_picks_by_table_rank(
    picks: list[IndiaEtfRecommendation],
    ranked_row_indices: list[int],
) -> list[IndiaEtfRecommendation]:
    """
    Same order as main RRG Top N @ rebalance (★ / tail momentum table order),
    not swing-score pick_rank order from recommend_india_etfs.
    """
    if not picks:
        return picks
    by_row = {p.row_idx: p for p in picks}
    ordered: list[IndiaEtfRecommendation] = []
    for j in ranked_row_indices:
        if j in by_row:
            ordered.append(by_row[j])
    for i, p in enumerate(ordered, start=1):
        ordered[i - 1] = replace(p, pick_rank=i)
    return ordered


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


def pick_leading_only(ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
    """Top N in Leading quadrant by momentum rank (no rank-Δ filter)."""
    eligible: list[int] = []
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref or ref in EXCLUDE_REF_ETFS:
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


def _pick_base(strategy: str, ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
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
    ctx: IndiaPickContext,
    base_picks: list[IndiaEtfRecommendation],
) -> list[IndiaEtfRecommendation]:
    """Keep prior holdings while rank ≤ threshold; refill gaps from ``base_picks``."""
    n = len(ctx.indices)
    ref_to_j: dict[str, int] = {}
    for j in range(n):
        ref = _bare_ref(ctx, j)
        if ref and ref not in ref_to_j:
            ref_to_j[ref] = j

    pick_by_ticker = {p.ticker: p for p in base_picks}

    held: list[IndiaEtfRecommendation] = []
    for ref in ctx.prev_holdings:
        bare = ref.strip().upper().replace(".NS", "")
        j = ref_to_j.get(bare)
        if j is None:
            continue
        rank = ctx.curr_ranks.get(j, n + 1)
        if rank <= ctx.max_hold_rank:
            held.append(pick_by_ticker.get(bare) or _row_indices_to_picks(ctx, [j])[0])

    held_tickers = {p.ticker for p in held}
    merged: list[IndiaEtfRecommendation] = list(held)
    for p in base_picks:
        if len(merged) >= ctx.top_n:
            break
        if p.ticker in held_tickers:
            continue
        merged.append(p)
        held_tickers.add(p.ticker)

    for i, p in enumerate(merged[: ctx.top_n], start=1):
        merged[i - 1] = IndiaEtfRecommendation(
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


def count_strategy_eligible(strategy: str, ctx: IndiaPickContext) -> int:
    """Universe rows passing this strategy's entry filters (before overlap dedupe)."""
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
    n = 0
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref or ref in EXCLUDE_REF_ETFS:
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


def pick_shortfall_hint(
    strategy: str, ctx: IndiaPickContext, picked_n: int
) -> str:
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


def pick_india_portfolio(strategy: str, ctx: IndiaPickContext) -> list[IndiaEtfRecommendation]:
    """Base strategy picks, optionally with hold-until-rank-worse overlay."""
    key = (strategy or "recommend").strip().lower()
    hold = ctx.hold_until_rank_exit or key == "top_n_rank_exit"
    base_key = "top_n" if key == "top_n_rank_exit" else key
    base = _pick_base(base_key, ctx)
    if not hold:
        return base
    return apply_rank_exit_overlay(ctx, base)


def pick_strategy_label(
    strategy: str,
    *,
    hold_until_rank_exit: bool = False,
    exit_below_9ema: bool = False,
) -> str:
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
        hold_until_rank_exit = True
    base = BASE_PICK_STRATEGIES.get(key, strategy)
    suffixes: list[str] = []
    if hold_until_rank_exit:
        suffixes.append("hold until rank worse")
    if exit_below_9ema:
        suffixes.append("exit below 9 EMA")
    if suffixes:
        return f"{base} + {' + '.join(suffixes)}"
    return base


def pick_strategy_subtitle(
    strategy: str,
    *,
    hold_until_rank_exit: bool = False,
    max_hold_rank: int = 10,
    exit_below_9ema: bool = False,
) -> str:
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
        hold_until_rank_exit = True
    if key == "leading_only":
        base = "Leading RRG quadrant only · best tail momentum first"
    elif key == "leading_improved":
        base = "Leading RRG quadrant · positive Rank Δ · best momentum first"
    elif key == "top_n":
        base = "Top N by tail momentum rank (no quadrant filter)"
    else:
        base = "Leading/Improving · Rank Δ>0 · momentum+reliability score"
    parts = [base]
    if hold_until_rank_exit:
        parts.append(f"keep holdings while rank ≤ {max_hold_rank}")
    if exit_below_9ema:
        parts.append("exit when close < 9 EMA · no refill until rebalance")
    return " · ".join(parts)
