"""Stock RRG portfolio pick rules (base strategies + optional rank-hold overlay)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from momentum.etf.us_rrg_recommendations import parse_rank_delta
from momentum.rrg_core import get_status
from momentum.rrg_portfolio_fill import (
    PORTFOLIO_FILL_ACCUMULATE,
    PORTFOLIO_FILL_MAINTAIN_TOP_N,
    PORTFOLIO_FILL_REPLACE,
    bare_symbol,
    merge_accumulate,
    merge_maintain_top_n,
)
from momentum.stock.stock_rrg_recommendations import (
    StockRecommendation,
    recommend_stocks,
)

BASE_PICK_STRATEGIES: dict[str, str] = {
    "recommend": "Recommended Top N (RRG swing score)",
    "leading_only": "Top N — Leading quadrant only",
    "leading_improved": "Top N — Leading quadrant + momentum rank ↑",
    "top_n": "Top N — momentum rank only (no filter)",
}

PICK_STRATEGIES = BASE_PICK_STRATEGIES


@dataclass(frozen=True)
class StockPickContext:
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
    portfolio_fill_mode: str = PORTFOLIO_FILL_REPLACE
    benchmark: str = "Nifty 500"


def ref_to_row_index(
    indices: list[str],
    ref_labels: list[str] | None = None,
) -> dict[str, int]:
    del ref_labels
    ref_to_j: dict[str, int] = {}
    for j, sym in enumerate(indices):
        bare = sym.strip().upper().replace(".NS", "")
        if bare and bare not in ref_to_j:
            ref_to_j[bare] = j
    return ref_to_j


def _bare_ref(ctx: StockPickContext, row_j: int) -> str:
    return ctx.indices[row_j].strip().upper().replace(".NS", "")


def _quadrant_at(ctx: StockPickContext, row_j: int) -> str:
    try:
        rsr = float(ctx.series_at_fn(ctx.rsr_series_by_row[row_j], ctx.end_ts))
        rsm = float(ctx.series_at_fn(ctx.rsm_series_by_row[row_j], ctx.end_ts))
    except (KeyError, TypeError, ValueError, IndexError):
        return "—"
    return get_status(rsr, rsm).capitalize()


def order_picks_by_table_rank(
    picks: list[StockRecommendation],
    ranked_row_indices: list[int],
) -> list[StockRecommendation]:
    if not picks:
        return picks
    by_row = {p.row_idx: p for p in picks}
    ordered: list[StockRecommendation] = []
    for j in ranked_row_indices:
        if j in by_row:
            ordered.append(by_row[j])
    for i, p in enumerate(ordered, start=1):
        ordered[i - 1] = replace(p, pick_rank=i)
    return ordered


def _row_indices_to_picks(
    ctx: StockPickContext, row_indices: list[int]
) -> list[StockRecommendation]:
    picks: list[StockRecommendation] = []
    for pick_rank, j in enumerate(row_indices, start=1):
        ref = _bare_ref(ctx, j)
        if not ref:
            continue
        chg = ctx.change_pct_fn(j)
        vol = ctx.vol_by_ref.get(ref, 0.0)
        delta_text = ctx.rank_delta_by_row.get(j, "—")
        picks.append(
            StockRecommendation(
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


def _eligible_by_momentum(ctx: StockPickContext) -> list[int]:
    out: list[int] = []
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref:
            continue
        if ctx.change_pct_fn(j) == float("-inf"):
            continue
        out.append(j)
    return out


def pick_leading_only(ctx: StockPickContext) -> list[StockRecommendation]:
    eligible: list[int] = []
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref:
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


def pick_leading_improved(ctx: StockPickContext) -> list[StockRecommendation]:
    eligible: list[int] = []
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref:
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


def pick_top_n_plain(ctx: StockPickContext) -> list[StockRecommendation]:
    eligible = _eligible_by_momentum(ctx)[: ctx.top_n]
    return _row_indices_to_picks(ctx, eligible)


def _pick_base(strategy: str, ctx: StockPickContext) -> list[StockRecommendation]:
    key = (strategy or "recommend").strip().lower()
    if key == "recommend":
        return recommend_stocks(
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
            benchmark=ctx.benchmark,
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


def _renumber_picks(
    picks: list[StockRecommendation],
) -> list[StockRecommendation]:
    out: list[StockRecommendation] = []
    for i, p in enumerate(picks, start=1):
        out.append(replace(p, pick_rank=i))
    return out


def _reconstruct_pick(ctx: StockPickContext, bare: str) -> StockRecommendation | None:
    ref_to_j = ref_to_row_index(ctx.indices)
    j = ref_to_j.get(bare)
    if j is None:
        return None
    rebuilt = _row_indices_to_picks(ctx, [j])
    return rebuilt[0] if rebuilt else None


def apply_portfolio_fill_mode(
    ctx: StockPickContext,
    base_picks: list[StockRecommendation],
) -> list[StockRecommendation]:
    mode = (ctx.portfolio_fill_mode or PORTFOLIO_FILL_REPLACE).strip().lower()
    if mode == PORTFOLIO_FILL_REPLACE or not ctx.prev_holdings:
        return base_picks
    pick_by_ticker = {bare_symbol(p.ticker): p for p in base_picks}
    if mode == PORTFOLIO_FILL_MAINTAIN_TOP_N:
        return merge_maintain_top_n(
            ctx.prev_holdings,
            base_picks,
            top_n=ctx.top_n,
            pick_by_ticker=pick_by_ticker,
            reconstruct=lambda bare: _reconstruct_pick(ctx, bare),
            renumber=_renumber_picks,
        )
    if mode == PORTFOLIO_FILL_ACCUMULATE:
        return merge_accumulate(
            ctx.prev_holdings,
            base_picks,
            pick_by_ticker=pick_by_ticker,
            reconstruct=lambda bare: _reconstruct_pick(ctx, bare),
            renumber=_renumber_picks,
        )
    return base_picks


def apply_rank_exit_overlay(
    ctx: StockPickContext,
    base_picks: list[StockRecommendation],
) -> list[StockRecommendation]:
    n = len(ctx.indices)
    ref_to_j = ref_to_row_index(ctx.indices)

    pick_by_ticker = {p.ticker: p for p in base_picks}

    held: list[StockRecommendation] = []
    for ref in ctx.prev_holdings:
        bare = ref.strip().upper().replace(".NS", "")
        j = ref_to_j.get(bare)
        if j is None:
            continue
        rank = ctx.curr_ranks.get(j, n + 1)
        if rank <= ctx.max_hold_rank:
            held.append(pick_by_ticker.get(bare) or _row_indices_to_picks(ctx, [j])[0])

    held_tickers = {p.ticker for p in held}
    merged: list[StockRecommendation] = list(held)
    for p in base_picks:
        if len(merged) >= ctx.top_n:
            break
        if p.ticker in held_tickers:
            continue
        merged.append(p)
        held_tickers.add(p.ticker)

    for i, p in enumerate(merged[: ctx.top_n], start=1):
        merged[i - 1] = StockRecommendation(
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


def count_strategy_eligible(strategy: str, ctx: StockPickContext) -> int:
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
    n = 0
    for j in ctx.ranked_row_indices:
        ref = _bare_ref(ctx, j)
        if not ref:
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
    strategy: str, ctx: StockPickContext, picked_n: int
) -> str:
    if picked_n >= ctx.top_n:
        return ""
    eligible = count_strategy_eligible(strategy, ctx)
    key = (strategy or "recommend").strip().lower()
    if key == "top_n_rank_exit":
        key = "top_n"
    if key == "recommend" and eligible > picked_n:
        return (
            f"{picked_n}/{ctx.top_n} picks — {eligible} passed Leading/Improving "
            f"+ RankΔ>0; industry dedupe skipped the rest"
        )
    if key == "leading_only":
        return (
            f"{picked_n}/{ctx.top_n} picks — only {eligible} stocks in Leading "
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


def pick_stock_portfolio(
    strategy: str, ctx: StockPickContext
) -> list[StockRecommendation]:
    key = (strategy or "recommend").strip().lower()
    hold = ctx.hold_until_rank_exit or key == "top_n_rank_exit"
    base_key = "top_n" if key == "top_n_rank_exit" else key
    base = _pick_base(base_key, ctx)
    mode = (ctx.portfolio_fill_mode or PORTFOLIO_FILL_REPLACE).strip().lower()
    picks = apply_portfolio_fill_mode(ctx, base)
    if hold and mode == PORTFOLIO_FILL_REPLACE:
        return apply_rank_exit_overlay(ctx, base)
    return picks


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
