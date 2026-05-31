"""Shared RRG table ranking helpers (tail-window change % and rank delta)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def series_at(series: pd.Series, ts) -> float:
    """Close at ``ts``, or last available bar on or before ``ts``."""
    try:
        return float(series.loc[ts])
    except KeyError:
        pos = series.index.get_indexer([pd.Timestamp(ts)], method="ffill")
        if pos[0] < 0:
            raise
        return float(series.iloc[pos[0]])


def tail_change_pct(price_series: pd.Series, start_ts, end_ts) -> float:
    """Percent price change over ``start_ts`` .. ``end_ts`` (inclusive)."""
    try:
        p_start = series_at(price_series, start_ts)
        p_end = series_at(price_series, end_ts)
        if p_start == 0:
            return float("-inf")
        return (p_end - p_start) / p_start * 100
    except (KeyError, TypeError, ValueError, IndexError):
        return float("-inf")


def rank_by_tail_change(
    n_rows: int,
    change_pct_fn: Callable[[int], float],
) -> dict[int, int]:
    """Row index -> rank (1 = best tail-window change)."""
    ranked = sorted(
        range(n_rows),
        key=lambda j: change_pct_fn(j),
        reverse=True,
    )
    return {j: display_rank + 1 for display_rank, j in enumerate(ranked)}


def ranked_row_indices(
    n_rows: int,
    change_pct_fn: Callable[[int], float],
) -> list[int]:
    """Row indices sorted best-to-worst by tail-window change."""
    return sorted(
        range(n_rows),
        key=lambda j: change_pct_fn(j),
        reverse=True,
    )


def format_rank_delta(curr_rank: int, prev_rank: int | None) -> str:
    """Change vs prior week (+ = moved up in rank)."""
    if prev_rank is None:
        return "—"
    delta = prev_rank - curr_rank
    if delta == 0:
        return "0"
    if delta > 0:
        return f"+{delta}"
    return str(delta)


def build_rank_delta_by_row(
    ranked: list[int],
    curr_ranks: dict[int, int],
    prev_ranks: dict[int, int] | None,
) -> dict[int, str]:
    prev_ranks = prev_ranks or {}
    out: dict[int, str] = {}
    for j in ranked:
        out[j] = format_rank_delta(curr_ranks.get(j, len(ranked)), prev_ranks.get(j))
    return out


def format_change_pct(chg: float) -> str:
    if chg == float("-inf"):
        return ""
    return f"{round(chg, 2):.2f}"
