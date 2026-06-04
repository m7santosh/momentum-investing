"""Portfolio fill modes for RRG backtest (replace vs maintain Top N vs accumulate)."""

from __future__ import annotations

from typing import Callable, TypeVar

PORTFOLIO_FILL_REPLACE = "replace"
PORTFOLIO_FILL_MAINTAIN_TOP_N = "maintain_top_n"
PORTFOLIO_FILL_ACCUMULATE = "accumulate"

PORTFOLIO_FILL_MODES: dict[str, str] = {
    PORTFOLIO_FILL_REPLACE: "Replace Top N each week",
    PORTFOLIO_FILL_MAINTAIN_TOP_N: "Maintain Top N (keep + fill to N)",
    PORTFOLIO_FILL_ACCUMULATE: "Add qualifying picks (keep all + add new)",
}

T = TypeVar("T")


def bare_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".NS", "")


def uses_prior_holdings(mode: str) -> bool:
    return (mode or PORTFOLIO_FILL_REPLACE).strip().lower() in (
        PORTFOLIO_FILL_MAINTAIN_TOP_N,
        PORTFOLIO_FILL_ACCUMULATE,
    )


def merge_maintain_top_n(
    prev_holdings: list[str],
    base_picks: list[T],
    *,
    top_n: int,
    pick_by_ticker: dict[str, T],
    reconstruct: Callable[[str], T | None],
    renumber: Callable[[list[T]], list[T]],
) -> list[T]:
    """Keep prior week holdings; fill gaps from strategy picks up to ``top_n``."""
    held: list[T] = []
    held_keys: set[str] = set()
    for ref in prev_holdings:
        if not ref:
            continue
        bare = bare_symbol(ref)
        if bare in held_keys:
            continue
        pick = pick_by_ticker.get(bare) or reconstruct(bare)
        if pick is None:
            continue
        held.append(pick)
        held_keys.add(bare)

    merged: list[T] = list(held)
    for pick in base_picks:
        if len(merged) >= top_n:
            break
        key = bare_symbol(getattr(pick, "ticker", "") or "")
        if not key or key in held_keys:
            continue
        merged.append(pick)
        held_keys.add(key)
    return renumber(merged[:top_n])


def merge_accumulate(
    prev_holdings: list[str],
    base_picks: list[T],
    *,
    pick_by_ticker: dict[str, T],
    reconstruct: Callable[[str], T | None],
    renumber: Callable[[list[T]], list[T]],
) -> list[T]:
    """Keep all prior holdings; append strategy picks not already held."""
    held: list[T] = []
    held_keys: set[str] = set()
    for ref in prev_holdings:
        if not ref:
            continue
        bare = bare_symbol(ref)
        if bare in held_keys:
            continue
        pick = pick_by_ticker.get(bare) or reconstruct(bare)
        if pick is None:
            continue
        held.append(pick)
        held_keys.add(bare)

    merged: list[T] = list(held)
    for pick in base_picks:
        key = bare_symbol(getattr(pick, "ticker", "") or "")
        if not key or key in held_keys:
            continue
        merged.append(pick)
        held_keys.add(key)
    return renumber(merged)


def equal_weight_port_return(week_rets: list[float], n_holdings: int) -> float:
    """Mean return over active holdings (ignore Top-N zero padding slots)."""
    if not week_rets or n_holdings <= 0:
        return 0.0
    return float(sum(week_rets[:n_holdings]) / n_holdings)
