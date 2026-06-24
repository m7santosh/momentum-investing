"""Nifty sector / thematic indices for candlestick backtests."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.universes import india
from utils.nse_bhavcopy import list_nse_index_names, nse_index_data_ticker


@dataclass(frozen=True)
class NiftyIndex:
    index_id: str
    yahoo_ticker: str
    label: str


def build_nifty_index_universe() -> list[NiftyIndex]:
    """All NSE ``ind_close_all`` indices (local cache) plus ETF-mapped names."""
    seen: set[str] = set()
    names: list[str] = []
    for index_name in dict.fromkeys(v for v in india.ETF_TO_NSE_INDEX.values() if v):
        if index_name in seen:
            continue
        seen.add(index_name)
        names.append(index_name)
    for index_name in list_nse_index_names():
        if index_name in seen:
            continue
        seen.add(index_name)
        names.append(index_name)
    return [
        NiftyIndex(
            index_id=index_name,
            yahoo_ticker=nse_index_data_ticker(index_name),
            label=index_name,
        )
        for index_name in names
    ]


NIFTY_INDICES: list[NiftyIndex] = build_nifty_index_universe()
NIFTY_INDEX_BY_ID: dict[str, NiftyIndex] = {i.index_id: i for i in NIFTY_INDICES}
NIFTY_INDEX_BY_LABEL: dict[str, NiftyIndex] = {i.label: i for i in NIFTY_INDICES}
NIFTY_INDEX_YAHOO_TICKERS: list[str] = list(dict.fromkeys(i.yahoo_ticker for i in NIFTY_INDICES))

DEFAULT_SELECTED_INDEX_IDS: tuple[str, ...] = ("Nifty 50",)


def resolve_selected_indices(values: list[str] | tuple[str, ...] | None) -> list[NiftyIndex]:
    """Resolve index ids or display labels to ``NiftyIndex`` rows (deduped, stable order)."""
    raw_values = list(values) if values else list(DEFAULT_SELECTED_INDEX_IDS)
    out: list[NiftyIndex] = []
    seen: set[str] = set()
    for raw in raw_values:
        key = (raw or "").strip()
        if not key:
            continue
        idx = NIFTY_INDEX_BY_ID.get(key) or NIFTY_INDEX_BY_LABEL.get(key)
        if idx is None:
            by_label = {k.lower(): v for k, v in NIFTY_INDEX_BY_LABEL.items()}
            idx = by_label.get(key.lower())
        if idx is None:
            raise ValueError(f"Unknown index: {raw!r}")
        if idx.index_id in seen:
            continue
        seen.add(idx.index_id)
        out.append(idx)
    if not out:
        raise ValueError("Select at least one index.")
    return out
