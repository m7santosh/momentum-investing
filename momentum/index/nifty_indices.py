"""Nifty sector / thematic indices and benchmark presets for candlestick backtests."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.etf_rrg_universe import RRG_ROWS
from utils.nse_bhavcopy import YAHOO_TO_NSE_INDEX

NSE_INDEX_TO_YAHOO: dict[str, str] = {v: k for k, v in YAHOO_TO_NSE_INDEX.items()}


@dataclass(frozen=True)
class NiftyBenchmark:
    key: str
    label: str
    yahoo_ticker: str


NIFTY_BENCHMARKS: dict[str, NiftyBenchmark] = {
    "nifty_50": NiftyBenchmark("nifty_50", "Nifty 50", "^NSEI"),
    "nifty_250": NiftyBenchmark("nifty_250", "Nifty 250", "NIFTY_LARGEMID250.NS"),
    "midcap_150": NiftyBenchmark("midcap_150", "Midcap 150", "NIFTYMIDCAP150.NS"),
    "smallcap_250": NiftyBenchmark("smallcap_250", "Smallcap 250", "NIFTYSMLCAP250.NS"),
    "nifty_500": NiftyBenchmark("nifty_500", "Nifty 500", "^CRSLDX"),
}

DEFAULT_BENCHMARK_KEY = "nifty_50"
BENCHMARK_KEYS: tuple[str, ...] = tuple(NIFTY_BENCHMARKS.keys())

_DEFAULT_BENCHMARK = NIFTY_BENCHMARKS[DEFAULT_BENCHMARK_KEY]
BENCHMARK_YAHOO = _DEFAULT_BENCHMARK.yahoo_ticker
BENCHMARK_LABEL = _DEFAULT_BENCHMARK.label


def resolve_benchmark(value: str) -> NiftyBenchmark:
    """Resolve preset key, display label, or Yahoo ticker to a benchmark."""
    raw = (value or "").strip()
    if not raw:
        return _DEFAULT_BENCHMARK
    if raw in NIFTY_BENCHMARKS:
        return NIFTY_BENCHMARKS[raw]
    by_label = {b.label.lower(): b for b in NIFTY_BENCHMARKS.values()}
    if raw.lower() in by_label:
        return by_label[raw.lower()]
    by_yahoo = {b.yahoo_ticker.upper(): b for b in NIFTY_BENCHMARKS.values()}
    if raw.upper() in by_yahoo:
        return by_yahoo[raw.upper()]
    raise ValueError(
        f"Unknown benchmark: {value!r}. "
        f"Choose: {', '.join(b.label for b in NIFTY_BENCHMARKS.values())}"
    )


@dataclass(frozen=True)
class NiftyIndex:
    index_id: str
    yahoo_ticker: str
    label: str


def build_nifty_index_universe() -> list[NiftyIndex]:
    """RRG index rows with Yahoo index symbol or tracking ETF fallback."""
    out: list[NiftyIndex] = []
    seen: set[str] = set()
    for row in RRG_ROWS:
        if row.kind != "index" or row.row_id in seen:
            continue
        seen.add(row.row_id)
        yahoo = NSE_INDEX_TO_YAHOO.get(row.row_id)
        if not yahoo and row.etf_ticker:
            yahoo = row.etf_ticker
        if not yahoo:
            continue
        out.append(NiftyIndex(index_id=row.row_id, yahoo_ticker=yahoo, label=row.label))
    return out


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
