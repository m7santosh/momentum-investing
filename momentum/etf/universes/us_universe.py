"""Canonical US ETF universe — single import for all US ETF Python scripts.

Reads ``us.py`` (tickers + ETF_LABELS). Call :func:`ensure_loaded` at process
start so RRG row caches stay in sync after ``us.py`` edits.
"""

from __future__ import annotations

from momentum.etf.universes import us as _us

KEY: str = _us.KEY
LABEL: str = _us.LABEL
DESCRIPTION: str = _us.DESCRIPTION
BENCHMARK_YAHOO: str = _us.BENCHMARK_YAHOO

TICKERS: list[str] = list(_us.tickers)
ETF_LABELS: dict[str, str] = dict(_us.ETF_LABELS)
DEFAULT_VISIBLE: set[str] = set(_us.DEFAULT_VISIBLE)


def refresh() -> int:
    """Reload tickers/labels from ``us.py``. Returns ETF count."""
    global TICKERS, ETF_LABELS, DEFAULT_VISIBLE, BENCHMARK_YAHOO
    TICKERS = list(_us.tickers)
    ETF_LABELS = dict(_us.ETF_LABELS)
    DEFAULT_VISIBLE = set(_us.DEFAULT_VISIBLE)
    BENCHMARK_YAHOO = _us.BENCHMARK_YAHOO
    return len(TICKERS)


def ensure_loaded() -> int:
    """Refresh universe + RRG caches; safe to call at every US script entry."""
    n = refresh()
    from momentum.etf.us_rrg_universe import sync_us_rrg_universe

    sync_us_rrg_universe()
    return n
