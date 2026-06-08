"""US ETF pool for RRG liquid helpers — mirrors ``us.py`` exactly.

Edit tickers only in ``momentum/etf/universes/us.py``. All US ETF scripts
(RRG, momentum rankers, backtest, ADV screener) use that canonical list.
"""

from __future__ import annotations

from momentum.etf.universes import us_universe as _us_core

KEY = "us_liquid"
LABEL = "US ETFs (us.py)"
DESCRIPTION = "Canonical US ETF list from momentum/etf/universes/us.py"
BENCHMARK_YAHOO = _us_core.BENCHMARK_YAHOO  # noqa: same as us.py

CATEGORY_CORE = "core"
CATEGORY_SECTOR = "sector"
CATEGORY_COUNTRY = "country"
CATEGORY_THEMATIC = "thematic"
ALL_CATEGORIES = (CATEGORY_CORE,)

# Single source: us.py (no extra discovery tickers).
ALWAYS_INCLUDE: frozenset[str] = frozenset(_us_core.TICKERS)
tickers: list[str] = list(_us_core.TICKERS)
ETF_LABELS: dict[str, str] = dict(_us_core.ETF_LABELS)
DEFAULT_VISIBLE: set[str] = set(_us_core.DEFAULT_VISIBLE)
ETF_CATEGORY: dict[str, str] = {sym: CATEGORY_CORE for sym in _us_core.TICKERS}
