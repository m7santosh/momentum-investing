"""Stock RRG universe — builds RRG rows/load lists from ``universes/*.py``."""

from __future__ import annotations

import types

from momentum.rrg_core import RrgRow
from momentum.stock.universes import BY_KEY, DEFAULT_KEY

RRG_BENCHMARK_NSE = "Nifty 500"
RRG_LOAD_NSE_INDEX_NAMES: list[str] = [RRG_BENCHMARK_NSE]
RRG_ROWS: list[RrgRow] = []
RRG_ALL_ROW_IDS: list[str] = []
RRG_ROW_BY_ID: dict[str, RrgRow] = {}
RRG_STOCK_ROW_IDS: list[str] = []
RRG_LOAD_NSE_STOCK_SYMBOLS: list[str] = []
RRG_DEFAULT_VISIBLE_IDS: set[str] = set()

_active_key = DEFAULT_KEY
_active_module: types.ModuleType = BY_KEY[DEFAULT_KEY]


def _yahoo_to_nse(yahoo_symbol: str) -> str:
    return yahoo_symbol.replace(".NS", "").replace(".BO", "").upper()


def _build_rrg_rows(tickers: list[dict[str, str]]) -> list[RrgRow]:
    rows: list[RrgRow] = []
    for item in tickers:
        sym = _yahoo_to_nse(item["symbol"])
        rows.append(
            RrgRow(
                row_id=sym,
                kind="stock",
                ref_label=item.get("industry", ""),
                label=sym,
            )
        )
    return rows


def use_universe_module(mod: types.ModuleType) -> None:
    """Point RRG at a universe module (must define ``tickers``, ``DEFAULT_VISIBLE``, …)."""
    global _active_key, _active_module
    global RRG_ROWS, RRG_ALL_ROW_IDS, RRG_ROW_BY_ID, RRG_STOCK_ROW_IDS
    global RRG_LOAD_NSE_STOCK_SYMBOLS, RRG_DEFAULT_VISIBLE_IDS, RRG_BENCHMARK_NSE
    global RRG_LOAD_NSE_INDEX_NAMES

    _active_module = mod
    _active_key = getattr(mod, "KEY", mod.__name__.rsplit(".", 1)[-1])
    RRG_BENCHMARK_NSE = getattr(mod, "BENCHMARK_NSE", "Nifty 500")
    RRG_LOAD_NSE_INDEX_NAMES = [RRG_BENCHMARK_NSE]

    tickers = mod.tickers
    RRG_ROWS = _build_rrg_rows(tickers)
    RRG_ALL_ROW_IDS = [r.row_id for r in RRG_ROWS]
    RRG_ROW_BY_ID = {r.row_id: r for r in RRG_ROWS}
    RRG_STOCK_ROW_IDS = list(RRG_ALL_ROW_IDS)
    RRG_LOAD_NSE_STOCK_SYMBOLS = list(RRG_STOCK_ROW_IDS)
    RRG_DEFAULT_VISIBLE_IDS = set(getattr(mod, "DEFAULT_VISIBLE", ()))


def use_universe_key(key: str) -> None:
    mod = BY_KEY.get(key)
    if mod is None:
        known = ", ".join(sorted(BY_KEY))
        raise KeyError(f"Unknown universe {key!r}. Known: {known}")
    use_universe_module(mod)


def active_universe_key() -> str:
    return _active_key


def active_universe_module() -> types.ModuleType:
    return _active_module


use_universe_key(DEFAULT_KEY)


def row_ref_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.ref_label if row else ""


def row_display_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.label if row else row_id


def row_kind(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.kind if row else "stock"
