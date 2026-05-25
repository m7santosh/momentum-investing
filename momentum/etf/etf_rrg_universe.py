"""ETF RRG universe — builds RRG rows/load lists from ``universes/india.py``."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.universes import india

# Aliases from universes/india.py (ticker source of truth lives there)
ETF_TICKERS: list[str] = india.tickers
ETF_TO_NSE_INDEX: dict[str, str | None] = india.ETF_TO_NSE_INDEX
RRG_BENCHMARK_NSE: str = india.BENCHMARK_NSE
RRG_ETF_LABELS: dict[str, str] = india.ETF_LABELS
RRG_DEFAULT_VISIBLE: set[str] = india.DEFAULT_VISIBLE

RRG_ETF_BHAVCOPY: list[str] = [t for t in ETF_TICKERS if ETF_TO_NSE_INDEX.get(t) is None]

RRG_ETF_TICKERS: list[str] = [t for t in ETF_TICKERS if ETF_TO_NSE_INDEX.get(t)]

RRG_NSE_INDICES: list[str] = list(dict.fromkeys(v for v in ETF_TO_NSE_INDEX.values() if v))

INDEX_REF_ETF: dict[str, str] = {}
for _etf, _index in ETF_TO_NSE_INDEX.items():
    if _index and _index not in INDEX_REF_ETF:
        INDEX_REF_ETF[_index] = _etf.replace(".NS", "")


@dataclass(frozen=True)
class RrgRow:
    """One RRG universe line (index EOD or ETF bhavcopy)."""

    row_id: str
    kind: str  # "index" | "etf"
    ref_etf: str
    label: str
    etf_ticker: str | None = None


def _etf_to_symbol(etf_ticker: str) -> str:
    return etf_ticker.replace(".NS", "")


def build_rrg_rows() -> list[RrgRow]:
    """Universe in ``ETF_TICKERS`` order: index rows once per NSE index, then ETF bhavcopy rows."""
    rows: list[RrgRow] = []
    seen_indices: set[str] = set()
    for etf in ETF_TICKERS:
        index_name = ETF_TO_NSE_INDEX.get(etf)
        if index_name:
            if index_name in seen_indices:
                continue
            seen_indices.add(index_name)
            sym = _etf_to_symbol(etf)
            rows.append(
                RrgRow(
                    row_id=index_name,
                    kind="index",
                    ref_etf=INDEX_REF_ETF.get(index_name, sym),
                    label=index_name,
                    etf_ticker=etf,
                )
            )
        elif etf in RRG_ETF_BHAVCOPY:
            sym = _etf_to_symbol(etf)
            rows.append(
                RrgRow(
                    row_id=sym,
                    kind="etf",
                    ref_etf=sym,
                    label=RRG_ETF_LABELS.get(etf, sym),
                    etf_ticker=etf,
                )
            )
    return rows


RRG_ROWS: list[RrgRow] = build_rrg_rows()
RRG_ALL_ROW_IDS: list[str] = [r.row_id for r in RRG_ROWS]
RRG_ROW_BY_ID: dict[str, RrgRow] = {r.row_id: r for r in RRG_ROWS}

RRG_INDEX_ROW_IDS: list[str] = [r.row_id for r in RRG_ROWS if r.kind == "index"]
RRG_ETF_ROW_IDS: list[str] = [r.row_id for r in RRG_ROWS if r.kind == "etf"]

RRG_LOAD_NSE_INDEX_NAMES: list[str] = list(
    dict.fromkeys([*RRG_INDEX_ROW_IDS, RRG_BENCHMARK_NSE])
)
RRG_LOAD_ETF_NSE_SYMBOLS: list[str] = list(RRG_ETF_ROW_IDS)

RRG_DEFAULT_VISIBLE_IDS: set[str] = set()
for _etf in RRG_DEFAULT_VISIBLE:
    _idx = ETF_TO_NSE_INDEX.get(_etf)
    if _idx:
        RRG_DEFAULT_VISIBLE_IDS.add(_idx)
    else:
        RRG_DEFAULT_VISIBLE_IDS.add(_etf_to_symbol(_etf))

RRG_DEFAULT_VISIBLE_INDICES = RRG_DEFAULT_VISIBLE_IDS


def index_ref_etf_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.ref_etf if row else ""


def row_display_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.label if row else row_id


def row_kind(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.kind if row else "index"
