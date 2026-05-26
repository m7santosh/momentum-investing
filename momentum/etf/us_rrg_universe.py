"""US ETF RRG universe — builds RRG rows from ``universes/us.py``."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.universes import us

ETF_TICKERS: list[str] = list(us.tickers)
RRG_BENCHMARK_YAHOO: str = us.BENCHMARK_YAHOO
RRG_ETF_LABELS: dict[str, str] = us.ETF_LABELS
RRG_DEFAULT_VISIBLE: set[str] = set(us.DEFAULT_VISIBLE)


@dataclass(frozen=True)
class RrgRow:
    """One RRG universe line (Yahoo ETF)."""

    row_id: str
    kind: str  # "etf"
    ref_etf: str
    label: str
    yahoo_ticker: str


def build_rrg_rows() -> list[RrgRow]:
    rows: list[RrgRow] = []
    for ticker in ETF_TICKERS:
        rows.append(
            RrgRow(
                row_id=ticker,
                kind="etf",
                ref_etf=ticker,
                label=RRG_ETF_LABELS.get(ticker, ticker),
                yahoo_ticker=ticker,
            )
        )
    return rows


RRG_ROWS: list[RrgRow] = build_rrg_rows()
RRG_ALL_ROW_IDS: list[str] = [r.row_id for r in RRG_ROWS]
RRG_ROW_BY_ID: dict[str, RrgRow] = {r.row_id: r for r in RRG_ROWS}
RRG_ETF_ROW_IDS: list[str] = [r.row_id for r in RRG_ROWS]

RRG_LOAD_YAHOO_TICKERS: list[str] = list(
    dict.fromkeys([*RRG_ETF_ROW_IDS, RRG_BENCHMARK_YAHOO])
)

RRG_DEFAULT_VISIBLE_IDS: set[str] = set(RRG_DEFAULT_VISIBLE)


def row_ref_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.ref_etf if row else ""


def row_display_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.label if row else row_id


def row_kind(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.kind if row else "etf"
