"""US ETF RRG universe — builds RRG rows from ``universes/us.py``."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.universes import us


@dataclass(frozen=True)
class RrgRow:
    """One RRG universe line (Yahoo ETF)."""

    row_id: str
    kind: str  # "etf"
    ref_etf: str
    label: str
    yahoo_ticker: str


RRG_BENCHMARK_YAHOO: str = us.BENCHMARK_YAHOO
RRG_ETF_LABELS: dict[str, str] = dict(us.ETF_LABELS)
RRG_DEFAULT_VISIBLE: set[str] = set(us.DEFAULT_VISIBLE)
ETF_TICKERS: list[str] = list(us.tickers)
RRG_ROWS: list[RrgRow] = []
RRG_ALL_ROW_IDS: list[str] = []
RRG_ROW_BY_ID: dict[str, RrgRow] = {}
RRG_ETF_ROW_IDS: list[str] = []
RRG_LOAD_YAHOO_TICKERS: list[str] = []
RRG_DEFAULT_VISIBLE_IDS: set[str] = set()


def build_rrg_rows(
    tickers: list[str] | None = None,
    labels: dict[str, str] | None = None,
) -> list[RrgRow]:
    syms = list(tickers if tickers is not None else us.tickers)
    name_map = labels if labels is not None else us.ETF_LABELS
    rows: list[RrgRow] = []
    for ticker in syms:
        rows.append(
            RrgRow(
                row_id=ticker,
                kind="etf",
                ref_etf=ticker,
                label=name_map.get(ticker, ticker),
                yahoo_ticker=ticker,
            )
        )
    return rows


def sync_us_rrg_universe() -> int:
    """Rebuild module-level RRG constants from current ``us.py``."""
    global ETF_TICKERS, RRG_BENCHMARK_YAHOO, RRG_ETF_LABELS, RRG_DEFAULT_VISIBLE
    global RRG_ROWS, RRG_ALL_ROW_IDS, RRG_ROW_BY_ID, RRG_ETF_ROW_IDS
    global RRG_LOAD_YAHOO_TICKERS, RRG_DEFAULT_VISIBLE_IDS

    ETF_TICKERS = list(us.tickers)
    RRG_BENCHMARK_YAHOO = us.BENCHMARK_YAHOO
    RRG_ETF_LABELS = dict(us.ETF_LABELS)
    RRG_DEFAULT_VISIBLE = set(us.DEFAULT_VISIBLE)
    RRG_ROWS = build_rrg_rows(ETF_TICKERS, RRG_ETF_LABELS)
    RRG_ALL_ROW_IDS = [r.row_id for r in RRG_ROWS]
    RRG_ROW_BY_ID = {r.row_id: r for r in RRG_ROWS}
    RRG_ETF_ROW_IDS = [r.row_id for r in RRG_ROWS]
    RRG_LOAD_YAHOO_TICKERS = list(
        dict.fromkeys([*RRG_ETF_ROW_IDS, RRG_BENCHMARK_YAHOO])
    )
    RRG_DEFAULT_VISIBLE_IDS = set(RRG_DEFAULT_VISIBLE)
    return len(RRG_ETF_ROW_IDS)


def row_ref_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.ref_etf if row else ""


def row_display_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.label if row else row_id


def row_kind(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.kind if row else "etf"


sync_us_rrg_universe()
