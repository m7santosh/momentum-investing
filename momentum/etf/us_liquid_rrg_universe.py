"""Build RRG rows from liquid-screened US ETF universe."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.us_liquid_screener import ScreenedEtf
from momentum.etf.universes import us_liquid_candidates as pool


@dataclass(frozen=True)
class RrgRow:
    row_id: str
    kind: str
    ref_etf: str
    label: str
    yahoo_ticker: str
    category: str


def build_rrg_rows(screened: list[ScreenedEtf]) -> list[RrgRow]:
    rows: list[RrgRow] = []
    for item in screened:
        rows.append(
            RrgRow(
                row_id=item.ticker,
                kind="etf",
                ref_etf=item.ticker,
                label=item.label,
                yahoo_ticker=item.ticker,
                category=item.category,
            )
        )
    return rows


def build_universe(screened: list[ScreenedEtf]) -> dict:
    """Runtime RRG universe dict consumed by ``RRGIndicatorUsEtfsLiquid3m``."""
    rows = build_rrg_rows(screened)
    row_ids = [r.row_id for r in rows]
    default_visible = {t for t in pool.DEFAULT_VISIBLE if t in row_ids}
    if not default_visible and row_ids:
        default_visible = set(row_ids[: min(16, len(row_ids))])

    labels = {r.row_id: r.label for r in rows}
    categories = {r.row_id: r.category for r in rows}
    row_by_id = {r.row_id: r for r in rows}
    load_tickers = list(dict.fromkeys([*row_ids, pool.BENCHMARK_YAHOO]))

    return {
        "benchmark": pool.BENCHMARK_YAHOO,
        "rows": rows,
        "row_ids": row_ids,
        "row_by_id": row_by_id,
        "labels": labels,
        "categories": categories,
        "default_visible": default_visible,
        "load_tickers": load_tickers,
    }
