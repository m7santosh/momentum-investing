"""ETF universe and RRG mappings: NSE index EOD + CM bhavcopy for commodity/international ETFs."""

from __future__ import annotations

from dataclasses import dataclass

# Same list as momentum_etfs.py
ETF_TICKERS: list[str] = [
    "ALPHA.NS",
    "AUTOBEES.NS",
    "BANKBEES.NS",
    "CONSUMBEES.NS",
    "CPSEETF.NS",
    "MOENERGY.NS",
    "FMCGIETF.NS",
    "GOLDBEES.NS",
    "GROWWPOWER.NS",
    "GROWWRAIL.NS",
    "HDFCSML250.NS",
    "HEALTHIETF.NS",
    "HNGSNGBEES.NS",
    "ICICIB22.NS",
    "INFRABEES.NS",
    "ITBEES.NS",
    "LIQUIDCASE.NS",
    "MAFANG.NS",
    "MAHKTECH.NS",
    "METALIETF.NS",
    "MIDCAPETF.NS",
    "MOCAPITAL.NS",
    "MODEFENCE.NS",
    "MON100.NS",
    "MOREALTY.NS",
    "MOTOUR.NS",
    "MOVALUE.NS",
    "NEXT50IETF.NS",
    "NIFTYBEES.NS",
    "OILIETF.NS",
    "PHARMABEES.NS",
    "PSUBNKBEES.NS",
    "PVTBANIETF.NS",
    "MOMIDMTM.NS",
    "SILVERBEES.NS",
    "SMALLCAP.NS",
    "CHEMICAL.NS",
    "GROWWNET.NS",
]

# RRG benchmark: Nifty 500 index EOD (matches ^CRSLDX in momentum scripts)
RRG_BENCHMARK_NSE = "Nifty 500"

# ETF Yahoo symbol -> exact ``Index Name`` in NSE ``ind_close_all`` CSV.
# ``None`` = no NSE equity index → use CM bhavcopy on the ETF symbol instead.
ETF_TO_NSE_INDEX: dict[str, str | None] = {
    "ALPHA.NS": "Nifty Alpha 50",
    "AUTOBEES.NS": "Nifty Auto",
    "BANKBEES.NS": "Nifty Bank",
    "CONSUMBEES.NS": "Nifty India Consumption",
    "CPSEETF.NS": "Nifty CPSE",
    "MOENERGY.NS": "Nifty Energy",
    "FMCGIETF.NS": "Nifty FMCG",
    "GOLDBEES.NS": None,
    "GROWWPOWER.NS": None,
    "GROWWRAIL.NS": "Nifty India Railways PSU",
    "HDFCSML250.NS": "Nifty Smallcap 250",
    "HEALTHIETF.NS": "Nifty Healthcare Index",
    "HNGSNGBEES.NS": None,
    "ICICIB22.NS": None,
    "INFRABEES.NS": "Nifty Infrastructure",
    "ITBEES.NS": "Nifty IT",
    "LIQUIDCASE.NS": None,
    "MAFANG.NS": None,
    "MAHKTECH.NS": None,
    "METALIETF.NS": "Nifty Metal",
    "MIDCAPETF.NS": "Nifty Midcap 150",
    "MOCAPITAL.NS": "Nifty Capital Markets",
    "MODEFENCE.NS": "Nifty India Defence",
    "MON100.NS": None,
    "MOREALTY.NS": "Nifty Realty",
    "MOTOUR.NS": "Nifty India Tourism",
    "MOVALUE.NS": "NIFTY500 Value 50",
    "NEXT50IETF.NS": "Nifty Next 50",
    "NIFTYBEES.NS": "Nifty 50",
    "OILIETF.NS": "Nifty Oil & Gas",
    "PHARMABEES.NS": "Nifty Pharma",
    "PSUBNKBEES.NS": "Nifty PSU Bank",
    "PVTBANIETF.NS": "Nifty Private Bank",
    "MOMIDMTM.NS": "Nifty Midcap150 Momentum 50",
    "SILVERBEES.NS": None,
    "SMALLCAP.NS": "Nifty Smallcap 250 Momentum Quality 100",
    "CHEMICAL.NS": "Nifty Chemicals",
    "GROWWNET.NS": "Nifty India Internet",
}

# ETFs without ``ind_close_all`` index — weekly EOD from NSE CM bhavcopy (listed ETF price).
RRG_ETF_BHAVCOPY: list[str] = [t for t in ETF_TICKERS if ETF_TO_NSE_INDEX.get(t) is None]

# Display names for bhavcopy-only rows (Index column in RRG table).
RRG_ETF_LABELS: dict[str, str] = {
    "GOLDBEES.NS": "Gold ETF",
    "SILVERBEES.NS": "Silver ETF",
    "MAFANG.NS": "US FANG+ ETF",
    "MON100.NS": "Nasdaq 100 ETF",
    "MAHKTECH.NS": "Hang Seng Tech ETF",
    "HNGSNGBEES.NS": "Hang Seng ETF",
    "LIQUIDCASE.NS": "Liquid ETF",
    "ICICIB22.NS": "G-Sec ETF",
    "GROWWPOWER.NS": "Power ETF",
}

RRG_ETF_TICKERS: list[str] = [t for t in ETF_TICKERS if ETF_TO_NSE_INDEX.get(t)]

RRG_NSE_INDICES: list[str] = list(
    dict.fromkeys(v for v in ETF_TO_NSE_INDEX.values() if v)
)

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

# Exact NSE names/symbols to download history for (RRG only — not the full exchange universe).
RRG_LOAD_NSE_INDEX_NAMES: list[str] = list(
    dict.fromkeys([*RRG_INDEX_ROW_IDS, RRG_BENCHMARK_NSE])
)
RRG_LOAD_ETF_NSE_SYMBOLS: list[str] = list(RRG_ETF_ROW_IDS)


def index_ref_etf_label(row_id: str) -> str:
    """Reference ETF ticker for a row (index or ETF)."""
    row = RRG_ROW_BY_ID.get(row_id)
    return row.ref_etf if row else ""


def row_display_label(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.label if row else row_id


def row_kind(row_id: str) -> str:
    row = RRG_ROW_BY_ID.get(row_id)
    return row.kind if row else "index"


# Default visible on launch (sector basket + key commodity/international sleeves)
RRG_DEFAULT_VISIBLE: set[str] = {
    "AUTOBEES.NS",
    "BANKBEES.NS",
    "CONSUMBEES.NS",
    "MOENERGY.NS",
    "FMCGIETF.NS",
    "HEALTHIETF.NS",
    "INFRABEES.NS",
    "ITBEES.NS",
    "METALIETF.NS",
    "MOCAPITAL.NS",
    "MODEFENCE.NS",
    "MOREALTY.NS",
    "MOTOUR.NS",
    "OILIETF.NS",
    "PHARMABEES.NS",
    "PSUBNKBEES.NS",
    "PVTBANIETF.NS",
    "CHEMICAL.NS",
    "GROWWRAIL.NS",
    "GOLDBEES.NS",
    "MON100.NS",
    "MAFANG.NS",
}

RRG_DEFAULT_VISIBLE_IDS: set[str] = set()
for _etf in RRG_DEFAULT_VISIBLE:
    _idx = ETF_TO_NSE_INDEX.get(_etf)
    if _idx:
        RRG_DEFAULT_VISIBLE_IDS.add(_idx)
    else:
        RRG_DEFAULT_VISIBLE_IDS.add(_etf_to_symbol(_etf))

# Backward-compatible alias
RRG_DEFAULT_VISIBLE_INDICES = RRG_DEFAULT_VISIBLE_IDS
