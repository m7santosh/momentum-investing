"""ETF universe and hardcoded NSE index mappings for RRG (index EOD, not ETF prices)."""

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
# ``None`` = no NSE equity index (commodity / offshore / cash); excluded from RRG load.
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

# ETFs with a mapped NSE index (plot in RRG)
RRG_ETF_TICKERS: list[str] = [t for t in ETF_TICKERS if ETF_TO_NSE_INDEX.get(t)]

# NSE index names for RRG (ETF list is only used to derive this universe)
RRG_NSE_INDICES: list[str] = list(
    dict.fromkeys(v for v in ETF_TO_NSE_INDEX.values() if v)
)

# One reference ETF ticker per index (for table label only; RRG uses index EOD)
INDEX_REF_ETF: dict[str, str] = {}
for _etf, _index in ETF_TO_NSE_INDEX.items():
    if _index and _index not in INDEX_REF_ETF:
        INDEX_REF_ETF[_index] = _etf.replace(".NS", "")


def index_ref_etf_label(index_name: str) -> str:
    """Reference ETF ticker for an NSE index, or empty if none."""
    return INDEX_REF_ETF.get(index_name, "")


# Default visible on launch (sector / thematic India indices)
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
}

RRG_DEFAULT_VISIBLE_INDICES: set[str] = {
    ETF_TO_NSE_INDEX[e]
    for e in RRG_DEFAULT_VISIBLE
    if ETF_TO_NSE_INDEX.get(e)
}
