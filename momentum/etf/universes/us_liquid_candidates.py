"""US ETF candidate pool for ADV$ screening — expand beyond ``us.py`` for RRG.

``us.py`` tickers are always included in the RRG (your core ~60). This file adds
extra sector / country / thematic candidates; the screener keeps core + any name
with ADV$ above your floor (vol filter optional, off by default).

Used by ``RRGIndicatorUsEtfsLiquid3m.py``.
"""

from __future__ import annotations

from momentum.etf.universes import us as _us_core

KEY = "us_liquid"
LABEL = "US ETFs (core + liquid discovery)"
DESCRIPTION = "Core us.py ETFs plus ADV$-screened sector, country, and thematic names"
BENCHMARK_YAHOO = "^GSPC"

CATEGORY_CORE = "core"
CATEGORY_SECTOR = "sector"
CATEGORY_COUNTRY = "country"
CATEGORY_THEMATIC = "thematic"
ALL_CATEGORIES = (CATEGORY_CORE, CATEGORY_SECTOR, CATEGORY_COUNTRY, CATEGORY_THEMATIC)

# Core list always in RRG — same as momentum/etf/universes/us.py
ALWAYS_INCLUDE: frozenset[str] = frozenset(_us_core.tickers)
# (ticker, label, category)
_CANDIDATE_ROWS: list[tuple[str, str, str]] = [
    # --- Sector (SPDR + liquid industry) ---
    ("XLC", "Communication", CATEGORY_SECTOR),
    ("XLY", "Consumer Disc.", CATEGORY_SECTOR),
    ("XLP", "Consumer Staples", CATEGORY_SECTOR),
    ("XLE", "Energy", CATEGORY_SECTOR),
    ("XLF", "Financials", CATEGORY_SECTOR),
    ("XLV", "Health Care", CATEGORY_SECTOR),
    ("XLI", "Industrials", CATEGORY_SECTOR),
    ("XLB", "Materials", CATEGORY_SECTOR),
    ("XLRE", "Real Estate", CATEGORY_SECTOR),
    ("XLK", "Technology", CATEGORY_SECTOR),
    ("XLU", "Utilities", CATEGORY_SECTOR),
    ("SOXX", "Semiconductors", CATEGORY_SECTOR),
    ("SMH", "Semiconductors VanEck", CATEGORY_SECTOR),
    ("XBI", "Biotech", CATEGORY_SECTOR),
    ("IBB", "Biotech iShares", CATEGORY_SECTOR),
    ("XHB", "Homebuilders", CATEGORY_SECTOR),
    ("ITB", "Home Construction", CATEGORY_SECTOR),
    ("XRT", "Retail", CATEGORY_SECTOR),
    ("KBE", "Banks", CATEGORY_SECTOR),
    ("KRE", "Regional Banks", CATEGORY_SECTOR),
    ("XME", "Metals & Mining", CATEGORY_SECTOR),
    ("XOP", "Oil & Gas E&P", CATEGORY_SECTOR),
    ("OIH", "Oil Services", CATEGORY_SECTOR),
    ("IYT", "Transportation", CATEGORY_SECTOR),
    ("PAVE", "Infrastructure", CATEGORY_SECTOR),
    # --- Country / region ---
    ("SPY", "S&P 500", CATEGORY_COUNTRY),
    ("QQQ", "Nasdaq 100", CATEGORY_COUNTRY),
    ("IWM", "Russell 2000", CATEGORY_COUNTRY),
    ("DIA", "Dow 30", CATEGORY_COUNTRY),
    ("EEM", "Emerging Markets", CATEGORY_COUNTRY),
    ("VWO", "EM Vanguard", CATEGORY_COUNTRY),
    ("IEMG", "Core EM", CATEGORY_COUNTRY),
    ("EFA", "EAFE", CATEGORY_COUNTRY),
    ("VEA", "Developed ex-US", CATEGORY_COUNTRY),
    ("EZU", "Eurozone", CATEGORY_COUNTRY),
    ("EWJ", "Japan", CATEGORY_COUNTRY),
    ("EWZ", "Brazil", CATEGORY_COUNTRY),
    ("EWY", "South Korea", CATEGORY_COUNTRY),
    ("EWT", "Taiwan", CATEGORY_COUNTRY),
    ("EWA", "Australia", CATEGORY_COUNTRY),
    ("EWC", "Canada", CATEGORY_COUNTRY),
    ("EWG", "Germany", CATEGORY_COUNTRY),
    ("EWU", "United Kingdom", CATEGORY_COUNTRY),
    ("EWP", "Spain", CATEGORY_COUNTRY),
    ("EWH", "Hong Kong", CATEGORY_COUNTRY),
    ("EWS", "Singapore", CATEGORY_COUNTRY),
    ("EWM", "Malaysia", CATEGORY_COUNTRY),
    ("EIS", "Israel", CATEGORY_COUNTRY),
    ("EZA", "South Africa", CATEGORY_COUNTRY),
    ("THD", "Thailand", CATEGORY_COUNTRY),
    ("EIDO", "Indonesia", CATEGORY_COUNTRY),
    ("INDA", "India", CATEGORY_COUNTRY),
    ("MCHI", "China", CATEGORY_COUNTRY),
    ("FXI", "China Large-Cap", CATEGORY_COUNTRY),
    ("ASHR", "China A-Shares", CATEGORY_COUNTRY),
    ("EPOL", "Poland", CATEGORY_COUNTRY),
    ("EWW", "Mexico", CATEGORY_COUNTRY),
    ("ARGT", "Argentina", CATEGORY_COUNTRY),
    ("TUR", "Turkey", CATEGORY_COUNTRY),
    ("GREK", "Greece", CATEGORY_COUNTRY),
    ("KSA", "Saudi Arabia", CATEGORY_COUNTRY),
    ("FM", "Frontier Markets", CATEGORY_COUNTRY),
    # --- Thematic / factor / commodity ---
    ("ARKK", "ARK Innovation", CATEGORY_THEMATIC),
    ("ARKG", "ARK Genomics", CATEGORY_THEMATIC),
    ("ARKW", "ARK Next Gen Internet", CATEGORY_THEMATIC),
    ("ICLN", "Clean Energy", CATEGORY_THEMATIC),
    ("TAN", "Solar", CATEGORY_THEMATIC),
    ("QCLN", "Clean Energy Alt", CATEGORY_THEMATIC),
    ("SKYY", "Cloud Computing", CATEGORY_THEMATIC),
    ("CLOU", "Cloud Computing Global", CATEGORY_THEMATIC),
    ("HACK", "Cybersecurity", CATEGORY_THEMATIC),
    ("CIBR", "Cybersecurity Nasdaq", CATEGORY_THEMATIC),
    ("ROBO", "Robotics", CATEGORY_THEMATIC),
    ("BOTZ", "Robotics & AI", CATEGORY_THEMATIC),
    ("LIT", "Lithium & Battery", CATEGORY_THEMATIC),
    ("URA", "Uranium", CATEGORY_THEMATIC),
    ("URNM", "Uranium Miners", CATEGORY_THEMATIC),
    ("VNQ", "REIT", CATEGORY_THEMATIC),
    ("IYR", "Real Estate", CATEGORY_THEMATIC),
    ("GLD", "Gold", CATEGORY_THEMATIC),
    ("SLV", "Silver", CATEGORY_THEMATIC),
    ("GDX", "Gold Miners", CATEGORY_THEMATIC),
    ("GDXJ", "Junior Gold Miners", CATEGORY_THEMATIC),
    ("USO", "Crude Oil", CATEGORY_THEMATIC),
    ("UNG", "Natural Gas", CATEGORY_THEMATIC),
    ("DBC", "Commodities", CATEGORY_THEMATIC),
    ("CPER", "Copper", CATEGORY_THEMATIC),
    ("DBA", "Agriculture", CATEGORY_THEMATIC),
    ("WEAT", "Wheat", CATEGORY_THEMATIC),
    ("CORN", "Corn", CATEGORY_THEMATIC),
    ("BITO", "Bitcoin Strategy", CATEGORY_THEMATIC),
    ("IBIT", "Bitcoin iShares", CATEGORY_THEMATIC),
    ("MTUM", "Momentum", CATEGORY_THEMATIC),
    ("QUAL", "Quality", CATEGORY_THEMATIC),
    ("USMV", "Min Volatility", CATEGORY_THEMATIC),
    ("VLUE", "Value", CATEGORY_THEMATIC),
    ("SIZE", "Size", CATEGORY_THEMATIC),
    ("PICK", "Metals & Mining iShares", CATEGORY_THEMATIC),
    ("UUP", "US Dollar", CATEGORY_THEMATIC),
    ("FXE", "Euro", CATEGORY_THEMATIC),
    ("FXY", "Yen", CATEGORY_THEMATIC),
    ("FXB", "Pound", CATEGORY_THEMATIC),
    # --- Extra discovery (liquid sector / theme / country not in us.py) ---
    ("RSP", "S&P 500 Equal Weight", CATEGORY_COUNTRY),
    ("MDY", "S&P MidCap 400", CATEGORY_COUNTRY),
    ("VTI", "Total US Market", CATEGORY_COUNTRY),
    ("VOO", "S&P 500 Vanguard", CATEGORY_COUNTRY),
    ("SCHD", "US Dividend Equity", CATEGORY_THEMATIC),
    ("SCHX", "Large-Cap Core", CATEGORY_COUNTRY),
    ("VGT", "Technology Vanguard", CATEGORY_SECTOR),
    ("VHT", "Health Care Vanguard", CATEGORY_SECTOR),
    ("VDE", "Energy Vanguard", CATEGORY_SECTOR),
    ("VFH", "Financials Vanguard", CATEGORY_SECTOR),
    ("VIS", "Industrials Vanguard", CATEGORY_SECTOR),
    ("VCR", "Consumer Disc. Vanguard", CATEGORY_SECTOR),
    ("VDC", "Consumer Staples Vanguard", CATEGORY_SECTOR),
    ("VPU", "Utilities Vanguard", CATEGORY_SECTOR),
    ("VAW", "Materials Vanguard", CATEGORY_SECTOR),
    ("IGV", "Software", CATEGORY_SECTOR),
    ("IGF", "Global Infrastructure", CATEGORY_SECTOR),
    ("ITA", "Aerospace & Defense", CATEGORY_SECTOR),
    ("KWEB", "China Internet", CATEGORY_COUNTRY),
    ("EMXC", "EM ex-China", CATEGORY_COUNTRY),
    ("EWUS", "Small-Cap UK", CATEGORY_COUNTRY),
    ("EPI", "India WisdomTree", CATEGORY_COUNTRY),
    ("VNM", "Vietnam", CATEGORY_COUNTRY),
    ("ECH", "Chile", CATEGORY_COUNTRY),
    ("EWL", "Switzerland", CATEGORY_COUNTRY),
    ("EWN", "Netherlands", CATEGORY_COUNTRY),
    ("EWD", "Sweden", CATEGORY_COUNTRY),
    ("EWI", "Italy", CATEGORY_COUNTRY),
    ("RSX", "Russia", CATEGORY_COUNTRY),
    ("XAR", "Aerospace & Defense SPDR", CATEGORY_SECTOR),
    ("XES", "Oil & Gas Equipment", CATEGORY_SECTOR),
    ("XPH", "Pharma", CATEGORY_SECTOR),
    ("XSW", "Software SPDR", CATEGORY_SECTOR),
    ("XSD", "Semiconductor SPDR", CATEGORY_SECTOR),
    ("JETS", "Global Airlines", CATEGORY_SECTOR),
    ("XHE", "Health Care Equipment", CATEGORY_SECTOR),
    ("KIE", "Insurance", CATEGORY_SECTOR),
    ("MOO", "Agribusiness", CATEGORY_THEMATIC),
    ("COPX", "Copper Miners", CATEGORY_THEMATIC),
    ("REMX", "Rare Earth", CATEGORY_THEMATIC),
    ("SIL", "Silver Miners", CATEGORY_THEMATIC),
    ("GSG", "Commodities iShares", CATEGORY_THEMATIC),
    ("PALL", "Palladium", CATEGORY_THEMATIC),
    ("PPLT", "Platinum", CATEGORY_THEMATIC),
    ("WGMI", "Bitcoin Miners", CATEGORY_THEMATIC),
    ("ETHA", "Ethereum", CATEGORY_THEMATIC),
    ("XLG", "Top 50 Mega-Cap", CATEGORY_COUNTRY),
    ("MGK", "Mega-Cap Growth", CATEGORY_THEMATIC),
    ("VUG", "Growth", CATEGORY_THEMATIC),
    ("VTV", "Value", CATEGORY_THEMATIC),
    ("IWF", "Russell 1000 Growth", CATEGORY_THEMATIC),
    ("IWD", "Russell 1000 Value", CATEGORY_THEMATIC),
]

# De-dupe by ticker (first category wins).
_seen: set[str] = set()
tickers: list[str] = []
ETF_LABELS: dict[str, str] = {}
ETF_CATEGORY: dict[str, str] = {}
for _sym, _label, _cat in _CANDIDATE_ROWS:
    if _sym in _seen:
        continue
    _seen.add(_sym)
    tickers.append(_sym)
    ETF_LABELS[_sym] = _label
    ETF_CATEGORY[_sym] = _cat

# Merge core us.py tickers (labels from us.py when present).
for _sym in _us_core.tickers:
    if _sym in _seen:
        if _sym in ALWAYS_INCLUDE:
            ETF_CATEGORY[_sym] = CATEGORY_CORE
        continue
    _seen.add(_sym)
    tickers.append(_sym)
    ETF_LABELS[_sym] = _us_core.ETF_LABELS.get(_sym, _sym)
    ETF_CATEGORY[_sym] = CATEGORY_CORE
for _sym in ALWAYS_INCLUDE:
    if _sym in ETF_CATEGORY:
        ETF_CATEGORY[_sym] = CATEGORY_CORE
    if _sym in _us_core.ETF_LABELS:
        ETF_LABELS[_sym] = _us_core.ETF_LABELS[_sym]

DEFAULT_VISIBLE = set(_us_core.DEFAULT_VISIBLE)
