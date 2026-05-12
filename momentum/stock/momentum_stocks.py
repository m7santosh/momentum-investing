import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_STOCK_DIR


def _symbol_for_excel(yahoo_ticker: str) -> str:
    """Strip Yahoo India suffix for display (NSE .NS / BSE .BO)."""
    if yahoo_ticker.endswith(".NS"):
        return yahoo_ticker[: -len(".NS")]
    if yahoo_ticker.endswith(".BO"):
        return yahoo_ticker[: -len(".BO")]
    return yahoo_ticker

# BSE LargeMidcap 250 EQ constituents
tickers = [
    {"symbol": "360ONE.NS", "industry": "Financial Services"},
    {"symbol": "3MINDIA.NS", "industry": "Diversified"},
    {"symbol": "AARTIIND.NS", "industry": "Commodities"},
    {"symbol": "ABB.NS", "industry": "Capital Goods"},
    {"symbol": "ABBOTINDIA.NS", "industry": "Healthcare"},
    {"symbol": "ACC.NS", "industry": "Construction Materials"},
    {"symbol": "ADANIENSOL.NS", "industry": "Power"},
    {"symbol": "ADANIENT.NS", "industry": "Metals & Mining"},
    {"symbol": "ADANIGREEN.NS", "industry": "Power"},
    {"symbol": "ADANIPORTS.NS", "industry": "Services"},
    {"symbol": "ADANIPOWER.NS", "industry": "Power"},
    {"symbol": "ATGL.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "ABCAPITAL.NS", "industry": "Financial Services"},
    {"symbol": "AEGISVOPAK.NS", "industry": "Energy"},
    {"symbol": "AIAENG.NS", "industry": "Capital Goods"},
    {"symbol": "AJANTPHARM.NS", "industry": "Healthcare"},
    {"symbol": "ALKEM.NS", "industry": "Healthcare"},
    {"symbol": "AMBUJACEM.NS", "industry": "Construction Materials"},
    {"symbol": "ANTHEM.NS", "industry": "Healthcare"},
    {"symbol": "APLAPOLLO.NS", "industry": "Capital Goods"},
    {"symbol": "APOLLOHOSP.NS", "industry": "Healthcare"},
    {"symbol": "ASHOKLEY.NS", "industry": "Capital Goods"},
    {"symbol": "ASIANPAINT.NS", "industry": "Consumer Durables"},
    {"symbol": "ASTRAL.NS", "industry": "Capital Goods"},
    {"symbol": "AUBANK.NS", "industry": "Financial Services"},
    {"symbol": "AUROPHARMA.NS", "industry": "Healthcare"},
    {"symbol": "DMART.NS", "industry": "Consumer Services"},
    {"symbol": "AWL.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "AXISBANK.NS", "industry": "Financial Services"},
    {"symbol": "BAJAJ-AUTO.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "BAJFINANCE.NS", "industry": "Financial Services"},
    {"symbol": "BAJAJFINSV.NS", "industry": "Financial Services"},
    {"symbol": "BAJAJHLDNG.NS", "industry": "Financial Services"},
    {"symbol": "BAJAJHFL.NS", "industry": "Financial Services"},
    {"symbol": "BALKRISIND.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "BANDHANBNK.NS", "industry": "Financial Services"},
    {"symbol": "BANKBARODA.NS", "industry": "Financial Services"},
    {"symbol": "BANKINDIA.NS", "industry": "Financial Services"},
    {"symbol": "MAHABANK.NS", "industry": "Financial Services"},
    {"symbol": "BAYERCROP.NS", "industry": "Commodities"},
    {"symbol": "BERGEPAINT.NS", "industry": "Consumer Durables"},
    {"symbol": "BDL.NS", "industry": "Capital Goods"},
    {"symbol": "BEL.NS", "industry": "Capital Goods"},
    {"symbol": "BHARATFORG.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "BHEL.NS", "industry": "Capital Goods"},
    {"symbol": "BPCL.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "BHARTIARTL.NS", "industry": "Telecommunication"},
    {"symbol": "BHARTIHEXA.NS", "industry": "Telecommunication"},
    {"symbol": "GROWW.NS", "industry": "Financial Services"},
    {"symbol": "BIOCON.NS", "industry": "Healthcare"},
    {"symbol": "BLUESTARCO.NS", "industry": "Consumer Durables"},
    {"symbol": "BOSCHLTD.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "FIRSTCRY.NS", "industry": "Consumer Discretionary"},
    {"symbol": "BRITANNIA.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "CANBK.NS", "industry": "Financial Services"},
    {"symbol": "CASTROLIND.NS", "industry": "Energy"},
    {"symbol": "CENTRALBK.NS", "industry": "Financial Services"},
    {"symbol": "CGPOWER.NS", "industry": "Capital Goods"},
    {"symbol": "CHOLAFIN.NS", "industry": "Financial Services"},
    {"symbol": "CIPLA.NS", "industry": "Healthcare"},
    {"symbol": "CLEAN.NS", "industry": "Commodities"},
    {"symbol": "COALINDIA.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "COCHINSHIP.NS", "industry": "Capital Goods"},
    {"symbol": "COFORGE.NS", "industry": "Information Technology"},
    {"symbol": "COLPAL.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "CONCOR.NS", "industry": "Services"},
    {"symbol": "COROMANDEL.NS", "industry": "Chemicals"},
    {"symbol": "CRISIL.NS", "industry": "Financial Services"},
    {"symbol": "CROMPTON.NS", "industry": "Consumer Discretionary"},
    {"symbol": "CUMMINSIND.NS", "industry": "Capital Goods"},
    {"symbol": "DABUR.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "DALBHARAT.NS", "industry": "Construction Materials"},
    {"symbol": "DEEPAKNTR.NS", "industry": "Commodities"},
    {"symbol": "DELHIVERY.NS", "industry": "Services"},
    {"symbol": "DIVISLAB.NS", "industry": "Healthcare"},
    {"symbol": "DIXON.NS", "industry": "Consumer Durables"},
    {"symbol": "DLF.NS", "industry": "Realty"},
    {"symbol": "DRREDDY.NS", "industry": "Healthcare"},
    {"symbol": "EICHERMOT.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "EMAMILTD.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "EMCURE.NS", "industry": "Healthcare"},
    {"symbol": "ENDURANCE.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "ESCORTS.NS", "industry": "Capital Goods"},
    {"symbol": "ETERNAL.NS", "industry": "Consumer Services"},
    {"symbol": "EXIDEIND.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "FEDERALBNK.NS", "industry": "Financial Services"},
    {"symbol": "FORTIS.NS", "industry": "Healthcare"},
    {"symbol": "NYKAA.NS", "industry": "Consumer Services"},
    {"symbol": "GAIL.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "GVT&D.NS", "industry": "Capital Goods"},
    {"symbol": "GICRE.NS", "industry": "Financial Services"},
    {"symbol": "GILLETTE.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "GLAND.NS", "industry": "Healthcare"},
    {"symbol": "GLAXO.NS", "industry": "Healthcare"},
    {"symbol": "GLENMARK.NS", "industry": "Healthcare"},
    {"symbol": "GMRAIRPORT.NS", "industry": "Services"},
    {"symbol": "GODIGIT.NS", "industry": "Financial Services"},
    {"symbol": "GODREJCP.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "GODREJIND.NS", "industry": "Diversified"},
    {"symbol": "GODREJPROP.NS", "industry": "Realty"},
    {"symbol": "GRASIM.NS", "industry": "Construction Materials"},
    {"symbol": "FLUOROCHEM.NS", "industry": "Chemicals"},
    {"symbol": "HAVELLS.NS", "industry": "Consumer Durables"},
    {"symbol": "HCLTECH.NS", "industry": "Information Technology"},
    {"symbol": "HDBFS.NS", "industry": "Financial Services"},
    {"symbol": "HDFCAMC.NS", "industry": "Financial Services"},
    {"symbol": "HDFCBANK.NS", "industry": "Financial Services"},
    {"symbol": "HDFCLIFE.NS", "industry": "Financial Services"},
    {"symbol": "HEROMOTOCO.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "HEXT.NS", "industry": "Information Technology"},
    {"symbol": "HINDALCO.NS", "industry": "Metals & Mining"},
    {"symbol": "HAL.NS", "industry": "Capital Goods"},
    {"symbol": "HINDPETRO.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "HINDUNILVR.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "HINDZINC.NS", "industry": "Metals & Mining"},
    {"symbol": "POWERINDIA.NS", "industry": "Capital Goods"},
    {"symbol": "HONAUT.NS", "industry": "Capital Goods"},
    {"symbol": "HUDCO.NS", "industry": "Financial Services"},
    {"symbol": "HYUNDAI.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "ICICIBANK.NS", "industry": "Financial Services"},
    {"symbol": "ICICIGI.NS", "industry": "Financial Services"},
    {"symbol": "ICICIAMC.NS", "industry": "Financial Services"},
    {"symbol": "ICICIPRULI.NS", "industry": "Financial Services"},
    {"symbol": "IDBI.NS", "industry": "Financial Services"},
    {"symbol": "IDFCFIRSTB.NS", "industry": "Financial Services"},
    {"symbol": "INDIANB.NS", "industry": "Financial Services"},
    {"symbol": "INDHOTEL.NS", "industry": "Consumer Services"},
    {"symbol": "IOC.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "IOB.NS", "industry": "Financial Services"},
    {"symbol": "IRCTC.NS", "industry": "Consumer Services"},
    {"symbol": "IRFC.NS", "industry": "Financial Services"},
    {"symbol": "IREDA.NS", "industry": "Financial Services"},
    {"symbol": "IGL.NS", "industry": "Energy"},
    {"symbol": "INDUSTOWER.NS", "industry": "Telecommunication"},
    {"symbol": "INDUSINDBK.NS", "industry": "Financial Services"},
    {"symbol": "NAUKRI.NS", "industry": "Consumer Services"},
    {"symbol": "INFY.NS", "industry": "Information Technology"},
    {"symbol": "INDIGO.NS", "industry": "Services"},
    {"symbol": "IKS.NS", "industry": "Information Technology"},
    {"symbol": "IPCALAB.NS", "industry": "Healthcare"},
    {"symbol": "ITCHOTELS.NS", "industry": "Consumer Services"},
    {"symbol": "ITC.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "JKCEMENT.NS", "industry": "Construction Materials"},
    {"symbol": "JSL.NS", "industry": "Metals & Mining"},
    {"symbol": "JINDALSTEL.NS", "industry": "Metals & Mining"},
    {"symbol": "JIOFIN.NS", "industry": "Financial Services"},
    {"symbol": "JSWENERGY.NS", "industry": "Power"},
    {"symbol": "JSWINFRA.NS", "industry": "Services"},
    {"symbol": "JSWSTEEL.NS", "industry": "Metals & Mining"},
    {"symbol": "JUBLFOOD.NS", "industry": "Consumer Services"},
    {"symbol": "KALYANKJIL.NS", "industry": "Consumer Durables"},
    {"symbol": "KAYNES.NS", "industry": "Industrials"},
    {"symbol": "KEI.NS", "industry": "Capital Goods"},
    {"symbol": "KOTAKBANK.NS", "industry": "Financial Services"},
    {"symbol": "KPITTECH.NS", "industry": "Information Technology"},
    {"symbol": "LTF.NS", "industry": "Financial Services"},
    {"symbol": "LTTS.NS", "industry": "Information Technology"},
    {"symbol": "LT.NS", "industry": "Construction"},
    {"symbol": "LAURUSLABS.NS", "industry": "Healthcare"},
    {"symbol": "LENSKART.NS", "industry": "Consumer Services"},
    {"symbol": "LGEINDIA.NS", "industry": "Consumer Durables"},
    {"symbol": "LICHSGFIN.NS", "industry": "Financial Services"},
    {"symbol": "LICI.NS", "industry": "Financial Services"},
    {"symbol": "LINDEINDIA.NS", "industry": "Chemicals"},
    {"symbol": "LLOYDSME.NS", "industry": "Metals & Mining"},
    {"symbol": "LODHA.NS", "industry": "Realty"},
    {"symbol": "LTM.NS", "industry": "Information Technology"},
    {"symbol": "LUPIN.NS", "industry": "Healthcare"},
    {"symbol": "M&MFIN.NS", "industry": "Financial Services"},
    {"symbol": "M&M.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "MANKIND.NS", "industry": "Healthcare"},
    {"symbol": "MARICO.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "MARUTI.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "MFSL.NS", "industry": "Financial Services"},
    {"symbol": "MAXHEALTH.NS", "industry": "Healthcare"},
    {"symbol": "MAZDOCK.NS", "industry": "Capital Goods"},
    {"symbol": "MEESHO.NS", "industry": "Consumer Discretionary"},
    {"symbol": "MOTILALOFS.NS", "industry": "Financial Services"},
    {"symbol": "MPHASIS.NS", "industry": "Information Technology"},
    {"symbol": "MRF.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "MUTHOOTFIN.NS", "industry": "Financial Services"},
    {"symbol": "NATIONALUM.NS", "industry": "Metals & Mining"},
    {"symbol": "CDSL.NS", "industry": "Financial Services"},
    {"symbol": "NESTLEIND.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "NHPC.NS", "industry": "Power"},
    {"symbol": "NAM-INDIA.NS", "industry": "Financial Services"},
    {"symbol": "NMDC.NS", "industry": "Metals & Mining"},
    {"symbol": "NTPCGREEN.NS", "industry": "Power"},
    {"symbol": "NTPC.NS", "industry": "Power"},
    {"symbol": "OBEROIRLTY.NS", "industry": "Realty"},
    {"symbol": "ONGC.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "OIL.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "OLAELEC.NS", "industry": "Consumer Discretionary"},
    {"symbol": "PAYTM.NS", "industry": "Financial Services"},
    {"symbol": "OFSS.NS", "industry": "Information Technology"},
    {"symbol": "PIIND.NS", "industry": "Chemicals"},
    {"symbol": "PAGEIND.NS", "industry": "Textiles"},
    {"symbol": "PATANJALI.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "POLICYBZR.NS", "industry": "Financial Services"},
    {"symbol": "PERSISTENT.NS", "industry": "Information Technology"},
    {"symbol": "PETRONET.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "PWL.NS", "industry": "Consumer Discretionary"},
    {"symbol": "PIDILITIND.NS", "industry": "Chemicals"},
    {"symbol": "PINELABS.NS", "industry": "Financial Services"},
    {"symbol": "PIRAMALFIN.NS", "industry": "Financial Services"},
    {"symbol": "POLYCAB.NS", "industry": "Capital Goods"},
    {"symbol": "PFC.NS", "industry": "Financial Services"},
    {"symbol": "POWERGRID.NS", "industry": "Power"},
    {"symbol": "PREMIERENE.NS", "industry": "Capital Goods"},
    {"symbol": "PRESTIGE.NS", "industry": "Realty"},
    {"symbol": "OLECTRA.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "PNB.NS", "industry": "Financial Services"},
    {"symbol": "RVNL.NS", "industry": "Construction"},
    {"symbol": "RECLTD.NS", "industry": "Financial Services"},
    {"symbol": "TRAVELFOOD.NS", "industry": "Consumer Discretionary"},
    {"symbol": "RELIANCE.NS", "industry": "Oil Gas & Consumable Fuels"},
    {"symbol": "MOTHERSON.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "SBICARD.NS", "industry": "Financial Services"},
    {"symbol": "SBILIFE.NS", "industry": "Financial Services"},
    {"symbol": "SCHAEFFLER.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "SHREECEM.NS", "industry": "Construction Materials"},
    {"symbol": "SHRIRAMFIN.NS", "industry": "Financial Services"},
    {"symbol": "ENRIN.NS", "industry": "Capital Goods"},
    {"symbol": "SIEMENS.NS", "industry": "Capital Goods"},
    {"symbol": "SJVN.NS", "industry": "Power"},
    {"symbol": "SOLARINDS.NS", "industry": "Chemicals"},
    {"symbol": "SONACOMS.NS", "industry": "Consumer Discretionary"},
    {"symbol": "SRF.NS", "industry": "Chemicals"},
    {"symbol": "STARHEALTH.NS", "industry": "Financial Services"},
    {"symbol": "SBIN.NS", "industry": "Financial Services"},
    {"symbol": "SAIL.NS", "industry": "Metals & Mining"},
    {"symbol": "SUNPHARMA.NS", "industry": "Healthcare"},
    {"symbol": "SUNTV.NS", "industry": "Consumer Discretionary"},
    {"symbol": "SUPREMEIND.NS", "industry": "Capital Goods"},
    {"symbol": "SUZLON.NS", "industry": "Capital Goods"},
    {"symbol": "SWIGGY.NS", "industry": "Consumer Services"},
    {"symbol": "TATACAP.NS", "industry": "Financial Services"},
    {"symbol": "TATACOMM.NS", "industry": "Telecommunication"},
    {"symbol": "TCS.NS", "industry": "Information Technology"},
    {"symbol": "TATACONSUM.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "TATAELXSI.NS", "industry": "Information Technology"},
    {"symbol": "TMCV.NS", "industry": "Capital Goods"},
    {"symbol": "TMPV.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "TATAPOWER.NS", "industry": "Power"},
    {"symbol": "TATASTEEL.NS", "industry": "Metals & Mining"},
    {"symbol": "TATATECH.NS", "industry": "Information Technology"},
    {"symbol": "TECHM.NS", "industry": "Information Technology"},
    {"symbol": "NIACL.NS", "industry": "Financial Services"},
    {"symbol": "PHOENIXLTD.NS", "industry": "Realty"},
    {"symbol": "RAMCOCEM.NS", "industry": "Commodities"},
    {"symbol": "THERMAX.NS", "industry": "Capital Goods"},
    {"symbol": "TITAN.NS", "industry": "Consumer Durables"},
    {"symbol": "TORNTPHARM.NS", "industry": "Healthcare"},
    {"symbol": "TORNTPOWER.NS", "industry": "Power"},
    {"symbol": "TRENT.NS", "industry": "Consumer Services"},
    {"symbol": "TIINDIA.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "TVSMOTOR.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "UCOBANK.NS", "industry": "Financial Services"},
    {"symbol": "ULTRACEMCO.NS", "industry": "Construction Materials"},
    {"symbol": "UNIONBANK.NS", "industry": "Financial Services"},
    {"symbol": "UBL.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "UNITDSPR.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "UNOMINDA.NS", "industry": "Automobile and Auto Components"},
    {"symbol": "UPL.NS", "industry": "Chemicals"},
    {"symbol": "VBL.NS", "industry": "Fast Moving Consumer Goods"},
    {"symbol": "MANYAVAR.NS", "industry": "Consumer Discretionary"},
    {"symbol": "VMM.NS", "industry": "Consumer Services"},
    {"symbol": "IDEA.NS", "industry": "Telecommunication"},
    {"symbol": "VOLTAS.NS", "industry": "Consumer Durables"},
    {"symbol": "WAAREEENER.NS", "industry": "Capital Goods"},
    {"symbol": "WHIRLPOOL.NS", "industry": "Consumer Discretionary"},
    {"symbol": "WIPRO.NS", "industry": "Information Technology"},
    {"symbol": "YESBANK.NS", "industry": "Financial Services"},
    {"symbol": "ZYDUSLIFE.NS", "industry": "Healthcare"},
]


# Function to fetch historical data
def get_data(ticker, start_date, end_date):
    return yf.download(
        ticker,
        start=start_date,
        end=end_date,
        multi_level_index=False,
        auto_adjust=False,
        progress=False,
    )

# Set dates
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 2)  # 2 year of data for moving averages

# Data dictionary to hold stock data
data = {}

# Fetch data for all tickers
for t in tickers:
    sym = t["symbol"]
    try:
        stock_data = get_data(sym, start_date, end_date)
        if len(stock_data) > 0:
            data[sym] = stock_data
    except Exception as e:
        print(f"Error fetching data for {sym}: {e}")

industry_by_symbol = {t["symbol"]: t["industry"] for t in tickers}

# Create a DataFrame for summary
summary = []

# Analyze each stock
for ticker, df in data.items():
    try:
        adj = df["Adj Close"]
        if isinstance(adj, pd.DataFrame):
            adj = adj.iloc[:, 0]
        adj = adj.squeeze()
        n = len(adj)
        if n < 21:
            continue

        df = df.copy()
        df["EMA200"] = adj.ewm(span=200).mean()

        # Last 1-year return (needs 252 sessions; newer listings may have less Yahoo history)
        if n >= 252:
            one_year_return = (adj.iloc[-1] / adj.iloc[-252] - 1) * 100
        else:
            one_year_return = float("nan")

        # 52-week high (up to 252 sessions, or all available bars)
        high_52_week = adj.iloc[-min(252, n) :].max()
        within_25_pct_high = adj.iloc[-1] >= high_52_week * 0.75  # within 25% of the 52-week high

        # More than 45% up days in the last 6 months (126 trading days)
        six_month_data = adj.iloc[-126:]
        up_days = (six_month_data.pct_change() > 0).sum()
        up_days_pct = up_days / len(six_month_data) * 100  # percentage of up days

        # Filtering criteria
        if (
            adj.iloc[-1] >= df["EMA200"].iloc[-1]
            and one_year_return >= 6.5
            and within_25_pct_high
            and up_days_pct > 45
        ):

            # Calculate returns (guard lookbacks if history is ever borderline)
            return_9m = (
                (adj.iloc[-1] / adj.iloc[-189] - 1) * 100 if n >= 189 else float("nan")
            )
            return_6m = (
                (adj.iloc[-1] / adj.iloc[-126] - 1) * 100 if n >= 126 else float("nan")
            )
            return_3m = (adj.iloc[-1] / adj.iloc[-63] - 1) * 100 if n >= 63 else float("nan")
            return_1m = (adj.iloc[-1] / adj.iloc[-21] - 1) * 100 if n >= 21 else float("nan")

            summary.append({
                "Symbol": _symbol_for_excel(ticker),
                'Industry': industry_by_symbol.get(ticker, ''),
                'Return_9M': return_9m,
                'Return_6M': return_6m,
                'Return_3M': return_3m,
                'Return_1M': return_1m,
            })
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Convert summary to DataFrame
df_summary = pd.DataFrame(summary)
if df_summary.empty:
    print("No tickers passed filters; no Excel file written.")
    raise SystemExit(0)

# Round off returns to 1 decimal place
df_summary['Return_9M'] = df_summary['Return_9M'].round(1)
df_summary['Return_6M'] = df_summary['Return_6M'].round(1)
df_summary['Return_3M'] = df_summary['Return_3M'].round(1)
df_summary['Return_1M'] = df_summary['Return_1M'].round(1)

# Ranking based on returns
df_summary['Rank_9M'] = df_summary['Return_9M'].rank(ascending=False)
df_summary['Rank_6M'] = df_summary['Return_6M'].rank(ascending=False)
df_summary['Rank_3M'] = df_summary['Return_3M'].rank(ascending=False)
#df_summary['Rank_1M'] = df_summary['Return_1M'].rank(ascending=False)

# Calculate final rank
df_summary['Final_Rank'] = 0.50*df_summary['Rank_3M'] + 0.30*df_summary['Rank_6M'] + 0.20*df_summary['Rank_9M'] # calculate the final rank based on the return in the last 3 months, 6 months and 9 months

# Sort by final rank and get top 30
df_summary_sorted = df_summary.sort_values('Final_Rank').head(30)

# Assign position based on final rank
df_summary_sorted['Position'] = np.arange(1, len(df_summary_sorted) + 1)

FINAL_RESULT_STOCK_DIR.mkdir(parents=True, exist_ok=True)
out_path = FINAL_RESULT_STOCK_DIR / "momentum_stocks_ranked.xlsx"
try:
    df_summary_sorted.to_excel(out_path, index=False, engine="openpyxl")
except ImportError:
    print("Missing dependency: pip install openpyxl")
    raise
print(f"Wrote {len(df_summary_sorted)} rows -> {out_path}")

