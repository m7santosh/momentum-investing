"""
Stock relative strength vs Nifty LargeMidcap 250 (NIFTY_LARGEMID250.NS).

Filters:
1. Trend: Price must be above 200-day EMA.
2. Proximity: Price must be within 30% of its 52-week high.
3. Liquidity: Average Daily Turnover (ADTV) must be > 5 Crores INR.

Blended Ranking Logic:
- Abs_Momentum_Rank: Weighted rank on raw returns (0.50·3M + 0.30·6M + 0.20·9M).
- Relative_Strength_Rank: Weighted rank on RS vs Benchmark (0.50·3M + 0.30·6M + 0.20·9M).
- Blended_Rank: Average of the above two. Lower is better.
- Volatility_Score: Standard deviation of last 21 days (lower = smoother trend).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Setup project root for utility imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_DIR

# --- Configuration ---
BENCHMARK_TICKER = "^CRSLDX"
MIN_ADTV_CRORES = 5.0  # Minimum 5 Crores daily trading volume

# Lookback periods (Sessions)
LB_1M = 21
LB_3M = 63
LB_6M = 126
LB_9M = 189

# Weights for Ranking (Focusing on the 3M trend for stocks)
W_3M, W_6M, W_9M = 0.50, 0.30, 0.20

# --- Ticker Universe (Nifty LargeMidcap 250) ---
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

# --- Helper Functions ---

def _symbol_for_excel(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace(".NS", "").replace(".BO", "")

def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()

def get_data(ticker: str, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False, progress=False)

def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)

    # 1. Fetch Benchmark Data
    try:
        nifty_df = get_data(BENCHMARK_TICKER, start_date, end_date)
        nifty_adj = _adj_close_series(nifty_df)
    except Exception as e:
        print(f"Error: Benchmark {BENCHMARK_TICKER} ({e})")
        return

    # 2. Fetch Stock Data and Analyze
    summary = []
    industry_by_symbol = {t["symbol"]: t["industry"] for t in tickers}

    for t in tickers:
        sym = t["symbol"]
        try:
            df = get_data(sym, start_date, end_date)
            if len(df) < LB_9M: continue
            
            adj = _adj_close_series(df)
            vol = df["Volume"]

            # --- LIQUIDITY FILTER ---
            # Calculates the average daily value of shares traded in Crores.
            daily_turnover = adj * vol
            adtv_crores = (daily_turnover.tail(20).mean()) / 10000000
            if adtv_crores < MIN_ADTV_CRORES: continue

            # --- TREND FILTERS ---
            ema200 = adj.ewm(span=200).mean().iloc[-1]
            high_52w = adj.iloc[-min(252, len(adj)):].max()
            
            # Must be above 200 EMA and within 30% of 52w High
            if adj.iloc[-1] < ema200 or adj.iloc[-1] < (high_52w * 0.7):
                continue

            # --- PERFORMANCE CALCULATIONS ---
            ret_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
            ret_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100
            ret_6m = (adj.iloc[-1] / adj.iloc[-LB_6M] - 1) * 100
            ret_9m = (adj.iloc[-1] / adj.iloc[-LB_9M] - 1) * 100

            # --- VOLATILITY SCORE ---
            # Measures the standard deviation of daily returns over the last month.
            vol_score = adj.pct_change().tail(21).std() * 100

            # --- RELATIVE STRENGTH ---
            nx = nifty_adj.reindex(adj.index).ffill()
            rs_3m = ret_3m - ((nx.iloc[-1] / nx.iloc[-LB_3M] - 1) * 100)
            rs_6m = ret_6m - ((nx.iloc[-1] / nx.iloc[-LB_6M] - 1) * 100)
            rs_9m = ret_9m - ((nx.iloc[-1] / nx.iloc[-LB_9M] - 1) * 100)

            summary.append({
                "Symbol": _symbol_for_excel(sym),
                "Industry": industry_by_symbol.get(sym, ""),
                "ADTV_Cr": adtv_crores,
                "Return_1M": ret_1m, 
                "Return_3M": ret_3m, 
                "Return_6M": ret_6m, 
                "Return_9M": ret_9m,
                "RS_3M_vs_Bench": rs_3m, 
                "RS_6M_vs_Bench": rs_6m, 
                "RS_9M_vs_Bench": rs_9m,
                "Volatility_Score": vol_score
            })
        except Exception as e:
            print(f"Error analyzing {sym}: {e}")

    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        print("No stocks passed the trend and liquidity filters.")
        return

    # --- RANKING ENGINE ---
    
    # 1. Absolute Return Ranks
    for c in ["3M", "6M", "9M"]:
        df_summary[f"Rank_{c}"] = df_summary[f"Return_{c}"].rank(ascending=False)
    
    # 2. Relative Strength Ranks (na_option=bottom: missing RS → worst rank, avoids NaN in composites)
    for c in ["3M", "6M", "9M"]:
        df_summary[f"Rank_RS_{c}"] = df_summary[f"RS_{c}_vs_Bench"].rank(
            ascending=False, na_option="bottom"
        )

    # 3. Composite Scoring
    df_summary["Abs_Momentum_Rank"] = (W_3M*df_summary["Rank_3M"] + W_6M*df_summary["Rank_6M"] + W_9M*df_summary["Rank_9M"]).rank()
    df_summary["Relative_Strength_Rank"] = (W_3M*df_summary["Rank_RS_3M"] + W_6M*df_summary["Rank_RS_6M"] + W_9M*df_summary["Rank_RS_9M"]).rank()

    # 4. BLENDED RANK (Average of Absolute and Relative Strength)
    df_summary["Blended_Rank"] = (df_summary["Abs_Momentum_Rank"] + df_summary["Relative_Strength_Rank"]) / 2

    # --- FINAL OUTPUT ---
    
    # Sort by Blended Rank
    df_out = df_summary.sort_values("Blended_Rank").head(30).reset_index(drop=True)
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))

    # Round columns for clean Excel output
    round_cols = ["ADTV_Cr", "Blended_Rank", "Volatility_Score", "Return_1M", "Return_3M", "Return_6M", "Return_9M"]
    for c in round_cols: df_out[c] = df_out[c].round(2)

    # Final Column Selection
    final_cols = [
        "Position", 
        "Symbol", 
        "Industry", 
        # "Blended_Rank", 
        "ADTV_Cr", 
        "Volatility_Score", 
        "Return_1M", 
        "Return_3M", 
        "Return_6M",
        "Return_9M",
        # "RS_3M_vs_Bench",
        # "RS_6M_vs_Bench",
        # "RS_9M_vs_Bench",
    ]
    
    # DECISION RULES COMMENTED FOR EXCEL OUTPUT:
    # 1. BLENDED_RANK: Primary factor. Shows stocks leading the market and rising.
    # 2. VOLATILITY_SCORE: 
    #    - < 1.8: Very steady trend (institutional quality).
    #    - > 3.0: Very jumpy (high risk of a "pump and dump" or news spike pullback).
    # 3. ADTV_Cr: Ensures you can sell your position. Never buy more than 1% of this value.

    FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_DIR / "stocks_momentum_final.xlsx"
    df_out[final_cols].to_excel(out_path, index=False)
    print(f"Success: Wrote top {len(df_out)} stocks to {out_path}")

if __name__ == "__main__":
    main()