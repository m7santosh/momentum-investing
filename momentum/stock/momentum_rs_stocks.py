"""
Nifty LargeMidcap 250 — relative strength vs Nifty LM250 benchmark.

Universe: universes/nifty_largemidcap.py
Filters: above 200 EMA, within 30% of 52w high, ADTV > 5 Cr.
Rank: blends abs-momentum and RS ranks on 3M/6M/9M → Blended_Rank (lower = better).

vs other stock scripts:
- momentum_stocks.py — BSE LargeMidcap; abs returns only; extra trend/up-day filters.
- quality_momentum_rs.py — Quality ~130; RS vs ^CRSLDX (N500 TR); daily scan.
- quality_momentum_rs_lv.py — Quality ~130; + low-vol; calendar rebalance + state.
- quality_momentum_rs_no_lv.py — Quality ~130; rebalance/state; no low-vol.
- quality_momentum_rs_lv_list.py — Quality ~130; + low-vol; list-only, no state.
- momentum_rs_lv_n500.py — Nifty 500; + low-vol; rebalance + state.
- RRGIndicatorStocks.py — visual RRG (selectable universe); not a ranker.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Setup project root for utility imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_DIR, FINAL_RESULT_STOCK_DIR

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

# --- Ticker universe: momentum/stock/universes/nifty_largemidcap.py (edit tickers there)
from momentum.stock.universes.nifty_largemidcap import tickers

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

    FINAL_RESULT_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_STOCK_DIR / "stocks_momentum_final.xlsx"
    df_out[final_cols].to_excel(out_path, index=False)
    print(f"Success: Wrote top {len(df_out)} stocks to {out_path}")

if __name__ == "__main__":
    main()