"""
Quality ~130 — relative strength vs Nifty 500 TR (^CRSLDX); daily ranked scan.

Universe: universes/quality.py (Quality 30 + Midcap Q50 + Smallcap Q50)
Filters: above 200 EMA, within 30% of 52w high, ADTV > 5 Cr (no low-vol gate).
Rank: blends abs-momentum and RS ranks on 3M/6M/9M → Blended_Rank; top-N mix by marketcap/industry.
No rebalance calendar or portfolio state — run fresh each day.

vs other stock scripts:
- quality_momentum_rs_lv.py — same universe/rank; + low-vol quantile; rebalance + state JSON.
- quality_momentum_rs_no_lv.py — same universe; rebalance/state; no low-vol (like this + rebalance).
- quality_momentum_rs_lv_list.py — same filters as _lv; list-only snapshot, no state.
- momentum_rs_lv_n500.py — Nifty 500 universe; + low-vol; rebalance + state.
- momentum_stocks.py — BSE LargeMidcap; abs returns only.
- momentum_rs_stocks.py — Nifty LargeMidcap; RS vs LM250 benchmark.
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
from utils.nse_bhavcopy import fetch_bhavcopy, fetch_nse_live_quotes, nse_symbol_from_yahoo, today_ist

# --- Configuration ---
BENCHMARK_TICKER = "^CRSLDX"
MIN_ADTV_CRORES = 5.0  # Minimum 5 Crores daily trading volume
PORTFOLIO_SIZE = 20  # Holdings size: mix summaries (Marketcap / Industry) use top N by Blended_Rank only
OUTPUT_RANKED_SIZE = 30  # Rows in Excel Sheet1: extend past portfolio to spot weaker names before rebalance

# Lookback periods (Sessions)
LB_1M = 21
LB_3M = 63
LB_6M = 126
LB_9M = 189

# Weights for Ranking (Focusing on the 3M trend for stocks)
W_3M, W_6M, W_9M = 0.50, 0.30, 0.20

# --- Ticker Universe: Nifty 100 Quality 30, Midcap 150 Quality 50, Smallcap 250 Quality 50 ---
from momentum.stock.universes.quality import tickers

# --- Helper Functions ---

def _symbol_for_excel(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace(".NS", "").replace(".BO", "")

def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()

def _fill_ohlcv_from_nse(df: pd.DataFrame, nse_row: dict) -> pd.DataFrame:
    """Patch the last row of *df* with OHLCV from an NSE source dict."""
    idx = df.index[-1]
    last_vol = df.iloc[-1].get("Volume")
    df.at[idx, "Close"] = nse_row["close"]
    df.at[idx, "Adj Close"] = nse_row["close"]
    df.at[idx, "Open"] = nse_row["open"]
    df.at[idx, "High"] = nse_row["high"]
    df.at[idx, "Low"] = nse_row["low"]
    if pd.isna(last_vol) or last_vol == 0:
        df.at[idx, "Volume"] = nse_row["volume"]
    return df


def get_data(ticker: str, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return df
    if pd.notna(df.iloc[-1].get("Close")):
        return df
    trade_dt = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
    if ticker.endswith(".NS"):
        nse_sym = nse_symbol_from_yahoo(ticker)
        bhav = fetch_bhavcopy(trade_dt)
        if nse_sym in bhav:
            return _fill_ohlcv_from_nse(df, bhav[nse_sym])
        if trade_dt == today_ist():
            live = fetch_nse_live_quotes()
            if nse_sym in live:
                return _fill_ohlcv_from_nse(df, live[nse_sym])
    df = df.dropna(subset=["Close"])
    return df


def portfolio_mix_summary(
    df_top: pd.DataFrame,
    column: str,
    *,
    label_header: str,
    fixed_order: list[str] | None = None,
) -> pd.DataFrame:
    """Counts and % of portfolio for `column` (e.g. Marketcap with fixed_order, or Industry sorted by count)."""
    total = len(df_top)
    counts = df_top[column].value_counts() if total else pd.Series(dtype=int)
    rows: list[dict] = []
    if fixed_order:
        for cat in fixed_order:
            n = int(counts.get(cat, 0)) if total else 0
            pct = (100.0 * n / total) if total else 0.0
            rows.append({label_header: cat, "Count": n, "Pct": round(pct, 2)})
        for cat in counts.index:
            if cat not in fixed_order:
                n = int(counts[cat])
                rows.append({label_header: cat, "Count": n, "Pct": round(100.0 * n / total, 2)})
    else:
        for cat, n in counts.items():
            rows.append({label_header: cat, "Count": int(n), "Pct": round(100.0 * int(n) / total, 2)})
    return pd.DataFrame(rows)


def write_combined_portfolio_summary_sheet(
    writer: pd.ExcelWriter,
    df_mcap: pd.DataFrame,
    df_industry: pd.DataFrame,
    *,
    sheet_name: str = "Portfolio_Summary",
    startrow_mcap: int = 0,
) -> None:
    """One sheet: Marketcap block, blank row, Industry block (each with its own header row)."""
    df_mcap.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow_mcap)
    startrow_ind = startrow_mcap + len(df_mcap) + 2
    df_industry.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow_ind)


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
    marketcap_by_symbol = {t["symbol"]: t["marketcap"] for t in tickers}

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
                "Marketcap": marketcap_by_symbol.get(sym, ""),
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
    df_sorted = df_summary.sort_values("Blended_Rank")

    # Portfolio mix: only the top PORTFOLIO_SIZE names (actual book), not the extended ranked list
    df_portfolio_slice = df_sorted.head(PORTFOLIO_SIZE)
    df_portfolio_mcap = portfolio_mix_summary(
        df_portfolio_slice,
        "Marketcap",
        label_header="Marketcap",
        fixed_order=["Largecap", "Midcap", "Smallcap"],
    )
    df_portfolio_industry = portfolio_mix_summary(
        df_portfolio_slice, "Industry", label_header="Industry", fixed_order=None
    )

    # Sheet1: top OUTPUT_RANKED_SIZE for decisions on borderline / lower-ranked names
    df_out = df_sorted.head(OUTPUT_RANKED_SIZE).reset_index(drop=True)
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))

    # Round columns for clean Excel output
    round_cols = ["ADTV_Cr", "Blended_Rank", "Volatility_Score", "Return_1M", "Return_3M", "Return_6M", "Return_9M"]
    for c in round_cols:
        if c in df_out.columns:
            df_out[c] = df_out[c].round(2)

    # Final Column Selection
    final_cols = [
        "Position",
        "Symbol",
        "Industry",
        "Marketcap",
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
    out_path = FINAL_RESULT_STOCK_DIR / "quality_momentum_rs.xlsx"

    print(f"\nPortfolio summary (holdings = top {PORTFOLIO_SIZE} by Blended_Rank, by Marketcap):")
    for _, r in df_portfolio_mcap.iterrows():
        print(f"  {r['Marketcap']}: {int(r['Count'])}  ({r['Pct']}%)")

    print(f"\nPortfolio summary (holdings = top {PORTFOLIO_SIZE} by Blended_Rank, by Industry):")
    for _, r in df_portfolio_industry.iterrows():
        print(f"  {r['Industry']}: {int(r['Count'])}  ({r['Pct']}%)")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out[final_cols].to_excel(writer, sheet_name="Sheet1", index=False)
        write_combined_portfolio_summary_sheet(writer, df_portfolio_mcap, df_portfolio_industry)

    print(
        f"\nSuccess: Wrote top {len(df_out)} ranked rows (OUTPUT_RANKED_SIZE={OUTPUT_RANKED_SIZE}); "
        f"mix from top {PORTFOLIO_SIZE} holdings → {out_path}"
    )

if __name__ == "__main__":
    main()