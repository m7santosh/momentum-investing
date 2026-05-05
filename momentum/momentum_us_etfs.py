"""
ETF momentum for a weekly rebalance: rank the universe on recent total returns (1W / 2W / 1M / 3M),
combine ranks, and keep names trading above a long trend (200 EMA) and not far below recent highs.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_DIR


# Universe: US ETFs/indices on Yahoo
tickers = [
    "XLC", "QQQ", "XLK", "SPY", "IWM", "VEA", "EEM", "MCHI", "CQQQ", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLY", "XLB", "SOXX", "SKYY", "ARKK", "ICLN", "TAN", "GLD", "GDX", "USO", "VNQ", "EWJ", "EWZ", "EWT", "EWY", "EWA", "EWG", "EWC", "EIDO", "FM", "KSA", "ARGT", "TUR", "THD", "GREK", "UUP", "MTUM", "QUAL", "USMV", "DXYZ", "DBC", "CPER", "FXI", "EFA", "PICK", "XME", "URA", "DBA", "UNG", "WEAT", "CORN", "FXE", "FXY", "FXB", "BITO"
]

# Function to fetch historical data
def get_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False)

# Set dates
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 2)  # 2 years of data for moving averages

# Data dictionary to hold stock data
data = {}

# Fetch data for all tickers
for ticker in tickers:
    try:
        stock_data = get_data(ticker, start_date, end_date)
        if len(stock_data) > 0:
            data[ticker] = stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

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
        if n < 64:
            print(f"Skip {ticker}: insufficient history ({n} rows, need >= 64 for 3M return).")
            continue

        # Calculate EMAs
        df = df.copy()
        df["EMA200"] = adj.ewm(span=200).mean()

        # 52-week high (use up to 252 sessions, or all data if shorter)
        high_52_week = adj.iloc[-min(252, n) :].max()
        within_30_pct_high = adj.iloc[-1] >= high_52_week * 0.7

        if adj.iloc[-1] >= df["EMA200"].iloc[-1] and within_30_pct_high:

            # Calculate returns (short / medium horizons)
            return_1w = (adj.iloc[-1] / adj.iloc[-6] - 1) * 100  # 5 trading sessions (~1 calendar week)
            return_2w = (adj.iloc[-1] / adj.iloc[-11] - 1) * 100  # 10 trading sessions (~2 calendar weeks)
            return_1m = (adj.iloc[-1] / adj.iloc[-21] - 1) * 100  # ~1 calendar month
            return_3m = (adj.iloc[-1] / adj.iloc[-64] - 1) * 100  # 63 trading sessions (~3 calendar months)

            summary.append({
                "Symbol": ticker,
                "Return_1W": return_1w,
                "Return_2W": return_2w,
                "Return_1M": return_1m,
                "Return_3M": return_3m,
            })
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Convert summary to DataFrame
df_summary = pd.DataFrame(summary)
if df_summary.empty:
    print("No tickers passed filters; no Excel file written.")
    raise SystemExit(0)

# Round off returns to 1 decimal place
df_summary["Return_1W"] = df_summary["Return_1W"].round(1)
df_summary["Return_2W"] = df_summary["Return_2W"].round(1)
df_summary["Return_1M"] = df_summary["Return_1M"].round(1)
df_summary["Return_3M"] = df_summary["Return_3M"].round(1)

# Ranking based on returns (lower Final_Rank = better)
df_summary["Rank_2W"] = df_summary["Return_2W"].rank(ascending=False)
df_summary["Rank_1M"] = df_summary["Return_1M"].rank(ascending=False)
df_summary["Rank_3M"] = df_summary["Return_3M"].rank(ascending=False)

# Final rank: more weight on longer horizons
df_summary["Final_Rank"] = (
    0.2 * df_summary["Rank_2W"]
    + 0.4 * df_summary["Rank_1M"]
    + 0.4 * df_summary["Rank_3M"]
)

# Sort by final rank and get top 10
df_summary_sorted = df_summary.sort_values('Final_Rank').head(10) # get the top 10 stocks based on the final rank

# Assign position based on final rank
df_summary_sorted['Position'] = np.arange(1, len(df_summary_sorted) + 1) # assign the position based on the final rank

FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
out_path = FINAL_RESULT_DIR / "momentum_us_etfs_ranked.xlsx"
try:
    df_summary_sorted.to_excel(out_path, index=False, engine="openpyxl")
except ImportError:
    print("Missing dependency: pip install openpyxl")
    raise
print(f"Wrote {len(df_summary_sorted)} rows -> {out_path}")
