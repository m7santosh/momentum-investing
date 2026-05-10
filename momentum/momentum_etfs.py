"""
ETF momentum for a weekly rebalance: rank the universe on recent total returns (1W / 2W / 1M),
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


def _symbol_for_excel(yahoo_ticker: str) -> str:
    """Strip Yahoo NSE suffix for spreadsheet display."""
    if yahoo_ticker.endswith(".NS"):
        return yahoo_ticker[: -len(".NS")]
    if yahoo_ticker.endswith(".BO"):
        return yahoo_ticker[: -len(".BO")]
    return yahoo_ticker


# Universe: India-listed ETFs/indices on Yahoo (.NS)
tickers = ["ALPHA.NS", "AUTOBEES.NS", "BANKBEES.NS","CONSUMBEES.NS","CPSEETF.NS","MOENERGY.NS", "FMCGIETF.NS", "GOLDBEES.NS", "GROWWPOWER.NS", "GROWWRAIL.NS", "HDFCSML250.NS", "HEALTHIETF.NS", "HNGSNGBEES.NS", "ICICIB22.NS", "INFRABEES.NS", "ITBEES.NS", "LIQUIDCASE.NS", "MAFANG.NS", "MAHKTECH.NS", "METALIETF.NS", "MIDCAPETF.NS", "MOCAPITAL.NS", "MODEFENCE.NS", "MON100.NS", "MOREALTY.NS", "MOTOUR.NS", "MOVALUE.NS", "NEXT50IETF.NS", "NIFTYBEES.NS", "OILIETF.NS", "PHARMABEES.NS", "PSUBNKBEES.NS", "PVTBANIETF.NS", "MOMIDMTM.NS", "SILVERBEES.NS", "SMALLCAP.NS"]

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
        if n < 21:
            print(f"Skip {ticker}: insufficient history ({n} rows, need >= 21 for 1M return).")
            continue

        # Calculate EMAs
        df = df.copy()
        df["EMA200"] = adj.ewm(span=200).mean()

        # Last 1-year return (only if a full year of bars exists)
        if n >= 252:
            one_year_return = (adj.iloc[-1] / adj.iloc[-252] - 1) * 100
        else:
            one_year_return = float("nan")

        # 52-week high (use up to 252 sessions, or all data if shorter)
        high_52_week = adj.iloc[-min(252, n) :].max()
        within_30_pct_high = adj.iloc[-1] >= high_52_week * 0.7 # check if the current price is within 30% of the 52-week high

        # More than 45% up days in the last 6 months (126 trading days)
        six_month_data = adj.iloc[-126:]
        up_days = (six_month_data.pct_change() > 0).sum()  # count the number of days that the price increased or closed higher than the previous day
        up_days_pct = up_days / len(six_month_data) * 100  # calculate the percentage of days that the price increased

        # Filtering criteria
        if (adj.iloc[-1] >= df["EMA200"].iloc[-1] and # check if the current price is above the 200-day EMA
            # one_year_return >= 6.5 and
            # up_days_pct > 45 and
            within_30_pct_high
            ): # check if the price has increased more than 45% days in the last 6 months

            # Calculate returns (short horizons only)
            return_1m = (adj.iloc[-1] / adj.iloc[-21] - 1) * 100 # calculate the return in the last month
            return_1w = (adj.iloc[-1] / adj.iloc[-6] - 1) * 100  # 5 trading sessions (~1 calendar week)
            return_2w = (adj.iloc[-1] / adj.iloc[-11] - 1) * 100  # 10 trading sessions (~2 calendar weeks)

            summary.append({
                "Symbol": _symbol_for_excel(ticker),
                'Return_1M': return_1m,
                'Return_2W': return_2w,
                'Return_1W': return_1w,
            })
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Convert summary to DataFrame
df_summary = pd.DataFrame(summary)
if df_summary.empty:
    print("No tickers passed filters; no Excel file written.")
    raise SystemExit(0)

# Round off returns to 1 decimal place
df_summary['Return_1M'] = df_summary['Return_1M'].round(1)
df_summary['Return_2W'] = df_summary['Return_2W'].round(1)
df_summary['Return_1W'] = df_summary['Return_1W'].round(1)

# Ranking based on returns
df_summary['Rank_1M'] = df_summary['Return_1M'].rank(ascending=False) # rank the stocks based on the return in the last month
df_summary['Rank_2W'] = df_summary['Return_2W'].rank(ascending=False)
df_summary['Rank_1W'] = df_summary['Return_1W'].rank(ascending=False)

# Final rank: 30% Rank_1W + 30% Rank_2W + 40% Rank_1M (lower = better after sort_values on this sum)
# more importance to recent performance
df_summary['Final_Rank'] = (
    0.4 * df_summary['Rank_1W']
    + 0.4 * df_summary['Rank_2W']
    + 0.2 * df_summary['Rank_1M']
)

# Sort by final rank and get top 10
df_summary_sorted = df_summary.sort_values('Final_Rank').head(10) # get the top 10 stocks based on the final rank

# Assign position based on final rank
df_summary_sorted['Position'] = np.arange(1, len(df_summary_sorted) + 1) # assign the position based on the final rank

FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
out_path = FINAL_RESULT_DIR / "momentum_etfs_ranked.xlsx"
try:
    df_summary_sorted.to_excel(out_path, index=False, engine="openpyxl")
except ImportError:
    print("Missing dependency: pip install openpyxl")
    raise
print(f"Wrote {len(df_summary_sorted)} rows -> {out_path}")
