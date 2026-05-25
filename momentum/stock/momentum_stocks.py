"""
BSE LargeMidcap 250 — absolute momentum rank (no relative strength vs index).

Universe: universes/bse_largemidcap.py
Filters: above 200 EMA, 1Y return ≥ 6.5%, within 25% of 52w high, >45% up-days (6M).
Rank: weighted 3M / 6M / 9M total-return ranks (lower Final_Rank = better) → top 30.

vs other stock scripts:
- momentum_rs_stocks.py — Nifty LargeMidcap universe; blends abs + RS vs Nifty LM250.
- quality_momentum_rs.py — Quality ~130; RS vs ^CRSLDX; daily ranked list + mix summaries.
- quality_momentum_rs_lv.py — Quality ~130; + low-vol filter; calendar rebalance + portfolio state.
- quality_momentum_rs_no_lv.py — Quality ~130; rebalance/state; no low-vol filter.
- quality_momentum_rs_lv_list.py — Quality ~130; + low-vol; fresh list only (no state).
- momentum_rs_lv_n500.py — Nifty 500; + low-vol; rebalance + portfolio state.
- RRGIndicatorStocks.py — visual RRG (selectable universe); not a ranker.
"""
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

# BSE LargeMidcap 250 EQ constituents — edit tickers in universes/bse_largemidcap.py
from momentum.stock.universes.bse_largemidcap import tickers


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

