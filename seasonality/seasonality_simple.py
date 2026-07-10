"""
Monthly seasonality analysis for all NSE indices (simplified version).

Usage:
    python seasonality_simple.py
    python seasonality_simple.py --years 5
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from io import StringIO

import numpy as np
import pandas as pd

# Suppress verbose NSE downloading messages
os.environ["NSE_VERBOSE"] = "0"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Redirect stderr to suppress bhavcopy messages during import
_stderr = sys.stderr
sys.stderr = StringIO()

try:
    from utils.nse_bhavcopy import fetch_index_ohlc_history, list_nse_index_names
finally:
    sys.stderr = _stderr


class MonthlySeasonality(NamedTuple):
    """Monthly seasonality metrics for one index."""
    index_name: str
    month: int  # 1-12
    month_name: str
    avg_return_pct: float
    win_rate: float  # % of years where return was positive
    avg_gain_pct: float  # Average return when positive
    avg_loss_pct: float  # Average return when negative
    sample_count: int  # Number of years with data


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def calculate_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly returns from daily OHLCV data."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    if "Close" not in df.columns:
        return pd.DataFrame()
    
    # Get month-end closes
    df['MonthYear'] = df.index.to_period('M')
    monthly_closes = df.groupby('MonthYear')['Close'].last()
    
    if len(monthly_closes) < 2:
        return pd.DataFrame()
    
    # Calculate returns
    monthly_returns = monthly_closes.pct_change() * 100
    
    # Build result dataframe
    result = pd.DataFrame({
        'Date': monthly_closes.index.to_timestamp(),
        'Close': monthly_closes.values,
        'Return': monthly_returns.values,
    })
    
    result['Month'] = result['Date'].dt.month
    result['Year'] = result['Date'].dt.year
    
    return result.iloc[1:]  # Skip first row (NaN return)


def compute_seasonality(monthly_returns: pd.DataFrame, index_name: str) -> list[MonthlySeasonality]:
    """Compute seasonality metrics for each month."""
    if monthly_returns.empty:
        return []
    
    results = []
    
    for month in range(1, 13):
        month_data = monthly_returns[monthly_returns['Month'] == month]
        
        if month_data.empty:
            continue
        
        returns = month_data['Return'].dropna().values
        if len(returns) == 0:
            continue
            
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        avg_return = returns.mean()
        win_rate = (len(positive_returns) / len(returns) * 100) if len(returns) > 0 else 0
        avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        results.append(MonthlySeasonality(
            index_name=index_name,
            month=month,
            month_name=MONTH_NAMES[month - 1],
            avg_return_pct=avg_return,
            win_rate=win_rate,
            avg_gain_pct=avg_gain,
            avg_loss_pct=avg_loss,
            sample_count=len(returns),
        ))
    
    return results


def analyze_index(index_name: str, start_date: date, end_date: date) -> list[MonthlySeasonality]:
    """Analyze one index."""
    try:
        # Suppress stderr during data load
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            df = fetch_index_ohlc_history(index_name, start_date, end_date, quiet=True)
        finally:
            sys.stderr = old_stderr
        
        if df is None or df.empty:
            return []
        
        monthly_returns = calculate_monthly_returns(df)
        if monthly_returns.empty:
            return []
        
        return compute_seasonality(monthly_returns, index_name)
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE Index Monthly Seasonality Analysis")
    parser.add_argument("--years", type=int, default=10, help="Number of years to analyze (default: 10)")
    parser.add_argument("--indices", nargs="*", help="Specific indices to analyze (default: top indices)")
    
    args = parser.parse_args()
    
    end_date = date.today()
    start_date = end_date - timedelta(days=args.years * 365)
    
    print(f"\n{'='*90}")
    print(f"NSE MONTHLY SEASONALITY ANALYSIS")
    print(f"{'='*90}")
    print(f"Period: {start_date} to {end_date} ({args.years} years)")
    print(f"{'='*90}\n")
    
    # Get index list
    if args.indices:
        indices_to_analyze = args.indices
    else:
        # Use top indices
        all_indices = list_nse_index_names()
        indices_to_analyze = [
            idx for idx in all_indices 
            if any(key in idx for key in ['Nifty', 'Sensex', 'Bank', 'IT', 'Pharma', 'Auto'])
        ][:15]  # Top 15
        if not indices_to_analyze:
            indices_to_analyze = all_indices[:15]
    
    print(f"Analyzing {len(indices_to_analyze)} indices...\n")
    
    # Analyze each index
    all_seasonality = []
    for i, idx_name in enumerate(indices_to_analyze, 1):
        print(f"  [{i:2d}] {idx_name:40s}", end=" ", flush=True)
        seasonality = analyze_index(idx_name, start_date, end_date)
        if seasonality:
            all_seasonality.extend(seasonality)
            print(f"OK ({len(seasonality)} months)")
        else:
            print("FAIL")
    
    if not all_seasonality:
        print("\n[ERROR] No data available")
        return
    
    # Create dataframe
    df_seasonality = pd.DataFrame(all_seasonality)
    
    # Report 1: Average returns by month across all indices
    print(f"\n{'='*90}")
    print("AVERAGE MONTHLY RETURNS (%) - All Indices")
    print(f"{'='*90}\n")
    
    avg_by_month = df_seasonality.groupby('month_name')['avg_return_pct'].mean()
    avg_by_month = avg_by_month.reindex([MONTH_NAMES[i] for i in range(12)])
    
    for month, ret in avg_by_month.items():
        bar = "[+]" * max(1, int(abs(ret) / 0.2)) if ret >= 0 else "[-]" * max(1, int(abs(ret) / 0.2))
        symbol = "UP" if ret >= 0 else "DN"
        print(f"  {month:12s}: {ret:7.2f}% {symbol} {bar}")
    
    # Report 2: Win rates
    print(f"\n{'='*90}")
    print("WIN RATE BY MONTH (% of indices/years positive)")
    print(f"{'='*90}\n")
    
    wr_by_month = df_seasonality.groupby('month_name')['win_rate'].mean()
    wr_by_month = wr_by_month.reindex([MONTH_NAMES[i] for i in range(12)])
    
    for month, wr in wr_by_month.items():
        bar = "[+]" * int(wr / 5)
        print(f"  {month:12s}: {wr:6.1f}% {bar}")
    
    # Report 3: Best and worst months
    print(f"\n{'='*90}")
    print("BEST & WORST MONTHS")
    print(f"{'='*90}\n")
    
    monthly_summary = df_seasonality.groupby('month_name').agg({
        'avg_return_pct': 'mean',
        'win_rate': 'mean',
        'sample_count': 'mean',
    }).round(2)
    monthly_summary = monthly_summary.reindex([MONTH_NAMES[i] for i in range(12)])
    monthly_summary.columns = ['Avg Return %', 'Win Rate %', 'Sample Size']
    
    best = monthly_summary.nlargest(3, 'Avg Return %')
    worst = monthly_summary.nsmallest(3, 'Avg Return %')
    
    print("BEST:")
    for month, row in best.iterrows():
        print(f"  UP  {month:12s}: {row['Avg Return %']:7.2f}% return, {row['Win Rate %']:5.1f}% win rate")
    
    print("\nWORST:")
    for month, row in worst.iterrows():
        print(f"  DN  {month:12s}: {row['Avg Return %']:7.2f}% return, {row['Win Rate %']:5.1f}% win rate")
    
    # Report 4: Index-wise performance
    print(f"\n{'='*90}")
    print("INDEX-WISE SUMMARY")
    print(f"{'='*90}\n")
    
    index_summary = df_seasonality.groupby('index_name').agg({
        'avg_return_pct': 'mean',
        'win_rate': 'mean',
    }).round(2)
    index_summary.columns = ['Avg Monthly Return %', 'Avg Win Rate %']
    index_summary = index_summary.sort_values('Avg Monthly Return %', ascending=False)
    
    print(index_summary.to_string())
    
    # Save CSV
    output_dir = Path(__file__).parent.parent / "final_result"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "seasonality_monthly_simple.csv"
    df_seasonality.to_csv(csv_path, index=False)
    
    print(f"\nOK: Detailed results saved to: {csv_path}")
    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
