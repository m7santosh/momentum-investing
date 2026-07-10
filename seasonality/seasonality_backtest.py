"""
Monthly seasonality analysis using the existing backtest infrastructure.

Usage:
    python seasonality_backtest.py
    python seasonality_backtest.py --years 5 --top-n 20
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress verbose output
old_stderr = sys.stderr
sys.stderr = StringIO()

try:
    from utils.india_market_data import get_india_market_data
    from momentum.index.nifty_indices import build_nifty_index_universe
finally:
    sys.stderr = old_stderr

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def calculate_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly returns from daily OHLCV data."""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Get month-end closes
    monthly_data = []
    for period, group in df.groupby(df.index.to_period('M')):
        close = group['Close'].iloc[-1]
        monthly_data.append({
            'Date': period.to_timestamp(),
            'Close': close,
            'Month': period.month,
            'Year': period.year,
        })
    
    if len(monthly_data) < 2:
        return pd.DataFrame()
    
    result = pd.DataFrame(monthly_data)
    result['Return'] = result['Close'].pct_change() * 100
    
    return result.iloc[1:]  # Skip first row


def compute_seasonality(monthly_returns: pd.DataFrame, index_name: str) -> dict:
    """Compute seasonality for all months."""
    if monthly_returns.empty:
        return {}
    
    results = {}
    for month in range(1, 13):
        month_data = monthly_returns[monthly_returns['Month'] == month]
        if month_data.empty:
            continue
        
        returns = month_data['Return'].dropna().values
        if len(returns) == 0:
            continue
        
        positive = returns[returns > 0]
        results[month] = {
            'month_name': MONTH_NAMES[month - 1],
            'avg_return': returns.mean(),
            'win_rate': len(positive) / len(returns) * 100,
            'avg_gain': positive.mean() if len(positive) > 0 else 0,
            'avg_loss': (returns[returns < 0]).mean() if len(returns[returns < 0]) > 0 else 0,
            'sample_size': len(returns),
        }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE Index Monthly Seasonality")
    parser.add_argument("--years", type=int, default=10, help="Years to analyze")
    parser.add_argument("--top-n", type=int, default=None, help="Top N indices to show")
    args = parser.parse_args()
    
    end_date = date.today()
    start_date = end_date - timedelta(days=args.years * 365)
    
    print(f"\n{'='*100}")
    print(f"NSE MONTHLY SEASONALITY ANALYSIS")
    print(f"{'='*100}")
    print(f"Period: {start_date} to {end_date} ({args.years} years)")
    print(f"{'='*100}\n")
    
    # Get indices
    indices = build_nifty_index_universe()
    print(f"Testing {len(indices)} indices...\n")
    
    # Analyze indices
    all_seasonality = {}
    successful = 0
    
    for i, idx in enumerate(indices):
        print(f"  [{i+1:3d}] {idx.label:40s}", end=" ", flush=True)
        
        try:
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                df = get_india_market_data(idx.yahoo_ticker, start_date, end_date)
            finally:
                sys.stderr = old_stderr
            
            if df is None or df.empty:
                print("FAIL")
                continue
            
            monthly_returns = calculate_monthly_returns(df)
            if monthly_returns.empty or len(monthly_returns) < 12:
                print(f"WARN ({len(monthly_returns)} months)")
                continue
            
            seasonality = compute_seasonality(monthly_returns, idx.label)
            if not seasonality or len(seasonality) < 12:
                print("FAIL")
                continue
            
            all_seasonality[idx.label] = seasonality
            successful += 1
            print(f"OK ({len(monthly_returns)} months)")
        except Exception as e:
            print(f"FAIL")
    
    if not all_seasonality:
        print("\n[ERROR] No usable data")
        return
    
    print(f"\nOK: Analyzed {successful} indices\n")
    
    # Aggregate results
    print(f"{'='*100}")
    print("AVERAGE MONTHLY RETURNS (%) ACROSS ALL INDICES")
    print(f"{'='*100}\n")
    
    monthly_avgs = {}
    for idx_name, seasonality in all_seasonality.items():
        for month, metrics in seasonality.items():
            if month not in monthly_avgs:
                monthly_avgs[month] = {'returns': [], 'win_rates': []}
            monthly_avgs[month]['returns'].append(metrics['avg_return'])
            monthly_avgs[month]['win_rates'].append(metrics['win_rate'])
    
    # Sort and display
    for month in range(1, 13):
        if month not in monthly_avgs:
            continue
        
        avg_ret = np.mean(monthly_avgs[month]['returns'])
        avg_wr = np.mean(monthly_avgs[month]['win_rates'])
        
        # Visual bar
        bar_len = int(abs(avg_ret) / 0.25)
        bar = ("[+]" * bar_len if avg_ret >= 0 else "[-]" * bar_len)
        symbol = "UP" if avg_ret >= 0 else "DN"
        
        print(f"  {MONTH_NAMES[month-1]:12s}: {avg_ret:7.2f}% (WR: {avg_wr:5.1f}%) {symbol} {bar}")
    
    # Best and worst months
    print(f"\n{'='*100}")
    print("BEST & WORST MONTHS")
    print(f"{'='*100}\n")
    
    sorted_months = sorted(
        [(m, np.mean(monthly_avgs[m]['returns'])) for m in monthly_avgs],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("BEST 3:")
    for month, ret in sorted_months[:3]:
        print(f"  UP  {MONTH_NAMES[month-1]:12s}: {ret:7.2f}%")
    
    print("\nWORST 3:")
    for month, ret in sorted_months[-3:]:
        print(f"  DN  {MONTH_NAMES[month-1]:12s}: {ret:7.2f}%")
    
    # Index-wise summary
    if args.top_n:
        print(f"\n{'='*100}")
        print(f"TOP {args.top_n} INDICES BY AVERAGE MONTHLY RETURN")
        print(f"{'='*100}\n")
        
        index_summary = []
        for idx_name, seasonality in all_seasonality.items():
            returns = [m['avg_return'] for m in seasonality.values()]
            win_rates = [m['win_rate'] for m in seasonality.values()]
            index_summary.append({
                'Index': idx_name,
                'Avg Return %': np.mean(returns),
                'Avg Win Rate %': np.mean(win_rates),
                'Best Month': MONTH_NAMES[sorted_months[0][0] - 1],
            })
        
        index_summary.sort(key=lambda x: x['Avg Return %'], reverse=True)
        
        for i, item in enumerate(index_summary[:args.top_n], 1):
            print(f"  {i:2d}. {item['Index']:35s}: {item['Avg Return %']:7.2f}% "
                  f"(WR: {item['Avg Win Rate %']:5.1f}%)")
    
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
