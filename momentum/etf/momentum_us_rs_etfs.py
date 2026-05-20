"""
US ETF relative strength vs S&P 500 (^GSPC): excess returns vs index over multiple horizons.

Trend filters: price above 200 EMA and within 30% of 52-week high (same universe as momentum_us_etfs.py).
Final_Rank blends absolute return ranks (2W / 1M / 3M) with RS ranks (equal weights).

RS_* columns are ETF minus ^GSPC over the same horizons (Adj Close, date-aligned).

Close_Below_9EMA: Exit if last Close is below 9 EMA of Close, else Hold (chart parity; returns use Adj Close).

Companion: momentum_us_rs_etfs_adaptive.py ranks by weighted 1W/2W/1M RS only (3M omitted).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_ETF_DIR

BENCHMARK_TICKER = "^GSPC"  # S&P 500 index (RS anchor; separate from ETF universe)

LB_1W = 6
LB_2W = 11
LB_1M = 21
LB_3M = 64
LB_52W = 252
PROXIMITY_OF_52W_HIGH = 0.7
EMA_SPAN = 200

BENCH_EMA_FAST = 50
BENCH_EMA_SLOW = 200

ETF_EMA_9 = 9
MIN_HISTORY_SESSIONS = LB_3M
TOP_N = 10
OUT_FILENAME = "momentum_us_rs_etfs.xlsx"

# Universe: US ETFs/indices on Yahoo (same as momentum_us_etfs.py)
tickers = [
    "XLC", "QQQ", "XLK", "SPY", "IWM", "VEA", "EEM", "MCHI", "CQQQ", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLY", "XLB", "SOXX", "SKYY", "ARKK", "ICLN", "TAN", "GLD", "GDX", "USO", "VNQ", "EWJ", "EWZ", "EWT", "EWY", "EWA", "EWG", "EWC", "EIDO", "FM", "KSA", "ARGT", "TUR", "THD", "GREK", "UUP", "MTUM", "QUAL", "USMV", "DXYZ", "DBC", "CPER", "FXI", "EFA", "PICK", "XME", "URA", "DBA", "UNG", "WEAT", "CORN", "FXE", "FXY", "FXB", "BITO"
]


def get_data(ticker: str, start_date, end_date):
    return yf.download(
        ticker,
        start=start_date,
        end=end_date,
        multi_level_index=False,
        auto_adjust=False,
    )


def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _passes_trend_gate(adj: pd.Series) -> bool:
    ema200 = adj.ewm(span=EMA_SPAN).mean().iloc[-1]
    high_52w = adj.iloc[-min(LB_52W, len(adj)) :].max()
    last = adj.iloc[-1]
    return last >= ema200 and last >= (high_52w * PROXIMITY_OF_52W_HIGH)


def classify_ema_regime(close: pd.Series, fast_span: int, slow_span: int) -> str:
    if fast_span >= slow_span:
        raise ValueError("fast_span must be less than slow_span")
    if len(close) < slow_span:
        return "Unknown"
    last = float(close.iloc[-1])
    e_fast = float(close.ewm(span=fast_span, adjust=False).mean().iloc[-1])
    e_slow = float(close.ewm(span=slow_span, adjust=False).mean().iloc[-1])
    if last >= e_slow and last >= e_fast:
        return "Trend_Up"
    if last < e_slow and last < e_fast:
        return "Trend_Down"
    if last >= e_fast:
        return "Mixed_Above50"
    return "Mixed_Below50"


def _rs_vs_benchmark(
    adj: pd.Series, bench_adj: pd.Series
) -> tuple[float, float, float, float, float, float, float, float]:
    return_1w = (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100
    return_2w = (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100
    return_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
    return_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100

    rs_1w = rs_2w = rs_1m = rs_3m = float("nan")
    if len(bench_adj) >= LB_3M:
        bx = bench_adj.reindex(adj.index).ffill()
        tail = bx.iloc[-LB_3M:]
        if not tail.isna().any() and (tail > 0).all():
            ret_b_1w = (bx.iloc[-1] / bx.iloc[-LB_1W] - 1) * 100
            ret_b_2w = (bx.iloc[-1] / bx.iloc[-LB_2W] - 1) * 100
            ret_b_1m = (bx.iloc[-1] / bx.iloc[-LB_1M] - 1) * 100
            ret_b_3m = (bx.iloc[-1] / bx.iloc[-LB_3M] - 1) * 100
            rs_1w = return_1w - ret_b_1w
            rs_2w = return_2w - ret_b_2w
            rs_1m = return_1m - ret_b_1m
            rs_3m = return_3m - ret_b_3m
    return return_1w, return_2w, return_1m, return_3m, rs_1w, rs_2w, rs_1m, rs_3m


def collect_us_etf_rows(
    bench_adj: pd.Series, start_date: datetime, end_date: datetime
) -> list[dict]:
    summary: list[dict] = []
    for ticker in tickers:
        try:
            df = get_data(ticker, start_date, end_date)
            if len(df) == 0:
                continue

            adj = _adj_close_series(df).dropna()
            if len(adj) < MIN_HISTORY_SESSIONS:
                print(
                    f"Skip {ticker}: insufficient history "
                    f"({len(adj)} rows, need >= {MIN_HISTORY_SESSIONS})."
                )
                continue
            if not _passes_trend_gate(adj):
                continue

            close = _close_series(df).reindex(adj.index).ffill().bfill()
            ema9_close = float(close.ewm(span=ETF_EMA_9, adjust=False).mean().iloc[-1])
            last_close = float(close.iloc[-1])
            close_below_9ema = "Exit" if last_close < ema9_close else "Hold"

            return_1w, return_2w, return_1m, return_3m, rs_1w, rs_2w, rs_1m, rs_3m = (
                _rs_vs_benchmark(adj, bench_adj)
            )

            summary.append(
                {
                    "Symbol": ticker,
                    "Close": round(last_close, 2),
                    "9EMA": round(ema9_close, 2),
                    "Close_Below_9EMA": close_below_9ema,
                    "Return_1W": return_1w,
                    "Return_2W": return_2w,
                    "Return_1M": return_1m,
                    "Return_3M": return_3m,
                    "RS_1W_vs_SP500": rs_1w,
                    "RS_2W_vs_SP500": rs_2w,
                    "RS_1M_vs_SP500": rs_1m,
                    "RS_3M_vs_SP500": rs_3m,
                }
            )
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
    return summary


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)

    try:
        sp500_df = get_data(BENCHMARK_TICKER, start_date, end_date)
        if len(sp500_df) == 0:
            print(f"Error: No rows for benchmark {BENCHMARK_TICKER}")
            return
        sp500_adj = _adj_close_series(sp500_df).dropna()
    except Exception as e:
        print(f"Error: Benchmark {BENCHMARK_TICKER} ({e})")
        return

    summary = collect_us_etf_rows(sp500_adj, start_date, end_date)
    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        print("No tickers passed filters; no Excel file written.")
        return

    round_cols = (
        "Return_1W", "Return_2W", "Return_1M", "Return_3M",
        "RS_1W_vs_SP500", "RS_2W_vs_SP500", "RS_1M_vs_SP500", "RS_3M_vs_SP500",
    )
    for col in round_cols:
        df_summary[col] = df_summary[col].round(1)

    df_summary["Rank_2W"] = df_summary["Return_2W"].rank(ascending=False)
    df_summary["Rank_1M"] = df_summary["Return_1M"].rank(ascending=False)
    df_summary["Rank_3M"] = df_summary["Return_3M"].rank(ascending=False)
    df_summary["Rank_RS_2W"] = df_summary["RS_2W_vs_SP500"].rank(
        ascending=False, na_option="bottom"
    )
    df_summary["Rank_RS_1M"] = df_summary["RS_1M_vs_SP500"].rank(
        ascending=False, na_option="bottom"
    )
    df_summary["Rank_RS_3M"] = df_summary["RS_3M_vs_SP500"].rank(
        ascending=False, na_option="bottom"
    )

    abs_rank_score = (
        0.2 * df_summary["Rank_2W"]
        + 0.4 * df_summary["Rank_1M"]
        + 0.4 * df_summary["Rank_3M"]
    )
    rs_rank_score = (
        0.2 * df_summary["Rank_RS_2W"]
        + 0.4 * df_summary["Rank_RS_1M"]
        + 0.4 * df_summary["Rank_RS_3M"]
    )
    df_summary["Final_Rank"] = (abs_rank_score + rs_rank_score) / 2

    df_out = df_summary.sort_values("Final_Rank").head(TOP_N).reset_index(drop=True)
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))

    cols = [
        "Position",
        "Symbol",
        "Close",
        "9EMA",
        "Close_Below_9EMA",
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
        "RS_1W_vs_SP500",
        "RS_2W_vs_SP500",
        "RS_1M_vs_SP500",
        "RS_3M_vs_SP500",
    ]
    df_out = df_out[cols]

    FINAL_RESULT_ETF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_ETF_DIR / OUT_FILENAME
    try:
        df_out.to_excel(out_path, index=False, engine="openpyxl")
    except ImportError:
        print("Missing dependency: pip install openpyxl")
        raise
    print(f"Wrote {len(df_out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
