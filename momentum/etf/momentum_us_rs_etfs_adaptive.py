"""
US ETFs — tactical RS vs S&P 500 (^GSPC); short horizons only.

Universe: universes/us.py
Filters: above 200 EMA, within 30% of 52w high.
Rank: weighted RS vs S&P on 1W/2W/1M only (3M shown but excluded from score).

vs other ETF scripts:
- momentum_us_etfs.py — abs returns only; no RS vs benchmark.
- momentum_us_rs_etfs.py — blends abs + RS on 2W/1M/3M; longer swing hold.

Output: final_result/etf/momentum_us_rs_etfs_adaptive.xlsx (Leaders + Run_Info).
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

from momentum.etf.universes.us import tickers
from utils.output_paths import FINAL_RESULT_ETF_DIR

BENCHMARK_TICKER = "^GSPC"

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
OUT_FILENAME = "momentum_us_rs_etfs_adaptive.xlsx"

W_RANK = (0.20, 0.40, 0.40)

RS_RANK_COLS = (
    "RS_1W_vs_SP500",
    "RS_2W_vs_SP500",
    "RS_1M_vs_SP500",
)

ROUND_COLS = [
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
    "RS_1W_vs_SP500",
    "RS_2W_vs_SP500",
    "RS_1M_vs_SP500",
    "RS_3M_vs_SP500",
]

FINAL_COLS = [
    "Position",
    "Symbol",
    "Close",
    "9EMA",
    "Close_Below_9EMA",
    "Weighted_RS_pct",
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
    "RS_1W_vs_SP500",
    "RS_2W_vs_SP500",
    "RS_1M_vs_SP500",
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


def _collect_etf_rows(
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


def _benchmark_return_1m(bench_adj: pd.Series) -> float:
    if len(bench_adj) < LB_1M:
        return float("nan")
    return float((bench_adj.iloc[-1] / bench_adj.iloc[-LB_1M] - 1) * 100)


def _weighted_excess_return(row: pd.Series) -> float:
    if any(pd.isna(row[c]) for c in RS_RANK_COLS):
        return float("nan")
    w1w, w2w, w1m = W_RANK
    return (
        w1w * row["RS_1W_vs_SP500"]
        + w2w * row["RS_2W_vs_SP500"]
        + w1m * row["RS_1M_vs_SP500"]
    )


def _weight_label() -> str:
    w1, w2, w1m = W_RANK
    return f"1W={w1:.0%} 2W={w2:.0%} 1M={w1m:.0%} (3M excluded)"


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

    market_regime = classify_ema_regime(sp500_adj, BENCH_EMA_FAST, BENCH_EMA_SLOW)
    benchmark_1m = _benchmark_return_1m(sp500_adj)

    summary = _collect_etf_rows(sp500_adj, start_date, end_date)
    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        print("No tickers passed filters (history or trend gate).")
        return

    for c in ROUND_COLS:
        df_summary[c] = df_summary[c].round(1)

    df_summary["Weighted_RS_pct"] = df_summary.apply(_weighted_excess_return, axis=1)
    df_ranked = df_summary.dropna(subset=["Weighted_RS_pct"]).copy()
    if df_ranked.empty:
        print("No ETFs with valid RS vs S&P 500 after trend filters.")
        return

    df_out = (
        df_ranked.sort_values("Weighted_RS_pct", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    df_out["Weighted_RS_pct"] = df_out["Weighted_RS_pct"].round(1)
    df_out = df_out[FINAL_COLS]

    run_info = pd.DataFrame(
        [
            {
                "Run_Date": end_date.strftime("%Y-%m-%d"),
                "Weight_Profile": _weight_label(),
                "Rank_By": "Weighted_RS_pct vs S&P 500 (highest first)",
                "Entry_Filters": "200 EMA + 52w proximity",
                "RS_Extra_Cutoffs": "None",
                "Benchmark_Return_1M_pct": round(benchmark_1m, 1)
                if not pd.isna(benchmark_1m)
                else None,
                "Market_Regime": market_regime,
                "ETFs_Ranked": len(df_ranked),
                "Rows_Written": len(df_out),
            }
        ]
    )

    FINAL_RESULT_ETF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_ETF_DIR / OUT_FILENAME
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Leaders", index=False)
        run_info.to_excel(writer, sheet_name="Run_Info", index=False)

    print(f"Success: Wrote {len(df_out)} rows to {out_path}")
    print(
        f"Rank=Weighted_RS_pct  Weights={_weight_label()}  "
        f"Ranked={len(df_ranked)}  Top={df_out.iloc[0]['Symbol']} "
        f"({df_out.iloc[0]['Weighted_RS_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()
