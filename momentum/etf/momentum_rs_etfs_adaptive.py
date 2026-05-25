"""
India NSE ETFs — tactical RS vs Nifty 500 (^CRSLDX); short horizons only.

Universe: universes/india.py
Filters: same as momentum_rs_etfs.py (200 EMA / 52w / ADTV for new listings).
Rank: weighted RS vs N500 on 1W/2W/1M only (3M shown but excluded from score).

vs other ETF scripts:
- momentum_etfs.py — abs returns only; no RS vs benchmark.
- momentum_rs_etfs.py — blends abs + RS on 1W–3M; longer swing hold.
- RRGIndicatorEtfs.py — visual RRG (indices + ETFs); not a ranker.

Output: final_result/etf/momentum_rs_etfs_adaptive.xlsx (Leaders + Run_Info).
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

from momentum.etf.universes.india import tickers
from utils.nse_bhavcopy import fetch_bhavcopy, fetch_nse_live_quotes, nse_symbol_from_yahoo, today_ist
from utils.output_paths import FINAL_RESULT_ETF_DIR

BENCHMARK_TICKER = "^CRSLDX"

LB_1W = 6
LB_2W = 11
LB_1M = 21
LB_3M = 63
LB_52W = 252

EMA_SPAN = 200
PROXIMITY_OF_52W_HIGH = 0.7
BENCH_EMA_FAST = 50
BENCH_EMA_SLOW = 200
ETF_EMA_9 = 9

MIN_HISTORY_SESSIONS = LB_3M
MIN_ADTV_NEW_ETF_CRORES = 2.5
ADTV_LOOKBACK = 20

TOP_N = 30
OUT_FILENAME = "momentum_rs_etfs_adaptive.xlsx"

W_RANK = (0.20, 0.40, 0.40)

RS_RANK_COLS = (
    "RS_1W_vs_N500",
    "RS_2W_vs_N500",
    "RS_1M_vs_N500",
)

ROUND_COLS = [
    "Peak_Proximity_Score",
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
    "RS_1W_vs_N500",
    "RS_2W_vs_N500",
    "RS_1M_vs_N500",
    "RS_3M_vs_N500",
    "Volatility_Score",
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
    "RS_1W_vs_N500",
    "RS_2W_vs_N500",
    "RS_1M_vs_N500",
]


def _symbol_for_excel(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace(".NS", "").replace(".BO", "")


def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _volume_series(df: pd.DataFrame) -> pd.Series:
    s = df["Volume"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _avg_adtv_crores(adj: pd.Series, vol: pd.Series) -> float:
    vol_aligned = vol.reindex(adj.index).fillna(0)
    daily_turnover = adj * vol_aligned
    n = min(ADTV_LOOKBACK, len(daily_turnover))
    if n == 0:
        return 0.0
    return float(daily_turnover.tail(n).mean()) / 1e7


def _passes_established_trend_gate(adj: pd.Series) -> bool:
    ema200 = adj.ewm(span=EMA_SPAN).mean().iloc[-1]
    high_52w = adj.iloc[-LB_52W:].max()
    last = adj.iloc[-1]
    return last >= ema200 and last >= (high_52w * PROXIMITY_OF_52W_HIGH)


def _fill_ohlcv_from_nse(df: pd.DataFrame, nse_row: dict) -> pd.DataFrame:
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
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        multi_level_index=False,
        auto_adjust=False,
        progress=False,
    )
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
    return df.dropna(subset=["Close"])


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


def _benchmark_return_1m(nifty_adj: pd.Series) -> float:
    if len(nifty_adj) < LB_1M:
        return float("nan")
    return float((nifty_adj.iloc[-1] / nifty_adj.iloc[-LB_1M] - 1) * 100)


def _weighted_excess_return(row: pd.Series) -> float:
    if any(pd.isna(row[c]) for c in RS_RANK_COLS):
        return float("nan")
    w1w, w2w, w1m = W_RANK
    return (
        w1w * row["RS_1W_vs_N500"]
        + w2w * row["RS_2W_vs_N500"]
        + w1m * row["RS_1M_vs_N500"]
    )


def _weight_label() -> str:
    w1, w2, w1m = W_RANK
    return f"1W={w1:.0%} 2W={w2:.0%} 1M={w1m:.0%} (3M excluded)"


def _collect_etf_rows(
    nifty_adj: pd.Series, start_date: datetime, end_date: datetime
) -> list[dict]:
    summary: list[dict] = []
    for sym in tickers:
        try:
            df = get_data(sym, start_date, end_date)
            if len(df) == 0:
                continue

            adj = _adj_close_series(df).dropna()
            if len(adj) < MIN_HISTORY_SESSIONS:
                continue

            if len(adj) >= LB_52W:
                if not _passes_established_trend_gate(adj):
                    continue
            else:
                vol = _volume_series(df)
                if _avg_adtv_crores(adj, vol) < MIN_ADTV_NEW_ETF_CRORES:
                    continue

            high_52w = adj.iloc[-min(LB_52W, len(adj)) :].max()
            last = adj.iloc[-1]

            close_on_adj_index = _close_series(df).reindex(adj.index).ffill().bfill()
            ema9_close = float(
                close_on_adj_index.ewm(span=ETF_EMA_9, adjust=False).mean().iloc[-1]
            )
            last_close = float(close_on_adj_index.iloc[-1])
            close_below_9ema = "Exit" if last_close < ema9_close else "Hold"

            high_ath = float(adj.max())
            ratio_52w = last / high_52w
            ratio_ath = last / high_ath if high_ath > 0 else float("nan")
            peak_proximity_score = (ratio_52w + ratio_ath) / 2.0

            return_1w = (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100
            return_2w = (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100
            return_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
            return_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100

            daily_returns = adj.pct_change()
            vol_score = daily_returns.tail(LB_1M).std() * 100

            rs_1w = rs_2w = rs_1m = rs_3m = float("nan")
            if len(nifty_adj) >= LB_3M:
                nx = nifty_adj.reindex(adj.index).ffill()
                tail = nx.iloc[-LB_3M :]
                if not tail.isna().any() and (tail > 0).all():
                    ret_n_1w = (nx.iloc[-1] / nx.iloc[-LB_1W] - 1) * 100
                    ret_n_2w = (nx.iloc[-1] / nx.iloc[-LB_2W] - 1) * 100
                    ret_n_1m = (nx.iloc[-1] / nx.iloc[-LB_1M] - 1) * 100
                    ret_n_3m = (nx.iloc[-1] / nx.iloc[-LB_3M] - 1) * 100
                    rs_1w = return_1w - ret_n_1w
                    rs_2w = return_2w - ret_n_2w
                    rs_1m = return_1m - ret_n_1m
                    rs_3m = return_3m - ret_n_3m

            summary.append(
                {
                    "Symbol": _symbol_for_excel(sym),
                    "Close": round(last_close, 2),
                    "9EMA": round(ema9_close, 2),
                    "Close_Below_9EMA": close_below_9ema,
                    "Peak_Proximity_Score": peak_proximity_score,
                    "Return_1W": return_1w,
                    "Return_2W": return_2w,
                    "Return_1M": return_1m,
                    "Return_3M": return_3m,
                    "RS_1W_vs_N500": rs_1w,
                    "RS_2W_vs_N500": rs_2w,
                    "RS_1M_vs_N500": rs_1m,
                    "RS_3M_vs_N500": rs_3m,
                    "Volatility_Score": vol_score,
                }
            )
        except Exception as e:
            print(f"Error analyzing {sym}: {e}")
    return summary


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)

    try:
        nifty_df = get_data(BENCHMARK_TICKER, start_date, end_date)
        if len(nifty_df) == 0:
            print(f"Error: No rows for benchmark {BENCHMARK_TICKER}")
            return
        nifty_adj = _adj_close_series(nifty_df).dropna()
    except Exception as e:
        print(f"Error: Benchmark {BENCHMARK_TICKER} ({e})")
        return

    market_regime = classify_ema_regime(nifty_adj, BENCH_EMA_FAST, BENCH_EMA_SLOW)
    benchmark_1m = _benchmark_return_1m(nifty_adj)

    summary = _collect_etf_rows(nifty_adj, start_date, end_date)
    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        print("No ETFs passed filters (history, liquidity, or trend gate).")
        return

    for c in ROUND_COLS:
        df_summary[c] = df_summary[c].round(2)

    df_summary["Weighted_RS_pct"] = df_summary.apply(_weighted_excess_return, axis=1)
    df_ranked = df_summary.dropna(subset=["Weighted_RS_pct"]).copy()
    if df_ranked.empty:
        print("No ETFs with valid RS vs N500 after trend filters.")
        return

    df_ranked["Rank_vs_Peak"] = (
        df_ranked["Peak_Proximity_Score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    df_out = (
        df_ranked.sort_values("Weighted_RS_pct", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    df_out["Weighted_RS_pct"] = df_out["Weighted_RS_pct"].round(2)
    df_out = df_out[FINAL_COLS]

    run_info = pd.DataFrame(
        [
            {
                "Run_Date": end_date.strftime("%Y-%m-%d"),
                "Weight_Profile": _weight_label(),
                "Rank_By": "Weighted_RS_pct vs N500 (highest first)",
                "Entry_Filters": "200 EMA + 52w proximity (or ADTV for new listings)",
                "RS_Extra_Cutoffs": "None",
                "Benchmark_Return_1M_pct": round(benchmark_1m, 2)
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
        f"({df_out.iloc[0]['Weighted_RS_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
