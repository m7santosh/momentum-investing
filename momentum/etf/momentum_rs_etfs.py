"""
ETF relative strength vs Nifty 500 (^CRSLDX): excess returns vs index over multiple horizons.

Trend filters:
1. Price must be above the 200-day EMA.
2. Price must be within 30% of its 52-week high (last close >= 70% of trailing ~252-session high).

Ranking (swing / all-weather blend):
- Horizons: 1W, 2W, 1M, 3M (trading sessions). Weights favor 3M; 6M omitted for ETFs (less idiosyncratic drift vs single stocks; 200 EMA / 52w already anchor slower trend).
- Abs_Momentum_Rank / Relative_Strength_Rank from weighted average of period ranks, then reranked.
- Blended_Rank: average of the two. Lower is better.

Market_Regime: ^CRSLDX vs 50 / 200 EMA (Trend_Up / Trend_Down / Mixed_Above50 / Mixed_Below50 / Unknown).

Close_Below_9EMA: per ETF, Yes if last **regular Close** is below the 9 EMA of **Close** (matches typical broker / TradingView “Close” charts). Trend gate and Return_* still use **Adj Close** (total return).

How to read Volatility_Score (informational):
- It is stdev of daily % returns over the last LB_1M sessions, ×100; higher = choppier last month.
- Use it only vs other names on the same output: lower = smoother when you already like the rank/RS story.
- Tiny differences (e.g. 1.1 vs 1.2) are noise; wide gaps (e.g. 0.9 vs 2.5) matter more.
- Broad index-style ETFs often ~0.5–2.0; thematic/commodity sleeves can be higher without being "bad".
- Tie-break or sizing hint, not a primary buy/sell rule (says nothing about gaps or tail risk).

How to read Rank_vs_Peak (informational):
- Built from adj close vs (1) trailing ~52-week high and (2) max adj close in the download window (~2y); score = average of the two ratios; then ranked across the filtered universe.
- Rank 1 = closest to those peaks (strongest “printing highs” posture in this data); larger rank = farther from peaks.
- Compare only within the same run: same date, same universe after trend filters.
- “ATH” here is max over downloaded history, not necessarily since listing; 52w uses LB_52W sessions.
- Not a buy rule on its own; use with Blended_Rank, RS, and trend filters.

Excel: Return_1W / Return_2W / Return_1M / Return_3M are **total return %** from **Adj Close** over a fixed number of **trading sessions** (see LB_1W … LB_3M): anchor is ``adj.iloc[-LB_*]``, end is last bar. They are not calendar “this week”, and they will **differ** from a broker chart that uses **unadjusted Close** or a different bar count — gaps of a few percent are normal after distributions or vs calendar windows.
RS_* columns are ETF minus benchmark over the **same** horizons, both from **Adj Close** on date-aligned rows.

Yahoo often leaves the latest calendar row as NaN for .NS symbols until the session prints (or timezone mismatch). Returns use the last **non-NaN** Adj Close bars so horizons are not applied to a phantom "today" row. A Return of 0.00 means the price was flat over that window (after rounding).
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

from utils.output_paths import FINAL_RESULT_ETF_DIR
from utils.nse_bhavcopy import fetch_bhavcopy, fetch_nse_live_quotes, nse_symbol_from_yahoo, today_ist

# --- Configuration ---
BENCHMARK_TICKER = "^CRSLDX"

# Lookback offsets: iloc[-LB_*] is the **start** bar; horizon spans (LB_* - 1) sessions to last bar
# (e.g. LB_1W=6 → 5 sessions from anchor to last, ~one trading week).
LB_1W = 6
LB_2W = 11
LB_1M = 21
LB_3M = 63

# Ranking weights (sum = 1): short windows + dominant 3M (no 6M)
W_1W, W_2W, W_1M, W_3M = 0.10, 0.10, 0.25, 0.55

EMA_SPAN = 200
LB_52W = 252
PROXIMITY_OF_52W_HIGH = 0.7

# Benchmark regime (^CRSLDX): slow / fast EMA spans
BENCH_EMA_FAST = 50
BENCH_EMA_SLOW = 200

# Short EMA for per-ETF “close below 9?” flag (regular Close, not Adj Close)
ETF_EMA_9 = 9

TOP_N = 20
OUT_FILENAME = "momentum_rs_etfs.xlsx"

RETURN_SUFFIXES = ("1W", "2W", "1M", "3M")

# --- Ticker universe ---
tickers = [
    "ALPHA.NS", "AUTOBEES.NS", "BANKBEES.NS", "CONSUMBEES.NS", "CPSEETF.NS",
    "MOENERGY.NS", "FMCGIETF.NS", "GOLDBEES.NS", "GROWWPOWER.NS", "GROWWRAIL.NS",
    "HEALTHIETF.NS", "HNGSNGBEES.NS", "ICICIB22.NS", "INFRABEES.NS", "ITBEES.NS",
    "LIQUIDCASE.NS", "MAHKTECH.NS", "METALIETF.NS", "MIDCAPETF.NS", "MOCAPITAL.NS",
    "MODEFENCE.NS", "MON100.NS", "MOREALTY.NS", "MOTOUR.NS", "MOVALUE.NS",
    "NEXT50IETF.NS", "NIFTYBEES.NS", "OILIETF.NS", "PHARMABEES.NS", "PSUBNKBEES.NS",
    "PVTBANIETF.NS", "MOMIDMTM.NS", "SILVERBEES.NS", "SMALLCAP.NS",
]


def _symbol_for_excel(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace(".NS", "").replace(".BO", "")


def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"]
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
    df = df.dropna(subset=["Close"])
    return df


def classify_ema_regime(close: pd.Series, fast_span: int, slow_span: int) -> str:
    """Benchmark only: last vs EMA(fast) and EMA(slow); fast_span < slow_span. (50/200) uses legacy Mixed_*50 names."""
    # Quadrant rule on price vs two EMAs (same structure if reused elsewhere with other spans).
    if fast_span >= slow_span:
        raise ValueError("fast_span must be less than slow_span")
    if len(close) < slow_span:
        return "Unknown"
    last = float(close.iloc[-1])
    e_fast = float(close.ewm(span=fast_span, adjust=False).mean().iloc[-1])
    e_slow = float(close.ewm(span=slow_span, adjust=False).mean().iloc[-1])
    mixed_above = "Mixed_Above50"
    mixed_below = "Mixed_Below50"
    if last >= e_slow and last >= e_fast:
        return "Trend_Up"
    if last < e_slow and last < e_fast:
        return "Trend_Down"
    if last >= e_fast:
        return mixed_above
    return mixed_below


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)  # ~2y history for 52w / EMA / RS

    # --- Benchmark: RS anchor + Market_Regime (one series for whole run) ---
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

    # --- Per ETF: fetch, trend gate, returns/RS/vol, append row ---
    summary = []
    for sym in tickers:
        try:
            df = get_data(sym, start_date, end_date)
            if len(df) == 0:
                continue

            adj = _adj_close_series(df).dropna()
            if len(adj) < LB_52W:
                continue

            # Trend gate: 200 EMA + within PROXIMITY_OF_52W_HIGH of trailing 52w high
            ema200 = adj.ewm(span=EMA_SPAN).mean().iloc[-1]
            high_52w = adj.iloc[-min(LB_52W, len(adj)):].max()
            last = adj.iloc[-1]
            if last < ema200 or last < (high_52w * PROXIMITY_OF_52W_HIGH):
                continue

            # Short swing flag: 9 EMA of **Close** vs last **Close** (chart/broker parity; not Adj Close).
            close_on_adj_index = _close_series(df).reindex(adj.index).ffill().bfill()
            ema9_close = float(close_on_adj_index.ewm(span=ETF_EMA_9, adjust=False).mean().iloc[-1])
            last_close = float(close_on_adj_index.iloc[-1])
            close_below_9ema = "Yes" if last_close < ema9_close else "No"

            # Peak proximity: avg(last/52w_high, last/max_in_window) → Rank_vs_Peak later
            high_ath = float(adj.max())
            ratio_52w = last / high_52w
            ratio_ath = last / high_ath if high_ath > 0 else float("nan")
            peak_proximity_score = (ratio_52w + ratio_ath) / 2.0

            # Trailing total returns (%), session offsets LB_* (see config)
            return_1w = (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100
            return_2w = (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100
            return_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
            return_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100

            daily_returns = adj.pct_change()
            # Volatility_Score — see module docstring "How to read Volatility_Score".
            vol_score = daily_returns.tail(LB_1M).std() * 100

            # RS = ETF return minus same-horizon benchmark return (aligned calendar via reindex/ffill)
            rs_1w = rs_2w = rs_1m = rs_3m = float("nan")
            if len(nifty_adj) >= LB_3M:
                nx = nifty_adj.reindex(adj.index).ffill()
                tail = nx.iloc[-LB_3M:]
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

    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        print("No ETFs passed the trend filters.")
        return

    # Round numeric columns before ranks (stable ordering / Excel display)
    round_cols = [
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
    for c in round_cols:
        df_summary[c] = df_summary[c].round(2)

    # Rank_vs_Peak — see module docstring "How to read Rank_vs_Peak".
    df_summary["Rank_vs_Peak"] = df_summary["Peak_Proximity_Score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)

    # --- Ranking: per-horizon ranks → weighted Abs_Score / RS_Score → rerank → Blended_Rank ---
    for suf in RETURN_SUFFIXES:
        df_summary[f"Rank_{suf}"] = df_summary[f"Return_{suf}"].rank(ascending=False)

    for suf in RETURN_SUFFIXES:
        df_summary[f"Rank_RS_{suf}"] = df_summary[f"RS_{suf}_vs_N500"].rank(
            ascending=False, na_option="bottom"
        )

    # Weighted average of horizon ranks (W_* in config); lower composite → better after rerank
    df_summary["Abs_Score"] = (
        W_1W * df_summary["Rank_1W"]
        + W_2W * df_summary["Rank_2W"]
        + W_1M * df_summary["Rank_1M"]
        + W_3M * df_summary["Rank_3M"]
    )
    df_summary["RS_Score"] = (
        W_1W * df_summary["Rank_RS_1W"]
        + W_2W * df_summary["Rank_RS_2W"]
        + W_1M * df_summary["Rank_RS_1M"]
        + W_3M * df_summary["Rank_RS_3M"]
    )

    df_summary["Abs_Momentum_Rank"] = df_summary["Abs_Score"].rank(ascending=True)
    df_summary["Relative_Strength_Rank"] = df_summary["RS_Score"].rank(ascending=True)
    # Mean of absolute-momentum rank and RS rank; lower Blended_Rank = better
    df_summary["Blended_Rank"] = (
        df_summary["Abs_Momentum_Rank"] + df_summary["Relative_Strength_Rank"]
    ) / 2
    df_summary["Market_Regime"] = market_regime

    # --- Excel: top TOP_N by Blended_Rank; column order for scan (regime → rank → vol → returns/RS) ---
    df_out = df_summary.sort_values("Blended_Rank").head(TOP_N).reset_index(drop=True)
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))

    final_cols = [
        "Position",
        "Symbol",
        "Close",
        "9EMA",
        # "Market_Regime",
        "Close_Below_9EMA",
        # "Blended_Rank",
        "Rank_vs_Peak",
        "Volatility_Score",
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
        # "RS_1W_vs_N500",
        # "RS_2W_vs_N500",
        # "RS_1M_vs_N500",
        # "RS_3M_vs_N500"
    ]
    df_out = df_out[final_cols]

    # df_out["Blended_Rank"] = df_out["Blended_Rank"].round(2)

    FINAL_RESULT_ETF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_ETF_DIR / OUT_FILENAME
    # openpyxl required for .xlsx write on most pandas installs
    df_out.to_excel(out_path, index=False, engine="openpyxl")

    print(f"Success: Wrote {len(df_out)} rows to {out_path}")
    print(f"Market_Regime={market_regime}")


if __name__ == "__main__":
    main()
