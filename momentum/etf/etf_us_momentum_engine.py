"""US ETF momentum rankers — computation engine (no Excel I/O).

Mirrors logic in momentum_us_etfs.py, momentum_us_rs_etfs.py, and
momentum_us_rs_etfs_adaptive.py for on-screen display and backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from momentum.etf.universes import us_universe

BENCHMARK_TICKER = "^GSPC"

LB_1W = 6
LB_2W = 11
LB_1M = 21
LB_3M = 64
LB_52W = 252

EMA_SPAN = 200
PROXIMITY_OF_52W_HIGH = 0.7
BENCH_EMA_FAST = 50
BENCH_EMA_SLOW = 200
ETF_EMA_9 = 9

MIN_HISTORY_SESSIONS = LB_3M
TOP_N = 30

W_ABS_LONG = (0.2, 0.4, 0.4)  # 2W, 1M, 3M
W_RS_ADAPTIVE = (0.20, 0.40, 0.40)

RETURN_SUFFIXES_LONG = ("2W", "1M", "3M")
RS_RANK_COLS = ("RS_1W_vs_SP500", "RS_2W_vs_SP500", "RS_1M_vs_SP500")


@dataclass
class UsEtfMomentumSnapshot:
    run_date: str
    market_regime: str
    benchmark_return_1m_pct: float | None
    abs_momentum: pd.DataFrame
    rs_blended: pd.DataFrame
    rs_adaptive: pd.DataFrame
    run_info: dict[str, str | int | float | None] = field(default_factory=dict)
    etfs_ranked_adaptive: int = 0


def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _etf_name(symbol: str) -> str:
    us_universe.ensure_loaded()
    return us_universe.ETF_LABELS.get(symbol, symbol)


def _passes_trend_gate(adj: pd.Series) -> bool:
    ema200 = adj.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1]
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


def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start_date,
        end=end_date,
        multi_level_index=False,
        auto_adjust=False,
        progress=False,
    )


def _benchmark_return_1m(bench_adj: pd.Series) -> float:
    if len(bench_adj) < LB_1M:
        return float("nan")
    return float((bench_adj.iloc[-1] / bench_adj.iloc[-LB_1M] - 1) * 100)


def _weight_label_adaptive() -> str:
    w1, w2, w1m = W_RS_ADAPTIVE
    return f"1W={w1:.0%} 2W={w2:.0%} 1M={w1m:.0%} (3M excluded)"


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


def _truncate_ohlcv(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.loc[:as_of]
    if out.empty:
        return out
    if "Close" in out.columns:
        return out.dropna(subset=["Close"])
    return out


def _resolve_as_of(as_of_date: str | None) -> pd.Timestamp:
    if as_of_date and str(as_of_date).strip():
        as_of = pd.Timestamp(str(as_of_date).strip()).normalize()
        if as_of > pd.Timestamp(datetime.today().date()):
            raise ValueError(f"As-of date cannot be in the future ({as_of_date}).")
        return as_of
    return pd.Timestamp(datetime.today().date())


def _collect_rs_rows(
    bench_adj: pd.Series,
    start_date,
    end_date,
    *,
    as_of: pd.Timestamp,
) -> list[dict]:
    summary: list[dict] = []
    for ticker in us_universe.TICKERS:
        try:
            df = get_data(ticker, start_date, end_date)
            df = _truncate_ohlcv(df, as_of)
            if df is None or len(df) == 0:
                continue

            adj = _adj_close_series(df).dropna()
            if len(adj) < MIN_HISTORY_SESSIONS:
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
                    "Name": _etf_name(ticker),
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
        except Exception:
            continue
    return summary


def _compute_abs_momentum(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary: list[dict] = []
    for ticker, df in data.items():
        try:
            adj = _adj_close_series(df).dropna()
            n = len(adj)
            if n < LB_3M:
                continue
            if not _passes_trend_gate(adj):
                continue

            close = _close_series(df).reindex(adj.index).ffill().bfill()
            ema9_close = float(close.ewm(span=ETF_EMA_9, adjust=False).mean().iloc[-1])
            last_close = float(close.iloc[-1])
            close_below_9ema = "Exit" if last_close < ema9_close else "Hold"

            summary.append(
                {
                    "Symbol": ticker,
                    "Name": _etf_name(ticker),
                    "Close": round(last_close, 2),
                    "9EMA": round(ema9_close, 2),
                    "Close_Below_9EMA": close_below_9ema,
                    "Return_1W": (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100,
                    "Return_2W": (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100,
                    "Return_1M": (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100,
                    "Return_3M": (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100,
                }
            )
        except Exception:
            continue

    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        return df_summary

    for col in ("Return_1W", "Return_2W", "Return_1M", "Return_3M"):
        df_summary[col] = df_summary[col].round(1)

    w2w, w1m, w3m = W_ABS_LONG
    df_summary["Rank_2W"] = df_summary["Return_2W"].rank(ascending=False)
    df_summary["Rank_1M"] = df_summary["Return_1M"].rank(ascending=False)
    df_summary["Rank_3M"] = df_summary["Return_3M"].rank(ascending=False)
    df_summary["Final_Rank"] = (
        w2w * df_summary["Rank_2W"]
        + w1m * df_summary["Rank_1M"]
        + w3m * df_summary["Rank_3M"]
    )

    df_out = df_summary.sort_values("Final_Rank").head(TOP_N).copy()
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    cols = [
        "Position",
        "Symbol",
        "Name",
        "Close",
        "9EMA",
        "Close_Below_9EMA",
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
    ]
    return df_out[cols].reset_index(drop=True)


def _compute_rs_blended(df_summary: pd.DataFrame) -> pd.DataFrame:
    if df_summary.empty:
        return df_summary

    round_cols = (
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
        "RS_1W_vs_SP500",
        "RS_2W_vs_SP500",
        "RS_1M_vs_SP500",
        "RS_3M_vs_SP500",
    )
    for c in round_cols:
        df_summary[c] = df_summary[c].round(1)

    w2w, w1m, w3m = W_ABS_LONG
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
        w2w * df_summary["Rank_2W"]
        + w1m * df_summary["Rank_1M"]
        + w3m * df_summary["Rank_3M"]
    )
    rs_rank_score = (
        w2w * df_summary["Rank_RS_2W"]
        + w1m * df_summary["Rank_RS_1M"]
        + w3m * df_summary["Rank_RS_3M"]
    )
    df_summary["Final_Rank"] = (abs_rank_score + rs_rank_score) / 2

    df_out = df_summary.sort_values("Final_Rank").head(TOP_N).reset_index(drop=True)
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    cols = [
        "Position",
        "Symbol",
        "Name",
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
    return df_out[cols]


def _weighted_excess_return(row: pd.Series) -> float:
    if any(pd.isna(row[c]) for c in RS_RANK_COLS):
        return float("nan")
    w1w, w2w, w1m = W_RS_ADAPTIVE
    return (
        w1w * row["RS_1W_vs_SP500"]
        + w2w * row["RS_2W_vs_SP500"]
        + w1m * row["RS_1M_vs_SP500"]
    )


def _compute_rs_adaptive(df_summary: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df_summary.empty:
        return df_summary, 0

    round_cols = (
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
        "RS_1W_vs_SP500",
        "RS_2W_vs_SP500",
        "RS_1M_vs_SP500",
        "RS_3M_vs_SP500",
    )
    for c in round_cols:
        df_summary[c] = df_summary[c].round(1)

    df_summary["Weighted_RS_pct"] = df_summary.apply(_weighted_excess_return, axis=1)
    df_ranked = df_summary.dropna(subset=["Weighted_RS_pct"]).copy()
    if df_ranked.empty:
        return pd.DataFrame(), 0

    df_out = (
        df_ranked.sort_values("Weighted_RS_pct", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    df_out["Weighted_RS_pct"] = df_out["Weighted_RS_pct"].round(2)

    cols = [
        "Position",
        "Symbol",
        "Name",
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
    return df_out[cols], len(df_ranked)


def fetch_us_etf_momentum_snapshot(as_of_date: str | None = None) -> UsEtfMomentumSnapshot:
    """Fetch data and compute all three US ranker outputs as of ``as_of_date`` (YYYY-MM-DD)."""
    us_universe.ensure_loaded()
    as_of = _resolve_as_of(as_of_date)
    end_date = as_of + pd.Timedelta(days=1)
    start_date = as_of - pd.Timedelta(days=365 * 2)
    run_date = as_of.strftime("%Y-%m-%d")

    abs_data: dict[str, pd.DataFrame] = {}
    for ticker in us_universe.TICKERS:
        try:
            stock_data = get_data(ticker, start_date, end_date)
            stock_data = _truncate_ohlcv(stock_data, as_of)
            if stock_data is not None and len(stock_data) > 0:
                abs_data[ticker] = stock_data
        except Exception:
            continue

    abs_df = _compute_abs_momentum(abs_data)

    sp500_df = get_data(BENCHMARK_TICKER, start_date, end_date)
    sp500_df = _truncate_ohlcv(sp500_df, as_of)
    if sp500_df is None or len(sp500_df) == 0:
        raise RuntimeError(f"No benchmark data for {BENCHMARK_TICKER} on or before {run_date}")

    sp500_adj = _adj_close_series(sp500_df).dropna()
    if sp500_adj.empty:
        raise RuntimeError(f"No benchmark prices on or before {run_date}")
    market_regime = classify_ema_regime(sp500_adj, BENCH_EMA_FAST, BENCH_EMA_SLOW)
    benchmark_1m = _benchmark_return_1m(sp500_adj)

    rs_rows = _collect_rs_rows(sp500_adj, start_date, end_date, as_of=as_of)
    rs_summary = pd.DataFrame(rs_rows)

    rs_blended = _compute_rs_blended(rs_summary.copy())
    rs_adaptive, etfs_ranked = _compute_rs_adaptive(rs_summary.copy())

    run_info: dict[str, str | int | float | None] = {
        "As_Of_Date": run_date,
        "Run_Date": run_date,
        "Weight_Profile": _weight_label_adaptive(),
        "Rank_By": "Weighted_RS_pct vs S&P 500 (highest first)",
        "Entry_Filters": "200 EMA + 52w proximity",
        "RS_Extra_Cutoffs": "None",
        "Benchmark_Return_1M_pct": round(benchmark_1m, 2)
        if not pd.isna(benchmark_1m)
        else None,
        "Market_Regime": market_regime,
        "ETFs_Ranked": etfs_ranked,
        "Rows_Written": len(rs_adaptive),
        "Abs_Momentum_Rows": len(abs_df),
        "RS_Blended_Rows": len(rs_blended),
    }

    return UsEtfMomentumSnapshot(
        run_date=run_date,
        market_regime=market_regime,
        benchmark_return_1m_pct=run_info["Benchmark_Return_1M_pct"],  # type: ignore[arg-type]
        abs_momentum=abs_df,
        rs_blended=rs_blended,
        rs_adaptive=rs_adaptive,
        run_info=run_info,
        etfs_ranked_adaptive=etfs_ranked,
    )
