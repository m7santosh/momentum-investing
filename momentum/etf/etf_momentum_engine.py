"""India NSE ETF momentum rankers — computation engine (no Excel I/O).

Mirrors logic in momentum_etfs.py, momentum_rs_etfs.py, and
momentum_rs_etfs_adaptive.py for on-screen display and later validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from momentum.etf.ema9_metrics import compute_ema9_metrics
from momentum.etf.universes.india import tickers
from utils.india_market_data import (
    MIN_BARS_NS_BHAVCOPY_FIRST,
    format_range_label,
    get_data,
    get_india_market_data_run_stats,
    prepare_india_market_data_range,
    summarize_etf_history_gaps,
)
from utils.nse_bhavcopy import today_ist

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

MIN_HISTORY_SESSIONS = LB_3M
MIN_ADTV_NEW_ETF_CRORES = 2.5
ADTV_LOOKBACK = 20

W_RS_BLEND = (0.10, 0.10, 0.25, 0.55)
W_RS_ADAPTIVE = (0.20, 0.40, 0.40)

RETURN_SUFFIXES = ("1W", "2W", "1M", "3M")
RS_RANK_COLS = ("RS_1W_vs_N500", "RS_2W_vs_N500", "RS_1M_vs_N500")


@dataclass
class EtfMomentumSnapshot:
    run_date: str
    market_regime: str
    benchmark_return_1m_pct: float | None
    abs_momentum: pd.DataFrame
    rs_blended: pd.DataFrame
    rs_adaptive: pd.DataFrame
    run_info: dict[str, str | int | float | None] = field(default_factory=dict)
    etfs_ranked_adaptive: int = 0


def _symbol_display(yahoo_ticker: str) -> str:
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


def _weight_label_adaptive() -> str:
    w1, w2, w1m = W_RS_ADAPTIVE
    return f"1W={w1:.0%} 2W={w2:.0%} 1M={w1m:.0%} (3M excluded)"


def _compute_abs_momentum(
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    summary: list[dict] = []
    for ticker, df in data.items():
        try:
            adj = df["Adj Close"]
            if isinstance(adj, pd.DataFrame):
                adj = adj.iloc[:, 0]
            adj = adj.squeeze()
            n = len(adj)
            if n < LB_1M:
                continue

            df = df.copy()
            df["EMA200"] = adj.ewm(span=EMA_SPAN).mean()

            high_52_week = adj.iloc[-min(LB_52W, n) :].max()
            within_30_pct_high = adj.iloc[-1] >= high_52_week * PROXIMITY_OF_52W_HIGH

            if adj.iloc[-1] >= df["EMA200"].iloc[-1] and within_30_pct_high:
                close_on_adj = _close_series(df).reindex(adj.index).ffill().bfill()
                ema9 = compute_ema9_metrics(close_on_adj)

                return_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
                return_1w = (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100
                return_2w = (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100
                summary.append(
                    {
                        "Symbol": _symbol_display(ticker),
                        "Close": ema9["last_close"],
                        "9EMA": ema9["ema9_close"],
                        "Close_Below_9EMA": ema9["close_below_9ema"],
                        "Above_9EMA_Since": ema9["above_9ema_since"],
                        "Pct_Above_9EMA": ema9["pct_since_cross"],
                        "Return_1M": return_1m,
                        "Return_2W": return_2w,
                        "Return_1W": return_1w,
                    }
                )
        except Exception:
            continue

    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        return df_summary

    df_summary["Return_1M"] = df_summary["Return_1M"].round(1)
    df_summary["Return_2W"] = df_summary["Return_2W"].round(1)
    df_summary["Return_1W"] = df_summary["Return_1W"].round(1)

    df_summary["Rank_1M"] = df_summary["Return_1M"].rank(ascending=False)
    df_summary["Rank_2W"] = df_summary["Return_2W"].rank(ascending=False)
    df_summary["Rank_1W"] = df_summary["Return_1W"].rank(ascending=False)
    df_summary["Final_Rank"] = (
        0.4 * df_summary["Rank_1W"]
        + 0.4 * df_summary["Rank_2W"]
        + 0.2 * df_summary["Rank_1M"]
    )

    df_out = df_summary.sort_values("Final_Rank").copy()
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    final_cols = [
        "Position",
        "Symbol",
        "Close",
        "9EMA",
        "Close_Below_9EMA",
        "Above_9EMA_Since",
        "Pct_Above_9EMA",
        "Return_1W",
        "Return_2W",
        "Return_1M",
    ]
    return df_out[final_cols].reset_index(drop=True)


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
        if as_of > pd.Timestamp(today_ist()):
            raise ValueError(f"As-of date cannot be in the future ({as_of_date}).")
        return as_of
    return pd.Timestamp(today_ist())


def _collect_rs_rows(
    nifty_adj: pd.Series,
    start_date,
    end_date,
    *,
    as_of: pd.Timestamp,
) -> list[dict]:
    summary: list[dict] = []
    for sym in tickers:
        try:
            df = get_data(sym, start_date, end_date)
            df = _truncate_ohlcv(df, as_of)
            if df is None or len(df) == 0:
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
            ema9 = compute_ema9_metrics(close_on_adj_index)

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
                    "Symbol": _symbol_display(sym),
                    "Close": ema9["last_close"],
                    "9EMA": ema9["ema9_close"],
                    "Close_Below_9EMA": ema9["close_below_9ema"],
                    "Above_9EMA_Since": ema9["above_9ema_since"],
                    "Pct_Above_9EMA": ema9["pct_since_cross"],
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
        except Exception:
            continue
    return summary


def _compute_rs_blended(df_summary: pd.DataFrame) -> pd.DataFrame:
    if df_summary.empty:
        return df_summary

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

    df_summary["Rank_vs_Peak"] = (
        df_summary["Peak_Proximity_Score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    w1w, w2w, w1m, w3m = W_RS_BLEND
    for suf in RETURN_SUFFIXES:
        df_summary[f"Rank_{suf}"] = df_summary[f"Return_{suf}"].rank(ascending=False)
    for suf in RETURN_SUFFIXES:
        df_summary[f"Rank_RS_{suf}"] = df_summary[f"RS_{suf}_vs_N500"].rank(
            ascending=False, na_option="bottom"
        )

    df_summary["Abs_Score"] = (
        w1w * df_summary["Rank_1W"]
        + w2w * df_summary["Rank_2W"]
        + w1m * df_summary["Rank_1M"]
        + w3m * df_summary["Rank_3M"]
    )
    df_summary["RS_Score"] = (
        w1w * df_summary["Rank_RS_1W"]
        + w2w * df_summary["Rank_RS_2W"]
        + w1m * df_summary["Rank_RS_1M"]
        + w3m * df_summary["Rank_RS_3M"]
    )
    df_summary["Abs_Momentum_Rank"] = df_summary["Abs_Score"].rank(ascending=True)
    df_summary["Relative_Strength_Rank"] = df_summary["RS_Score"].rank(ascending=True)
    df_summary["Blended_Rank"] = (
        df_summary["Abs_Momentum_Rank"] + df_summary["Relative_Strength_Rank"]
    ) / 2

    df_out = df_summary.sort_values("Blended_Rank").reset_index(drop=True)
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))

    final_cols = [
        "Position",
        "Symbol",
        "Close",
        "9EMA",
        "Close_Below_9EMA",
        "Above_9EMA_Since",
        "Pct_Above_9EMA",
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
    ]
    return df_out[final_cols]


def _weighted_excess_return(row: pd.Series) -> float:
    if any(pd.isna(row[c]) for c in RS_RANK_COLS):
        return float("nan")
    w1w, w2w, w1m = W_RS_ADAPTIVE
    return (
        w1w * row["RS_1W_vs_N500"]
        + w2w * row["RS_2W_vs_N500"]
        + w1m * row["RS_1M_vs_N500"]
    )


def _compute_rs_adaptive(df_summary: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df_summary.empty:
        return df_summary, 0

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

    df_summary["Weighted_RS_pct"] = df_summary.apply(_weighted_excess_return, axis=1)
    etfs_ranked = int(df_summary["Weighted_RS_pct"].notna().sum())
    if etfs_ranked == 0:
        return pd.DataFrame(), 0

    df_out = (
        df_summary.sort_values("Weighted_RS_pct", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    df_out["Weighted_RS_pct"] = df_out["Weighted_RS_pct"].round(2)

    final_cols = [
        "Position",
        "Symbol",
        "Close",
        "9EMA",
        "Close_Below_9EMA",
        "Above_9EMA_Since",
        "Pct_Above_9EMA",
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
    ]
    return df_out[final_cols], etfs_ranked


def fetch_etf_momentum_snapshot(as_of_date: str | None = None) -> EtfMomentumSnapshot:
    """Fetch data and compute all three ranker outputs as of ``as_of_date`` (YYYY-MM-DD)."""
    as_of = _resolve_as_of(as_of_date)
    end_date = as_of + pd.Timedelta(days=1)
    start_date = as_of - pd.Timedelta(days=365 * 2)
    run_date = as_of.strftime("%Y-%m-%d")
    prepare_india_market_data_range(start_date, as_of)
    abs_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            stock_data = get_data(ticker, start_date, end_date)
            stock_data = _truncate_ohlcv(stock_data, as_of)
            if stock_data is not None and len(stock_data) > 0:
                abs_data[ticker] = stock_data
        except Exception:
            continue

    abs_df = _compute_abs_momentum(abs_data)

    nifty_df = get_data(BENCHMARK_TICKER, start_date, end_date)
    nifty_df = _truncate_ohlcv(nifty_df, as_of)
    if nifty_df is None or len(nifty_df) == 0:
        raise RuntimeError(f"No benchmark data for {BENCHMARK_TICKER} on or before {run_date}")

    nifty_adj = _adj_close_series(nifty_df).dropna()
    if nifty_adj.empty:
        raise RuntimeError(f"No benchmark prices on or before {run_date}")
    market_regime = classify_ema_regime(nifty_adj, BENCH_EMA_FAST, BENCH_EMA_SLOW)
    benchmark_1m = _benchmark_return_1m(nifty_adj)

    rs_rows = _collect_rs_rows(nifty_adj, start_date, end_date, as_of=as_of)
    rs_summary = pd.DataFrame(rs_rows)

    rs_blended = _compute_rs_blended(rs_summary.copy())
    rs_adaptive, etfs_ranked = _compute_rs_adaptive(rs_summary.copy())

    run_info: dict[str, str | int | float | None] = {
        "As_Of_Date": run_date,
        "Run_Date": run_date,
        "Weight_Profile": _weight_label_adaptive(),
        "Rank_By": "Weighted_RS_pct vs N500 (highest first)",
        "Entry_Filters": "200 EMA + 52w proximity (or ADTV for new listings)",
        "RS_Extra_Cutoffs": "None",
        "Benchmark_Return_1M_pct": round(benchmark_1m, 2)
        if not pd.isna(benchmark_1m)
        else None,
        "Market_Regime": market_regime,
        "ETFs_Ranked": etfs_ranked,
        "Rows_Written": len(rs_adaptive),
        "Abs_Momentum_Rows": len(abs_df),
        "RS_Blended_Rows": len(rs_blended),
        "Data_Window": format_range_label(start_date, end_date),
        "Data_Cache": get_india_market_data_run_stats().summary(),
        **summarize_etf_history_gaps(tickers, min_bars=MIN_BARS_NS_BHAVCOPY_FIRST),
    }

    return EtfMomentumSnapshot(
        run_date=run_date,
        market_regime=market_regime,
        benchmark_return_1m_pct=run_info["Benchmark_Return_1M_pct"],  # type: ignore[arg-type]
        abs_momentum=abs_df,
        rs_blended=rs_blended,
        rs_adaptive=rs_adaptive,
        run_info=run_info,
        etfs_ranked_adaptive=etfs_ranked,
    )
