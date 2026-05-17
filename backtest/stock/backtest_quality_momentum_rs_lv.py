"""
Walk-forward backtest for the Quality Momentum RS + low-volatility stock strategy
in momentum/stock/quality_momentum_rs_lv.py.

Bulk-downloads adj close + volume once, then ranks point-in-time each rebalance:
  trend (200 EMA), 52w-high proximity, ADTV, low-vol quantile, blended momentum/RS vs ^CRSLDX.

Holdings: top PORTFOLIO_SIZE, equal-weight between rebalances.
EXIT_RANK_THRESHOLD: keep prior holdings while universe rank <= this (hysteresis).

Configure in this file (below) or override via CLI:
    --portfolio-size, --exit-rank-threshold, --benchmark-ticker
    (aliases: --top-n, --worst-rank; presets: --benchmark nifty_500 | nifty_50)

Examples:
    python backtest/stock/backtest_quality_momentum_rs_lv.py
    python backtest/stock/backtest_quality_momentum_rs_lv.py --start 2023-01-01 --end 2025-12-31
    python backtest/stock/backtest_quality_momentum_rs_lv.py --portfolio-size 20 --exit-rank-threshold 30
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_STOCK_DIR


def _load_strategy_module():
    path = _PROJECT_ROOT / "momentum" / "stock" / "quality_momentum_rs_lv.py"
    spec = importlib.util.spec_from_file_location("quality_momentum_rs_lv", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QM = _load_strategy_module()

# ──────────────────────────────────────────────
# Configurable parameters (defaults mirror live script)
# ──────────────────────────────────────────────

BACKTEST_START = "2026-01-01"
BACKTEST_END = "2026-05-15"

# Portfolio sizing (edit here or pass --portfolio-size / --exit-rank-threshold on CLI)
PORTFOLIO_SIZE = 20
EXIT_RANK_THRESHOLD = 30

# RS benchmark (Yahoo ticker; edit here or --benchmark-ticker / --benchmark on CLI)
BENCHMARK_TICKER = "^CRSLDX"
BENCHMARK_PRESETS: dict[str, str] = {
    "nifty_500": "^CRSLDX",
    "nifty_50": "^NSEI",
}

REBALANCE_PERIOD = "biweekly"
INITIAL_CAPITAL = 100_000

STRATEGY_TAG = "quality_momentum_rs_lv"

LB_1M = QM.LB_1M
LB_3M = QM.LB_3M
LB_6M = QM.LB_6M
LB_9M = QM.LB_9M
MIN_ADTV_CRORES = QM.MIN_ADTV_CRORES
HIGH_52W_LOOKBACK = 252
# Min price / 52w high (0.70 = within 30% of high). CLI: --52w-proximity. Env: QUALITY_RS_52W_PROXIMITY.
PROXIMITY_OF_52W_HIGH = 0.70
EMA_SPAN = 200
ADTV_WINDOW = 20
VOL_WINDOW = 21

MIN_HISTORY = LB_9M + HIGH_52W_LOOKBACK + 30

BACKTEST_OUT_DIR = FINAL_RESULT_STOCK_DIR / "backtest"

PERIODS_PER_YEAR = {
    "weekly": 52,
    "biweekly": 26,
    "monthly": 12,
}


def resolve_proximity_of_52w_high(*, proximity_52w: float | None = None) -> float:
    if proximity_52w is not None:
        v = float(proximity_52w)
        if not (0 < v <= 1):
            raise ValueError(f"52w proximity must be in (0, 1], got {proximity_52w}")
        return v
    raw = (os.environ.get("QUALITY_RS_52W_PROXIMITY") or "").strip()
    if raw:
        v = float(raw)
        if not (0 < v <= 1):
            raise ValueError(
                "QUALITY_RS_52W_PROXIMITY must be a float in (0, 1], got " + repr(raw)
            )
        return v
    return float(PROXIMITY_OF_52W_HIGH)


def resolve_benchmark_ticker(
    *,
    benchmark_ticker: str | None = None,
    benchmark_preset: str | None = None,
) -> str:
    """File default BENCHMARK_TICKER, unless CLI preset or explicit ticker is set."""
    if benchmark_preset:
        key = benchmark_preset.strip().lower().replace("-", "_")
        if key not in BENCHMARK_PRESETS:
            raise ValueError(
                f"Unknown benchmark preset {benchmark_preset!r}; "
                f"choose from {sorted(BENCHMARK_PRESETS)}"
            )
        return BENCHMARK_PRESETS[key]
    raw = (benchmark_ticker or BENCHMARK_TICKER).strip()
    if not raw:
        raise ValueError("benchmark_ticker must be non-empty")
    return raw

REBALANCE_ALIASES = {
    "bi-weekly": "biweekly",
    "bi_weekly": "biweekly",
    "biweekly": "biweekly",
    "weekly": "weekly",
    "monthly": "monthly",
}


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────


def _adj_col(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _vol_col(df: pd.DataFrame) -> pd.Series:
    s = df["Volume"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _download_adj_vol(
    yahoo_symbols: list[str], start: str, end: str
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    extra_start = (pd.Timestamp(start) - pd.Timedelta(days=int(MIN_HISTORY * 1.6))).strftime("%Y-%m-%d")
    adj_store: dict[str, pd.Series] = {}
    vol_store: dict[str, pd.Series] = {}
    for sym in yahoo_symbols:
        try:
            df = yf.download(
                sym,
                start=extra_start,
                end=end,
                multi_level_index=False,
                auto_adjust=False,
                progress=False,
            )
            if df is None or len(df) == 0:
                continue
            adj_store[sym] = _adj_col(df)
            vol_store[sym] = _vol_col(df)
        except Exception:
            pass
    return adj_store, vol_store


# ──────────────────────────────────────────────
# Point-in-time ranking (mirrors quality_momentum_rs_lv.py)
# ──────────────────────────────────────────────


def _rank_at_date(
    stock_adj: dict[str, pd.Series],
    stock_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    bench_slice = bench_adj.loc[:as_of]
    if len(bench_slice) < LB_9M:
        return pd.DataFrame()

    summary: list[dict] = []
    for t in QM.tickers:
        sym = t["symbol"]
        if sym not in stock_adj:
            continue
        adj = stock_adj[sym].loc[:as_of]
        if len(adj) < LB_9M:
            continue

        vol_s = stock_vol.get(sym)
        if vol_s is None or len(vol_s) == 0:
            continue
        vol = vol_s.reindex(adj.index).fillna(0.0)

        daily_turnover = adj * vol
        adtv_crores = float(daily_turnover.tail(ADTV_WINDOW).mean()) / 10_000_000
        if adtv_crores < MIN_ADTV_CRORES:
            continue

        ema200 = float(adj.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])
        high_52w = float(adj.iloc[-min(HIGH_52W_LOOKBACK, len(adj)) :].max())
        last = float(adj.iloc[-1])

        if last < ema200 or last < (high_52w * proximity_of_52w_high):
            continue

        ret_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
        ret_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100
        ret_6m = (adj.iloc[-1] / adj.iloc[-LB_6M] - 1) * 100
        ret_9m = (adj.iloc[-1] / adj.iloc[-LB_9M] - 1) * 100
        vol_score = float(adj.pct_change().tail(VOL_WINDOW).std() * 100)

        nx = bench_slice.reindex(adj.index).ffill()
        rs_3m = ret_3m - ((nx.iloc[-1] / nx.iloc[-LB_3M] - 1) * 100)
        rs_6m = ret_6m - ((nx.iloc[-1] / nx.iloc[-LB_6M] - 1) * 100)
        rs_9m = ret_9m - ((nx.iloc[-1] / nx.iloc[-LB_9M] - 1) * 100)

        summary.append(
            {
                "Symbol": QM._symbol_for_excel(sym),
                "Industry": t.get("industry", ""),
                "Marketcap": t.get("marketcap", ""),
                "Return_1M": ret_1m,
                "Return_3M": ret_3m,
                "Return_6M": ret_6m,
                "Return_9M": ret_9m,
                "RS_3M_vs_Bench": rs_3m,
                "RS_6M_vs_Bench": rs_6m,
                "RS_9M_vs_Bench": rs_9m,
                "Volatility_Score": vol_score,
            }
        )

    if not summary:
        return pd.DataFrame()

    df_summary = pd.DataFrame(summary)
    df_summary = QM._apply_low_volatility_filter(df_summary, quiet=True)
    if df_summary.empty:
        return pd.DataFrame()
    return QM._apply_ranking_engine(df_summary)


def _select_holdings(
    ranked_df: pd.DataFrame,
    prev_holdings: list[str],
    top_n: int,
    worst_rank_held: int,
) -> list[str]:
    """Top-N entry; retain prior names while universe rank <= worst_rank_held."""
    if ranked_df.empty:
        return []

    rank_map = dict(zip(ranked_df["Symbol"], ranked_df["Rank_Position"]))

    retained = [
        sym for sym in prev_holdings if sym in rank_map and rank_map[sym] <= worst_rank_held
    ]
    retained_set = set(retained)
    new_entries = [
        sym
        for sym in ranked_df["Symbol"]
        if sym not in retained_set and rank_map[sym] <= top_n
    ]
    retained.sort(key=lambda s: rank_map[s])
    return (retained + new_entries)[:top_n]


def _rebalance_dates(
    bench_adj: pd.Series, backtest_start: str, backtest_end: str, period: str
) -> pd.DatetimeIndex:
    all_dates = bench_adj.loc[backtest_start:backtest_end].index
    if len(all_dates) == 0:
        raise RuntimeError("No trading dates in the backtest window")

    period = REBALANCE_ALIASES.get(period.strip().lower(), period.strip().lower())

    if period == "weekly":
        dates = all_dates.to_series().groupby(all_dates.to_period("W")).last().values
    elif period == "monthly":
        dates = all_dates.to_series().groupby(all_dates.to_period("M")).last().values
    elif period == "biweekly":
        rebal: list[pd.Timestamp] = []
        last_rebal: pd.Timestamp | None = None
        for d in all_dates:
            if last_rebal is None or (d - last_rebal).days >= 14:
                rebal.append(d)
                last_rebal = d
        dates = rebal
    else:
        raise ValueError(f"Unknown rebalance period {period!r}; use weekly, biweekly, or monthly")

    out = pd.DatetimeIndex(dates)
    return out[out >= all_dates[0]]


# ──────────────────────────────────────────────
# Walk-forward simulation
# ──────────────────────────────────────────────


def run_backtest(
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    rebalance_period: str = REBALANCE_PERIOD,
    portfolio_size: int = PORTFOLIO_SIZE,
    exit_rank_threshold: int = EXIT_RANK_THRESHOLD,
    benchmark_ticker: str = BENCHMARK_TICKER,
    proximity_of_52w_high: float | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    rebalance_period = REBALANCE_ALIASES.get(
        rebalance_period.strip().lower(), rebalance_period.strip().lower()
    )
    periods_per_year = PERIODS_PER_YEAR.get(rebalance_period, 26)
    prox_52w = resolve_proximity_of_52w_high(proximity_52w=proximity_of_52w_high)

    yahoo_symbols = [t["symbol"] for t in QM.tickers]

    print(f"[{STRATEGY_TAG}] Backtest")
    print(f"Benchmark : {benchmark_ticker}")
    print(f"Period    : {backtest_start} to {backtest_end}")
    print(f"Rebalance : {rebalance_period}")
    print(f"Portfolio : {portfolio_size}  |  Exit rank threshold: {exit_rank_threshold}")
    print(f"52w proximity: {prox_52w:.2f}")
    print(f"Universe  : {len(yahoo_symbols)} Quality tickers")
    print(f"Capital   : {initial_capital:,.0f}\n")

    print("Downloading stock data ...")
    stock_adj, stock_vol = _download_adj_vol(yahoo_symbols, backtest_start, backtest_end)
    print(f"  {len(stock_adj)} / {len(yahoo_symbols)} symbols loaded")

    print(f"Downloading benchmark {benchmark_ticker} ...")
    bench_adj, _ = _download_adj_vol([benchmark_ticker], backtest_start, backtest_end)
    if benchmark_ticker not in bench_adj:
        raise RuntimeError(f"Could not download benchmark {benchmark_ticker}")
    bench_series = bench_adj[benchmark_ticker]
    print(f"  Benchmark rows: {len(bench_series)}")

    rebal_dates = _rebalance_dates(bench_series, backtest_start, backtest_end, rebalance_period)
    if len(rebal_dates) < 2:
        raise RuntimeError("Need at least two rebalance dates in the backtest window")
    print(f"  Rebalance dates: {len(rebal_dates)}\n")

    records: list[dict] = []
    portfolio_value = initial_capital
    prev_holdings: list[str] = []
    total_trades = 0

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        ranked_df = _rank_at_date(
            stock_adj, stock_vol, bench_series, rebal_date, proximity_of_52w_high=prox_52w
        )
        holdings = _select_holdings(
            ranked_df, prev_holdings, portfolio_size, exit_rank_threshold
        )

        if holdings:
            period_rets: list[float] = []
            for sym_excel in holdings:
                yh = QM._EXCEL_SYMBOL_TO_YAHOO.get(sym_excel)
                if not yh or yh not in stock_adj:
                    period_rets.append(0.0)
                    continue
                s = stock_adj[yh]
                s_from = s.loc[:rebal_date]
                s_to = s.loc[:next_date]
                if len(s_from) == 0 or len(s_to) == 0:
                    period_rets.append(0.0)
                    continue
                p0 = float(s_from.iloc[-1])
                p1 = float(s_to.iloc[-1])
                period_rets.append((p1 / p0 - 1) if p0 > 0 else 0.0)
            port_ret = float(np.mean(period_rets))
        else:
            port_ret = 0.0

        b_from = bench_series.loc[:rebal_date]
        b_to = bench_series.loc[:next_date]
        bench_ret = (
            (float(b_to.iloc[-1]) / float(b_from.iloc[-1]) - 1)
            if len(b_from) > 0 and len(b_to) > 0
            else 0.0
        )

        turnover = 0.0
        new_entries_count = 0
        if prev_holdings:
            old_set = set(prev_holdings)
            new_set = set(holdings)
            new_entries_count = len(new_set - old_set)
            changed = len(old_set.symmetric_difference(new_set))
            turnover = changed / max(len(old_set | new_set), 1)
        else:
            new_entries_count = len(holdings)

        total_trades += new_entries_count
        portfolio_value *= 1 + port_ret

        records.append(
            {
                "Rebal_Date": rebal_date,
                "End_Date": next_date,
                "Holdings": ", ".join(holdings) if holdings else "CASH",
                "Num_Holdings": len(holdings),
                "Universe_Ranked": len(ranked_df),
                "Port_Return": port_ret,
                "Bench_Return": bench_ret,
                "Excess_Return": port_ret - bench_ret,
                "Turnover": turnover,
                "Portfolio_Value": portfolio_value,
            }
        )
        prev_holdings = holdings

    df_trades = pd.DataFrame(records)
    if df_trades.empty:
        print("No rebalance periods — check date range and data availability.")
        return {}

    df_trades["Bench_Value"] = initial_capital * (1 + df_trades["Bench_Return"]).cumprod()

    metrics = _compute_metrics(
        df_trades,
        initial_capital,
        rebalance_period,
        portfolio_size,
        exit_rank_threshold,
        benchmark_ticker,
        prox_52w,
        periods_per_year,
        total_trades,
    )
    _print_metrics(metrics)
    BACKTEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_excel(df_trades, metrics, rebalance_period)
    _plot_equity(
        df_trades,
        metrics,
        rebalance_period,
        portfolio_size,
        exit_rank_threshold,
        benchmark_ticker,
    )
    _append_run_log(metrics)

    return metrics


# ──────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────


def _compute_metrics(
    df: pd.DataFrame,
    capital: float,
    rebalance_period: str,
    portfolio_size: int,
    exit_rank_threshold: int,
    benchmark_ticker: str,
    proximity_of_52w_high: float,
    periods_per_year: int,
    total_trades: int,
) -> dict:
    port_rets = df["Port_Return"].values
    bench_rets = df["Bench_Return"].values
    n_periods = len(port_rets)

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    bench_total = df["Bench_Value"].iloc[-1] / capital - 1
    years = n_periods / periods_per_year if periods_per_year > 0 else 0.0

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    bench_cum = (1 + pd.Series(bench_rets)).cumprod()
    bench_max_dd = float((bench_cum / bench_cum.cummax() - 1).min())

    ann_vol = np.std(port_rets, ddof=1) * np.sqrt(periods_per_year) if n_periods > 1 else 0.0
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    downside = port_rets[port_rets < 0]
    downside_vol = (
        np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    )
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    win_rate = float(np.mean(port_rets > 0) * 100)
    avg_period = float(np.mean(port_rets) * 100)
    avg_turnover = float(df["Turnover"].mean() * 100)

    alpha = cagr - bench_cagr
    excess = pd.Series(port_rets) - pd.Series(bench_rets)
    te = float(excess.std(ddof=1)) * np.sqrt(periods_per_year) if n_periods > 1 else 0.0
    info_ratio = alpha / te if te > 0 else 0.0

    return {
        "Strategy": STRATEGY_TAG,
        "BENCHMARK_TICKER": benchmark_ticker,
        "PROXIMITY_OF_52W_HIGH": proximity_of_52w_high,
        "Rebalance_Period": rebalance_period,
        "Period": (
            f"{df['Rebal_Date'].iloc[0].strftime('%Y-%m-%d')} to "
            f"{df['End_Date'].iloc[-1].strftime('%Y-%m-%d')}"
        ),
        "PORTFOLIO_SIZE": portfolio_size,
        "EXIT_RANK_THRESHOLD": exit_rank_threshold,
        "Periods": n_periods,
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(win_rate, 1),
        f"Avg_{rebalance_period.capitalize()}_Return_%": round(avg_period, 2),
        "Total_Trades": total_trades,
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Ann_Volatility_%": round(ann_vol * 100, 2),
        "Avg_Turnover_%": round(avg_turnover, 1),
        "Alpha_%": round(alpha * 100, 2),
        "Information_Ratio": round(info_ratio, 2),
        "Bench_Total_Return_%": round(bench_total * 100, 2),
        "Bench_CAGR_%": round(bench_cagr * 100, 2),
        "Bench_Max_Drawdown_%": round(bench_max_dd * 100, 2),
        "Final_Value": round(df["Portfolio_Value"].iloc[-1], 2),
        "Bench_Final_Value": round(df["Bench_Value"].iloc[-1], 2),
    }


def _print_metrics(m: dict) -> None:
    period_label = m.get("Rebalance_Period", "period")
    avg_key = f"Avg_{period_label.capitalize()}_Return_%"
    print("\n" + "=" * 55)
    print(f"  BACKTEST RESULTS [{STRATEGY_TAG}]")
    print("=" * 55)
    fmt = [
        ("BENCHMARK_TICKER", m.get("BENCHMARK_TICKER", "")),
        ("PROXIMITY_OF_52W_HIGH", m.get("PROXIMITY_OF_52W_HIGH", "")),
        ("Period", m["Period"]),
        ("Rebalance", m.get("Rebalance_Period", "")),
        ("PORTFOLIO_SIZE", m.get("PORTFOLIO_SIZE", "")),
        ("EXIT_RANK_THRESHOLD", m.get("EXIT_RANK_THRESHOLD", "")),
        ("Rebalance periods", m.get("Periods", "")),
        ("", ""),
        ("Strategy Total Return", f"{m['Total_Return_%']:+.2f} %"),
        ("Strategy CAGR", f"{m['CAGR_%']:+.2f} %"),
        ("Max Drawdown", f"{m['Max_Drawdown_%']:.2f} %"),
        ("Sharpe Ratio", f"{m['Sharpe']:.2f}"),
        ("Sortino Ratio", f"{m['Sortino']:.2f}"),
        ("Calmar Ratio", f"{m['Calmar']:.2f}"),
        ("Win Rate", f"{m['Win_Rate_%']:.1f} %"),
        (f"Avg {period_label} return", f"{m.get(avg_key, 0):+.2f} %"),
        ("Annual Volatility", f"{m['Ann_Volatility_%']:.2f} %"),
        ("Avg Turnover", f"{m['Avg_Turnover_%']:.1f} %"),
        ("", ""),
        ("Benchmark Total Return", f"{m['Bench_Total_Return_%']:+.2f} %"),
        ("Benchmark CAGR", f"{m['Bench_CAGR_%']:+.2f} %"),
        ("Benchmark Max DD", f"{m['Bench_Max_Drawdown_%']:.2f} %"),
        ("", ""),
        ("Alpha (ann.)", f"{m['Alpha_%']:+.2f} %"),
        ("Information Ratio", f"{m['Information_Ratio']:.2f}"),
        ("", ""),
        ("Final Portfolio Value", f"Rs {m['Final_Value']:,.2f}"),
        ("Final Benchmark Value", f"Rs {m['Bench_Final_Value']:,.2f}"),
    ]
    for label, val in fmt:
        if label == "":
            print()
        else:
            print(f"  {label:<26s} {val}")
    print("=" * 55 + "\n")


def _write_excel(df_trades: pd.DataFrame, metrics: dict, rebalance_period: str) -> None:
    out_path = BACKTEST_OUT_DIR / f"backtest_quality_momentum_rs_lv_{rebalance_period}.xlsx"
    if out_path.exists():
        try:
            out_path.open("a").close()
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_path.with_stem(f"{out_path.stem}_{ts}")
            print(f"  (original file locked, writing to {out_path.name})")

    df_summary = pd.DataFrame([metrics]).T
    df_summary.columns = ["Value"]
    df_summary.index.name = "Metric"

    df_equity = df_trades[["End_Date", "Portfolio_Value", "Bench_Value"]].copy()
    df_equity.columns = ["Date", "Strategy", "Benchmark"]

    pct_cols = ["Port_Return", "Bench_Return", "Excess_Return", "Turnover"]
    df_t = df_trades.copy()
    for c in pct_cols:
        df_t[c] = (df_t[c] * 100).round(2)
    df_t["Portfolio_Value"] = df_t["Portfolio_Value"].round(2)
    df_t["Bench_Value"] = df_t["Bench_Value"].round(2)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary")
        df_equity.to_excel(writer, sheet_name="Equity_Curve", index=False)
        df_t.to_excel(writer, sheet_name="Trades", index=False)

    print(f"Excel -> {out_path}")


def _plot_equity(
    df_trades: pd.DataFrame,
    metrics: dict,
    rebalance_period: str,
    portfolio_size: int,
    exit_rank_threshold: int,
    benchmark_ticker: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    dates = df_trades["End_Date"]
    strat_vals = df_trades["Portfolio_Value"]
    bench_vals = df_trades["Bench_Value"]

    ax1 = axes[0]
    ax1.plot(dates, strat_vals, label="Strategy", linewidth=1.5, color="#1f77b4")
    ax1.plot(
        dates,
        bench_vals,
        label=f"Benchmark ({benchmark_ticker})",
        linewidth=1.2,
        color="#aaaaaa",
        linestyle="--",
    )
    ax1.set_ylabel("Portfolio Value (Rs)")
    ax1.set_title(
        f"Quality Momentum RS+LV  |  {rebalance_period}  |  "
        f"N={portfolio_size} (hold to rank {exit_rank_threshold})  |  "
        f"CAGR {metrics['CAGR_%']:+.1f}%  vs  {metrics['Bench_CAGR_%']:+.1f}%  |  "
        f"Sharpe {metrics['Sharpe']:.2f}  |  MaxDD {metrics['Max_Drawdown_%']:.1f}%",
        fontsize=11,
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    cum_ret = (1 + df_trades["Port_Return"]).cumprod()
    dd = (cum_ret / cum_ret.cummax() - 1) * 100

    ax2 = axes[1]
    ax2.fill_between(dates, dd, 0, color="#d62728", alpha=0.35)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    png_path = BACKTEST_OUT_DIR / f"backtest_quality_momentum_rs_lv_{rebalance_period}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart -> {png_path}")


RUN_LOG_PATH = BACKTEST_OUT_DIR / "backtest_quality_momentum_rs_lv_run_log.csv"


def _append_run_log(metrics: dict) -> None:
    row = {"Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    df_row = pd.DataFrame([row])

    if RUN_LOG_PATH.exists():
        df_existing = pd.read_csv(RUN_LOG_PATH)
        df_log = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_log = df_row

    df_log.to_csv(RUN_LOG_PATH, index=False)
    print(f"Run log -> {RUN_LOG_PATH}  ({len(df_log)} runs recorded)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest Quality Momentum RS + low-volatility stocks (quality_momentum_rs_lv.py)",
    )
    parser.add_argument("--start", default=BACKTEST_START, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=BACKTEST_END, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--rebalance",
        default=REBALANCE_PERIOD,
        choices=["weekly", "biweekly", "bi-weekly", "monthly"],
        help="Rebalance frequency (default: biweekly, matches live script)",
    )
    parser.add_argument(
        "--portfolio-size",
        "--top-n",
        type=int,
        default=PORTFOLIO_SIZE,
        dest="portfolio_size",
        help=f"Holdings count (file default PORTFOLIO_SIZE={PORTFOLIO_SIZE})",
    )
    parser.add_argument(
        "--exit-rank-threshold",
        "--worst-rank",
        type=int,
        default=EXIT_RANK_THRESHOLD,
        dest="exit_rank_threshold",
        help=(
            f"Keep prior holdings while rank <= this "
            f"(file default EXIT_RANK_THRESHOLD={EXIT_RANK_THRESHOLD})"
        ),
    )
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument(
        "--benchmark-ticker",
        default=None,
        metavar="TICKER",
        help=f"Yahoo benchmark symbol (file default BENCHMARK_TICKER={BENCHMARK_TICKER})",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        choices=sorted(BENCHMARK_PRESETS),
        metavar="PRESET",
        help=(
            "Benchmark preset (overrides --benchmark-ticker when set): "
            + ", ".join(f"{k}={v}" for k, v in BENCHMARK_PRESETS.items())
        ),
    )
    parser.add_argument(
        "--52w-proximity",
        type=float,
        default=None,
        dest="proximity_52w",
        metavar="RATIO",
        help=(
            f"Min price / 52w high (file PROXIMITY_OF_52W_HIGH={PROXIMITY_OF_52W_HIGH}). "
            "Env: QUALITY_RS_52W_PROXIMITY."
        ),
    )

    args = parser.parse_args()

    resolved_benchmark = resolve_benchmark_ticker(
        benchmark_ticker=args.benchmark_ticker,
        benchmark_preset=args.benchmark,
    )

    run_backtest(
        backtest_start=args.start,
        backtest_end=args.end,
        rebalance_period=args.rebalance,
        portfolio_size=args.portfolio_size,
        exit_rank_threshold=max(args.exit_rank_threshold, args.portfolio_size),
        benchmark_ticker=resolved_benchmark,
        proximity_of_52w_high=args.proximity_52w,
        initial_capital=args.capital,
    )
