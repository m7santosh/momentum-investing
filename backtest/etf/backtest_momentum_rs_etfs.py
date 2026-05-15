"""
Walk-forward backtest for the ETF relative-strength momentum strategy
defined in momentum/etf/momentum_rs_etfs.py.

Plain weekly rebalance: re-runs the ranking pipeline each week, picks
Top-N ETFs, equal-weights them, holds for one week, repeats.

Entry / Exit rules:
    - Entry : ETF must rank <= TOP_N to enter the portfolio.
    - Hold  : rank <= WORST_RANK_HELD -> keep (buffer zone).
    - Exit  : rank > WORST_RANK_HELD -> drop at rebalance.

Configurable:
    BACKTEST_START / BACKTEST_END  - calendar date range.
    BENCHMARK_TICKER               - RS anchor (^CRSLDX = Nifty 500,
                                     ^NSEI = Nifty 50).
    TOP_N                          - number of ETFs to hold.
    WORST_RANK_HELD                - rank buffer for existing holdings
                                     (>= TOP_N).
    USE_REGIME_FILTER              - go 100% cash when Trend_Down.
    INITIAL_CAPITAL                - starting value.

Examples:
    python backtest/etf/backtest_momentum_rs_etfs.py --top-n 5 --worst-rank 5
    python backtest/etf/backtest_momentum_rs_etfs.py --top-n 5 --worst-rank 10
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_ETF_DIR

# ──────────────────────────────────────────────
# Configurable parameters
# ──────────────────────────────────────────────

BACKTEST_START = "2024-09-01"
BACKTEST_END   = "2025-03-31"

BENCHMARKS = {
    "Nifty_500": "^CRSLDX",
    "Nifty_50":  "^NSEI",
}
BENCHMARK_NAME = "Nifty_500"

TOP_N = 5
WORST_RANK_HELD = 10
USE_REGIME_FILTER = False
INITIAL_CAPITAL = 100_000

STRATEGY_TAG = "base"

# ── Strategy constants (mirror momentum_rs_etfs.py) ──

LB_1W  = 6
LB_2W  = 11
LB_1M  = 21
LB_3M  = 63
LB_52W = 252

W_1W, W_2W, W_1M, W_3M = 0.10, 0.10, 0.25, 0.55

EMA_SPAN = 200
PROXIMITY_OF_52W_HIGH = 0.70

BENCH_EMA_FAST = 50
BENCH_EMA_SLOW = 200

RETURN_SUFFIXES = ("1W", "2W", "1M", "3M")

TICKERS = [
    "ALPHA.NS", "AUTOBEES.NS", "BANKBEES.NS", "CONSUMBEES.NS", "CPSEETF.NS",
    "MOENERGY.NS", "FMCGIETF.NS", "GOLDBEES.NS", "GROWWPOWER.NS", "GROWWRAIL.NS",
    "HEALTHIETF.NS", "HNGSNGBEES.NS", "ICICIB22.NS", "INFRABEES.NS", "ITBEES.NS",
    "LIQUIDCASE.NS", "MAHKTECH.NS", "METALIETF.NS", "MIDCAPETF.NS", "MOCAPITAL.NS",
    "MODEFENCE.NS", "MON100.NS", "MOREALTY.NS", "MOTOUR.NS", "MOVALUE.NS",
    "NEXT50IETF.NS", "NIFTYBEES.NS", "OILIETF.NS", "PHARMABEES.NS", "PSUBNKBEES.NS",
    "PVTBANIETF.NS", "MOMIDMTM.NS", "SILVERBEES.NS", "SMALLCAP.NS",
]

BACKTEST_OUT_DIR = FINAL_RESULT_ETF_DIR / "backtest"
MIN_HISTORY = LB_52W + LB_3M


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _adj_col(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _download_all(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """Bulk download Adj Close for every ticker; returns {ticker: Series}."""
    extra_start = (pd.Timestamp(start) - pd.Timedelta(days=int(MIN_HISTORY * 1.6))).strftime("%Y-%m-%d")
    store: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            df = yf.download(t, start=extra_start, end=end,
                             multi_level_index=False, auto_adjust=False, progress=False)
            if df is not None and len(df) > 0:
                store[t] = _adj_col(df)
        except Exception:
            pass
    return store


def _classify_regime(close: pd.Series) -> str:
    if len(close) < BENCH_EMA_SLOW:
        return "Unknown"
    last = float(close.iloc[-1])
    e_fast = float(close.ewm(span=BENCH_EMA_FAST, adjust=False).mean().iloc[-1])
    e_slow = float(close.ewm(span=BENCH_EMA_SLOW, adjust=False).mean().iloc[-1])
    if last >= e_slow and last >= e_fast:
        return "Trend_Up"
    if last < e_slow and last < e_fast:
        return "Trend_Down"
    return "Mixed_Above50" if last >= e_fast else "Mixed_Below50"


# ──────────────────────────────────────────────
# Core ranking logic (point-in-time)
# ──────────────────────────────────────────────

def _rank_at_date(
    etf_data: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []

    bench_slice = bench_adj.loc[:as_of]
    if len(bench_slice) < LB_3M:
        return pd.DataFrame()

    for sym, full_adj in etf_data.items():
        adj = full_adj.loc[:as_of]
        if len(adj) < LB_52W:
            continue

        ema200_val = float(adj.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])
        high_52w = adj.iloc[-min(LB_52W, len(adj)):].max()
        last = float(adj.iloc[-1])

        if last < ema200_val or last < (high_52w * PROXIMITY_OF_52W_HIGH):
            continue

        ret_1w = (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100
        ret_2w = (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100
        ret_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
        ret_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100

        nx = bench_slice.reindex(adj.index).ffill()
        tail = nx.iloc[-LB_3M:]
        rs_1w = rs_2w = rs_1m = rs_3m = float("nan")
        if not tail.isna().any() and (tail > 0).all():
            bn_1w = (nx.iloc[-1] / nx.iloc[-LB_1W] - 1) * 100
            bn_2w = (nx.iloc[-1] / nx.iloc[-LB_2W] - 1) * 100
            bn_1m = (nx.iloc[-1] / nx.iloc[-LB_1M] - 1) * 100
            bn_3m = (nx.iloc[-1] / nx.iloc[-LB_3M] - 1) * 100
            rs_1w = ret_1w - bn_1w
            rs_2w = ret_2w - bn_2w
            rs_1m = ret_1m - bn_1m
            rs_3m = ret_3m - bn_3m

        rows.append({
            "Symbol": sym,
            "Return_1W": ret_1w, "Return_2W": ret_2w,
            "Return_1M": ret_1m, "Return_3M": ret_3m,
            "RS_1W": rs_1w, "RS_2W": rs_2w,
            "RS_1M": rs_1m, "RS_3M": rs_3m,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for suf in RETURN_SUFFIXES:
        df[f"Rank_{suf}"] = df[f"Return_{suf}"].rank(ascending=False)
    for suf in RETURN_SUFFIXES:
        df[f"Rank_RS_{suf}"] = df[f"RS_{suf}"].rank(ascending=False, na_option="bottom")

    df["Abs_Score"] = (
        W_1W * df["Rank_1W"] + W_2W * df["Rank_2W"]
        + W_1M * df["Rank_1M"] + W_3M * df["Rank_3M"]
    )
    df["RS_Score"] = (
        W_1W * df["Rank_RS_1W"] + W_2W * df["Rank_RS_2W"]
        + W_1M * df["Rank_RS_1M"] + W_3M * df["Rank_RS_3M"]
    )

    df["Abs_Rank"] = df["Abs_Score"].rank(ascending=True)
    df["RS_Rank"]  = df["RS_Score"].rank(ascending=True)
    df["Blended"]  = (df["Abs_Rank"] + df["RS_Rank"]) / 2.0
    df["Blended_Rank"] = df["Blended"].rank(ascending=True, method="first").astype(int)

    return df.sort_values("Blended_Rank").reset_index(drop=True)


def _select_holdings(
    ranked_df: pd.DataFrame,
    prev_holdings: list[str],
    top_n: int,
    worst_rank_held: int,
) -> list[str]:
    """
    Entry rule : rank <= top_n to enter.
    Exit rule  : rank > worst_rank_held -> drop at rebalance.
    Portfolio is capped at top_n names.
    """
    if ranked_df.empty:
        return []

    rank_map = dict(zip(ranked_df["Symbol"], ranked_df["Blended_Rank"]))

    retained = [
        sym for sym in prev_holdings
        if sym in rank_map and rank_map[sym] <= worst_rank_held
    ]

    retained_set = set(retained)
    new_entries = [
        sym for sym in ranked_df["Symbol"]
        if sym not in retained_set and rank_map[sym] <= top_n
    ]

    retained.sort(key=lambda s: rank_map[s])
    holdings = retained + new_entries
    return holdings[:top_n]


# ──────────────────────────────────────────────
# Walk-forward simulation
# ──────────────────────────────────────────────

def run_backtest(
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    benchmark_name: str = BENCHMARK_NAME,
    top_n: int = TOP_N,
    worst_rank_held: int = WORST_RANK_HELD,
    use_regime_filter: bool = USE_REGIME_FILTER,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    benchmark_ticker = BENCHMARKS[benchmark_name]
    print(f"[BASE] Backtest")
    print(f"Benchmark : {benchmark_name} ({benchmark_ticker})")
    print(f"Period    : {backtest_start} to {backtest_end}")
    print(f"Top N     : {top_n}  |  Worst rank held: {worst_rank_held}  |  Regime filter: {use_regime_filter}")
    print(f"Capital   : {initial_capital:,.0f}\n")

    print("Downloading ETF data ...")
    etf_data = _download_all(TICKERS, backtest_start, backtest_end)
    print(f"  {len(etf_data)} / {len(TICKERS)} ETFs loaded")

    print(f"Downloading benchmark {benchmark_ticker} ...")
    bench_data = _download_all([benchmark_ticker], backtest_start, backtest_end)
    if benchmark_ticker not in bench_data:
        raise RuntimeError(f"Could not download benchmark {benchmark_ticker}")
    bench_adj = bench_data[benchmark_ticker]
    print(f"  Benchmark rows: {len(bench_adj)}")

    all_dates = bench_adj.loc[backtest_start:backtest_end].index
    if len(all_dates) == 0:
        raise RuntimeError("No trading dates in the backtest window")

    weekly_dates = all_dates.to_series().groupby(all_dates.to_period("W")).last().values
    weekly_dates = pd.DatetimeIndex(weekly_dates)
    weekly_dates = weekly_dates[weekly_dates >= all_dates[0]]
    print(f"  Rebalance dates: {len(weekly_dates)}\n")

    records: list[dict] = []
    portfolio_value = initial_capital
    prev_holdings: list[str] = []
    total_trades = 0

    for i, rebal_date in enumerate(weekly_dates[:-1]):
        next_date = weekly_dates[i + 1]

        regime = _classify_regime(bench_adj.loc[:rebal_date])
        go_cash = use_regime_filter and regime == "Trend_Down"

        if go_cash:
            holdings = []
        else:
            ranked_df = _rank_at_date(etf_data, bench_adj, rebal_date)
            holdings = _select_holdings(ranked_df, prev_holdings, top_n, worst_rank_held)

        if holdings:
            week_rets: list[float] = []
            for sym in holdings:
                s = etf_data[sym]
                s_from = s.loc[:rebal_date]
                s_to   = s.loc[:next_date]
                if len(s_from) == 0 or len(s_to) == 0:
                    week_rets.append(0.0)
                    continue
                p0 = float(s_from.iloc[-1])
                p1 = float(s_to.iloc[-1])
                week_rets.append((p1 / p0 - 1) if p0 > 0 else 0.0)
            port_ret = np.mean(week_rets)
        else:
            port_ret = 0.0

        b_from = bench_adj.loc[:rebal_date]
        b_to   = bench_adj.loc[:next_date]
        bench_ret = (float(b_to.iloc[-1]) / float(b_from.iloc[-1]) - 1) if len(b_from) > 0 and len(b_to) > 0 else 0.0

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
        portfolio_value *= (1 + port_ret)

        records.append({
            "Rebal_Date": rebal_date,
            "End_Date": next_date,
            "Holdings": ", ".join([s.replace(".NS", "") for s in holdings]) or "CASH",
            "Num_Holdings": len(holdings),
            "Regime": regime,
            "Port_Return": port_ret,
            "Bench_Return": bench_ret,
            "Excess_Return": port_ret - bench_ret,
            "Turnover": turnover,
            "Portfolio_Value": portfolio_value,
        })
        prev_holdings = holdings

    df_trades = pd.DataFrame(records)
    if df_trades.empty:
        print("No rebalance periods -- check date range and data availability.")
        return {}

    df_trades["Bench_Value"] = initial_capital * (
        (1 + df_trades["Bench_Return"]).cumprod()
    )

    metrics = _compute_metrics(df_trades, initial_capital, benchmark_name, top_n, worst_rank_held, use_regime_filter, total_trades)
    _print_metrics(metrics)

    BACKTEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_excel(df_trades, metrics, benchmark_name)
    _plot_equity(df_trades, metrics, benchmark_name, top_n, worst_rank_held)
    _append_run_log(metrics)

    return metrics


# ──────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────

def _compute_metrics(df: pd.DataFrame, capital: float, bench_name: str, top_n: int = TOP_N, worst_rank_held: int = WORST_RANK_HELD, regime_filter: bool = False, total_trades: int = 0) -> dict:
    port_rets = df["Port_Return"].values
    bench_rets = df["Bench_Return"].values
    n_weeks = len(port_rets)
    weeks_per_year = 52

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    bench_total = df["Bench_Value"].iloc[-1] / capital - 1
    years = n_weeks / weeks_per_year

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    rolling_max = cum.cummax()
    drawdowns = cum / rolling_max - 1
    max_dd = float(drawdowns.min())

    bench_cum = (1 + pd.Series(bench_rets)).cumprod()
    bench_roll_max = bench_cum.cummax()
    bench_max_dd = float((bench_cum / bench_roll_max - 1).min())

    ann_vol = np.std(port_rets, ddof=1) * np.sqrt(weeks_per_year)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    downside = port_rets[port_rets < 0]
    downside_vol = np.std(downside, ddof=1) * np.sqrt(weeks_per_year) if len(downside) > 1 else 0.0
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    win_rate = np.mean(port_rets > 0) * 100
    avg_weekly = np.mean(port_rets) * 100
    avg_turnover = df["Turnover"].mean() * 100

    alpha = cagr - bench_cagr
    excess = pd.Series(port_rets) - pd.Series(bench_rets)
    te = float(excess.std(ddof=1)) * np.sqrt(weeks_per_year)
    info_ratio = alpha / te if te > 0 else 0.0

    return {
        "Strategy": STRATEGY_TAG,
        "Benchmark": bench_name,
        "Period": f"{df['Rebal_Date'].iloc[0].strftime('%Y-%m-%d')} to {df['End_Date'].iloc[-1].strftime('%Y-%m-%d')}",
        "Top_N": top_n,
        "Worst_Rank_Held": worst_rank_held,
        "Regime_Filter": regime_filter,
        "Weeks": n_weeks,
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(win_rate, 1),
        "Avg_Weekly_Return_%": round(avg_weekly, 2),
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
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS [BASE]")
    print("=" * 55)
    fmt = [
        ("Benchmark",            m["Benchmark"]),
        ("Period",               m["Period"]),
        ("Worst Rank Held",      m.get("Worst_Rank_Held", "N/A")),
        ("Rebalance Weeks",      m["Weeks"]),
        ("",                     ""),
        ("Strategy Total Return", f"{m['Total_Return_%']:+.2f} %"),
        ("Strategy CAGR",        f"{m['CAGR_%']:+.2f} %"),
        ("Max Drawdown",         f"{m['Max_Drawdown_%']:.2f} %"),
        ("Sharpe Ratio",         f"{m['Sharpe']:.2f}"),
        ("Sortino Ratio",        f"{m['Sortino']:.2f}"),
        ("Calmar Ratio",         f"{m['Calmar']:.2f}"),
        ("Win Rate",             f"{m['Win_Rate_%']:.1f} %"),
        ("Avg Weekly Return",    f"{m['Avg_Weekly_Return_%']:+.2f} %"),
        ("Annual Volatility",    f"{m['Ann_Volatility_%']:.2f} %"),
        ("Avg Turnover",         f"{m['Avg_Turnover_%']:.1f} %"),
        ("",                     ""),
        ("Benchmark Total Return", f"{m['Bench_Total_Return_%']:+.2f} %"),
        ("Benchmark CAGR",       f"{m['Bench_CAGR_%']:+.2f} %"),
        ("Benchmark Max DD",     f"{m['Bench_Max_Drawdown_%']:.2f} %"),
        ("",                     ""),
        ("Alpha (ann.)",         f"{m['Alpha_%']:+.2f} %"),
        ("Information Ratio",    f"{m['Information_Ratio']:.2f}"),
        ("",                     ""),
        ("Final Portfolio Value", f"Rs {m['Final_Value']:,.2f}"),
        ("Final Benchmark Value", f"Rs {m['Bench_Final_Value']:,.2f}"),
    ]
    for label, val in fmt:
        if label == "":
            print()
        else:
            print(f"  {label:<26s} {val}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# Excel output
# ──────────────────────────────────────────────

def _write_excel(df_trades: pd.DataFrame, metrics: dict, bench_name: str) -> None:
    out_path = BACKTEST_OUT_DIR / f"backtest_rs_etfs_{bench_name.lower()}.xlsx"
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


# ──────────────────────────────────────────────
# Equity curve chart
# ──────────────────────────────────────────────

def _plot_equity(df_trades: pd.DataFrame, metrics: dict, bench_name: str, top_n: int = TOP_N, worst_rank: int = WORST_RANK_HELD) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    dates = df_trades["End_Date"]
    strat_vals = df_trades["Portfolio_Value"]
    bench_vals = df_trades["Bench_Value"]

    ax1 = axes[0]
    ax1.plot(dates, strat_vals, label="Strategy (base)", linewidth=1.5, color="#1f77b4")
    ax1.plot(dates, bench_vals, label=f"Benchmark ({bench_name})", linewidth=1.2, color="#aaaaaa", linestyle="--")
    ax1.set_ylabel("Portfolio Value (Rs)")
    ax1.set_title(
        f"ETF RS Momentum [Base]  |  Top {top_n} (hold to {worst_rank})  |  "
        f"CAGR {metrics['CAGR_%']:+.1f}%  vs  {metrics['Bench_CAGR_%']:+.1f}%  |  "
        f"Sharpe {metrics['Sharpe']:.2f}  |  MaxDD {metrics['Max_Drawdown_%']:.1f}%",
        fontsize=11,
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    cum_ret = (1 + df_trades["Port_Return"]).cumprod()
    rolling_max = cum_ret.cummax()
    dd = (cum_ret / rolling_max - 1) * 100

    ax2 = axes[1]
    ax2.fill_between(dates, dd, 0, color="#d62728", alpha=0.35)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    png_path = BACKTEST_OUT_DIR / f"backtest_rs_etfs_{bench_name.lower()}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart -> {png_path}")


# ──────────────────────────────────────────────
# Persistent run log
# ──────────────────────────────────────────────

RUN_LOG_PATH = BACKTEST_OUT_DIR / "backtest_run_log.csv"


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


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest ETF RS Momentum Strategy [Base]")
    parser.add_argument("--start", default=BACKTEST_START, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=BACKTEST_END,   help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--benchmark", default=BENCHMARK_NAME,
                        choices=list(BENCHMARKS.keys()),
                        help="Benchmark index for RS calculation")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="Number of ETFs to hold")
    parser.add_argument("--worst-rank", type=int, default=WORST_RANK_HELD,
                        help="Exit buffer: existing holding stays until rank exceeds this (>= top-n)")
    parser.add_argument("--regime-filter", action="store_true", default=USE_REGIME_FILTER,
                        help="Go to cash when Market_Regime = Trend_Down")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")

    args = parser.parse_args()

    run_backtest(
        backtest_start=args.start,
        backtest_end=args.end,
        benchmark_name=args.benchmark,
        top_n=args.top_n,
        worst_rank_held=max(args.worst_rank, args.top_n),
        use_regime_filter=args.regime_filter,
        initial_capital=args.capital,
    )
