"""
ETF relative strength vs ^CRSLDX (companion to momentum_rs_etfs.py).

Same universe and entry filters as momentum_rs_etfs.py (history, trend gate, new-listing liquidity).
No extra RS cutoffs — each ETF is compared to N500 on 1W / 2W / 1M / 3M; ranking uses **1W / 2W / 1M
RS only** (3M omitted for tactical rotation). Weighted_RS_pct = W_RANK on those horizons (highest first).
Return_3M and RS_3M_vs_N500 are still shown in Excel for context.

Output: final_result/etf/momentum_rs_etfs_adaptive.xlsx (Leaders + Run_Info).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_ETF_DIR

_base_path = Path(__file__).resolve().parent / "momentum_rs_etfs.py"
_spec = importlib.util.spec_from_file_location("momentum_rs_etfs_base", _base_path)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

OUT_FILENAME = "momentum_rs_etfs_adaptive.xlsx"
TOP_N = base.TOP_N

# Ranking weights on RS vs N500: 1W / 2W / 1M only (sum = 1; 3M excluded)
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
    # "Rank_vs_Peak",
    # "Volatility_Score",
    "Weighted_RS_pct",
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
    "RS_1W_vs_N500",
    "RS_2W_vs_N500",
    "RS_1M_vs_N500",
    # "RS_3M_vs_N500",
]


def _benchmark_return_1m(nifty_adj: pd.Series) -> float:
    if len(nifty_adj) < base.LB_1M:
        return float("nan")
    return float((nifty_adj.iloc[-1] / nifty_adj.iloc[-base.LB_1M] - 1) * 100)


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
    for sym in base.tickers:
        try:
            df = base.get_data(sym, start_date, end_date)
            if len(df) == 0:
                continue

            adj = base._adj_close_series(df).dropna()
            if len(adj) < base.MIN_HISTORY_SESSIONS:
                continue

            if len(adj) >= base.LB_52W:
                if not base._passes_established_trend_gate(adj):
                    continue
            else:
                vol = base._volume_series(df)
                if base._avg_adtv_crores(adj, vol) < base.MIN_ADTV_NEW_ETF_CRORES:
                    continue

            high_52w = adj.iloc[-min(base.LB_52W, len(adj)) :].max()
            last = adj.iloc[-1]

            close_on_adj_index = base._close_series(df).reindex(adj.index).ffill().bfill()
            ema9_close = float(
                close_on_adj_index.ewm(span=base.ETF_EMA_9, adjust=False).mean().iloc[-1]
            )
            last_close = float(close_on_adj_index.iloc[-1])
            close_below_9ema = "Exit" if last_close < ema9_close else "Hold"

            high_ath = float(adj.max())
            ratio_52w = last / high_52w
            ratio_ath = last / high_ath if high_ath > 0 else float("nan")
            peak_proximity_score = (ratio_52w + ratio_ath) / 2.0

            return_1w = (adj.iloc[-1] / adj.iloc[-base.LB_1W] - 1) * 100
            return_2w = (adj.iloc[-1] / adj.iloc[-base.LB_2W] - 1) * 100
            return_1m = (adj.iloc[-1] / adj.iloc[-base.LB_1M] - 1) * 100
            return_3m = (adj.iloc[-1] / adj.iloc[-base.LB_3M] - 1) * 100

            daily_returns = adj.pct_change()
            vol_score = daily_returns.tail(base.LB_1M).std() * 100

            rs_1w = rs_2w = rs_1m = rs_3m = float("nan")
            if len(nifty_adj) >= base.LB_3M:
                nx = nifty_adj.reindex(adj.index).ffill()
                tail = nx.iloc[-base.LB_3M :]
                if not tail.isna().any() and (tail > 0).all():
                    ret_n_1w = (nx.iloc[-1] / nx.iloc[-base.LB_1W] - 1) * 100
                    ret_n_2w = (nx.iloc[-1] / nx.iloc[-base.LB_2W] - 1) * 100
                    ret_n_1m = (nx.iloc[-1] / nx.iloc[-base.LB_1M] - 1) * 100
                    ret_n_3m = (nx.iloc[-1] / nx.iloc[-base.LB_3M] - 1) * 100
                    rs_1w = return_1w - ret_n_1w
                    rs_2w = return_2w - ret_n_2w
                    rs_1m = return_1m - ret_n_1m
                    rs_3m = return_3m - ret_n_3m

            summary.append(
                {
                    "Symbol": base._symbol_for_excel(sym),
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
        nifty_df = base.get_data(base.BENCHMARK_TICKER, start_date, end_date)
        if len(nifty_df) == 0:
            print(f"Error: No rows for benchmark {base.BENCHMARK_TICKER}")
            return
        nifty_adj = base._adj_close_series(nifty_df).dropna()
    except Exception as e:
        print(f"Error: Benchmark {base.BENCHMARK_TICKER} ({e})")
        return

    market_regime = base.classify_ema_regime(
        nifty_adj, base.BENCH_EMA_FAST, base.BENCH_EMA_SLOW
    )
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
                "Entry_Filters": "Same as momentum_rs_etfs.py (trend/liquidity/history)",
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
