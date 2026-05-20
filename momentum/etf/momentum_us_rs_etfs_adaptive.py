"""
US ETF relative strength vs ^GSPC (companion to momentum_us_rs_etfs.py).

Same universe and entry filters as momentum_us_rs_etfs.py (history, 200 EMA, 52w proximity).
No extra RS cutoffs — each ETF is compared to S&P 500 on 1W / 2W / 1M / 3M; ranking uses
**1W / 2W / 1M RS only** (3M omitted for tactical rotation). Weighted_RS_pct = W_RANK on
those horizons (highest first). Return_3M and RS_3M_vs_SP500 are still shown for context.

Output: final_result/etf/momentum_us_rs_etfs_adaptive.xlsx (Leaders + Run_Info).
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

_base_path = Path(__file__).resolve().parent / "momentum_us_rs_etfs.py"
_spec = importlib.util.spec_from_file_location("momentum_us_rs_etfs_base", _base_path)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

OUT_FILENAME = "momentum_us_rs_etfs_adaptive.xlsx"
TOP_N = base.TOP_N

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


def _benchmark_return_1m(bench_adj: pd.Series) -> float:
    if len(bench_adj) < base.LB_1M:
        return float("nan")
    return float((bench_adj.iloc[-1] / bench_adj.iloc[-base.LB_1M] - 1) * 100)


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
        sp500_df = base.get_data(base.BENCHMARK_TICKER, start_date, end_date)
        if len(sp500_df) == 0:
            print(f"Error: No rows for benchmark {base.BENCHMARK_TICKER}")
            return
        sp500_adj = base._adj_close_series(sp500_df).dropna()
    except Exception as e:
        print(f"Error: Benchmark {base.BENCHMARK_TICKER} ({e})")
        return

    market_regime = base.classify_ema_regime(
        sp500_adj, base.BENCH_EMA_FAST, base.BENCH_EMA_SLOW
    )
    benchmark_1m = _benchmark_return_1m(sp500_adj)

    summary = base.collect_us_etf_rows(sp500_adj, start_date, end_date)
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
                "Entry_Filters": "Same as momentum_us_rs_etfs.py (200 EMA + 52w proximity)",
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
