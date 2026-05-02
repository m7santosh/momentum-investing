"""
ETF relative strength vs Nifty 50 (^NSEI): excess returns (ETF % − index %) over 1W / 2W / 1M,
ranked on the same horizons. Same universe as momentum_etfs.py (keep lists in sync manually).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

NIFTY50_BENCHMARK = "^NSEI"

# Same universe as momentum_etfs.py
tickers = [
    "ALPHA.NS",
    "AUTOBEES.NS",
    "BANKBEES.NS",
    "CONSUMBEES.NS",
    "CPSEETF.NS",
    "MOENERGY.NS",
    "FMCGIETF.NS",
    "GOLDBEES.NS",
    "GROWWPOWER.NS",
    "GROWWRAIL.NS",
    "HDFCSML250.NS",
    "HEALTHIETF.NS",
    "HNGSNGBEES.NS",
    "ICICIB22.NS",
    "INFRABEES.NS",
    "ITBEES.NS",
    "LIQUIDCASE.NS",
    "MAFANG.NS",
    "MAHKTECH.NS",
    "METALIETF.NS",
    "MIDCAPETF.NS",
    "MOCAPITAL.NS",
    "MODEFENCE.NS",
    "MON100.NS",
    "MOREALTY.NS",
    "MOTOUR.NS",
    "MOVALUE.NS",
    "NEXT50IETF.NS",
    "NIFTYBEES.NS",
    "OILIETF.NS",
    "PHARMABEES.NS",
    "PSUBNKBEES.NS",
    "PVTBANIETF.NS",
    "MOMIDMTM.NS",
    "SILVERBEES.NS",
    "SMALLCAP.NS",
]


def _symbol_for_excel(yahoo_ticker: str) -> str:
    if yahoo_ticker.endswith(".NS"):
        return yahoo_ticker[: -len(".NS")]
    if yahoo_ticker.endswith(".BO"):
        return yahoo_ticker[: -len(".BO")]
    return yahoo_ticker


def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze()


def get_data(ticker: str, start_date, end_date):
    return yf.download(
        ticker,
        start=start_date,
        end=end_date,
        multi_level_index=False,
        auto_adjust=False,
        progress=False,
    )


end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 2)

data = {}
for ticker in tickers:
    try:
        stock_data = get_data(ticker, start_date, end_date)
        if len(stock_data) > 0:
            data[ticker] = stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

nifty_adj: pd.Series | None = None
try:
    nifty_df = get_data(NIFTY50_BENCHMARK, start_date, end_date)
    if len(nifty_df) > 0:
        nifty_adj = _adj_close_series(nifty_df)
except Exception as e:
    print(f"Error: Nifty 50 benchmark {NIFTY50_BENCHMARK} ({e})")
    raise SystemExit(1)

summary = []
for ticker, df in data.items():
    try:
        adj = _adj_close_series(df)
        n = len(adj)
        if n < 21:
            print(f"Skip {ticker}: insufficient history ({n} rows).")
            continue

        return_1m = (adj.iloc[-1] / adj.iloc[-21] - 1) * 100
        return_1w = (adj.iloc[-1] / adj.iloc[-6] - 1) * 100
        return_2w = (adj.iloc[-1] / adj.iloc[-11] - 1) * 100

        rs_1w = rs_2w = rs_1m = float("nan")
        if len(nifty_adj) >= 21:
            nx = nifty_adj.reindex(adj.index).ffill()
            if not nx.iloc[-21:].isna().any() and (nx.iloc[-21:] > 0).all():
                ret_n_1w = (nx.iloc[-1] / nx.iloc[-6] - 1) * 100
                ret_n_2w = (nx.iloc[-1] / nx.iloc[-11] - 1) * 100
                ret_n_1m = (nx.iloc[-1] / nx.iloc[-21] - 1) * 100
                rs_1w = return_1w - ret_n_1w
                rs_2w = return_2w - ret_n_2w
                rs_1m = return_1m - ret_n_1m

        summary.append(
            {
                "Symbol": _symbol_for_excel(ticker),
                "Return_1W": return_1w,
                "Return_2W": return_2w,
                "Return_1M": return_1m,
                "RS_1W_vs_N50": rs_1w,
                "RS_2W_vs_N50": rs_2w,
                "RS_1M_vs_N50": rs_1m,
            }
        )
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

df_summary = pd.DataFrame(summary)
if df_summary.empty:
    print("No rows; no Excel file written.")
    raise SystemExit(0)

for c in ("Return_1W", "Return_2W", "Return_1M", "RS_1W_vs_N50", "RS_2W_vs_N50", "RS_1M_vs_N50"):
    df_summary[c] = df_summary[c].round(1)

for c in ("Return_1W", "Return_2W", "Return_1M"):
    df_summary[f"Rank_{c.replace('Return_', '')}"] = df_summary[c].rank(ascending=False)

for c, rc in (
    ("RS_1W_vs_N50", "Rank_RS_1W"),
    ("RS_2W_vs_N50", "Rank_RS_2W"),
    ("RS_1M_vs_N50", "Rank_RS_1M"),
):
    df_summary[rc] = df_summary[c].rank(ascending=False, na_option="bottom")

df_summary["Final_RS_Rank"] = (
    0.3 * df_summary["Rank_RS_1W"]
    + 0.3 * df_summary["Rank_RS_2W"]
    + 0.4 * df_summary["Rank_RS_1M"]
)

df_out = df_summary.sort_values("Final_RS_Rank").head(10).reset_index(drop=True)
df_out["Position"] = np.arange(1, len(df_out) + 1)

out_path = Path(__file__).resolve().parent / "momentum_rs_etfs_ranked.xlsx"
try:
    df_out.to_excel(out_path, index=False, engine="openpyxl")
except ImportError:
    print("Missing dependency: pip install openpyxl")
    raise
print(f"Wrote {len(df_out)} rows -> {out_path}")
