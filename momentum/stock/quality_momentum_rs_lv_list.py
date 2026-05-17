"""
Quality Momentum RS + low-volatility — ranked stock list only (standalone).

Run any day for a fresh ranked list. No rebalance history or portfolio state.

Filters: 200 EMA, 52w-high proximity, ADTV > 5 Cr, low-vol quantile, blended momentum/RS vs ^CRSLDX.
Universe: Nifty Quality 30 + Midcap Quality 50 + Smallcap Quality 50.

Examples:
    python momentum/stock/quality_momentum_rs_lv_list.py
    python momentum/stock/quality_momentum_rs_lv_list.py --as-of 2026-05-15 --limit 50
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_STOCK_DIR
from utils.nse_bhavcopy import fetch_bhavcopy, fetch_nse_live_quotes, nse_symbol_from_yahoo, today_ist

# --- Configuration ---
BENCHMARK_TICKER = "^CRSLDX"
MIN_ADTV_CRORES = 5.0
PROXIMITY_OF_52W_HIGH = 0.70
_RUNTIME_52W_PROXIMITY: float | None = None
LOW_VOLATILITY_MAX_QUANTILE = 0.67
LOW_VOLATILITY_MIN_UNIVERSE = 8
LIST_SIZE = 30
RUN_AS_OF: date | str | None = None
OUTPUT_FILENAME = "quality_momentum_rs_lv_list.xlsx"

LB_1M = 21
LB_3M = 63
LB_6M = 126
LB_9M = 189
W_3M, W_6M, W_9M = 0.50, 0.30, 0.20


def _coerce_run_as_of_config(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        return datetime.strptime(s, "%Y-%m-%d").date()
    raise TypeError(
        f"RUN_AS_OF must be None, date, datetime, or YYYY-MM-DD str, got {type(value).__name__}"
    )


tickers = [
    {"symbol": "ABB.NS", "industry": "Capital Goods", "marketcap": "Largecap"},
    {"symbol": "ASIANPAINT.NS", "industry": "Consumer Durables", "marketcap": "Largecap"},
    {"symbol": "BAJAJ-AUTO.NS", "industry": "Automobile and Auto Components", "marketcap": "Largecap"},
    {"symbol": "BEL.NS", "industry": "Capital Goods", "marketcap": "Largecap"},
    {"symbol": "BOSCHLTD.NS", "industry": "Automobile and Auto Components", "marketcap": "Largecap"},
    {"symbol": "BRITANNIA.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "COALINDIA.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Largecap"},
    {"symbol": "DIVISLAB.NS", "industry": "Healthcare", "marketcap": "Largecap"},
    {"symbol": "DRREDDY.NS", "industry": "Healthcare", "marketcap": "Largecap"},
    {"symbol": "EICHERMOT.NS", "industry": "Automobile and Auto Components", "marketcap": "Largecap"},
    {"symbol": "GODREJCP.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "HCLTECH.NS", "industry": "Information Technology", "marketcap": "Largecap"},
    {"symbol": "HAVELLS.NS", "industry": "Consumer Durables", "marketcap": "Largecap"},
    {"symbol": "HAL.NS", "industry": "Capital Goods", "marketcap": "Largecap"},
    {"symbol": "HINDUNILVR.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "HINDZINC.NS", "industry": "Metals & Mining", "marketcap": "Largecap"},
    {"symbol": "ITC.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "INFY.NS", "industry": "Information Technology", "marketcap": "Largecap"},
    {"symbol": "LTM.NS", "industry": "Information Technology", "marketcap": "Largecap"},
    {"symbol": "MARUTI.NS", "industry": "Automobile and Auto Components", "marketcap": "Largecap"},
    {"symbol": "MAZDOCK.NS", "industry": "Capital Goods", "marketcap": "Largecap"},
    {"symbol": "NESTLEIND.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "PIDILITIND.NS", "industry": "Chemicals", "marketcap": "Largecap"},
    {"symbol": "SOLARINDS.NS", "industry": "Chemicals", "marketcap": "Largecap"},
    {"symbol": "TCS.NS", "industry": "Information Technology", "marketcap": "Largecap"},
    {"symbol": "TECHM.NS", "industry": "Information Technology", "marketcap": "Largecap"},
    {"symbol": "UNITDSPR.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "VBL.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Largecap"},
    {"symbol": "WIPRO.NS", "industry": "Information Technology", "marketcap": "Largecap"},
    {"symbol": "ZYDUSLIFE.NS", "industry": "Healthcare", "marketcap": "Largecap"},

    {"symbol": "360ONE.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "3MINDIA.NS", "industry": "Diversified", "marketcap": "Midcap"},
    {"symbol": "AIAENG.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "APLAPOLLO.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "ABBOTINDIA.NS", "industry": "Healthcare", "marketcap": "Midcap"},
    {"symbol": "AJANTPHARM.NS", "industry": "Healthcare", "marketcap": "Midcap"},
    {"symbol": "ALKEM.NS", "industry": "Healthcare", "marketcap": "Midcap"},
    {"symbol": "APARINDS.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "ASTRAL.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "BALKRISIND.NS", "industry": "Automobile and Auto Components", "marketcap": "Midcap"},
    {"symbol": "MAHABANK.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "BERGEPAINT.NS", "industry": "Consumer Durables", "marketcap": "Midcap"},
    {"symbol": "BDL.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "CRISIL.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "COFORGE.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "COLPAL.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Midcap"},
    {"symbol": "COROMANDEL.NS", "industry": "Chemicals", "marketcap": "Midcap"},
    {"symbol": "CUMMINSIND.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "DIXON.NS", "industry": "Consumer Durables", "marketcap": "Midcap"},
    {"symbol": "GLAXO.NS", "industry": "Healthcare", "marketcap": "Midcap"},
    {"symbol": "GODFRYPHLP.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Midcap"},
    {"symbol": "GUJGASLTD.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Midcap"},
    {"symbol": "HDFCAMC.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "HEROMOTOCO.NS", "industry": "Automobile and Auto Components", "marketcap": "Midcap"},
    {"symbol": "HONAUT.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "IRCTC.NS", "industry": "Consumer Services", "marketcap": "Midcap"},
    {"symbol": "IGL.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Midcap"},
    {"symbol": "KPRMILL.NS", "industry": "Textiles", "marketcap": "Midcap"},
    {"symbol": "KEI.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "KPITTECH.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "LTTS.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "MARICO.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Midcap"},
    {"symbol": "MOTILALOFS.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "MPHASIS.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "MUTHOOTFIN.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "NMDC.NS", "industry": "Metals & Mining", "marketcap": "Midcap"},
    {"symbol": "NAM-INDIA.NS", "industry": "Financial Services", "marketcap": "Midcap"},
    {"symbol": "OFSS.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "PIIND.NS", "industry": "Chemicals", "marketcap": "Midcap"},
    {"symbol": "PAGEIND.NS", "industry": "Textiles", "marketcap": "Midcap"},
    {"symbol": "PERSISTENT.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "PETRONET.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Midcap"},
    {"symbol": "POLYCAB.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "PGHH.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Midcap"},
    {"symbol": "SCHAEFFLER.NS", "industry": "Automobile and Auto Components", "marketcap": "Midcap"},
    {"symbol": "SONACOMS.NS", "industry": "Automobile and Auto Components", "marketcap": "Midcap"},
    {"symbol": "SUPREMEIND.NS", "industry": "Capital Goods", "marketcap": "Midcap"},
    {"symbol": "SYNGENE.NS", "industry": "Healthcare", "marketcap": "Midcap"},
    {"symbol": "TATAELXSI.NS", "industry": "Information Technology", "marketcap": "Midcap"},
    {"symbol": "TIINDIA.NS", "industry": "Automobile and Auto Components", "marketcap": "Midcap"},

    {"symbol": "ACE.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "ABSLAMC.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "AFFLE.NS", "industry": "Information Technology", "marketcap": "Smallcap"},
    {"symbol": "ARE&M.NS", "industry": "Automobile and Auto Components", "marketcap": "Smallcap"},
    {"symbol": "ANGELONE.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "APTUS.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "BLS.NS", "industry": "Consumer Services", "marketcap": "Smallcap"},
    {"symbol": "BAYERCROP.NS", "industry": "Chemicals", "marketcap": "Smallcap"},
    {"symbol": "BSOFT.NS", "industry": "Information Technology", "marketcap": "Smallcap"},
    {"symbol": "MAPMYINDIA.NS", "industry": "Information Technology", "marketcap": "Smallcap"},
    {"symbol": "CANFINHOME.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "CASTROLIND.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Smallcap"},
    {"symbol": "CDSL.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "CHAMBLFERT.NS", "industry": "Chemicals", "marketcap": "Smallcap"},
    {"symbol": "CLEAN.NS", "industry": "Chemicals", "marketcap": "Smallcap"},
    {"symbol": "CAMS.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "CYIENT.NS", "industry": "Information Technology", "marketcap": "Smallcap"},
    {"symbol": "LALPATHLAB.NS", "industry": "Healthcare", "marketcap": "Smallcap"},
    {"symbol": "ELGIEQUIP.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "EMAMILTD.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Smallcap"},
    {"symbol": "ENGINERSIN.NS", "industry": "Construction", "marketcap": "Smallcap"},
    {"symbol": "FINCABLES.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "GILLETTE.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Smallcap"},
    {"symbol": "GPIL.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "GRAVITA.NS", "industry": "Metals & Mining", "marketcap": "Smallcap"},
    {"symbol": "GSPL.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Smallcap"},
    {"symbol": "INDIAMART.NS", "industry": "Consumer Services", "marketcap": "Smallcap"},
    {"symbol": "IEX.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "JBCHEPHARM.NS", "industry": "Healthcare", "marketcap": "Smallcap"},
    {"symbol": "JSWDULUX.NS", "industry": "Consumer Durables", "marketcap": "Smallcap"},
    {"symbol": "KAJARIACER.NS", "industry": "Consumer Durables", "marketcap": "Smallcap"},
    {"symbol": "KARURVYSYA.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "KIRLOSBROS.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "LTFOODS.NS", "industry": "Fast Moving Consumer Goods", "marketcap": "Smallcap"},
    {"symbol": "MGL.NS", "industry": "Oil Gas & Consumable Fuels", "marketcap": "Smallcap"},
    {"symbol": "METROPOLIS.NS", "industry": "Healthcare", "marketcap": "Smallcap"},
    {"symbol": "MSUMI.NS", "industry": "Automobile and Auto Components", "marketcap": "Smallcap"},
    {"symbol": "PFIZER.NS", "industry": "Healthcare", "marketcap": "Smallcap"},
    {"symbol": "POLYMED.NS", "industry": "Healthcare", "marketcap": "Smallcap"},
    {"symbol": "PRAJIND.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "RITES.NS", "industry": "Construction", "marketcap": "Smallcap"},
    {"symbol": "RAILTEL.NS", "industry": "Telecommunication", "marketcap": "Smallcap"},
    {"symbol": "SONATSOFTW.NS", "industry": "Information Technology", "marketcap": "Smallcap"},
    {"symbol": "SUMICHEM.NS", "industry": "Chemicals", "marketcap": "Smallcap"},
    {"symbol": "SUNTV.NS", "industry": "Media Entertainment & Publication", "marketcap": "Smallcap"},
    {"symbol": "TIMKEN.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "TRITURBINE.NS", "industry": "Capital Goods", "marketcap": "Smallcap"},
    {"symbol": "UTIAMC.NS", "industry": "Financial Services", "marketcap": "Smallcap"},
    {"symbol": "ZENSARTECH.NS", "industry": "Information Technology", "marketcap": "Smallcap"},
    {"symbol": "ECLERX.NS", "industry": "Services", "marketcap": "Smallcap"},
]

def _symbol_for_excel(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace(".NS", "").replace(".BO", "")


_EXCEL_SYMBOL_TO_YAHOO: dict[str, str] = {
    _symbol_for_excel(t["symbol"]): t["symbol"] for t in tickers
}


def _adj_close_series(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _raw_close_series(df: pd.DataFrame) -> pd.Series:
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
    df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False, progress=False)
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


def _last_bar_adj_close_for_session(
    sym_excel: str, session_as_of: date
) -> tuple[float | None, float | None]:
    """Last session bar Adj Close / Close for a Yahoo ticker (not from ranked df)."""
    yh = _EXCEL_SYMBOL_TO_YAHOO.get(sym_excel)
    if not yh:
        return None, None
    end_dt = datetime.combine(session_as_of, datetime.min.time()) + timedelta(days=1)
    start_dt = end_dt - timedelta(days=120)
    try:
        df = get_data(yh, start_dt, end_dt)
        if df is None or len(df) == 0:
            return None, None
        adj = _adj_close_series(df)
        raw = _raw_close_series(df)
        la = adj.iloc[-1]
        lc = raw.iloc[-1]
        return (
            round(float(la), 2) if pd.notna(la) else None,
            round(float(lc), 2) if pd.notna(lc) else None,
        )
    except Exception:
        return None, None


def _current_price_fallback_for_symbols(
    symbols: list[str],
    session_as_of: date,
    symbols_in_ranked: set[str],
) -> dict[str, tuple[float | None, float | None]]:
    """For prior holdings not in this run's df_ranked: last-bar prices for rebalance sheet."""
    out: dict[str, tuple[float | None, float | None]] = {}
    for sym in symbols:
        if sym in symbols_in_ranked:
            continue
        a, c = _last_bar_adj_close_for_session(sym, session_as_of)
        if a is not None or c is not None:
            out[sym] = (a, c)
    return out


def _pct_change_vs_baseline(
    baseline: float | None, current: float | None
) -> float | None:
    """Percent change from baseline session to current: (current / baseline - 1) * 100."""
    if baseline is None or current is None:
        return None
    try:
        b = float(baseline)
        c = float(current)
    except (TypeError, ValueError):
        return None
    if b == 0 or not np.isfinite(b) or not np.isfinite(c):
        return None
    return round((c / b - 1.0) * 100.0, 2)


def _effective_low_vol_max_quantile() -> float:
    raw = (os.environ.get("QUALITY_RS_LV_MAX_QUANTILE") or "").strip()
    if not raw:
        return float(LOW_VOLATILITY_MAX_QUANTILE)
    v = float(raw)
    if not (0 < v <= 1):
        raise ValueError(
            "QUALITY_RS_LV_MAX_QUANTILE must be a float in (0, 1], got " + repr(raw)
        )
    return v


def _effective_proximity_of_52w_high() -> float:
    if _RUNTIME_52W_PROXIMITY is not None:
        return _RUNTIME_52W_PROXIMITY
    raw = (os.environ.get("QUALITY_RS_52W_PROXIMITY") or "").strip()
    if not raw:
        return float(PROXIMITY_OF_52W_HIGH)
    v = float(raw)
    if not (0 < v <= 1):
        raise ValueError(
            "QUALITY_RS_52W_PROXIMITY must be a float in (0, 1], got " + repr(raw)
        )
    return v


def _set_runtime_52w_proximity(value: float | None) -> None:
    global _RUNTIME_52W_PROXIMITY
    if value is not None and not (0 < value <= 1):
        raise ValueError(f"52w proximity must be in (0, 1], got {value}")
    _RUNTIME_52W_PROXIMITY = value


def _apply_low_volatility_filter(
    df_summary: pd.DataFrame, *, quiet: bool
) -> pd.DataFrame:
    """Drop high-volatility names using cross-sectional quantile of Volatility_Score."""
    if df_summary.empty:
        return df_summary
    vs = df_summary["Volatility_Score"]
    valid = vs.dropna()
    n0 = len(df_summary)
    if len(valid) < LOW_VOLATILITY_MIN_UNIVERSE:
        if not quiet:
            print(
                f"Low-vol filter skipped: only {len(valid)} valid Volatility_Score rows "
                f"(need >= {LOW_VOLATILITY_MIN_UNIVERSE})."
            )
        return df_summary
    q = _effective_low_vol_max_quantile()
    cap = float(valid.quantile(q))
    out = df_summary.loc[vs.notna() & (vs <= cap)].copy()
    if not quiet:
        print(
            f"Low-vol filter: quantile={q:.3f}, cutoff Volatility_Score<={cap:.4f} "
            f"→ {len(out)}/{n0} names."
        )
    return out


def _apply_ranking_engine(df_summary: pd.DataFrame) -> pd.DataFrame:
    df = df_summary.copy()
    for c in ["3M", "6M", "9M"]:
        df[f"Rank_{c}"] = df[f"Return_{c}"].rank(ascending=False)
    for c in ["3M", "6M", "9M"]:
        df[f"Rank_RS_{c}"] = df[f"RS_{c}_vs_Bench"].rank(ascending=False, na_option="bottom")
    df["Abs_Momentum_Rank"] = (
        W_3M * df["Rank_3M"] + W_6M * df["Rank_6M"] + W_9M * df["Rank_9M"]
    ).rank()
    df["Relative_Strength_Rank"] = (
        W_3M * df["Rank_RS_3M"] + W_6M * df["Rank_RS_6M"] + W_9M * df["Rank_RS_9M"]
    ).rank()
    df["Blended_Rank"] = (df["Abs_Momentum_Rank"] + df["Relative_Strength_Rank"]) / 2
    out = df.sort_values("Blended_Rank").reset_index(drop=True)
    out["Rank_Position"] = np.arange(1, len(out) + 1)
    return out


def compute_ranked_universe(session_as_of: date, *, quiet: bool = False) -> pd.DataFrame | None:
    """Full download + filters + ranks for one session date (yfinance bars through session_as_of)."""
    # end exclusive: last bar is session_as_of, not the calendar day of end_date.
    end_date = datetime.combine(session_as_of, datetime.min.time()) + timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 2)
    try:
        nifty_df = get_data(BENCHMARK_TICKER, start_date, end_date)
        nifty_adj = _adj_close_series(nifty_df)
    except Exception as e:
        if not quiet:
            print(f"Error: Benchmark {BENCHMARK_TICKER} ({e})")
        return None

    summary: list[dict] = []
    industry_by_symbol = {t["symbol"]: t["industry"] for t in tickers}
    marketcap_by_symbol = {t["symbol"]: t["marketcap"] for t in tickers}

    for t in tickers:
        sym = t["symbol"]
        try:
            df = get_data(sym, start_date, end_date)
            if len(df) < LB_9M:
                continue

            adj = _adj_close_series(df)
            vol = df["Volume"]

            daily_turnover = adj * vol
            adtv_crores = (daily_turnover.tail(20).mean()) / 10000000
            if adtv_crores < MIN_ADTV_CRORES:
                continue

            ema200 = adj.ewm(span=200).mean().iloc[-1]
            high_52w = adj.iloc[-min(252, len(adj)) :].max()

            prox = _effective_proximity_of_52w_high()
            if adj.iloc[-1] < ema200 or adj.iloc[-1] < (high_52w * prox):
                continue

            ret_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
            ret_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100
            ret_6m = (adj.iloc[-1] / adj.iloc[-LB_6M] - 1) * 100
            ret_9m = (adj.iloc[-1] / adj.iloc[-LB_9M] - 1) * 100

            vol_score = adj.pct_change().tail(21).std() * 100

            raw_close = _raw_close_series(df)
            last_adj_close = float(adj.iloc[-1])
            last_close = float(raw_close.iloc[-1])

            nx = nifty_adj.reindex(adj.index).ffill()
            rs_3m = ret_3m - ((nx.iloc[-1] / nx.iloc[-LB_3M] - 1) * 100)
            rs_6m = ret_6m - ((nx.iloc[-1] / nx.iloc[-LB_6M] - 1) * 100)
            rs_9m = ret_9m - ((nx.iloc[-1] / nx.iloc[-LB_9M] - 1) * 100)

            summary.append(
                {
                    "Symbol": _symbol_for_excel(sym),
                    "Industry": industry_by_symbol.get(sym, ""),
                    "Marketcap": marketcap_by_symbol.get(sym, ""),
                    "Adj_Close": last_adj_close,
                    "Close": last_close,
                    "ADTV_Cr": adtv_crores,
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
        except Exception as e:
            if not quiet:
                print(f"Error analyzing {sym}: {e}")

    df_summary = pd.DataFrame(summary)
    if df_summary.empty:
        return None
    df_summary = _apply_low_volatility_filter(df_summary, quiet=quiet)
    if df_summary.empty:
        return None
    return _apply_ranking_engine(df_summary)


DISPLAY_COLS = [
    "Position",
    "Symbol",
    "Industry",
    "Marketcap",
    "Adj_Close",
    "Close",
    "ADTV_Cr",
    "Volatility_Score",
    "Return_1M",
    "Return_3M",
    "Return_6M",
    "Return_9M",
]
ROUND_COLS = [
    "ADTV_Cr",
    "Blended_Rank",
    "Volatility_Score",
    "Adj_Close",
    "Close",
    "Return_1M",
    "Return_3M",
    "Return_6M",
    "Return_9M",
]


def _resolve_as_of(as_of: date | None) -> date:
    if as_of is not None:
        return as_of
    file_as_of = _coerce_run_as_of_config(RUN_AS_OF)
    if file_as_of is not None:
        return file_as_of
    env_raw = (os.environ.get("QUALITY_RS_RUN_AS_OF") or "").strip()
    if env_raw:
        return datetime.strptime(env_raw, "%Y-%m-%d").date()
    return datetime.now().date()


def _format_output(df_ranked: pd.DataFrame, list_size: int) -> pd.DataFrame:
    df = df_ranked.head(list_size).copy()
    df.insert(0, "Position", np.arange(1, len(df) + 1))
    df.drop(columns=["Rank_Position"], inplace=True, errors="ignore")
    for c in ROUND_COLS:
        if c in df.columns:
            df[c] = df[c].round(2)
    return df[[c for c in DISPLAY_COLS if c in df.columns]]


def main(
    *,
    as_of: date | None = None,
    list_size: int = LIST_SIZE,
    proximity_52w: float | None = None,
) -> None:
    _set_runtime_52w_proximity(proximity_52w)
    as_of_day = _resolve_as_of(as_of)
    list_size = max(1, int(list_size))

    print("Quality momentum RS+LV — ranked list")
    print(
        f"Session as_of: {as_of_day}  |  Benchmark: {BENCHMARK_TICKER}  |  "
        f"52w proximity: {_effective_proximity_of_52w_high():.2f}"
    )
    print(f"Showing top {list_size} of ranked universe\n")

    df_ranked = compute_ranked_universe(as_of_day, quiet=False)
    if df_ranked is None or len(df_ranked) == 0:
        print("No stocks passed the trend, liquidity, and low-volatility filters.")
        return

    df_out = _format_output(df_ranked, list_size)
    print(f"Universe after filters: {len(df_ranked)} names\n")
    print(df_out.to_string(index=False))

    FINAL_RESULT_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_STOCK_DIR / OUTPUT_FILENAME
    df_out.to_excel(out_path, sheet_name="Ranked", index=False)
    print(f"\nWrote {len(df_out)} rows → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality momentum RS+LV: ranked stock list (standalone).",
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        default=None,
        help="Session date (bars through this day). Default: today.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIST_SIZE,
        help=f"Number of ranked rows (default LIST_SIZE={LIST_SIZE})",
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
    resolved_as_of = (
        datetime.strptime(args.as_of.strip(), "%Y-%m-%d").date() if args.as_of else None
    )
    main(as_of=resolved_as_of, list_size=args.limit, proximity_52w=args.proximity_52w)
