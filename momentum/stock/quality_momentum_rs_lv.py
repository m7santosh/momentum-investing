"""
Stock relative strength vs Nifty 500 TR (^CRSLDX). CLI/env/file: session date + rebalance period. Calendar rebalance uses a deterministic context JSON under portfolio_state_archive when rebalance session date + period + sizing params match; otherwise reranks baseline (as_of−N). each_run uses latest state.json only. Weekly_Rebalance + Within_exit_rank_band vs EXIT_RANK_THRESHOLD.

Filters:
1. Trend: Price must be above 200-day EMA.
2. Proximity: Price must be within 30% of its 52-week high.
3. Liquidity: Average Daily Turnover (ADTV) must be > 5 Crores INR.
4. Low volatility: Among names passing 1–3, keep those with Volatility_Score at or below the
   LOW_VOLATILITY_MAX_QUANTILE cross-sectional cutoff (21d adj-close daily return stdev %).
   Override: env QUALITY_RS_LV_MAX_QUANTILE in (0, 1] (e.g. 0.5 = bottom half only).

#### SAME AS quality_momentum_rs_lv_n500.py, BUT WITH ONLY QUALITY 30, 50, 50 AS UNIVERSE. ####

Blended Ranking Logic:
- Abs_Momentum_Rank: Weighted rank on raw returns (0.50·3M + 0.30·6M + 0.20·9M).
- Relative_Strength_Rank: Weighted rank on RS vs Benchmark (0.50·3M + 0.30·6M + 0.20·9M).
- Blended_Rank: Average of the above two. Lower is better.
- Volatility_Score: Standard deviation of last 21 days (lower = smoother trend).
"""

import argparse
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import sys
from pathlib import Path

# Setup project root for utility imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.output_paths import FINAL_RESULT_DIR, FINAL_RESULT_STOCK_DIR
from utils.nse_bhavcopy import fetch_bhavcopy, fetch_nse_live_quotes, nse_symbol_from_yahoo, today_ist

# --- Configuration ---
BENCHMARK_TICKER = "^CRSLDX"
MIN_ADTV_CRORES = 5.0  # Minimum 5 Crores daily trading volume
# After trend/liquidity: keep symbols with Volatility_Score <= universe quantile (lower vol = calmer tape).
LOW_VOLATILITY_MAX_QUANTILE = 0.67
LOW_VOLATILITY_MIN_UNIVERSE = 8  # skip vol filter if fewer names (stable quantile needs breadth)
PORTFOLIO_SIZE = 20  # Holdings size: mix summaries (Marketcap / Industry) use top N by Blended_Rank only
OUTPUT_RANKED_SIZE = 30  # Rows in Excel Sheet1: extend past portfolio to spot weaker names before rebalance
EXIT_RANK_THRESHOLD = 30  # Exit monitor: previous holding with universe rank above this → review / exit line
PORTFOLIO_STATE_FILENAME = "quality_momentum_portfolio_state.json"  # Last run top PORTFOLIO_SIZE for each_run IN/OUT
# Optional: deterministic rebalance context JSON (weekly|biweekly|monthly) keyed by session date + params; overwrites same context.
PORTFOLIO_STATE_ARCHIVE_DIR: Path | None = FINAL_RESULT_DIR / "portfolio_state_archive"
# Rebalance IN/OUT baseline (file defaults). Override: `python ... --rebalance monthly` or env QUALITY_RS_REBALANCE.
#   each_run → last run JSON only. weekly|biweekly|monthly → context cache or rerank as-of session−7/14/30d.
REBALANCE_COMPARE_PERIOD = "biweekly"
# Session date (last bar you want): None | date | "YYYY-MM-DD". yfinance `end` is exclusive, so code passes end = day+1 00:00.
# RUN_AS_OF: date | str | None = '2026-05-15'
RUN_AS_OF: date | str | None = None


def _coerce_run_as_of_config(value: object) -> date | None:
    """Normalize RUN_AS_OF from date, datetime, or YYYY-MM-DD string."""
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
    raise TypeError(f"RUN_AS_OF must be None, date, datetime, or YYYY-MM-DD str, got {type(value).__name__}")


def _parse_session_date_label(value: object) -> date | None:
    """Parse ``run_as_of`` from saved JSON (YYYY-MM-DD or ISO datetime prefix)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if "T" in s:
        try:
            return datetime.fromisoformat(s.replace("Z", "")).date()
        except ValueError:
            return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# Lookback periods (Sessions)
LB_1M = 21
LB_3M = 63
LB_6M = 126
LB_9M = 189

# Weights for Ranking (Focusing on the 3M trend for stocks)
W_3M, W_6M, W_9M = 0.50, 0.30, 0.20

# --- Ticker Universe: Nifty 100 Quality 30, Midcap 150 Quality 50, Smallcap 250 Quality 50 ---
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

# --- Helper Functions ---

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


def _compute_ranked_universe_for_session(session_as_of: date, *, quiet: bool = False) -> pd.DataFrame | None:
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

            if adj.iloc[-1] < ema200 or adj.iloc[-1] < (high_52w * 0.7):
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


def portfolio_mix_summary(
    df_top: pd.DataFrame,
    column: str,
    *,
    label_header: str,
    fixed_order: list[str] | None = None,
) -> pd.DataFrame:
    """Counts and % of portfolio for `column` (e.g. Marketcap with fixed_order, or Industry sorted by count)."""
    total = len(df_top)
    counts = df_top[column].value_counts() if total else pd.Series(dtype=int)
    rows: list[dict] = []
    if fixed_order:
        for cat in fixed_order:
            n = int(counts.get(cat, 0)) if total else 0
            pct = (100.0 * n / total) if total else 0.0
            rows.append({label_header: cat, "Count": n, "Pct": round(pct, 2)})
        for cat in counts.index:
            if cat not in fixed_order:
                n = int(counts[cat])
                rows.append({label_header: cat, "Count": n, "Pct": round(100.0 * n / total, 2)})
    else:
        for cat, n in counts.items():
            rows.append({label_header: cat, "Count": int(n), "Pct": round(100.0 * int(n) / total, 2)})
    return pd.DataFrame(rows)


def write_combined_portfolio_summary_sheet(
    writer: pd.ExcelWriter,
    df_mcap: pd.DataFrame,
    df_industry: pd.DataFrame,
    *,
    sheet_name: str = "Portfolio_Summary",
    startrow_mcap: int = 0,
) -> None:
    """One sheet: Marketcap block, blank row, Industry block (each with its own header row)."""
    df_mcap.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow_mcap)
    startrow_ind = startrow_mcap + len(df_mcap) + 2
    df_industry.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow_ind)


def _portfolio_state_path() -> Path:
    return FINAL_RESULT_DIR / PORTFOLIO_STATE_FILENAME


def load_portfolio_state() -> dict | None:
    path = _portfolio_state_path()
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


_REBALANCE_CONTEXT_SCHEMA_VERSION = 1


def _rebalance_context_lv_tag() -> str:
    lv = _effective_low_vol_max_quantile()
    return f"{float(lv):.6f}".replace(".", "p").replace("-", "m")


def _rebalance_context_basename(rebalance_as_of: date, period: str) -> str:
    return (
        f"quality_momentum_rebalance_context_{rebalance_as_of:%Y-%m-%d}_{period}_"
        f"p{PORTFOLIO_SIZE}_or{OUTPUT_RANKED_SIZE}_ex{EXIT_RANK_THRESHOLD}_"
        f"lv{_rebalance_context_lv_tag()}"
    )


def _rebalance_context_archive_path(rebalance_as_of: date, period: str) -> Path | None:
    if PORTFOLIO_STATE_ARCHIVE_DIR is None:
        return None
    return Path(PORTFOLIO_STATE_ARCHIVE_DIR) / f"{_rebalance_context_basename(rebalance_as_of, period)}.json"


def _baseline_rebalance_context_matches(
    payload: dict, *, rebalance_as_of: date, period: str, baseline_as_of: date
) -> bool:
    if int(payload.get("schema_version", 0)) != _REBALANCE_CONTEXT_SCHEMA_VERSION:
        return False
    if _parse_session_date_label(payload.get("rebalance_session_date")) != rebalance_as_of:
        return False
    if _parse_session_date_label(payload.get("baseline_session_date")) != baseline_as_of:
        return False
    if (payload.get("rebalance_compare_period") or "").strip().lower() != period:
        return False
    try:
        if int(payload.get("portfolio_size", -1)) != PORTFOLIO_SIZE:
            return False
        if int(payload.get("output_ranked_size", -1)) != OUTPUT_RANKED_SIZE:
            return False
        if int(payload.get("exit_rank_threshold", -1)) != EXIT_RANK_THRESHOLD:
            return False
    except (TypeError, ValueError):
        return False
    try:
        file_lv = float(payload["low_volatility_max_quantile"])
        cur_lv = float(_effective_low_vol_max_quantile())
    except (KeyError, TypeError, ValueError):
        return False
    if abs(file_lv - cur_lv) > 1e-9:
        return False
    rec = payload.get("baseline_ranked_records")
    if not rec or not isinstance(rec, list):
        return False
    return True


def _try_load_rebalance_baseline_context(
    path: Path, *, rebalance_as_of: date, period: str, baseline_as_of: date
) -> tuple[pd.DataFrame | None, dict | None]:
    if not path.is_file():
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    if not d or not _baseline_rebalance_context_matches(
        d, rebalance_as_of=rebalance_as_of, period=period, baseline_as_of=baseline_as_of
    ):
        return None, None
    try:
        df_b = pd.DataFrame(d["baseline_ranked_records"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if df_b is None or len(df_b) == 0:
        return None, None
    return df_b, d


def _write_rebalance_baseline_context(
    path: Path,
    *,
    rebalance_as_of: date,
    baseline_as_of: date,
    period: str,
    df_b: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    syms = df_b.head(PORTFOLIO_SIZE)["Symbol"].tolist()
    records = json.loads(df_b.to_json(orient="records", date_format="iso"))
    payload = {
        "schema_version": _REBALANCE_CONTEXT_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "rebalance_session_date": f"{rebalance_as_of:%Y-%m-%d}",
        "baseline_session_date": f"{baseline_as_of:%Y-%m-%d}",
        "rebalance_compare_period": period,
        "portfolio_size": PORTFOLIO_SIZE,
        "output_ranked_size": OUTPUT_RANKED_SIZE,
        "exit_rank_threshold": EXIT_RANK_THRESHOLD,
        "low_volatility_max_quantile": float(_effective_low_vol_max_quantile()),
        "top_portfolio_symbols": syms,
        "baseline_ranked_records": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_rebalance_baseline(
    as_of: date, compare_period: str
) -> tuple[list[str], str, str, str, date | None, pd.DataFrame | None]:
    """IN/OUT baseline: (symbols, description, prev_saved_at, prev_run_as_of, baseline_price_session, df_baseline_cached).

    ``baseline_price_session`` is the calendar date for baseline **closes** (``as_of − N`` in calendar
    modes, or from JSON ``run_as_of`` in ``each_run``), vs current session ``as_of``.

    For ``weekly`` / ``biweekly`` / ``monthly``, load a **deterministic context** JSON under
    ``PORTFOLIO_STATE_ARCHIVE_DIR`` when rebalance session date, baseline calendar date, period, and
    sizing/quantile params match; otherwise re-rank the baseline session and **write** that file.
    ``df_baseline_cached`` is the ranked baseline dataframe (from cache or fresh).

    ``each_run`` → latest ``quality_momentum_portfolio_state.json`` only (no calendar context file).
    """
    raw = (compare_period or "each_run").strip().lower()
    period_aliases = {
        "each_run": "each_run",
        "last_run": "each_run",
        "eachrun": "each_run",
        "each-run": "each_run",
        "last-run": "each_run",
        "bi-weekly": "biweekly",
        "biweekly": "biweekly",
        "bi_weekly": "biweekly",
    }
    period = period_aliases.get(raw, raw)
    if period in ("each_run", "last_run", ""):
        s = load_portfolio_state()
        if s is None:
            return [], "each_run: no latest state file", "", "", None, None
        syms = list(s.get("top_portfolio_symbols") or [])
        ra = str(s.get("run_as_of") or "")
        return (
            syms,
            "each_run: latest quality_momentum_portfolio_state.json",
            str(s.get("saved_at") or ""),
            ra,
            _parse_session_date_label(s.get("run_as_of")),
            None,
        )

    days = {"weekly": 7, "biweekly": 14, "monthly": 30}.get(period)
    if days is None:
        s = load_portfolio_state()
        if s is None:
            return [], f"unknown rebalance period={compare_period!r}; no state file", "", "", None, None
        syms = list(s.get("top_portfolio_symbols") or [])
        ra = str(s.get("run_as_of") or "")
        return (
            syms,
            f"unknown rebalance period={compare_period!r}; using latest state file",
            str(s.get("saved_at") or ""),
            ra,
            _parse_session_date_label(s.get("run_as_of")),
            None,
        )

    baseline_as_of = as_of - timedelta(days=days)
    ctx_path = _rebalance_context_archive_path(as_of, period)
    if ctx_path is not None:
        df_cached, meta = _try_load_rebalance_baseline_context(
            ctx_path, rebalance_as_of=as_of, period=period, baseline_as_of=baseline_as_of
        )
        if df_cached is not None and meta is not None:
            print(
                f"Rebalance baseline ({period}): using context cache {ctx_path.name} "
                f"(baseline {baseline_as_of}; rebalance session {as_of})"
            )
            syms = list(
                meta.get("top_portfolio_symbols")
                or df_cached.head(PORTFOLIO_SIZE)["Symbol"].tolist()
            )
            return (
                syms,
                f"{period}: baseline from context file {ctx_path.name} "
                f"(n_ranked={len(df_cached)}; baseline {baseline_as_of}; session {as_of})",
                str(meta.get("saved_at") or ""),
                str(meta.get("baseline_session_date") or f"{baseline_as_of:%Y-%m-%d}"),
                baseline_as_of,
                df_cached,
            )

    print(
        f"Rebalance baseline ({period}): ranked universe top {PORTFOLIO_SIZE} as-of {baseline_as_of} "
        f"(rebalance session {as_of} − {days}d)…"
    )
    df_b = _compute_ranked_universe_for_session(baseline_as_of, quiet=True)
    if df_b is None or len(df_b) == 0:
        return (
            [],
            f"{period}: empty ranking for baseline session {baseline_as_of} (rebalance session {as_of})",
            "(computed)",
            str(baseline_as_of),
            baseline_as_of,
            None,
        )
    syms = df_b.head(PORTFOLIO_SIZE)["Symbol"].tolist()
    if ctx_path is not None:
        try:
            _write_rebalance_baseline_context(
                ctx_path,
                rebalance_as_of=as_of,
                baseline_as_of=baseline_as_of,
                period=period,
                df_b=df_b,
            )
            print(f"Rebalance baseline ({period}): wrote context cache → {ctx_path.name}")
        except OSError as exc:
            print(f"Rebalance baseline ({period}): could not write context cache ({exc})")
    return (
        syms,
        f"{period}: baseline top {PORTFOLIO_SIZE} from ranked universe as-of {baseline_as_of} "
        f"(n_ranked={len(df_b)}); rebalance session {as_of}",
        datetime.now().isoformat(timespec="seconds"),
        str(baseline_as_of),
        baseline_as_of,
        df_b,
    )


def save_portfolio_state(
    *,
    top_portfolio_symbols: list[str],
    run_as_of: str,
    as_of_day: date,
    rebalance_compare_period: str,
    explicit_session_date: bool,
) -> tuple[Path, Path | None]:
    """Writes ``quality_momentum_portfolio_state.json`` (latest run). Calendar baseline cache uses separate context files in ``PORTFOLIO_STATE_ARCHIVE_DIR`` from ``resolve_rebalance_baseline``."""
    FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_as_of": run_as_of,
        "run_as_of_is_historical": explicit_session_date,
        "portfolio_size": PORTFOLIO_SIZE,
        "output_ranked_size": OUTPUT_RANKED_SIZE,
        "exit_rank_threshold": EXIT_RANK_THRESHOLD,
        "low_volatility_max_quantile": _effective_low_vol_max_quantile(),
        "rebalance_compare_period": rebalance_compare_period,
        "top_portfolio_symbols": top_portfolio_symbols,
    }
    latest = _portfolio_state_path()
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return latest, None


_REBALANCE_PRICE_COLS = (
    "Symbol",
    "Adj_Close",
    "Pct_vs_baseline",
)


def rebalance_symbol_prices(
    df_ranked: pd.DataFrame,
    symbols: list[str],
    *,
    current_price_fallback: dict[str, tuple[float | None, float | None]] | None = None,
    df_baseline: pd.DataFrame | None = None,
    pct_vs_baseline_only_if_symbols: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Symbol, session Adj_Close, and % vs baseline Adj_Close (adjusted prices only).

    When ``pct_vs_baseline_only_if_symbols`` is set, baseline closes are used for the
    percent change only if the symbol is in that set (prior baseline top-N book). Other
    symbols still get current ``Adj_Close`` but ``Pct_vs_baseline`` is None.
    """
    if not symbols:
        return pd.DataFrame(columns=list(_REBALANCE_PRICE_COLS))
    by_sym = df_ranked.set_index("Symbol")
    by_base = (
        df_baseline.set_index("Symbol")
        if df_baseline is not None and len(df_baseline)
        else None
    )
    rows: list[dict] = []
    for sym in symbols:
        adj = None
        if sym in by_sym.index:
            r = by_sym.loc[sym]
            adj = round(float(r["Adj_Close"]), 2) if pd.notna(r.get("Adj_Close")) else None
        elif current_price_fallback and sym in current_price_fallback:
            adj, _raw_c = current_price_fallback[sym]

        base_adj = None
        if by_base is not None and sym in by_base.index:
            if pct_vs_baseline_only_if_symbols is None or sym in pct_vs_baseline_only_if_symbols:
                br = by_base.loc[sym]
                base_adj = (
                    round(float(br["Adj_Close"]), 2) if pd.notna(br.get("Adj_Close")) else None
                )

        rows.append(
            {
                "Symbol": sym,
                "Adj_Close": adj,
                "Pct_vs_baseline": _pct_change_vs_baseline(base_adj, adj),
            }
        )
    return pd.DataFrame(rows)


def prev_holdings_rebalance_table(
    df_ranked: pd.DataFrame,
    prev_symbols: list[str],
    current_top_portfolio: set[str],
    df_baseline_ranked: pd.DataFrame | None = None,
    *,
    current_price_fallback: dict[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Each prior holding: baseline vs current Adj_Close, rank, exit monitor, short note."""
    if not prev_symbols:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Baseline_Adj_Close",
                "Current_Adj_Close",
                "Pct_vs_baseline",
                "Current_Rank",
                "In_top_portfolio",
                "Within_exit_rank_band",
                "Note",
            ]
        )
    by_sym = df_ranked.set_index("Symbol")
    by_base = (
        df_baseline_ranked.set_index("Symbol")
        if df_baseline_ranked is not None and len(df_baseline_ranked)
        else None
    )
    rows = []
    for sym in prev_symbols:
        base_adj = None
        if by_base is not None and sym in by_base.index:
            br = by_base.loc[sym]
            base_adj = round(float(br["Adj_Close"]), 2) if pd.notna(br.get("Adj_Close")) else None
        if sym not in by_sym.index:
            fa = None
            if current_price_fallback and sym in current_price_fallback:
                fa, _raw_c = current_price_fallback[sym]
            rows.append(
                {
                    "Symbol": sym,
                    "Baseline_Adj_Close": base_adj,
                    "Current_Adj_Close": fa,
                    "Pct_vs_baseline": _pct_change_vs_baseline(base_adj, fa),
                    "Current_Rank": None,
                    "In_top_portfolio": "N",
                    "Within_exit_rank_band": "N",
                    "Note": "Failed screen — not in ranked universe this run",
                }
            )
            continue
        r = by_sym.loc[sym]
        rp = int(r["Rank_Position"])
        in_top = "Y" if sym in current_top_portfolio else "N"
        within = "Y" if rp <= EXIT_RANK_THRESHOLD else "N"
        cur_adj = round(float(r["Adj_Close"]), 2) if pd.notna(r.get("Adj_Close")) else None
        if sym in current_top_portfolio:
            note = ""
        else:
            note = f"Below top {PORTFOLIO_SIZE} (universe rank {rp})"
        rows.append(
            {
                "Symbol": sym,
                "Baseline_Adj_Close": base_adj,
                "Current_Adj_Close": cur_adj,
                "Pct_vs_baseline": _pct_change_vs_baseline(base_adj, cur_adj),
                "Current_Rank": rp,
                "In_top_portfolio": in_top,
                "Within_exit_rank_band": within,
                "Note": note,
            }
        )
    return pd.DataFrame(rows)


def _excel_rows_used(n_data_rows: int) -> int:
    """Rows consumed by to_excel: 1 header + n_data_rows."""
    return 1 + n_data_rows


def _rebalance_section_title(
    writer: pd.ExcelWriter, sheet_name: str, title: str, startrow: int
) -> int:
    """One full-width title row + one blank row before the next table."""
    pd.DataFrame([[title]]).to_excel(
        writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow
    )
    return startrow + 2


def write_weekly_rebalance_sheet(
    writer: pd.ExcelWriter,
    *,
    df_ranked: pd.DataFrame,
    df_baseline_ranked: pd.DataFrame | None,
    baseline_price_session: date | None,
    current_top_syms: list[str],
    prev_saved_at: str,
    prev_run_as_of: str,
    current_run_as_of: str,
    prev_symbols: list[str],
    coming_in: list[str],
    going_out: list[str],
    df_prev_status: pd.DataFrame,
    baseline_description: str,
    rebalance_compare_period: str,
    current_price_fallback: dict[str, tuple[float | None, float | None]] | None = None,
    sheet_name: str = "Weekly_Rebalance",
) -> None:
    row = 0
    meta = pd.DataFrame(
        {
            "Field": [
                "Session_date (effective)",
                "Baseline_price_session",
                "Rebalance_period (effective)",
                "RUN_AS_OF (file default)",
                "REBALANCE_COMPARE_PERIOD (file default)",
                "Baseline_selection",
                "Previous_state_saved_at",
                "Previous_run_as_of",
                "Wall_clock_excel_write",
                "PORTFOLIO_SIZE",
                "OUTPUT_RANKED_SIZE",
                "EXIT_RANK_THRESHOLD",
                "LOW_VOLATILITY_MAX_QUANTILE (effective)",
            ],
            "Value": [
                current_run_as_of,
                str(baseline_price_session) if baseline_price_session else "(none)",
                rebalance_compare_period,
                str(RUN_AS_OF) if RUN_AS_OF else "(None)",
                REBALANCE_COMPARE_PERIOD,
                baseline_description,
                prev_saved_at or "(none)",
                prev_run_as_of or "(none)",
                datetime.now().isoformat(timespec="seconds"),
                PORTFOLIO_SIZE,
                OUTPUT_RANKED_SIZE,
                EXIT_RANK_THRESHOLD,
                _effective_low_vol_max_quantile(),
            ],
        }
    )
    meta.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    row += _excel_rows_used(len(meta)) + 1

    pct_vs_baseline_book = frozenset(prev_symbols) if prev_symbols else None
    ba = str(baseline_price_session) if baseline_price_session else "n/a"

    row = _rebalance_section_title(
        writer,
        sheet_name,
        f"1) Current session — top {PORTFOLIO_SIZE} portfolio (this run, session {current_run_as_of})",
        row,
    )
    if current_top_syms:
        df_cur = rebalance_symbol_prices(
            df_ranked,
            current_top_syms,
            current_price_fallback=current_price_fallback,
            df_baseline=df_baseline_ranked,
            pct_vs_baseline_only_if_symbols=pct_vs_baseline_book,
        )
    else:
        df_cur = pd.DataFrame(
            [
                {
                    "Symbol": "(empty)",
                    "Adj_Close": None,
                    "Pct_vs_baseline": None,
                }
            ]
        )
    df_cur.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    row += _excel_rows_used(len(df_cur)) + 1

    row = _rebalance_section_title(
        writer,
        sheet_name,
        "2) Entering top portfolio — new names vs previous baseline",
        row,
    )
    if coming_in:
        df_in = rebalance_symbol_prices(
            df_ranked,
            coming_in,
            current_price_fallback=current_price_fallback,
            df_baseline=df_baseline_ranked,
            pct_vs_baseline_only_if_symbols=pct_vs_baseline_book,
        )
    else:
        df_in = pd.DataFrame(
            [
                {
                    "Symbol": "(none)",
                    "Adj_Close": None,
                    "Pct_vs_baseline": None,
                }
            ]
        )
    df_in.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    row += _excel_rows_used(len(df_in)) + 1

    row = _rebalance_section_title(
        writer,
        sheet_name,
        "3) Leaving top portfolio — names that dropped out of top vs previous baseline",
        row,
    )
    if going_out:
        df_outm = rebalance_symbol_prices(
            df_ranked,
            going_out,
            current_price_fallback=current_price_fallback,
            df_baseline=df_baseline_ranked,
            pct_vs_baseline_only_if_symbols=pct_vs_baseline_book,
        )
    else:
        df_outm = pd.DataFrame(
            [
                {
                    "Symbol": "(none)",
                    "Adj_Close": None,
                    "Pct_vs_baseline": None,
                }
            ]
        )
    df_outm.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    row += _excel_rows_used(len(df_outm)) + 1

    row = _rebalance_section_title(
        writer,
        sheet_name,
        f"4) Prior baseline top {PORTFOLIO_SIZE} — baseline session {ba} vs {current_run_as_of}: "
        f"adj. close, % change, rank, exit monitor",
        row,
    )
    if not df_prev_status.empty:
        df_prev_status.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    else:
        pd.DataFrame([["(No prior baseline list — nothing to track here)"]]).to_excel(
            writer, sheet_name=sheet_name, index=False, header=False, startrow=row
        )


def main(
    *,
    as_of: date | None = None,
    rebalance_compare_period: str | None = None,
) -> None:
    run_as_of_file = _coerce_run_as_of_config(RUN_AS_OF)
    as_of_day: date = (
        as_of
        if as_of is not None
        else (run_as_of_file if run_as_of_file is not None else datetime.now().date())
    )
    rebalance_effective = (rebalance_compare_period or REBALANCE_COMPARE_PERIOD or "each_run").strip()
    explicit_session_date = (as_of is not None) or (run_as_of_file is not None)

    # yfinance `end` is exclusive (no bar at end): use midnight *after* session so last included bar is as_of_day.
    end_date = datetime.combine(as_of_day, datetime.min.time()) + timedelta(days=1)
    print(f"Session as_of: {as_of_day}  |  Rebalance compare: {rebalance_effective}")
    if as_of_day != datetime.now().date():
        print(
            f"  Data through {as_of_day}; yfinance end is exclusive → passed end={end_date:%Y-%m-%d %H:%M:%S}"
        )

    df_ranked = _compute_ranked_universe_for_session(as_of_day, quiet=False)
    if df_ranked is None or len(df_ranked) == 0:
        print("No stocks passed the trend, liquidity, and low-volatility filters.")
        return

    current_top_syms = df_ranked.head(PORTFOLIO_SIZE)["Symbol"].tolist()
    current_top_set = set(current_top_syms)

    (
        prev_syms,
        rebalance_baseline_desc,
        prev_saved_at,
        prev_run_as_of,
        baseline_price_session,
        df_baseline_cached,
    ) = resolve_rebalance_baseline(as_of_day, rebalance_effective)

    df_baseline_ranked: pd.DataFrame | None = None
    if prev_syms and baseline_price_session is not None:
        if df_baseline_cached is not None:
            df_baseline_ranked = df_baseline_cached
        elif baseline_price_session == as_of_day:
            df_baseline_ranked = df_ranked
        else:
            df_baseline_ranked = _compute_ranked_universe_for_session(baseline_price_session, quiet=True)

    if prev_syms:
        coming_in = sorted(set(current_top_syms) - set(prev_syms))
        going_out = sorted(set(prev_syms) - set(current_top_syms))
    else:
        coming_in = []
        going_out = []

    ranked_symbol_set = set(df_ranked["Symbol"].tolist())
    current_price_fallback = _current_price_fallback_for_symbols(
        prev_syms, as_of_day, ranked_symbol_set
    )

    df_prev_status = prev_holdings_rebalance_table(
        df_ranked,
        prev_syms,
        current_top_set,
        df_baseline_ranked=df_baseline_ranked,
        current_price_fallback=current_price_fallback,
    )

    # Portfolio mix: only the top PORTFOLIO_SIZE names (actual book), not the extended ranked list
    df_portfolio_slice = df_ranked.head(PORTFOLIO_SIZE)
    df_portfolio_mcap = portfolio_mix_summary(
        df_portfolio_slice,
        "Marketcap",
        label_header="Marketcap",
        fixed_order=["Largecap", "Midcap", "Smallcap"],
    )
    df_portfolio_industry = portfolio_mix_summary(
        df_portfolio_slice, "Industry", label_header="Industry", fixed_order=None
    )

    # Sheet1: top OUTPUT_RANKED_SIZE for decisions on borderline / lower-ranked names
    df_out = df_ranked.head(OUTPUT_RANKED_SIZE).copy()
    df_out.insert(0, "Position", np.arange(1, len(df_out) + 1))
    df_out.drop(columns=["Rank_Position"], inplace=True, errors="ignore")

    # Round columns for clean Excel output
    round_cols = [
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
    for c in round_cols:
        if c in df_out.columns:
            df_out[c] = df_out[c].round(2)

    # Final Column Selection
    final_cols = [
        "Position",
        "Symbol",
        "Industry",
        "Marketcap",
        "Adj_Close",
        "Close",
        # "Blended_Rank",
        "ADTV_Cr",
        "Volatility_Score",
        "Return_1M",
        "Return_3M",
        "Return_6M",
        "Return_9M",
        # "RS_3M_vs_Bench",
        # "RS_6M_vs_Bench",
        # "RS_9M_vs_Bench",
    ]
    
    # DECISION RULES COMMENTED FOR EXCEL OUTPUT:
    # 1. BLENDED_RANK: Primary factor. Shows stocks leading the market and rising.
    # 2. VOLATILITY_SCORE: 
    #    - < 1.8: Very steady trend (institutional quality).
    #    - > 3.0: Very jumpy (high risk of a "pump and dump" or news spike pullback).
    # 3. ADTV_Cr: Ensures you can sell your position. Never buy more than 1% of this value.

    FINAL_RESULT_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_RESULT_STOCK_DIR / "quality_momentum_rs_lv.xlsx"

    print(f"\nPortfolio summary (holdings = top {PORTFOLIO_SIZE} by Blended_Rank, by Marketcap):")
    for _, r in df_portfolio_mcap.iterrows():
        print(f"  {r['Marketcap']}: {int(r['Count'])}  ({r['Pct']}%)")

    print(f"\nPortfolio summary (holdings = top {PORTFOLIO_SIZE} by Blended_Rank, by Industry):")
    for _, r in df_portfolio_industry.iterrows():
        print(f"  {r['Industry']}: {int(r['Count'])}  ({r['Pct']}%)")

    run_as_of = str(as_of_day)
    print("\n--- Rebalance vs baseline ---")
    print(f"Compare mode: {rebalance_effective} — {rebalance_baseline_desc}")
    if prev_syms:
        print(
            f"Baseline run as_of: {prev_run_as_of} (state saved: {prev_saved_at}); "
            f"closes vs current: baseline session {baseline_price_session or 'n/a'} vs {run_as_of}"
        )
        print(f"Baseline top {PORTFOLIO_SIZE} holdings: {', '.join(prev_syms)}")
        print(f"Coming into portfolio ({len(coming_in)}): {', '.join(coming_in) if coming_in else '(none)'}")
        print(f"Leaving portfolio ({len(going_out)}): {', '.join(going_out) if going_out else '(none)'}")
        print(
            f"Exit monitor: rank in full universe > {EXIT_RANK_THRESHOLD} → Within_exit_rank_band = N "
            f"(see Weekly_Rebalance sheet)."
        )
    else:
        print("No baseline holdings (each_run: no state file, or calendar mode: empty computed baseline).")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out[final_cols].to_excel(writer, sheet_name="Sheet1", index=False)
        write_combined_portfolio_summary_sheet(writer, df_portfolio_mcap, df_portfolio_industry)
        write_weekly_rebalance_sheet(
            writer,
            df_ranked=df_ranked,
            df_baseline_ranked=df_baseline_ranked,
            baseline_price_session=baseline_price_session,
            current_top_syms=current_top_syms,
            prev_saved_at=prev_saved_at,
            prev_run_as_of=prev_run_as_of,
            current_run_as_of=run_as_of,
            prev_symbols=prev_syms,
            coming_in=coming_in,
            going_out=going_out,
            df_prev_status=df_prev_status,
            baseline_description=rebalance_baseline_desc,
            rebalance_compare_period=rebalance_effective,
            current_price_fallback=current_price_fallback,
        )

    latest_state_path, _archive_state_path = save_portfolio_state(
        top_portfolio_symbols=current_top_syms,
        run_as_of=run_as_of,
        as_of_day=as_of_day,
        rebalance_compare_period=rebalance_effective,
        explicit_session_date=explicit_session_date,
    )
    print(f"\nSaved portfolio state for next week → {latest_state_path}")

    print(
        f"\nSuccess: Wrote top {len(df_out)} ranked rows (OUTPUT_RANKED_SIZE={OUTPUT_RANKED_SIZE}); "
        f"mix from top {PORTFOLIO_SIZE} holdings; Weekly_Rebalance sheet + state file → {out_path}"
    )

def _env_run_as_of() -> date | None:
    raw = (os.environ.get("QUALITY_RS_RUN_AS_OF") or "").strip()
    if not raw:
        return None
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _env_rebalance_period() -> str | None:
    v = (os.environ.get("QUALITY_RS_REBALANCE") or "").strip()
    return v or None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality momentum RS: ranks, Excel, portfolio state, rebalance IN/OUT.",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        metavar="YYYY-MM-DD",
        default=None,
        help="Session date (bars through this day). Beats env QUALITY_RS_RUN_AS_OF and file RUN_AS_OF.",
    )
    parser.add_argument(
        "--rebalance",
        dest="rebalance",
        default=None,
        metavar="PERIOD",
        help="each_run | weekly | biweekly | bi-weekly | monthly. Beats env QUALITY_RS_REBALANCE and file default.",
    )
    args = parser.parse_args()

    resolved_as_of: date | None = None
    if args.as_of:
        resolved_as_of = datetime.strptime(args.as_of.strip(), "%Y-%m-%d").date()
    else:
        resolved_as_of = _env_run_as_of()

    resolved_rebalance = (args.rebalance or "").strip() or None
    if not resolved_rebalance:
        resolved_rebalance = _env_rebalance_period()

    main(as_of=resolved_as_of, rebalance_compare_period=resolved_rebalance)