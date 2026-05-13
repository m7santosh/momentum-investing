"""
NSE data helpers: bhavcopy (after-hours EOD) and live quotes (market hours).

Fallback chain when Yahoo Finance returns NaN prices for the latest session:
  1. Bhavcopy — official EOD CSV, available ~7 PM IST after close.
  2. Live quotes — NSE API (Nifty 500 + ETFs), works during market hours
     and the window between close and bhavcopy publication.
  3. Drop NaN row — last resort if both NSE sources are unavailable.
"""

import io
import time
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

_IST = timezone(timedelta(hours=5, minutes=30))

_CACHE: dict[date, dict[str, dict] | None] = {}
_SESSION: requests.Session | None = None
_ANNOUNCED: set[date] = set()

_LIVE_CACHE: dict[str, dict] = {}
_LIVE_CACHE_TS: float = 0
_LIVE_ANNOUNCED: bool = False


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.nseindia.com/",
        })
        try:
            _SESSION.get("https://www.nseindia.com", timeout=10)
        except Exception:
            pass
    return _SESSION


def today_ist() -> date:
    return datetime.now(_IST).date()


def _bhavcopy_urls(trade_date: date) -> list[str]:
    """Candidate download URLs — new format first, legacy fallback."""
    yyyymmdd = trade_date.strftime("%Y%m%d")
    mon = trade_date.strftime("%b").upper()
    ddMONyyyy = trade_date.strftime("%d") + mon + str(trade_date.year)
    return [
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{trade_date.year}/{mon}/cm{ddMONyyyy}bhav.csv.zip",
    ]


def _parse_bhavcopy_df(df: pd.DataFrame) -> dict[str, dict]:
    """Parse either new-format or legacy-format bhavcopy CSV into a dict."""
    if "TckrSymb" in df.columns:
        sym_col, series_col = "TckrSymb", "SctySrs"
        cols = {
            "open": "OpnPric", "high": "HghPric",
            "low": "LwPric", "close": "ClsPric", "volume": "TtlTradgVol",
        }
    elif "SYMBOL" in df.columns:
        sym_col, series_col = "SYMBOL", "SERIES"
        cols = {
            "open": "OPEN", "high": "HIGH",
            "low": "LOW", "close": "CLOSE", "volume": "TOTTRDQTY",
        }
    else:
        return {}

    eq = df[df[series_col].astype(str).str.strip() == "EQ"]
    result: dict[str, dict] = {}
    for _, row in eq.iterrows():
        sym = str(row[sym_col]).strip()
        try:
            result[sym] = {
                k: (int(row[v]) if k == "volume" else float(row[v]))
                for k, v in cols.items()
            }
        except (ValueError, KeyError):
            continue
    return result


def fetch_bhavcopy(trade_date: date) -> dict[str, dict]:
    """Download NSE CM equity bhavcopy for *trade_date*.

    Returns ``{NSE_SYMBOL: {open, high, low, close, volume}}`` or ``{}``
    on failure.  Result is cached; repeated calls for the same date are free.
    """
    if trade_date in _CACHE:
        return _CACHE[trade_date] or {}

    sess = _session()
    for url in _bhavcopy_urls(trade_date):
        try:
            resp = sess.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    continue
                df = pd.read_csv(zf.open(csv_names[0]))
            result = _parse_bhavcopy_df(df)
            if result:
                _CACHE[trade_date] = result
                if trade_date not in _ANNOUNCED:
                    print(f"  [NSE Bhavcopy] Fetched official EOD data for {trade_date} ({len(result)} symbols)")
                    _ANNOUNCED.add(trade_date)
                return result
        except Exception:
            continue

    _CACHE[trade_date] = None
    if trade_date not in _ANNOUNCED:
        print(f"  [NSE Bhavcopy] Not available for {trade_date} — trying live quotes")
        _ANNOUNCED.add(trade_date)
    return {}


def _safe_float(val: object) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val.replace(",", ""))
    return 0.0


def _safe_int(val: object) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return int(float(val.replace(",", "")))
    return 0


def _parse_index_item(item: dict) -> dict | None:
    """Parse a row from /api/equity-stockIndices (Nifty 500 etc.)."""
    ltp = item.get("lastPrice")
    if ltp is None:
        return None
    return {
        "open": _safe_float(item.get("open", 0)),
        "high": _safe_float(item.get("dayHigh", 0)),
        "low": _safe_float(item.get("dayLow", 0)),
        "close": _safe_float(ltp),
        "volume": _safe_int(item.get("totalTradedVolume", 0)),
    }


def _parse_etf_item(item: dict) -> dict | None:
    """Parse a row from /api/etf (different field names: ltP, qty, etc.)."""
    ltp = item.get("ltP")
    if ltp is None:
        return None
    return {
        "open": _safe_float(item.get("open", 0)),
        "high": _safe_float(item.get("high", 0)),
        "low": _safe_float(item.get("low", 0)),
        "close": _safe_float(ltp),
        "volume": _safe_int(item.get("qty", 0)),
    }


def fetch_nse_live_quotes() -> dict[str, dict]:
    """Bulk live quotes from NSE (Nifty 500 stocks + ETFs).

    Returns ``{NSE_SYMBOL: {open, high, low, close, volume}}`` or ``{}``.
    Cached for 30 seconds so repeated calls within the same run are free.
    """
    global _LIVE_CACHE, _LIVE_CACHE_TS, _LIVE_ANNOUNCED
    now = time.time()
    if _LIVE_CACHE and (now - _LIVE_CACHE_TS) < 30:
        return _LIVE_CACHE

    sess = _session()
    result: dict[str, dict] = {}

    _SOURCES: list[tuple[str, callable]] = [
        ("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500", _parse_index_item),
        ("https://www.nseindia.com/api/etf", _parse_etf_item),
    ]
    for url, parser in _SOURCES:
        try:
            resp = sess.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            for item in resp.json().get("data", []):
                sym = str(item.get("symbol", "")).strip()
                if not sym:
                    continue
                parsed = parser(item)
                if parsed:
                    result[sym] = parsed
        except Exception:
            continue

    if result:
        _LIVE_CACHE = result
        _LIVE_CACHE_TS = now
        if not _LIVE_ANNOUNCED:
            print(f"  [NSE Live] Fetched live quotes ({len(result)} symbols)")
            _LIVE_ANNOUNCED = True

    return result


def nse_symbol_from_yahoo(yahoo_ticker: str) -> str:
    """``RELIANCE.NS`` → ``RELIANCE``, ``M&M.NS`` → ``M&M``."""
    return yahoo_ticker.split(".")[0]
