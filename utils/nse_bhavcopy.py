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


# --- NSE index EOD (ind_close_all_*.csv) — sector indices are NOT in CM bhavcopy ---

_INDEX_CLOSE_CACHE: dict[date, dict[str, float] | None] = {}
_INDEX_CLOSE_ANNOUNCED: set[date] = set()

# Yahoo / custom symbol → name as listed in NSE ind_close_all CSV
YAHOO_TO_NSE_INDEX: dict[str, str] = {
    "^NSEBANK": "Nifty Bank",
    "^CNXENERGY": "Nifty Energy",
    "^CNXMETAL": "Nifty Metal",
    "^CNXAUTO": "Nifty Auto",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXIT": "Nifty IT",
    "^CNXFMCG": "Nifty FMCG",
    "^CNXREALTY": "Nifty Realty",
    "^NSEI": "Nifty 50",
    "^CRSLDX": "Nifty 500",
    "NIFTY_CAPITAL_MKT.NS": "Nifty Capital Markets",
    "NIFTY_RAILWAYSPSU.NS": "Nifty India Railways PSU",
    "NIFTY_OIL_AND_GAS.NS": "Nifty Oil & Gas",
    "NIFTY_CHEMICALS.NS": "Nifty Chemicals",
}


def yahoo_ticker_to_nse_index(yahoo_ticker: str) -> str | None:
    """Map a Yahoo-style ticker to the NSE index name in ``ind_close_all`` CSV."""
    if yahoo_ticker in YAHOO_TO_NSE_INDEX:
        return YAHOO_TO_NSE_INDEX[yahoo_ticker]
    base = yahoo_ticker.split(".")[0].replace("_", " ").replace("^", "").strip()
    if not base:
        return None
    if base.upper().startswith("NIFTY"):
        return " ".join(part.capitalize() for part in base.split())
    return base.upper()


def _normalize_index_key(name: str) -> str:
    return " ".join(name.upper().split())


def _ind_close_all_url(trade_date: date) -> str:
    return (
        "https://nsearchives.nseindia.com/content/indices/"
        f"ind_close_all_{trade_date.strftime('%d%m%Y')}.csv"
    )


def _parse_ind_close_all(text: str) -> dict[str, float]:
    """``Index Name`` → closing value from NSE ``ind_close_all`` CSV."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("<") or line.lower().startswith("index name"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        name = parts[0].strip()
        try:
            result[name] = float(parts[5])
        except ValueError:
            continue
    return result


def fetch_index_close_all(trade_date: date, *, quiet: bool = False) -> dict[str, float]:
    """All index closes for one session from NSE ``ind_close_all`` archive."""
    if trade_date in _INDEX_CLOSE_CACHE:
        return _INDEX_CLOSE_CACHE[trade_date] or {}

    sess = _session()
    url = _ind_close_all_url(trade_date)
    try:
        resp = sess.get(url, timeout=30)
        if resp.status_code == 200 and resp.text and "Closing Index Value" in resp.text:
            result = _parse_ind_close_all(resp.text)
            if result:
                _INDEX_CLOSE_CACHE[trade_date] = result
                if not quiet and trade_date not in _INDEX_CLOSE_ANNOUNCED:
                    print(f"  [NSE Index EOD] {trade_date} ({len(result)} indices)")
                    _INDEX_CLOSE_ANNOUNCED.add(trade_date)
                return result
    except Exception:
        pass

    _INDEX_CLOSE_CACHE[trade_date] = None
    return {}


def resolve_index_name(requested: str, available: dict[str, float]) -> str | None:
    """Match *requested* to a key in *available* (case/spacing insensitive, exact only)."""
    if not available:
        return None
    req = _normalize_index_key(requested)
    by_norm = {_normalize_index_key(k): k for k in available}
    return by_norm.get(req)


def fetch_index_close_history(
    index_name: str,
    start_date: date,
    end_date: date,
) -> "pd.Series":
    """Daily index closes from ``ind_close_all`` archives (walk trading days)."""
    import pandas as pd

    batch = fetch_index_close_histories([index_name], start_date, end_date)
    return batch.get(index_name, pd.Series(dtype=float))


def fetch_index_close_histories(
    index_names: list[str],
    start_date: date,
    end_date: date,
    *,
    quiet: bool = True,
) -> dict[str, "pd.Series"]:
    """Daily closes for many indices — one archive download per trading day."""
    import pandas as pd

    buckets: dict[str, list[tuple[pd.Timestamp, float]]] = {n: [] for n in index_names}
    resolved: dict[str, str | None] = {n: None for n in index_names}
    sessions_loaded = 0

    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            day_map = fetch_index_close_all(d, quiet=quiet)
            if day_map:
                sessions_loaded += 1
                for name in index_names:
                    if resolved[name] is None:
                        resolved[name] = resolve_index_name(name, day_map)
                    canonical = resolved[name]
                    if canonical and canonical in day_map:
                        buckets[name].append((pd.Timestamp(d), day_map[canonical]))
        d += timedelta(days=1)

    if quiet and sessions_loaded:
        print(
            f"  [NSE Index EOD] Loaded {sessions_loaded} sessions "
            f"for {len(index_names)} index(es)"
        )

    out: dict[str, pd.Series] = {}
    for name, rows in buckets.items():
        if rows:
            out[name] = pd.Series({ts: val for ts, val in rows}).sort_index()
            out[name].name = name
        else:
            out[name] = pd.Series(dtype=float)
    return out


def load_nse_index_weekly_histories(
    index_names: list[str],
    *,
    period: str = "1y",
    min_points: int = 15,
) -> dict[str, "pd.Series"]:
    """Weekly closes from NSE ``ind_close_all`` (Friday week-end)."""
    import pandas as pd

    unique = list(dict.fromkeys(index_names))
    if not unique:
        return {}

    end = today_ist()
    days_back = 400 if period == "1y" else 800
    start = end - timedelta(days=days_back)
    daily_batch = fetch_index_close_histories(unique, start, end, quiet=False)

    out: dict[str, pd.Series] = {}
    for name in unique:
        daily = daily_batch.get(name, pd.Series(dtype=float))
        if len(daily) < min_points:
            out[name] = pd.Series(dtype=float)
            continue
        weekly = daily.resample("W-FRI").last().dropna()
        out[name] = weekly if len(weekly) >= min_points else pd.Series(dtype=float)
    return out


def fetch_equity_close_history_bhavcopy(
    nse_symbol: str,
    start_date: date,
    end_date: date,
) -> "pd.Series":
    """Daily equity closes from CM bhavcopy (EQ series only)."""
    import pandas as pd

    rows: list[tuple[pd.Timestamp, float]] = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            day_map = fetch_bhavcopy(d)
            if nse_symbol in day_map:
                rows.append((pd.Timestamp(d), day_map[nse_symbol]["close"]))
        d += timedelta(days=1)

    if not rows:
        return pd.Series(dtype=float)
    series = pd.Series({ts: val for ts, val in rows}).sort_index()
    series.name = nse_symbol
    return series


def fetch_weekly_close_series(
    yahoo_ticker: str,
    period: str = "1y",
    min_points: int = 15,
) -> "pd.Series":
    """Weekly closes: Yahoo first, then NSE index archive or CM bhavcopy."""
    import pandas as pd

    try:
        import yfinance as yf

        raw = yf.download(
            yahoo_ticker,
            period=period,
            interval="1wk",
            progress=False,
            auto_adjust=True,
        )
        if raw is not None and len(raw) > 0:
            close = raw["Close"] if "Close" in raw.columns else raw.squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) >= min_points:
                return close
    except Exception:
        pass

    end = today_ist()
    days_back = 400 if period == "1y" else 800
    start = end - timedelta(days=days_back)

    index_name = yahoo_ticker_to_nse_index(yahoo_ticker)
    if index_name:
        daily = fetch_index_close_histories([index_name], start, end).get(
            index_name, pd.Series(dtype=float)
        )
        if len(daily) >= min_points:
            weekly = daily.resample("W-FRI").last().dropna()
            if len(weekly) >= min_points:
                print(f"  [NSE Index EOD] Weekly series for {yahoo_ticker} ({len(weekly)} wks)")
                return weekly

    if yahoo_ticker.endswith(".NS"):
        nse_sym = nse_symbol_from_yahoo(yahoo_ticker)
        daily = fetch_equity_close_history_bhavcopy(nse_sym, start, end)
        if len(daily) >= min_points:
            weekly = daily.resample("W-FRI").last().dropna()
            if len(weekly) >= min_points:
                print(f"  [NSE Bhavcopy] Weekly series for {yahoo_ticker} ({len(weekly)} wks)")
                return weekly

    return pd.Series(dtype=float)


def load_weekly_histories_batch(
    yahoo_tickers: list[str],
    period: str = "1y",
    min_points: int = 15,
) -> dict[str, "pd.Series"]:
    """Load weekly closes for many tickers (one NSE archive pass per trading day)."""
    import pandas as pd

    import yfinance as yf

    result: dict[str, pd.Series] = {t: pd.Series(dtype=float) for t in yahoo_tickers}
    try:
        raw = yf.download(
            yahoo_tickers,
            period=period,
            interval="1wk",
            progress=False,
            auto_adjust=True,
        )
        if raw is not None and len(raw) > 0:
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if isinstance(close, pd.DataFrame):
                for t in yahoo_tickers:
                    if t in close.columns:
                        s = close[t].dropna()
                        if len(s) >= min_points:
                            result[t] = s
            elif len(yahoo_tickers) == 1:
                s = pd.Series(close).dropna()
                if len(s) >= min_points:
                    result[yahoo_tickers[0]] = s
    except Exception:
        pass

    need_nse_index: dict[str, str] = {}
    for t in yahoo_tickers:
        if len(result[t]) < min_points:
            n = yahoo_ticker_to_nse_index(t)
            if n:
                need_nse_index[t] = n

    if need_nse_index:
        end = today_ist()
        days_back = 400 if period == "1y" else 800
        start = end - timedelta(days=days_back)
        unique_names = list(dict.fromkeys(need_nse_index.values()))
        daily_batch = fetch_index_close_histories(unique_names, start, end)
        for ticker, index_name in need_nse_index.items():
            daily = daily_batch.get(index_name, pd.Series(dtype=float))
            if len(daily) < min_points:
                continue
            weekly = daily.resample("W-FRI").last().dropna()
            if len(weekly) >= min_points:
                result[ticker] = weekly
                print(
                    f"  [NSE Index EOD] Weekly series for {ticker} ({len(weekly)} wks)"
                )

    for ticker in yahoo_tickers:
        if len(result[ticker]) >= min_points or not ticker.endswith(".NS"):
            continue
        end = today_ist()
        days_back = 400 if period == "1y" else 800
        start = end - timedelta(days=days_back)
        nse_sym = nse_symbol_from_yahoo(ticker)
        daily = fetch_equity_close_history_bhavcopy(nse_sym, start, end)
        if len(daily) < min_points:
            continue
        weekly = daily.resample("W-FRI").last().dropna()
        if len(weekly) >= min_points:
            result[ticker] = weekly
            print(f"  [NSE Bhavcopy] Weekly series for {ticker} ({len(weekly)} wks)")

    return result
