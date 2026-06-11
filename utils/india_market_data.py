"""
Unified India market OHLCV loader for momentum screeners and backtests.

Policy:
  -``.NS`` symbols: Yahoo first; if Yahoo has fewer than 63 valid bars, rebuild from
    NSE CM bhavcopy for the requested window (then overlay Yahoo for today).
  - Other past sessions: NSE EOD overlay on Yahoo (bhavcopy / index archive).
  - Today / realtime: Yahoo Finance; NSE live/bhavcopy only if Yahoo Close is NaN.

Cache:
  - Each ticker stores the exact ``[start, end)`` window last downloaded.
  - Same window again → reuse. Any date change → fresh download for that ticker.
  - NSE bhavcopy/index day archives stay cached per trading day (shared across tickers).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from utils.nse_bhavcopy import (
    clear_nse_data_caches,
    fetch_bhavcopy,
    fetch_index_close_all,
    fetch_nse_live_quotes,
    nse_symbol_from_yahoo,
    resolve_index_name,
    today_ist,
    yahoo_ticker_to_nse_index,
)

# Align with momentum engine MIN_HISTORY_SESSIONS (LB_3M).
MIN_BARS_NS_BHAVCOPY_FIRST = 63

_OHLCV_COLS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


@dataclass(frozen=True)
class IndiaTickerLoadMeta:
    ticker: str
    source: str
    yahoo_bars: int
    bhavcopy_bars: int
    effective_bars: int

    @property
    def sufficient(self) -> bool:
        return self.effective_bars >= MIN_BARS_NS_BHAVCOPY_FIRST


@dataclass(frozen=True)
class _OhlcvCacheEntry:
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    df: pd.DataFrame
    meta: IndiaTickerLoadMeta | None = None


_OHLCV_STORE: dict[str, _OhlcvCacheEntry] = {}
_LOAD_META: dict[str, IndiaTickerLoadMeta] = {}


@dataclass(frozen=True)
class IndiaMarketDataRunStats:
    cached: int = 0
    downloaded: int = 0

    def summary(self) -> str:
        parts: list[str] = []
        if self.cached:
            parts.append(f"{self.cached} cached")
        if self.downloaded:
            parts.append(f"{self.downloaded} downloaded")
        return ", ".join(parts) if parts else "no symbols"


_run_stats = IndiaMarketDataRunStats()


def get_india_market_data_run_stats() -> IndiaMarketDataRunStats:
    return _run_stats


def reset_india_market_data_run_stats() -> None:
    global _run_stats
    _run_stats = IndiaMarketDataRunStats()


def clear_india_market_data_cache(*, clear_nse_archives: bool = False) -> None:
    """Drop per-ticker OHLCV store (and optionally NSE day archives)."""
    _OHLCV_STORE.clear()
    _LOAD_META.clear()
    if clear_nse_archives:
        clear_nse_data_caches()


def get_ticker_load_meta(ticker: str) -> IndiaTickerLoadMeta | None:
    return _LOAD_META.get(ticker)


def _display_symbol(yahoo_ticker: str) -> str:
    return yahoo_ticker.replace(".NS", "").replace(".BO", "")


def summarize_etf_history_gaps(
    universe_tickers: list[str],
    *,
    min_bars: int = MIN_BARS_NS_BHAVCOPY_FIRST,
) -> dict[str, str]:
    """Report NSE bhavcopy backfills and symbols still below ``min_bars`` after load."""
    backfilled: list[str] = []
    insufficient: list[str] = []
    for ticker in universe_tickers:
        if not ticker.endswith(".NS"):
            continue
        meta = _LOAD_META.get(ticker)
        if meta is None:
            insufficient.append(f"{_display_symbol(ticker)} (not loaded)")
            continue
        if meta.source == "bhavcopy":
            backfilled.append(
                f"{_display_symbol(ticker)} (Yahoo {meta.yahoo_bars} -> NSE {meta.effective_bars})"
            )
        if meta.effective_bars < min_bars:
            insufficient.append(f"{_display_symbol(ticker)} ({meta.effective_bars} bars)")
    return {
        "Bhavcopy_Backfill": ", ".join(backfilled) if backfilled else "None",
        "Insufficient_History": ", ".join(insufficient) if insufficient else "None",
    }


def begin_fresh_india_market_data_session() -> None:
    """Clear all in-memory India data caches (explicit reset only)."""
    clear_india_market_data_cache(clear_nse_archives=True)


def format_range_label(start_date, end_date) -> str:
    """Human-readable download window ``[start, end)``."""
    start, end_exclusive = _normalize_range(start_date, end_date)
    end_inclusive = end_exclusive - pd.Timedelta(days=1)
    return f"{start.date():%Y-%m-%d} .. {end_inclusive.date():%Y-%m-%d}"


def _trade_date_from_index(idx) -> date:
    if hasattr(idx, "date"):
        return idx.date()
    return pd.Timestamp(idx).date()


def _as_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _normalize_range(start_date, end_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _as_timestamp(start_date)
    end = _as_timestamp(end_date)
    if end <= start:
        end = start + pd.Timedelta(days=1)
    return start, end


def _bump_stat(*, cached: bool = False, downloaded: bool = False) -> None:
    global _run_stats
    _run_stats = IndiaMarketDataRunStats(
        cached=_run_stats.cached + (1 if cached else 0),
        downloaded=_run_stats.downloaded + (1 if downloaded else 0),
    )


def _apply_nse_ohlcv(df: pd.DataFrame, idx, nse_row: dict) -> None:
    df.at[idx, "Close"] = nse_row["close"]
    df.at[idx, "Adj Close"] = nse_row["close"]
    df.at[idx, "Open"] = nse_row["open"]
    df.at[idx, "High"] = nse_row["high"]
    df.at[idx, "Low"] = nse_row["low"]
    if "Volume" in df.columns:
        last_vol = df.at[idx, "Volume"]
        if pd.isna(last_vol) or last_vol == 0:
            df.at[idx, "Volume"] = nse_row.get("volume", last_vol)


def _patch_cm_bhavcopy_history(df: pd.DataFrame, nse_sym: str, *, before: date) -> None:
    for idx in df.index:
        trade_dt = _trade_date_from_index(idx)
        if trade_dt >= before:
            continue
        day_map = fetch_bhavcopy(trade_dt, symbols={nse_sym})
        if nse_sym in day_map:
            _apply_nse_ohlcv(df, idx, day_map[nse_sym])


def _patch_index_eod_history(df: pd.DataFrame, index_name: str, *, before: date) -> None:
    resolved_by_date: dict[date, str | None] = {}
    for idx in df.index:
        trade_dt = _trade_date_from_index(idx)
        if trade_dt >= before:
            continue
        day_map = fetch_index_close_all(trade_dt, quiet=True)
        if not day_map:
            continue
        if trade_dt not in resolved_by_date:
            resolved_by_date[trade_dt] = resolve_index_name(index_name, day_map)
        key = resolved_by_date[trade_dt]
        if not key or key not in day_map:
            continue
        close = day_map[key]
        df.at[idx, "Close"] = close
        df.at[idx, "Adj Close"] = close


def _patch_today_yahoo_fallback(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Keep Yahoo on today; NSE fallback only when the last bar Close is missing."""
    if df is None or len(df) == 0:
        return df
    if pd.notna(df.iloc[-1].get("Close")):
        return df
    if not ticker.endswith(".NS"):
        return df.dropna(subset=["Close"])

    nse_sym = nse_symbol_from_yahoo(ticker)
    idx = df.index[-1]
    trade_dt = _trade_date_from_index(idx)
    bhav = fetch_bhavcopy(trade_dt, symbols={nse_sym})
    if nse_sym in bhav:
        _apply_nse_ohlcv(df, idx, bhav[nse_sym])
        return df
    if trade_dt == today_ist():
        live = fetch_nse_live_quotes()
        if nse_sym in live:
            _apply_nse_ohlcv(df, idx, live[nse_sym])
            return df
    return df.dropna(subset=["Close"])


def patch_yahoo_with_nse_eod(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Overlay NSE EOD on Yahoo bars before today IST; leave today on Yahoo."""
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    today = today_ist()

    if ticker.endswith(".NS"):
        nse_sym = nse_symbol_from_yahoo(ticker)
        _patch_cm_bhavcopy_history(out, nse_sym, before=today)
        return _patch_today_yahoo_fallback(out, ticker)

    index_name = yahoo_ticker_to_nse_index(ticker)
    if index_name:
        _patch_index_eod_history(out, index_name, before=today)
        return out

    return out


def _download_yahoo(ticker: str, start, end_exclusive) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        multi_level_index=False,
        auto_adjust=False,
        progress=False,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df


def _series_col(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def _yahoo_valid_bar_count(df: pd.DataFrame) -> int:
    if df is None or df.empty or "Close" not in df.columns:
        return 0
    return int(_series_col(df, "Close").dropna().shape[0])


def _effective_bar_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "Adj Close" in df.columns:
        return int(_series_col(df, "Adj Close").dropna().shape[0])
    if "Close" in df.columns:
        return int(_series_col(df, "Close").dropna().shape[0])
    return len(df)


def _build_ohlcv_from_cm_bhavcopy(
    nse_sym: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Build daily OHLCV from CM bhavcopy for ``[start, end_exclusive)``."""
    start_d = start.date()
    end_d = (end_exclusive - pd.Timedelta(days=1)).date()
    if end_d < start_d:
        return pd.DataFrame(), 0

    rows: list[dict] = []
    d = start_d
    while d <= end_d:
        if d.weekday() < 5:
            day_map = fetch_bhavcopy(d, symbols={nse_sym})
            if nse_sym in day_map:
                bar = day_map[nse_sym]
                rows.append(
                    {
                        "Date": pd.Timestamp(d),
                        "Open": bar["open"],
                        "High": bar["high"],
                        "Low": bar["low"],
                        "Close": bar["close"],
                        "Adj Close": bar["close"],
                        "Volume": bar.get("volume", 0),
                    }
                )
        d += timedelta(days=1)

    if not rows:
        return pd.DataFrame(), 0
    out = pd.DataFrame(rows).set_index("Date").sort_index()
    return out, len(out)


def _merge_yahoo_onto_bhavcopy(
    bhav_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Bhavcopy base; overlay Yahoo rows (typically today / sparse fixes)."""
    if yahoo_df is None or yahoo_df.empty:
        return _patch_today_yahoo_fallback(bhav_df, ticker)

    out = bhav_df.copy()
    yahoo = yahoo_df.copy()
    for idx in yahoo.index:
        if idx in out.index:
            for col in _OHLCV_COLS:
                if col not in yahoo.columns or col not in out.columns:
                    continue
                val = yahoo.at[idx, col]
                if pd.notna(val):
                    out.at[idx, col] = val
        else:
            row = yahoo.loc[[idx]]
            out = pd.concat([out, row])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return _patch_today_yahoo_fallback(out, ticker)


def _load_ns_equity_ohlcv(
    ticker: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, IndiaTickerLoadMeta]:
    nse_sym = nse_symbol_from_yahoo(ticker)
    yahoo_df = _download_yahoo(ticker, start, end_exclusive)
    yahoo_n = _yahoo_valid_bar_count(yahoo_df)

    if yahoo_n >= MIN_BARS_NS_BHAVCOPY_FIRST:
        patched = patch_yahoo_with_nse_eod(yahoo_df, ticker)
        meta = IndiaTickerLoadMeta(
            ticker=ticker,
            source="yahoo",
            yahoo_bars=yahoo_n,
            bhavcopy_bars=0,
            effective_bars=_effective_bar_count(patched),
        )
        return patched, meta

    bhav_df, bhav_n = _build_ohlcv_from_cm_bhavcopy(nse_sym, start, end_exclusive)
    if bhav_df.empty:
        patched = patch_yahoo_with_nse_eod(yahoo_df, ticker) if not yahoo_df.empty else yahoo_df
        meta = IndiaTickerLoadMeta(
            ticker=ticker,
            source="yahoo" if yahoo_n else "none",
            yahoo_bars=yahoo_n,
            bhavcopy_bars=0,
            effective_bars=_effective_bar_count(patched),
        )
        return patched, meta

    merged = _merge_yahoo_onto_bhavcopy(bhav_df, yahoo_df, ticker)
    meta = IndiaTickerLoadMeta(
        ticker=ticker,
        source="bhavcopy",
        yahoo_bars=yahoo_n,
        bhavcopy_bars=bhav_n,
        effective_bars=_effective_bar_count(merged),
    )
    return merged, meta


def _store_ticker_data(
    ticker: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    df: pd.DataFrame,
    meta: IndiaTickerLoadMeta | None,
) -> pd.DataFrame:
    _OHLCV_STORE[ticker] = _OhlcvCacheEntry(start, end_exclusive, df, meta)
    if meta is not None:
        _LOAD_META[ticker] = meta
    return df


def get_india_market_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Return OHLCV for ``[start_date, end_date)``; reuse only on an exact range match."""
    start, end_exclusive = _normalize_range(start_date, end_date)
    entry = _OHLCV_STORE.get(ticker)
    if (
        entry is not None
        and entry.start == start
        and entry.end_exclusive == end_exclusive
        and not entry.df.empty
    ):
        if entry.meta is not None:
            _LOAD_META[ticker] = entry.meta
        _bump_stat(cached=True)
        return entry.df.copy()

    if ticker.endswith(".NS"):
        loaded, meta = _load_ns_equity_ohlcv(ticker, start, end_exclusive)
    else:
        raw = _download_yahoo(ticker, start, end_exclusive)
        loaded = patch_yahoo_with_nse_eod(raw, ticker) if not raw.empty else raw
        meta = IndiaTickerLoadMeta(
            ticker=ticker,
            source="yahoo" if not raw.empty else "none",
            yahoo_bars=_yahoo_valid_bar_count(raw),
            bhavcopy_bars=0,
            effective_bars=_effective_bar_count(loaded),
        )

    _store_ticker_data(ticker, start, end_exclusive, loaded, meta)
    _bump_stat(downloaded=True)
    return loaded.copy()


def get_data(ticker: str, start_date, end_date, *, patch_nse: bool = True) -> pd.DataFrame:
    """Backward-compatible alias (``patch_nse`` is ignored; policy is always unified)."""
    del patch_nse
    return get_india_market_data(ticker, start_date, end_date)


def _as_date(value) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def warm_nse_eod_caches(
    start_date,
    end_date,
    *,
    include_cm_bhavcopy: bool = True,
    include_index_archive: bool = True,
) -> int:
    """Pre-load NSE EOD archives for weekdays in range that are not yet cached."""
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        return 0

    today = today_ist()
    end_past = min(end, today - timedelta(days=1))
    if start > end_past:
        return 0

    sessions = 0
    d = start
    while d <= end_past:
        if d.weekday() < 5:
            if include_cm_bhavcopy:
                fetch_bhavcopy(d)
            if include_index_archive:
                fetch_index_close_all(d, quiet=True)
            sessions += 1
        d += timedelta(days=1)
    return sessions


def prepare_india_market_data_range(
    start_date,
    end_date,
    *,
    reset_stats: bool = True,
) -> IndiaMarketDataRunStats:
    """Warm NSE day archives for a date window; reset per-run symbol stats when requested."""
    if reset_stats:
        reset_india_market_data_run_stats()
        _LOAD_META.clear()
    warm_nse_eod_caches(start_date, end_date)
    return get_india_market_data_run_stats()

