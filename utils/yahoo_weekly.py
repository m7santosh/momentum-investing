"""Yahoo Finance weekly close loaders for RRG and other momentum tools."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf

from utils.nse_bhavcopy import log_rrg_data_fetch_plan, today_ist


def _daily_to_weekly(daily: pd.Series, min_points: int) -> pd.Series:
    if daily is None or len(daily) < 2:
        return pd.Series(dtype=float)
    daily = pd.to_numeric(daily, errors="coerce").dropna().sort_index()
    weekly = daily.resample("W-FRI").last().dropna()
    return weekly if len(weekly) >= min_points else pd.Series(dtype=float)


def _closes_from_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    """Extract daily adjusted close series per ticker from a yfinance download frame."""
    out: dict[str, pd.Series] = {}
    if raw is None or len(raw) == 0:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            close = raw["Close"]
        elif "Adj Close" in level0:
            close = raw["Adj Close"]
        else:
            close = raw
        if isinstance(close, pd.Series):
            if len(tickers) == 1:
                out[tickers[0]] = close
            return out
        for ticker in tickers:
            if ticker in close.columns:
                out[ticker] = close[ticker]
        return out

    if "Close" in raw.columns:
        series = raw["Close"]
        if isinstance(series, pd.DataFrame):
            for ticker in tickers:
                if ticker in series.columns:
                    out[ticker] = series[ticker]
        elif len(tickers) == 1:
            out[tickers[0]] = series.squeeze()
    return out


def load_yahoo_histories(
    yahoo_tickers: list[str],
    *,
    period: str = "3m",
    min_points: int = 15,
    rrg_window: int = 14,
    quiet: bool = False,
    freq: str = "week",
) -> dict[str, pd.Series]:
    """Yahoo adjusted closes: weekly (W-FRI) or daily, keyed by ticker symbol."""
    from momentum.rrg_core import rrg_normalize_bar_unit

    bar_unit = rrg_normalize_bar_unit(freq)
    unique = list(dict.fromkeys(t.strip() for t in yahoo_tickers if t and t.strip()))
    if not unique:
        return {}

    end = today_ist()
    start, _ = log_rrg_data_fetch_plan(
        period,
        source="Yahoo Finance",
        item_count=len(unique),
        rrg_window=rrg_window,
    )
    if quiet:
        pass

    out: dict[str, pd.Series] = {t: pd.Series(dtype=float) for t in unique}
    end_dl = end + timedelta(days=1)

    try:
        raw = yf.download(
            unique,
            start=start,
            end=end_dl,
            group_by="column",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        daily_map = _closes_from_download(raw, unique)
        for ticker, daily in daily_map.items():
            daily = pd.to_numeric(daily, errors="coerce").dropna().sort_index()
            if bar_unit == "day":
                if len(daily) >= min_points:
                    out[ticker] = daily
            else:
                weekly = _daily_to_weekly(daily, min_points)
                if len(weekly):
                    out[ticker] = weekly
    except Exception as exc:
        print(f"[Yahoo Finance] batch download failed: {exc}")

    for ticker in unique:
        if len(out[ticker]) >= min_points:
            continue
        try:
            raw_one = yf.download(
                ticker,
                start=start,
                end=end_dl,
                auto_adjust=True,
                progress=False,
            )
            daily_map = _closes_from_download(raw_one, [ticker])
            daily = pd.to_numeric(
                daily_map.get(ticker, pd.Series(dtype=float)), errors="coerce"
            ).dropna().sort_index()
            if bar_unit == "day":
                if len(daily) >= min_points:
                    out[ticker] = daily
                    print(
                        f"  [Yahoo Finance] Daily series for {ticker} ({len(daily)} days)"
                    )
            else:
                weekly = _daily_to_weekly(daily, min_points)
                if len(weekly):
                    out[ticker] = weekly
                    print(
                        f"  [Yahoo Finance] Weekly series for {ticker} ({len(weekly)} wks)"
                    )
        except Exception as exc:
            print(f"  [Yahoo Finance] {ticker}: {exc}")

    return out


def load_yahoo_weekly_histories(*args, **kwargs) -> dict[str, pd.Series]:
    """Backward-compatible alias for weekly Yahoo loads."""
    kwargs.setdefault("freq", "week")
    return load_yahoo_histories(*args, **kwargs)


def load_yahoo_histories_range(
    yahoo_tickers: list[str],
    start_date,
    end_date,
    *,
    min_points: int = 1,
    quiet: bool = True,
    freq: str = "week",
) -> dict[str, pd.Series]:
    """Yahoo adjusted closes between ``start_date`` and ``end_date`` (inclusive)."""
    from datetime import date as date_cls

    from momentum.rrg_core import rrg_normalize_bar_unit

    bar_unit = rrg_normalize_bar_unit(freq)
    unique = list(dict.fromkeys(t.strip() for t in yahoo_tickers if t and t.strip()))
    if not unique:
        return {}
    start = pd.Timestamp(start_date).date() if not isinstance(start_date, date_cls) else start_date
    end = pd.Timestamp(end_date).date() if not isinstance(end_date, date_cls) else end_date
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    end_dl = end + timedelta(days=1)
    out: dict[str, pd.Series] = {t: pd.Series(dtype=float) for t in unique}

    batch_size = 40
    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end_dl,
                group_by="column",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            daily_map = _closes_from_download(raw, batch)
            for ticker, daily in daily_map.items():
                daily = pd.to_numeric(daily, errors="coerce").dropna().sort_index()
                if bar_unit == "day":
                    if len(daily) >= min_points:
                        out[ticker] = daily
                else:
                    weekly = _daily_to_weekly(daily, min_points)
                    if len(weekly):
                        out[ticker] = weekly
        except Exception as exc:
            if not quiet:
                print(f"[Yahoo Finance] batch failed: {exc}")

    for ticker in unique:
        if len(out[ticker]) >= min_points:
            continue
        try:
            raw_one = yf.download(
                ticker,
                start=start,
                end=end_dl,
                auto_adjust=True,
                progress=False,
            )
            daily_map = _closes_from_download(raw_one, [ticker])
            daily = pd.to_numeric(
                daily_map.get(ticker, pd.Series(dtype=float)), errors="coerce"
            ).dropna().sort_index()
            if bar_unit == "day":
                if len(daily) >= min_points:
                    out[ticker] = daily
            else:
                weekly = _daily_to_weekly(daily, min_points)
                if len(weekly):
                    out[ticker] = weekly
        except Exception as exc:
            if not quiet:
                print(f"[Yahoo Finance] {ticker}: {exc}")

    if not quiet and unique:
        print(
            f"  [Yahoo Finance] {start:%Y-%m-%d}..{end:%Y-%m-%d}: "
            f"{len(unique)} ticker(s)"
        )
    return out
