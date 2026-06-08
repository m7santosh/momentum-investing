"""Volume Breakout sheet calculations — mirrors Google Sheet formulas."""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from utils.nse_bhavcopy import fetch_nse_live_quotes

_DMA_CALENDAR_DAYS = {50: 75, 100: 150, 200: 300}

_IST = timezone(timedelta(hours=5, minutes=30))
_FILTER_KEYWORDS = "BEES|ETF|GOLD|LIQUID|CASE|SILVER|LIQ|GSEC|MOSMALL"
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

TOP_250_VOLUME_COLUMNS: tuple[str, ...] = (
    "NSE Code",
    "Volume",
    "Close Price",
    "CMP",
    "50 DMA",
    "100 DMA",
    "200 DMA",
    "Output",
    "Difference from 200 DMA",
    "CAR",
)

TOP_250_TURNOVER_COLUMNS: tuple[str, ...] = (
    "NSE Code",
    "Turnover",
    "Close Price",
    "CMP",
    "50 DMA",
    "100 DMA",
    "200 DMA",
    "Output",
    "Difference from 200 DMA",
    "CAR",
)

FINAL_LIST_VOLUME_COLUMNS: tuple[str, ...] = (
    "NSE Code",
    "Volume",
    "Previous Close",
    "CMP",
    "Difference from 200 DMA",
    "CAR",
)

FINAL_LIST_TURNOVER_COLUMNS: tuple[str, ...] = (
    "NSE Code",
    "Volume",
    "Previous Close",
    "CMP",
    "Difference from 200 DMA",
    "CAR",
)


@dataclass(frozen=True)
class SymbolEnrichment:
    nse_code: str
    close_price: float
    cmp: float
    dma50: float | None
    dma100: float | None
    dma200: float | None
    output: str
    diff_200_dma: float | None
    car: str


@dataclass(frozen=True)
class SheetRow:
    nse_code: str
    metric: float
    close_price: float
    cmp: float
    dma50: float | None
    dma100: float | None
    dma200: float | None
    output: str
    diff_200_dma: float | None
    car: str

    def top250_values(self) -> tuple:
        return (
            self.nse_code,
            self.metric,
            self.close_price,
            self.cmp,
            self.dma50,
            self.dma100,
            self.dma200,
            self.output,
            self.diff_200_dma,
            self.car,
        )

    def final_list_values(self) -> tuple:
        return (
            self.nse_code,
            self.metric,
            self.close_price,
            self.cmp,
            self.diff_200_dma,
            self.car,
        )


@dataclass(frozen=True)
class Top250Snapshot:
    volume_rows: list[SheetRow]
    turnover_rows: list[SheetRow]
    final_volume: list[SheetRow]
    final_turnover: list[SheetRow]
    data_date: str
    status: str


def _today_ist() -> date:
    return datetime.now(_IST).date()


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    sym_col, close_col, series_col, vol_col, turnover_col, _high_col = (
        _resolve_columns_extended(df)
    )
    return sym_col, close_col, series_col, vol_col, turnover_col


def _resolve_columns_extended(
    df: pd.DataFrame,
) -> tuple[str, str, str, str, str, str]:
    sym_col = "TckrSymb" if "TckrSymb" in df.columns else "SYMBOL"
    close_col = "ClsPric" if "ClsPric" in df.columns else "CLOSE"
    series_col = "SctySrs" if "SctySrs" in df.columns else "SERIES"
    high_col = "HghPric" if "HghPric" in df.columns else "HIGH"

    vol_col = "TtlTradgVol"
    for c in ("TtlTradgVol", "TOTTRDQTY", "TtlTrdQty", "TotTrdQty"):
        if c in df.columns:
            vol_col = c
            break

    turnover_col = "TtlTrfVal"
    for c in ("TtlTrfVal", "TOTTRDVAL", "TtlTrdVal", "TotTrdVal"):
        if c in df.columns:
            turnover_col = c
            break

    return sym_col, close_col, series_col, vol_col, turnover_col, high_col


def _filter_eq_stocks(df: pd.DataFrame) -> pd.DataFrame:
    sym_col, _, series_col, _, _ = _resolve_columns(df)
    if series_col in df.columns:
        df = df[df[series_col].astype(str).str.strip() == "EQ"]
    return df[~df[sym_col].astype(str).str.contains(_FILTER_KEYWORDS, case=False, na=False)]


def _base_rows_from_bhavcopy_df(
    df: pd.DataFrame,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    sym_col, close_col, _, vol_col, turnover_col = _resolve_columns(df)
    df = _filter_eq_stocks(df)

    df_vol = df.sort_values(by=vol_col, ascending=False).head(250)
    volume_rows = [
        (str(row[sym_col]).strip(), float(row[vol_col]), float(row[close_col]))
        for _, row in df_vol.iterrows()
    ]

    df_turnover = df.sort_values(by=turnover_col, ascending=False).head(250)
    turnover_rows = [
        (str(row[sym_col]).strip(), float(row[turnover_col]), float(row[close_col]))
        for _, row in df_turnover.iterrows()
    ]
    return volume_rows, turnover_rows


@dataclass(frozen=True)
class BhavcopyDayData:
    trade_date: date
    volume_top250: list[tuple[str, float, float]]
    turnover_top250: list[tuple[str, float, float]]
    ohlc: dict[str, dict[str, float]]


_BHAVCOPY_DAY_CACHE: dict[date, BhavcopyDayData | None] = {}


def _parse_bhavcopy_day(date_obj: date, df: pd.DataFrame) -> BhavcopyDayData:
    sym_col, close_col, _, vol_col, turnover_col, high_col = _resolve_columns_extended(df)
    filtered = _filter_eq_stocks(df)
    ohlc: dict[str, dict[str, float]] = {}
    for _, row in filtered.iterrows():
        sym = str(row[sym_col]).strip()
        try:
            ohlc[sym] = {
                "close": float(row[close_col]),
                "high": float(row[high_col]),
                "volume": float(row[vol_col]),
                "turnover": float(row[turnover_col]),
            }
        except (ValueError, KeyError, TypeError):
            continue
    volume_top250, turnover_top250 = _base_rows_from_bhavcopy_df(filtered)
    return BhavcopyDayData(
        trade_date=date_obj,
        volume_top250=volume_top250,
        turnover_top250=turnover_top250,
        ohlc=ohlc,
    )


def _download_bhavcopy_df(date_obj: date) -> pd.DataFrame | None:
    from utils.nse_bhavcopy import _bhavcopy_urls, _session

    sess = _session()
    for url in _bhavcopy_urls(date_obj):
        try:
            response = sess.get(url, timeout=30)
            if response.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


def load_bhavcopy_day(date_obj: date) -> BhavcopyDayData | None:
    """Cached NSE bhavcopy for one session (top-250 lists + full EQ OHLC)."""
    if date_obj in _BHAVCOPY_DAY_CACHE:
        return _BHAVCOPY_DAY_CACHE[date_obj]

    try:
        df = _download_bhavcopy_df(date_obj)
        if df is None or df.empty:
            _BHAVCOPY_DAY_CACHE[date_obj] = None
            return None
        result = _parse_bhavcopy_day(date_obj, df)
        _BHAVCOPY_DAY_CACHE[date_obj] = result
        return result
    except Exception:
        _BHAVCOPY_DAY_CACHE[date_obj] = None
        return None


def _fetch_bhavcopy_for_date(date_obj: date) -> tuple[list, list] | None:
    day = load_bhavcopy_day(date_obj)
    if day is None:
        return None
    return day.volume_top250, day.turnover_top250


def _dma_google_style(
    closes: pd.Series,
    cmp: float,
    data_date: date,
    n: int,
) -> float | None:
    """Match GOOGLEFINANCE price window + QUERY limit N (partial history allowed)."""
    cal_days = _DMA_CALENDAR_DAYS[n]
    start = pd.Timestamp(data_date) - pd.Timedelta(days=cal_days)
    end = pd.Timestamp(data_date)
    s = closes[(closes.index >= start) & (closes.index <= end)]
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    if s.empty:
        return None

    s = s.copy()
    if s.index[-1].normalize() == end.normalize():
        s.iloc[-1] = cmp
    else:
        s = pd.concat([s, pd.Series([cmp], index=[end])])

    recent = s.sort_index(ascending=False).iloc[: min(n, len(s))]
    return float(recent.mean())


def _diff_from_200_dma(cmp: float, dma200: float | None) -> float | None:
    if dma200 is None or dma200 == 0:
        return None
    return ((cmp - dma200) * 100) / dma200


def _cmp_above_dma(cmp: float, dma: float) -> bool:
    """Allow minor Yahoo vs Google Finance DMA drift (e.g. GRSE turnover list)."""
    return cmp >= dma * 0.998


def _cmp_below_dma(cmp: float, dma: float) -> bool:
    return cmp <= dma * 1.002


def _output_label(cmp: float, dma50: float | None, dma100: float | None, dma200: float | None, diff: float | None) -> str:
    if None in (dma50, dma100, dma200, diff):
        return "Unconfirmed"
    assert dma50 is not None and dma100 is not None and dma200 is not None and diff is not None
    if (
        _cmp_above_dma(cmp, dma50)
        and _cmp_above_dma(cmp, dma100)
        and _cmp_above_dma(cmp, dma200)
        and 0.01 <= diff <= 10
    ):
        # Yahoo vs Google Finance DMA can diverge near the 10% bull-run cap (e.g. YESBANK).
        if diff >= 9.95:
            return "Unconfirmed"
        return "In Bull Run"
    if (
        _cmp_below_dma(cmp, dma50)
        and _cmp_below_dma(cmp, dma100)
        and _cmp_below_dma(cmp, dma200)
        and -10 <= diff <= -0.01
    ):
        return "In Bear Run"
    return "Unconfirmed"


def _compute_car(high: pd.Series, close: pd.Series) -> str:
    high = pd.to_numeric(high, errors="coerce").dropna()
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return "Short History"

    cutoff = close.index.max() - pd.Timedelta(days=365)
    tail_high = high.loc[high.index >= cutoff]
    if tail_high.empty:
        tail_high = high
    high_date = tail_high.idxmax()

    raw = close.loc[high_date:]
    prices = raw.iloc[1:].to_numpy(dtype=float)
    if len(prices) < 2:
        fallback = close.iloc[-10:]
        prices = fallback.iloc[1:].to_numpy(dtype=float)

    count_rows = len(prices)
    if count_rows < 10:
        return "Short History"

    cum_avg = np.array([float(np.mean(prices[:n])) for n in range(1, count_rows + 1)])
    last_10 = cum_avg[-10:]
    check = sum(1 for i in range(9) if last_10[i + 1] > last_10[i])
    return "Buy/Average Out" if check == 9 else "Avoid/Hold"


def _extract_ohlc(raw: pd.DataFrame, yahoo_ticker: str) -> tuple[pd.Series, pd.Series] | None:
    if raw is None or raw.empty:
        return None

    sub: pd.DataFrame | None = None
    if isinstance(raw.columns, pd.MultiIndex):
        tickers = raw.columns.get_level_values(0).unique()
        if yahoo_ticker in tickers:
            sub = raw[yahoo_ticker]
        elif yahoo_ticker in raw.columns.get_level_values(1):
            sub = raw.xs(yahoo_ticker, axis=1, level=1)
    elif "High" in raw.columns and "Close" in raw.columns:
        sub = raw

    if sub is None or "High" not in sub.columns or "Close" not in sub.columns:
        return None

    high = pd.to_numeric(sub["High"], errors="coerce").dropna()
    close = pd.to_numeric(sub["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    return high, close


def _yahoo_cmp(sym: str) -> float | None:
    ticker = yf.Ticker(f"{sym}.NS")
    for attempt in range(2):
        try:
            price = ticker.info.get("regularMarketPrice")
            if price is not None:
                return float(price)
        except Exception:
            pass
        try:
            price = ticker.fast_info.get("last_price")
            if price is not None:
                return float(price)
        except Exception:
            pass
        if attempt == 0:
            time.sleep(0.35)
    return None


def _fetch_cmp_prices(
    symbols: list[str],
    live: dict[str, dict],
    *,
    close_by_symbol: dict[str, float],
    histories: dict[str, tuple[pd.Series, pd.Series]],
    data_date: date,
) -> dict[str, float]:
    """CMP: NSE live, then Yahoo regularMarketPrice (matches Google Sheet), else history/close."""
    out: dict[str, float] = {}
    for sym in symbols:
        quote = live.get(sym)
        if quote and quote.get("close"):
            out[sym] = float(quote["close"])

    for sym in symbols:
        if sym in out:
            continue
        price = _yahoo_cmp(sym)
        if price is not None:
            out[sym] = price
        else:
            out[sym] = _cmp_from_history(
                histories, sym, data_date, close_by_symbol[sym]
            )
        time.sleep(0.15)
    return out


def _download_one_history(sym: str) -> tuple[pd.Series, pd.Series] | None:
    try:
        raw = yf.download(
            f"{sym}.NS",
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    return _extract_ohlc(raw, f"{sym}.NS")


def _download_histories(symbols: list[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    tickers = [f"{s}.NS" for s in symbols]
    out: dict[str, tuple[pd.Series, pd.Series]] = {}
    chunk_size = 20

    def _download_chunk(chunk_syms: list[str], chunk_tickers: list[str]) -> None:
        raw = yf.download(
            chunk_tickers,
            period="2y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return
        for sym, yahoo in zip(chunk_syms, chunk_tickers, strict=True):
            extracted = _extract_ohlc(raw, yahoo)
            if extracted is not None:
                out[sym] = extracted

    for i in range(0, len(tickers), chunk_size):
        chunk_syms = symbols[i : i + chunk_size]
        chunk_tickers = tickers[i : i + chunk_size]
        for attempt in range(3):
            try:
                _download_chunk(chunk_syms, chunk_tickers)
                break
            except Exception:
                time.sleep(2.0 * (attempt + 1))
        time.sleep(1.2)

    missing = [sym for sym in symbols if sym not in out]
    for sym in missing:
        extracted = _download_one_history(sym)
        if extracted is not None:
            out[sym] = extracted
        time.sleep(0.6)
    return out


def _cmp_from_history(
    histories: dict[str, tuple[pd.Series, pd.Series]],
    symbol: str,
    data_date: date,
    fallback: float,
) -> float:
    hist = histories.get(symbol)
    if hist is None:
        return fallback
    _, close = hist
    closes = pd.to_numeric(close, errors="coerce").dropna()
    closes = closes[closes.index <= pd.Timestamp(data_date)]
    if closes.empty:
        return fallback
    return float(closes.iloc[-1])


def enrich_symbol_eod(
    symbol: str,
    close_price: float,
    *,
    data_date: date,
    closes: pd.Series,
    highs: pd.Series,
) -> SymbolEnrichment:
    """Point-in-time enrichment using bhavcopy EOD close as CMP (backtest)."""
    cmp = close_price
    close = pd.to_numeric(closes, errors="coerce").dropna()
    close = close[close.index <= pd.Timestamp(data_date)]
    high = pd.to_numeric(highs, errors="coerce").dropna()
    high = high[high.index <= pd.Timestamp(data_date)]
    if close.empty:
        return SymbolEnrichment(
            nse_code=symbol,
            close_price=close_price,
            cmp=cmp,
            dma50=None,
            dma100=None,
            dma200=None,
            output="Unconfirmed",
            diff_200_dma=None,
            car="Short History",
        )

    dma50 = _dma_google_style(close, cmp, data_date, 50)
    dma100 = _dma_google_style(close, cmp, data_date, 100)
    dma200 = _dma_google_style(close, cmp, data_date, 200)
    diff = _diff_from_200_dma(cmp, dma200)
    output = _output_label(cmp, dma50, dma100, dma200, diff)
    car = _compute_car(high, close)
    return SymbolEnrichment(
        nse_code=symbol,
        close_price=close_price,
        cmp=cmp,
        dma50=dma50,
        dma100=dma100,
        dma200=dma200,
        output=output,
        diff_200_dma=diff,
        car=car,
    )


def _enrich_symbol(
    symbol: str,
    close_price: float,
    *,
    data_date: date,
    histories: dict[str, tuple[pd.Series, pd.Series]],
    cmp_prices: dict[str, float],
) -> SymbolEnrichment:
    cmp = cmp_prices[symbol]
    hist = histories.get(symbol)
    if hist is None:
        return SymbolEnrichment(
            nse_code=symbol,
            close_price=close_price,
            cmp=cmp,
            dma50=None,
            dma100=None,
            dma200=None,
            output="Unconfirmed",
            diff_200_dma=None,
            car="Short History",
        )

    high, close = hist
    return enrich_symbol_eod(
        symbol,
        close_price,
        data_date=data_date,
        closes=close,
        highs=high,
    )


def _build_enrichment_cache(
    close_by_symbol: dict[str, float],
    *,
    data_date: date,
    histories: dict[str, tuple[pd.Series, pd.Series]],
    cmp_prices: dict[str, float],
) -> dict[str, SymbolEnrichment]:
    return {
        symbol: _enrich_symbol(
            symbol,
            close_price,
            data_date=data_date,
            histories=histories,
            cmp_prices=cmp_prices,
        )
        for symbol, close_price in close_by_symbol.items()
    }


def sheet_row(enrichment: SymbolEnrichment, metric: float) -> SheetRow:
    return SheetRow(
        nse_code=enrichment.nse_code,
        metric=metric,
        close_price=enrichment.close_price,
        cmp=enrichment.cmp,
        dma50=enrichment.dma50,
        dma100=enrichment.dma100,
        dma200=enrichment.dma200,
        output=enrichment.output,
        diff_200_dma=enrichment.diff_200_dma,
        car=enrichment.car,
    )


def _rows_from_base(
    base_rows: list[tuple[str, float, float]],
    cache: dict[str, SymbolEnrichment],
) -> list[SheetRow]:
    return [sheet_row(cache[symbol], metric) for symbol, metric, _ in base_rows]


def pick_final_list(rows: list[SheetRow]) -> list[SheetRow]:
    picked = [
        r
        for r in rows
        if r.output == "In Bull Run" and r.car == "Buy/Average Out"
    ]
    return sorted(picked, key=lambda r: r.metric, reverse=True)


def _final_list(rows: list[SheetRow]) -> list[SheetRow]:
    return pick_final_list(rows)


def fetch_top250_snapshot(*, lookback_days: int = 7) -> Top250Snapshot:
    start = _today_ist()
    volume_base: list[tuple[str, float, float]] | None = None
    turnover_base: list[tuple[str, float, float]] | None = None
    data_date = ""

    for i in range(lookback_days):
        test_date = start - timedelta(days=i)
        if test_date.weekday() >= 5:
            continue
        result = _fetch_bhavcopy_for_date(test_date)
        if result is not None:
            volume_base, turnover_base = result
            data_date = test_date.strftime("%d-%b-%Y")
            break

    if volume_base is None or turnover_base is None:
        raise RuntimeError("No NSE bhavcopy found for the last 7 trading days.")

    symbols = list(dict.fromkeys([r[0] for r in volume_base] + [r[0] for r in turnover_base]))
    close_by_symbol: dict[str, float] = {}
    for sym, _, close in volume_base + turnover_base:
        close_by_symbol.setdefault(sym, close)
    live = fetch_nse_live_quotes()
    trade_date = datetime.strptime(data_date, "%d-%b-%Y").date()
    histories = _download_histories(symbols)
    cmp_prices = _fetch_cmp_prices(
        symbols,
        live,
        close_by_symbol=close_by_symbol,
        histories=histories,
        data_date=trade_date,
    )
    enrichment_cache = _build_enrichment_cache(
        close_by_symbol,
        data_date=trade_date,
        histories=histories,
        cmp_prices=cmp_prices,
    )

    volume_rows = _rows_from_base(volume_base, enrichment_cache)
    turnover_rows = _rows_from_base(turnover_base, enrichment_cache)

    ist_now = datetime.now(_IST).strftime("%d-%b %H:%M")
    status = f"Data Date: {data_date} | Last Update: {ist_now} (IST)"

    return Top250Snapshot(
        volume_rows=volume_rows,
        turnover_rows=turnover_rows,
        final_volume=_final_list(volume_rows),
        final_turnover=_final_list(turnover_rows),
        data_date=data_date,
        status=status,
    )
