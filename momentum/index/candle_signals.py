"""Standard candlestick and Heikin Ashi trend signals."""

from __future__ import annotations

from typing import Literal

import pandas as pd

CandleMode = Literal["candlestick", "heikin_ashi"]
Timeframe = Literal["day", "week", "month"]

TIMEFRAME_LABELS: dict[Timeframe, str] = {
    "day": "Daily",
    "week": "Weekly",
    "month": "Monthly",
}

_PERIODS_PER_YEAR: dict[Timeframe, int] = {
    "day": 252,
    "week": 52,
    "month": 12,
}


def resample_ohlc(ohlc: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate daily OHLC to weekly (Fri) or month-end bars."""
    base = normalize_ohlc(ohlc)
    if base.empty or timeframe == "day":
        return base
    rule = "W-FRI" if timeframe == "week" else "ME"
    out = base.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    return out.dropna(subset=["Close"])


def _ohlc_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return Open/High/Low/Close columns as a clean DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    out = pd.DataFrame(index=df.index)
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"Missing OHLC column: {col}")
        out[col] = pd.to_numeric(_ohlc_series(df, col), errors="coerce")
    return out.dropna(how="any")


def compute_heikin_ashi(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi OHLC from standard OHLC."""
    base = normalize_ohlc(ohlc)
    if base.empty:
        return base.copy()

    o = base["Open"]
    h = base["High"]
    l = base["Low"]
    c = base["Close"]

    ha_close = (o + h + l + c) / 4.0
    ha_open = pd.Series(index=base.index, dtype=float)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    for i in range(1, len(base)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame(
        {"Open": ha_open, "High": ha_high, "Low": ha_low, "Close": ha_close},
        index=base.index,
    )


def candle_frame(ohlc: pd.DataFrame, mode: CandleMode) -> pd.DataFrame:
    if mode == "heikin_ashi":
        return compute_heikin_ashi(ohlc)
    return normalize_ohlc(ohlc)


def is_bullish_bar(opens: pd.Series, closes: pd.Series, idx: int) -> bool:
    if idx < 0 or idx >= len(opens):
        return False
    o = opens.iloc[idx]
    c = closes.iloc[idx]
    if pd.isna(o) or pd.isna(c):
        return False
    return float(c) > float(o)


def consecutive_bullish_bars(opens: pd.Series, closes: pd.Series, end_idx: int) -> int:
    count = 0
    for i in range(end_idx, -1, -1):
        if is_bullish_bar(opens, closes, i):
            count += 1
        else:
            break
    return count


def bullish_signal(
    ohlc: pd.DataFrame,
    *,
    mode: CandleMode,
    as_of: pd.Timestamp,
    min_bullish_bars: int = 1,
) -> bool:
    """True when the latest bar on/before *as_of* meets the bullish rule."""
    frame = candle_frame(ohlc, mode)
    sliced = frame[frame.index <= as_of]
    if sliced.empty:
        return False
    end_idx = len(sliced) - 1
    if consecutive_bullish_bars(sliced["Open"], sliced["Close"], end_idx) < min_bullish_bars:
        return False
    return True


def bearish_signal(
    ohlc: pd.DataFrame,
    *,
    mode: CandleMode,
    as_of: pd.Timestamp,
) -> bool:
    frame = candle_frame(ohlc, mode)
    sliced = frame[frame.index <= as_of]
    if sliced.empty:
        return False
    end_idx = len(sliced) - 1
    return not is_bullish_bar(sliced["Open"], sliced["Close"], end_idx)


def return_over_bars(closes: pd.Series, as_of: pd.Timestamp, bars: int) -> float:
    sliced = closes[closes.index <= as_of].dropna()
    if len(sliced) < bars + 1:
        return float("-inf")
    end_px = float(sliced.iloc[-1])
    start_px = float(sliced.iloc[-1 - bars])
    if start_px <= 0:
        return float("-inf")
    return end_px / start_px - 1.0
