"""Technical indicators for Nifty index backtest entry/exit signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from momentum.index.candle_signals import CandleMode, candle_frame, is_bullish_bar, normalize_ohlc

IndicatorKind = Literal["candle", "sma", "ema", "supertrend"]

INDICATOR_LABELS: dict[IndicatorKind, str] = {
    "candle": "Candle color",
    "sma": "SMA",
    "ema": "EMA",
    "supertrend": "Supertrend",
}

DEFAULT_INDICATOR: IndicatorKind = "sma"
DEFAULT_INDICATOR_PERIOD = 20
DEFAULT_SUPERTREND_ATR = 10
DEFAULT_SUPERTREND_MULTIPLIER = 3.0


def indicator_warmup_start(
    display_start: pd.Timestamp,
    *,
    period: int,
    timeframe: Literal["day", "week", "month"] = "day",
) -> pd.Timestamp:
    """Load extra history so MA / Supertrend match chart platforms at display start."""
    p = max(2, int(period))
    if timeframe == "month":
        return display_start - pd.DateOffset(months=max(24, p * 3))
    if timeframe == "week":
        return display_start - pd.Timedelta(weeks=max(52, p * 5))
    return display_start - pd.Timedelta(days=max(120, p * 10))


def resolve_indicator(value: str) -> IndicatorKind:
    raw = (value or "").strip().lower().replace(" ", "_")
    aliases = {
        "candle": "candle",
        "candle_color": "candle",
        "candlestick": "candle",
        "sma": "sma",
        "moving_average": "sma",
        "ma": "sma",
        "ema": "ema",
        "supertrend": "supertrend",
        "super_trend": "supertrend",
    }
    key = aliases.get(raw, raw)
    if key not in INDICATOR_LABELS:
        return DEFAULT_INDICATOR
    return key  # type: ignore[return-value]


def indicator_display(
    kind: IndicatorKind,
    *,
    period: int,
    multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> str:
    label = INDICATOR_LABELS[kind]
    if kind in ("sma", "ema"):
        return f"{label}({period})"
    if kind == "supertrend":
        return f"{label}({period}, {multiplier:g})"
    return label


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def indicator_ohlc(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
) -> pd.DataFrame:
    """OHLC series used to compute an indicator (matches chart candle type)."""
    base = normalize_ohlc(ohlc)
    if indicator == "supertrend" and candle_mode == "heikin_ashi":
        return candle_frame(base, "heikin_ashi")
    return base


def _atr(ohlc: pd.DataFrame, period: int) -> pd.Series:
    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_supertrend(
    ohlc: pd.DataFrame,
    *,
    atr_period: int = DEFAULT_SUPERTREND_ATR,
    multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> pd.DataFrame:
    """Return supertrend line and direction (+1 bullish, -1 bearish).

    Matches TradingView's built-in Supertrend (hl2 source, Wilder ATR/RMA bands).
    """
    base = normalize_ohlc(ohlc)
    if base.empty:
        return pd.DataFrame(columns=["supertrend", "direction"])

    close = base["Close"].astype(float)
    hl2 = (base["High"] + base["Low"]) / 2.0
    atr = _atr(base, atr_period)
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    basic_upper_v = basic_upper.to_numpy(dtype=float)
    basic_lower_v = basic_lower.to_numpy(dtype=float)

    n = len(base)
    final_upper = basic_upper_v.copy()
    final_lower = basic_lower_v.copy()
    supertrend = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    close_v = close.to_numpy(dtype=float)

    for i in range(n):
        if i == 0:
            supertrend[i] = final_lower[i]
            direction[i] = 1
            continue

        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        prev_close = close_v[i - 1]

        if basic_lower_v[i] > prev_lower or prev_close < prev_lower:
            final_lower[i] = basic_lower_v[i]
        else:
            final_lower[i] = prev_lower

        if basic_upper_v[i] < prev_upper or prev_close > prev_upper:
            final_upper[i] = basic_upper_v[i]
        else:
            final_upper[i] = prev_upper

        if direction[i - 1] == -1 and close_v[i] > prev_upper:
            direction[i] = 1
        elif direction[i - 1] == 1 and close_v[i] < prev_lower:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame(
        {"supertrend": supertrend, "direction": direction},
        index=base.index,
    )


def _slice_as_of(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return frame[frame.index <= as_of]


def bullish_signal(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    as_of: pd.Timestamp,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> bool:
    """True when the selected indicator is bullish on/before *as_of*."""
    if indicator == "candle":
        frame = candle_frame(ohlc, candle_mode)
        sliced = _slice_as_of(frame, as_of)
        if sliced.empty:
            return False
        end_idx = len(sliced) - 1
        return is_bullish_bar(sliced["Open"], sliced["Close"], end_idx)

    base = normalize_ohlc(ohlc)
    sliced = _slice_as_of(base, as_of)
    if sliced.empty:
        return False

    close = float(sliced["Close"].iloc[-1])

    if indicator == "sma":
        ma = compute_sma(sliced["Close"], period).iloc[-1]
        return bool(pd.notna(ma) and close > float(ma))

    if indicator == "ema":
        ma = compute_ema(sliced["Close"], period).iloc[-1]
        return bool(pd.notna(ma) and close > float(ma))

    if indicator == "supertrend":
        src = indicator_ohlc(base, indicator=indicator, candle_mode=candle_mode)
        st = compute_supertrend(
            src,
            atr_period=period,
            multiplier=supertrend_multiplier,
        )
        st = _slice_as_of(st, as_of)
        if st.empty:
            return False
        return int(st["direction"].iloc[-1]) == 1

    return False


def bearish_signal(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    as_of: pd.Timestamp,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> bool:
    """True when the selected indicator is bearish on/before *as_of*."""
    if indicator == "candle":
        frame = candle_frame(ohlc, candle_mode)
        sliced = _slice_as_of(frame, as_of)
        if sliced.empty:
            return False
        end_idx = len(sliced) - 1
        return not is_bullish_bar(sliced["Open"], sliced["Close"], end_idx)

    return not bullish_signal(
        ohlc,
        indicator=indicator,
        candle_mode=candle_mode,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
    )


def indicator_series(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> pd.Series | pd.DataFrame:
    """Compute indicator values for chart overlay."""
    base = normalize_ohlc(ohlc)
    if base.empty:
        return pd.Series(dtype=float)

    if indicator == "sma":
        return compute_sma(base["Close"], period)
    if indicator == "ema":
        return compute_ema(base["Close"], period)
    if indicator == "supertrend":
        src = indicator_ohlc(base, indicator=indicator, candle_mode=candle_mode)
        return compute_supertrend(
            src,
            atr_period=period,
            multiplier=supertrend_multiplier,
        )
    return candle_frame(ohlc, candle_mode)["Close"]
