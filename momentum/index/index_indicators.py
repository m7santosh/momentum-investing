"""Technical indicators for Nifty index backtest entry/exit signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from momentum.index.candle_signals import (
    CandleMode,
    Timeframe,
    candle_frame,
    is_bullish_bar,
    normalize_ohlc,
    ohlc_uses_precomputed_ha,
)

IndicatorKind = Literal["candle", "sma", "ema", "supertrend"]

INDICATOR_LABELS: dict[IndicatorKind, str] = {
    "candle": "Candle color",
    "sma": "SMA",
    "ema": "EMA",
    "supertrend": "Supertrend",
}

DEFAULT_INDICATOR: IndicatorKind = "sma"
DEFAULT_INDICATOR_PERIOD = 20
DEFAULT_SUPERTREND_ATR = 9
DEFAULT_SUPERTREND_MULTIPLIER = 2.0


def indicator_warmup_start(
    display_start: pd.Timestamp,
    *,
    period: int,
    timeframe: Literal["day", "week", "month"] = "day",
) -> pd.Timestamp:
    """Load extra history so MA / Supertrend match chart platforms at display start."""
    p = max(2, int(period))
    if timeframe == "month":
        return display_start - pd.DateOffset(months=max(36, p * 6))
    if timeframe == "week":
        return display_start - pd.Timedelta(weeks=max(104, p * 12))
    return display_start - pd.Timedelta(days=max(252, p * 15))


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
    timeframe: Timeframe = "day",
) -> pd.DataFrame:
    """OHLC series used to compute an indicator (HA OHLC when chart mode is HA)."""
    base = normalize_ohlc(ohlc)
    if ohlc_uses_precomputed_ha(timeframe, candle_mode):
        return base
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

    Matches TradingView ``ta.supertrend`` (Pine Script reference implementation).
    """
    base = normalize_ohlc(ohlc)
    if base.empty:
        return pd.DataFrame(columns=["supertrend", "direction"])

    src = (base["High"] + base["Low"]) / 2.0
    close = base["Close"].astype(float)
    atr = _atr(base, atr_period)

    lower = (src - multiplier * atr).to_numpy(dtype=float).copy()
    upper = (src + multiplier * atr).to_numpy(dtype=float).copy()
    close_v = close.to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float)

    n = len(base)
    # Pine: direction -1 = up/bull, +1 = down/bear
    pine_dir = np.zeros(n, dtype=int)
    st = np.full(n, np.nan)

    for i in range(n):
        if i > 0:
            prev_lower = lower[i - 1]
            prev_upper = upper[i - 1]
            if lower[i] > prev_lower or close_v[i - 1] < prev_lower:
                pass
            else:
                lower[i] = prev_lower
            if upper[i] < prev_upper or close_v[i - 1] > prev_upper:
                pass
            else:
                upper[i] = prev_upper

        prev_st = st[i - 1] if i > 0 else np.nan
        prev_upper = upper[i - 1] if i > 0 else upper[i]

        if i == 0 or np.isnan(atr_v[i - 1]):
            pine_dir[i] = 1
        elif prev_st == prev_upper:
            pine_dir[i] = -1 if close_v[i] > upper[i] else 1
        else:
            pine_dir[i] = 1 if close_v[i] < lower[i] else -1

        st[i] = lower[i] if pine_dir[i] == -1 else upper[i]

    direction = np.where(pine_dir == -1, 1, -1)
    return pd.DataFrame(
        {"supertrend": st, "direction": direction},
        index=base.index,
    )


def _slice_as_of(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return frame[frame.index <= as_of]


def _uses_weekly_ha_supertrend(
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    timeframe: Timeframe,
) -> bool:
    return indicator == "supertrend" and candle_mode == "heikin_ashi" and timeframe == "week"


def _is_monday_holiday_week_bar(bar_ts: pd.Timestamp) -> bool:
    """Weekly bar dated Tuesday — first NSE session after a Monday holiday."""
    return pd.Timestamp(bar_ts).normalize().dayofweek == 1


def _supertrend_direction_at(
    ohlc: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    period: int,
    supertrend_multiplier: float,
    candle_mode: CandleMode,
    timeframe: Timeframe,
) -> int | None:
    src = indicator_ohlc(
        ohlc,
        indicator="supertrend",
        candle_mode=candle_mode,
        timeframe=timeframe,
    )
    st = compute_supertrend(
        src,
        atr_period=period,
        multiplier=supertrend_multiplier,
    )
    st = _slice_as_of(st, as_of)
    if st.empty:
        return None
    return int(st["direction"].iloc[-1])


def supertrend_weekly_ha_directions(
    ha_ohlc: pd.DataFrame,
    std_ohlc: pd.DataFrame,
    *,
    period: int = DEFAULT_SUPERTREND_ATR,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> pd.Series:
    """Weekly HA Supertrend chart/backtest direction (matches TradingView in/out coloring)."""
    ha = normalize_ohlc(ha_ohlc)
    std = normalize_ohlc(std_ohlc)
    if ha.empty or std.empty:
        return pd.Series(dtype=int)

    in_position = False
    out: list[int] = []
    for ts in ha.index:
        as_of = pd.Timestamp(ts)
        bull = _weekly_ha_supertrend_bullish(
            ha,
            std,
            as_of=as_of,
            period=period,
            supertrend_multiplier=supertrend_multiplier,
        )
        bear = _weekly_ha_supertrend_bearish(
            ha,
            std,
            as_of=as_of,
            period=period,
            supertrend_multiplier=supertrend_multiplier,
        )
        if not in_position and bull:
            in_position = True
            out.append(1)
        elif in_position and bear:
            in_position = False
            out.append(-1)
        elif in_position:
            out.append(1)
        else:
            out.append(-1)
    return pd.Series(out, index=ha.index, dtype=int)


def supertrend_weekly_ha_frame(
    ha_ohlc: pd.DataFrame,
    std_ohlc: pd.DataFrame,
    *,
    period: int = DEFAULT_SUPERTREND_ATR,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
) -> pd.DataFrame:
    """HA Supertrend line for charts with TV-aligned bull/bear direction."""
    ha = normalize_ohlc(ha_ohlc)
    if ha.empty:
        return pd.DataFrame(columns=["supertrend", "direction"])

    line = compute_supertrend(ha, atr_period=period, multiplier=supertrend_multiplier)
    directions = supertrend_weekly_ha_directions(
        ha,
        std_ohlc,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
    )
    return pd.DataFrame(
        {"supertrend": line["supertrend"], "direction": directions.reindex(line.index)},
        index=line.index,
    )


def _weekly_ha_supertrend_bullish(
    ha_ohlc: pd.DataFrame,
    std_ohlc: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    period: int,
    supertrend_multiplier: float,
) -> bool:
    ha_dir = _supertrend_direction_at(
        ha_ohlc,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
        candle_mode="heikin_ashi",
        timeframe="week",
    )
    std_dir = _supertrend_direction_at(
        std_ohlc,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
        candle_mode="candlestick",
        timeframe="week",
    )
    if ha_dir is None or std_dir is None:
        return False
    return ha_dir == 1 and std_dir == 1


def _weekly_ha_supertrend_bearish(
    ha_ohlc: pd.DataFrame,
    std_ohlc: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    period: int,
    supertrend_multiplier: float,
) -> bool:
    ha_dir = _supertrend_direction_at(
        ha_ohlc,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
        candle_mode="heikin_ashi",
        timeframe="week",
    )
    if ha_dir is None:
        return False
    if ha_dir == -1:
        return True
    if _is_monday_holiday_week_bar(as_of):
        std_dir = _supertrend_direction_at(
            std_ohlc,
            as_of=as_of,
            period=period,
            supertrend_multiplier=supertrend_multiplier,
            candle_mode="candlestick",
            timeframe="week",
        )
        return std_dir == -1
    return False


def bullish_signal(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    as_of: pd.Timestamp,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
    timeframe: Timeframe = "day",
    standard_ohlc: pd.DataFrame | None = None,
) -> bool:
    """True when the selected indicator is bullish on/before *as_of*."""
    if indicator == "candle":
        if ohlc_uses_precomputed_ha(timeframe, candle_mode):
            frame = normalize_ohlc(ohlc)
        else:
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
        if (
            _uses_weekly_ha_supertrend(indicator, candle_mode, timeframe)
            and standard_ohlc is not None
            and not standard_ohlc.empty
        ):
            return _weekly_ha_supertrend_bullish(
                base,
                standard_ohlc,
                as_of=as_of,
                period=period,
                supertrend_multiplier=supertrend_multiplier,
            )
        src = indicator_ohlc(
            base,
            indicator=indicator,
            candle_mode=candle_mode,
            timeframe=timeframe,
        )
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
    timeframe: Timeframe = "day",
    standard_ohlc: pd.DataFrame | None = None,
) -> bool:
    """True when the selected indicator is bearish on/before *as_of*."""
    if indicator == "candle":
        if ohlc_uses_precomputed_ha(timeframe, candle_mode):
            frame = normalize_ohlc(ohlc)
        else:
            frame = candle_frame(ohlc, candle_mode)
        sliced = _slice_as_of(frame, as_of)
        if sliced.empty:
            return False
        end_idx = len(sliced) - 1
        return not is_bullish_bar(sliced["Open"], sliced["Close"], end_idx)

    if (
        indicator == "supertrend"
        and _uses_weekly_ha_supertrend(indicator, candle_mode, timeframe)
        and standard_ohlc is not None
        and not standard_ohlc.empty
    ):
        return _weekly_ha_supertrend_bearish(
            normalize_ohlc(ohlc),
            standard_ohlc,
            as_of=as_of,
            period=period,
            supertrend_multiplier=supertrend_multiplier,
        )

    return not bullish_signal(
        ohlc,
        indicator=indicator,
        candle_mode=candle_mode,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
        timeframe=timeframe,
        standard_ohlc=standard_ohlc,
    )


def entry_signal(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    as_of: pd.Timestamp,
    prev_bull: bool | None,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
    timeframe: Timeframe = "day",
    standard_ohlc: pd.DataFrame | None = None,
) -> bool:
    """True on a bullish flip (matches chart entry markers)."""
    if prev_bull is None:
        return False
    bull = bullish_signal(
        ohlc,
        indicator=indicator,
        candle_mode=candle_mode,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
        timeframe=timeframe,
        standard_ohlc=standard_ohlc,
    )
    return bull and not prev_bull


def exit_signal(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    as_of: pd.Timestamp,
    prev_bull: bool | None,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
    timeframe: Timeframe = "day",
    standard_ohlc: pd.DataFrame | None = None,
) -> bool:
    """True on a bearish flip (matches chart exit markers)."""
    if prev_bull is None:
        return False
    bull = bullish_signal(
        ohlc,
        indicator=indicator,
        candle_mode=candle_mode,
        as_of=as_of,
        period=period,
        supertrend_multiplier=supertrend_multiplier,
        timeframe=timeframe,
        standard_ohlc=standard_ohlc,
    )
    return not bull and prev_bull


def indicator_series(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
    timeframe: Timeframe = "day",
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
        src = indicator_ohlc(
            base,
            indicator=indicator,
            candle_mode=candle_mode,
            timeframe=timeframe,
        )
        return compute_supertrend(
            src,
            atr_period=period,
            multiplier=supertrend_multiplier,
        )
    if ohlc_uses_precomputed_ha(timeframe, candle_mode):
        return base["Close"]
    return candle_frame(ohlc, candle_mode)["Close"]
