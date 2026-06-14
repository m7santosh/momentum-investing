"""Matplotlib candlestick / Heikin Ashi charts for index backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from momentum.index.candle_signals import CandleMode, candle_frame, normalize_ohlc
from momentum.index.index_indicators import (
    DEFAULT_INDICATOR_PERIOD,
    DEFAULT_SUPERTREND_MULTIPLIER,
    IndicatorKind,
    bullish_signal,
    compute_ema,
    compute_sma,
    compute_supertrend,
    indicator_display,
    indicator_ohlc,
)

_BULL_COLOR = "#26a69a"
_BEAR_COLOR = "#ef5350"
_WICK_COLOR = "#424242"
_INDICATOR_COLOR = "#FF9800"
_ST_BULL_COLOR = "#26a69a"
_ST_BEAR_COLOR = "#ef5350"


@dataclass
class CandleHoverData:
    ax: Any
    frame: pd.DataFrame
    xnums: np.ndarray
    bar_width: float
    chart_mode_label: str
    indicator: IndicatorKind
    indicator_label: str
    indicator_period: int
    supertrend_multiplier: float
    ma_line: pd.Series | None = None
    supertrend: pd.DataFrame | None = None


class CandleChartHover:
    """Show OHLC details for the bar under the cursor (via *set_detail* callback)."""

    DEFAULT_DETAIL = "Hover a bar for OHLC details"

    def __init__(self, canvas, *, set_detail: Callable[[str], None] | None = None) -> None:
        self._canvas = canvas
        self._set_detail = set_detail or (lambda _text: None)
        self._data: CandleHoverData | None = None
        self._last_i: int | None = None
        self._cid = canvas.mpl_connect("motion_notify_event", self._on_move)

    def update(self, data: CandleHoverData | None) -> None:
        self._data = data
        self._last_i = None
        if data is None:
            self._set_detail(self.DEFAULT_DETAIL)
            return
        self._set_detail(self.DEFAULT_DETAIL)

    def _hide(self) -> None:
        self._last_i = None
        self._set_detail(self.DEFAULT_DETAIL)

    def _format_bar(self, data: CandleHoverData, i: int) -> str:
        row = data.frame.iloc[i]
        ts = pd.Timestamp(data.frame.index[i])
        o, h, l, c = (float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]))
        if i > 0:
            prev_c = float(data.frame.iloc[i - 1]["Close"])
            if prev_c > 0:
                chg_abs = c - prev_c
                chg_pct = (c / prev_c - 1.0) * 100.0
                chg_part = f"Chg {chg_pct:+.2f}% ({chg_abs:+,.2f})"
            else:
                chg_part = "Chg —"
        else:
            chg_part = "Chg —"
        parts = [
            ts.strftime("%d-%m-%Y"),
            f"O {o:,.2f}  H {h:,.2f}  L {l:,.2f}  C {c:,.2f}",
            chg_part,
        ]
        if data.indicator in ("sma", "ema") and data.ma_line is not None:
            val = data.ma_line.reindex(data.frame.index).iloc[i]
            if pd.notna(val):
                parts.append(f"{data.indicator_label}: {float(val):,.2f}")
        elif data.indicator == "supertrend" and data.supertrend is not None:
            st_row = data.supertrend.reindex(data.frame.index).iloc[i]
            if pd.notna(st_row["supertrend"]):
                side = "Bull" if int(st_row["direction"]) == 1 else "Bear"
                parts.append(f"{data.indicator_label}: {float(st_row['supertrend']):,.2f} ({side})")
        return "  |  ".join(parts)

    def _on_move(self, event) -> None:
        data = self._data
        if data is None or event.inaxes != data.ax:
            if self._last_i is not None:
                self._hide()
            return
        if event.xdata is None or len(data.xnums) == 0:
            if self._last_i is not None:
                self._hide()
            return

        half = data.bar_width * 0.55
        dists = np.abs(data.xnums - float(event.xdata))
        i = int(np.argmin(dists))
        if dists[i] > half:
            if self._last_i is not None:
                self._hide()
            return

        if i == self._last_i:
            return

        self._set_detail(self._format_bar(data, i))
        self._last_i = i


def _bar_width(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.6
    gaps = pd.Series(index[1:] - index[:-1]).dt.total_seconds() / 86400.0
    median_gap = float(gaps.median()) if not gaps.empty else 1.0
    return max(0.2, min(0.85, median_gap * 0.7))


def plot_candles(
    ax,
    ohlc: pd.DataFrame,
    *,
    mode: CandleMode = "candlestick",
    title: str = "",
    mark_signals: bool = False,
    min_bullish_bars: int = 1,
) -> pd.DataFrame:
    """Draw OHLC or Heikin Ashi candles on *ax*. Returns the frame that was plotted."""
    frame = candle_frame(ohlc, mode)
    if frame.empty:
        ax.set_title(title or "No OHLC data")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return frame

    x = mdates.date2num(frame.index.to_pydatetime())
    width = _bar_width(frame.index)
    opens = frame["Open"].astype(float).values
    highs = frame["High"].astype(float).values
    lows = frame["Low"].astype(float).values
    closes = frame["Close"].astype(float).values

    for i, (xo, o, h, l, c) in enumerate(zip(x, opens, highs, lows, closes)):
        bullish = c >= o
        color = _BULL_COLOR if bullish else _BEAR_COLOR
        ax.vlines(xo, l, h, color=_WICK_COLOR, linewidth=0.8, zorder=1)
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height == 0:
            body_height = max((h - l) * 0.08, 1e-6)
        ax.add_patch(
            Rectangle(
                (xo - width / 2.0, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                zorder=2,
            )
        )

    if mark_signals and len(frame) > 0:
        opens_s = frame["Open"]
        closes_s = frame["Close"]
        for i in range(len(frame)):
            end_idx = i
            count = 0
            for j in range(end_idx, -1, -1):
                if closes_s.iloc[j] > opens_s.iloc[j]:
                    count += 1
                else:
                    break
            if count >= min_bullish_bars and (i == 0 or count == min_bullish_bars):
                ax.scatter(
                    x[i],
                    lows[i] * 0.998,
                    marker="^",
                    s=28,
                    color=_BULL_COLOR,
                    zorder=3,
                )
            elif closes_s.iloc[i] <= opens_s.iloc[i] and (
                i == 0 or closes_s.iloc[i - 1] > opens_s.iloc[i - 1]
            ):
                ax.scatter(
                    x[i],
                    highs[i] * 1.002,
                    marker="v",
                    s=28,
                    color=_BEAR_COLOR,
                    zorder=3,
                )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y"))
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("Price")
    mode_label = "Heikin Ashi" if mode == "heikin_ashi" else "Candlestick"
    ax.set_title(title or mode_label)
    fig = ax.get_figure()
    if fig is not None:
        fig.autofmt_xdate()
    return frame


def plot_candle_comparison(
    fig,
    ohlc: pd.DataFrame,
    *,
    index_label: str,
    min_bullish_bars: int = 1,
) -> None:
    """Standard vs Heikin Ashi side-by-side on one figure."""
    fig.clear()
    ax_left, ax_right = fig.subplots(1, 2, sharey=True)
    plot_candles(
        ax_left,
        ohlc,
        mode="candlestick",
        title=f"{index_label} — Candlestick",
        mark_signals=True,
        min_bullish_bars=min_bullish_bars,
    )
    plot_candles(
        ax_right,
        ohlc,
        mode="heikin_ashi",
        title=f"{index_label} — Heikin Ashi",
        mark_signals=True,
        min_bullish_bars=min_bullish_bars,
    )
    fig.tight_layout()


def plot_close_overlay(
    ax,
    ohlc: pd.DataFrame,
    *,
    modes: tuple[CandleMode, ...] = ("candlestick", "heikin_ashi"),
) -> None:
    """Line overlay of close series for each candle mode (raw vs HA)."""
    base = normalize_ohlc(ohlc)
    if base.empty:
        return
    ax.plot(base.index, base["Close"], label="Close (raw)", color="#1565C0", linewidth=1.2)
    if "heikin_ashi" in modes:
        ha = candle_frame(ohlc, "heikin_ashi")
        ax.plot(ha.index, ha["Close"], label="HA Close", color="#2E7D32", linewidth=1.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)


def _signal_transitions(
    ohlc: pd.DataFrame,
    *,
    indicator: IndicatorKind,
    candle_mode: CandleMode,
    period: int,
    supertrend_multiplier: float,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Return (entry_dates, exit_dates) where indicator flips."""
    base = normalize_ohlc(ohlc)
    if base.empty:
        return [], []

    entries: list[pd.Timestamp] = []
    exits: list[pd.Timestamp] = []
    prev_bull: bool | None = None

    for ts in base.index:
        bull = bullish_signal(
            ohlc,
            indicator=indicator,
            candle_mode=candle_mode,
            as_of=pd.Timestamp(ts),
            period=period,
            supertrend_multiplier=supertrend_multiplier,
        )
        if prev_bull is not None:
            if bull and not prev_bull:
                entries.append(pd.Timestamp(ts))
            elif not bull and prev_bull:
                exits.append(pd.Timestamp(ts))
        prev_bull = bull

    return entries, exits


def _plot_supertrend_line(ax, st: pd.DataFrame, *, label: str, linewidth: float = 1.6) -> None:
    """TradingView-style Supertrend: one continuous line, green bull / red bear."""
    if st.empty or len(st) < 2:
        if not st.empty:
            x = mdates.date2num(st.index.to_pydatetime())
            color = _ST_BULL_COLOR if int(st["direction"].iloc[0]) == 1 else _ST_BEAR_COLOR
            ax.plot(x, st["supertrend"].astype(float), color=color, linewidth=linewidth, zorder=3)
        return

    xnums = mdates.date2num(st.index.to_pydatetime())
    y = st["supertrend"].astype(float).values
    dirs = st["direction"].astype(int).values

    segments = [
        [(xnums[i], y[i]), (xnums[i + 1], y[i + 1])]
        for i in range(len(xnums) - 1)
    ]
    colors = [_ST_BULL_COLOR if dirs[i] == 1 else _ST_BEAR_COLOR for i in range(len(xnums) - 1)]

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    ax.add_collection(lc)
    ax.autoscale_view()

    # Single legend entry (line is green/red on chart; legend shows one Supertrend label).
    ax.plot([], [], color=_ST_BULL_COLOR, linewidth=linewidth, label=label)


def _slice_display(
    frame: pd.DataFrame,
    *,
    display_start: pd.Timestamp | None,
    display_end: pd.Timestamp | None,
) -> pd.DataFrame:
    if display_start is None and display_end is None:
        return frame
    mask = pd.Series(True, index=frame.index)
    if display_start is not None:
        mask &= frame.index >= display_start
    if display_end is not None:
        mask &= frame.index < display_end
    return frame.loc[mask]


def plot_index_with_indicator(
    fig,
    ohlc: pd.DataFrame,
    *,
    index_label: str,
    indicator: IndicatorKind,
    candle_mode: CandleMode = "candlestick",
    chart_candle_mode: CandleMode | None = None,
    period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
    timeframe_label: str = "Daily",
    mark_signals: bool = True,
    display_start: pd.Timestamp | None = None,
    display_end: pd.Timestamp | None = None,
) -> CandleHoverData | None:
    """Price chart with selected indicator overlay. Returns hover metadata."""
    fig.clear()
    ax = fig.add_subplot(111)

    if ohlc is None or ohlc.empty:
        ax.set_title(f"{index_label} — no data")
        ax.text(0.5, 0.5, "Load data to plot", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return None

    base_full = normalize_ohlc(ohlc)
    display_ohlc = _slice_display(
        base_full, display_start=display_start, display_end=display_end
    )
    if display_ohlc.empty:
        ax.set_title(f"{index_label} — no data in range")
        fig.tight_layout()
        return None

    plot_mode = chart_candle_mode or candle_mode
    frame = plot_candles(
        ax,
        display_ohlc,
        mode=plot_mode,
        title="",
        mark_signals=False,
    )
    if frame.empty:
        fig.tight_layout()
        return None

    ind_label = indicator_display(indicator, period=period, multiplier=supertrend_multiplier)
    ma_line: pd.Series | None = None
    st_frame: pd.DataFrame | None = None

    if indicator in ("sma", "ema"):
        ma_full = compute_sma(base_full["Close"], period) if indicator == "sma" else compute_ema(base_full["Close"], period)
        ma_line = ma_full.reindex(frame.index)
        ax.plot(
            ma_line.index,
            ma_line.values,
            color=_INDICATOR_COLOR,
            linewidth=1.6,
            label=ind_label,
            zorder=3,
        )
    elif indicator == "supertrend":
        ind_full = indicator_ohlc(base_full, indicator=indicator, candle_mode=candle_mode)
        st_full = compute_supertrend(ind_full, atr_period=period, multiplier=supertrend_multiplier)
        st_frame = st_full.reindex(frame.index)
        _plot_supertrend_line(ax, st_frame, label=ind_label)

    if mark_signals:
        entries, exits = _signal_transitions(
            base_full,
            indicator=indicator,
            candle_mode=candle_mode,
            period=period,
            supertrend_multiplier=supertrend_multiplier,
        )
        if display_start is not None or display_end is not None:
            entries = [d for d in entries if (display_start is None or pd.Timestamp(d) >= display_start) and (display_end is None or pd.Timestamp(d) < display_end)]
            exits = [d for d in exits if (display_start is None or pd.Timestamp(d) >= display_start) and (display_end is None or pd.Timestamp(d) < display_end)]
        x = mdates.date2num(frame.index.to_pydatetime())
        lows = frame["Low"].astype(float).values
        highs = frame["High"].astype(float).values
        idx_map = {pd.Timestamp(t): i for i, t in enumerate(frame.index)}
        for dt in entries:
            i = idx_map.get(pd.Timestamp(dt))
            if i is not None:
                ax.scatter(x[i], lows[i] * 0.998, marker="^", s=36, color=_BULL_COLOR, zorder=4)
        for dt in exits:
            i = idx_map.get(pd.Timestamp(dt))
            if i is not None:
                ax.scatter(x[i], highs[i] * 1.002, marker="v", s=36, color=_BEAR_COLOR, zorder=4)

    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{index_label} — {ind_label} ({timeframe_label})")
    fig.tight_layout()

    chart_mode_label = "Heikin Ashi" if plot_mode == "heikin_ashi" else "Candlestick"
    return CandleHoverData(
        ax=ax,
        frame=frame,
        xnums=mdates.date2num(frame.index.to_pydatetime()),
        bar_width=_bar_width(frame.index),
        chart_mode_label=chart_mode_label,
        indicator=indicator,
        indicator_label=ind_label,
        indicator_period=period,
        supertrend_multiplier=supertrend_multiplier,
        ma_line=ma_line,
        supertrend=st_frame,
    )
