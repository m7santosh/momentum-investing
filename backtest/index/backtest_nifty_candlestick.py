"""
Signal-based backtest for Nifty indices using a technical indicator.

No rebalancing — enter when the indicator turns bullish, exit when bearish,
at the selected timeframe (daily / weekly / monthly).

Examples:
    python backtest/index/backtest_nifty_candlestick.py --mode candlestick
    python backtest/index/backtest_nifty_candlestick.py --timeframe weekly --indices "Nifty Bank"
    python momentum/nifty_candlestick_backtest_ui.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.backtest_cancel import check_cancelled  # noqa: E402
from momentum.index.candle_signals import (  # noqa: E402
    TIMEFRAME_LABELS,
    CandleMode,
    Timeframe,
    _PERIODS_PER_YEAR,
    ohlc_for_timeframe,
    resample_ohlc,
)
from momentum.index.index_indicators import (  # noqa: E402
    DEFAULT_INDICATOR,
    DEFAULT_INDICATOR_PERIOD,
    DEFAULT_SUPERTREND_ATR,
    DEFAULT_SUPERTREND_MULTIPLIER,
    INDICATOR_LABELS,
    IndicatorKind,
    bullish_signal,
    bearish_signal,
    indicator_display,
    indicator_warmup_start,
    resolve_indicator,
)
from momentum.index.nifty_indices import (  # noqa: E402
    DEFAULT_SELECTED_INDEX_IDS,
    NiftyIndex,
    resolve_selected_indices,
)
from utils.india_market_data import (  # noqa: E402
    format_range_label,
    get_india_market_data,
    get_india_market_data_run_stats,
    prepare_india_market_data_range,
)
from utils.nse_bhavcopy import today_ist  # noqa: E402

CANDLE_MODES: tuple[CandleMode, ...] = ("candlestick", "heikin_ashi")

CANDLE_MODE_LABELS: dict[CandleMode, str] = {
    "candlestick": "Candlestick",
    "heikin_ashi": "Heikin Ashi",
}

_TIMEFRAME_ALIASES: dict[str, Timeframe] = {
    "daily": "day",
    "day": "day",
    "weekly": "week",
    "week": "week",
    "monthly": "month",
    "month": "month",
}


@dataclass
class OpenPosition:
    index_id: str
    yahoo_ticker: str
    label: str
    entry_date: date
    entry_price: float
    shares: float


@dataclass
class NiftyCandleBacktestConfig:
    backtest_start: str
    backtest_end: str
    candle_mode: CandleMode = "candlestick"
    timeframe: Timeframe = "day"
    selected_index_ids: tuple[str, ...] = DEFAULT_SELECTED_INDEX_IDS
    indicator: IndicatorKind = DEFAULT_INDICATOR
    indicator_period: int = DEFAULT_INDICATOR_PERIOD
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER
    initial_capital: float = 500_000.0

    def __post_init__(self) -> None:
        tf = _TIMEFRAME_ALIASES.get(str(self.timeframe).lower(), self.timeframe)
        if tf not in TIMEFRAME_LABELS:
            tf = "day"
        object.__setattr__(self, "timeframe", tf)
        selected = resolve_selected_indices(self.selected_index_ids)
        object.__setattr__(
            self,
            "selected_index_ids",
            tuple(idx.index_id for idx in selected),
        )
        ind = resolve_indicator(str(self.indicator))
        object.__setattr__(self, "indicator", ind)
        period = max(1, int(self.indicator_period))
        object.__setattr__(self, "indicator_period", period)
        object.__setattr__(
            self,
            "supertrend_multiplier",
            max(0.1, float(self.supertrend_multiplier)),
        )

    def selected_indices(self) -> list[NiftyIndex]:
        return resolve_selected_indices(self.selected_index_ids)


_DISPLAY_DATE_FMT = "%d-%m-%Y"


def _format_display_date(value) -> str:
    if value is None or value == "":
        return ""
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return ""
    if pd.isna(ts):
        return ""
    return ts.strftime(_DISPLAY_DATE_FMT)


def _parse_user_date(text: str) -> pd.Timestamp:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Date is required (DD-MM-YYYY).")
    parts = raw.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Use DD-MM-YYYY only (e.g. 03-06-2026). Got: {raw!r}")
    day, month, year = (int(parts[0]), int(parts[1]), int(parts[2]))
    if len(parts[2]) != 4:
        raise ValueError(f"Use DD-MM-YYYY only (4-digit year). Got: {raw!r}")
    try:
        ts = pd.Timestamp(datetime(year, month, day))
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date: {raw!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {raw!r}")
    if ts.strftime(_DISPLAY_DATE_FMT) != raw:
        raise ValueError(f"Use DD-MM-YYYY only (e.g. 03-06-2026). Got: {raw!r}")
    return ts


def normalize_backtest_date(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Backtest date is required.")
    parts = raw.split("-")
    if len(parts) == 3 and len(parts[0]) == 4 and parts[0].isdigit():
        pd.Timestamp(raw)
        return raw
    return _parse_user_date(raw).strftime("%Y-%m-%d")


def _series_col(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()


@dataclass
class NiftyCandleBacktestEngine:
    config: NiftyCandleBacktestConfig
    progress_cb: Callable[[str], None] | None = None
    cancel_check: Callable[[], bool] | None = None
    _ohlc_daily: dict[str, pd.DataFrame] = field(default_factory=dict)
    _ohlc: dict[str, pd.DataFrame] = field(default_factory=dict)
    _ohlc_std: dict[str, pd.DataFrame] = field(default_factory=dict)
    _closes: dict[str, pd.Series] = field(default_factory=dict)
    _universe: list[NiftyIndex] = field(default_factory=list)
    _bar_dates: list[date] = field(default_factory=list)
    _positions: dict[str, OpenPosition] = field(default_factory=dict)
    _cash: float = 0.0
    _records: list[dict] = field(default_factory=list)
    _period_idx: int = 0
    _portfolio_value: float = 0.0
    _loaded: bool = False

    def __post_init__(self) -> None:
        self._cash = self.config.initial_capital
        self._portfolio_value = self.config.initial_capital

    def _log(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def total_periods(self) -> int:
        return len(self._bar_dates)

    @property
    def current_period(self) -> int:
        return self._period_idx

    @property
    def finished(self) -> bool:
        return self._period_idx >= len(self._bar_dates)

    def ohlc_for_chart(self, yahoo_ticker: str) -> pd.DataFrame:
        """OHLC at the configured timeframe (includes warmup bars for indicators)."""
        return self._ohlc.get(yahoo_ticker, pd.DataFrame())

    def ohlc_daily_for_chart(self, yahoo_ticker: str) -> pd.DataFrame:
        """Raw daily OHLC (includes warmup bars) for chart recompute by candle mode."""
        return self._ohlc_daily.get(yahoo_ticker, pd.DataFrame())

    def chart_display_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(normalize_backtest_date(self.config.backtest_start))
        end = pd.Timestamp(normalize_backtest_date(self.config.backtest_end))
        return start, end

    @property
    def first_bar_date(self) -> date | None:
        return self._bar_dates[0] if self._bar_dates else None

    @property
    def last_bar_date(self) -> date | None:
        return self._bar_dates[-1] if self._bar_dates else None

    def data_starts_after_backtest(self) -> bool:
        """True when the first bar in-range is later than the configured backtest start."""
        if not self._bar_dates:
            return False
        req = pd.Timestamp(normalize_backtest_date(self.config.backtest_start)).date()
        return self._bar_dates[0] > req

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        df.attrs["candle_mode"] = self.config.candle_mode
        df.attrs["timeframe"] = self.config.timeframe
        df.attrs["selected_indices"] = list(self.config.selected_index_ids)
        df.attrs["indicator"] = self.config.indicator
        df.attrs["indicator_period"] = self.config.indicator_period
        df.attrs["supertrend_multiplier"] = self.config.supertrend_multiplier
        df.attrs["indicator_label"] = indicator_display(
            self.config.indicator,
            period=self.config.indicator_period,
            multiplier=self.config.supertrend_multiplier,
        )
        return df

    def reset_run(self) -> None:
        self._period_idx = 0
        self._records = []
        self._positions = {}
        self._cash = self.config.initial_capital
        self._portfolio_value = self.config.initial_capital

    def _loaded_data_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for df in self._ohlc_daily.values():
            if df is None or df.empty:
                continue
            starts.append(pd.Timestamp(df.index.min()))
            ends.append(pd.Timestamp(df.index.max()))
        if not starts:
            return pd.Timestamp.min, pd.Timestamp.max
        return min(starts), max(ends)

    def apply_run_context(self, config: NiftyCandleBacktestConfig) -> None:
        """Apply UI/backtest parameters from *config* using cached daily OHLC."""
        if not self._loaded:
            return

        req_end = pd.Timestamp(normalize_backtest_date(config.backtest_end))
        _, loaded_end = self._loaded_data_bounds()
        if req_end > loaded_end + pd.Timedelta(days=7):
            raise ValueError(
                f"End date {_format_display_date(config.backtest_end)} is beyond loaded data "
                f"({_format_display_date(loaded_end.date())}). Click Load Data."
            )

        selected = config.selected_indices()
        available = [idx for idx in selected if idx.yahoo_ticker in self._ohlc_daily]
        missing = [idx.label for idx in selected if idx.yahoo_ticker not in self._ohlc_daily]
        if not available:
            raise ValueError("No selected indices have loaded data. Click Load Data.")

        self.config = config
        self._universe = available
        if missing:
            self._log(f"  Skipping indices without loaded data: {', '.join(missing)}")

        self._apply_timeframe()
        self.reset_run()

    def reapply_chart_context(
        self,
        *,
        timeframe: Timeframe,
        candle_mode: CandleMode,
        indicator_period: int | None = None,
        supertrend_multiplier: float | None = None,
    ) -> None:
        """Rebuild bars from cached daily OHLC — same path as the chart (TradingView rules)."""
        if not self._loaded:
            return
        updated = NiftyCandleBacktestConfig(
            backtest_start=self.config.backtest_start,
            backtest_end=self.config.backtest_end,
            candle_mode=candle_mode,
            timeframe=timeframe,
            selected_index_ids=self.config.selected_index_ids,
            indicator=self.config.indicator,
            indicator_period=(
                indicator_period
                if indicator_period is not None
                else self.config.indicator_period
            ),
            supertrend_multiplier=(
                supertrend_multiplier
                if supertrend_multiplier is not None
                else self.config.supertrend_multiplier
            ),
            initial_capital=self.config.initial_capital,
        )
        self.apply_run_context(updated)

    def _close_on(self, yahoo_ticker: str, on_date: date) -> float | None:
        series = self._closes.get(yahoo_ticker)
        if series is None or series.empty:
            return None
        sliced = series[series.index <= pd.Timestamp(on_date)]
        if sliced.empty:
            return None
        return float(sliced.iloc[-1])

    def _bar_before(self, on_date: date) -> date | None:
        prior = [d for d in self._bar_dates if d < on_date]
        return prior[-1] if prior else None

    def _portfolio_value_on(self, on_date: date) -> float:
        total = self._cash
        for pos in self._positions.values():
            px = self._close_on(pos.yahoo_ticker, on_date)
            if px is not None:
                total += pos.shares * px
        return total

    def _exit_position(self, index_id: str, exit_date: date, exit_price: float) -> dict:
        pos = self._positions.pop(index_id)
        proceeds = pos.shares * exit_price
        self._cash += proceeds
        pl_pct = (exit_price / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0
        return {
            "ticker": pos.label,
            "index_id": index_id,
            "status": "Exit",
            "entry": pos.entry_price,
            "exit": exit_price,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "pl_pct": pl_pct,
            "exit_reason": "Exit signal",
        }

    def _enter_position(self, idx: NiftyIndex, entry_date: date) -> dict | None:
        px = self._close_on(idx.yahoo_ticker, entry_date)
        if px is None or px <= 0:
            return None
        slots_left = len(self._universe) - len(self._positions)
        if slots_left <= 0:
            return None
        deploy = self._cash / slots_left if slots_left > 0 else 0.0
        if deploy <= 0:
            return None
        shares = deploy / px
        self._cash -= deploy
        self._positions[idx.index_id] = OpenPosition(
            index_id=idx.index_id,
            yahoo_ticker=idx.yahoo_ticker,
            label=idx.label,
            entry_date=entry_date,
            entry_price=px,
            shares=shares,
        )
        return {
            "ticker": idx.label,
            "index_id": idx.index_id,
            "status": "Entry",
            "entry": px,
            "exit": px,
            "entry_date": entry_date,
            "exit_date": None,
            "pl_pct": 0.0,
            "exit_reason": "Open",
        }

    def _open_position_row(self, pos: OpenPosition, mark_date: date) -> dict:
        mark = self._close_on(pos.yahoo_ticker, mark_date) or pos.entry_price
        pl_pct = (mark / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0
        return {
            "ticker": pos.label,
            "index_id": pos.index_id,
            "status": "Open",
            "entry": pos.entry_price,
            "exit": mark,
            "entry_date": pos.entry_date,
            "exit_date": None,
            "pl_pct": pl_pct,
            "exit_reason": "Open",
        }

    def _apply_timeframe(self) -> None:
        tf = self.config.timeframe
        mode = self.config.candle_mode
        self._ohlc.clear()
        self._ohlc_std.clear()
        self._closes.clear()
        for ticker, daily in self._ohlc_daily.items():
            resampled = ohlc_for_timeframe(daily, tf, mode)
            if resampled.empty:
                continue
            self._ohlc[ticker] = resampled
            if tf == "week" and mode == "heikin_ashi":
                std_frame = ohlc_for_timeframe(daily, tf, "candlestick")
                if not std_frame.empty:
                    self._ohlc_std[ticker] = std_frame
            self._closes[ticker] = resampled["Close"].astype(float)

        start = pd.Timestamp(normalize_backtest_date(self.config.backtest_start))
        end = pd.Timestamp(normalize_backtest_date(self.config.backtest_end))
        bar_dates: set[date] = set()
        for series in self._closes.values():
            for ts in series.index:
                ts_norm = pd.Timestamp(ts)
                if start <= ts_norm < end:
                    bar_dates.add(ts_norm.date())
        self._bar_dates = sorted(bar_dates)

    def load_data(self) -> None:
        start = pd.Timestamp(normalize_backtest_date(self.config.backtest_start))
        end = pd.Timestamp(normalize_backtest_date(self.config.backtest_end))
        load_start = indicator_warmup_start(
            start,
            period=self.config.indicator_period,
            timeframe=self.config.timeframe,
        )

        self._universe = self.config.selected_indices()
        tf_label = TIMEFRAME_LABELS[self.config.timeframe]
        if load_start < start:
            self._log(
                f"Loading {len(self._universe)} index(es) — "
                f"{format_range_label(start, end)} ({tf_label} bars) "
                f"[warmup from {load_start.date()}]"
            )
        else:
            self._log(
                f"Loading {len(self._universe)} index(es) — "
                f"{format_range_label(start, end)} ({tf_label} bars)"
            )
        prepare_india_market_data_range(
            load_start,
            end,
            reset_stats=True,
            include_cm_bhavcopy=False,
            include_index_archive=True,
            cancel_check=self.cancel_check,
        )

        self._ohlc_daily.clear()

        for idx in self._universe:
            check_cancelled(self.cancel_check)
            df = get_india_market_data(idx.yahoo_ticker, load_start, end)
            if df is None or df.empty or "Close" not in df.columns:
                self._log(f"  Skip {idx.label}: no OHLC")
                continue
            self._ohlc_daily[idx.yahoo_ticker] = df

        self._apply_timeframe()
        self._loaded = True
        stats = get_india_market_data_run_stats()
        self._log(
            f"Loaded {len(self._ohlc)}/{len(self._universe)} indices, "
            f"{len(self._bar_dates)} {tf_label.lower()} bars ({stats.summary()})"
        )
        if self._bar_dates:
            req_start = pd.Timestamp(start).date()
            first = self._bar_dates[0]
            last = self._bar_dates[-1]
            if first > req_start:
                self._log(
                    f"  Note: first {tf_label.lower()} bar is {_format_display_date(first)} "
                    f"(no NSE history before your start {_format_display_date(req_start)})"
                )
            else:
                self._log(
                    f"  Bars in backtest window: {_format_display_date(first)} .. "
                    f"{_format_display_date(last)}"
                )

    def _signal_kwargs(self) -> dict:
        return {
            "indicator": self.config.indicator,
            "candle_mode": self.config.candle_mode,
            "period": self.config.indicator_period,
            "supertrend_multiplier": self.config.supertrend_multiplier,
            "timeframe": self.config.timeframe,
        }

    def _standard_ohlc_for(self, yahoo_ticker: str) -> pd.DataFrame | None:
        return self._ohlc_std.get(yahoo_ticker)

    def step_period(self) -> dict | None:
        """Advance one bar — enter when bullish, exit when bearish (close vs indicator)."""
        if not self._loaded or self.finished:
            return None

        check_cancelled(self.cancel_check)
        idx = self._period_idx
        bar_date = self._bar_dates[idx]
        bar_ts = pd.Timestamp(bar_date)
        mode = self.config.candle_mode
        ind = self.config.indicator
        period = self.config.indicator_period
        st_mult = self.config.supertrend_multiplier
        sig_kw = self._signal_kwargs()

        before = self._bar_before(bar_date)
        if before is None or (idx == 0 and not self._positions):
            val_start = self.config.initial_capital if idx == 0 else self._portfolio_value
        else:
            val_start = self._portfolio_value_on(before)

        period_exits: list[dict] = []
        period_entries: list[dict] = []

        for index_id in list(self._positions.keys()):
            pos = self._positions[index_id]
            if bar_date <= pos.entry_date:
                continue
            ohlc = self._ohlc.get(pos.yahoo_ticker)
            if ohlc is None:
                continue
            if bearish_signal(
                ohlc,
                as_of=bar_ts,
                standard_ohlc=self._standard_ohlc_for(pos.yahoo_ticker),
                **sig_kw,
            ):
                px = self._close_on(pos.yahoo_ticker, bar_date)
                if px is not None:
                    period_exits.append(self._exit_position(index_id, bar_date, px))

        for cand in self._universe:
            if cand.index_id in self._positions:
                continue
            ohlc = self._ohlc.get(cand.yahoo_ticker)
            if ohlc is None:
                continue
            if bullish_signal(
                ohlc,
                as_of=bar_ts,
                standard_ohlc=self._standard_ohlc_for(cand.yahoo_ticker),
                **sig_kw,
            ):
                entered = self._enter_position(cand, bar_date)
                if entered is not None:
                    period_entries.append(entered)

        val_end = self._portfolio_value_on(bar_date)
        port_ret = (val_end / val_start - 1.0) if val_start > 0 else 0.0
        self._portfolio_value = val_end

        open_rows = [self._open_position_row(pos, bar_date) for pos in self._positions.values()]
        position_rows = period_exits + open_rows

        prev_open = set(self._records[-1].get("Open_Tickers") or []) if self._records else set()
        open_tickers = [self._positions[k].label for k in self._positions]
        new_entries = [r["ticker"] for r in period_entries]

        record = {
            "Period": idx + 1,
            "Bar_Date": pd.Timestamp(bar_date),
            "End_Date": pd.Timestamp(bar_date),
            "Timeframe": TIMEFRAME_LABELS[self.config.timeframe],
            "Candle_Mode": CANDLE_MODE_LABELS[mode],
            "Indicator": indicator_display(ind, period=period, multiplier=st_mult),
            "Selected_Count": len(self._universe),
            "Held": len(self._positions),
            "New_Entries": len(new_entries),
            "Signal_Exits": len(period_exits),
            "Port_Return": port_ret,
            "Portfolio_Value": val_end,
            "Open_Tickers": open_tickers,
            "New_Entry_Tickers": new_entries,
            "Position_Rows": position_rows,
        }
        self._records.append(record)
        self._period_idx += 1
        return record

    def run_all(self) -> pd.DataFrame:
        self.reset_run()
        while not self.finished:
            check_cancelled(self.cancel_check)
            self.step_period()
        return self.trades_df


def compute_metrics(df: pd.DataFrame, capital: float) -> dict:
    if df.empty:
        return {}

    port_rets = df["Port_Return"].values
    n_periods = len(port_rets)
    tf_key = str(df.attrs.get("timeframe", "day"))
    periods_per_year = _PERIODS_PER_YEAR.get(tf_key, 252)

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    years = n_periods / periods_per_year if periods_per_year > 0 else 0.0

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    ann_vol = (
        float(np.std(port_rets, ddof=1) * np.sqrt(periods_per_year))
        if n_periods > 1
        else 0.0
    )
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    downside = port_rets[port_rets < 0]
    downside_vol = (
        float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))
        if len(downside) > 1
        else 0.0
    )
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    mode = CANDLE_MODE_LABELS.get(str(df.attrs.get("candle_mode", "candlestick")), "Candlestick")
    tf = TIMEFRAME_LABELS.get(tf_key, "Daily")
    indicator_label = str(df.attrs.get("indicator_label", INDICATOR_LABELS[DEFAULT_INDICATOR]))
    selected = df.attrs.get("selected_indices") or []
    selected_s = ", ".join(selected) if selected else "—"

    return {
        "Candle_Mode": mode if str(df.attrs.get("indicator", DEFAULT_INDICATOR)) == "candle" else "—",
        "Indicator": indicator_label,
        "Timeframe": tf,
        "Indices": len(selected),
        "Index_List": selected_s,
        "Period": (
            f"{_format_display_date(df['Bar_Date'].iloc[0])} to "
            f"{_format_display_date(df['Bar_Date'].iloc[-1])}"
        ),
        "Bars": n_periods,
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(float(np.mean(port_rets > 0) * 100), 1),
        "Avg_Bar_Return_%": round(float(np.mean(port_rets) * 100), 2),
        "Entries": int(df["New_Entries"].sum()),
        "Exits": int(df["Signal_Exits"].sum()),
        "Final_Value": round(float(df["Portfolio_Value"].iloc[-1]), 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Ann_Volatility_%": round(ann_vol * 100, 2),
    }


def run_backtest(
    backtest_start: str,
    backtest_end: str,
    *,
    candle_mode: CandleMode = "candlestick",
    timeframe: Timeframe = "day",
    selected_index_ids: tuple[str, ...] | list[str] = DEFAULT_SELECTED_INDEX_IDS,
    indicator: IndicatorKind = DEFAULT_INDICATOR,
    indicator_period: int = DEFAULT_INDICATOR_PERIOD,
    supertrend_multiplier: float = DEFAULT_SUPERTREND_MULTIPLIER,
    initial_capital: float = 500_000.0,
    progress_cb: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    engine = NiftyCandleBacktestEngine(
        NiftyCandleBacktestConfig(
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            candle_mode=candle_mode,
            timeframe=timeframe,
            selected_index_ids=tuple(selected_index_ids),
            indicator=indicator,
            indicator_period=indicator_period,
            supertrend_multiplier=supertrend_multiplier,
            initial_capital=initial_capital,
        ),
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )
    engine.load_data()
    return engine.run_all()


def main() -> int:
    parser = argparse.ArgumentParser(description="Nifty index candlestick / Heikin Ashi backtest")
    parser.add_argument("--start", default=f"{today_ist().year}-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--mode",
        choices=list(CANDLE_MODES),
        default="candlestick",
        help="Signal candle type (default: candlestick)",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help='Comma-separated index names (e.g. "Nifty Bank,Nifty IT"). Default: sector preset.',
    )
    parser.add_argument(
        "--indicator",
        default=DEFAULT_INDICATOR,
        choices=list(INDICATOR_LABELS.keys()),
    )
    parser.add_argument("--period", type=int, default=None, help="SMA/EMA period or Supertrend ATR length")
    parser.add_argument(
        "--supertrend-multiplier",
        type=float,
        default=DEFAULT_SUPERTREND_MULTIPLIER,
    )
    parser.add_argument("--capital", type=float, default=500_000.0)
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=["daily", "weekly", "monthly", "day", "week", "month"],
    )
    args = parser.parse_args()

    end = args.end or pd.Timestamp(today_ist()).strftime("%Y-%m-%d")
    tf = _TIMEFRAME_ALIASES.get(args.timeframe.lower(), "day")

    if args.indices:
        selected = tuple(part.strip() for part in args.indices.split(",") if part.strip())
    else:
        selected = DEFAULT_SELECTED_INDEX_IDS

    indicator = resolve_indicator(args.indicator)
    if args.period is not None:
        indicator_period = args.period
    elif indicator == "supertrend":
        indicator_period = DEFAULT_SUPERTREND_ATR
    else:
        indicator_period = DEFAULT_INDICATOR_PERIOD

    df = run_backtest(
        args.start,
        end,
        candle_mode=args.mode,
        timeframe=tf,
        selected_index_ids=selected,
        indicator=indicator,
        indicator_period=indicator_period,
        supertrend_multiplier=args.supertrend_multiplier,
        initial_capital=args.capital,
        progress_cb=print,
    )
    if df.empty:
        print("No results.")
        return 1

    metrics = compute_metrics(df, args.capital)
    ind_label = metrics.get("Indicator", INDICATOR_LABELS[resolve_indicator(args.indicator)])
    print(f"\n{ind_label} — {metrics.get('Period', '')}")
    for key, value in metrics.items():
        if key != "Period":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
