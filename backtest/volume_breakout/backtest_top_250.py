"""
Walk-forward backtest for Volume Breakout final-list strategy.

Entries on rebalance dates (daily / weekly / fortnight / monthly). Positions stay open
until EOD close falls below 50 DMA — no forced exit on rebalance.

Examples:
    python backtest/volume_breakout/backtest_top_250.py --start 2024-01-01 --end 2025-03-31
    python backtest/volume_breakout/backtest_top_250.py --universe turnover --rebalance month
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VB_DIR = _PROJECT_ROOT / "volume-breakout"
for _path in (_PROJECT_ROOT, _VB_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from momentum.rrg_core import rrg_format_date  # noqa: E402
from top_250_sheet_logic import (  # noqa: E402
    enrich_symbol_eod,
    load_bhavcopy_day,
    pick_final_list,
    sheet_row,
)
from utils.nse_bhavcopy import (  # noqa: E402
    load_nse_index_weekly_histories_range,
    today_ist,
)

BENCHMARK_NSE = "Nifty 500"
_WARMUP_CALENDAR_DAYS = 370
UniverseMode = Literal["volume", "turnover"]
RebalanceFreq = Literal["day", "week", "fortnight", "month"]

_REBALANCE_LABELS: dict[RebalanceFreq, str] = {
    "day": "Daily",
    "week": "Weekly",
    "fortnight": "Fortnight",
    "month": "Monthly",
}

_PERIODS_PER_YEAR: dict[RebalanceFreq, int] = {
    "day": 252,
    "week": 52,
    "fortnight": 26,
    "month": 12,
}


@dataclass
class OpenPosition:
    symbol: str
    entry_date: date
    entry_price: float
    shares: float


@dataclass
class Top250BacktestConfig:
    backtest_start: str
    backtest_end: str
    universe: UniverseMode = "volume"
    top_n: int = 20
    initial_capital: float = 500_000.0
    rebalance_freq: RebalanceFreq = "week"


def build_rebalance_dates(
    start: date,
    end: date,
    available: set[date],
    freq: RebalanceFreq,
) -> list[date]:
    if freq == "day":
        return sorted(d for d in available if start <= d <= end)

    fridays: list[date] = []
    d = start
    while d <= end:
        if d.weekday() == 4 and d in available:
            fridays.append(d)
        d += timedelta(days=1)

    if not fridays:
        return []

    if freq == "week":
        return fridays
    if freq == "fortnight":
        return fridays[::2]

    monthly: list[date] = []
    seen: set[tuple[int, int]] = set()
    for friday in fridays:
        key = (friday.year, friday.month)
        if key not in seen:
            seen.add(key)
            monthly.append(friday)
    return monthly


@dataclass
class Top250BacktestEngine:
    config: Top250BacktestConfig
    progress_cb: Callable[[str], None] | None = None
    _closes: dict[str, pd.Series] = field(default_factory=dict)
    _highs: dict[str, pd.Series] = field(default_factory=dict)
    _bench_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    _rebal_dates: list[date] = field(default_factory=list)
    _available_dates: list[date] = field(default_factory=list)
    _backtest_end_date: date | None = None
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
    def total_weeks(self) -> int:
        return len(self._rebal_dates)

    @property
    def total_periods(self) -> int:
        return len(self._rebal_dates)

    @property
    def current_week(self) -> int:
        return self._period_idx

    @property
    def current_period(self) -> int:
        return self._period_idx

    @property
    def finished(self) -> bool:
        return self._period_idx >= len(self._rebal_dates)

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        if "Port_Return" in df.columns and len(df):
            df["Bench_Value"] = self.config.initial_capital * (
                (1 + df["Bench_Return"]).cumprod()
            )
        df.attrs["universe"] = self.config.universe
        df.attrs["top_n"] = self.config.top_n
        df.attrs["rebalance_freq"] = self.config.rebalance_freq
        return df

    def reset_run(self) -> None:
        self._period_idx = 0
        self._records = []
        self._positions = {}
        self._cash = self.config.initial_capital
        self._portfolio_value = self.config.initial_capital

    def _close_on(self, symbol: str, on_date: date) -> float | None:
        series = self._closes.get(symbol)
        if series is None or series.empty:
            return None
        sliced = series[series.index <= pd.Timestamp(on_date)]
        if sliced.empty:
            return None
        return float(sliced.iloc[-1])

    def _day_before(self, on_date: date) -> date | None:
        prior = [d for d in self._available_dates if d < on_date]
        return prior[-1] if prior else None

    def _portfolio_value_on(self, on_date: date) -> float:
        total = self._cash
        for sym, pos in self._positions.items():
            px = self._close_on(sym, on_date)
            if px is not None:
                total += pos.shares * px
        return total

    def _bench_return(self, start: date, end: date) -> float:
        if self._bench_daily.empty:
            return 0.0
        s = self._bench_daily[self._bench_daily.index <= pd.Timestamp(start)]
        e = self._bench_daily[self._bench_daily.index <= pd.Timestamp(end)]
        if s.empty or e.empty:
            return 0.0
        p0 = float(s.iloc[-1])
        p1 = float(e.iloc[-1])
        if p0 <= 0:
            return 0.0
        return p1 / p0 - 1

    def _below_dma50(self, sym: str, check_date: date) -> bool:
        close = self._close_on(sym, check_date)
        if close is None:
            return False
        closes = self._closes.get(sym, pd.Series(dtype=float))
        highs = self._highs.get(sym, pd.Series(dtype=float))
        enr = enrich_symbol_eod(
            sym,
            close,
            data_date=check_date,
            closes=closes,
            highs=highs,
        )
        return enr.dma50 is not None and close < enr.dma50

    def _final_list_tickers(self, rebal_date: date) -> tuple[list[str], int]:
        day = load_bhavcopy_day(rebal_date)
        if day is None:
            return [], 0

        base = (
            day.volume_top250
            if self.config.universe == "volume"
            else day.turnover_top250
        )
        rows = []
        for sym, metric, close in base:
            closes = self._closes.get(sym, pd.Series(dtype=float))
            highs = self._highs.get(sym, pd.Series(dtype=float))
            enr = enrich_symbol_eod(
                sym,
                close,
                data_date=rebal_date,
                closes=closes,
                highs=highs,
            )
            rows.append(sheet_row(enr, metric))

        final = pick_final_list(rows)
        return [r.nse_code for r in final], len(final)

    def _enter_position(self, sym: str, entry_date: date) -> dict | None:
        if sym in self._positions or len(self._positions) >= self.config.top_n:
            return None
        price = self._close_on(sym, entry_date)
        if price is None or price <= 0:
            return None

        port = self._portfolio_value_on(entry_date)
        # Equal-weight: each slot targets 1/top_n of total portfolio at entry.
        slot_size = port / self.config.top_n
        invest = min(self._cash, slot_size)
        if invest <= 0:
            return None

        shares = invest / price
        self._cash -= invest
        self._positions[sym] = OpenPosition(
            symbol=sym,
            entry_date=entry_date,
            entry_price=price,
            shares=shares,
        )
        return {
            "ticker": sym,
            "entry": round(price, 2),
            "exit": None,
            "exit_date": None,
            "exit_reason": "Open",
            "pl_pct": None,
        }

    def _exit_position(self, sym: str, exit_date: date, exit_price: float) -> dict:
        pos = self._positions.pop(sym)
        self._cash += pos.shares * exit_price
        pl_pct = round((exit_price / pos.entry_price - 1.0) * 100.0, 2)
        return {
            "ticker": sym,
            "entry": round(pos.entry_price, 2),
            "exit": round(exit_price, 2),
            "exit_date": pd.Timestamp(exit_date),
            "exit_reason": "50 DMA",
            "pl_pct": pl_pct,
        }

    def _open_position_row(self, pos: OpenPosition, as_of: date) -> dict:
        mark = self._close_on(pos.symbol, as_of)
        pl_pct: float | None = None
        if mark is not None and pos.entry_price > 0:
            pl_pct = round((mark / pos.entry_price - 1.0) * 100.0, 2)
        return {
            "ticker": pos.symbol,
            "entry": round(pos.entry_price, 2),
            "exit": round(mark, 2) if mark is not None else None,
            "exit_date": None,
            "exit_reason": "Open",
            "pl_pct": pl_pct,
        }

    def load_data(self) -> None:
        cfg = self.config
        start_ts = pd.Timestamp(cfg.backtest_start)
        end_ts = pd.Timestamp(cfg.backtest_end)
        if start_ts >= end_ts:
            raise ValueError("Backtest start must be before end")

        dl_start = (start_ts - pd.Timedelta(days=_WARMUP_CALENDAR_DAYS)).date()
        dl_end = max(end_ts.date(), today_ist())

        self._log(
            f"Loading NSE bhavcopy {rrg_format_date(dl_start)} .. "
            f"{rrg_format_date(dl_end)} (warmup + backtest)..."
        )

        close_buckets: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        high_buckets: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        available: list[date] = []
        sessions = 0
        d = dl_start
        while d <= dl_end:
            if d.weekday() < 5:
                day = load_bhavcopy_day(d)
                if day is not None:
                    sessions += 1
                    available.append(d)
                    ts = pd.Timestamp(d)
                    for sym, ohlc in day.ohlc.items():
                        close_buckets.setdefault(sym, []).append((ts, ohlc["close"]))
                        high_buckets.setdefault(sym, []).append((ts, ohlc["high"]))
            d += timedelta(days=1)

        if not available:
            raise RuntimeError("No NSE bhavcopy sessions found in the requested range.")

        self._closes = {
            sym: pd.Series({t: v for t, v in rows}).sort_index()
            for sym, rows in close_buckets.items()
            if rows
        }
        self._highs = {
            sym: pd.Series({t: v for t, v in rows}).sort_index()
            for sym, rows in high_buckets.items()
            if rows
        }

        self._log(f"Bhavcopy: {sessions} sessions, {len(self._closes)} symbols.")

        bench_batch = load_nse_index_weekly_histories_range(
            [BENCHMARK_NSE],
            dl_start,
            dl_end,
            min_points=1,
            quiet=True,
            freq="day",
        )
        self._bench_daily = bench_batch.get(BENCHMARK_NSE, pd.Series(dtype=float))
        if self._bench_daily.empty:
            raise RuntimeError(f"Could not load benchmark {BENCHMARK_NSE}")

        avail_set = set(available)
        rebal_dates = build_rebalance_dates(
            start_ts.date(),
            end_ts.date(),
            avail_set,
            cfg.rebalance_freq,
        )
        if not rebal_dates:
            raise RuntimeError(
                "No rebalance dates in range for the selected frequency."
            )

        self._rebal_dates = rebal_dates
        self._available_dates = available
        self._backtest_end_date = end_ts.date()
        self._loaded = True
        self.reset_run()
        freq_label = _REBALANCE_LABELS[cfg.rebalance_freq]
        self._log(
            f"Ready: {len(rebal_dates)} {freq_label.lower()} periods "
            f"({rrg_format_date(rebal_dates[0])} .. {rrg_format_date(rebal_dates[-1])})."
        )

    def _period_end_date(self, idx: int) -> date:
        rebal = self._rebal_dates[idx]
        if idx + 1 < len(self._rebal_dates):
            next_rebal = self._rebal_dates[idx + 1]
            prior = [d for d in self._available_dates if rebal <= d < next_rebal]
            return prior[-1] if prior else rebal

        end_cap = self._backtest_end_date or rebal
        tail = [d for d in self._available_dates if rebal <= d <= end_cap]
        return tail[-1] if tail else rebal

    def step_week(self) -> dict | None:
        """Advance one rebalance period (alias: step_period)."""
        return self.step_period()

    def step_period(self) -> dict | None:
        if not self._loaded or self.finished:
            return None

        idx = self._period_idx
        rebal_date = self._rebal_dates[idx]
        end_date = self._period_end_date(idx)

        before = self._day_before(rebal_date)
        if before is None or (idx == 0 and not self._positions):
            val_start = self.config.initial_capital if idx == 0 else self._portfolio_value
        else:
            val_start = self._portfolio_value_on(before)

        period_days = [d for d in self._available_dates if rebal_date <= d <= end_date]
        period_exits: list[dict] = []
        period_entries: list[dict] = []
        final_count = 0

        for d in period_days:
            for sym in list(self._positions.keys()):
                pos = self._positions[sym]
                if d <= pos.entry_date:
                    continue
                if self._below_dma50(sym, d):
                    px = self._close_on(sym, d)
                    if px is not None:
                        period_exits.append(self._exit_position(sym, d, px))

            if d == rebal_date:
                candidates, final_count = self._final_list_tickers(rebal_date)
                for sym in candidates:
                    if len(self._positions) >= self.config.top_n:
                        break
                    if sym in self._positions:
                        continue
                    entered = self._enter_position(sym, d)
                    if entered is not None:
                        period_entries.append(entered)

        val_end = self._portfolio_value_on(end_date)
        port_ret = (val_end / val_start - 1.0) if val_start > 0 else 0.0
        self._portfolio_value = val_end

        bench_start = before if before is not None else rebal_date
        bench_ret = self._bench_return(bench_start, end_date)

        open_rows = [
            self._open_position_row(pos, end_date)
            for pos in self._positions.values()
        ]
        position_rows = period_exits + open_rows

        prev_open = set(self._records[-1].get("Open_Tickers") or []) if self._records else set()
        open_tickers = list(self._positions.keys())
        new_entries = [r["ticker"] for r in period_entries]

        record = {
            "Week": idx + 1,
            "Period": idx + 1,
            "Rebal_Date": pd.Timestamp(rebal_date),
            "End_Date": pd.Timestamp(end_date),
            "Rebalance": _REBALANCE_LABELS[self.config.rebalance_freq],
            "Universe": self.config.universe,
            "Final_List_Count": final_count,
            "New_Entries": len(new_entries),
            "New_Entry_Tickers": new_entries,
            "DMA_Exits": len(period_exits),
            "Exit_Tickers": [r["ticker"] for r in period_exits],
            "Open_Tickers": open_tickers,
            "Open_Count": len(open_tickers),
            "Position_Rows": position_rows,
            "Closed_Rows": period_exits,
            "Entry_Rows": period_entries,
            "Holdings": ", ".join(open_tickers) if open_tickers else "CASH",
            "Port_Return": port_ret,
            "Bench_Return": bench_ret,
            "Portfolio_Value": val_end,
            "Turnover": len(set(open_tickers) ^ prev_open) / max(self.config.top_n, 1),
        }

        self._records.append(record)
        self._period_idx += 1
        return record

    def run_all(self) -> pd.DataFrame:
        while not self.finished:
            self.step_period()
        return self.trades_df


def compute_metrics(df: pd.DataFrame, capital: float) -> dict:
    if df.empty:
        return {}

    port_rets = df["Port_Return"].values
    bench_rets = df["Bench_Return"].values
    n_periods = len(port_rets)
    freq_key = str(df.attrs.get("rebalance_freq", "week"))
    periods_per_year = _PERIODS_PER_YEAR.get(freq_key, 52)  # type: ignore[arg-type]

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    bench_total = df["Bench_Value"].iloc[-1] / capital - 1
    years = n_periods / periods_per_year

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    bench_cum = (1 + pd.Series(bench_rets)).cumprod()
    bench_max_dd = float((bench_cum / bench_cum.cummax() - 1).min())

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
    alpha = cagr - bench_cagr
    total_trades = int(df["New_Entries"].sum()) if "New_Entries" in df.columns else 0
    dma_exits = int(df["DMA_Exits"].sum()) if "DMA_Exits" in df.columns else 0
    universe = str(df.attrs.get("universe", "volume"))
    top_n = int(df.attrs.get("top_n", 20))
    rebal = _REBALANCE_LABELS.get(
        str(df.attrs.get("rebalance_freq", "week")), "Weekly"
    )

    return {
        "Universe": universe.title(),
        "Rebalance": rebal,
        "Top_N": top_n,
        "Period": (
            f"{rrg_format_date(df['Rebal_Date'].iloc[0])} to "
            f"{rrg_format_date(df['End_Date'].iloc[-1])}"
        ),
        "Periods": n_periods,
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(float(np.mean(port_rets > 0) * 100), 1),
        "Avg_Period_Return_%": round(float(np.mean(port_rets) * 100), 2),
        "New_Entries": total_trades,
        "DMA_Exits": dma_exits,
        "Bench_Total_Return_%": round(bench_total * 100, 2),
        "Bench_CAGR_%": round(bench_cagr * 100, 2),
        "Bench_Max_Drawdown_%": round(bench_max_dd * 100, 2),
        "Final_Value": round(float(df["Portfolio_Value"].iloc[-1]), 2),
        "Bench_Final_Value": round(float(df["Bench_Value"].iloc[-1]), 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Ann_Volatility_%": round(ann_vol * 100, 2),
        "Avg_Turnover_%": round(float(df["Turnover"].mean() * 100), 1),
        "Alpha_%": round(alpha * 100, 2),
    }


def run_backtest(
    backtest_start: str,
    backtest_end: str,
    *,
    universe: UniverseMode = "volume",
    top_n: int = 20,
    initial_capital: float = 500_000.0,
    rebalance_freq: RebalanceFreq = "week",
    progress_cb: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    engine = Top250BacktestEngine(
        Top250BacktestConfig(
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            universe=universe,
            top_n=top_n,
            initial_capital=initial_capital,
            rebalance_freq=rebalance_freq,
        ),
        progress_cb=progress_cb,
    )
    engine.load_data()
    return engine.run_all()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volume Breakout top-250 backtest")
    p.add_argument("--start", required=True, help="Backtest start YYYY-MM-DD")
    p.add_argument("--end", required=True, help="Backtest end YYYY-MM-DD")
    p.add_argument("--universe", choices=("volume", "turnover"), default="volume")
    p.add_argument("--rebalance", choices=("day", "week", "fortnight", "month"), default="week")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--capital", type=float, default=500_000.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    def _progress(msg: str) -> None:
        print(msg)

    df = run_backtest(
        args.start,
        args.end,
        universe=args.universe,
        top_n=args.top_n,
        initial_capital=args.capital,
        rebalance_freq=args.rebalance,
        progress_cb=_progress,
    )
    metrics = compute_metrics(df, args.capital)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
