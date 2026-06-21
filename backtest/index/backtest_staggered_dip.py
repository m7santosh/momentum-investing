"""
Staggered dip-buying backtest on a Nifty index or Gold/Silver with a matching ETF.

Pick an index or commodity (dip buys: −Y%% from last lot; profit: +X%% on each lot's ETF).
At most N open lots at once; when full, wait for a profit exit. Freed slots redeploy next day only.
Dip buys extend the ladder (1→2→…→N); they never back-fill a slot left by profit.
All N lots use the same ETF. When every lot is closed, capital redeploys lot 1 next day.

Examples:
    python backtest/index/backtest_staggered_dip.py --index "Nifty Bank" --etf BANKBEES
    python backtest/index/backtest_staggered_dip.py --index Gold
    python momentum/staggered_dip_backtest_ui.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.backtest_cancel import check_cancelled  # noqa: E402
from momentum.etf.universes.india import (  # noqa: E402
    ETF_LABELS,
    ETF_TO_NSE_INDEX,
    tickers as INDIA_ETF_TICKERS,
)
from momentum.index.nifty_indices import (  # noqa: E402
    NIFTY_INDICES,
    resolve_selected_indices,
)
from utils.india_market_data import (  # noqa: E402
    format_range_label,
    get_india_market_data,
    get_india_market_data_run_stats,
    prepare_india_market_data_range,
)
from utils.nse_bhavcopy import today_ist  # noqa: E402

DEFAULT_INDEX_ID = "Nifty Bank"
_PERIODS_PER_YEAR = 252
_DISPLAY_DATE_FMT = "%d-%m-%Y"

GOLD_ETF = "GOLDBEES.NS"
SILVER_ETF = "SILVERBEES.NS"


@dataclass(frozen=True)
class StaggeredDipInstrument:
    index_id: str
    label: str
    index_yahoo: str
    etf_ticker: str


COMMODITY_INSTRUMENTS: dict[str, StaggeredDipInstrument] = {
    "Gold": StaggeredDipInstrument("Gold", "Gold", GOLD_ETF, GOLD_ETF),
    "Silver": StaggeredDipInstrument("Silver", "Silver", SILVER_ETF, SILVER_ETF),
}


def _resolve_commodity(index_id: str) -> StaggeredDipInstrument | None:
    key = (index_id or "").strip()
    if key in COMMODITY_INSTRUMENTS:
        return COMMODITY_INSTRUMENTS[key]
    by_label = {v.label: v for v in COMMODITY_INSTRUMENTS.values()}
    return by_label.get(key)


def staggered_dip_index_choices() -> list[tuple[str, str]]:
    """Display label and index id for UI / CLI (Nifty indices + Gold / Silver)."""
    choices = [(i.label, i.index_id) for i in NIFTY_INDICES]
    for inst in COMMODITY_INSTRUMENTS.values():
        choices.append((inst.label, inst.index_id))
    return sorted(choices, key=lambda row: row[0])


def resolve_staggered_dip_signal(index_id: str) -> tuple[str, str, str]:
    """Return ``(index_id, label, signal_yahoo)`` for dip-buy signals."""
    commodity = _resolve_commodity(index_id)
    if commodity:
        return commodity.index_id, commodity.label, commodity.index_yahoo
    idx = resolve_selected_indices([index_id])[0]
    return idx.index_id, idx.label, idx.yahoo_ticker


def _etf_display(ticker: str) -> str:
    raw = (ticker or "").strip()
    if not raw:
        return "—"
    label = ETF_LABELS.get(raw) or ETF_TO_NSE_INDEX.get(raw)
    if label:
        return f"{raw.replace('.NS', '')} ({label})"
    return raw.replace(".NS", "")


def _normalize_etf_ticker(value: str) -> str:
    raw = (value or "").strip().upper()
    if not raw:
        raise ValueError("ETF ticker is required.")
    if not raw.endswith(".NS"):
        raw = f"{raw}.NS"
    known = {t.upper() for t in INDIA_ETF_TICKERS}
    if raw not in known:
        raise ValueError(f"Unknown ETF ticker: {value!r}")
    return raw


def etfs_for_index(index_id: str) -> list[str]:
    """All NSE ETFs mapped to ``index_id`` (stable ticker order)."""
    commodity = _resolve_commodity(index_id)
    if commodity:
        return [commodity.etf_ticker]
    idx = resolve_selected_indices([index_id])[0]
    out: list[str] = []
    for etf in INDIA_ETF_TICKERS:
        if ETF_TO_NSE_INDEX.get(etf) == idx.index_id:
            out.append(etf)
    return out


def resolve_trade_etf(index_id: str, trade_etf: str | None = None) -> str:
    """Pick the trading ETF for an index (must track that index)."""
    commodity = _resolve_commodity(index_id)
    if commodity:
        if not trade_etf:
            return commodity.etf_ticker
        normalized = _normalize_etf_ticker(trade_etf)
        if normalized != commodity.etf_ticker:
            bare = commodity.etf_ticker.replace(".NS", "")
            raise ValueError(
                f"ETF {trade_etf!r} does not track {commodity.label!r}. Choose: {bare}"
            )
        return normalized
    options = etfs_for_index(index_id)
    if not options:
        idx = resolve_selected_indices([index_id])[0]
        raise ValueError(
            f"No listed ETF tracks {idx.label!r}. "
            "Choose an index that has a mapped ETF in universes/india.py."
        )
    if not trade_etf:
        return options[0]
    normalized = _normalize_etf_ticker(trade_etf)
    if normalized not in options:
        idx = resolve_selected_indices([index_id])[0]
        choices = ", ".join(t.replace(".NS", "") for t in options)
        raise ValueError(
            f"ETF {trade_etf!r} does not track {idx.label!r}. Choose: {choices}"
        )
    return normalized


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


def normalize_backtest_date(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Backtest date is required.")
    parts = raw.split("-")
    if len(parts) == 3 and len(parts[0]) == 4 and parts[0].isdigit():
        pd.Timestamp(raw)
        return raw
    from backtest.index.backtest_nifty_candlestick import _parse_user_date  # noqa: E402

    return _parse_user_date(raw).strftime("%Y-%m-%d")


@dataclass
class LotPosition:
    lot_num: int
    yahoo_ticker: str
    label: str
    entry_date: date
    entry_price: float
    entry_index_price: float
    shares: float
    deploy_amount: float


@dataclass
class StaggeredDipBacktestConfig:
    backtest_start: str
    backtest_end: str
    num_lots: int = 5
    profit_pct: float = 5.0
    dip_pct: float = 5.0
    initial_capital: float = 500_000.0
    index_id: str = DEFAULT_INDEX_ID
    trade_etf: str | None = None
    index_yahoo: str = field(init=False)
    index_label: str = field(init=False)
    trade_etf_ticker: str = field(init=False)

    def __post_init__(self) -> None:
        resolved_id, label, signal_yahoo = resolve_staggered_dip_signal(self.index_id)
        object.__setattr__(self, "index_id", resolved_id)
        object.__setattr__(self, "index_label", label)
        object.__setattr__(self, "index_yahoo", signal_yahoo)
        object.__setattr__(
            self,
            "trade_etf_ticker",
            resolve_trade_etf(resolved_id, self.trade_etf),
        )
        n = max(1, int(self.num_lots))
        object.__setattr__(self, "num_lots", n)
        object.__setattr__(self, "profit_pct", max(0.1, float(self.profit_pct)))
        object.__setattr__(self, "dip_pct", max(0.1, float(self.dip_pct)))

    @property
    def lot_size(self) -> float:
        return self.initial_capital / self.num_lots


@dataclass
class StaggeredDipBacktestEngine:
    config: StaggeredDipBacktestConfig
    progress_cb: Callable[[str], None] | None = None
    cancel_check: Callable[[], bool] | None = None
    _closes: dict[str, pd.Series] = field(default_factory=dict)
    _bar_dates: list[date] = field(default_factory=list)
    _positions: list[LotPosition] = field(default_factory=list)
    _cash: float = 0.0
    _anchor_price: float | None = None
    _last_lot_index_price: float | None = None
    _cycle_lot_size: float = 0.0
    _pending_redeploy: bool = False
    _pending_redeploy_lot: int | None = None
    _lots_deployed: int = 0
    _cycle_max_lot: int = 0
    _records: list[dict] = field(default_factory=list)
    _trade_rows: list[dict] = field(default_factory=list)
    _period_idx: int = 0
    _portfolio_value: float = 0.0
    _loaded: bool = False
    _index_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def __post_init__(self) -> None:
        self._cash = self.config.initial_capital
        self._portfolio_value = self.config.initial_capital
        self._cycle_lot_size = self.config.lot_size

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

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        if "Port_Return" in df.columns and len(df):
            df["Bench_Value"] = self.config.initial_capital * (
                (1 + df["Bench_Return"]).cumprod()
            )
        df.attrs["num_lots"] = self.config.num_lots
        df.attrs["profit_pct"] = self.config.profit_pct
        df.attrs["dip_pct"] = self.config.dip_pct
        df.attrs["index_label"] = self.config.index_label
        df.attrs["trade_etf"] = self.config.trade_etf_ticker
        return df

    @property
    def trade_log_df(self) -> pd.DataFrame:
        if not self._trade_rows:
            return pd.DataFrame()
        rows = [dict(r) for r in self._trade_rows]
        if self.finished and self._positions:
            self._apply_open_mtm(rows)
        return pd.DataFrame(rows)

    def _apply_open_mtm(self, rows: list[dict]) -> None:
        """Fill mark, unrealized P/L on buy rows still held at backtest end."""
        if not self._bar_dates:
            return
        last_date = self._bar_dates[-1]
        mark = self._close_on(self.config.trade_etf_ticker, last_date)
        if mark is None or mark <= 0:
            return
        open_keys = {(p.lot_num, p.entry_date) for p in self._positions}
        mark_label = _format_display_date(last_date)
        for row in rows:
            if row.get("Exit_Date") is not None:
                continue
            if row.get("Action") != "Buy":
                continue
            entry_date = pd.Timestamp(row["Date"]).date()
            if (int(row["Lot"]), entry_date) not in open_keys:
                continue
            entry = float(row["Entry"])
            amount = float(row["Amount"])
            if entry <= 0:
                continue
            shares = amount / entry
            pl_amt = shares * mark - amount
            pl_pct = (mark / entry - 1.0) * 100.0
            row["Exit"] = round(mark, 2)
            row["PL_%"] = round(pl_pct, 2)
            row["PL_Amt"] = round(pl_amt, 2)
            reason = str(row.get("Reason") or "")
            if "· open" not in reason:
                row["Reason"] = f"{reason} · open @ {mark_label}"

    def reset_run(self) -> None:
        self._period_idx = 0
        self._records = []
        self._trade_rows = []
        self._positions = []
        self._cash = self.config.initial_capital
        self._portfolio_value = self.config.initial_capital
        self._anchor_price = None
        self._last_lot_index_price = None
        self._cycle_lot_size = self.config.lot_size
        self._pending_redeploy = False
        self._pending_redeploy_lot = None
        self._lots_deployed = 0
        self._cycle_max_lot = 0

    def _close_on(self, yahoo_ticker: str, on_date: date) -> float | None:
        series = self._closes.get(yahoo_ticker)
        if series is None or series.empty:
            return None
        sliced = series[series.index <= pd.Timestamp(on_date)]
        if sliced.empty:
            return None
        return float(sliced.iloc[-1])

    def _index_return(self, start_date: date, end_date: date) -> float:
        if self._index_series.empty:
            return 0.0
        s = self._index_series[self._index_series.index <= pd.Timestamp(start_date)]
        e = self._index_series[self._index_series.index <= pd.Timestamp(end_date)]
        if s.empty or e.empty:
            return 0.0
        start_px = float(s.iloc[-1])
        end_px = float(e.iloc[-1])
        if start_px <= 0:
            return 0.0
        return end_px / start_px - 1.0

    def _portfolio_value_on(self, on_date: date) -> float:
        total = self._cash
        for pos in self._positions:
            px = self._close_on(pos.yahoo_ticker, on_date)
            if px is not None:
                total += pos.shares * px
        return total

    def _move_pct_from(self, index_px: float, ref_px: float | None) -> float | None:
        if ref_px is None or ref_px <= 0:
            return None
        return (index_px / ref_px - 1.0) * 100.0

    def _bench_move_pct(self, index_px: float) -> float | None:
        return self._move_pct_from(index_px, self._anchor_price)

    def _dip_move_from_last_lot(self, index_px: float) -> float | None:
        return self._move_pct_from(index_px, self._last_lot_index_price)

    def _available_lot_slots(self) -> int:
        return max(0, self.config.num_lots - len(self._positions))

    def _open_lot_numbers(self) -> set[int]:
        return {p.lot_num for p in self._positions}

    def _next_free_lot_num(self) -> int | None:
        """Lowest lot slot in 1..N not currently open."""
        open_nums = self._open_lot_numbers()
        for lot_num in range(1, self.config.num_lots + 1):
            if lot_num not in open_nums:
                return lot_num
        return None

    def _next_dip_lot_num(self) -> int | None:
        """Next lot for a dip buy: extend ladder 1→2→…→N only (never back-fill a freed slot)."""
        if self._available_lot_slots() <= 0:
            return None
        open_nums = self._open_lot_numbers()
        if not open_nums:
            return None
        next_lot = max(open_nums) + 1
        if next_lot > self.config.num_lots or next_lot in open_nums:
            return None
        return next_lot

    def _recalc_lot_size(self) -> None:
        slots = self._available_lot_slots()
        if slots > 0:
            self._cycle_lot_size = self._cash / slots

    def _next_dip_target_price(self) -> float | None:
        if self._last_lot_index_price is None or self._available_lot_slots() <= 0:
            return None
        return self._last_lot_index_price * (1.0 - self.config.dip_pct / 100.0)

    def _etf_profit_pct(self, pos: LotPosition, bar_date: date) -> float | None:
        etf_px = self._close_on(pos.yahoo_ticker, bar_date)
        if etf_px is None or pos.entry_price <= 0:
            return None
        return (etf_px / pos.entry_price - 1.0) * 100.0

    def _trade_label(self) -> str:
        return _etf_display(self.config.trade_etf_ticker)

    def _append_trade(
        self,
        *,
        bar_date: date,
        action: str,
        lot_num: int,
        label: str,
        entry: float | None,
        exit_px: float | None,
        amount: float,
        pl_pct: float | None,
        bench_pct: float | None,
        reason: str,
        pl_amt: float | None = None,
    ) -> None:
        self._trade_rows.append(
            {
                "Date": pd.Timestamp(bar_date),
                "Exit_Date": None,
                "Action": action,
                "Lot": lot_num,
                "Instrument": label,
                "Entry": entry,
                "Exit": exit_px,
                "Amount": round(amount, 2),
                "PL_%": round(pl_pct, 2) if pl_pct is not None else None,
                "PL_Amt": round(pl_amt, 2) if pl_amt is not None else None,
                "Index_%": round(bench_pct, 2) if bench_pct is not None else None,
                "Reason": reason,
            }
        )

    def _close_buy_row(
        self,
        pos: LotPosition,
        bar_date: date,
        etf_px: float,
        pl_pct: float,
        pl_amt: float,
    ) -> None:
        for row in self._trade_rows:
            if row.get("Exit_Date") is not None:
                continue
            if row.get("Action") != "Buy":
                continue
            if int(row["Lot"]) != pos.lot_num:
                continue
            if pd.Timestamp(row["Date"]).date() != pos.entry_date:
                continue
            row["Exit"] = round(etf_px, 2)
            row["Exit_Date"] = pd.Timestamp(bar_date)
            row["PL_%"] = round(pl_pct, 2)
            row["PL_Amt"] = round(pl_amt, 2)
            base = str(row.get("Reason") or "").split(" → ")[0]
            row["Reason"] = (
                f"{base} → profit book (+{self.config.profit_pct:g}% ETF @ "
                f"{_format_display_date(bar_date)})"
            )
            return

    def _deploy_lot(self, lot_num: int, bar_date: date, index_px: float, reason: str) -> bool:
        if lot_num < 1 or lot_num > self.config.num_lots:
            return False
        if len(self._positions) >= self.config.num_lots:
            return False
        if lot_num in self._open_lot_numbers():
            return False
        ticker = self.config.trade_etf_ticker
        label = self._trade_label()
        px = self._close_on(ticker, bar_date)
        if px is None or px <= 0:
            return False
        deploy = min(self._cycle_lot_size, self._cash)
        if deploy <= 0:
            return False
        shares = deploy / px
        self._cash -= deploy
        self._positions.append(
            LotPosition(
                lot_num=lot_num,
                yahoo_ticker=ticker,
                label=label,
                entry_date=bar_date,
                entry_price=px,
                entry_index_price=index_px,
                shares=shares,
                deploy_amount=deploy,
            )
        )
        self._lots_deployed = lot_num
        self._cycle_max_lot = max(self._cycle_max_lot, lot_num)
        self._last_lot_index_price = index_px
        index_pct = (
            (index_px / self._anchor_price - 1.0) * 100.0 if self._anchor_price else 0.0
        )
        self._append_trade(
            bar_date=bar_date,
            action="Buy",
            lot_num=lot_num,
            label=label,
            entry=px,
            exit_px=None,
            amount=deploy,
            pl_pct=None,
            bench_pct=index_pct,
            reason=reason,
        )
        return True

    def _exit_lot_on_profit(
        self,
        pos: LotPosition,
        bar_date: date,
        index_px: float,
        etf_px: float,
    ) -> dict:
        proceeds = pos.shares * etf_px
        self._cash += proceeds
        pl_pct = (etf_px / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0
        pl_amt = proceeds - pos.deploy_amount
        self._positions.remove(pos)
        self._close_buy_row(pos, bar_date, etf_px, pl_pct, pl_amt)
        return {
            "lot": pos.lot_num,
            "ticker": pos.label,
            "entry": pos.entry_price,
            "exit": etf_px,
            "pl_pct": pl_pct,
        }

    def _exit_lots_on_etf_profit(self, bar_date: date, index_px: float) -> list[dict]:
        exits: list[dict] = []
        for pos in list(self._positions):
            pl_pct = self._etf_profit_pct(pos, bar_date)
            if pl_pct is None or pl_pct + 1e-9 < self.config.profit_pct:
                continue
            etf_px = self._close_on(pos.yahoo_ticker, bar_date)
            if etf_px is None:
                continue
            exits.append(self._exit_lot_on_profit(pos, bar_date, index_px, etf_px))

        if exits:
            if not self._positions:
                self._schedule_redeploy_cycle()
            else:
                self._recalc_lot_size()
                if self._available_lot_slots() > 0 and self._cycle_lot_size > 0:
                    self._pending_redeploy_lot = exits[-1]["lot"]
                    self._pending_redeploy = True
        return exits

    def _schedule_redeploy_cycle(self) -> None:
        self._cycle_max_lot = 0
        self._lots_deployed = 0
        self._last_lot_index_price = None
        self._anchor_price = None
        self._pending_redeploy_lot = None
        self._recalc_lot_size()
        self._pending_redeploy = True

    def _redeploy_lot_num(self) -> int:
        if self._pending_redeploy_lot is not None:
            return self._pending_redeploy_lot
        return 1

    def load_data(self) -> None:
        start = pd.Timestamp(normalize_backtest_date(self.config.backtest_start))
        end = pd.Timestamp(normalize_backtest_date(self.config.backtest_end))

        tickers = {self.config.index_yahoo, self.config.trade_etf_ticker}

        self._log(
            f"Loading {self.config.index_label} + {_etf_display(self.config.trade_etf_ticker)} — "
            f"{format_range_label(start, end)} (daily)"
        )
        prepare_india_market_data_range(
            start,
            end,
            reset_stats=True,
            include_cm_bhavcopy=True,
            include_index_archive=True,
            cancel_check=self.cancel_check,
        )

        self._closes.clear()
        for ticker in sorted(tickers):
            check_cancelled(self.cancel_check)
            df = get_india_market_data(ticker, start, end)
            if df is None or df.empty or "Close" not in df.columns:
                self._log(f"  Skip {ticker}: no OHLC")
                continue
            self._closes[ticker] = df["Close"].astype(float)

        index_closes = self._closes.get(self.config.index_yahoo)
        if index_closes is None or index_closes.empty:
            raise ValueError(f"No index data for {self.config.index_label}")

        etf_closes = self._closes.get(self.config.trade_etf_ticker)
        if etf_closes is None or etf_closes.empty:
            raise ValueError(
                f"No ETF data for {_etf_display(self.config.trade_etf_ticker)}"
            )

        self._index_series = index_closes
        bar_dates: set[date] = set()
        for ts in index_closes.index:
            ts_norm = pd.Timestamp(ts)
            if start <= ts_norm < end:
                bar_dates.add(ts_norm.date())
        self._bar_dates = sorted(bar_dates)
        self._loaded = True

        stats = get_india_market_data_run_stats()
        self._log(
            f"Loaded index + ETF, {len(self._bar_dates)} daily bars ({stats.summary()})"
        )

    def apply_run_context(self, config: StaggeredDipBacktestConfig) -> None:
        if not self._loaded:
            return
        self.config = config
        self.reset_run()

    def step_period(self) -> dict | None:
        if not self._loaded or self.finished:
            return None

        check_cancelled(self.cancel_check)
        idx = self._period_idx
        bar_date = self._bar_dates[idx]
        index_px = self._close_on(self.config.index_yahoo, bar_date)
        if index_px is None or index_px <= 0:
            self._period_idx += 1
            return None

        before = self._bar_dates[idx - 1] if idx > 0 else None
        val_start = (
            self.config.initial_capital
            if idx == 0
            else self._portfolio_value_on(before)
        )

        period_buys: list[dict] = []
        period_sells: list[dict] = []

        if self._pending_redeploy:
            self._recalc_lot_size()
            if self._anchor_price is None:
                self._anchor_price = index_px
            lot_num = self._redeploy_lot_num()
            if self._available_lot_slots() > 0 and self._cycle_lot_size > 0:
                reason = (
                    "Redeploy after all lots closed (next day)"
                    if lot_num == 1 and not self._positions
                    else f"Redeploy after profit book (next day, ₹{self._cycle_lot_size:,.0f}/lot)"
                )
                if self._deploy_lot(lot_num, bar_date, index_px, reason):
                    period_buys.append({"lot": lot_num})
            self._pending_redeploy = False
            self._pending_redeploy_lot = None
        elif self._anchor_price is None:
            self._anchor_price = index_px
            self._recalc_lot_size()
            if self._deploy_lot(1, bar_date, index_px, "Initial lot"):
                period_buys.append({"lot": 1})
        else:
            period_sells = self._exit_lots_on_etf_profit(bar_date, index_px)
            if not self._pending_redeploy and self._last_lot_index_price is not None:
                next_lot = self._next_dip_lot_num()
                if next_lot is not None:
                    dip_move = self._dip_move_from_last_lot(index_px)
                    if dip_move is not None and dip_move - 1e-9 <= -self.config.dip_pct:
                        reason = (
                            f"Dip buy (lot {next_lot}, "
                            f"-{self.config.dip_pct:g}% from last lot)"
                        )
                        if self._deploy_lot(next_lot, bar_date, index_px, reason):
                            period_buys.append({"lot": next_lot})

        val_end = self._portfolio_value_on(bar_date)
        port_ret = (val_end / val_start - 1.0) if val_start > 0 else 0.0
        self._portfolio_value = val_end

        index_start = before if before is not None else bar_date
        index_ret = self._index_return(index_start, bar_date)

        open_labels = [f"L{p.lot_num}" for p in self._positions]
        record = {
            "Period": idx + 1,
            "Bar_Date": pd.Timestamp(bar_date),
            "Index_Close": index_px,
            "Anchor_Price": self._anchor_price,
            "Lots_Deployed": self._lots_deployed,
            "Cash": round(self._cash, 2),
            "Buys": len(period_buys),
            "Sells": len(period_sells),
            "Port_Return": port_ret,
            "Bench_Return": index_ret,
            "Portfolio_Value": val_end,
            "Open_Lots": open_labels,
            "Next_Dip_Target": self._next_dip_target_price(),
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
    index_rets = df["Bench_Return"].values
    n_periods = len(port_rets)
    periods_per_year = _PERIODS_PER_YEAR

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    index_total = df["Bench_Value"].iloc[-1] / capital - 1
    years = n_periods / periods_per_year if periods_per_year > 0 else 0.0

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    index_cagr = (1 + index_total) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    index_cum = (1 + pd.Series(index_rets)).cumprod()
    index_max_dd = float((index_cum / index_cum.cummax() - 1).min())

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
    alpha = cagr - index_cagr

    return {
        "Strategy": "Staggered dip buy / profit book",
        "Index": str(df.attrs.get("index_label", "—")),
        "ETF": _etf_display(str(df.attrs.get("trade_etf", ""))),
        "Lots_N": int(df.attrs.get("num_lots", 0)),
        "Profit_X_%": float(df.attrs.get("profit_pct", 0)),
        "Dip_Y_%": float(df.attrs.get("dip_pct", 0)),
        "Period": (
            f"{_format_display_date(df['Bar_Date'].iloc[0])} to "
            f"{_format_display_date(df['Bar_Date'].iloc[-1])}"
        ),
        "Trading_Days": n_periods,
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(float(np.mean(port_rets > 0) * 100), 1),
        "Avg_Daily_Return_%": round(float(np.mean(port_rets) * 100), 3),
        "Profit_Books": int(df["Sells"].sum()) if "Sells" in df.columns else 0,
        "Dip_Buys": int(df["Buys"].sum()) if "Buys" in df.columns else 0,
        "Index_Total_Return_%": round(index_total * 100, 2),
        "Index_CAGR_%": round(index_cagr * 100, 2),
        "Index_Max_Drawdown_%": round(index_max_dd * 100, 2),
        "Final_Value": round(float(df["Portfolio_Value"].iloc[-1]), 2),
        "Index_BuyHold_Value": round(float(df["Bench_Value"].iloc[-1]), 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Ann_Volatility_%": round(ann_vol * 100, 2),
        "Alpha_%": round(alpha * 100, 2),
    }


def run_backtest(
    backtest_start: str,
    backtest_end: str,
    *,
    num_lots: int = 5,
    profit_pct: float = 5.0,
    dip_pct: float = 5.0,
    initial_capital: float = 500_000.0,
    index_id: str = DEFAULT_INDEX_ID,
    trade_etf: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    engine = StaggeredDipBacktestEngine(
        StaggeredDipBacktestConfig(
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            num_lots=num_lots,
            profit_pct=profit_pct,
            dip_pct=dip_pct,
            initial_capital=initial_capital,
            index_id=index_id,
            trade_etf=trade_etf,
        ),
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )
    engine.load_data()
    return engine.run_all()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staggered dip-buying backtest (N lots, X%% profit book, Y%% dip steps)",
    )
    parser.add_argument("--start", default=f"{today_ist().year}-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lots", "-n", type=int, default=5, help="Number of equal capital lots (N)")
    parser.add_argument(
        "--profit",
        "-x",
        type=float,
        default=5.0,
        help="Exit each lot when its ETF price is up X%% from that lot's entry",
    )
    parser.add_argument(
        "--dip",
        "-y",
        type=float,
        default=5.0,
        help="Deploy next lot when index falls Y%% from the previous lot buy level",
    )
    parser.add_argument("--capital", type=float, default=500_000.0)
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_ID,
        help='Signal source (Nifty index or "Gold" / "Silver"; commodities use ETF price for dips)',
    )
    parser.add_argument(
        "--etf",
        default=None,
        help="ETF to trade (default: first ETF mapped to the index)",
    )
    args = parser.parse_args()

    end = args.end or pd.Timestamp(today_ist()).strftime("%Y-%m-%d")

    df = run_backtest(
        args.start,
        end,
        num_lots=args.lots,
        profit_pct=args.profit,
        dip_pct=args.dip,
        initial_capital=args.capital,
        index_id=args.index,
        trade_etf=args.etf,
        progress_cb=print,
    )
    if df.empty:
        print("No results.")
        return 1

    metrics = compute_metrics(df, args.capital)
    print(f"\nStaggered dip backtest — {metrics.get('Period', '')}")
    for key, value in metrics.items():
        if key != "Period":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
