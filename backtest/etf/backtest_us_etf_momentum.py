"""
Walk-forward backtest engine for US ETF momentum rankers.

Strategies (mirror live scanners):
  - momentum_us_etfs.py           — abs momentum (2W/1M/3M)
  - momentum_us_rs_etfs.py        — abs + RS blended (2W/1M/3M)
  - momentum_us_rs_etfs_adaptive.py — weighted RS (1W/2W/1M)

Used by momentum/etf_us_momentum_backtest_ui.py and optional CLI.

Examples:
    python backtest/etf/backtest_us_etf_momentum.py --strategy momentum_us_rs_etfs
    python momentum/etf_us_momentum_backtest_ui.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.etf_us_momentum_engine import (  # noqa: E402
    BENCH_EMA_FAST,
    BENCH_EMA_SLOW,
    BENCHMARK_TICKER,
    EMA_SPAN,
    LB_1M,
    LB_1W,
    LB_2W,
    LB_3M,
    LB_52W,
    MIN_HISTORY_SESSIONS,
    PROXIMITY_OF_52W_HIGH,
    RETURN_SUFFIXES_LONG,
    RS_RANK_COLS,
    W_ABS_LONG,
    W_RS_ADAPTIVE,
    _weighted_excess_return,
    classify_ema_regime,
)
from momentum.etf.universes import us_universe  # noqa: E402
from momentum.rrg_core import rrg_config_date_str  # noqa: E402
from momentum.rrg_ema_exit import (  # noqa: E402
    _daily_close_series,
    close_below_9ema,
    simulate_week_with_exits,
    split_holdings_at_stop_loss,
    split_holdings_below_9ema,
)
from momentum.rrg_portfolio_fill import equal_weight_port_return  # noqa: E402

STRATEGY_KEYS = (
    "momentum_us_etfs",
    "momentum_us_rs_etfs",
    "momentum_us_rs_etfs_adaptive",
)

STRATEGY_LABELS: dict[str, str] = {
    "momentum_us_etfs": "Abs Momentum",
    "momentum_us_rs_etfs": "RS Blended",
    "momentum_us_rs_etfs_adaptive": "RS Adaptive",
}


def _symbol_display(yahoo_ticker: str) -> str:
    return yahoo_ticker


def _etf_tickers() -> list[str]:
    us_universe.ensure_loaded()
    return list(us_universe.TICKERS)

REBALANCE_ALIASES = {
    "bi-weekly": "biweekly",
    "bi_weekly": "biweekly",
    "biweekly": "biweekly",
    "weekly": "weekly",
    "monthly": "monthly",
}

REBALANCE_LABELS = {
    "weekly": "Weekly",
    "biweekly": "Biweekly",
    "monthly": "Monthly",
}

PERIODS_PER_YEAR = {
    "weekly": 52,
    "biweekly": 26,
    "monthly": 12,
}

MIN_HISTORY = LB_52W + LB_3M + 30

_SYMBOL_MAP: dict[str, str] | None = None


def _get_symbol_map() -> dict[str, str]:
    global _SYMBOL_MAP
    if _SYMBOL_MAP is None:
        _SYMBOL_MAP = {sym: sym for sym in _etf_tickers()}
    return _SYMBOL_MAP


def normalize_backtest_date(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Backtest date is required.")
    parts = raw.split("-")
    if len(parts) == 3 and len(parts[0]) == 4 and parts[0].isdigit():
        pd.Timestamp(raw)
        return raw
    return rrg_config_date_str(raw)


def _yf_download_end_exclusive(backtest_end: str) -> str:
    end_iso = normalize_backtest_date(backtest_end)
    return (pd.Timestamp(end_iso) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _adj_col(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _vol_col(df: pd.DataFrame) -> pd.Series:
    s = df["Volume"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _close_col(df: pd.DataFrame) -> pd.Series:
    s = df["Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _download_adj_vol_close(
    yahoo_symbols: list[str], start: str, end: str
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]]:
    start_iso = normalize_backtest_date(start)
    end_iso = _yf_download_end_exclusive(end)
    extra_start = (
        pd.Timestamp(start_iso) - pd.Timedelta(days=int(MIN_HISTORY * 1.6))
    ).strftime("%Y-%m-%d")
    adj_store: dict[str, pd.Series] = {}
    vol_store: dict[str, pd.Series] = {}
    close_store: dict[str, pd.Series] = {}
    for sym in yahoo_symbols:
        try:
            df = yf.download(
                sym,
                start=extra_start,
                end=end_iso,
                multi_level_index=False,
                auto_adjust=False,
                progress=False,
            )
            if df is None or len(df) == 0:
                continue
            adj_store[sym] = _adj_col(df)
            vol_store[sym] = _vol_col(df)
            close_store[sym] = _close_col(df)
        except Exception:
            pass
    return adj_store, vol_store, close_store


def strategy_defaults(strategy_key: str) -> dict[str, Any]:
    return {
        "portfolio_size": 7,
        "exit_rank_threshold": 10,
        "benchmark_ticker": BENCHMARK_TICKER,
        "proximity_of_52w_high": PROXIMITY_OF_52W_HIGH,
        "rebalance_period": "weekly",
        "use_regime_filter": False,
    }


def _rank_abs_at_date(
    etf_adj: dict[str, pd.Series],
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    summary: list[dict] = []
    for yahoo_sym in _etf_tickers():
        if yahoo_sym not in etf_adj:
            continue
        adj = etf_adj[yahoo_sym].loc[:as_of]
        n = len(adj)
        if n < LB_3M:
            continue

        ema200 = adj.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1]
        high_52w = adj.iloc[-min(LB_52W, n) :].max()
        last = float(adj.iloc[-1])
        if last < ema200 or last < high_52w * proximity_of_52w_high:
            continue

        summary.append(
            {
                "Symbol": _symbol_display(yahoo_sym),
                "Return_2W": (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100,
                "Return_1M": (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100,
                "Return_3M": (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100,
            }
        )

    if not summary:
        return pd.DataFrame()

    w2w, w1m, w3m = W_ABS_LONG
    df = pd.DataFrame(summary)
    df["Rank_2W"] = df["Return_2W"].rank(ascending=False)
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False)
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False)
    df["Final_Rank"] = (
        w2w * df["Rank_2W"] + w1m * df["Rank_1M"] + w3m * df["Rank_3M"]
    )
    out = df.sort_values("Final_Rank").reset_index(drop=True)
    out["Rank_Position"] = np.arange(1, len(out) + 1)
    return out


def _collect_rs_rows_at_date(
    etf_adj: dict[str, pd.Series],
    etf_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    bench_slice = bench_adj.loc[:as_of]
    if len(bench_slice) < LB_3M:
        return pd.DataFrame()

    rows: list[dict] = []
    for yahoo_sym in _etf_tickers():
        if yahoo_sym not in etf_adj:
            continue
        adj = etf_adj[yahoo_sym].loc[:as_of]
        if len(adj) < MIN_HISTORY_SESSIONS:
            continue
        ema200 = float(adj.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])
        high_52w = float(adj.iloc[-min(LB_52W, len(adj)) :].max())
        last = float(adj.iloc[-1])
        if last < ema200 or last < high_52w * proximity_of_52w_high:
            continue

        ret_1w = (adj.iloc[-1] / adj.iloc[-LB_1W] - 1) * 100
        ret_2w = (adj.iloc[-1] / adj.iloc[-LB_2W] - 1) * 100
        ret_1m = (adj.iloc[-1] / adj.iloc[-LB_1M] - 1) * 100
        ret_3m = (adj.iloc[-1] / adj.iloc[-LB_3M] - 1) * 100

        nx = bench_slice.reindex(adj.index).ffill()
        tail = nx.iloc[-LB_3M:]
        rs_1w = rs_2w = rs_1m = rs_3m = float("nan")
        if not tail.isna().any() and (tail > 0).all():
            bn_1w = (nx.iloc[-1] / nx.iloc[-LB_1W] - 1) * 100
            bn_2w = (nx.iloc[-1] / nx.iloc[-LB_2W] - 1) * 100
            bn_1m = (nx.iloc[-1] / nx.iloc[-LB_1M] - 1) * 100
            bn_3m = (nx.iloc[-1] / nx.iloc[-LB_3M] - 1) * 100
            rs_1w = ret_1w - bn_1w
            rs_2w = ret_2w - bn_2w
            rs_1m = ret_1m - bn_1m
            rs_3m = ret_3m - bn_3m

        rows.append(
            {
                "Symbol": _symbol_display(yahoo_sym),
                "Return_1W": ret_1w,
                "Return_2W": ret_2w,
                "Return_1M": ret_1m,
                "Return_3M": ret_3m,
                "RS_1W_vs_SP500": rs_1w,
                "RS_2W_vs_SP500": rs_2w,
                "RS_1M_vs_SP500": rs_1m,
                "RS_3M_vs_SP500": rs_3m,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _rank_rs_blended_at_date(
    etf_adj: dict[str, pd.Series],
    etf_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    df = _collect_rs_rows_at_date(
        etf_adj, etf_vol, bench_adj, as_of, proximity_of_52w_high=proximity_of_52w_high
    )
    if df.empty:
        return df

    w2w, w1m, w3m = W_ABS_LONG
    for suf in RETURN_SUFFIXES_LONG:
        df[f"Rank_{suf}"] = df[f"Return_{suf}"].rank(ascending=False)
        df[f"Rank_RS_{suf}"] = df[f"RS_{suf}_vs_SP500"].rank(
            ascending=False, na_option="bottom"
        )

    abs_rank_score = (
        w2w * df["Rank_2W"] + w1m * df["Rank_1M"] + w3m * df["Rank_3M"]
    )
    rs_rank_score = (
        w2w * df["Rank_RS_2W"] + w1m * df["Rank_RS_1M"] + w3m * df["Rank_RS_3M"]
    )
    df["Final_Rank"] = (abs_rank_score + rs_rank_score) / 2
    out = df.sort_values("Final_Rank").reset_index(drop=True)
    out["Rank_Position"] = np.arange(1, len(out) + 1)
    return out


def _rank_rs_adaptive_at_date(
    etf_adj: dict[str, pd.Series],
    etf_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    df = _collect_rs_rows_at_date(
        etf_adj, etf_vol, bench_adj, as_of, proximity_of_52w_high=proximity_of_52w_high
    )
    if df.empty:
        return df

    df["Weighted_RS_pct"] = df.apply(_weighted_excess_return, axis=1)
    df = df.dropna(subset=["Weighted_RS_pct"]).copy()
    if df.empty:
        return df

    out = df.sort_values("Weighted_RS_pct", ascending=False).reset_index(drop=True)
    out["Rank_Position"] = np.arange(1, len(out) + 1)
    return out


def rank_at_date(
    strategy_key: str,
    etf_adj: dict[str, pd.Series],
    etf_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    if strategy_key == "momentum_us_etfs":
        return _rank_abs_at_date(
            etf_adj, as_of, proximity_of_52w_high=proximity_of_52w_high
        )
    if strategy_key == "momentum_us_rs_etfs":
        return _rank_rs_blended_at_date(
            etf_adj,
            etf_vol,
            bench_adj,
            as_of,
            proximity_of_52w_high=proximity_of_52w_high,
        )
    if strategy_key == "momentum_us_rs_etfs_adaptive":
        return _rank_rs_adaptive_at_date(
            etf_adj,
            etf_vol,
            bench_adj,
            as_of,
            proximity_of_52w_high=proximity_of_52w_high,
        )
    raise ValueError(f"Unknown strategy {strategy_key!r}")


def _rebalance_dates(
    bench_adj: pd.Series, backtest_start: str, backtest_end: str, period: str
) -> pd.DatetimeIndex:
    start_iso = normalize_backtest_date(backtest_start)
    end_iso = normalize_backtest_date(backtest_end)
    all_dates = bench_adj.loc[start_iso:end_iso].index
    if len(all_dates) == 0:
        raise RuntimeError("No trading dates in the backtest window")

    period = REBALANCE_ALIASES.get(period.strip().lower(), period.strip().lower())

    if period == "weekly":
        dates = all_dates.to_series().groupby(all_dates.to_period("W")).last().values
    elif period == "monthly":
        dates = all_dates.to_series().groupby(all_dates.to_period("M")).last().values
    elif period == "biweekly":
        rebal: list[pd.Timestamp] = []
        last_rebal: pd.Timestamp | None = None
        for d in all_dates:
            if last_rebal is None or (d - last_rebal).days >= 14:
                rebal.append(d)
                last_rebal = d
        dates = rebal
    else:
        raise ValueError(f"Unknown rebalance period {period!r}")

    out = pd.DatetimeIndex(dates)
    return out[out >= all_dates[0]]


def _above_9ema_at_rebal(
    sym: str,
    daily_close: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> bool:
    close = _daily_close_series(daily_close, sym)
    if close is None:
        return True
    return not close_below_9ema(close, as_of)


def _select_holdings(
    ranked_df: pd.DataFrame,
    prev_holdings: list[str],
    top_n: int,
    worst_rank_held: int,
    *,
    exit_rank_enabled: bool = False,
    entry_above_9ema: bool = True,
    daily_close: dict[str, pd.Series] | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[str]:
    if ranked_df.empty:
        return []

    rank_map = dict(zip(ranked_df["Symbol"], ranked_df["Rank_Position"]))
    if exit_rank_enabled:
        retained = [
            sym for sym in prev_holdings if sym in rank_map and rank_map[sym] <= worst_rank_held
        ]
    else:
        retained = [sym for sym in prev_holdings if sym in rank_map]
    retained_set = set(retained)
    retained.sort(key=lambda s: rank_map[s])
    result = list(retained)
    slots_left = top_n - len(result)
    if slots_left <= 0:
        return result[:top_n]

    for sym in ranked_df["Symbol"]:
        if sym in retained_set:
            continue
        if entry_above_9ema:
            if daily_close is None or as_of is None:
                raise ValueError("entry_above_9ema requires daily_close and as_of")
            if not _above_9ema_at_rebal(sym, daily_close, as_of):
                continue
        elif rank_map[sym] > top_n:
            continue
        result.append(sym)
        slots_left -= 1
        if slots_left <= 0:
            break

    return result[:top_n]


def _exit_reason(
    sym: str,
    *,
    rank_map: dict[str, int],
    holdings: list[str],
    top_n: int,
    exit_rank_threshold: int,
    exit_rank_enabled: bool,
) -> str:
    if sym not in rank_map:
        return "Not in ranked universe (failed screen)"
    rank_pos = rank_map[sym]
    if exit_rank_enabled and rank_pos > exit_rank_threshold:
        return f"Rank {rank_pos} > exit rank {exit_rank_threshold}"
    if sym not in holdings:
        return f"Displaced — rank {rank_pos} (portfolio top {top_n} filled)"
    return "Exit"


@dataclass
class EtfMomentumBacktestConfig:
    strategy_key: str
    backtest_start: str
    backtest_end: str
    rebalance_period: str = "weekly"
    portfolio_size: int = 7
    exit_rank_threshold: int = 10
    exit_rank_enabled: bool = False
    benchmark_ticker: str = BENCHMARK_TICKER
    proximity_of_52w_high: float = PROXIMITY_OF_52W_HIGH
    initial_capital: float = 100_000.0
    use_regime_filter: bool = False
    exit_below_9ema: bool = True
    entry_above_9ema: bool = True
    exit_stop_loss: bool = False
    stop_loss_pct: float = 5.0


def _intraweek_exits_enabled(cfg: EtfMomentumBacktestConfig) -> bool:
    return bool(cfg.exit_below_9ema) or bool(cfg.exit_stop_loss)


@dataclass
class EtfMomentumBacktestEngine:
    config: EtfMomentumBacktestConfig
    progress_cb: Callable[[str], None] | None = None
    _etf_adj: dict[str, pd.Series] = field(default_factory=dict)
    _etf_vol: dict[str, pd.Series] = field(default_factory=dict)
    _etf_close: dict[str, pd.Series] = field(default_factory=dict)
    _bench_adj: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    _rebal_dates: list[pd.Timestamp] = field(default_factory=list)
    _records: list[dict] = field(default_factory=list)
    _period_idx: int = 0
    _portfolio_value: float = 0.0
    _prev_holdings: list[str] = field(default_factory=list)
    _entry_prices: dict[str, float] = field(default_factory=dict)
    _loaded: bool = False

    def __post_init__(self) -> None:
        self._portfolio_value = self.config.initial_capital

    def _log(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def total_periods(self) -> int:
        return max(0, len(self._rebal_dates) - 1)

    @property
    def current_period(self) -> int:
        return self._period_idx

    @property
    def finished(self) -> bool:
        return self._period_idx >= self.total_periods

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        if "Port_Return" in df.columns and len(df):
            df["Bench_Value"] = self.config.initial_capital * (
                (1 + df["Bench_Return"]).cumprod()
            )
        df.attrs["strategy_key"] = self.config.strategy_key
        df.attrs["rebalance_period"] = self.config.rebalance_period
        df.attrs["portfolio_size"] = self.config.portfolio_size
        df.attrs["benchmark_ticker"] = self.config.benchmark_ticker
        df.attrs["exit_rank_enabled"] = self.config.exit_rank_enabled
        df.attrs["use_regime_filter"] = self.config.use_regime_filter
        df.attrs["exit_below_9ema"] = self.config.exit_below_9ema
        df.attrs["entry_above_9ema"] = self.config.entry_above_9ema
        df.attrs["exit_stop_loss"] = self.config.exit_stop_loss
        df.attrs["stop_loss_pct"] = self.config.stop_loss_pct
        return df

    def reset_run(self) -> None:
        self._period_idx = 0
        self._records = []
        self._portfolio_value = self.config.initial_capital
        self._prev_holdings = []
        self._entry_prices = {}

    @staticmethod
    def _adj_price(etf_adj: dict[str, pd.Series], yahoo_sym: str, as_of: pd.Timestamp) -> float | None:
        series = etf_adj.get(yahoo_sym)
        if series is None or series.empty:
            return None
        sliced = series.loc[:as_of]
        if sliced.empty:
            return None
        return round(float(sliced.iloc[-1]), 2)

    @staticmethod
    def _pl_pct(entry: float | None, mark: float | None) -> float | None:
        if entry is None or mark is None or entry <= 0:
            return None
        return round((mark / entry - 1.0) * 100.0, 2)

    def _ensure_entry_prices(self, holdings: list[str], rebal_date: pd.Timestamp) -> None:
        for sym_excel in holdings:
            if sym_excel in self._entry_prices:
                continue
            yh = _get_symbol_map().get(sym_excel)
            if not yh:
                continue
            px = self._adj_price(self._etf_adj, yh, rebal_date)
            if px is not None:
                self._entry_prices[sym_excel] = px

    def _daily_close_excel(self) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for yh, close in self._etf_close.items():
            out[_symbol_display(yh)] = close
        return out

    def _weekly_adj_excel(self) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for yh, adj in self._etf_adj.items():
            out[_symbol_display(yh)] = adj
        return out

    def _rebal_stop_loss_exit_rows(
        self,
        dropped: list[str],
        rebal_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        rows: list[dict[str, Any]] = []
        for sym_excel in dropped:
            yh = _get_symbol_map().get(sym_excel)
            entry = self._entry_prices.get(sym_excel)
            exit_px = self._adj_price(self._etf_adj, yh, rebal_date) if yh else None
            if entry is None:
                entry = exit_px
            rows.append(
                {
                    "ticker": sym_excel,
                    "status": "Exit",
                    "entry": entry,
                    "exit": exit_px,
                    "exit_date": rebal_date,
                    "exit_reason": (
                        f"Stop loss ({cfg.stop_loss_pct:g}% from entry "
                        f"{entry:,.2f}) at rebalance"
                        if entry is not None
                        else f"Stop loss ({cfg.stop_loss_pct:g}%) at rebalance"
                    ),
                    "pl_pct": self._pl_pct(entry, exit_px),
                }
            )
            self._entry_prices.pop(sym_excel, None)
        return rows

    def _rebal_9ema_exit_rows(
        self,
        dropped: list[str],
        rebal_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sym_excel in dropped:
            yh = _get_symbol_map().get(sym_excel)
            entry = self._entry_prices.get(sym_excel)
            exit_px = self._adj_price(self._etf_adj, yh, rebal_date) if yh else None
            if entry is None:
                entry = exit_px
            rows.append(
                {
                    "ticker": sym_excel,
                    "status": "Exit",
                    "entry": entry,
                    "exit": exit_px,
                    "exit_date": rebal_date,
                    "exit_reason": "Close below 9 EMA at rebalance",
                    "pl_pct": self._pl_pct(entry, exit_px),
                }
            )
            self._entry_prices.pop(sym_excel, None)
        return rows

    def _intraweek_exit_rows(
        self,
        mid_week_9ema: list[tuple[str, pd.Timestamp]],
        mid_week_stop_loss: list[tuple[str, pd.Timestamp]],
        rebal_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        rows: list[dict[str, Any]] = []
        for sym_excel, exit_day in mid_week_9ema:
            yh = _get_symbol_map().get(sym_excel)
            entry = self._entry_prices.get(sym_excel)
            rebal_px = self._adj_price(self._etf_adj, yh, rebal_date) if yh else None
            if entry is None:
                entry = rebal_px
            exit_px = self._adj_price(self._etf_adj, yh, exit_day) if yh else None
            rows.append(
                {
                    "ticker": sym_excel,
                    "status": "Exit",
                    "entry": entry,
                    "exit": exit_px,
                    "exit_date": exit_day,
                    "exit_reason": "Close below 9 EMA (mid-period)",
                    "pl_pct": self._pl_pct(entry, exit_px),
                }
            )
            self._entry_prices.pop(sym_excel, None)
        for sym_excel, exit_day in mid_week_stop_loss:
            yh = _get_symbol_map().get(sym_excel)
            entry = self._entry_prices.get(sym_excel)
            rebal_px = self._adj_price(self._etf_adj, yh, rebal_date) if yh else None
            if entry is None:
                entry = rebal_px
            exit_px = self._adj_price(self._etf_adj, yh, exit_day) if yh else None
            rows.append(
                {
                    "ticker": sym_excel,
                    "status": "Exit",
                    "entry": entry,
                    "exit": exit_px,
                    "exit_date": exit_day,
                    "exit_reason": (
                        f"Stop loss ({cfg.stop_loss_pct:g}% from entry "
                        f"{entry:,.2f})"
                        if entry is not None
                        else f"Stop loss ({cfg.stop_loss_pct:g}%)"
                    ),
                    "pl_pct": self._pl_pct(entry, exit_px),
                }
            )
            self._entry_prices.pop(sym_excel, None)
        return rows

    def _simulate_period_returns(
        self,
        holdings: list[str],
        rebal_date: pd.Timestamp,
        next_date: pd.Timestamp,
    ) -> tuple[float, list[str], list[dict[str, Any]]]:
        """Equal-weight period return with optional 9 EMA / stop-loss intraperiod exits."""
        cfg = self.config
        if not holdings:
            return 0.0, [], []

        if _intraweek_exits_enabled(cfg):
            slot_returns, end_holdings, mid_9ema, mid_sl = simulate_week_with_exits(
                holdings,
                rebal_date,
                next_date,
                self._daily_close_excel(),
                self._weekly_adj_excel(),
                cfg.portfolio_size,
                exit_below_9ema=bool(cfg.exit_below_9ema),
                exit_stop_loss=bool(cfg.exit_stop_loss),
                stop_loss_pct=float(cfg.stop_loss_pct),
                entry_prices=dict(self._entry_prices),
                daily_adj=self._weekly_adj_excel(),
            )
            port_ret = equal_weight_port_return(slot_returns, len(holdings))
            intraweek_exits = self._intraweek_exit_rows(mid_9ema, mid_sl, rebal_date)
            return port_ret, end_holdings, intraweek_exits

        period_rets: list[float] = []
        end_holdings: list[str] = []
        for sym_excel in holdings:
            yh = _get_symbol_map().get(sym_excel)
            if not yh or yh not in self._etf_adj:
                period_rets.append(0.0)
                end_holdings.append(sym_excel)
                continue
            series = self._etf_adj[yh]
            rebal_slice = series.loc[:rebal_date]
            if len(rebal_slice) == 0:
                period_rets.append(0.0)
                end_holdings.append(sym_excel)
                continue
            p_rebal = float(rebal_slice.iloc[-1])
            mark_px = self._adj_price(self._etf_adj, yh, next_date)
            period_rets.append(
                self._pl_pct(p_rebal, mark_px) / 100.0 if mark_px is not None else 0.0
            )
            end_holdings.append(sym_excel)

        port_ret = float(np.mean(period_rets)) if period_rets else 0.0
        return port_ret, end_holdings, []

    def _build_period_position_rows(
        self,
        *,
        rebal_date: pd.Timestamp,
        next_date: pd.Timestamp,
        prev_holdings: list[str],
        holdings_at_rebal: list[str],
        end_holdings: list[str],
        rebal_9ema_exits: list[dict[str, Any]],
        intraweek_exits: list[dict[str, Any]],
        rebal_sl_exits: list[dict[str, Any]],
        ranked_df: pd.DataFrame,
        top_n: int,
        exit_rank_threshold: int,
        exit_rank_enabled: bool,
    ) -> list[dict[str, Any]]:
        prev_set = set(prev_holdings)
        rank_map = (
            dict(zip(ranked_df["Symbol"], ranked_df["Rank_Position"]))
            if not ranked_df.empty
            else {}
        )
        rows: list[dict[str, Any]] = []

        for sym_excel in sorted(prev_set - set(holdings_at_rebal)):
            yh = _get_symbol_map().get(sym_excel)
            entry = self._entry_prices.get(sym_excel)
            exit_px = self._adj_price(self._etf_adj, yh, rebal_date) if yh else None
            if entry is None:
                entry = exit_px
            rows.append(
                {
                    "ticker": sym_excel,
                    "status": "Exit",
                    "entry": entry,
                    "exit": exit_px,
                    "exit_date": rebal_date,
                    "exit_reason": _exit_reason(
                        sym_excel,
                        rank_map=rank_map,
                        holdings=holdings_at_rebal,
                        top_n=top_n,
                        exit_rank_threshold=exit_rank_threshold,
                        exit_rank_enabled=exit_rank_enabled,
                    ),
                    "pl_pct": self._pl_pct(entry, exit_px),
                }
            )
            self._entry_prices.pop(sym_excel, None)

        rows.extend(rebal_9ema_exits)
        rows.extend(rebal_sl_exits)
        rows.extend(intraweek_exits)

        for sym_excel in end_holdings:
            yh = _get_symbol_map().get(sym_excel)
            rebal_px = self._adj_price(self._etf_adj, yh, rebal_date) if yh else None
            mark_px = self._adj_price(self._etf_adj, yh, next_date) if yh else None
            entry_orig = self._entry_prices.get(sym_excel, rebal_px)
            status = "New" if sym_excel not in prev_set else "Held"
            rank_pos = rank_map.get(sym_excel)
            rows.append(
                {
                    "ticker": sym_excel,
                    "status": status,
                    "entry": entry_orig,
                    "exit": mark_px,
                    "exit_date": None,
                    "exit_reason": (
                        f"Open (rank {rank_pos})" if rank_pos is not None else "Open"
                    ),
                    "pl_pct": self._pl_pct(entry_orig, mark_px),
                    "period_pl_pct": self._pl_pct(rebal_px, mark_px),
                }
            )

        return rows

    def load_data(self) -> None:
        cfg = self.config
        tickers = _etf_tickers()
        _get_symbol_map()
        self._log(f"Downloading {len(tickers)} ETFs …")
        self._etf_adj, self._etf_vol, self._etf_close = _download_adj_vol_close(
            tickers, cfg.backtest_start, cfg.backtest_end
        )
        self._log(f"  {len(self._etf_adj)} ETFs loaded")

        self._log(f"Downloading benchmark {cfg.benchmark_ticker} …")
        bench_adj, _, _ = _download_adj_vol_close(
            [cfg.benchmark_ticker], cfg.backtest_start, cfg.backtest_end
        )
        if cfg.benchmark_ticker not in bench_adj:
            raise RuntimeError(f"Could not download benchmark {cfg.benchmark_ticker}")
        self._bench_adj = bench_adj[cfg.benchmark_ticker]

        self._rebal_dates = list(
            _rebalance_dates(
                self._bench_adj,
                cfg.backtest_start,
                cfg.backtest_end,
                cfg.rebalance_period,
            )
        )
        if len(self._rebal_dates) < 2:
            raise RuntimeError("Need at least two rebalance dates in the backtest window")

        self.reset_run()
        self._loaded = True
        self._log(f"  {len(self._rebal_dates)} rebalance dates; {self.total_periods} periods")

    def step_period(self) -> dict | None:
        if not self._loaded or self.finished:
            return None

        cfg = self.config
        i = self._period_idx
        rebal_date = self._rebal_dates[i]
        next_date = self._rebal_dates[i + 1]

        regime = classify_ema_regime(
            self._bench_adj.loc[:rebal_date], BENCH_EMA_FAST, BENCH_EMA_SLOW
        )
        go_cash = cfg.use_regime_filter and regime == "Trend_Down"

        if go_cash:
            ranked_df = pd.DataFrame()
            holdings = []
        else:
            ranked_df = rank_at_date(
                cfg.strategy_key,
                self._etf_adj,
                self._etf_vol,
                self._bench_adj,
                rebal_date,
                proximity_of_52w_high=cfg.proximity_of_52w_high,
            )
            holdings = _select_holdings(
                ranked_df,
                self._prev_holdings,
                cfg.portfolio_size,
                cfg.exit_rank_threshold,
                exit_rank_enabled=cfg.exit_rank_enabled,
                entry_above_9ema=cfg.entry_above_9ema,
                daily_close=self._daily_close_excel() if cfg.entry_above_9ema else None,
                as_of=rebal_date if cfg.entry_above_9ema else None,
            )

        self._ensure_entry_prices(holdings, rebal_date)

        rebal_sl_exits: list[dict[str, Any]] = []
        if cfg.exit_stop_loss and holdings:
            kept, dropped = split_holdings_at_stop_loss(
                holdings,
                self._weekly_adj_excel(),
                self._entry_prices,
                rebal_date,
                float(cfg.stop_loss_pct),
            )
            rebal_sl_exits = self._rebal_stop_loss_exit_rows(dropped, rebal_date)
            holdings = kept

        rebal_9ema_exits: list[dict[str, Any]] = []
        if cfg.exit_below_9ema and holdings:
            kept, dropped = split_holdings_below_9ema(
                holdings, self._daily_close_excel(), rebal_date
            )
            rebal_9ema_exits = self._rebal_9ema_exit_rows(dropped, rebal_date)
            holdings = kept
        port_ret, end_holdings, intraweek_exits = self._simulate_period_returns(
            holdings, rebal_date, next_date
        )
        position_rows = self._build_period_position_rows(
            rebal_date=rebal_date,
            next_date=next_date,
            prev_holdings=list(self._prev_holdings),
            holdings_at_rebal=holdings,
            end_holdings=end_holdings,
            rebal_9ema_exits=rebal_9ema_exits,
            intraweek_exits=intraweek_exits,
            rebal_sl_exits=rebal_sl_exits,
            ranked_df=ranked_df,
            top_n=cfg.portfolio_size,
            exit_rank_threshold=cfg.exit_rank_threshold,
            exit_rank_enabled=cfg.exit_rank_enabled,
        )

        b_from = self._bench_adj.loc[:rebal_date]
        b_to = self._bench_adj.loc[:next_date]
        bench_ret = (
            (float(b_to.iloc[-1]) / float(b_from.iloc[-1]) - 1)
            if len(b_from) > 0 and len(b_to) > 0
            else 0.0
        )

        turnover = 0.0
        new_entries_count = 0
        if self._prev_holdings:
            old_set = set(self._prev_holdings)
            new_set = set(end_holdings)
            new_entries_count = len(new_set - old_set)
            changed = len(old_set.symmetric_difference(new_set))
            turnover = changed / max(len(old_set | new_set), 1)
        else:
            new_entries_count = len(end_holdings)

        self._portfolio_value *= 1 + port_ret

        record = {
            "Period": i + 1,
            "Rebal_Date": rebal_date,
            "End_Date": next_date,
            "Holdings": ", ".join(end_holdings) if end_holdings else "CASH",
            "Open_Tickers": list(end_holdings),
            "Num_Holdings": len(end_holdings),
            "Universe_Ranked": len(ranked_df),
            "Regime": regime,
            "Port_Return": port_ret,
            "Bench_Return": bench_ret,
            "Excess_Return": port_ret - bench_ret,
            "Turnover": turnover,
            "New_Entries": new_entries_count,
            "EMA_9_Exits_Rebal": len(rebal_9ema_exits),
            "Stop_Loss_Exits_Rebal": len(rebal_sl_exits),
            "EMA_9_Exits_Midweek": sum(
                1
                for row in intraweek_exits
                if "9 EMA (mid-period)" in str(row.get("exit_reason", ""))
            ),
            "Stop_Loss_Exits": len(rebal_sl_exits)
            + sum(
                1
                for row in intraweek_exits
                if str(row.get("exit_reason", "")).startswith("Stop loss")
            ),
            "Portfolio_Value": self._portfolio_value,
            "Strategy": STRATEGY_LABELS.get(cfg.strategy_key, cfg.strategy_key),
            "Rebalance": REBALANCE_LABELS.get(cfg.rebalance_period, cfg.rebalance_period),
            "Position_Rows": position_rows,
        }
        self._records.append(record)
        self._prev_holdings = end_holdings
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
    freq_key = str(df.attrs.get("rebalance_period", "weekly"))
    periods_per_year = PERIODS_PER_YEAR.get(freq_key, 52)

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    bench_total = df["Bench_Value"].iloc[-1] / capital - 1
    years = n_periods / periods_per_year if periods_per_year > 0 else 0.0

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    bench_cum = (1 + pd.Series(bench_rets)).cumprod()
    bench_max_dd = float((bench_cum / bench_cum.cummax() - 1).min())

    ann_vol = np.std(port_rets, ddof=1) * np.sqrt(periods_per_year) if n_periods > 1 else 0.0
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    downside = port_rets[port_rets < 0]
    downside_vol = (
        np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    )
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    win_rate = float(np.mean(port_rets > 0) * 100)
    avg_period = float(np.mean(port_rets) * 100)
    avg_turnover = float(df["Turnover"].mean() * 100)
    total_trades = int(df["New_Entries"].sum()) if "New_Entries" in df.columns else 0
    total_sl_exits = (
        int(df["Stop_Loss_Exits"].sum())
        if df.attrs.get("exit_stop_loss") and "Stop_Loss_Exits" in df.columns
        else 0
    )

    alpha = cagr - bench_cagr
    excess = pd.Series(port_rets) - pd.Series(bench_rets)
    te = float(excess.std(ddof=1)) * np.sqrt(periods_per_year) if n_periods > 1 else 0.0
    info_ratio = alpha / te if te > 0 else 0.0

    strategy_key = str(df.attrs.get("strategy_key", ""))
    benchmark = str(df.attrs.get("benchmark_ticker", BENCHMARK_TICKER))

    return {
        "Strategy": STRATEGY_LABELS.get(strategy_key, strategy_key),
        "Period": (
            f"{df['Rebal_Date'].iloc[0].strftime('%Y-%m-%d')} to "
            f"{df['End_Date'].iloc[-1].strftime('%Y-%m-%d')}"
        ),
        "Rebalance": REBALANCE_LABELS.get(freq_key, freq_key),
        "Periods": n_periods,
        "Total return %": f"{total_ret * 100:+.2f}",
        "CAGR %": f"{cagr * 100:+.2f}",
        "Max drawdown %": f"{max_dd * 100:.2f}",
        "Final value": f"{df['Portfolio_Value'].iloc[-1]:,.0f}",
        "Total trades": total_trades,
        "Stop loss exits": total_sl_exits,
        "Benchmark": benchmark,
        "Bench total return %": f"{bench_total * 100:+.2f}",
        "Bench CAGR %": f"{bench_cagr * 100:+.2f}",
        "Bench max DD %": f"{bench_max_dd * 100:.2f}",
        "Bench final value": f"{df['Bench_Value'].iloc[-1]:,.0f}",
        "Win rate %": f"{win_rate:.1f}",
        f"Avg {freq_key} return %": f"{avg_period:+.2f}",
        "Avg turnover %": f"{avg_turnover:.1f}",
        "Sortino": f"{sortino:.2f}",
        "Calmar": f"{calmar:.2f}",
        "Sharpe": f"{sharpe:.2f}",
        "Alpha %": f"{alpha * 100:+.2f}",
        "Information ratio": f"{info_ratio:.2f}",
    }


def build_config_from_ui(
    *,
    strategy_key: str,
    backtest_start: str,
    backtest_end: str,
    rebalance_period: str = "weekly",
    portfolio_size: int | None = None,
    exit_rank_threshold: int | None = None,
    exit_rank_enabled: bool = False,
    benchmark_ticker: str | None = None,
    proximity_of_52w_high: float | None = None,
    initial_capital: float = 100_000.0,
    use_regime_filter: bool = False,
    exit_below_9ema: bool = True,
    entry_above_9ema: bool = True,
    exit_stop_loss: bool = False,
    stop_loss_pct: float = 5.0,
) -> EtfMomentumBacktestConfig:
    defaults = strategy_defaults(strategy_key)
    ps = portfolio_size if portfolio_size is not None else defaults["portfolio_size"]
    ex = exit_rank_threshold if exit_rank_threshold is not None else defaults["exit_rank_threshold"]
    return EtfMomentumBacktestConfig(
        strategy_key=strategy_key,
        backtest_start=normalize_backtest_date(backtest_start),
        backtest_end=normalize_backtest_date(backtest_end),
        rebalance_period=REBALANCE_ALIASES.get(rebalance_period, rebalance_period),
        portfolio_size=ps,
        exit_rank_threshold=max(ex, ps),
        exit_rank_enabled=exit_rank_enabled,
        benchmark_ticker=benchmark_ticker or defaults["benchmark_ticker"],
        proximity_of_52w_high=(
            proximity_of_52w_high
            if proximity_of_52w_high is not None
            else defaults["proximity_of_52w_high"]
        ),
        initial_capital=initial_capital,
        use_regime_filter=use_regime_filter,
        exit_below_9ema=exit_below_9ema,
        entry_above_9ema=entry_above_9ema,
        exit_stop_loss=exit_stop_loss,
        stop_loss_pct=float(stop_loss_pct),
    )


def run_backtest_cli(
    *,
    strategy_key: str,
    backtest_start: str,
    backtest_end: str,
    rebalance_period: str = "weekly",
    portfolio_size: int | None = None,
    exit_rank_threshold: int | None = None,
    exit_rank_enabled: bool = False,
    benchmark_ticker: str | None = None,
    proximity_of_52w_high: float | None = None,
    initial_capital: float = 100_000.0,
    use_regime_filter: bool = False,
    exit_below_9ema: bool = True,
    entry_above_9ema: bool = True,
    exit_stop_loss: bool = False,
    stop_loss_pct: float = 5.0,
) -> dict:
    cfg = build_config_from_ui(
        strategy_key=strategy_key,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        rebalance_period=rebalance_period,
        portfolio_size=portfolio_size,
        exit_rank_threshold=exit_rank_threshold,
        exit_rank_enabled=exit_rank_enabled,
        benchmark_ticker=benchmark_ticker,
        proximity_of_52w_high=proximity_of_52w_high,
        initial_capital=initial_capital,
        use_regime_filter=use_regime_filter,
        exit_below_9ema=exit_below_9ema,
        entry_above_9ema=entry_above_9ema,
        exit_stop_loss=exit_stop_loss,
        stop_loss_pct=stop_loss_pct,
    )
    engine = EtfMomentumBacktestEngine(cfg)
    engine.load_data()
    df = engine.run_all()
    if df.empty:
        return {}
    return compute_metrics(df, cfg.initial_capital)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest US ETF momentum rankers")
    parser.add_argument(
        "--strategy",
        default="momentum_us_rs_etfs",
        choices=list(STRATEGY_KEYS),
    )
    parser.add_argument("--start", default=f"{date.today().year}-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--rebalance",
        default="weekly",
        choices=["weekly", "biweekly", "bi-weekly", "monthly"],
    )
    parser.add_argument("--portfolio-size", type=int, default=None)
    parser.add_argument("--exit-rank-threshold", type=int, default=None)
    parser.add_argument(
        "--exit-rank",
        action="store_true",
        help="Enable exit-rank hysteresis (default: off)",
    )
    parser.add_argument("--benchmark-ticker", default=None)
    parser.add_argument("--52w-proximity", type=float, default=None, dest="proximity_52w")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--regime-filter", action="store_true")
    parser.add_argument(
        "--no-exit-below-9ema",
        action="store_true",
        help="Disable 9 EMA exit (default: on)",
    )
    parser.add_argument(
        "--no-entry-above-9ema",
        action="store_true",
        help="Disable 9 EMA entry filter (default: on)",
    )
    parser.add_argument(
        "--exit-stop-loss",
        action="store_true",
        help="Exit when price falls stop_loss_pct below entry (daily check)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=5.0,
        help="Stop loss %% below entry (default 5)",
    )
    args = parser.parse_args()

    end = args.end or date.today().strftime("%Y-%m-%d")
    metrics = run_backtest_cli(
        strategy_key=args.strategy,
        backtest_start=args.start,
        backtest_end=end,
        rebalance_period=args.rebalance,
        portfolio_size=args.portfolio_size,
        exit_rank_threshold=args.exit_rank_threshold,
        exit_rank_enabled=args.exit_rank,
        benchmark_ticker=args.benchmark_ticker,
        proximity_of_52w_high=args.proximity_52w,
        initial_capital=args.capital,
        use_regime_filter=args.regime_filter,
        exit_below_9ema=not args.no_exit_below_9ema,
        entry_above_9ema=not args.no_entry_above_9ema,
        exit_stop_loss=args.exit_stop_loss,
        stop_loss_pct=args.stop_loss_pct,
    )
    if not metrics:
        print("No results.")
        return 1
    for k, v in metrics.items():
        print(f"  {k:<28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
