"""
Walk-forward backtest engine for India stock momentum rankers.

Supports:
  - quality_momentum_rs_lv.py
  - quality_momentum_rs_no_lv.py
  - quality_momentum_rs.py
  - momentum_rs_lv_n500.py
  - momentum_rs_stocks.py
  - momentum_stocks.py

Equal-weight rebalance with rank-based exit hysteresis (top-N entry, hold while
universe rank <= exit threshold). Used by CLI and stock_momentum_backtest_ui.

Examples:
    python backtest/stock/backtest_stock_momentum.py --strategy quality_momentum_rs_lv
    python backtest/stock/backtest_stock_momentum.py --strategy momentum_stocks --start 2024-01-01
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.nse_bhavcopy import today_ist  # noqa: E402
from utils.output_paths import FINAL_RESULT_STOCK_DIR  # noqa: E402
from momentum.rrg_core import rrg_config_date_str  # noqa: E402
from momentum.rrg_ema_exit import first_stop_loss_exit_day  # noqa: E402

STRATEGY_KEYS = (
    "quality_momentum_rs_lv",
    "quality_momentum_rs_no_lv",
    "quality_momentum_rs",
    "momentum_rs_lv_n500",
    "momentum_rs_stocks",
    "momentum_stocks",
)

STRATEGY_LABELS: dict[str, str] = {
    "quality_momentum_rs_lv": "Quality RS + Low Vol",
    "quality_momentum_rs_no_lv": "Quality RS (no LV)",
    "quality_momentum_rs": "Quality RS (daily scan)",
    "momentum_rs_lv_n500": "N500 RS + Low Vol",
    "momentum_rs_stocks": "Nifty LM RS",
    "momentum_stocks": "BSE LM Abs Momentum",
}

STRATEGY_MODULES: dict[str, str] = {
    "quality_momentum_rs_lv": "quality_momentum_rs_lv.py",
    "quality_momentum_rs_no_lv": "quality_momentum_rs_no_lv.py",
    "quality_momentum_rs": "quality_momentum_rs.py",
    "momentum_rs_lv_n500": "momentum_rs_lv_n500.py",
    "momentum_rs_stocks": "momentum_rs_stocks.py",
    "momentum_stocks": "momentum_stocks.py",
}

DEFAULT_BENCHMARK = "^CRSLDX"
HIGH_52W_LOOKBACK = 252
EMA_SPAN = 200
ADTV_WINDOW = 20
VOL_WINDOW = 21
MIN_HISTORY = 189 + HIGH_52W_LOOKBACK + 30

REBALANCE_ALIASES = {
    "bi-weekly": "biweekly",
    "bi_weekly": "biweekly",
    "biweekly": "biweekly",
    "weekly": "weekly",
    "monthly": "monthly",
}

PERIODS_PER_YEAR = {
    "weekly": 52,
    "biweekly": 26,
    "monthly": 12,
}

REBALANCE_LABELS = {
    "weekly": "Weekly",
    "biweekly": "Biweekly",
    "monthly": "Monthly",
}


def load_strategy_module(strategy_key: str) -> Any:
    if strategy_key not in STRATEGY_MODULES:
        raise ValueError(
            f"Unknown strategy {strategy_key!r}; choose from {list(STRATEGY_KEYS)}"
        )
    path = _PROJECT_ROOT / "momentum" / "stock" / STRATEGY_MODULES[strategy_key]
    spec = importlib.util.spec_from_file_location(strategy_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strategy_defaults(strategy_key: str) -> dict[str, Any]:
    mod = load_strategy_module(strategy_key)
    portfolio_size = int(getattr(mod, "PORTFOLIO_SIZE", 30))
    exit_rank = int(getattr(mod, "EXIT_RANK_THRESHOLD", max(portfolio_size, 30)))
    if strategy_key == "momentum_stocks":
        proximity = 0.75
    else:
        proximity = float(getattr(mod, "PROXIMITY_OF_52W_HIGH", 0.70))
    benchmark = str(getattr(mod, "BENCHMARK_TICKER", DEFAULT_BENCHMARK))
    rebalance = str(getattr(mod, "REBALANCE_COMPARE_PERIOD", "biweekly"))
    return {
        "portfolio_size": portfolio_size,
        "exit_rank_threshold": exit_rank,
        "proximity_of_52w_high": proximity,
        "benchmark_ticker": benchmark,
        "rebalance_period": REBALANCE_ALIASES.get(rebalance, rebalance),
        "use_low_vol": strategy_key
        in ("quality_momentum_rs_lv", "momentum_rs_lv_n500"),
        "rank_style": "abs" if strategy_key == "momentum_stocks" else "rs_blended",
    }


def normalize_backtest_date(value: str) -> str:
    """Accept YYYY-MM-DD or DD-MM-YYYY; return YYYY-MM-DD for data loaders."""
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Backtest date is required.")
    parts = raw.split("-")
    if len(parts) == 3 and len(parts[0]) == 4 and parts[0].isdigit():
        pd.Timestamp(raw)  # validate
        return raw
    return rrg_config_date_str(raw)


def _yf_download_end_exclusive(backtest_end: str) -> str:
    """yfinance ``end`` is exclusive — include the last backtest session."""
    end_iso = normalize_backtest_date(backtest_end)
    return (pd.Timestamp(end_iso) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _adj_col(df: pd.DataFrame) -> pd.Series:
    s = df["Adj Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _vol_col(df: pd.DataFrame) -> pd.Series:
    s = df["Volume"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s.squeeze()).dropna()


def _download_adj_vol(
    yahoo_symbols: list[str], start: str, end: str
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    start_iso = normalize_backtest_date(start)
    end_iso = _yf_download_end_exclusive(end)
    extra_start = (
        pd.Timestamp(start_iso) - pd.Timedelta(days=int(MIN_HISTORY * 1.6))
    ).strftime("%Y-%m-%d")
    adj_store: dict[str, pd.Series] = {}
    vol_store: dict[str, pd.Series] = {}
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
        except Exception:
            pass
    return adj_store, vol_store


def _excel_to_yahoo(mod: Any) -> dict[str, str]:
    mapping = getattr(mod, "_EXCEL_SYMBOL_TO_YAHOO", None)
    if mapping:
        return dict(mapping)
    sym_fn = getattr(mod, "_symbol_for_excel", lambda s: s.replace(".NS", "").replace(".BO", ""))
    return {sym_fn(t["symbol"]): t["symbol"] for t in mod.tickers}


def _rank_rs_blended(
    mod: Any,
    stock_adj: dict[str, pd.Series],
    stock_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
    use_low_vol: bool,
) -> pd.DataFrame:
    lb_1m = mod.LB_1M
    lb_3m = mod.LB_3M
    lb_6m = mod.LB_6M
    lb_9m = mod.LB_9M
    min_adtv = mod.MIN_ADTV_CRORES

    bench_slice = bench_adj.loc[:as_of]
    if len(bench_slice) < lb_9m:
        return pd.DataFrame()

    summary: list[dict] = []
    for t in mod.tickers:
        sym = t["symbol"]
        if sym not in stock_adj:
            continue
        adj = stock_adj[sym].loc[:as_of]
        if len(adj) < lb_9m:
            continue

        vol_s = stock_vol.get(sym)
        if vol_s is None or len(vol_s) == 0:
            continue
        vol = vol_s.reindex(adj.index).fillna(0.0)

        daily_turnover = adj * vol
        adtv_crores = float(daily_turnover.tail(ADTV_WINDOW).mean()) / 10_000_000
        if adtv_crores < min_adtv:
            continue

        ema200 = float(adj.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])
        high_52w = float(adj.iloc[-min(HIGH_52W_LOOKBACK, len(adj)) :].max())
        last = float(adj.iloc[-1])

        if last < ema200 or last < (high_52w * proximity_of_52w_high):
            continue

        ret_1m = (adj.iloc[-1] / adj.iloc[-lb_1m] - 1) * 100
        ret_3m = (adj.iloc[-1] / adj.iloc[-lb_3m] - 1) * 100
        ret_6m = (adj.iloc[-1] / adj.iloc[-lb_6m] - 1) * 100
        ret_9m = (adj.iloc[-1] / adj.iloc[-lb_9m] - 1) * 100
        vol_score = float(adj.pct_change().tail(VOL_WINDOW).std() * 100)

        nx = bench_slice.reindex(adj.index).ffill()
        rs_3m = ret_3m - ((nx.iloc[-1] / nx.iloc[-lb_3m] - 1) * 100)
        rs_6m = ret_6m - ((nx.iloc[-1] / nx.iloc[-lb_6m] - 1) * 100)
        rs_9m = ret_9m - ((nx.iloc[-1] / nx.iloc[-lb_9m] - 1) * 100)

        row: dict[str, Any] = {
            "Symbol": mod._symbol_for_excel(sym),
            "Return_1M": ret_1m,
            "Return_3M": ret_3m,
            "Return_6M": ret_6m,
            "Return_9M": ret_9m,
            "RS_3M_vs_Bench": rs_3m,
            "RS_6M_vs_Bench": rs_6m,
            "RS_9M_vs_Bench": rs_9m,
            "Volatility_Score": vol_score,
        }
        if "marketcap" in t:
            row["Marketcap"] = t.get("marketcap", "")
        summary.append(row)

    if not summary:
        return pd.DataFrame()

    df_summary = pd.DataFrame(summary)
    if use_low_vol and hasattr(mod, "_apply_low_volatility_filter"):
        df_summary = mod._apply_low_volatility_filter(df_summary, quiet=True)
        if df_summary.empty:
            return pd.DataFrame()

    if hasattr(mod, "_apply_ranking_engine"):
        return mod._apply_ranking_engine(df_summary)

    w3, w6, w9 = mod.W_3M, mod.W_6M, mod.W_9M
    df = df_summary.copy()
    for c in ["3M", "6M", "9M"]:
        df[f"Rank_{c}"] = df[f"Return_{c}"].rank(ascending=False)
    for c in ["3M", "6M", "9M"]:
        df[f"Rank_RS_{c}"] = df[f"RS_{c}_vs_Bench"].rank(ascending=False, na_option="bottom")
    df["Abs_Momentum_Rank"] = (
        w3 * df["Rank_3M"] + w6 * df["Rank_6M"] + w9 * df["Rank_9M"]
    ).rank()
    df["Relative_Strength_Rank"] = (
        w3 * df["Rank_RS_3M"] + w6 * df["Rank_RS_6M"] + w9 * df["Rank_RS_9M"]
    ).rank()
    df["Blended_Rank"] = (df["Abs_Momentum_Rank"] + df["Relative_Strength_Rank"]) / 2
    out = df.sort_values("Blended_Rank").reset_index(drop=True)
    out["Rank_Position"] = np.arange(1, len(out) + 1)
    return out


def _rank_abs_momentum(
    mod: Any,
    stock_adj: dict[str, pd.Series],
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
) -> pd.DataFrame:
    summary: list[dict] = []
    industry_by_symbol = {t["symbol"]: t.get("industry", "") for t in mod.tickers}

    for t in mod.tickers:
        sym = t["symbol"]
        if sym not in stock_adj:
            continue
        adj = stock_adj[sym].loc[:as_of]
        n = len(adj)
        if n < 21:
            continue

        ema200 = adj.ewm(span=EMA_SPAN, adjust=False).mean()
        if n >= 252:
            one_year_return = (adj.iloc[-1] / adj.iloc[-252] - 1) * 100
        else:
            one_year_return = float("nan")

        high_52w = adj.iloc[-min(HIGH_52W_LOOKBACK, n) :].max()
        within_high = adj.iloc[-1] >= high_52w * proximity_of_52w_high

        six_month_data = adj.iloc[-min(126, n) :]
        up_days_pct = (six_month_data.pct_change() > 0).sum() / len(six_month_data) * 100

        if (
            adj.iloc[-1] < ema200.iloc[-1]
            or one_year_return < 6.5
            or not within_high
            or up_days_pct <= 45
        ):
            continue

        ret_9m = (adj.iloc[-1] / adj.iloc[-189] - 1) * 100 if n >= 189 else float("nan")
        ret_6m = (adj.iloc[-1] / adj.iloc[-126] - 1) * 100 if n >= 126 else float("nan")
        ret_3m = (adj.iloc[-1] / adj.iloc[-63] - 1) * 100 if n >= 63 else float("nan")
        ret_1m = (adj.iloc[-1] / adj.iloc[-21] - 1) * 100 if n >= 21 else float("nan")

        summary.append(
            {
                "Symbol": mod._symbol_for_excel(sym),
                "Industry": industry_by_symbol.get(sym, ""),
                "Return_9M": ret_9m,
                "Return_6M": ret_6m,
                "Return_3M": ret_3m,
                "Return_1M": ret_1m,
            }
        )

    if not summary:
        return pd.DataFrame()

    df = pd.DataFrame(summary)
    for c in ["9M", "6M", "3M"]:
        df[f"Rank_{c}"] = df[f"Return_{c}"].rank(ascending=False)
    df["Final_Rank"] = (
        0.50 * df["Rank_3M"] + 0.30 * df["Rank_6M"] + 0.20 * df["Rank_9M"]
    )
    out = df.sort_values("Final_Rank").reset_index(drop=True)
    out["Rank_Position"] = np.arange(1, len(out) + 1)
    return out


def rank_at_date(
    mod: Any,
    strategy_key: str,
    stock_adj: dict[str, pd.Series],
    stock_vol: dict[str, pd.Series],
    bench_adj: pd.Series,
    as_of: pd.Timestamp,
    *,
    proximity_of_52w_high: float,
    use_low_vol: bool,
) -> pd.DataFrame:
    defaults = strategy_defaults(strategy_key)
    if defaults["rank_style"] == "abs":
        return _rank_abs_momentum(
            mod, stock_adj, as_of, proximity_of_52w_high=proximity_of_52w_high
        )
    return _rank_rs_blended(
        mod,
        stock_adj,
        stock_vol,
        bench_adj,
        as_of,
        proximity_of_52w_high=proximity_of_52w_high,
        use_low_vol=use_low_vol,
    )


def _select_holdings(
    ranked_df: pd.DataFrame,
    prev_holdings: list[str],
    top_n: int,
    worst_rank_held: int,
    *,
    exit_rank_enabled: bool = True,
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
    new_entries = [
        sym
        for sym in ranked_df["Symbol"]
        if sym not in retained_set and rank_map[sym] <= top_n
    ]
    retained.sort(key=lambda s: rank_map[s])
    return (retained + new_entries)[:top_n]


def _exit_reason(
    sym: str,
    *,
    rank_map: dict[str, int],
    holdings: list[str],
    top_n: int,
    exit_rank_threshold: int,
    exit_rank_enabled: bool = True,
) -> str:
    if sym not in rank_map:
        return "Not in ranked universe (failed screen)"
    rank_pos = rank_map[sym]
    if exit_rank_enabled and rank_pos > exit_rank_threshold:
        return f"Rank {rank_pos} > exit rank {exit_rank_threshold}"
    if sym not in holdings:
        return f"Displaced — rank {rank_pos} (portfolio top {top_n} filled)"
    return "Exit"


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
        raise ValueError(f"Unknown rebalance period {period!r}; use weekly, biweekly, or monthly")

    out = pd.DatetimeIndex(dates)
    return out[out >= all_dates[0]]


@dataclass
class StockMomentumBacktestConfig:
    strategy_key: str
    backtest_start: str
    backtest_end: str
    rebalance_period: str = "biweekly"
    portfolio_size: int = 20
    exit_rank_threshold: int = 30
    exit_rank_enabled: bool = True
    benchmark_ticker: str = DEFAULT_BENCHMARK
    proximity_of_52w_high: float = 0.70
    initial_capital: float = 100_000.0
    use_low_vol: bool | None = None
    exit_stop_loss: bool = False
    stop_loss_pct: float = 10.0


@dataclass
class StockMomentumBacktestEngine:
    config: StockMomentumBacktestConfig
    progress_cb: Callable[[str], None] | None = None
    _mod: Any = field(default=None, repr=False)
    _excel_to_yahoo: dict[str, str] = field(default_factory=dict)
    _stock_adj: dict[str, pd.Series] = field(default_factory=dict)
    _stock_vol: dict[str, pd.Series] = field(default_factory=dict)
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
        defaults = strategy_defaults(self.config.strategy_key)
        if self.config.use_low_vol is None:
            object.__setattr__(self.config, "use_low_vol", defaults["use_low_vol"])

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
    def total_weeks(self) -> int:
        return self.total_periods

    @property
    def current_period(self) -> int:
        return self._period_idx

    @property
    def current_week(self) -> int:
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
        df.attrs["exit_stop_loss"] = self.config.exit_stop_loss
        df.attrs["stop_loss_pct"] = self.config.stop_loss_pct
        df.attrs["exit_rank_enabled"] = self.config.exit_rank_enabled
        return df

    def reset_run(self) -> None:
        self._period_idx = 0
        self._records = []
        self._portfolio_value = self.config.initial_capital
        self._prev_holdings = []
        self._entry_prices = {}

    @staticmethod
    def _adj_price(stock_adj: dict[str, pd.Series], yahoo_sym: str, as_of: pd.Timestamp) -> float | None:
        series = stock_adj.get(yahoo_sym)
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
            yh = self._excel_to_yahoo.get(sym_excel)
            if not yh:
                continue
            px = self._adj_price(self._stock_adj, yh, rebal_date)
            if px is not None:
                self._entry_prices[sym_excel] = px

    def _simulate_period_returns(
        self,
        holdings: list[str],
        rebal_date: pd.Timestamp,
        next_date: pd.Timestamp,
    ) -> tuple[float, list[str], list[dict[str, Any]]]:
        """Equal-weight period return with optional intraperiod stop-loss from entry."""
        cfg = self.config
        if not holdings:
            return 0.0, [], []

        period_rets: list[float] = []
        end_holdings: list[str] = []
        stop_exits: list[dict[str, Any]] = []

        for sym_excel in holdings:
            yh = self._excel_to_yahoo.get(sym_excel)
            if not yh or yh not in self._stock_adj:
                period_rets.append(0.0)
                end_holdings.append(sym_excel)
                continue

            series = self._stock_adj[yh]
            rebal_slice = series.loc[:rebal_date]
            if len(rebal_slice) == 0:
                period_rets.append(0.0)
                end_holdings.append(sym_excel)
                continue

            p_rebal = float(rebal_slice.iloc[-1])
            entry_for_stop = self._entry_prices.get(sym_excel, p_rebal)
            sl_day: pd.Timestamp | None = None
            if cfg.exit_stop_loss and cfg.stop_loss_pct > 0:
                sl_day = first_stop_loss_exit_day(
                    series,
                    float(entry_for_stop),
                    rebal_date,
                    next_date,
                    float(cfg.stop_loss_pct),
                )

            if sl_day is not None:
                exit_px = self._adj_price(self._stock_adj, yh, sl_day)
                period_rets.append(
                    self._pl_pct(p_rebal, exit_px) / 100.0 if exit_px is not None else 0.0
                )
                stop_exits.append(
                    {
                        "ticker": sym_excel,
                        "status": "Exit",
                        "entry": entry_for_stop,
                        "exit": exit_px,
                        "exit_date": sl_day,
                        "exit_reason": (
                            f"Stop loss ({cfg.stop_loss_pct:g}% from entry "
                            f"{entry_for_stop:,.2f})"
                        ),
                        "pl_pct": self._pl_pct(entry_for_stop, exit_px),
                    }
                )
                self._entry_prices.pop(sym_excel, None)
                continue

            mark_px = self._adj_price(self._stock_adj, yh, next_date)
            period_rets.append(
                self._pl_pct(p_rebal, mark_px) / 100.0 if mark_px is not None else 0.0
            )
            end_holdings.append(sym_excel)

        port_ret = float(np.mean(period_rets)) if period_rets else 0.0
        return port_ret, end_holdings, stop_exits

    def _build_period_position_rows(
        self,
        *,
        rebal_date: pd.Timestamp,
        next_date: pd.Timestamp,
        prev_holdings: list[str],
        holdings_at_rebal: list[str],
        end_holdings: list[str],
        stop_loss_exits: list[dict[str, Any]],
        ranked_df: pd.DataFrame,
        top_n: int,
        exit_rank_threshold: int,
        exit_rank_enabled: bool,
    ) -> list[dict[str, Any]]:
        prev_set = set(prev_holdings)
        rebal_set = set(holdings_at_rebal)
        rank_map = (
            dict(zip(ranked_df["Symbol"], ranked_df["Rank_Position"]))
            if not ranked_df.empty
            else {}
        )
        rows: list[dict[str, Any]] = []

        for sym_excel in sorted(prev_set - rebal_set):
            yh = self._excel_to_yahoo.get(sym_excel)
            entry = self._entry_prices.get(sym_excel)
            exit_px = self._adj_price(self._stock_adj, yh, rebal_date) if yh else None
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

        rows.extend(stop_loss_exits)

        for sym_excel in end_holdings:
            yh = self._excel_to_yahoo.get(sym_excel)
            rebal_px = self._adj_price(self._stock_adj, yh, rebal_date) if yh else None
            mark_px = self._adj_price(self._stock_adj, yh, next_date) if yh else None
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
                        f"Open (rank {rank_pos})"
                        if rank_pos is not None
                        else "Open"
                    ),
                    "pl_pct": self._pl_pct(entry_orig, mark_px),
                    "period_pl_pct": self._pl_pct(rebal_px, mark_px),
                }
            )

        return rows

    def load_data(self) -> None:
        cfg = self.config
        self._mod = load_strategy_module(cfg.strategy_key)
        self._excel_to_yahoo = _excel_to_yahoo(self._mod)

        yahoo_symbols = [t["symbol"] for t in self._mod.tickers]
        self._log(f"Downloading {len(yahoo_symbols)} symbols …")
        self._stock_adj, self._stock_vol = _download_adj_vol(
            yahoo_symbols, cfg.backtest_start, cfg.backtest_end
        )
        self._log(f"  {len(self._stock_adj)} symbols loaded")

        self._log(f"Downloading benchmark {cfg.benchmark_ticker} …")
        bench_adj, _ = _download_adj_vol(
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

        ranked_df = rank_at_date(
            self._mod,
            cfg.strategy_key,
            self._stock_adj,
            self._stock_vol,
            self._bench_adj,
            rebal_date,
            proximity_of_52w_high=cfg.proximity_of_52w_high,
            use_low_vol=bool(cfg.use_low_vol),
        )
        holdings = _select_holdings(
            ranked_df,
            self._prev_holdings,
            cfg.portfolio_size,
            cfg.exit_rank_threshold,
            exit_rank_enabled=cfg.exit_rank_enabled,
        )
        self._ensure_entry_prices(holdings, rebal_date)
        port_ret, end_holdings, stop_loss_exits = self._simulate_period_returns(
            holdings, rebal_date, next_date
        )
        position_rows = self._build_period_position_rows(
            rebal_date=rebal_date,
            next_date=next_date,
            prev_holdings=list(self._prev_holdings),
            holdings_at_rebal=holdings,
            end_holdings=end_holdings,
            stop_loss_exits=stop_loss_exits,
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
            "Week": i + 1,
            "Rebal_Date": rebal_date,
            "End_Date": next_date,
            "Holdings": ", ".join(end_holdings) if end_holdings else "CASH",
            "Open_Tickers": list(end_holdings),
            "Num_Holdings": len(end_holdings),
            "Universe_Ranked": len(ranked_df),
            "Port_Return": port_ret,
            "Bench_Return": bench_ret,
            "Excess_Return": port_ret - bench_ret,
            "Turnover": turnover,
            "New_Entries": new_entries_count,
            "Stop_Loss_Exits": len(stop_loss_exits),
            "Portfolio_Value": self._portfolio_value,
            "Strategy": STRATEGY_LABELS.get(cfg.strategy_key, cfg.strategy_key),
            "Rebalance": REBALANCE_LABELS.get(cfg.rebalance_period, cfg.rebalance_period),
            "Position_Rows": position_rows,
        }
        self._records.append(record)
        self._prev_holdings = end_holdings
        self._period_idx += 1
        return record

    def step_week(self) -> dict | None:
        return self.step_period()

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
    freq_key = str(df.attrs.get("rebalance_period", "biweekly"))
    periods_per_year = PERIODS_PER_YEAR.get(freq_key, 26)

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

    alpha = cagr - bench_cagr
    excess = pd.Series(port_rets) - pd.Series(bench_rets)
    te = float(excess.std(ddof=1)) * np.sqrt(periods_per_year) if n_periods > 1 else 0.0
    info_ratio = alpha / te if te > 0 else 0.0

    strategy_key = str(df.attrs.get("strategy_key", ""))
    portfolio_size = int(df.attrs.get("portfolio_size", 20))

    return {
        "Strategy": STRATEGY_LABELS.get(strategy_key, strategy_key),
        "Period": (
            f"{df['Rebal_Date'].iloc[0].strftime('%Y-%m-%d')} to "
            f"{df['End_Date'].iloc[-1].strftime('%Y-%m-%d')}"
        ),
        "Rebalance": REBALANCE_LABELS.get(freq_key, freq_key),
        "Periods": n_periods,
        "Portfolio size": portfolio_size,
        "Total return %": f"{total_ret * 100:+.2f}",
        "CAGR %": f"{cagr * 100:+.2f}",
        "Max drawdown %": f"{max_dd * 100:.2f}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Calmar": f"{calmar:.2f}",
        "Win rate %": f"{win_rate:.1f}",
        f"Avg {freq_key} return %": f"{avg_period:+.2f}",
        "Avg turnover %": f"{avg_turnover:.1f}",
        "Total entries": total_trades,
        "Bench total return %": f"{bench_total * 100:+.2f}",
        "Bench CAGR %": f"{bench_cagr * 100:+.2f}",
        "Bench max DD %": f"{bench_max_dd * 100:.2f}",
        "Alpha %": f"{alpha * 100:+.2f}",
        "Information ratio": f"{info_ratio:.2f}",
        "Final value": f"{df['Portfolio_Value'].iloc[-1]:,.0f}",
        "Bench final value": f"{df['Bench_Value'].iloc[-1]:,.0f}",
    }


def build_config_from_ui(
    *,
    strategy_key: str,
    backtest_start: str,
    backtest_end: str,
    rebalance_period: str = "biweekly",
    portfolio_size: int | None = None,
    exit_rank_threshold: int | None = None,
    exit_rank_enabled: bool = True,
    benchmark_ticker: str | None = None,
    proximity_of_52w_high: float | None = None,
    initial_capital: float = 100_000.0,
    exit_stop_loss: bool = False,
    stop_loss_pct: float = 10.0,
) -> StockMomentumBacktestConfig:
    defaults = strategy_defaults(strategy_key)
    ps = portfolio_size if portfolio_size is not None else defaults["portfolio_size"]
    ex = exit_rank_threshold if exit_rank_threshold is not None else defaults["exit_rank_threshold"]
    return StockMomentumBacktestConfig(
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
        exit_stop_loss=exit_stop_loss,
        stop_loss_pct=float(stop_loss_pct),
    )


def run_backtest_cli(
    *,
    strategy_key: str,
    backtest_start: str,
    backtest_end: str,
    rebalance_period: str = "biweekly",
    portfolio_size: int | None = None,
    exit_rank_threshold: int | None = None,
    exit_rank_enabled: bool = True,
    benchmark_ticker: str | None = None,
    proximity_of_52w_high: float | None = None,
    initial_capital: float = 100_000.0,
    exit_stop_loss: bool = False,
    stop_loss_pct: float = 10.0,
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
        exit_stop_loss=exit_stop_loss,
        stop_loss_pct=stop_loss_pct,
    )
    engine = StockMomentumBacktestEngine(cfg)
    engine.load_data()
    df = engine.run_all()
    if df.empty:
        return {}
    return compute_metrics(df, cfg.initial_capital)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest stock momentum rankers")
    parser.add_argument(
        "--strategy",
        default="quality_momentum_rs_lv",
        choices=list(STRATEGY_KEYS),
        help="Strategy script to backtest",
    )
    parser.add_argument("--start", default=f"{today_ist().year}-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--rebalance",
        default="biweekly",
        choices=["weekly", "biweekly", "bi-weekly", "monthly"],
    )
    parser.add_argument("--portfolio-size", type=int, default=None)
    parser.add_argument("--exit-rank-threshold", type=int, default=None)
    parser.add_argument(
        "--no-exit-rank",
        action="store_true",
        help="Disable exit-rank hold band (keep prior names while they pass screen)",
    )
    parser.add_argument("--benchmark-ticker", default=None)
    parser.add_argument("--52w-proximity", type=float, default=None, dest="proximity_52w")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument(
        "--exit-stop-loss",
        action="store_true",
        help="Exit when price falls stop_loss_pct below entry (daily check)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=10.0,
        help="Stop loss %% below entry (default 10)",
    )
    args = parser.parse_args()

    end = args.end or today_ist().strftime("%Y-%m-%d")
    metrics = run_backtest_cli(
        strategy_key=args.strategy,
        backtest_start=args.start,
        backtest_end=end,
        rebalance_period=args.rebalance,
        portfolio_size=args.portfolio_size,
        exit_rank_threshold=args.exit_rank_threshold,
        exit_rank_enabled=not args.no_exit_rank,
        benchmark_ticker=args.benchmark_ticker,
        proximity_of_52w_high=args.proximity_52w,
        initial_capital=args.capital,
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
