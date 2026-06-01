"""
Walk-forward backtest for India NSE ETF RRG swing recommendations.

Replays the same weekly ranking + ``recommend_india_etfs`` pipeline as the live
RRG screen: tail-window rank, Leading/Improving quadrant, rank delta, vol scoring.

RRG inputs use NSE index EOD (index rows) or Yahoo ETF weekly (ETF rows).
P&L uses equal-weight ref ETF weekly closes (Yahoo .NS).

Examples:
    python backtest/etf/backtest_rrg_india.py --start 2024-01-01 --end 2025-03-31
    python backtest/etf/backtest_rrg_india.py --start 2023-06-01 --end 2024-12-31 --top-n 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.etf_rrg_universe import (  # noqa: E402
    RRG_BENCHMARK_NSE,
    RRG_ETF_ROW_IDS,
    RRG_ROWS,
    row_display_label,
)
from momentum.etf.india_rrg_pick_strategies import (  # noqa: E402
    PICK_STRATEGIES,
    IndiaPickContext,
    pick_india_portfolio,
    pick_strategy_label,
)
from momentum.etf.india_rrg_recommendations import (  # noqa: E402
    load_india_etf_vol_pct,
)
from momentum.rrg_core import compute_rrg_indicators, rrg_effective_window, rrg_warmup_weeks  # noqa: E402
from momentum.rrg_ema_exit import (  # noqa: E402
    filter_holdings_below_9ema,
    simulate_week_with_9ema_exits,
)
from momentum.rrg_ranking import (  # noqa: E402
    build_rank_delta_by_row,
    rank_by_tail_change,
    ranked_row_indices,
    series_at,
    tail_change_pct,
)
from utils.nse_bhavcopy import (  # noqa: E402
    fetch_index_close_histories,
    load_nse_cm_histories_range,
    today_ist,
)
from utils.output_paths import FINAL_RESULT_ETF_DIR  # noqa: E402

BACKTEST_OUT_DIR = FINAL_RESULT_ETF_DIR / "backtest_rrg_india"
STRATEGY_TAG = "india_rrg_swing"


@dataclass
class IndiaRrgBacktestConfig:
    backtest_start: str
    backtest_end: str
    top_n: int = 7
    tail: int = 1
    rrg_window: int = 10
    initial_capital: float = 100_000.0
    vol_days: int = 63
    pick_strategy: str = "recommend"
    hold_until_rank_exit: bool = False
    max_hold_rank: int = 10
    exit_below_9ema: bool = False


@dataclass
class IndiaRrgBacktestEngine:
    """Walk-forward India RRG backtest; supports step-by-step or run-all."""

    config: IndiaRrgBacktestConfig
    progress_cb: Callable[[str], None] | None = None
    _row_ids: list[str] = field(default_factory=list)
    _ref_labels: list[str] = field(default_factory=list)
    _display_labels: list[str] = field(default_factory=list)
    _row_kinds: list[str] = field(default_factory=list)
    _row_price_weekly: dict[str, pd.Series] = field(default_factory=dict)
    _ref_etf_weekly: dict[str, pd.Series] = field(default_factory=dict)
    _ref_etf_daily: dict[str, pd.Series] = field(default_factory=dict)
    _bench_weekly: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    _weekly_index: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    _rebal_dates: list[pd.Timestamp] = field(default_factory=list)
    _rsr_series: list[pd.Series | None] = field(default_factory=list)
    _rsm_series: list[pd.Series | None] = field(default_factory=list)
    _vol_by_ref: dict[str, float] = field(default_factory=dict)
    _records: list[dict] = field(default_factory=list)
    _week_idx: int = 0
    _portfolio_value: float = 0.0
    _prev_holdings: list[str] = field(default_factory=list)
    _loaded: bool = False

    def __post_init__(self) -> None:
        self._portfolio_value = self.config.initial_capital
        for row in RRG_ROWS:
            self._row_ids.append(row.row_id)
            self._ref_labels.append(row.ref_etf)
            self._display_labels.append(row_display_label(row.row_id))
            self._row_kinds.append(row.kind)

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
    def current_week(self) -> int:
        return self._week_idx

    @property
    def finished(self) -> bool:
        return self._week_idx >= len(self._rebal_dates)

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        if "Port_Return" in df.columns and len(df):
            df["Bench_Value"] = self.config.initial_capital * (
                (1 + df["Bench_Return"]).cumprod()
            )
        return df

    @property
    def rebal_dates(self) -> list[pd.Timestamp]:
        return list(self._rebal_dates)

    def reset_run(self) -> None:
        self._week_idx = 0
        self._records = []
        self._portfolio_value = self.config.initial_capital
        self._prev_holdings = []

    @staticmethod
    def _collapse_tail_window_dates(
        candidates: list[pd.Timestamp],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """One tail window (start Fri → end Fri): single as-of decision at end Fri."""
        if len(candidates) == 2:
            d0, d1 = candidates[0], candidates[1]
            if (d1 - d0).days == 7:
                if (
                    d0.normalize() == start_ts.normalize()
                    and d1.normalize() == end_ts.normalize()
                ):
                    return [d1]
        return candidates

    def load_data(self) -> None:
        cfg = self.config
        start_ts = pd.Timestamp(cfg.backtest_start)
        end_ts = pd.Timestamp(cfg.backtest_end)
        if start_ts >= end_ts:
            raise ValueError("Backtest start must be before end")

        warmup_weeks = rrg_warmup_weeks(cfg.rrg_window) + cfg.tail + 2
        dl_start = (start_ts - pd.Timedelta(days=warmup_weeks * 7 + 21)).date()
        dl_end = max(end_ts.date(), today_ist())

        self._log(
            f"Loading NSE index EOD {dl_start} .. {dl_end} "
            f"({len(self._row_ids)} RRG rows + benchmark)..."
        )

        index_names = list(
            dict.fromkeys(
                [
                    *[
                        rid
                        for rid, kind in zip(self._row_ids, self._row_kinds)
                        if kind == "index"
                    ],
                    RRG_BENCHMARK_NSE,
                ]
            )
        )
        daily_index = fetch_index_close_histories(
            index_names, dl_start, dl_end, quiet=True
        )
        for name in index_names:
            daily = daily_index.get(name, pd.Series(dtype=float))
            if len(daily):
                weekly = daily.resample("W-FRI").last().dropna()
                self._row_price_weekly[name] = weekly

        self._bench_weekly = self._row_price_weekly.get(
            RRG_BENCHMARK_NSE, pd.Series(dtype=float)
        )
        if self._bench_weekly.empty:
            raise RuntimeError(f"Could not load benchmark {RRG_BENCHMARK_NSE}")

        etf_row_ids = [rid for rid, kind in zip(self._row_ids, self._row_kinds) if kind == "etf"]
        ref_etfs = list(
            dict.fromkeys(
                [
                    (ref or rid).upper().replace(".NS", "")
                    for ref, rid in zip(self._ref_labels, self._row_ids)
                ]
            )
        )
        etf_bhavcopy_syms = list(
            dict.fromkeys([*RRG_ETF_ROW_IDS, *ref_etfs])
        )

        vol_start = dl_start - timedelta(days=cfg.vol_days + 30)
        self._log(
            f"Loading NSE ETF bhavcopy {vol_start} .. {dl_end} "
            f"({len(etf_bhavcopy_syms)} symbols, same source as live RRG)..."
        )
        etf_daily_batch = load_nse_cm_histories_range(
            etf_bhavcopy_syms,
            vol_start,
            dl_end,
            min_points=5,
            quiet=True,
            asset_label="ETF symbol",
            freq="day",
        )
        for bare in etf_bhavcopy_syms:
            daily = etf_daily_batch.get(bare, pd.Series(dtype=float))
            if len(daily):
                daily = daily.sort_index()
                self._ref_etf_daily[bare] = daily
                self._ref_etf_weekly[bare] = daily.resample("W-FRI").last().dropna()

        for sym in etf_row_ids:
            bare = sym.upper().replace(".NS", "")
            if bare in self._ref_etf_weekly:
                self._row_price_weekly[sym] = self._ref_etf_weekly[bare]

        for j, row_id in enumerate(self._row_ids):
            if row_id not in self._row_price_weekly:
                bare = (self._ref_labels[j] or row_id).upper().replace(".NS", "")
                if bare in self._ref_etf_weekly:
                    self._row_price_weekly[row_id] = self._ref_etf_weekly[bare]

        min_history = rrg_effective_window(cfg.rrg_window, "week")
        active_row_ids: list[str] = []
        active_ref_labels: list[str] = []
        active_display_labels: list[str] = []
        active_kinds: list[str] = []
        for j, row_id in enumerate(self._row_ids):
            prices = self._row_price_weekly.get(row_id, pd.Series(dtype=float))
            if prices.notna().sum() > min_history:
                active_row_ids.append(row_id)
                active_ref_labels.append(self._ref_labels[j])
                active_display_labels.append(self._display_labels[j])
                active_kinds.append(self._row_kinds[j])
        skipped = len(self._row_ids) - len(active_row_ids)
        if skipped:
            self._log(f"Skipping {skipped} rows with insufficient weekly history.")
        if not active_row_ids:
            raise RuntimeError("No RRG rows with enough history for backtest.")
        self._row_ids = active_row_ids
        self._ref_labels = active_ref_labels
        self._display_labels = active_display_labels
        self._row_kinds = active_kinds

        self._weekly_index = self._bench_weekly.index.sort_values()

        self._rsr_series = []
        self._rsm_series = []
        bench = self._bench_weekly
        for row_id in self._row_ids:
            prices = self._row_price_weekly.get(row_id, pd.Series(dtype=float))
            if prices.empty:
                self._rsr_series.append(None)
                self._rsm_series.append(None)
                continue
            rsr, _, rsm = compute_rrg_indicators(prices, bench, cfg.rrg_window)
            self._rsr_series.append(rsr)
            self._rsm_series.append(rsm)

        self._log("Loading Vol% for recommendations (same source as live RRG)...")
        try:
            self._vol_by_ref = load_india_etf_vol_pct(
                self._row_ids,
                self._ref_labels,
                vol_days=cfg.vol_days,
                history_days=120,
            )
        except Exception as exc:
            self._log(f"Vol% load skipped ({exc}); using bhavcopy fallback.")
            self._vol_by_ref = {}

        warmup_bars = rrg_warmup_weeks(cfg.rrg_window) + cfg.tail
        start_ts = pd.Timestamp(cfg.backtest_start)
        end_ts = pd.Timestamp(cfg.backtest_end)

        candidates: list[pd.Timestamp] = []
        for d in self._weekly_index:
            if d.normalize() < start_ts.normalize() or d.normalize() > end_ts.normalize():
                continue
            pos = self._weekly_index.get_loc(d)
            if pos >= warmup_bars and pos < len(self._weekly_index) - 1:
                candidates.append(d)

        self._rebal_dates = self._collapse_tail_window_dates(
            candidates, start_ts, end_ts
        )

        if len(self._rebal_dates) < 1:
            raise RuntimeError(
                "No rebalance weeks in range — extend the period or check data "
                f"(warmup needs {warmup_bars} weekly bars before each date)"
            )
        self.reset_run()
        self._loaded = True
        first = self._rebal_dates[0].strftime("%Y-%m-%d")
        last = self._rebal_dates[-1].strftime("%Y-%m-%d")
        self._log(
            f"Ready: {len(self._rebal_dates)} as-of week(s) "
            f"({first} .. {last}). Match main RRG Date slider to as-of date."
        )

    def step_week(self) -> dict | None:
        if not self._loaded:
            raise RuntimeError("Call load_data() first")
        if self.finished:
            return None
        record = self._simulate_week(self._week_idx)
        self._week_idx += 1
        return record

    def step_back(self) -> dict | None:
        """Undo the last simulated week; return the new current record (if any)."""
        if not self._loaded or self._week_idx <= 0:
            return None
        removed = self._records.pop()
        self._week_idx -= 1
        port_ret = float(removed["Port_Return"])
        if port_ret != -1.0:
            self._portfolio_value /= 1.0 + port_ret
        else:
            self._portfolio_value = self.config.initial_capital
            for rec in self._records:
                self._portfolio_value *= 1.0 + float(rec["Port_Return"])
        if self._records:
            last = self._records[-1]
            picks = last.get("Picks") or []
            self._prev_holdings = [p.ticker for p in picks]
        else:
            self._prev_holdings = []
        return self._records[-1] if self._records else None

    def run_all(self) -> pd.DataFrame:
        if not self._loaded:
            raise RuntimeError("Call load_data() first")
        while not self.finished:
            self.step_week()
        return self.trades_df

    def _simulate_week(self, idx: int) -> dict:
        cfg = self.config
        decision_date = self._rebal_dates[idx]
        cal_pos = self._weekly_index.get_loc(decision_date)
        if cal_pos + 1 >= len(self._weekly_index):
            next_date = decision_date
        else:
            next_date = self._weekly_index[cal_pos + 1]
        tail_start = (
            self._weekly_index[cal_pos - cfg.tail]
            if cal_pos >= cfg.tail
            else decision_date
        )

        picks = self._recommend_at(decision_date)
        holdings = [p.ticker for p in picks]
        mid_week_9ema_exits = 0

        if cfg.exit_below_9ema and holdings:
            holdings = filter_holdings_below_9ema(
                holdings, self._ref_etf_daily, decision_date
            )

        turnover = 0.0
        new_entries = 0
        if self._prev_holdings:
            old_set = set(self._prev_holdings)
            new_set = set(holdings)
            new_entries = len(new_set - old_set)
            turnover = len(old_set.symmetric_difference(new_set)) / max(
                len(old_set | new_set), 1
            )
        else:
            new_entries = len(holdings)

        if holdings:
            if cfg.exit_below_9ema:
                week_rets, end_holdings, mid_week_9ema_exits = simulate_week_with_9ema_exits(
                    holdings,
                    decision_date,
                    next_date,
                    self._ref_etf_daily,
                    self._ref_etf_weekly,
                    cfg.top_n,
                )
                port_ret = float(np.mean(week_rets))
                holdings = end_holdings
            else:
                week_rets = []
                for ref in holdings:
                    series = self._ref_etf_weekly.get(ref)
                    if series is None or series.empty:
                        week_rets.append(0.0)
                        continue
                    s_from = series.loc[:decision_date]
                    s_to = series.loc[:next_date]
                    if len(s_from) == 0 or len(s_to) == 0:
                        week_rets.append(0.0)
                        continue
                    p0 = float(s_from.iloc[-1])
                    p1 = float(s_to.iloc[-1])
                    week_rets.append((p1 / p0 - 1) if p0 > 0 else 0.0)
                port_ret = float(np.mean(week_rets))
        else:
            port_ret = 0.0

        b_from = self._bench_weekly.loc[:decision_date]
        b_to = self._bench_weekly.loc[:next_date]
        bench_ret = (
            (float(b_to.iloc[-1]) / float(b_from.iloc[-1]) - 1)
            if len(b_from) > 0 and len(b_to) > 0
            else 0.0
        )

        self._portfolio_value *= 1 + port_ret

        pick_detail = "; ".join(
            f"{p.ticker}({p.quadrant[0]}{p.rank_delta})" for p in picks
        ) or "CASH"

        pick_prices = {
            p.ticker: self._ref_price_at(p.ticker, decision_date) for p in picks
        }

        record = {
            "Week": idx + 1,
            "Rebal_Date": decision_date,
            "Tail_Start": tail_start,
            "End_Date": next_date,
            "Holdings": ", ".join(holdings) or "CASH",
            "Pick_Detail": pick_detail,
            "Num_Holdings": len(holdings),
            "Port_Return": port_ret,
            "Bench_Return": bench_ret,
            "Excess_Return": port_ret - bench_ret,
            "Turnover": turnover,
            "New_Entries": new_entries,
            "Mid_Week_9EMA_Exits": mid_week_9ema_exits,
            "Portfolio_Value": self._portfolio_value,
            "Picks": picks,
            "Pick_Prices": pick_prices,
        }
        self._records.append(record)
        self._prev_holdings = holdings
        return record

    def _recommend_at(self, end_ts: pd.Timestamp):
        cfg = self.config
        n = len(self._row_ids)

        rsr_series = list(self._rsr_series)
        rsm_series = list(self._rsm_series)
        while len(rsr_series) < n:
            rsr_series.append(None)
            rsm_series.append(None)

        cal_pos = self._weekly_index.get_indexer([end_ts], method="ffill")[0]
        if cal_pos < 0:
            return []
        end_i = int(cal_pos)
        if end_i < cfg.tail:
            return []

        start_ts = self._weekly_index[end_i - cfg.tail]

        def change_pct_fn(j: int) -> float:
            row_id = self._row_ids[j]
            prices = self._row_price_weekly.get(row_id, pd.Series(dtype=float))
            return tail_change_pct(prices, start_ts, end_ts)

        def series_at_fn(series, ts):
            if series is None:
                raise KeyError
            return series_at(series, ts)

        curr_ranks = rank_by_tail_change(n, change_pct_fn)
        prev_end = self._weekly_index[end_i - 1] if end_i > 0 else None
        prev_ranks = (
            rank_by_tail_change(
                n,
                lambda j: tail_change_pct(
                    self._row_price_weekly.get(self._row_ids[j], pd.Series(dtype=float)),
                    self._weekly_index[end_i - 1 - cfg.tail],
                    prev_end,
                ),
            )
            if prev_end is not None and end_i > cfg.tail
            else {}
        )

        ranked = ranked_row_indices(n, change_pct_fn)
        rank_delta_by_row = build_rank_delta_by_row(ranked, curr_ranks, prev_ranks)
        vol_by_ref = self._vol_by_ref if self._vol_by_ref else self._vol_by_ref_at(end_ts)

        ctx = IndiaPickContext(
            ranked_row_indices=ranked,
            indices=self._row_ids,
            ref_labels=self._ref_labels,
            display_labels=self._display_labels,
            vol_by_ref=vol_by_ref,
            end_ts=end_ts,
            rsr_series_by_row=rsr_series,
            rsm_series_by_row=rsm_series,
            rank_delta_by_row=rank_delta_by_row,
            change_pct_fn=change_pct_fn,
            series_at_fn=series_at_fn,
            curr_ranks=curr_ranks,
            prev_ranks=prev_ranks,
            top_n=cfg.top_n,
            prev_holdings=list(self._prev_holdings),
            hold_until_rank_exit=cfg.hold_until_rank_exit,
            max_hold_rank=cfg.max_hold_rank,
        )
        return pick_india_portfolio(cfg.pick_strategy, ctx)

    def _vol_by_ref_at(self, as_of: pd.Timestamp) -> dict[str, float]:
        cfg = self.config
        out: dict[str, float] = {}
        for bare, daily in self._ref_etf_daily.items():
            truncated = daily.loc[:as_of]
            rets = truncated.pct_change().dropna()
            vol_slice = rets.tail(cfg.vol_days)
            if len(vol_slice) < max(10, cfg.vol_days // 3):
                continue
            vol_ann = float(vol_slice.std(ddof=0) * np.sqrt(252) * 100)
            if np.isfinite(vol_ann) and vol_ann > 0:
                out[bare] = vol_ann
        return out

    def ref_price_at(self, ref: str, as_of: pd.Timestamp) -> float | None:
        return self._ref_price_at(ref, as_of)

    def _ref_price_at(self, ref: str, as_of: pd.Timestamp) -> float | None:
        bare = (ref or "").upper().replace(".NS", "")
        series = self._ref_etf_weekly.get(bare)
        if series is None or series.empty:
            return None
        try:
            return round(series_at(series, as_of), 2)
        except (KeyError, TypeError, ValueError, IndexError):
            return None


def compute_metrics(df: pd.DataFrame, capital: float) -> dict:
    if df.empty:
        return {}
    port_rets = df["Port_Return"].values
    bench_rets = df["Bench_Return"].values
    n_weeks = len(port_rets)
    weeks_per_year = 52

    total_ret = df["Portfolio_Value"].iloc[-1] / capital - 1
    bench_total = df["Bench_Value"].iloc[-1] / capital - 1
    years = n_weeks / weeks_per_year

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0.0

    cum = (1 + pd.Series(port_rets)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    bench_cum = (1 + pd.Series(bench_rets)).cumprod()
    bench_max_dd = float((bench_cum / bench_cum.cummax() - 1).min())

    ann_vol = float(np.std(port_rets, ddof=1) * np.sqrt(weeks_per_year))
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    downside = port_rets[port_rets < 0]
    downside_vol = (
        float(np.std(downside, ddof=1) * np.sqrt(weeks_per_year))
        if len(downside) > 1
        else 0.0
    )
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    alpha = cagr - bench_cagr
    excess = pd.Series(port_rets) - pd.Series(bench_rets)
    te = float(excess.std(ddof=1)) * np.sqrt(weeks_per_year)
    info_ratio = alpha / te if te > 0 else 0.0

    total_trades = int(df["New_Entries"].sum()) if "New_Entries" in df.columns else 0

    return {
        "Strategy": STRATEGY_TAG,
        "Pick_Strategy": pick_strategy_label(
            str(df.attrs.get("pick_strategy", "recommend")),
            hold_until_rank_exit=bool(df.attrs.get("hold_until_rank_exit", False)),
            exit_below_9ema=bool(df.attrs.get("exit_below_9ema", False)),
        ),
        "Max_Hold_Rank": df.attrs.get("max_hold_rank"),
        "Benchmark": RRG_BENCHMARK_NSE,
        "Period": (
            f"{df['Rebal_Date'].iloc[0].strftime('%Y-%m-%d')} to "
            f"{df['End_Date'].iloc[-1].strftime('%Y-%m-%d')}"
        ),
        "Weeks": n_weeks,
        "Top_N": int(df.attrs.get("top_n", 7)),
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(float(np.mean(port_rets > 0) * 100), 1),
        "Avg_Weekly_Return_%": round(float(np.mean(port_rets) * 100), 2),
        "Total_Trades": total_trades,
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Ann_Volatility_%": round(ann_vol * 100, 2),
        "Avg_Turnover_%": round(float(df["Turnover"].mean() * 100), 1),
        "Alpha_%": round(alpha * 100, 2),
        "Information_Ratio": round(info_ratio, 2),
        "Bench_Total_Return_%": round(bench_total * 100, 2),
        "Bench_CAGR_%": round(bench_cagr * 100, 2),
        "Bench_Max_Drawdown_%": round(bench_max_dd * 100, 2),
        "Final_Value": round(float(df["Portfolio_Value"].iloc[-1]), 2),
        "Bench_Final_Value": round(float(df["Bench_Value"].iloc[-1]), 2),
    }


def run_backtest(
    backtest_start: str,
    backtest_end: str,
    *,
    top_n: int = 7,
    tail: int = 1,
    rrg_window: int = 10,
    initial_capital: float = 100_000.0,
    pick_strategy: str = "recommend",
    hold_until_rank_exit: bool = False,
    max_hold_rank: int = 10,
    exit_below_9ema: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = IndiaRrgBacktestConfig(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        top_n=top_n,
        tail=tail,
        rrg_window=rrg_window,
        initial_capital=initial_capital,
        pick_strategy=pick_strategy,
        hold_until_rank_exit=hold_until_rank_exit,
        max_hold_rank=max_hold_rank,
        exit_below_9ema=exit_below_9ema,
    )
    engine = IndiaRrgBacktestEngine(config=cfg, progress_cb=progress_cb)
    engine.load_data()
    df = engine.run_all()
    if not df.empty:
        df.attrs["top_n"] = top_n
        df.attrs["pick_strategy"] = pick_strategy
        df.attrs["hold_until_rank_exit"] = hold_until_rank_exit
        df.attrs["max_hold_rank"] = max_hold_rank
        df.attrs["exit_below_9ema"] = exit_below_9ema
    metrics = compute_metrics(df, initial_capital)
    return df, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="India RRG swing backtest")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--top-n", type=int, default=7)
    parser.add_argument("--tail", type=int, default=1)
    parser.add_argument("--window", type=int, default=10, choices=(10, 14))
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument(
        "--pick-strategy",
        choices=tuple(PICK_STRATEGIES),
        default="recommend",
        help="Base portfolio pick rule (default: recommend)",
    )
    parser.add_argument(
        "--hold-until-rank-exit",
        action="store_true",
        help="Keep holdings while momentum rank <= max-hold-rank; refill from base strategy",
    )
    parser.add_argument(
        "--max-hold-rank",
        type=int,
        default=10,
        metavar="RANK",
        help="With --hold-until-rank-exit: drop when rank is worse than R (default: 10)",
    )
    parser.add_argument(
        "--exit-below-9ema",
        action="store_true",
        help="Exit any holding when close < 9 EMA (mid-week); no refill until rebalance",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    end = args.end or today_ist().strftime("%Y-%m-%d")
    print(f"[{STRATEGY_TAG}] Backtest {args.start} .. {end}")
    print(
        f"  Pick: {pick_strategy_label(args.pick_strategy, hold_until_rank_exit=args.hold_until_rank_exit)}"
    )
    df, metrics = run_backtest(
        args.start,
        end,
        top_n=args.top_n,
        tail=args.tail,
        rrg_window=args.window,
        initial_capital=args.capital,
        pick_strategy=args.pick_strategy,
        hold_until_rank_exit=args.hold_until_rank_exit,
        max_hold_rank=args.max_hold_rank,
        exit_below_9ema=args.exit_below_9ema,
        progress_cb=print,
    )
    if metrics:
        print("\n" + "=" * 55)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        BACKTEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTEST_OUT_DIR / f"backtest_rrg_india_{args.start}_{end}.csv"
        export = df.drop(columns=["Picks"], errors="ignore")
        export.to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
