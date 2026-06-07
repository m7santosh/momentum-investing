"""
Walk-forward backtest for India NSE ETF RRG swing recommendations.

Replays the same weekly ranking + ``recommend_india_etfs`` pipeline as the live
RRG screen: tail-window rank, Leading/Improving quadrant, rank delta, vol scoring.

RRG row prices use the same NSE loaders as live RRG, over the backtest date range
(``RRGIndicatorEtfs._load_all_histories_range``).
P&L uses equal-weight ref ETF weekly closes from NSE CM bhavcopy.

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
    ref_to_row_index,
)
from momentum.etf.india_rrg_recommendations import (  # noqa: E402
    load_india_etf_vol_pct,
)
from momentum.etf.RRGIndicatorEtfs import _load_all_histories_range  # noqa: E402
from momentum.rrg_core import (  # noqa: E402
    compute_rrg_indicators,
    rrg_effective_window,
    rrg_format_date,
    rrg_min_history_bars,
    rrg_warmup_weeks,
)
from momentum.rrg_backtest_positions import (  # noqa: E402
    build_week_position_rows,
    register_new_week_entries,
    update_entry_prices_after_week,
)
from momentum.rrg_backtest_week_sim import intraweek_exits_enabled, simulate_backtest_week  # noqa: E402
from momentum.rrg_ema_exit import midweek_9ema_exit_count  # noqa: E402
from momentum.rrg_portfolio_exits import (  # noqa: E402
    build_week_exits,
    format_exit_summary,
    mid_week_9ema_label,
    mid_week_stop_loss_label,
    rebal_9ema_label,
)
from momentum.rrg_portfolio_fill import (  # noqa: E402
    PORTFOLIO_FILL_MAINTAIN_TOP_N,
    PORTFOLIO_FILL_MODES,
    PORTFOLIO_FILL_REPLACE,
    equal_weight_port_return,
    uses_prior_holdings,
)
from momentum.rrg_ref_price import (  # noqa: E402
    filter_picks_with_ref_price,
    filter_tickers_with_ref_price,
)
from momentum.rrg_ranking import (  # noqa: E402
    build_rank_delta_by_row,
    rank_by_tail_change,
    ranked_row_indices,
    series_at,
    tail_change_pct,
)
from utils.nse_bhavcopy import load_nse_cm_histories_range, today_ist  # noqa: E402
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
    exit_below_9ema: bool = True
    exit_stop_loss: bool = False
    stop_loss_pct: float = 3.0
    analysis_period: str = "3m"
    portfolio_fill_mode: str = PORTFOLIO_FILL_MAINTAIN_TOP_N


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
    _prev_rebalance_tickers: list[str] = field(default_factory=list)
    _prev_rank_at_rebal: dict[str, int] = field(default_factory=dict)
    _last_pick_ctx: IndiaPickContext | None = None
    _entry_prices: dict[str, float] = field(default_factory=dict)
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
        self._prev_rebalance_tickers = []
        self._prev_rank_at_rebal = {}
        self._entry_prices = {}

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
        min_history = rrg_effective_window(cfg.rrg_window, "week")
        min_weekly_points = rrg_min_history_bars(cfg.rrg_window, "week")

        self._log(
            f"Loading RRG weekly histories {rrg_format_date(dl_start)} .. "
            f"{rrg_format_date(dl_end)} (warmup + backtest range)..."
        )
        histories = _load_all_histories_range(
            dl_start,
            dl_end,
            min_weekly_points,
            cfg.rrg_window,
            freq="week",
        )
        for row_id in self._row_ids:
            self._row_price_weekly[row_id] = histories.get(
                row_id, pd.Series(dtype=float)
            )
        self._bench_weekly = histories.get(
            RRG_BENCHMARK_NSE, pd.Series(dtype=float)
        )
        if self._bench_weekly.empty:
            raise RuntimeError(f"Could not load benchmark {RRG_BENCHMARK_NSE}")

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
            f"Loading NSE ETF bhavcopy {rrg_format_date(vol_start)} .. "
            f"{rrg_format_date(dl_end)} "
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

        active_row_ids: list[str] = []
        active_ref_labels: list[str] = []
        active_display_labels: list[str] = []
        active_kinds: list[str] = []
        self._rsr_series = []
        self._rsm_series = []
        bench = self._bench_weekly
        skipped_history = 0
        skipped_rrg = 0
        for j, row_id in enumerate(self._row_ids):
            prices = self._row_price_weekly.get(row_id, pd.Series(dtype=float))
            if prices.notna().sum() <= min_history:
                skipped_history += 1
                continue
            rsr, _, rsm = compute_rrg_indicators(prices, bench, cfg.rrg_window)
            if rsr is None:
                skipped_rrg += 1
                continue
            active_row_ids.append(row_id)
            active_ref_labels.append(self._ref_labels[j])
            active_display_labels.append(self._display_labels[j])
            active_kinds.append(self._row_kinds[j])
            self._rsr_series.append(rsr)
            self._rsm_series.append(rsm)
        if skipped_history:
            self._log(
                f"Skipping {skipped_history} row(s) with insufficient weekly history "
                f"(same rule as live RRG)."
            )
        if skipped_rrg:
            self._log(
                f"Skipping {skipped_rrg} row(s) with no valid RRG series "
                f"(same rule as live RRG)."
            )
        if not active_row_ids:
            raise RuntimeError("No RRG rows with enough history for backtest.")
        self._row_ids = active_row_ids
        self._ref_labels = active_ref_labels
        self._display_labels = active_display_labels
        self._row_kinds = active_kinds

        self._weekly_index = self._bench_weekly.index.sort_values()

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

        first = rrg_format_date(self._rebal_dates[0])
        last = rrg_format_date(self._rebal_dates[-1])
        self._log(
            f"Ready: {len(self._rebal_dates)} weekly rebalance(s) "
            f"({first} .. {last}). Click Run All or Next Week."
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
            self._prev_holdings = list(last.get("Held_Tickers") or [])
            if not self._prev_holdings:
                raw = last.get("Holdings") or ""
                if raw and str(raw).strip().upper() != "CASH":
                    self._prev_holdings = [
                        t.strip() for t in str(raw).split(",") if t.strip()
                    ]
            self._prev_rebalance_tickers = list(last.get("Rebalance_Tickers") or [])
            self._prev_rank_at_rebal = dict(last.get("Rank_At_Rebal") or {})
            self._entry_prices = dict(last.get("Entry_Prices") or {})
        else:
            self._prev_holdings = []
            self._prev_rebalance_tickers = []
            self._prev_rank_at_rebal = {}
            self._entry_prices = {}
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

        was_portfolio = list(self._prev_rebalance_tickers)
        was_rank_at_rebal = dict(self._prev_rank_at_rebal)
        prev_rebal_date = (
            self._records[-1]["Rebal_Date"] if self._records else None
        )

        picks = self._recommend_at(decision_date)
        if self._last_pick_ctx is not None:
            from momentum.etf.india_rrg_pick_strategies import order_picks_by_table_rank

            picks = order_picks_by_table_rank(
                picks, self._last_pick_ctx.ranked_row_indices
            )
        rebalance_holdings = [p.ticker for p in picks]
        rebal_slots = list(rebalance_holdings)
        dropped_pick_9ema: list[str] = []
        dropped_9ema_rebal: list[str] = []
        mid_week_9ema: list = []
        mid_week_stop_loss: list = []

        if cfg.exit_below_9ema:
            from momentum.rrg_ema_exit import (
                apply_9ema_rebalance_slots,
                rebalance_9ema_dropped,
                rebalance_holdings_entered,
            )

            rebal_slots, dropped_pick_9ema = apply_9ema_rebalance_slots(
                rebalance_holdings,
                self._ref_etf_daily,
                decision_date,
                enabled=True,
            )
            holdings = rebalance_holdings_entered(rebal_slots)
            holdings, dropped_was_9ema = rebalance_9ema_dropped(
                holdings,
                was_portfolio or None,
                self._ref_etf_daily,
                decision_date,
            )
            dropped_9ema_rebal = list(dropped_pick_9ema)
            seen_drop = {
                sym.strip().upper().replace(".NS", "") for sym in dropped_9ema_rebal
            }
            for sym in dropped_was_9ema:
                bare = sym.strip().upper().replace(".NS", "")
                if bare and bare not in seen_drop:
                    seen_drop.add(bare)
                    dropped_9ema_rebal.append(sym)
        else:
            holdings = list(rebalance_holdings)

        holdings = filter_tickers_with_ref_price(
            list(holdings), self._ref_etf_weekly, decision_date
        )
        held_at_rebal = list(holdings)

        register_new_week_entries(
            self._entry_prices,
            held_at_rebal=held_at_rebal,
            prev_holdings=self._prev_holdings,
            decision_date=decision_date,
            price_weekly=self._ref_etf_weekly,
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
            port_ret, holdings, mid_week_9ema, mid_week_stop_loss = simulate_backtest_week(
                cfg,
                holdings,
                decision_date,
                next_date,
                self._ref_etf_daily,
                self._ref_etf_weekly,
                len(held_at_rebal),
            )
        else:
            port_ret = 0.0
            mid_week_stop_loss = []

        mid_week_9ema_exits = midweek_9ema_exit_count(mid_week_9ema)
        pick_ctx = self._last_pick_ctx
        if pick_ctx is not None:
            week_exits = build_week_exits(
                prev_holdings=was_portfolio,
                rebalance_holdings=held_at_rebal,
                hold_until_rank_exit=cfg.hold_until_rank_exit,
                curr_ranks=pick_ctx.curr_ranks,
                ref_to_j=ref_to_row_index(pick_ctx.indices, pick_ctx.ref_labels),
                max_hold_rank=cfg.max_hold_rank,
                exit_below_9ema=cfg.exit_below_9ema,
                dropped_9ema_rebal=dropped_9ema_rebal,
                mid_week_9ema=mid_week_9ema,
                decision_date=decision_date,
                exit_stop_loss=cfg.exit_stop_loss,
                mid_week_stop_loss=mid_week_stop_loss,
                stop_loss_pct=cfg.stop_loss_pct,
            )
        else:
            week_exits = []

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

        from momentum.rrg_portfolio_panel import norm_ticker

        rank_at_rebal: dict[str, int] = {}
        if pick_ctx is not None:
            for p in picks:
                rk = pick_ctx.curr_ranks.get(p.row_idx)
                if rk is not None:
                    rank_at_rebal[norm_ticker(p.ticker)] = int(rk)

        pick_shortfall = ""
        rebal_n_count = len([t for t in rebal_slots if t])
        if pick_ctx is not None and rebal_n_count < cfg.top_n:
            from momentum.etf.india_rrg_pick_strategies import pick_shortfall_hint

            pick_shortfall = pick_shortfall_hint(
                cfg.pick_strategy, pick_ctx, rebal_n_count
            )

        position_rows = build_week_position_rows(
            held_at_rebal=held_at_rebal,
            end_holdings=holdings,
            mid_week_9ema=mid_week_9ema,
            mid_week_stop_loss=mid_week_stop_loss,
            week_exits=week_exits,
            entry_prices=self._entry_prices,
            decision_date=decision_date,
            end_date=next_date,
            price_weekly=self._ref_etf_weekly,
            daily_close=self._ref_etf_daily,
            strategy_order=rebalance_holdings,
        )
        update_entry_prices_after_week(
            self._entry_prices,
            end_holdings=holdings,
            mid_week_9ema=mid_week_9ema,
            mid_week_stop_loss=mid_week_stop_loss,
            week_exits=week_exits,
        )

        record = {
            "Week": idx + 1,
            "Rebal_Date": decision_date,
            "Tail_Start": tail_start,
            "End_Date": next_date,
            "Holdings": ", ".join(holdings) or "CASH",
            "Held_Tickers": list(holdings),
            "Held_At_Rebal": list(held_at_rebal),
            "Rebalance_Tickers": list(rebal_slots),
            "Strategy_Tickers": list(rebalance_holdings),
            "Rebal_9EMA_Label": rebal_9ema_label(
                dropped_pick_9ema,
                decision_date,
                rebalance_tickers=rebalance_holdings,
            ),
            "Mid_Week_9EMA": list(mid_week_9ema),
            "Mid_Week_9EMA_Label": mid_week_9ema_label(
                mid_week_9ema, rebalance_tickers=held_at_rebal
            ),
            "Mid_Week_Stop_Loss": list(mid_week_stop_loss),
            "Mid_Week_Stop_Loss_Label": mid_week_stop_loss_label(
                mid_week_stop_loss, rebalance_tickers=held_at_rebal
            ),
            "Was_Portfolio": was_portfolio,
            "Was_Rank_At_Rebal": was_rank_at_rebal,
            "Rank_At_Rebal": rank_at_rebal,
            "Prev_Rebal_Date": prev_rebal_date,
            "Pick_Detail": pick_detail,
            "Pick_Shortfall": pick_shortfall,
            "Num_Holdings": len(holdings),
            "Port_Return": port_ret,
            "Bench_Return": bench_ret,
            "Excess_Return": port_ret - bench_ret,
            "Turnover": turnover,
            "New_Entries": new_entries,
            "Mid_Week_9EMA_Exits": mid_week_9ema_exits,
            "Exits": week_exits,
            "Exit_Summary": format_exit_summary(
                week_exits, rebalance_tickers=held_at_rebal
            ),
            "Portfolio_Value": self._portfolio_value,
            "Picks": picks,
            "Pick_Prices": pick_prices,
            "Position_Rows": position_rows,
            "Entry_Prices": dict(self._entry_prices),
        }
        self._records.append(record)
        self._prev_holdings = holdings
        self._prev_rebalance_tickers = list(rebal_slots)
        self._prev_rank_at_rebal = dict(rank_at_rebal)
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
            prev_holdings=(
                list(self._prev_holdings)
                if cfg.hold_until_rank_exit or uses_prior_holdings(cfg.portfolio_fill_mode)
                else []
            ),
            hold_until_rank_exit=cfg.hold_until_rank_exit,
            max_hold_rank=cfg.max_hold_rank,
            portfolio_fill_mode=cfg.portfolio_fill_mode,
        )
        self._last_pick_ctx = ctx
        picks = pick_india_portfolio(cfg.pick_strategy, ctx)
        return filter_picks_with_ref_price(
            picks, self._ref_etf_weekly, end_ts
        )

    def _rebalance_picks_at(
        self, decision_date: pd.Timestamp
    ) -> tuple[list, list[str], list[str], dict[str, int], str]:
        """Strategy Top N + 9 EMA slots at ``decision_date`` (no week simulation)."""
        cfg = self.config
        picks = self._recommend_at(decision_date)
        pick_ctx = self._last_pick_ctx
        if pick_ctx is not None:
            from momentum.etf.india_rrg_pick_strategies import order_picks_by_table_rank

            picks = order_picks_by_table_rank(
                picks, pick_ctx.ranked_row_indices
            )
        rebalance_holdings = [p.ticker for p in picks]
        rebal_slots = list(rebalance_holdings)
        if cfg.exit_below_9ema:
            from momentum.rrg_ema_exit import apply_9ema_rebalance_slots

            rebal_slots, _ = apply_9ema_rebalance_slots(
                rebalance_holdings,
                self._ref_etf_daily,
                decision_date,
                enabled=True,
            )
        from momentum.rrg_portfolio_panel import norm_ticker

        rank_at_rebal: dict[str, int] = {}
        if pick_ctx is not None:
            for p in picks:
                rk = pick_ctx.curr_ranks.get(p.row_idx)
                if rk is not None:
                    rank_at_rebal[norm_ticker(p.ticker)] = int(rk)
        pick_shortfall = ""
        rebal_n_count = len([t for t in rebal_slots if t])
        if pick_ctx is not None and rebal_n_count < cfg.top_n:
            from momentum.etf.india_rrg_pick_strategies import pick_shortfall_hint

            pick_shortfall = pick_shortfall_hint(
                cfg.pick_strategy, pick_ctx, rebal_n_count
            )
        return picks, rebalance_holdings, rebal_slots, rank_at_rebal, pick_shortfall

    def _prior_panel_was_slots(
        self, prev_bar_ts: pd.Timestamp
    ) -> tuple[list[str], dict[str, int]]:
        """
        Prior bar Top N slots (★ order) — matches main ``_prior_week_top_n_portfolio``.

        Uses empty ``prev_holdings`` so Was is not contaminated by simulation state.
        """
        saved = list(self._prev_holdings)
        self._prev_holdings = []
        try:
            _, _, slots, ranks, _ = self._rebalance_picks_at(prev_bar_ts)
            return list(slots), dict(ranks)
        finally:
            self._prev_holdings = saved

    def _week_exits_live_at(
        self,
        decision_date: pd.Timestamp,
        *,
        prev_was_slots: list[str],
        next_date: pd.Timestamp,
        rebal_slots: list[str] | None = None,
        rebalance_holdings: list[str] | None = None,
    ) -> list:
        """Exit events for one rebalance bar (same rules as ``_simulate_week``)."""
        cfg = self.config
        from momentum.etf.india_rrg_pick_strategies import (
            order_picks_by_table_rank,
            ref_to_row_index,
        )
        from momentum.rrg_portfolio_exits import build_week_exits

        was_portfolio = [t for t in prev_was_slots if t]
        saved = list(self._prev_holdings)
        self._prev_holdings = was_portfolio
        try:
            picks = self._recommend_at(decision_date)
            pick_ctx = self._last_pick_ctx
            if pick_ctx is not None:
                picks = order_picks_by_table_rank(
                    picks, pick_ctx.ranked_row_indices
                )
            if rebalance_holdings is None:
                rebalance_holdings = [p.ticker for p in picks]
            if rebal_slots is None:
                rebal_slots = list(rebalance_holdings)
                if cfg.exit_below_9ema:
                    from momentum.rrg_ema_exit import apply_9ema_rebalance_slots

                    rebal_slots, dropped_pick_9ema = apply_9ema_rebalance_slots(
                        rebalance_holdings,
                        self._ref_etf_daily,
                        decision_date,
                        enabled=True,
                    )
                else:
                    dropped_pick_9ema = []
            else:
                entered = {t for t in rebal_slots if t}
                dropped_pick_9ema = [
                    t for t in rebalance_holdings if t and t not in entered
                ]

            held_at_rebal = [t for t in rebal_slots if t]
            dropped_9ema_rebal: list[str] = list(dropped_pick_9ema)
            mid_week_9ema: list = []
            mid_week_stop_loss: list = []

            if cfg.exit_below_9ema:
                from momentum.rrg_ema_exit import (
                    rebalance_9ema_dropped,
                    simulate_week_with_exits,
                )

                holdings, dropped_was_9ema = rebalance_9ema_dropped(
                    held_at_rebal,
                    was_portfolio or None,
                    self._ref_etf_daily,
                    decision_date,
                )
                seen_drop = {
                    sym.strip().upper().replace(".NS", "")
                    for sym in dropped_9ema_rebal
                }
                for sym in dropped_was_9ema:
                    bare = sym.strip().upper().replace(".NS", "")
                    if bare and bare not in seen_drop:
                        seen_drop.add(bare)
                        dropped_9ema_rebal.append(sym)
                if (
                    next_date is not None
                    and pd.Timestamp(next_date) > pd.Timestamp(decision_date)
                    and intraweek_exits_enabled(cfg)
                ):
                    _, _, mid_week_9ema, mid_week_stop_loss = simulate_week_with_exits(
                        holdings,
                        decision_date,
                        next_date,
                        self._ref_etf_daily,
                        self._ref_etf_weekly,
                        cfg.top_n,
                        exit_below_9ema=cfg.exit_below_9ema,
                        exit_stop_loss=cfg.exit_stop_loss,
                        stop_loss_pct=cfg.stop_loss_pct,
                    )
            if pick_ctx is None:
                return []
            return build_week_exits(
                prev_holdings=was_portfolio,
                rebalance_holdings=held_at_rebal,
                hold_until_rank_exit=cfg.hold_until_rank_exit,
                curr_ranks=pick_ctx.curr_ranks,
                ref_to_j=ref_to_row_index(pick_ctx.indices, pick_ctx.ref_labels),
                max_hold_rank=cfg.max_hold_rank,
                exit_below_9ema=cfg.exit_below_9ema,
                dropped_9ema_rebal=dropped_9ema_rebal,
                mid_week_9ema=mid_week_9ema,
                decision_date=decision_date,
                exit_stop_loss=cfg.exit_stop_loss,
                mid_week_stop_loss=mid_week_stop_loss,
                stop_loss_pct=cfg.stop_loss_pct,
            )
        finally:
            self._prev_holdings = saved

    def _panel_exit_slices_live(
        self,
        *,
        prev_rebal_ts: pd.Timestamp | None,
        panel_rebal_ts: pd.Timestamp,
        panel_end_ts: pd.Timestamp,
        was_portfolio: list[str],
        rebal_tickers: list[str],
        strategy_tickers: list[str],
        wi: pd.DatetimeIndex,
        panel_i: int,
        tail_n: int,
    ) -> list[tuple]:
        slices: list[tuple] = []
        if prev_rebal_ts is not None:
            prev_rec = self._record_by_rebal(prev_rebal_ts)
            if prev_rec is not None:
                slices.append(
                    (pd.Timestamp(prev_rebal_ts), prev_rec.get("Exits") or [])
                )
            else:
                prev_i = panel_i - 1
                prev_prev_was: list[str] = []
                if prev_i - 1 >= tail_n:
                    prev_prev_was, _ = self._prior_panel_was_slots(
                        pd.Timestamp(wi[prev_i - 1])
                    )
                slices.append(
                    (
                        pd.Timestamp(prev_rebal_ts),
                        self._week_exits_live_at(
                            prev_rebal_ts,
                            prev_was_slots=prev_prev_was,
                            next_date=panel_rebal_ts,
                        ),
                    )
                )
        cur_rec = self._record_by_rebal(panel_rebal_ts)
        if cur_rec is not None:
            slices.append(
                (pd.Timestamp(panel_rebal_ts), cur_rec.get("Exits") or [])
            )
        else:
            slices.append(
                (
                    pd.Timestamp(panel_rebal_ts),
                    self._week_exits_live_at(
                        panel_rebal_ts,
                        prev_was_slots=was_portfolio,
                        next_date=panel_end_ts,
                        rebal_slots=rebal_tickers,
                        rebalance_holdings=strategy_tickers,
                    ),
                )
            )
        return slices

    def portfolio_panel_context(self, record: dict) -> dict:
        """
        Portfolio panel at record Rebal_Date (backtest Current week).
        Was = prior week picks; Top N = this week's rebalance picks.
        """
        from momentum.rrg_portfolio_exits import panel_was_out_exits

        rebal_ts = pd.Timestamp(record["Rebal_Date"])
        end_ts = pd.Timestamp(record["End_Date"])
        was_portfolio = list(record.get("Was_Portfolio") or [])
        was_ranks = dict(record.get("Was_Rank_At_Rebal") or {})
        strategy_tickers = list(record.get("Strategy_Tickers") or [])
        rebal_tickers = list(record.get("Rebalance_Tickers") or [])
        curr_ranks = dict(record.get("Rank_At_Rebal") or {})
        pick_shortfall = str(record.get("Pick_Shortfall") or "")
        end_prev_week_holdings: list[str] | None = None
        week_num = int(record.get("Week") or 0)
        if week_num > 1 and week_num - 2 < len(self._records):
            prev_row = self._records[week_num - 2]
            end_prev_week_holdings = list(prev_row.get("Held_Tickers") or [])

        prev_rebal = record.get("Prev_Rebal_Date")
        exit_slices: list[tuple] = []
        if prev_rebal is not None and week_num > 1:
            prev_rec = self._records[week_num - 2]
            exit_slices.append(
                (pd.Timestamp(prev_rebal), prev_rec.get("Exits") or [])
            )
        exit_slices.append((rebal_ts, record.get("Exits") or []))
        prev_rebal_ts = pd.Timestamp(prev_rebal) if prev_rebal is not None else None

        def _daily_for_panel(sym: str) -> pd.Series | None:
            bare = sym.strip().upper().replace(".NS", "")
            return self._ref_etf_daily.get(bare)

        panel_exits = panel_was_out_exits(
            exit_slices,
            end_ts,
            prev_rebal_ts=prev_rebal_ts,
            panel_rebal_ts=rebal_ts,
            prev_holdings=was_portfolio,
            rebalance_holdings=[t for t in rebal_tickers if t],
            exit_below_9ema=self.config.exit_below_9ema,
            daily_for_ticker=_daily_for_panel,
        )
        from momentum.rrg_core import rrg_format_date

        was_label = (
            rrg_format_date(prev_rebal) if prev_rebal is not None else "—"
        )
        return {
            "rebal_ts": rebal_ts,
            "end_ts": end_ts,
            "prev_rebal_ts": (
                pd.Timestamp(prev_rebal) if prev_rebal is not None else None
            ),
            "was_portfolio": was_portfolio,
            "was_ranks": was_ranks,
            "was_label": was_label,
            "rebalance_label": rrg_format_date(rebal_ts),
            "strategy_tickers": strategy_tickers,
            "rebal_tickers": rebal_tickers,
            "curr_ranks": curr_ranks,
            "pick_shortfall": pick_shortfall,
            "end_prev_week_holdings": end_prev_week_holdings,
            "panel_exits": panel_exits,
            "mid_week_9ema": list(record.get("Mid_Week_9EMA") or []),
        }

    def _record_by_rebal(self, rebal_ts: pd.Timestamp | None) -> dict | None:
        if rebal_ts is None:
            return None
        target = pd.Timestamp(rebal_ts).normalize()
        for rec in self._records:
            if pd.Timestamp(rec["Rebal_Date"]).normalize() == target:
                return rec
        return None

    def _exit_slices_for_panel(
        self,
        prev_rebal_ts: pd.Timestamp | None,
        panel_rebal_ts: pd.Timestamp,
    ) -> list[tuple]:
        slices: list[tuple] = []
        if prev_rebal_ts is not None:
            prev_rec = self._record_by_rebal(prev_rebal_ts)
            if prev_rec is not None:
                slices.append(
                    (pd.Timestamp(prev_rebal_ts), prev_rec.get("Exits") or [])
                )
        cur_rec = self._record_by_rebal(panel_rebal_ts)
        slices.append(
            (
                pd.Timestamp(panel_rebal_ts),
                (cur_rec.get("Exits") or []) if cur_rec is not None else [],
            )
        )
        return slices

    def portfolio_panel_at_asof(
        self,
        as_of_ts: pd.Timestamp,
        *,
        tail_bars: int | None = None,
        weekly_index: pd.DatetimeIndex | None = None,
    ) -> dict:
        """
        Portfolio panel at ``as_of`` using the same rebalance bar rule as main RRG.

        On a weekly bar that starts a new hold week (e.g. 08-05), Top N picks are
        computed at that bar; Was comes from the prior weekly bar (01-05).
        """
        from momentum.rrg_core import panel_rebal_bar_index, rrg_format_date
        from momentum.rrg_portfolio_exits import panel_was_out_exits

        as_of = pd.Timestamp(as_of_ts)
        cfg = self.config
        tail_n = max(1, int(tail_bars if tail_bars is not None else cfg.tail))
        wi = (
            pd.DatetimeIndex(weekly_index).sort_values()
            if weekly_index is not None and len(weekly_index)
            else self._weekly_index
        )
        panel_i = panel_rebal_bar_index(wi, as_of, tail_n)
        panel_rebal_ts = pd.Timestamp(wi[panel_i])
        panel_end_i = min(panel_i + 1, len(wi) - 1)
        panel_end_ts = pd.Timestamp(wi[panel_end_i])
        prev_i = panel_i - 1 if panel_i > tail_n else None
        prev_rebal_ts = (
            pd.Timestamp(wi[prev_i]) if prev_i is not None else None
        )

        was_portfolio: list[str] = []
        was_ranks: dict = {}
        if prev_i is not None and prev_i >= tail_n:
            prev_bar_ts = pd.Timestamp(wi[prev_i])
            was_portfolio, was_ranks = self._prior_panel_was_slots(prev_bar_ts)

        prev_holdings_for_picks: list[str] = []
        if prev_rebal_ts is not None:
            prev_rec = self._record_by_rebal(prev_rebal_ts)
            if prev_rec is not None and (
                cfg.hold_until_rank_exit
                or cfg.exit_below_9ema
                or uses_prior_holdings(cfg.portfolio_fill_mode)
            ):
                prev_holdings_for_picks = list(
                    prev_rec.get("Held_Tickers") or []
                )
            else:
                prev_holdings_for_picks = [t for t in was_portfolio if t]

        saved_holdings = list(self._prev_holdings)
        self._prev_holdings = prev_holdings_for_picks
        try:
            _, strategy_tickers, rebal_tickers, curr_ranks, pick_shortfall = (
                self._rebalance_picks_at(panel_rebal_ts)
            )
        finally:
            self._prev_holdings = saved_holdings

        end_prev_week_holdings: list[str] | None = None
        if prev_rebal_ts is not None:
            prev_rec = self._record_by_rebal(prev_rebal_ts)
            if prev_rec is not None:
                end_prev_week_holdings = list(prev_rec.get("Held_Tickers") or [])
        if end_prev_week_holdings is None:
            end_prev_week_holdings = [t for t in was_portfolio if t] or None

        exit_slices = self._panel_exit_slices_live(
            prev_rebal_ts=prev_rebal_ts,
            panel_rebal_ts=panel_rebal_ts,
            panel_end_ts=panel_end_ts,
            was_portfolio=was_portfolio,
            rebal_tickers=rebal_tickers,
            strategy_tickers=strategy_tickers,
            wi=wi,
            panel_i=panel_i,
            tail_n=tail_n,
        )

        def _daily_for_panel(sym: str) -> pd.Series | None:
            bare = sym.strip().upper().replace(".NS", "")
            return self._ref_etf_daily.get(bare)

        panel_exits = panel_was_out_exits(
            exit_slices,
            panel_end_ts,
            prev_rebal_ts=prev_rebal_ts,
            panel_rebal_ts=panel_rebal_ts,
            prev_holdings=[t for t in was_portfolio if t],
            rebalance_holdings=[t for t in rebal_tickers if t],
            exit_below_9ema=cfg.exit_below_9ema,
            daily_for_ticker=_daily_for_panel,
        )

        mid_week_9ema: list = []
        mid_week_stop_loss: list = []
        cur_rec = self._record_by_rebal(panel_rebal_ts)
        if cur_rec is not None:
            mid_week_9ema = list(cur_rec.get("Mid_Week_9EMA") or [])
            mid_week_stop_loss = list(cur_rec.get("Mid_Week_Stop_Loss") or [])
        elif (
            intraweek_exits_enabled(cfg)
            and pd.Timestamp(panel_end_ts) > pd.Timestamp(panel_rebal_ts)
        ):
            from momentum.rrg_ema_exit import (
                rebalance_9ema_dropped,
                rebalance_holdings_entered,
                simulate_week_with_exits,
            )

            held = rebalance_holdings_entered(rebal_tickers)
            held, _ = rebalance_9ema_dropped(
                held,
                [t for t in was_portfolio if t] or None,
                self._ref_etf_daily,
                panel_rebal_ts,
            )
            _, _, mid_week_9ema, mid_week_stop_loss = simulate_week_with_exits(
                held,
                panel_rebal_ts,
                panel_end_ts,
                self._ref_etf_daily,
                self._ref_etf_weekly,
                cfg.top_n,
                exit_below_9ema=cfg.exit_below_9ema,
                exit_stop_loss=cfg.exit_stop_loss,
                stop_loss_pct=cfg.stop_loss_pct,
            )

        was_label = (
            rrg_format_date(prev_rebal_ts) if prev_rebal_ts is not None else "—"
        )
        return {
            "rebal_ts": panel_rebal_ts,
            "panel_end_ts": panel_end_ts,
            "end_ts": as_of,
            "prev_rebal_ts": prev_rebal_ts,
            "was_portfolio": was_portfolio,
            "was_ranks": was_ranks,
            "was_label": was_label,
            "rebalance_label": rrg_format_date(panel_rebal_ts),
            "exits_through_label": rrg_format_date(panel_end_ts),
            "strategy_tickers": list(strategy_tickers),
            "rebal_tickers": list(rebal_tickers),
            "curr_ranks": dict(curr_ranks),
            "pick_shortfall": pick_shortfall,
            "end_prev_week_holdings": end_prev_week_holdings,
            "panel_exits": panel_exits,
            "mid_week_9ema": mid_week_9ema,
            "mid_week_stop_loss": mid_week_stop_loss,
        }

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
    from momentum.rrg_core import rrg_format_date

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

    total_trades = int(df["New_Entries"].sum()) if "New_Entries" in df.columns else 0

    return {
        "Pick_Strategy": pick_strategy_label(
            str(df.attrs.get("pick_strategy", "recommend")),
            hold_until_rank_exit=bool(df.attrs.get("hold_until_rank_exit", False)),
            exit_below_9ema=bool(df.attrs.get("exit_below_9ema", True)),
        ),
        "Max_Hold_Rank": df.attrs.get("max_hold_rank"),
        "Period": (
            f"{rrg_format_date(df['Rebal_Date'].iloc[0])} to "
            f"{rrg_format_date(df['End_Date'].iloc[-1])}"
        ),
        "Weeks": n_weeks,
        "Total_Return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Win_Rate_%": round(float(np.mean(port_rets > 0) * 100), 1),
        "Avg_Weekly_Return_%": round(float(np.mean(port_rets) * 100), 2),
        "Total_Trades": total_trades,
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
    top_n: int = 7,
    tail: int = 1,
    rrg_window: int = 10,
    initial_capital: float = 100_000.0,
    pick_strategy: str = "recommend",
    hold_until_rank_exit: bool = False,
    max_hold_rank: int = 10,
    exit_below_9ema: bool = True,
    exit_stop_loss: bool = False,
    stop_loss_pct: float = 3.0,
    analysis_period: str = "3m",
    portfolio_fill_mode: str = PORTFOLIO_FILL_MAINTAIN_TOP_N,
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
        exit_stop_loss=exit_stop_loss,
        stop_loss_pct=stop_loss_pct,
        analysis_period=analysis_period,
        portfolio_fill_mode=portfolio_fill_mode,
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
        df.attrs["exit_stop_loss"] = exit_stop_loss
        df.attrs["stop_loss_pct"] = stop_loss_pct
        df.attrs["portfolio_fill_mode"] = portfolio_fill_mode
    metrics = compute_metrics(df, initial_capital)
    return df, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="India RRG swing backtest")
    parser.add_argument("--start", default=None, help="Backtest start (default: 1 Jan this year)")
    parser.add_argument("--end", default=None, help="Backtest end (default: today)")
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit when close < 9 EMA (mid-week); default on",
    )
    parser.add_argument(
        "--exit-stop-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit when loss from rebalance entry hits stop-loss %% (mid-week)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=3.0,
        metavar="PCT",
        help="Stop-loss %% below rebalance entry (default: 3)",
    )
    parser.add_argument(
        "--portfolio-fill",
        choices=tuple(PORTFOLIO_FILL_MODES),
        default=PORTFOLIO_FILL_MAINTAIN_TOP_N,
        metavar="MODE",
        help="How rebalance picks combine with prior holdings (default: maintain_top_n)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = args.start or f"{today_ist().year}-01-01"
    end = args.end or today_ist().strftime("%Y-%m-%d")
    print(f"[{STRATEGY_TAG}] Backtest {start} .. {end}")
    print(
        f"  Pick: {pick_strategy_label(args.pick_strategy, hold_until_rank_exit=args.hold_until_rank_exit)}"
    )
    df, metrics = run_backtest(
        start,
        end,
        top_n=args.top_n,
        tail=args.tail,
        rrg_window=args.window,
        initial_capital=args.capital,
        pick_strategy=args.pick_strategy,
        hold_until_rank_exit=args.hold_until_rank_exit,
        max_hold_rank=args.max_hold_rank,
        exit_below_9ema=args.exit_below_9ema,
        exit_stop_loss=args.exit_stop_loss,
        stop_loss_pct=args.stop_loss_pct,
        portfolio_fill_mode=args.portfolio_fill,
        progress_cb=print,
    )
    if metrics:
        print("\n" + "=" * 55)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        BACKTEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTEST_OUT_DIR / f"backtest_rrg_india_{start}_{end}.csv"
        export = df.drop(columns=["Picks"], errors="ignore")
        export.to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
