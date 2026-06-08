"""Standalone Tkinter UI for RRG backtest (India / US). Not tied to main RRG table."""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from momentum.rrg_backtest_positions import (  # noqa: E402
    format_exit_date,
    format_pl_pct,
    format_price,
    sort_position_rows,
)
from momentum.rrg_core import (
    RRG_MAX_TAIL,
    rrg_config_date_str,
    rrg_format_date,
    rrg_parse_user_date,
)
from momentum.rrg_busy import RrgBusyOverlay
from momentum.rrg_ui_copy import install_copy_support
from utils.nse_bhavcopy import today_ist


@dataclass(frozen=True)
class _BacktestUiProfile:
    title: str
    bench_chart: str
    load_status: str
    Config: type
    Engine: type
    compute_metrics: Callable[..., dict]
    format_vol_pct: Callable[[float], str]
    recommendation_row_bg: Callable[[str], str]


def _backtest_profile(name: str) -> _BacktestUiProfile:
    if name == "us":
        from backtest.etf.backtest_rrg_us import (
            UsRrgBacktestConfig,
            UsRrgBacktestEngine,
            compute_metrics,
        )
        from momentum.etf.us_rrg_recommendations import (
            format_vol_pct,
            recommendation_row_bg,
        )

        return _BacktestUiProfile(
            title="US RRG Backtest",
            bench_chart="S&P 500",
            load_status="Loading universe and Yahoo data (may take a few minutes)...",
            Config=UsRrgBacktestConfig,
            Engine=UsRrgBacktestEngine,
            compute_metrics=compute_metrics,
            format_vol_pct=format_vol_pct,
            recommendation_row_bg=recommendation_row_bg,
        )
    if name == "stock":
        from backtest.stock.backtest_rrg_stocks import (
            StockRrgBacktestConfig,
            StockRrgBacktestEngine,
            compute_metrics,
        )
        from momentum.stock.stock_rrg_recommendations import (
            format_vol_pct,
            recommendation_row_bg,
        )

        return _BacktestUiProfile(
            title="Stock RRG Backtest",
            bench_chart="Benchmark",
            load_status="Loading NSE stock data (may take a few minutes)...",
            Config=StockRrgBacktestConfig,
            Engine=StockRrgBacktestEngine,
            compute_metrics=compute_metrics,
            format_vol_pct=format_vol_pct,
            recommendation_row_bg=recommendation_row_bg,
        )
    from backtest.etf.backtest_rrg_india import (
        IndiaRrgBacktestConfig,
        IndiaRrgBacktestEngine,
        compute_metrics,
    )
    from momentum.etf.india_rrg_recommendations import (
        format_vol_pct,
        recommendation_row_bg,
    )

    return _BacktestUiProfile(
        title="India RRG Backtest",
        bench_chart="Nifty 500",
        load_status="Loading NSE data (may take a few minutes)...",
        Config=IndiaRrgBacktestConfig,
        Engine=IndiaRrgBacktestEngine,
        compute_metrics=compute_metrics,
        format_vol_pct=format_vol_pct,
        recommendation_row_bg=recommendation_row_bg,
    )


def backtest_week_pick_detail(record: dict) -> dict[str, str]:
    """Strategy vs entered Top N for Selected week (matches main RRG panel semantics)."""
    strategy = [t for t in (record.get("Strategy_Tickers") or []) if t]
    entered = [t for t in (record.get("Rebalance_Tickers") or []) if t]
    strategy_s = ", ".join(strategy) if strategy else "—"
    n_strat = len(strategy)
    n_ent = len(entered)
    if n_ent:
        entered_s = (
            f"{', '.join(entered)} ({n_ent}/{n_strat})"
            if n_strat
            else ", ".join(entered)
        )
    else:
        entered_s = f"— (0/{n_strat})" if n_strat else "—"

    reason_parts: list[str] = []
    shortfall = (record.get("Pick_Shortfall") or "").strip()
    if shortfall:
        reason_parts.append(shortfall)
    rebal_9ema = (record.get("Rebal_9EMA_Label") or "").strip()
    if rebal_9ema:
        reason_parts.append(f"Below 9 EMA @ rebalance: {rebal_9ema}")
    elif n_ent < n_strat:
        reason_parts.append(
            f"{n_ent}/{n_strat} entered — remaining strategy slots empty at rebalance"
        )
    reason_s = " | ".join(reason_parts) if reason_parts else "—"

    return {
        "Strategy picks": strategy_s,
        "Entered picks": entered_s,
        "Pick gap reason": reason_s,
    }


def backtest_drawdown_pct_series(port_returns: pd.Series) -> pd.Series:
    """Running drawdown % from weekly ``Port_Return`` (same rule as summary Max DD)."""
    if port_returns.empty:
        return pd.Series(dtype=float)
    cum = (1 + port_returns).cumprod()
    return (cum / cum.cummax() - 1) * 100


def backtest_cum_pl_pct(portfolio_value: float, initial_capital: float) -> float:
    """Total portfolio P/L % since backtest start."""
    if initial_capital <= 0:
        return 0.0
    return (portfolio_value / initial_capital - 1) * 100


def _week_detail_fields(
    record: dict,
    *,
    total_weeks: int,
    initial_capital: float,
) -> dict[str, str]:
    from momentum.rrg_portfolio_exits import format_exit_summary

    rebal = rrg_format_date(record["Rebal_Date"])
    end = rrg_format_date(record["End_Date"])
    exits = format_exit_summary(
        record.get("Exits") or [],
        rebalance_tickers=list(record.get("Rebalance_Tickers") or []),
    )
    cum_pl = backtest_cum_pl_pct(float(record["Portfolio_Value"]), initial_capital)
    fields = {
        "Week": f"{record['Week']} of {total_weeks}",
        "Hold start": rebal,
        "Hold end": end,
        "Portfolio % (week)": f"{record['Port_Return'] * 100:+.2f}",
        "Portfolio P/L % (since start)": f"{cum_pl:+.2f}",
        "Benchmark %": f"{record['Bench_Return'] * 100:+.2f}",
        "Portfolio value": f"{record['Portfolio_Value']:,.0f}",
        "Holdings at week-end": record.get("Holdings") or "CASH",
        "Exits": exits or "—",
    }
    fields.update(backtest_week_pick_detail(record))
    return fields


def _fill_position_tree(
    tree: ttk.Treeview,
    rows: list[dict] | None,
    *,
    strategy_order: list[str] | None = None,
) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if not rows:
        tree.insert("", tk.END, values=("—", "—", "—", "—", "—", "—", "—"))
        return
    ordered = sort_position_rows(rows, strategy_order)
    for row in ordered:
        tree.insert(
            "",
            tk.END,
            values=(
                row.get("ticker") or "—",
                row.get("status") or "—",
                format_price(row.get("entry")),
                format_price(row.get("exit")),
                format_exit_date(row.get("exit_date")),
                format_price(row.get("running")),
                format_pl_pct(row.get("pl_pct")),
            ),
        )


def _fill_kv_tree(tree: ttk.Treeview, rows: dict[str, str] | None, *, empty: str) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if not rows:
        tree.insert("", tk.END, values=("—", empty))
        return
    for key, value in rows.items():
        tree.insert("", tk.END, values=(key, value))


def _make_kv_tree(
    parent: tk.Misc,
    *,
    height: int,
    col0_width: int = 140,
    col1_width: int = 220,
) -> ttk.Treeview:
    cols = ("Field", "Value")
    tree = ttk.Treeview(parent, columns=cols, show="headings", height=height, selectmode="none")
    tree.heading("Field", text="Field")
    tree.heading("Value", text="Value")
    tree.column("Field", width=col0_width, stretch=False)
    tree.column("Value", width=col1_width, stretch=True)
    scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    return tree


def open_rrg_backtest(
    parent: tk.Misc,
    *,
    profile: str = "india",
    rrg_window: int = 10,
    tail: int = 1,
    bar_unit: str = "week",  # ignored — backtest uses weekly rebalance only
    analysis_period: str = "3m",
    top_n: int = 7,
    backtest_extra: dict[str, Any] | None = None,
    pick_strategy: str | None = None,
    hold_until_rank_exit: bool | None = None,
    max_hold_rank: int | None = None,
    exit_below_9ema: bool | None = None,
    initial_as_of: str | None = None,  # ignored — use Start/End dates only
    initial_tail_from: str | None = None,  # ignored
    initial_start: str | None = None,
    initial_end: str | None = None,
    shutdown_root: tk.Misc | None = None,
) -> tk.Toplevel:
    del initial_as_of, initial_tail_from, bar_unit

    prof = _backtest_profile(profile)
    bt_extra = dict(backtest_extra or {})
    bt_extra.setdefault("analysis_period", analysis_period)
    if profile == "stock" and not bt_extra.get("universe_key"):
        from momentum.stock.universes import DEFAULT_KEY

        bt_extra.setdefault("universe_key", DEFAULT_KEY)
    if profile == "us" and not bt_extra.get("universe_row_ids"):
        from momentum.etf.us_liquid_rrg_config import (
            DEFAULT_MIN_ADV,
            DEFAULT_VOL_PERCENTILE,
        )

        bt_extra.setdefault("universe_mode", "expanded")
        bt_extra.setdefault("min_adv_usd", DEFAULT_MIN_ADV)
        bt_extra.setdefault("vol_percentile", DEFAULT_VOL_PERCENTILE)
        bt_extra.setdefault("screen_categories", ("all",))

    win = tk.Toplevel(parent)
    win.title(prof.title)
    win.geometry("1100x780")
    win.minsize(900, 560)
    try:
        win.attributes("-toolwindow", False)
    except tk.TclError:
        pass
    win.lift()

    engine: Any | None = None
    _busy = False
    _load_run_all_after = False
    _overlay = RrgBusyOverlay(win)

    params = tk.Frame(win, padx=10, pady=8)
    params.pack(fill=tk.X)

    tk.Label(params, text="Start date:").grid(row=0, column=0, sticky="w")
    start_default = initial_start or f"{today_ist().year}-01-01"
    start_var = tk.StringVar(value=rrg_format_date(start_default))
    start_entry = tk.Entry(params, textvariable=start_var, width=12)
    start_entry.grid(row=0, column=1, padx=(4, 16))

    tk.Label(params, text="End date:").grid(row=0, column=2, sticky="w")
    end_var = tk.StringVar(
        value=rrg_format_date(initial_end) if initial_end else rrg_format_date(today_ist())
    )
    end_entry = tk.Entry(params, textvariable=end_var, width=12)
    end_entry.grid(row=0, column=3, padx=(4, 16))

    def _register_dd_mm_yyyy_entry(entry: tk.Entry, var: tk.StringVar) -> None:
        def _allow_char(proposed: str) -> bool:
            if proposed == "":
                return True
            if len(proposed) > 10:
                return False
            return all(c.isdigit() or c == "-" for c in proposed)

        entry.config(
            validate="key",
            validatecommand=(win.register(_allow_char), "%P"),
        )

        def _on_leave(_event=None) -> None:
            raw = var.get().strip()
            if not raw:
                return
            try:
                var.set(rrg_format_date(rrg_parse_user_date(raw)))
            except ValueError as exc:
                messagebox.showerror("Invalid date", str(exc), parent=win)
                entry.focus_set()
                entry.selection_range(0, tk.END)

        entry.bind("<FocusOut>", _on_leave)

    _register_dd_mm_yyyy_entry(start_entry, start_var)
    _register_dd_mm_yyyy_entry(end_entry, end_var)

    tk.Label(params, text="Top N:").grid(row=0, column=4, sticky="w")
    top_n_var = tk.IntVar(value=top_n)
    ttk.Spinbox(params, from_=1, to=15, width=4, textvariable=top_n_var).grid(
        row=0, column=5, padx=(4, 16)
    )

    tk.Label(params, text="RRG window:").grid(row=0, column=6, sticky="w")
    window_combo = ttk.Combobox(
        params, values=("10", "14"), width=4, state="readonly"
    )
    window_combo.set(str(rrg_window))
    window_combo.grid(row=0, column=7, padx=(4, 16))

    tk.Label(params, text="RRG tail (wks):").grid(row=0, column=8, sticky="w")
    tail_var = tk.IntVar(value=max(1, min(int(tail), RRG_MAX_TAIL)))
    ttk.Spinbox(
        params, from_=1, to=RRG_MAX_TAIL, width=4, textvariable=tail_var
    ).grid(row=0, column=9, padx=(4, 16))

    tk.Label(params, text="Capital:").grid(row=0, column=10, sticky="w")
    capital_var = tk.StringVar(value="100000")
    tk.Entry(params, textvariable=capital_var, width=10).grid(row=0, column=11, padx=4)

    pick_strategy_var = tk.StringVar()
    hold_until_rank_exit_var = tk.BooleanVar(value=False)
    exit_below_9ema_var = tk.BooleanVar(value=bool(bt_extra.get("exit_below_9ema", True)))
    _default_stop_loss_pct = 10.0 if profile == "stock" else 3.0
    exit_stop_loss_var = tk.BooleanVar(value=bool(bt_extra.get("exit_stop_loss", False)))
    stop_loss_pct_var = tk.DoubleVar(
        value=float(bt_extra.get("stop_loss_pct", _default_stop_loss_pct))
    )
    max_hold_rank_var = tk.IntVar(value=10)
    _pick_label_to_key: dict[str, str] = {}
    max_rank_label: tk.Label | None = None
    max_rank_spin: ttk.Spinbox | None = None
    us_universe_var = tk.StringVar()
    _us_universe_label_to_key: dict[str, str] = {}
    params_row = 1

    from momentum.rrg_portfolio_fill import (
        PORTFOLIO_FILL_MAINTAIN_TOP_N,
        PORTFOLIO_FILL_MODES,
    )

    _portfolio_fill_label_to_key = {
        label: key for key, label in PORTFOLIO_FILL_MODES.items()
    }
    portfolio_fill_var = tk.StringVar(
        value=PORTFOLIO_FILL_MODES[PORTFOLIO_FILL_MAINTAIN_TOP_N]
    )

    def _pack_portfolio_fill(parent: tk.Misc) -> None:
        tk.Label(parent, text="Portfolio fill:").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Combobox(
            parent,
            textvariable=portfolio_fill_var,
            values=list(PORTFOLIO_FILL_MODES.values()),
            width=36,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 0))

    if profile == "us":
        from momentum.etf.us_rrg_universe_modes import (
            US_UNIVERSE_DROPDOWN_VALUES,
            US_UNIVERSE_LABELS,
            normalize_us_universe_mode,
        )

        _us_universe_label_to_key = {
            label: key for key, label in US_UNIVERSE_LABELS.items()
        }
        init_mode = normalize_us_universe_mode(
            str(bt_extra.get("universe_mode", "expanded"))
        )
        us_universe_var.set(
            US_UNIVERSE_LABELS.get(init_mode, US_UNIVERSE_DROPDOWN_VALUES[0])
        )
        uni_row = tk.Frame(params)
        uni_row.grid(row=params_row, column=0, columnspan=12, sticky="w", pady=(6, 0))
        tk.Label(uni_row, text="ETF universe:").pack(side=tk.LEFT)
        ttk.Combobox(
            uni_row,
            textvariable=us_universe_var,
            values=list(US_UNIVERSE_DROPDOWN_VALUES),
            width=36,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 0))
        _pack_portfolio_fill(uni_row)
        params_row += 1

    if profile in ("india", "us", "stock"):
        if profile == "us":
            from momentum.etf.us_rrg_pick_strategies import PICK_STRATEGIES
        elif profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import PICK_STRATEGIES
        else:
            from momentum.etf.india_rrg_pick_strategies import PICK_STRATEGIES

        _pick_label_to_key = {label: key for key, label in PICK_STRATEGIES.items()}
        pick_strategy_var.set(PICK_STRATEGIES["recommend"])
        strat_row = tk.Frame(params)
        strat_row.grid(
            row=params_row, column=0, columnspan=12, sticky="w", pady=(6, 0)
        )
        params_row += 1
        tk.Label(strat_row, text="Strategy:").pack(side=tk.LEFT)
        ttk.Combobox(
            strat_row,
            textvariable=pick_strategy_var,
            values=list(PICK_STRATEGIES.values()),
            width=36,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 12))
        if profile != "us":
            _pack_portfolio_fill(strat_row)
        ttk.Checkbutton(
            strat_row, text="Hold until rank worse", variable=hold_until_rank_exit_var
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            strat_row, text="Exit below 9 EMA", variable=exit_below_9ema_var
        ).pack(side=tk.LEFT, padx=(0, 8))
        stop_loss_cb = ttk.Checkbutton(
            strat_row, text="Exit stop loss", variable=exit_stop_loss_var
        )
        stop_loss_cb.pack(side=tk.LEFT, padx=(0, 4))
        stop_loss_label = tk.Label(strat_row, text="Stop %:")
        stop_loss_spin = ttk.Spinbox(
            strat_row,
            from_=1,
            to=50,
            increment=0.5,
            width=5,
            textvariable=stop_loss_pct_var,
        )

        def _toggle_stop_loss_pct(*_) -> None:
            if exit_stop_loss_var.get():
                stop_loss_label.pack(side=tk.LEFT)
                stop_loss_spin.pack(side=tk.LEFT, padx=(4, 8))
            else:
                stop_loss_label.pack_forget()
                stop_loss_spin.pack_forget()

        exit_stop_loss_var.trace_add("write", _toggle_stop_loss_pct)
        _toggle_stop_loss_pct()
        max_rank_label = tk.Label(strat_row, text="Max hold rank:")
        max_rank_spin = ttk.Spinbox(
            strat_row, from_=5, to=60, width=4, textvariable=max_hold_rank_var
        )

        def _toggle_max_hold_rank(*_) -> None:
            if max_rank_label is None or max_rank_spin is None:
                return
            if hold_until_rank_exit_var.get():
                max_rank_label.pack(side=tk.LEFT)
                max_rank_spin.pack(side=tk.LEFT, padx=(4, 0))
            else:
                max_rank_label.pack_forget()
                max_rank_spin.pack_forget()

        hold_until_rank_exit_var.trace_add("write", _toggle_max_hold_rank)
        _toggle_max_hold_rank()
    else:
        portfolio_fill_var = tk.StringVar(value="")
        _portfolio_fill_label_to_key = {}

    if pick_strategy is not None and profile in ("india", "us", "stock"):
        if profile == "us":
            from momentum.etf.us_rrg_pick_strategies import PICK_STRATEGIES as _PS
        elif profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import PICK_STRATEGIES as _PS
        else:
            from momentum.etf.india_rrg_pick_strategies import PICK_STRATEGIES as _PS
        ps_key = pick_strategy
        if ps_key == "top_n_rank_exit":
            ps_key = "top_n"
            hold_until_rank_exit_var.set(True)
        if ps_key in _PS:
            pick_strategy_var.set(_PS[ps_key])
    if hold_until_rank_exit is not None:
        hold_until_rank_exit_var.set(bool(hold_until_rank_exit))
    if max_hold_rank is not None:
        max_hold_rank_var.set(int(max_hold_rank))
    if "exit_below_9ema" in bt_extra:
        exit_below_9ema_var.set(bool(bt_extra["exit_below_9ema"]))
    elif exit_below_9ema is not None:
        exit_below_9ema_var.set(bool(exit_below_9ema))
    if "exit_stop_loss" in bt_extra:
        exit_stop_loss_var.set(bool(bt_extra["exit_stop_loss"]))
    if "stop_loss_pct" in bt_extra:
        stop_loss_pct_var.set(float(bt_extra["stop_loss_pct"]))
    if profile in ("india", "us", "stock") and bt_extra.get("portfolio_fill_mode"):
        from momentum.rrg_portfolio_fill import PORTFOLIO_FILL_MODES as _PFM

        pf_key = str(bt_extra["portfolio_fill_mode"]).strip().lower()
        if pf_key in _PFM:
            portfolio_fill_var.set(_PFM[pf_key])

    mode_var = tk.StringVar(value="all")
    mode_row = tk.Frame(params)
    mode_row.grid(row=params_row, column=0, columnspan=12, sticky="w", pady=(8, 0))
    ttk.Radiobutton(mode_row, text="Week-by-week", variable=mode_var, value="step").pack(
        side=tk.LEFT, padx=(0, 12)
    )
    ttk.Radiobutton(mode_row, text="Run all after load", variable=mode_var, value="all").pack(
        side=tk.LEFT, padx=(0, 12)
    )
    show_equity_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(mode_row, text="Show equity curve", variable=show_equity_var).pack(
        side=tk.LEFT
    )

    btn_row = tk.Frame(win, pady=6)
    btn_row.pack(fill=tk.X, padx=8)

    status_var = tk.StringVar(
        value="Set Start date and End date, then Load Data."
    )
    tk.Label(win, textvariable=status_var, anchor="w", fg="#333", padx=10).pack(fill=tk.X)

    body = ttk.PanedWindow(win, orient=tk.VERTICAL)
    body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    top_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(top_pane, weight=3)

    chart_frame = tk.Frame(top_pane)
    top_pane.add(chart_frame, weight=3)
    fig = Figure(figsize=(7, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Portfolio vs {prof.bench_chart}")
    ax.set_xlabel("Week #")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    metrics_frame = tk.LabelFrame(top_pane, text="Summary metrics", padx=4, pady=4)
    top_pane.add(metrics_frame, weight=1)
    metrics_tree = _make_kv_tree(
        metrics_frame,
        height=14,
        col0_width=150,
        col1_width=180,
    )

    bottom_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(bottom_pane, weight=2)

    detail_frame = tk.LabelFrame(bottom_pane, text="Selected week", padx=4, pady=4)
    bottom_pane.add(detail_frame, weight=1)
    detail_pane = ttk.PanedWindow(detail_frame, orient=tk.VERTICAL)
    detail_pane.pack(fill=tk.BOTH, expand=True)

    summary_frame = tk.Frame(detail_pane)
    detail_pane.add(summary_frame, weight=1)
    detail_tree = _make_kv_tree(
        summary_frame,
        height=6,
        col0_width=130,
        col1_width=280,
    )

    pos_frame = tk.LabelFrame(detail_pane, text="ETF prices & P/L", padx=4, pady=4)
    detail_pane.add(pos_frame, weight=2)
    pos_cols = ("ETF", "Status", "Entry", "Exit", "Exit date", "Running", "P/L %")
    pos_tree = ttk.Treeview(
        pos_frame, columns=pos_cols, show="headings", height=8, selectmode="none"
    )
    for col in pos_cols:
        pos_tree.heading(col, text=col)
        w = (
            120
            if col == "Status"
            else 84
            if col in ("Entry", "Exit", "Running", "P/L %", "Exit date")
            else 80
        )
        pos_tree.column(col, width=w, stretch=True)
    pos_scroll = ttk.Scrollbar(pos_frame, orient=tk.VERTICAL, command=pos_tree.yview)
    pos_tree.configure(yscrollcommand=pos_scroll.set)
    pos_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    pos_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    log_frame = tk.LabelFrame(bottom_pane, text="Rebalance log", padx=4, pady=4)
    bottom_pane.add(log_frame, weight=3)
    log_cols = (
        "Week",
        "Rebal",
        "End",
        "Holdings",
        "Rebal_9EMA",
        "Mid_9EMA",
        "Mid_SL",
        "Exits",
        "Port%",
        "Bench%",
        "Drawdown%",
        "Value",
    )
    log_tree = ttk.Treeview(
        log_frame, columns=log_cols, show="headings", height=10, selectmode="browse"
    )
    _log_heading = {
        "Rebal_9EMA": "9EMA @ reb",
        "Mid_9EMA": "9EMA mid-out",
        "Mid_SL": "SL mid-out",
        "Port%": "Port%",
        "Bench%": "Bench%",
        "Drawdown%": "Drawdown %",
    }
    for col in log_cols:
        log_tree.heading(col, text=_log_heading.get(col, col))
        w = (
            200
            if col == "Exits"
            else 160
            if col == "Holdings"
            else 92
            if col == "Drawdown%"
            else 88
        )
        log_tree.column(col, width=w, stretch=True)
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_tree.yview)
    log_tree.configure(yscrollcommand=log_scroll.set)
    log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_busy(busy: bool, message: str | None = None) -> None:
        nonlocal _busy
        _busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        load_btn.config(state=state)
        next_btn.config(state=state)
        prev_btn.config(state=state)
        run_all_btn.config(state=state)
        reset_btn.config(state=state)
        if busy:
            _overlay.show(message or status_var.get())
        else:
            _overlay.hide()
        if not busy:
            _refresh_nav_buttons()

    def _refresh_nav_buttons() -> None:
        can_prev = (
            not _busy
            and engine is not None
            and engine.loaded
            and engine.current_week > 0
        )
        prev_btn.config(state=tk.NORMAL if can_prev else tk.DISABLED)
        can_next = (
            not _busy
            and engine is not None
            and engine.loaded
            and not engine.finished
        )
        next_btn.config(state=tk.NORMAL if can_next else tk.DISABLED)

    def _update_metrics() -> None:
        if engine is None or engine.trades_df.empty:
            _fill_kv_tree(metrics_tree, None, empty="No results yet.")
            return
        df = engine.trades_df
        df.attrs["top_n"] = int(top_n_var.get())
        if profile == "us" and engine is not None:
            df.attrs["universe_mode"] = getattr(
                engine.config, "universe_mode", "expanded"
            )
            df.attrs["universe_size"] = len(getattr(engine, "_row_ids", ()))
        if profile in ("india", "us", "stock") and _pick_label_to_key:
            df.attrs["pick_strategy"] = _pick_label_to_key.get(
                pick_strategy_var.get(), "recommend"
            )
            df.attrs["hold_until_rank_exit"] = bool(hold_until_rank_exit_var.get())
            df.attrs["max_hold_rank"] = int(max_hold_rank_var.get())
            df.attrs["exit_below_9ema"] = bool(exit_below_9ema_var.get())
            df.attrs["exit_stop_loss"] = bool(exit_stop_loss_var.get())
            df.attrs["stop_loss_pct"] = _parse_stop_loss_pct()
            df.attrs["portfolio_fill_mode"] = _portfolio_fill_label_to_key.get(
                portfolio_fill_var.get(), "maintain_top_n"
            )
        cap = float(capital_var.get())
        if profile == "us" and hasattr(engine, "_benchmark"):
            m = prof.compute_metrics(df, cap, benchmark=engine._benchmark)
        else:
            m = prof.compute_metrics(df, cap)
        _fill_kv_tree(
            metrics_tree,
            {str(k): str(v) for k, v in m.items()},
            empty="No results yet.",
        )

    def _chart_in_top_pane() -> bool:
        try:
            return str(chart_frame) in top_pane.panes()
        except tk.TclError:
            return False

    def _set_equity_curve_visible(visible: bool) -> None:
        if visible:
            if not _chart_in_top_pane():
                try:
                    top_pane.add(chart_frame, weight=3)
                except tk.TclError:
                    pass
        elif _chart_in_top_pane():
            top_pane.forget(chart_frame)
        win.update_idletasks()

    def _on_equity_toggle(*_) -> None:
        _set_equity_curve_visible(bool(show_equity_var.get()))
        if show_equity_var.get():
            _update_chart()

    show_equity_var.trace_add("write", _on_equity_toggle)
    _set_equity_curve_visible(bool(show_equity_var.get()))

    def _update_chart() -> None:
        if not show_equity_var.get():
            return
        ax.clear()
        if engine is None or engine.trades_df.empty:
            ax.set_title(f"Portfolio vs {prof.bench_chart}")
            canvas.draw_idle()
            return
        df = engine.trades_df
        cap = float(capital_var.get())
        port_vals = df["Portfolio_Value"].values
        bench_vals = cap * (1 + df["Bench_Return"]).cumprod()
        weeks = range(1, len(df) + 1)
        ax.plot(weeks, port_vals, label="RRG portfolio", color="#1565C0", linewidth=2)
        ax.plot(weeks, bench_vals, label=prof.bench_chart, color="#757575", linewidth=1.5)
        if engine.current_week < engine.total_weeks:
            ax.axvline(engine.current_week, color="#E65100", linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Equity curve ({len(df)} week(s))")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        canvas.draw_idle()

    def _show_week_detail(record: dict | None) -> None:
        if record is None or engine is None:
            _fill_kv_tree(
                detail_tree,
                None,
                empty="Load data, then Next Week or Run All.",
            )
            _fill_position_tree(pos_tree, None)
            return
        cap = float(capital_var.get())
        _fill_kv_tree(
            detail_tree,
            _week_detail_fields(
                record,
                total_weeks=engine.total_weeks,
                initial_capital=cap,
            ),
            empty="—",
        )
        _fill_position_tree(
            pos_tree,
            record.get("Position_Rows") or [],
            strategy_order=list(record.get("Strategy_Tickers") or []),
        )

    def _refresh_log_table() -> None:
        from momentum.rrg_portfolio_exits import format_exit_summary

        for item in log_tree.get_children():
            log_tree.delete(item)
        if engine is None or engine.trades_df.empty:
            return
        dd_vals = backtest_drawdown_pct_series(engine.trades_df["Port_Return"]).values
        for pos, (_, row) in enumerate(engine.trades_df.iterrows()):
            rebal_tickers = row.get("Rebalance_Tickers")
            exits = row.get("Exits") or []
            exit_text = format_exit_summary(
                exits,
                rebalance_tickers=list(rebal_tickers) if rebal_tickers else None,
            )
            dd_pct = float(dd_vals[pos]) if pos < len(dd_vals) else 0.0
            log_tree.insert(
                "",
                tk.END,
                iid=str(row["Week"]),
                values=(
                    row["Week"],
                    rrg_format_date(row["Rebal_Date"]),
                    rrg_format_date(row["End_Date"]),
                    row["Holdings"],
                    row.get("Rebal_9EMA_Label") or "—",
                    row.get("Mid_Week_9EMA_Label") or "—",
                    row.get("Mid_Week_Stop_Loss_Label") or "—",
                    exit_text or "—",
                    f"{row['Port_Return'] * 100:+.2f}",
                    f"{row['Bench_Return'] * 100:+.2f}",
                    f"{dd_pct:+.2f}",
                    f"{row['Portfolio_Value']:,.0f}",
                ),
            )
        log_tree.yview_moveto(1.0)

    def _clear_results() -> None:
        _refresh_log_table()
        _show_week_detail(None)
        _update_chart()
        _update_metrics()
        _refresh_nav_buttons()

    def _validate_dates() -> bool:
        try:
            start_norm = rrg_format_date(rrg_parse_user_date(start_var.get()))
            end_norm = rrg_format_date(rrg_parse_user_date(end_var.get()))
            start_var.set(start_norm)
            end_var.set(end_norm)
            if pd.Timestamp(rrg_config_date_str(start_norm)) > pd.Timestamp(
                rrg_config_date_str(end_norm)
            ):
                messagebox.showerror(
                    "Invalid dates",
                    "Start date must be on or before End date (DD-MM-YYYY).",
                    parent=win,
                )
                return False
            return True
        except ValueError as exc:
            messagebox.showerror("Invalid date", str(exc), parent=win)
            return False

    def _ui_config():
        kw = dict(
            backtest_start=rrg_config_date_str(start_var.get()),
            backtest_end=rrg_config_date_str(end_var.get()),
            top_n=int(top_n_var.get()),
            tail=int(tail_var.get()),
            rrg_window=int(window_combo.get()),
            initial_capital=float(capital_var.get()),
        )
        if profile in ("india", "us", "stock") and _pick_label_to_key:
            kw["pick_strategy"] = _pick_label_to_key.get(
                pick_strategy_var.get(), "recommend"
            )
            kw["hold_until_rank_exit"] = bool(hold_until_rank_exit_var.get())
            kw["max_hold_rank"] = int(max_hold_rank_var.get())
            kw["exit_below_9ema"] = bool(exit_below_9ema_var.get())
            kw["exit_stop_loss"] = bool(exit_stop_loss_var.get())
            kw["stop_loss_pct"] = _parse_stop_loss_pct()
            kw["portfolio_fill_mode"] = _portfolio_fill_label_to_key.get(
                portfolio_fill_var.get(), "maintain_top_n"
            )
            if profile in ("india", "stock", "us"):
                kw["analysis_period"] = bt_extra.get("analysis_period", "3m")
        if profile == "stock":
            kw["universe_key"] = bt_extra.get("universe_key", "quality")
        if profile == "us":
            row_ids = bt_extra.get("universe_row_ids")
            extra_us = {
                k: v
                for k, v in bt_extra.items()
                if k
                in (
                    "min_adv_usd",
                    "vol_percentile",
                    "screen_categories",
                )
            }
            if row_ids:
                kw["universe_row_ids"] = tuple(row_ids)
                kw["universe_mode"] = "main_table"
            else:
                kw.update(extra_us)
                kw["universe_mode"] = _us_universe_label_to_key.get(
                    us_universe_var.get(), "expanded"
                )
                kw.setdefault("min_adv_usd", bt_extra.get("min_adv_usd", 10_000_000.0))
                kw.setdefault("vol_percentile", bt_extra.get("vol_percentile", 100.0))
                kw.setdefault(
                    "screen_categories", bt_extra.get("screen_categories", ("all",))
                )
        return prof.Config(**kw)

    def _parse_stop_loss_pct() -> float:
        default_pct = 10.0 if profile == "stock" else 3.0
        try:
            pct = float(stop_loss_pct_var.get())
        except (TypeError, ValueError, tk.TclError):
            pct = default_pct
        return max(0.1, min(pct, 50.0))

    def _config_needs_reload(stored, ui) -> bool:
        base = (
            stored.backtest_start != ui.backtest_start
            or stored.backtest_end != ui.backtest_end
            or stored.tail != ui.tail
            or stored.rrg_window != ui.rrg_window
        )
        if profile in ("india", "us", "stock"):
            base = base or stored.analysis_period != ui.analysis_period
        if profile == "stock":
            base = base or stored.universe_key != ui.universe_key
        if profile == "us":
            base = base or stored.universe_mode != ui.universe_mode
            base = base or (
                tuple(getattr(stored, "universe_row_ids", None) or ())
                != tuple(getattr(ui, "universe_row_ids", None) or ())
            )
        return base

    def _simulation_settings_changed(stored, ui) -> bool:
        """Pick / exit / fill params that affect weekly simulation (no data reload)."""
        keys = (
            "top_n",
            "pick_strategy",
            "hold_until_rank_exit",
            "max_hold_rank",
            "exit_below_9ema",
            "exit_stop_loss",
            "stop_loss_pct",
            "portfolio_fill_mode",
        )
        for key in keys:
            if getattr(stored, key, None) != getattr(ui, key, None):
                return True
        return False

    def _exit_rules_summary(cfg) -> str:
        rules: list[str] = []
        if getattr(cfg, "exit_below_9ema", False):
            rules.append("9 EMA")
        if getattr(cfg, "exit_stop_loss", False):
            rules.append(f"stop loss {getattr(cfg, 'stop_loss_pct', _default_stop_loss_pct):g}%")
        return ", ".join(rules) if rules else "none"

    def _on_progress(msg: str) -> None:
        status_var.set(msg)
        _overlay.update_message(msg)

    def _build_engine():
        return prof.Engine(
            config=_ui_config(),
            progress_cb=lambda msg: win.after(0, lambda m=msg: _on_progress(m)),
        )

    def _on_load_done(eng, err: Exception | None) -> None:
        nonlocal engine, _load_run_all_after
        _set_busy(False)
        if err is not None:
            messagebox.showerror("Load failed", str(err), parent=win)
            status_var.set(f"Load failed: {err}")
            _load_run_all_after = False
            return
        engine = eng
        _clear_results()
        first = rrg_format_date(engine.rebal_dates[0])
        last = rrg_format_date(engine.rebal_dates[-1])
        status_var.set(
            f"Ready: {engine.total_weeks} weekly rebalance(s) from {first} to {last}. "
            f"Universe: {getattr(engine.config, 'universe_mode', 'n/a')} "
            f"({len(getattr(engine, '_row_ids', ()))} ETFs). "
            f"Click Next Week or Run All."
        )
        run_after = _load_run_all_after or mode_var.get() == "all"
        _load_run_all_after = False
        if run_after:
            _do_run_all(from_load=True)

    def _load_data(*, run_all_after: bool = False) -> None:
        nonlocal _load_run_all_after
        if _busy:
            return
        if not _validate_dates():
            return
        _load_run_all_after = run_all_after
        status_var.set(prof.load_status)
        _set_busy(True, prof.load_status)

        def worker():
            err = None
            eng = None
            try:
                eng = _build_engine()
                eng.load_data()
            except Exception as exc:
                err = exc
            win.after(0, lambda: _on_load_done(eng, err))

        threading.Thread(target=worker, daemon=True).start()

    def _do_step() -> None:
        if engine is None or not engine.loaded:
            messagebox.showinfo("Backtest", "Load data first.", parent=win)
            return
        if not _validate_dates():
            return
        ui_cfg = _ui_config()
        if _config_needs_reload(engine.config, ui_cfg):
            messagebox.showinfo(
                "Backtest",
                "Settings changed — click Load Data again.",
                parent=win,
            )
            return
        prior_cfg = engine.config
        engine.config = ui_cfg
        if engine.finished:
            if _simulation_settings_changed(prior_cfg, ui_cfg):
                engine.reset_run()
                _clear_results()
            else:
                status_var.set("All weeks done. Change exit rules or Reset, then run again.")
                return
        record = engine.step_week()
        if record:
            _refresh_log_table()
            _show_week_detail(record)
            log_tree.selection_set(str(record["Week"]))
            _update_chart()
            _update_metrics()
            status_var.set(f"Week {engine.current_week}/{engine.total_weeks} complete.")
        _refresh_nav_buttons()

    def _do_previous() -> None:
        if engine is None or not engine.loaded:
            return
        if engine.current_week <= 0:
            status_var.set("Already at the first week.")
            return
        record = engine.step_back()
        _refresh_log_table()
        if record:
            _show_week_detail(record)
            log_tree.selection_set(str(record["Week"]))
            status_var.set(f"Showing week {engine.current_week}/{engine.total_weeks}.")
        else:
            _show_week_detail(None)
            status_var.set("At start — click Next Week.")
        _update_chart()
        _update_metrics()
        _refresh_nav_buttons()

    def _do_run_all(*, from_load: bool = False) -> None:
        if _busy and not from_load:
            return
        if engine is None or not engine.loaded:
            _load_data(run_all_after=True)
            return
        if not _validate_dates():
            return
        ui_cfg = _ui_config()
        if _config_needs_reload(engine.config, ui_cfg):
            _load_data(run_all_after=True)
            return
        # Always apply current UI exit/pick settings before simulating.
        engine.config = ui_cfg
        engine.reset_run()
        _clear_results()
        exit_rules = _exit_rules_summary(ui_cfg)
        run_msg = f"Running all weeks (exits: {exit_rules})…"
        status_var.set(run_msg)
        _set_busy(True, run_msg)

        def worker():
            err = None
            last_record = None
            try:
                eng = engine
                if eng is None:
                    return
                eng.config = ui_cfg
                eng.reset_run()
                while not eng.finished:
                    last_record = eng.step_week()
                    w, t = eng.current_week, eng.total_weeks
                    win.after(
                        0,
                        lambda w=w, t=t: _on_progress(f"Running week {w}/{t}…"),
                    )
            except Exception as exc:
                err = exc
            win.after(0, lambda: _on_run_all_done(err, last_record, ui_cfg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_all_done(
        err: Exception | None,
        last_record: dict | None,
        ui_cfg=None,
    ) -> None:
        _set_busy(False)
        if err is not None:
            messagebox.showerror("Run failed", str(err), parent=win)
            return
        _refresh_log_table()
        if last_record:
            _show_week_detail(last_record)
            log_tree.selection_set(str(last_record["Week"]))
        _update_chart()
        _update_metrics()
        total = engine.total_weeks if engine else 0
        cfg = ui_cfg if ui_cfg is not None else getattr(engine, "config", None)
        exit_rules = _exit_rules_summary(cfg) if cfg is not None else "n/a"
        sl_exits = 0
        if engine is not None and engine._records and "Mid_Week_Stop_Loss" in engine._records[0]:
            sl_exits = sum(len(r.get("Mid_Week_Stop_Loss") or []) for r in engine._records)
        status_var.set(
            f"Finished all {total} week(s). Exits: {exit_rules}. "
            f"Mid-week stop-loss exits: {sl_exits}."
        )
        _refresh_nav_buttons()

    def _reset() -> None:
        if engine is not None and engine.loaded:
            engine.reset_run()
        _clear_results()
        if engine is not None and engine.loaded:
            first = rrg_format_date(engine.rebal_dates[0])
            last = rrg_format_date(engine.rebal_dates[-1])
            status_var.set(
                f"Reset. {engine.total_weeks} week(s) ({first} → {last}). "
                f"Click Next Week or Run All."
            )
        else:
            status_var.set("Reset. Load data to begin.")

    def _on_log_select(_event=None) -> None:
        if engine is None or not engine._records:
            return
        sel = log_tree.selection()
        if not sel:
            return
        try:
            week_num = int(sel[0])
        except ValueError:
            return
        if 1 <= week_num <= len(engine._records):
            _show_week_detail(engine._records[week_num - 1])

    log_tree.bind("<<TreeviewSelect>>", _on_log_select)

    load_btn = ttk.Button(btn_row, text="Load Data", command=_load_data)
    load_btn.pack(side=tk.LEFT, padx=(10, 6))
    prev_btn = ttk.Button(btn_row, text="Previous Week", command=_do_previous)
    prev_btn.pack(side=tk.LEFT, padx=6)
    next_btn = ttk.Button(btn_row, text="Next Week", command=_do_step)
    next_btn.pack(side=tk.LEFT, padx=6)
    run_all_btn = ttk.Button(btn_row, text="Run All", command=_do_run_all)
    run_all_btn.pack(side=tk.LEFT, padx=6)
    reset_btn = ttk.Button(btn_row, text="Reset", command=_reset)
    reset_btn.pack(side=tk.LEFT, padx=6)

    def _close_window() -> None:
        win.destroy()
        if shutdown_root is not None:
            try:
                shutdown_root.quit()
            except tk.TclError:
                pass

    ttk.Button(btn_row, text="Close", command=_close_window).pack(side=tk.RIGHT, padx=10)
    win.protocol("WM_DELETE_WINDOW", _close_window)

    install_copy_support(win)
    _fill_kv_tree(metrics_tree, None, empty="No results yet.")
    _show_week_detail(None)
    _refresh_nav_buttons()
    win.focus_set()
    return win


def open_stock_rrg_backtest(
    parent: tk.Misc,
    *,
    rrg_window: int = 10,
    tail: int = 1,
    top_n: int = 7,
    universe_key: str = "quality",
) -> tk.Toplevel:
    return open_rrg_backtest(
        parent,
        profile="stock",
        rrg_window=rrg_window,
        tail=tail,
        top_n=top_n,
        backtest_extra={"universe_key": universe_key},
    )


def open_india_rrg_backtest(
    parent: tk.Misc,
    *,
    rrg_window: int = 10,
    tail: int = 1,
    top_n: int = 7,
) -> tk.Toplevel:
    return open_rrg_backtest(
        parent,
        profile="india",
        rrg_window=rrg_window,
        tail=tail,
        top_n=top_n,
    )


def open_us_rrg_backtest(
    parent: tk.Misc,
    *,
    rrg_window: int = 10,
    tail: int = 1,
    analysis_period: str = "3m",
    top_n: int = 7,
) -> tk.Toplevel:
    return open_rrg_backtest(
        parent,
        profile="us",
        rrg_window=rrg_window,
        tail=tail,
        analysis_period=analysis_period,
        top_n=top_n,
    )


def launch_standalone_rrg_backtest(
    *,
    profile: str = "india",
    rrg_window: int = 10,
    tail: int | None = None,
    top_n: int = 7,
    start: str | None = None,
    end: str | None = None,
    backtest_extra: dict[str, Any] | None = None,
    ready_file: str | None = None,
) -> None:
    """Open backtest UI in its own process (no main RRG window)."""
    from pathlib import Path

    root = tk.Tk()
    root.withdraw()
    if tail is None:
        tail = 1
    us_extra = None
    if profile == "us":
        from momentum.etf.us_rrg_universe_modes import (
            US_UNIVERSE_LABELS,
            normalize_us_universe_mode,
        )
        from momentum.etf.us_liquid_rrg_config import (
            DEFAULT_MIN_ADV,
            DEFAULT_VOL_PERCENTILE,
        )

        init_mode = "expanded"
        if backtest_extra and backtest_extra.get("universe_mode"):
            init_mode = normalize_us_universe_mode(str(backtest_extra["universe_mode"]))
        us_extra = {
            "universe_mode": init_mode,
            "min_adv_usd": (backtest_extra or {}).get("min_adv_usd", DEFAULT_MIN_ADV),
            "vol_percentile": (backtest_extra or {}).get(
                "vol_percentile", DEFAULT_VOL_PERCENTILE
            ),
            "screen_categories": (backtest_extra or {}).get(
                "screen_categories", ("all",)
            ),
            "analysis_period": "3m",
        }
    stock_extra = None
    if profile == "stock":
        from momentum.stock.universes import DEFAULT_KEY

        stock_extra = {
            "universe_key": DEFAULT_KEY,
            "analysis_period": "3m",
        }
    win = open_rrg_backtest(
        root,
        profile=profile,
        rrg_window=rrg_window,
        tail=tail,
        top_n=top_n,
        initial_start=start,
        initial_end=end,
        backtest_extra=backtest_extra or us_extra or stock_extra,
        shutdown_root=root,
    )

    if ready_file:
        ready_path = Path(ready_file)

        def _write_ready() -> None:
            try:
                ready_path.write_text("ready\n", encoding="utf-8")
            except OSError:
                pass

        def _poll_ready() -> None:
            try:
                if win.winfo_exists() and win.winfo_viewable():
                    _write_ready()
                    return
            except tk.TclError:
                return
            win.after(50, _poll_ready)

        win.after_idle(_poll_ready)

    root.mainloop()
