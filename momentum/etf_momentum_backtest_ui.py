"""Tkinter backtest UI for India ETF momentum rankers."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.etf.backtest_etf_momentum import (  # noqa: E402
    REBALANCE_ALIASES,
    REBALANCE_LABELS,
    STRATEGY_KEYS,
    STRATEGY_LABELS,
    EtfMomentumBacktestConfig,
    EtfMomentumBacktestEngine,
    build_config_from_ui,
    compute_metrics,
    strategy_defaults,
)
from momentum.rrg_backtest_positions import (  # noqa: E402
    format_exit_date,
    format_pl_pct,
    format_price,
)
from momentum.rrg_backtest_ui import backtest_cum_pl_pct, backtest_drawdown_pct_series  # noqa: E402
from momentum.rrg_core import rrg_config_date_str, rrg_format_date, rrg_parse_user_date  # noqa: E402
from momentum.rrg_ui_copy import install_copy_support  # noqa: E402
from utils.nse_bhavcopy import today_ist  # noqa: E402


def _format_quantity(value) -> str:
    if value is None:
        return "—"
    return str(int(value))


def _format_alloc_pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}%"


def _format_allocated(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}"


def _format_portfolio_value(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}"


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


def _fill_kv_tree(tree: ttk.Treeview, rows: dict[str, str] | None, *, empty: str) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if not rows:
        tree.insert("", tk.END, values=("—", empty))
        return
    for key, value in rows.items():
        tree.insert("", tk.END, values=(key, value))


def _fill_pick_tree(tree: ttk.Treeview, rows: list[dict] | None) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if not rows:
        tree.insert("", tk.END, values=("—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"))
        return
    for row in rows:
        status = row.get("status") or "—"
        if status == "Exit":
            date_s = format_exit_date(row.get("exit_date"))
            mark_or_exit = format_price(row.get("exit"))
        elif status == "Skipped":
            date_s = "—"
            mark_or_exit = "—"
        else:
            date_s = "Open"
            mark_or_exit = format_price(row.get("exit"))
        alloc_pct = row.get("alloc_pct")
        alloc_amt = row.get("alloc_amount")
        tree.insert(
            "",
            tk.END,
            values=(
                row.get("ticker") or "—",
                status,
                format_price(row.get("entry")),
                mark_or_exit,
                date_s,
                _format_alloc_pct(alloc_pct),
                _format_quantity(row.get("alloc_qty")),
                _format_allocated(alloc_amt),
                format_pl_pct(row.get("period_pl_pct")),
                format_pl_pct(row.get("pl_pct")),
                row.get("exit_reason") or "—",
            ),
        )


def _strategy_key_from_label(label: str) -> str:
    for key, text in STRATEGY_LABELS.items():
        if text == label:
            return key
    return STRATEGY_KEYS[0]


def _apply_strategy_defaults(
    strategy_var: tk.StringVar,
    portfolio_var: tk.IntVar,
    entry_rank_var: tk.IntVar,
    exit_rank_var: tk.IntVar,
    benchmark_var: tk.StringVar,
    proximity_var: tk.StringVar,
    rebalance_var: tk.StringVar,
) -> None:
    key = _strategy_key_from_label(strategy_var.get())
    defaults = strategy_defaults(key)
    ps = int(defaults["portfolio_size"])
    portfolio_var.set(ps)
    entry_rank_var.set(max(int(defaults["entry_rank_depth"]), ps))
    exit_rank_var.set(defaults["exit_rank_threshold"])
    benchmark_var.set(defaults["benchmark_ticker"])
    proximity_var.set(f"{defaults['proximity_of_52w_high']:.2f}")
    rebalance_var.set(REBALANCE_LABELS.get(defaults["rebalance_period"], "Weekly"))


def open_etf_momentum_backtest(
    parent: tk.Misc,
    *,
    initial_strategy: str = "momentum_rs_etfs",
    initial_start: str | None = None,
    initial_end: str | None = None,
) -> tk.Toplevel:
    if initial_strategy not in STRATEGY_KEYS:
        initial_strategy = STRATEGY_KEYS[0]

    win = tk.Toplevel(parent)
    win.title("ETF Momentum Backtest")
    win.geometry("1140x820")
    win.minsize(940, 600)
    win.lift()

    engine: EtfMomentumBacktestEngine | None = None
    _busy = False
    _load_run_all_after = False

    params = tk.Frame(win, padx=10, pady=6)
    params.pack(fill=tk.X)

    _pad = dict(padx=(4, 14), sticky="w")

    tk.Label(params, text="Strategy:").grid(row=0, column=0, sticky="w")
    strategy_var = tk.StringVar(value=STRATEGY_LABELS[initial_strategy])
    strategy_combo = ttk.Combobox(
        params,
        textvariable=strategy_var,
        values=[STRATEGY_LABELS[k] for k in STRATEGY_KEYS],
        width=18,
        state="readonly",
    )
    strategy_combo.grid(row=0, column=1, **_pad)

    tk.Label(params, text="Start:").grid(row=0, column=2, sticky="w")
    start_default = initial_start or f"{today_ist().year}-01-01"
    start_var = tk.StringVar(value=rrg_format_date(start_default))
    start_entry = tk.Entry(params, textvariable=start_var, width=11)
    start_entry.grid(row=0, column=3, **_pad)

    tk.Label(params, text="End:").grid(row=0, column=4, sticky="w")
    end_var = tk.StringVar(
        value=rrg_format_date(initial_end) if initial_end else rrg_format_date(today_ist())
    )
    end_entry = tk.Entry(params, textvariable=end_var, width=11)
    end_entry.grid(row=0, column=5, **_pad)

    tk.Label(params, text="Rebalance:").grid(row=0, column=6, sticky="w")
    rebalance_var = tk.StringVar(value="Weekly")
    ttk.Combobox(
        params,
        textvariable=rebalance_var,
        values=tuple(REBALANCE_LABELS.values()),
        width=10,
        state="readonly",
    ).grid(row=0, column=7, **_pad)

    tk.Label(params, text="Portfolio size:").grid(row=0, column=8, sticky="w")
    portfolio_var = tk.IntVar(value=5)
    portfolio_spin = ttk.Spinbox(
        params, from_=1, to=30, width=4, textvariable=portfolio_var
    )
    portfolio_spin.grid(row=0, column=9, **_pad)

    tk.Label(params, text="Entry rank:").grid(row=0, column=10, sticky="w")
    entry_rank_var = tk.IntVar(value=5)
    entry_rank_spin = ttk.Spinbox(
        params, from_=5, to=50, width=4, textvariable=entry_rank_var
    )
    entry_rank_spin.grid(row=0, column=11, padx=(4, 14), sticky="w")

    def _sync_entry_rank_floor(*_args: object) -> None:
        ps = max(int(portfolio_var.get()), 1)
        entry_rank_spin.config(from_=ps)
        if int(entry_rank_var.get()) < ps:
            entry_rank_var.set(ps)

    portfolio_var.trace_add("write", _sync_entry_rank_floor)

    exit_rank_enabled_var = tk.BooleanVar(value=False)
    exit_rank_cb = ttk.Checkbutton(params, text="Exit rank", variable=exit_rank_enabled_var)
    exit_rank_cb.grid(row=0, column=12, padx=(0, 2), sticky="w")
    exit_rank_var = tk.IntVar(value=10)
    exit_rank_spin = ttk.Spinbox(
        params, from_=1, to=50, width=4, textvariable=exit_rank_var
    )
    exit_rank_spin.grid(row=0, column=13, padx=(4, 0), sticky="w")

    def _toggle_exit_rank_spin() -> None:
        exit_rank_spin.config(
            state=tk.NORMAL if exit_rank_enabled_var.get() else tk.DISABLED
        )

    exit_rank_cb.config(command=_toggle_exit_rank_spin)
    _toggle_exit_rank_spin()

    def _register_date_entry(entry: tk.Entry, var: tk.StringVar) -> None:
        def _allow_char(proposed: str) -> bool:
            if proposed == "":
                return True
            if len(proposed) > 10:
                return False
            return all(c.isdigit() or c == "-" for c in proposed)

        entry.config(validate="key", validatecommand=(win.register(_allow_char), "%P"))

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

    _register_date_entry(start_entry, start_var)
    _register_date_entry(end_entry, end_var)

    tk.Label(params, text="Benchmark:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    benchmark_var = tk.StringVar(value="^CRSLDX")
    tk.Entry(params, textvariable=benchmark_var, width=11).grid(
        row=1, column=1, pady=(6, 0), **_pad
    )

    tk.Label(params, text="52w prox:").grid(row=1, column=2, sticky="w", pady=(6, 0))
    proximity_var = tk.StringVar(value="0.70")
    tk.Entry(params, textvariable=proximity_var, width=6).grid(
        row=1, column=3, pady=(6, 0), **_pad
    )

    tk.Label(params, text="Capital:").grid(row=1, column=4, sticky="w", pady=(6, 0))
    capital_var = tk.StringVar(value="100000")
    tk.Entry(params, textvariable=capital_var, width=10).grid(
        row=1, column=5, pady=(6, 0), **_pad
    )

    exit_stop_loss_var = tk.BooleanVar(value=False)
    stop_loss_pct_var = tk.DoubleVar(value=5.0)
    stop_loss_cb = ttk.Checkbutton(params, text="Stop loss", variable=exit_stop_loss_var)
    stop_loss_cb.grid(row=1, column=6, padx=(0, 4), pady=(6, 0), sticky="w")
    tk.Label(params, text="%:").grid(row=1, column=7, sticky="w", pady=(6, 0))
    stop_loss_spin = ttk.Spinbox(
        params,
        from_=0.5,
        to=50.0,
        increment=0.5,
        width=5,
        textvariable=stop_loss_pct_var,
    )
    stop_loss_spin.grid(row=1, column=8, pady=(6, 0), padx=(4, 14), sticky="w")

    def _toggle_stop_loss_spin() -> None:
        stop_loss_spin.config(
            state=tk.NORMAL if exit_stop_loss_var.get() else tk.DISABLED
        )

    stop_loss_cb.config(command=_toggle_stop_loss_spin)
    _toggle_stop_loss_spin()

    park_empty_slots_var = tk.BooleanVar(value=True)
    park_slot_var = tk.StringVar(value="LIQUIDCASE")
    park_cb = ttk.Checkbutton(
        params, text="Park empty slots", variable=park_empty_slots_var
    )
    park_cb.grid(row=1, column=9, padx=(0, 4), pady=(6, 0), sticky="w")
    tk.Label(params, text="Park:").grid(row=1, column=10, sticky="w", pady=(6, 0))
    park_combo = ttk.Combobox(
        params,
        textvariable=park_slot_var,
        values=("LIQUIDCASE", "LTGILTBEES"),
        width=11,
        state="readonly",
    )
    park_combo.grid(row=1, column=11, padx=(4, 0), pady=(6, 0), sticky="w")

    def _toggle_park_combo() -> None:
        park_combo.config(
            state="readonly" if park_empty_slots_var.get() else tk.DISABLED
        )

    park_cb.config(command=_toggle_park_combo)
    _toggle_park_combo()

    exit_below_9ema_var = tk.BooleanVar(value=True)
    entry_above_9ema_var = tk.BooleanVar(value=False)

    mode_var = tk.StringVar(value="all")
    mode_row = tk.Frame(params)
    mode_row.grid(row=2, column=0, columnspan=12, sticky="w", pady=(6, 0))
    ttk.Checkbutton(mode_row, text="9 EMA exit", variable=exit_below_9ema_var).pack(
        side=tk.LEFT, padx=(0, 12)
    )
    ttk.Checkbutton(mode_row, text="9 EMA entry", variable=entry_above_9ema_var).pack(
        side=tk.LEFT, padx=(0, 12)
    )
    regime_filter_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(mode_row, text="Cash when Trend_Down", variable=regime_filter_var).pack(
        side=tk.LEFT, padx=(0, 12)
    )
    ttk.Radiobutton(mode_row, text="Period-by-period", variable=mode_var, value="step").pack(
        side=tk.LEFT, padx=(0, 10)
    )
    ttk.Radiobutton(mode_row, text="Run all after load", variable=mode_var, value="all").pack(
        side=tk.LEFT, padx=(0, 10)
    )
    show_equity_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(mode_row, text="Show equity curve", variable=show_equity_var).pack(
        side=tk.LEFT
    )

    _apply_strategy_defaults(
        strategy_var,
        portfolio_var,
        entry_rank_var,
        exit_rank_var,
        benchmark_var,
        proximity_var,
        rebalance_var,
    )
    _sync_entry_rank_floor()

    btn_row = tk.Frame(params)
    btn_row.grid(row=3, column=0, columnspan=12, sticky="w", pady=(8, 0))

    status_var = tk.StringVar(value="Choose strategy, set dates, then Load Data.")
    tk.Label(win, textvariable=status_var, anchor="w", fg="#333", padx=10).pack(fill=tk.X)

    body = ttk.PanedWindow(win, orient=tk.VERTICAL)
    body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    top_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(top_pane, weight=3)

    chart_frame = tk.Frame(top_pane)
    top_pane.add(chart_frame, weight=3)
    fig = Figure(figsize=(7, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Portfolio vs benchmark")
    ax.set_xlabel("Period #")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    metrics_frame = tk.LabelFrame(top_pane, text="Summary metrics", padx=4, pady=4)
    top_pane.add(metrics_frame, weight=1)
    metrics_tree = _make_kv_tree(metrics_frame, height=18, col0_width=150, col1_width=180)

    bottom_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(bottom_pane, weight=2)

    detail_frame = tk.LabelFrame(bottom_pane, text="Selected period", padx=4, pady=4)
    bottom_pane.add(detail_frame, weight=2)
    detail_pane = ttk.PanedWindow(detail_frame, orient=tk.VERTICAL)
    detail_pane.pack(fill=tk.BOTH, expand=True)

    summary_frame = tk.Frame(detail_pane)
    detail_pane.add(summary_frame, weight=1)
    detail_tree = _make_kv_tree(summary_frame, height=7, col0_width=130, col1_width=280)

    pick_frame = tk.LabelFrame(detail_pane, text="Rebalance tickers", padx=4, pady=4)
    detail_pane.add(pick_frame, weight=2)
    pick_cols = (
        "Symbol",
        "Status",
        "Entry",
        "Mark/Exit",
        "Date",
        "Alloc %",
        "Qty",
        "Allocated",
        "Period P/L %",
        "Since entry %",
        "Reason",
    )
    pick_table = tk.Frame(pick_frame)
    pick_table.pack(fill=tk.BOTH, expand=True)
    pick_tree = ttk.Treeview(
        pick_table, columns=pick_cols, show="headings", height=8, selectmode="browse"
    )
    _pick_col_widths: dict[str, int] = {
        "Symbol": 92,
        "Status": 52,
        "Entry": 76,
        "Mark/Exit": 76,
        "Date": 84,
        "Alloc %": 52,
        "Qty": 72,
        "Allocated": 78,
        "Period P/L %": 72,
        "Since entry %": 72,
        "Reason": 220,
    }
    for col in pick_cols:
        pick_tree.heading(col, text=col)
        w = _pick_col_widths.get(col, 72)
        anchor = "w" if col in ("Symbol", "Status", "Date", "Reason") else "e"
        pick_tree.column(
            col,
            width=w,
            minwidth=w if col != "Reason" else 180,
            stretch=(col == "Reason"),
            anchor=anchor,
        )
    pick_v_scroll = ttk.Scrollbar(pick_table, orient=tk.VERTICAL, command=pick_tree.yview)
    pick_h_scroll = ttk.Scrollbar(pick_table, orient=tk.HORIZONTAL, command=pick_tree.xview)
    pick_tree.configure(yscrollcommand=pick_v_scroll.set, xscrollcommand=pick_h_scroll.set)
    pick_tree.grid(row=0, column=0, sticky="nsew")
    pick_v_scroll.grid(row=0, column=1, sticky="ns")
    pick_h_scroll.grid(row=1, column=0, sticky="ew")
    pick_table.grid_rowconfigure(0, weight=1)
    pick_table.grid_columnconfigure(0, weight=1)

    log_frame = tk.LabelFrame(bottom_pane, text="Rebalance log", padx=4, pady=4)
    bottom_pane.add(log_frame, weight=3)
    log_cols = (
        "Period",
        "Rebal",
        "End",
        "Holdings",
        "Ranked",
        "Port%",
        "Bench%",
        "DD%",
        "Val start",
        "Val end",
    )
    _log_col_widths = {
        "Period": 52,
        "Rebal": 88,
        "End": 88,
        "Holdings": 200,
        "Ranked": 56,
        "Port%": 64,
        "Bench%": 64,
        "DD%": 56,
        "Val start": 88,
        "Val end": 88,
    }
    log_table = tk.Frame(log_frame)
    log_table.pack(fill=tk.BOTH, expand=True)
    log_tree = ttk.Treeview(
        log_table, columns=log_cols, show="headings", height=10, selectmode="browse"
    )
    for col in log_cols:
        log_tree.heading(col, text=col)
        w = _log_col_widths.get(col, 72)
        log_tree.column(
            col,
            width=w,
            minwidth=w,
            stretch=(col == "Holdings"),
            anchor="e" if col not in ("Holdings",) else "w",
        )
    log_v_scroll = ttk.Scrollbar(log_table, orient=tk.VERTICAL, command=log_tree.yview)
    log_h_scroll = ttk.Scrollbar(log_table, orient=tk.HORIZONTAL, command=log_tree.xview)
    log_tree.configure(yscrollcommand=log_v_scroll.set, xscrollcommand=log_h_scroll.set)
    log_tree.grid(row=0, column=0, sticky="nsew")
    log_v_scroll.grid(row=0, column=1, sticky="ns")
    log_h_scroll.grid(row=1, column=0, sticky="ew")
    log_table.grid_rowconfigure(0, weight=1)
    log_table.grid_columnconfigure(0, weight=1)

    load_btn = ttk.Button(btn_row, text="Load Data")
    load_btn.pack(side=tk.LEFT, padx=4)
    prev_btn = ttk.Button(btn_row, text="Prev Period", state=tk.DISABLED)
    prev_btn.pack(side=tk.LEFT, padx=4)
    next_btn = ttk.Button(btn_row, text="Next Period", state=tk.DISABLED)
    next_btn.pack(side=tk.LEFT, padx=4)
    run_all_btn = ttk.Button(btn_row, text="Run All", state=tk.DISABLED)
    run_all_btn.pack(side=tk.LEFT, padx=4)
    reset_btn = ttk.Button(btn_row, text="Reset Run", state=tk.DISABLED)
    reset_btn.pack(side=tk.LEFT, padx=4)

    install_copy_support(win)

    def _rebalance_key() -> str:
        label = rebalance_var.get().strip().lower()
        for key, text in REBALANCE_LABELS.items():
            if text.lower() == label:
                return key
        return REBALANCE_ALIASES.get(label, "weekly")

    def _parse_capital() -> float:
        return float(capital_var.get().replace(",", "").strip())

    def _parse_proximity() -> float:
        v = float(proximity_var.get().strip())
        if not (0 < v <= 1):
            raise ValueError("52w proximity must be in (0, 1]")
        return v

    def _parse_stop_loss_pct() -> float:
        v = float(stop_loss_pct_var.get())
        if v <= 0:
            raise ValueError("Stop loss % must be greater than 0")
        return v

    def _ui_config() -> EtfMomentumBacktestConfig:
        return build_config_from_ui(
            strategy_key=_strategy_key_from_label(strategy_var.get()),
            backtest_start=rrg_config_date_str(start_var.get()),
            backtest_end=rrg_config_date_str(end_var.get()),
            rebalance_period=_rebalance_key(),
            portfolio_size=int(portfolio_var.get()),
            entry_rank_depth=max(int(entry_rank_var.get()), int(portfolio_var.get())),
            exit_rank_threshold=int(exit_rank_var.get()),
            exit_rank_enabled=bool(exit_rank_enabled_var.get()),
            benchmark_ticker=benchmark_var.get().strip(),
            proximity_of_52w_high=_parse_proximity(),
            initial_capital=_parse_capital(),
            use_regime_filter=bool(regime_filter_var.get()),
            exit_below_9ema=bool(exit_below_9ema_var.get()),
            entry_above_9ema=bool(entry_above_9ema_var.get()),
            exit_stop_loss=bool(exit_stop_loss_var.get()),
            stop_loss_pct=_parse_stop_loss_pct() if exit_stop_loss_var.get() else 5.0,
            park_empty_slots=bool(park_empty_slots_var.get()),
            park_slot_etf=park_slot_var.get().strip(),
        )

    def _sync_engine_config() -> None:
        if engine is None:
            return
        ui = _ui_config()
        engine.config.portfolio_size = ui.portfolio_size
        engine.config.entry_rank_depth = ui.entry_rank_depth
        engine.config.exit_rank_threshold = ui.exit_rank_threshold
        engine.config.exit_rank_enabled = ui.exit_rank_enabled
        engine.config.proximity_of_52w_high = ui.proximity_of_52w_high
        engine.config.use_regime_filter = ui.use_regime_filter
        engine.config.exit_below_9ema = ui.exit_below_9ema
        engine.config.entry_above_9ema = ui.entry_above_9ema
        engine.config.exit_stop_loss = ui.exit_stop_loss
        engine.config.stop_loss_pct = ui.stop_loss_pct
        engine.config.park_empty_slots = ui.park_empty_slots
        engine.config.park_slot_etf = ui.park_slot_etf

    def _config_needs_reload(
        old: EtfMomentumBacktestConfig, new: EtfMomentumBacktestConfig
    ) -> bool:
        return (
            old.strategy_key != new.strategy_key
            or old.backtest_start != new.backtest_start
            or old.backtest_end != new.backtest_end
            or old.rebalance_period != new.rebalance_period
            or old.benchmark_ticker != new.benchmark_ticker
        )

    def _set_busy(busy: bool) -> None:
        nonlocal _busy
        _busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        load_btn.config(state=state)
        next_btn.config(state=state)
        prev_btn.config(state=state)
        run_all_btn.config(state=state)
        reset_btn.config(state=state)
        strategy_combo.config(state="disabled" if busy else "readonly")
        win.config(cursor="watch" if busy else "")
        if not busy:
            _refresh_nav_buttons()

    def _refresh_nav_buttons() -> None:
        can_prev = (
            not _busy
            and engine is not None
            and engine.loaded
            and engine.current_period > 0
        )
        prev_btn.config(state=tk.NORMAL if can_prev else tk.DISABLED)
        can_next = (
            not _busy
            and engine is not None
            and engine.loaded
            and not engine.finished
        )
        next_btn.config(state=tk.NORMAL if can_next else tk.DISABLED)
        can_reset = not _busy and engine is not None and engine.loaded
        reset_btn.config(state=tk.NORMAL if can_reset else tk.DISABLED)
        run_all_btn.config(
            state=tk.NORMAL if can_reset and not engine.finished else tk.DISABLED
        )

    def _clear_results() -> None:
        for item in log_tree.get_children():
            log_tree.delete(item)
        _fill_kv_tree(detail_tree, None, empty="Load data, then Next Period or Run All.")
        _fill_pick_tree(pick_tree, None)
        _fill_kv_tree(metrics_tree, None, empty="No results yet.")
        _update_chart()

    def _update_metrics() -> None:
        if engine is None or engine.trades_df.empty:
            _fill_kv_tree(metrics_tree, None, empty="No results yet.")
            return
        m = compute_metrics(engine.trades_df, _parse_capital())
        _fill_kv_tree(metrics_tree, {str(k): str(v) for k, v in m.items()}, empty="No results yet.")

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

    def _update_chart() -> None:
        if not show_equity_var.get():
            return
        ax.clear()
        if engine is None or engine.trades_df.empty:
            ax.set_title("Portfolio vs benchmark")
            canvas.draw_idle()
            return
        df = engine.trades_df
        cap = _parse_capital()
        port_vals = df["Portfolio_Value"].values
        bench_vals = cap * (1 + df["Bench_Return"]).cumprod()
        periods = range(1, len(df) + 1)
        bench_label = engine.config.benchmark_ticker
        ax.plot(periods, port_vals, label="Strategy", color="#1565C0", linewidth=2)
        ax.plot(periods, bench_vals, label=bench_label, color="#757575", linewidth=1.5)
        if engine.current_period < engine.total_periods:
            ax.axvline(engine.current_period, color="#E65100", linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Equity curve ({len(df)} period(s))")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        canvas.draw_idle()

    def _refresh_log_table() -> None:
        for item in log_tree.get_children():
            log_tree.delete(item)
        if engine is None or engine.trades_df.empty:
            return
        cap = _parse_capital()
        dd_series = backtest_drawdown_pct_series(engine.trades_df["Port_Return"]).values
        for i, row in engine.trades_df.iterrows():
            cum_pl = backtest_cum_pl_pct(float(row["Portfolio_Value"]), cap)
            log_tree.insert(
                "",
                tk.END,
                iid=str(row["Period"]),
                values=(
                    row["Period"],
                    rrg_format_date(row["Rebal_Date"]),
                    rrg_format_date(row["End_Date"]),
                    row.get("Holdings", "—"),
                    row.get("Universe_Ranked", "—"),
                    f"{row['Port_Return'] * 100:+.2f}",
                    f"{row['Bench_Return'] * 100:+.2f}",
                    f"{dd_series[i]:.2f}",
                    _format_portfolio_value(
                        row.get("Portfolio_Value_Start", row["Portfolio_Value"])
                    ),
                    _format_portfolio_value(row.get("Portfolio_Value_End", row["Portfolio_Value"])),
                ),
            )

    def _show_period_detail(record: dict | None) -> None:
        if record is None or engine is None:
            _fill_kv_tree(detail_tree, None, empty="Load data, then Next Period or Run All.")
            _fill_pick_tree(pick_tree, None)
            return
        cap = _parse_capital()
        port_end = float(record.get("Portfolio_Value_End", record["Portfolio_Value"]))
        cum_pl = backtest_cum_pl_pct(port_end, cap)
        fields = {
            "Period start": rrg_format_date(record["Rebal_Date"]),
            "Period end": rrg_format_date(record["End_Date"]),
            "Portfolio value (start)": _format_portfolio_value(
                record.get("Portfolio_Value_Start")
            ),
            "Portfolio value (end)": _format_portfolio_value(
                record.get("Portfolio_Value_End", record.get("Portfolio_Value"))
            ),
            "Market regime": str(record.get("Regime", "—")),
            "Holdings": record.get("Holdings") or "CASH",
            "Momentum ETFs": record.get("Momentum_Holdings") or "—",
            "Park slots": str(record.get("Park_Slots", 0)),
            "Equal-weight slots": str(record.get("Equal_Weight_Slots", "—")),
            "Slot allocation": (
                f"{record.get('Slot_Alloc_Amount', 0):,.2f}"
                if record.get("Slot_Alloc_Amount") is not None
                else "—"
            ),
            "Ranked ETFs (screen)": str(record.get("Universe_Ranked", 0)),
            "New entries": str(record.get("New_Entries", 0)),
            "9 EMA exits (rebal)": str(record.get("EMA_9_Exits_Rebal", 0)),
            "9 EMA exits (mid-period)": str(record.get("EMA_9_Exits_Midweek", 0)),
            "Stop loss exits": str(record.get("Stop_Loss_Exits", 0)),
            "Turnover": f"{record.get('Turnover', 0) * 100:.1f} %",
            "Portfolio % (period)": f"{record['Port_Return'] * 100:+.2f}",
            "Portfolio P/L % (since start)": f"{cum_pl:+.2f}",
            "Benchmark %": f"{record['Bench_Return'] * 100:+.2f}",
        }
        _fill_kv_tree(detail_tree, fields, empty="—")
        _fill_pick_tree(pick_tree, record.get("Position_Rows"))

    def _on_log_select(_event=None) -> None:
        if engine is None or not engine.trades_df.shape[0]:
            return
        sel = log_tree.selection()
        if not sel:
            return
        period = int(sel[0])
        rows = engine.trades_df[engine.trades_df["Period"] == period]
        if not rows.empty:
            _show_period_detail(rows.iloc[0].to_dict())

    log_tree.bind("<<TreeviewSelect>>", _on_log_select)

    def _on_step_done(err: Exception | None) -> None:
        if err is not None:
            messagebox.showerror("Backtest", str(err), parent=win)
            status_var.set(f"Error: {err}")
            return
        _refresh_log_table()
        _update_chart()
        _update_metrics()
        if engine is not None and engine.trades_df.shape[0]:
            last = engine.trades_df.iloc[-1].to_dict()
            _show_period_detail(last)
            log_tree.selection_set(str(last["Period"]))
            status_var.set(
                f"Period {engine.current_period}/{engine.total_periods} complete."
                if not engine.finished
                else "All periods complete."
            )
        _refresh_nav_buttons()

    def _on_load() -> None:
        nonlocal engine, _load_run_all_after
        if _busy:
            return
        try:
            ui_cfg = _ui_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=win)
            return
        _load_run_all_after = mode_var.get() == "all"
        _set_busy(True)
        status_var.set("Loading Yahoo data (may take a few minutes)…")

        def worker() -> None:
            err: Exception | None = None
            eng: EtfMomentumBacktestEngine | None = None
            try:
                eng = EtfMomentumBacktestEngine(ui_cfg)
                eng.load_data()
            except Exception as exc:
                err = exc
            win.after(0, lambda: _on_load_done(eng, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_done(eng: EtfMomentumBacktestEngine | None, err: Exception | None) -> None:
        nonlocal engine, _load_run_all_after
        _set_busy(False)
        if err is not None:
            messagebox.showerror("Load failed", str(err), parent=win)
            status_var.set(f"Load failed: {err}")
            return
        engine = eng
        _clear_results()
        status_var.set(
            f"Loaded {engine.total_periods} periods for {strategy_var.get()}. "
            + ("Running all…" if _load_run_all_after else "Click Next Period or Run All.")
        )
        _refresh_nav_buttons()
        if _load_run_all_after:
            _on_run_all(from_load=True)

    def _on_next() -> None:
        if _busy or engine is None or not engine.loaded or engine.finished:
            return
        try:
            new_cfg = _ui_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=win)
            return
        if _config_needs_reload(engine.config, new_cfg):
            messagebox.showwarning(
                "Reload required",
                "Strategy, dates, benchmark, or rebalance changed — click Load Data again.",
                parent=win,
            )
            return
        _set_busy(True)
        status_var.set("Simulating period…")

        def worker() -> None:
            err: Exception | None = None
            try:
                _sync_engine_config()
                engine.step_period()
            except Exception as exc:
                err = exc
            win.after(0, lambda: (_set_busy(False), _on_step_done(err)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_all(*, from_load: bool = False) -> None:
        if _busy or engine is None or not engine.loaded:
            return
        if engine.finished and not from_load:
            return
        _set_busy(True)
        status_var.set("Running all periods…")

        def worker() -> None:
            err: Exception | None = None
            try:
                _sync_engine_config()
                engine.run_all()
            except Exception as exc:
                err = exc
            win.after(0, lambda: (_set_busy(False), _on_step_done(err)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_prev() -> None:
        if _busy or engine is None or engine.current_period <= 0:
            return
        target = engine.current_period - 1
        _set_busy(True)

        def worker() -> None:
            err: Exception | None = None
            try:
                _sync_engine_config()
                engine.reset_run()
                for _ in range(target):
                    engine.step_period()
            except Exception as exc:
                err = exc
            win.after(0, lambda: (_set_busy(False), _on_step_done(err)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_reset() -> None:
        if _busy or engine is None:
            return
        engine.reset_run()
        _clear_results()
        status_var.set(f"Reset. {engine.total_periods} periods ready.")
        _refresh_nav_buttons()

    def _on_strategy_change(_event=None) -> None:
        _apply_strategy_defaults(
            strategy_var,
            portfolio_var,
            entry_rank_var,
            exit_rank_var,
            benchmark_var,
            proximity_var,
            rebalance_var,
        )
        _sync_entry_rank_floor()

    strategy_combo.bind("<<ComboboxSelected>>", _on_strategy_change)

    load_btn.config(command=_on_load)
    next_btn.config(command=_on_next)
    prev_btn.config(command=_on_prev)
    run_all_btn.config(command=lambda: _on_run_all(from_load=False))
    reset_btn.config(command=_on_reset)

    show_equity_var.trace_add("write", _on_equity_toggle)
    _set_equity_curve_visible(bool(show_equity_var.get()))

    return win


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    open_etf_momentum_backtest(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
