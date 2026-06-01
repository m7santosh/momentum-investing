"""Tkinter UI for RRG backtest (India / US; week-by-week and run-all modes)."""

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

from momentum.rrg_core import rrg_row_fg_color
from momentum.rrg_ui_copy import (
    TableRegionCopy,
    configure_readonly_text,
    install_copy_support,
)
from utils.nse_bhavcopy import today_ist


@dataclass(frozen=True)
class _BacktestUiProfile:
    title: str
    bench_chart: str
    load_status: str
    Config: type
    Engine: type
    compute_metrics: Callable[[pd.DataFrame, float], dict]
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
            load_status=(
                "Screening liquid universe and loading Yahoo Finance "
                "(may take a few minutes)..."
            ),
            Config=UsRrgBacktestConfig,
            Engine=UsRrgBacktestEngine,
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


def open_rrg_backtest(
    parent: tk.Misc,
    *,
    profile: str = "india",
    rrg_window: int = 10,
    tail: int = 1,
    top_n: int = 7,
    backtest_extra: dict[str, Any] | None = None,
    pick_strategy: str | None = None,
    hold_until_rank_exit: bool | None = None,
    max_hold_rank: int | None = None,
    exit_below_9ema: bool | None = None,
) -> tk.Toplevel:
    """Open RRG backtest as a normal secondary window (not modal)."""
    prof = _backtest_profile(profile)
    bt_extra = dict(backtest_extra or {})
    if pick_strategy is not None:
        bt_extra.setdefault("pick_strategy", pick_strategy)
    if hold_until_rank_exit is not None:
        bt_extra.setdefault("hold_until_rank_exit", hold_until_rank_exit)
    if max_hold_rank is not None:
        bt_extra.setdefault("max_hold_rank", max_hold_rank)
    if exit_below_9ema is not None:
        bt_extra.setdefault("exit_below_9ema", exit_below_9ema)
    win = tk.Toplevel(parent)
    win.title(prof.title)
    win.geometry("1280x820")
    win.minsize(1000, 640)
    # Do not use transient() — on Windows that hides min/max buttons and traps
    # focus above the main RRG window so you cannot switch back without closing.
    try:
        win.attributes("-toolwindow", False)
    except tk.TclError:
        pass
    win.lift()

    engine: Any | None = None
    _busy = False
    _load_run_all_after = False

    # ── Params ──
    params = tk.Frame(win, padx=10, pady=8)
    params.pack(fill=tk.X)

    tk.Label(params, text="Tail from (Fri):").grid(row=0, column=0, sticky="w")
    start_var = tk.StringVar(value=f"{today_ist().year - 1}-01-01")
    tk.Entry(params, textvariable=start_var, width=12).grid(row=0, column=1, padx=(4, 16))

    tk.Label(params, text="As-of (Fri):").grid(row=0, column=2, sticky="w")
    end_var = tk.StringVar(value=today_ist().strftime("%Y-%m-%d"))
    tk.Entry(params, textvariable=end_var, width=12).grid(row=0, column=3, padx=(4, 16))

    tk.Label(params, text="Top N:").grid(row=0, column=4, sticky="w")
    top_n_var = tk.IntVar(value=top_n)
    ttk.Spinbox(params, from_=1, to=15, width=4, textvariable=top_n_var).grid(
        row=0, column=5, padx=(4, 16)
    )

    tk.Label(params, text="Tail:").grid(row=0, column=6, sticky="w")
    tail_var = tk.IntVar(value=tail)
    ttk.Spinbox(params, from_=1, to=10, width=4, textvariable=tail_var).grid(
        row=0, column=7, padx=(4, 16)
    )

    tk.Label(params, text="RRG window:").grid(row=0, column=8, sticky="w")
    window_var = tk.IntVar(value=rrg_window)
    window_combo = ttk.Combobox(
        params, values=("10", "14"), width=4, state="readonly"
    )
    window_combo.set(str(rrg_window))
    window_combo.grid(row=0, column=9, padx=(4, 16))

    tk.Label(params, text="Capital:").grid(row=0, column=10, sticky="w")
    capital_var = tk.StringVar(value="100000")
    tk.Entry(params, textvariable=capital_var, width=10).grid(row=0, column=11, padx=4)

    pick_strategy_var = tk.StringVar()
    hold_until_rank_exit_var = tk.BooleanVar(value=False)
    exit_below_9ema_var = tk.BooleanVar(value=False)
    max_hold_rank_var = tk.IntVar(value=10)
    _pick_label_to_key: dict[str, str] = {}
    max_rank_label: tk.Label | None = None
    max_rank_spin: ttk.Spinbox | None = None

    if profile in ("india", "us"):
        from momentum.etf.india_rrg_pick_strategies import PICK_STRATEGIES

        _pick_label_to_key = {label: key for key, label in PICK_STRATEGIES.items()}
        pick_strategy_var.set(PICK_STRATEGIES["recommend"])
        strat_row = tk.Frame(params)
        strat_row.grid(row=1, column=0, columnspan=12, sticky="w", pady=(6, 0))
        tk.Label(strat_row, text="Base strategy:").pack(side=tk.LEFT)
        ttk.Combobox(
            strat_row,
            textvariable=pick_strategy_var,
            values=list(PICK_STRATEGIES.values()),
            width=40,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(
            strat_row,
            text="Hold until rank worse",
            variable=hold_until_rank_exit_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            strat_row,
            text="Exit below 9 EMA",
            variable=exit_below_9ema_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        max_rank_label = tk.Label(strat_row, text="Max hold rank:")
        max_rank_spin = ttk.Spinbox(
            strat_row,
            from_=5,
            to=60,
            width=4,
            textvariable=max_hold_rank_var,
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

    if profile in ("india", "us") and bt_extra.get("pick_strategy"):
        from momentum.etf.india_rrg_pick_strategies import PICK_STRATEGIES as _PS

        ps_key = str(bt_extra["pick_strategy"])
        if ps_key == "top_n_rank_exit":
            ps_key = "top_n"
            hold_until_rank_exit_var.set(True)
        if ps_key in _PS:
            pick_strategy_var.set(_PS[ps_key])
    if bt_extra.get("hold_until_rank_exit"):
        hold_until_rank_exit_var.set(True)
    if bt_extra.get("max_hold_rank") is not None:
        max_hold_rank_var.set(int(bt_extra["max_hold_rank"]))
    if bt_extra.get("exit_below_9ema"):
        exit_below_9ema_var.set(True)

    mode_var = tk.StringVar(value="step")
    mode_row = tk.Frame(params)
    mode_row.grid(
        row=2 if profile in ("india", "us") else 1,
        column=0,
        columnspan=12,
        sticky="w",
        pady=(8, 0),
    )
    ttk.Radiobutton(
        mode_row, text="Week-by-week", variable=mode_var, value="step"
    ).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Radiobutton(
        mode_row, text="Run all (one go)", variable=mode_var, value="all"
    ).pack(side=tk.LEFT)

    btn_row = tk.Frame(win, pady=6)
    btn_row.pack(fill=tk.X, padx=8)

    status_var = tk.StringVar(value="Set dates and click Load Data.")
    tk.Label(win, textvariable=status_var, anchor="w", fg="#333", padx=10).pack(
        fill=tk.X
    )
    # ── Main split: chart + metrics | week detail + log ──
    body = ttk.PanedWindow(win, orient=tk.VERTICAL)
    body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    top_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(top_pane, weight=3)

    chart_frame = tk.Frame(top_pane)
    top_pane.add(chart_frame, weight=3)

    fig = Figure(figsize=(7, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Portfolio vs {prof.bench_chart}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    metrics_frame = tk.Frame(top_pane, padx=8)
    top_pane.add(metrics_frame, weight=1)
    metrics_title = tk.Label(metrics_frame, text="Metrics", font=("Arial", 10, "bold"))
    metrics_title.pack(anchor="w")
    metrics_text = tk.Text(
        metrics_frame, width=36, height=18, font=("Consolas", 9)
    )
    metrics_text.pack(fill=tk.BOTH, expand=True)
    configure_readonly_text(metrics_text)

    bottom_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(bottom_pane, weight=2)

    week_frame = tk.LabelFrame(bottom_pane, text="Current week", padx=6, pady=4)
    bottom_pane.add(week_frame, weight=1)

    week_header = tk.Label(week_frame, text="—", anchor="w", font=("Arial", 10, "bold"))
    week_header.pack(fill=tk.X)
    week_return_label = tk.Label(week_frame, text="", anchor="w")
    week_return_label.pack(fill=tk.X)

    picks_table = tk.Frame(week_frame)
    picks_table.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
    _pick_cols = ("Rank", "ETF", "Price", "Quadrant", "Rank Δ", "Chg%", "Vol%")
    _pick_col_anchors = ("e", "w", "e", "w", "e", "e", "e")
    pick_header: list[tk.Label] = []
    for c, h in enumerate(_pick_cols):
        lbl = tk.Label(
            picks_table,
            text=h,
            font=("Arial", 9, "bold"),
            relief=tk.RIDGE,
            padx=4,
            anchor=_pick_col_anchors[c],
        )
        lbl.grid(row=0, column=c, sticky="ew")
        pick_header.append(lbl)
    _MAX_TOP_N = 15
    pick_rows: list[list[tk.Label]] = []
    for r in range(_MAX_TOP_N):
        row_cells: list[tk.Label] = []
        for c in range(len(_pick_cols)):
            lbl = tk.Label(
                picks_table,
                text="",
                relief=tk.RIDGE,
                padx=4,
                anchor=_pick_col_anchors[c],
            )
            lbl.grid(row=r + 1, column=c, sticky="ew")
            row_cells.append(lbl)
        pick_rows.append(row_cells)

    def _apply_top_n_rows(n: int | None = None) -> None:
        count = max(1, min(_MAX_TOP_N, int(n if n is not None else top_n_var.get())))
        panel_bg = win.cget("bg")
        for i, row in enumerate(pick_rows):
            if i < count:
                for c, lbl in enumerate(row):
                    lbl.grid(row=i + 1, column=c, sticky="ew")
            else:
                for lbl in row:
                    lbl.grid_remove()
                    lbl.config(text="", bg=panel_bg, fg="black")

    _apply_top_n_rows(top_n)
    top_n_var.trace_add("write", lambda *_: _apply_top_n_rows())

    log_frame = tk.LabelFrame(bottom_pane, text="Rebalance log", padx=4, pady=4)
    bottom_pane.add(log_frame, weight=2)

    log_cols = ("Week", "Rebal", "End", "Holdings", "Port%", "Bench%", "Value")
    log_tree = ttk.Treeview(
        log_frame, columns=log_cols, show="headings", height=12, selectmode="extended"
    )
    for col in log_cols:
        log_tree.heading(col, text=col)
        log_tree.column(col, width=100 if col != "Holdings" else 220, stretch=True)
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_tree.yview)
    log_tree.configure(yscrollcommand=log_scroll.set)
    log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    _bt_copy = TableRegionCopy.for_window(win)
    picks_copy_grid = _bt_copy.register_grid([pick_header, *pick_rows])

    def _set_busy(busy: bool, cursor: str = "") -> None:
        nonlocal _busy
        _busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        load_btn.config(state=state)
        next_btn.config(state=state)
        prev_btn.config(state=state)
        run_all_btn.config(state=state)
        reset_btn.config(state=state)
        win.config(cursor="watch" if busy else "")
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
    def _update_metrics() -> None:
        if engine is None or engine.trades_df.empty:
            metrics_text.delete("1.0", tk.END)
            metrics_text.insert(tk.END, "No results yet.")
            return
        df = engine.trades_df
        df.attrs["top_n"] = int(top_n_var.get())
        if profile in ("india", "us") and _pick_label_to_key:
            df.attrs["pick_strategy"] = _pick_label_to_key.get(
                pick_strategy_var.get(), "recommend"
            )
            df.attrs["hold_until_rank_exit"] = bool(hold_until_rank_exit_var.get())
            df.attrs["max_hold_rank"] = int(max_hold_rank_var.get())
            df.attrs["exit_below_9ema"] = bool(exit_below_9ema_var.get())
        cap = float(capital_var.get())
        if profile == "us" and hasattr(engine, "_benchmark"):
            m = prof.compute_metrics(df, cap, benchmark=engine._benchmark)
        else:
            m = prof.compute_metrics(df, cap)
        lines = [f"{k}: {v}" for k, v in m.items()]
        metrics_text.delete("1.0", tk.END)
        metrics_text.insert(tk.END, "\n".join(lines))

    def _update_chart() -> None:
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
        ax.set_title(f"Equity curve ({len(df)} weeks simulated)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        canvas.draw_idle()

    def _clear_week_display() -> None:
        week_header.config(text="—")
        week_return_label.config(text="")
        panel_bg = win.cget("bg")
        visible_rows = max(1, min(_MAX_TOP_N, int(top_n_var.get())))
        for slot, row in enumerate(pick_rows):
            if slot >= visible_rows:
                continue
            for lbl in row:
                lbl.config(text="", bg=panel_bg, fg="black")
        TableRegionCopy.for_window(win).sync_styles(picks_copy_grid)

    def _show_week_record(record: dict | None) -> None:
        if record is None:
            week_header.config(text="Done — all weeks processed.")
            week_return_label.config(text="")
            panel_bg = win.cget("bg")
            visible_rows = max(1, min(_MAX_TOP_N, int(top_n_var.get())))
            for slot, row in enumerate(pick_rows):
                if slot >= visible_rows:
                    continue
                for lbl in row:
                    lbl.config(text="", bg=panel_bg, fg="black")
            TableRegionCopy.for_window(win).sync_styles(picks_copy_grid)
            return

        rebal = pd.Timestamp(record["Rebal_Date"]).strftime("%Y-%m-%d")
        end = pd.Timestamp(record["End_Date"]).strftime("%Y-%m-%d")
        tail_start = record.get("Tail_Start")
        if tail_start is not None:
            tail_s = pd.Timestamp(tail_start).strftime("%Y-%m-%d")
            header = (
                f"Week {record['Week']} — as of {rebal} "
                f"(tail {tail_s} → {rebal}) · hold → {end}"
            )
        else:
            header = f"Week {record['Week']} — as of {rebal} · hold → {end}"
        week_header.config(text=header)
        week_return_label.config(
            text=(
                f"Portfolio: {record['Port_Return'] * 100:+.2f}%  |  "
                f"Benchmark: {record['Bench_Return'] * 100:+.2f}%  |  "
                f"Value: {record['Portfolio_Value']:,.0f}"
            )
        )
        picks = record.get("Picks") or []
        pick_prices: dict[str, float | None] = record.get("Pick_Prices") or {}
        rebal_ts = pd.Timestamp(record["Rebal_Date"])
        panel_bg = win.cget("bg")
        visible_rows = max(1, min(_MAX_TOP_N, int(top_n_var.get())))
        for slot, row in enumerate(pick_rows):
            if slot >= visible_rows:
                continue
            if slot >= len(picks):
                for lbl in row:
                    lbl.config(text="", bg=panel_bg, fg="black")
                continue
            p = picks[slot]
            bg = prof.recommendation_row_bg(p.quadrant)
            fg = rrg_row_fg_color(bg)
            price = pick_prices.get(p.ticker)
            if price is None and engine is not None:
                price = engine.ref_price_at(p.ticker, rebal_ts)
            price_text = f"{price:.2f}" if price is not None else ""
            vals = (
                str(p.pick_rank),
                p.ticker,
                price_text,
                p.quadrant,
                p.rank_delta,
                f"{p.change_pct:.2f}",
                prof.format_vol_pct(p.vol_pct),
            )
            for lbl, val in zip(row, vals):
                lbl.config(text=val, bg=bg, fg=fg)
        TableRegionCopy.for_window(win).sync_styles(picks_copy_grid)

    def _refresh_log_table() -> None:
        for item in log_tree.get_children():
            log_tree.delete(item)
        if engine is None or engine.trades_df.empty:
            return
        for _, row in engine.trades_df.iterrows():
            log_tree.insert(
                "",
                tk.END,
                values=(
                    row["Week"],
                    pd.Timestamp(row["Rebal_Date"]).strftime("%Y-%m-%d"),
                    pd.Timestamp(row["End_Date"]).strftime("%Y-%m-%d"),
                    row["Holdings"],
                    f"{row['Port_Return'] * 100:+.2f}",
                    f"{row['Bench_Return'] * 100:+.2f}",
                    f"{row['Portfolio_Value']:,.0f}",
                ),
            )
        log_tree.yview_moveto(1.0)

    def _clear_results() -> None:
        _refresh_log_table()
        _clear_week_display()
        _update_chart()
        _update_metrics()
        _refresh_nav_buttons()

    def _build_engine():
        return prof.Engine(
            config=_ui_config(),
            progress_cb=lambda msg: win.after(0, lambda m=msg: status_var.set(m)),
        )

    def _ui_config():
        kw = dict(
            backtest_start=start_var.get().strip(),
            backtest_end=end_var.get().strip(),
            top_n=int(top_n_var.get()),
            tail=int(tail_var.get()),
            rrg_window=int(window_combo.get()),
            initial_capital=float(capital_var.get()),
        )
        if profile in ("india", "us") and _pick_label_to_key:
            kw["pick_strategy"] = _pick_label_to_key.get(
                pick_strategy_var.get(), "recommend"
            )
            kw["hold_until_rank_exit"] = bool(hold_until_rank_exit_var.get())
            kw["max_hold_rank"] = int(max_hold_rank_var.get())
            kw["exit_below_9ema"] = bool(exit_below_9ema_var.get())
        if profile == "us":
            kw.update(
                {
                    k: v
                    for k, v in bt_extra.items()
                    if k
                    in (
                        "universe_mode",
                        "min_adv_usd",
                        "vol_percentile",
                        "screen_categories",
                    )
                }
            )
        return prof.Config(**kw)

    def _config_needs_reload(stored, ui) -> bool:
        base = (
            stored.backtest_start != ui.backtest_start
            or stored.backtest_end != ui.backtest_end
            or stored.tail != ui.tail
            or stored.rrg_window != ui.rrg_window
        )
        if profile == "us":
            base = base or (
                getattr(stored, "universe_mode", "expanded")
                != getattr(ui, "universe_mode", "expanded")
                or getattr(stored, "min_adv_usd", 0) != getattr(ui, "min_adv_usd", 0)
                or getattr(stored, "vol_percentile", 100)
                != getattr(ui, "vol_percentile", 100)
            )
        return base

    def _sync_engine_config_from_ui(*, require_loaded: bool = True) -> bool:
        """Apply current inputs. Returns False if a full reload is required first."""
        if engine is None or not engine.loaded:
            return False
        ui_cfg = _ui_config()
        if _config_needs_reload(engine.config, ui_cfg):
            return False
        engine.config = ui_cfg
        return True

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
        status_var.set(
            f"Loaded {engine.total_weeks} rebalance week(s) "
            f"({engine.rebal_dates[0].strftime('%Y-%m-%d')} .. "
            f"{engine.rebal_dates[-1].strftime('%Y-%m-%d')}). "
            f"Match main RRG Date slider to as-of date. "
            f"{'Click Next Week or Run All.' if mode_var.get() == 'step' else 'Click Run All.'}"
        )
        run_after = _load_run_all_after or mode_var.get() == "all"
        _load_run_all_after = False
        if run_after:
            _do_run_all(from_load=True)

    def _load_data(*, run_all_after: bool = False) -> None:
        nonlocal _load_run_all_after
        if _busy:
            return
        _load_run_all_after = run_all_after
        _set_busy(True)
        status_var.set(prof.load_status)

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
        if not _sync_engine_config_from_ui():
            messagebox.showinfo(
                "Backtest",
                "Dates, tail, or RRG window changed — click Load Data first.",
                parent=win,
            )
            return
        if engine.finished:
            status_var.set("All weeks done. Click Run All or Reset.")
            return
        record = engine.step_week()
        if record:
            _refresh_log_table()
            _show_week_record(record)
            _update_chart()
            _update_metrics()
            status_var.set(
                f"Week {engine.current_week}/{engine.total_weeks} complete."
            )
        if engine.finished:
            status_var.set(f"Finished all {engine.total_weeks} weeks.")
        _refresh_nav_buttons()

    def _do_previous() -> None:
        if engine is None or not engine.loaded:
            messagebox.showinfo("Backtest", "Load data first.", parent=win)
            return
        if engine.current_week <= 0:
            status_var.set("Already at the first week.")
            return
        record = engine.step_back()
        _refresh_log_table()
        if record:
            _show_week_record(record)
            status_var.set(
                f"Showing week {engine.current_week}/{engine.total_weeks}."
            )
        else:
            _clear_week_display()
            status_var.set("At start — click Next Week to begin.")
        _update_chart()
        _update_metrics()
        _refresh_nav_buttons()

    def _do_run_all(*, from_load: bool = False) -> None:
        if _busy and not from_load:
            return
        if engine is None or not engine.loaded:
            _load_data(run_all_after=True)
            return
        if not from_load:
            ui_cfg = _ui_config()
            if _config_needs_reload(engine.config, ui_cfg):
                _load_data(run_all_after=True)
                return
            engine.config = ui_cfg
            engine.reset_run()
            _clear_results()
        _set_busy(True)
        status_var.set("Running all weeks...")

        def worker():
            err = None
            last_record = None
            try:
                eng = engine
                if eng is None:
                    return
                eng.reset_run()
                while not eng.finished:
                    last_record = eng.step_week()
                    w = eng.current_week
                    total = eng.total_weeks
                    win.after(0, lambda w=w, t=total: status_var.set(f"Running week {w}/{t}..."))
            except Exception as exc:
                err = exc
            win.after(0, lambda: _on_run_all_done(err, last_record))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_all_done(err: Exception | None, last_record: dict | None) -> None:
        _set_busy(False)
        if err is not None:
            messagebox.showerror("Run failed", str(err), parent=win)
            return
        _refresh_log_table()
        if last_record:
            _show_week_record(last_record)
        _update_chart()
        _update_metrics()
        status_var.set(f"Finished all {engine.total_weeks if engine else 0} weeks.")
        _refresh_nav_buttons()

    def _reset() -> None:
        if engine is not None and engine.loaded:
            engine.reset_run()
        _clear_results()
        status_var.set("Reset. Click Next Week or Run All.")

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
    ttk.Button(btn_row, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=10)

    install_copy_support(win)
    _refresh_nav_buttons()
    win.focus_set()
    return win


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
    tail: int = 2,
    top_n: int = 7,
) -> tk.Toplevel:
    return open_rrg_backtest(
        parent,
        profile="us",
        rrg_window=rrg_window,
        tail=tail,
        top_n=top_n,
    )
