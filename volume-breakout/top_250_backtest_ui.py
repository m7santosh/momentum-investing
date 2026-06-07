"""Tkinter backtest UI for Volume Breakout top-250 strategy."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_VB_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _VB_DIR.parent
for _path in (_PROJECT_ROOT, _VB_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backtest.volume_breakout.backtest_top_250 import (  # noqa: E402
    BENCHMARK_NSE,
    Top250BacktestConfig,
    Top250BacktestEngine,
    compute_metrics,
)
from momentum.rrg_backtest_positions import (  # noqa: E402
    format_exit_date,
    format_pl_pct,
    format_price,
)
from momentum.rrg_backtest_ui import (  # noqa: E402
    backtest_cum_pl_pct,
    backtest_drawdown_pct_series,
)
from momentum.rrg_core import rrg_format_date, rrg_parse_user_date  # noqa: E402
from momentum.rrg_ui_copy import install_copy_support  # noqa: E402
from utils.nse_bhavcopy import today_ist  # noqa: E402


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


def _fill_pick_tree(tree: ttk.Treeview, rows: list[dict] | None) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if not rows:
        tree.insert("", tk.END, values=("—", "—", "—", "—", "—"))
        return
    for row in rows:
        if row.get("exit_reason") == "Open":
            exit_date_s = "Open"
        else:
            exit_date_s = format_exit_date(row.get("exit_date"))
        tree.insert(
            "",
            tk.END,
            values=(
                row.get("ticker") or "—",
                format_price(row.get("entry")),
                format_price(row.get("exit")),
                exit_date_s,
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


def open_top250_backtest(
    parent: tk.Misc,
    *,
    initial_start: str | None = None,
    initial_end: str | None = None,
    universe: str = "volume",
) -> tk.Toplevel:
    win = tk.Toplevel(parent)
    win.title("Volume Breakout Backtest")
    win.geometry("1100x760")
    win.minsize(900, 540)
    win.lift()

    engine: Top250BacktestEngine | None = None
    _busy = False
    _load_run_all_after = False

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

    tk.Label(params, text="Universe:").grid(row=0, column=4, sticky="w")
    universe_var = tk.StringVar(value="Volume" if universe != "turnover" else "Turnover")
    ttk.Combobox(
        params,
        textvariable=universe_var,
        values=("Volume", "Turnover"),
        width=10,
        state="readonly",
    ).grid(row=0, column=5, padx=(4, 16))

    tk.Label(params, text="Top N:").grid(row=0, column=6, sticky="w")
    top_n_var = tk.IntVar(value=20)
    ttk.Spinbox(params, from_=1, to=30, width=4, textvariable=top_n_var).grid(
        row=0, column=7, padx=(4, 16)
    )

    tk.Label(params, text="Capital:").grid(row=0, column=8, sticky="w")
    capital_var = tk.StringVar(value="500000")
    tk.Entry(params, textvariable=capital_var, width=10).grid(row=0, column=9, padx=4)

    tk.Label(params, text="Rebalance:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    rebalance_var = tk.StringVar(value="Weekly")
    ttk.Combobox(
        params,
        textvariable=rebalance_var,
        values=("Daily", "Weekly", "Fortnight", "Monthly"),
        width=10,
        state="readonly",
    ).grid(row=1, column=1, padx=(4, 16), pady=(8, 0), sticky="w")

    mode_var = tk.StringVar(value="all")
    mode_row = tk.Frame(params)
    mode_row.grid(row=1, column=2, columnspan=8, sticky="w", pady=(8, 0))
    ttk.Radiobutton(mode_row, text="Period-by-period", variable=mode_var, value="step").pack(
        side=tk.LEFT, padx=(0, 12)
    )
    ttk.Radiobutton(mode_row, text="Run all after load", variable=mode_var, value="all").pack(
        side=tk.LEFT, padx=(0, 12)
    )
    show_equity_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(mode_row, text="Show equity curve", variable=show_equity_var).pack(
        side=tk.LEFT
    )

    btn_row = tk.Frame(win, pady=6)
    btn_row.pack(fill=tk.X, padx=8)

    status_var = tk.StringVar(value="Set dates and Load Data.")
    tk.Label(win, textvariable=status_var, anchor="w", fg="#333", padx=10).pack(fill=tk.X)

    body = ttk.PanedWindow(win, orient=tk.VERTICAL)
    body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    top_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(top_pane, weight=3)

    chart_frame = tk.Frame(top_pane)
    top_pane.add(chart_frame, weight=3)
    fig = Figure(figsize=(7, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Portfolio vs {BENCHMARK_NSE}")
    ax.set_xlabel("Period #")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    metrics_frame = tk.LabelFrame(top_pane, text="Summary metrics", padx=4, pady=4)
    top_pane.add(metrics_frame, weight=1)
    metrics_tree = _make_kv_tree(metrics_frame, height=14, col0_width=150, col1_width=180)

    bottom_pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
    body.add(bottom_pane, weight=2)

    detail_frame = tk.LabelFrame(bottom_pane, text="Selected week", padx=4, pady=4)
    bottom_pane.add(detail_frame, weight=1)
    detail_pane = ttk.PanedWindow(detail_frame, orient=tk.VERTICAL)
    detail_pane.pack(fill=tk.BOTH, expand=True)

    summary_frame = tk.Frame(detail_pane)
    detail_pane.add(summary_frame, weight=1)
    detail_tree = _make_kv_tree(summary_frame, height=6, col0_width=130, col1_width=280)

    pick_frame = tk.LabelFrame(detail_pane, text="Pick detail", padx=4, pady=4)
    detail_pane.add(pick_frame, weight=2)
    pick_cols = ("NSE Code", "Entry", "Exit", "Exit date", "P/L %")
    pick_tree = ttk.Treeview(
        pick_frame, columns=pick_cols, show="headings", height=8, selectmode="none"
    )
    for col in pick_cols:
        pick_tree.heading(col, text=col)
        w = 88 if col != "NSE Code" else 80
        anchor = "w" if col == "NSE Code" else "e"
        pick_tree.column(col, width=w, stretch=True, anchor=anchor)
    pick_scroll = ttk.Scrollbar(pick_frame, orient=tk.VERTICAL, command=pick_tree.yview)
    pick_tree.configure(yscrollcommand=pick_scroll.set)
    pick_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    pick_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    log_frame = tk.LabelFrame(bottom_pane, text="Rebalance log", padx=4, pady=4)
    bottom_pane.add(log_frame, weight=3)
    log_cols = (
        "Period",
        "Rebal",
        "End",
        "Open",
        "Exits",
        "Port%",
        "Bench%",
        "Drawdown%",
        "Value",
    )
    log_tree = ttk.Treeview(
        log_frame, columns=log_cols, show="headings", height=10, selectmode="browse"
    )
    for col in log_cols:
        log_tree.heading(col, text=col)
        w = 160 if col in ("Open", "Exits") else 88
        log_tree.column(col, width=w, stretch=True)
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_tree.yview)
    log_tree.configure(yscrollcommand=log_scroll.set)
    log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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

    def _universe_key() -> str:
        return "turnover" if universe_var.get().strip().lower() == "turnover" else "volume"

    def _rebalance_key() -> str:
        label = rebalance_var.get().strip().lower()
        if label == "daily":
            return "day"
        if label == "monthly":
            return "month"
        if label == "fortnight":
            return "fortnight"
        return "week"

    def _parse_capital() -> float:
        return float(capital_var.get().replace(",", "").strip())

    def _set_busy(busy: bool) -> None:
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

    def _update_metrics() -> None:
        if engine is None or engine.trades_df.empty:
            _fill_kv_tree(metrics_tree, None, empty="No results yet.")
            return
        cap = _parse_capital()
        m = compute_metrics(engine.trades_df, cap)
        _fill_kv_tree(
            metrics_tree,
            {str(k): str(v) for k, v in m.items()},
            empty="No results yet.",
        )

    def _update_chart() -> None:
        if not show_equity_var.get():
            ax.clear()
            canvas.draw_idle()
            return
        ax.clear()
        if engine is None or engine.trades_df.empty:
            ax.set_title(f"Portfolio vs {BENCHMARK_NSE}")
            canvas.draw_idle()
            return
        df = engine.trades_df
        cap = _parse_capital()
        port_vals = df["Portfolio_Value"].values
        bench_vals = cap * (1 + df["Bench_Return"]).cumprod()
        weeks = range(1, len(df) + 1)
        ax.plot(weeks, port_vals, label="Portfolio", color="#1565C0", linewidth=2)
        ax.plot(weeks, bench_vals, label=BENCHMARK_NSE, color="#757575", linewidth=1.5)
        if engine.current_week < engine.total_weeks:
            ax.axvline(engine.current_week, color="#E65100", linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Equity curve ({len(df)} period(s))")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        canvas.draw_idle()

    def _show_week_detail(record: dict | None) -> None:
        if record is None or engine is None:
            _fill_kv_tree(detail_tree, None, empty="Load data, then Next Period or Run All.")
            _fill_pick_tree(pick_tree, None)
            return
        cap = _parse_capital()
        cum_pl = backtest_cum_pl_pct(float(record["Portfolio_Value"]), cap)
        fields = {
            "Period": f"{record['Period']} of {engine.total_periods}",
            "Rebalance type": record.get("Rebalance", rebalance_var.get()),
            "Period start": rrg_format_date(record["Rebal_Date"]),
            "Period end": rrg_format_date(record["End_Date"]),
            "Universe": record.get("Universe", _universe_key()),
            "Final list size": str(record.get("Final_List_Count", "—")),
            "New entries": ", ".join(record.get("New_Entry_Tickers") or []) or "—",
            "50 DMA exits": ", ".join(record.get("Exit_Tickers") or []) or "—",
            "Open holdings": record.get("Holdings") or "CASH",
            "Portfolio % (period)": f"{record['Port_Return'] * 100:+.2f}",
            "Portfolio P/L % (since start)": f"{cum_pl:+.2f}",
            "Benchmark %": f"{record['Bench_Return'] * 100:+.2f}",
            "Portfolio value": f"{record['Portfolio_Value']:,.0f}",
        }
        _fill_kv_tree(detail_tree, fields, empty="—")
        closed = record.get("Closed_Rows") or []
        open_rows = [
            r
            for r in (record.get("Position_Rows") or [])
            if r.get("exit_reason") == "Open"
        ]
        _fill_pick_tree(pick_tree, closed + open_rows)

    def _refresh_log_table() -> None:
        for item in log_tree.get_children():
            log_tree.delete(item)
        if engine is None or engine.trades_df.empty:
            return
        dd_vals = backtest_drawdown_pct_series(engine.trades_df["Port_Return"]).values
        for pos, (_, row) in enumerate(engine.trades_df.iterrows()):
            dd_pct = float(dd_vals[pos]) if pos < len(dd_vals) else 0.0
            log_tree.insert(
                "",
                tk.END,
                iid=str(row["Period"]),
                values=(
                    row["Period"],
                    rrg_format_date(row["Rebal_Date"]),
                    rrg_format_date(row["End_Date"]),
                    row.get("Open_Count", "—"),
                    row.get("DMA_Exits", "—"),
                    f"{row['Port_Return'] * 100:+.2f}",
                    f"{row['Bench_Return'] * 100:+.2f}",
                    f"{dd_pct:+.2f}",
                    f"{row['Portfolio_Value']:,.0f}",
                ),
            )
        log_tree.yview_moveto(1.0)

    def _on_log_select(_event=None) -> None:
        if engine is None or engine.trades_df.empty:
            return
        sel = log_tree.selection()
        if not sel:
            return
        period = int(sel[0])
        row = engine.trades_df[engine.trades_df["Period"] == period]
        if row.empty:
            return
        _show_week_detail(row.iloc[0].to_dict())

    log_tree.bind("<<TreeviewSelect>>", _on_log_select)

    def _clear_results() -> None:
        _fill_kv_tree(metrics_tree, None, empty="No results yet.")
        _fill_kv_tree(detail_tree, None, empty="Load data, then Next Period or Run All.")
        _fill_pick_tree(pick_tree, None)
        for item in log_tree.get_children():
            log_tree.delete(item)
        _update_chart()

    def _after_step() -> None:
        _refresh_log_table()
        _update_metrics()
        _update_chart()
        if engine and engine.trades_df is not None and not engine.trades_df.empty:
            last = engine.trades_df.iloc[-1].to_dict()
            _show_week_detail(last)
            log_tree.selection_set(str(last["Period"]))
        _refresh_nav_buttons()
        if engine:
            status_var.set(
                f"Period {engine.current_period} / {engine.total_periods}"
                + (" — finished." if engine.finished else "")
            )

    def _build_engine() -> Top250BacktestEngine:
        return Top250BacktestEngine(
            Top250BacktestConfig(
                backtest_start=rrg_format_date(rrg_parse_user_date(start_var.get())),
                backtest_end=rrg_format_date(rrg_parse_user_date(end_var.get())),
                universe=_universe_key(),  # type: ignore[arg-type]
                top_n=int(top_n_var.get()),
                initial_capital=_parse_capital(),
                rebalance_freq=_rebalance_key(),  # type: ignore[arg-type]
            ),
            progress_cb=lambda msg: win.after(0, lambda m=msg: status_var.set(m)),
        )

    def _on_load_done(err: Exception | None) -> None:
        nonlocal engine, _load_run_all_after
        _set_busy(False)
        if err is not None:
            messagebox.showerror("Load failed", str(err), parent=win)
            status_var.set(f"Load failed: {err}")
            return
        status_var.set(
            f"Loaded {engine.total_periods} periods ({rebalance_var.get()}). "
            + ("Running all…" if _load_run_all_after else "Click Next Period or Run All.")
        )
        if _load_run_all_after:
            _load_run_all_after = False
            _on_run_all()
        else:
            _refresh_nav_buttons()

    def _on_load() -> None:
        nonlocal engine, _load_run_all_after
        if _busy:
            return
        try:
            _parse_capital()
            rrg_parse_user_date(start_var.get())
            rrg_parse_user_date(end_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=win)
            return

        _set_busy(True)
        _clear_results()
        engine = _build_engine()
        _load_run_all_after = mode_var.get() == "all"

        def worker() -> None:
            err: Exception | None = None
            try:
                engine.load_data()
            except Exception as exc:
                err = exc
            win.after(0, lambda: _on_load_done(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_next() -> None:
        if _busy or engine is None or not engine.loaded or engine.finished:
            return
        _set_busy(True)

        def worker() -> None:
            err: Exception | None = None
            try:
                engine.step_week()
            except Exception as exc:
                err = exc
            win.after(0, lambda: (_set_busy(False), _on_step_done(err)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_step_done(err: Exception | None) -> None:
        if err is not None:
            messagebox.showerror("Step failed", str(err), parent=win)
            status_var.set(f"Step failed: {err}")
            return
        _after_step()

    def _on_run_all() -> None:
        if _busy or engine is None or not engine.loaded:
            return
        _set_busy(True)
        status_var.set("Running all periods…")

        def worker() -> None:
            err: Exception | None = None
            try:
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

    load_btn.config(command=_on_load)
    next_btn.config(command=_on_next)
    prev_btn.config(command=_on_prev)
    run_all_btn.config(command=_on_run_all)
    reset_btn.config(command=_on_reset)

    show_equity_var.trace_add("write", lambda *_: _update_chart())

    return win


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    open_top250_backtest(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
