"""Tkinter UI for staggered dip-buying / profit-booking backtest."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.index.backtest_staggered_dip import (  # noqa: E402
    DEFAULT_INDEX_ID,
    StaggeredDipBacktestConfig,
    StaggeredDipBacktestEngine,
    _etf_display,
    compute_metrics,
    etfs_for_index,
    normalize_backtest_date,
    resolve_trade_etf,
    staggered_dip_index_choices,
)
from momentum.backtest_cancel import BacktestCancelled  # noqa: E402
from momentum.rrg_core import rrg_format_date, rrg_parse_user_date  # noqa: E402
from momentum.rrg_ui_copy import install_copy_support  # noqa: E402
from utils.nse_bhavcopy import today_ist  # noqa: E402


def _make_kv_tree(
    parent: tk.Misc,
    *,
    height: int,
    col0_width: int = 150,
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


def open_staggered_dip_backtest(
    parent: tk.Misc,
    *,
    initial_start: str | None = None,
    initial_end: str | None = None,
) -> tk.Toplevel:
    win = tk.Toplevel(parent)
    win.title("Staggered Dip Backtest")
    win.geometry("1100x780")
    win.minsize(900, 600)
    win.lift()

    engine: StaggeredDipBacktestEngine | None = None
    _busy = False
    _cancel_event: threading.Event | None = None

    _index_choices = staggered_dip_index_choices()
    _index_labels = [label for label, _ in _index_choices]
    _index_label_to_id = {label: index_id for label, index_id in _index_choices}

    params = tk.Frame(win, padx=10, pady=8)
    params.pack(fill=tk.X)

    tk.Label(params, text="Start:").grid(row=0, column=0, sticky="w")
    start_var = tk.StringVar(value=rrg_format_date(initial_start or f"{today_ist().year}-01-01"))
    start_entry = tk.Entry(params, textvariable=start_var, width=12)
    start_entry.grid(row=0, column=1, padx=(4, 16))

    tk.Label(params, text="End:").grid(row=0, column=2, sticky="w")
    end_var = tk.StringVar(value=rrg_format_date(initial_end or today_ist()))
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

        entry.bind("<FocusOut>", _on_leave)

    _register_date_entry(start_entry, start_var)
    _register_date_entry(end_entry, end_var)

    tk.Label(params, text="Capital:").grid(row=0, column=4, sticky="w")
    capital_var = tk.StringVar(value="500000")
    tk.Entry(params, textvariable=capital_var, width=10).grid(row=0, column=5, padx=(4, 16))

    tk.Label(params, text="Lots (N):").grid(row=0, column=6, sticky="w")
    lots_var = tk.StringVar(value="5")
    ttk.Spinbox(params, from_=1, to=20, width=4, textvariable=lots_var).grid(
        row=0, column=7, padx=(4, 16)
    )

    tk.Label(params, text="Profit X%:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    profit_var = tk.StringVar(value="5")
    tk.Entry(params, textvariable=profit_var, width=6).grid(row=1, column=1, padx=(4, 16), pady=(8, 0))

    tk.Label(params, text="Dip Y%:").grid(row=1, column=2, sticky="w", pady=(8, 0))
    dip_var = tk.StringVar(value="5")
    tk.Entry(params, textvariable=dip_var, width=6).grid(row=1, column=3, padx=(4, 16), pady=(8, 0))

    tk.Label(params, text="Index:").grid(row=1, column=4, sticky="w", pady=(8, 0))
    _default_index_label = next(
        (label for label, index_id in _index_choices if index_id == DEFAULT_INDEX_ID),
        _index_labels[0] if _index_labels else "",
    )
    index_var = tk.StringVar(value=_default_index_label)
    index_combo = ttk.Combobox(
        params,
        textvariable=index_var,
        values=_index_labels,
        width=22,
        state="readonly",
    )
    index_combo.grid(row=1, column=5, padx=(4, 16), pady=(8, 0), sticky="w")

    tk.Label(params, text="ETF:").grid(row=1, column=6, sticky="w", pady=(8, 0))
    etf_var = tk.StringVar()
    etf_combo = ttk.Combobox(
        params,
        textvariable=etf_var,
        width=28,
        state="readonly",
    )
    etf_combo.grid(row=1, column=7, padx=4, pady=(8, 0), sticky="w")

    _etf_label_to_ticker: dict[str, str] = {}

    def _refresh_etf_choices(*, keep_selection: bool = True) -> None:
        index_id = _index_label_to_id.get(index_var.get().strip(), DEFAULT_INDEX_ID)
        try:
            tickers = etfs_for_index(index_id)
        except ValueError:
            tickers = []
        _etf_label_to_ticker.clear()
        labels: list[str] = []
        for ticker in tickers:
            label = _etf_display(ticker)
            labels.append(label)
            _etf_label_to_ticker[label] = ticker
        etf_combo.config(values=labels)
        if not labels:
            etf_var.set("")
            return
        if keep_selection and etf_var.get().strip() in _etf_label_to_ticker:
            return
        etf_var.set(labels[0])

    _refresh_etf_choices(keep_selection=False)
    index_combo.bind("<<ComboboxSelected>>", lambda _e: _refresh_etf_choices(keep_selection=False))

    hint_row = tk.Frame(win, padx=10)
    hint_row.pack(fill=tk.X, pady=(0, 4))
    tk.Label(
        hint_row,
        text="Max N open lots. Dips extend ladder 1→N only; profit exit redeploys that slot next day. Size = cash ÷ free slots.",
        fg="#666",
    ).pack(anchor="w")

    btn_row = tk.Frame(win, pady=6)
    btn_row.pack(fill=tk.X, padx=8)
    status_var = tk.StringVar(value="Set parameters and Load Data.")
    tk.Label(win, textvariable=status_var, anchor="w", fg="#333", padx=10).pack(fill=tk.X)

    main_pane = ttk.PanedWindow(win, orient=tk.VERTICAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    chart_frame = tk.LabelFrame(main_pane, text="Equity curve", padx=4, pady=4)
    main_pane.add(chart_frame, weight=2)
    chart_fig = Figure(figsize=(9, 3.2), dpi=100)
    chart_canvas = FigureCanvasTkAgg(chart_fig, master=chart_frame)
    chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    body = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
    main_pane.add(body, weight=2)

    metrics_frame = tk.LabelFrame(body, text="Summary metrics", padx=4, pady=4)
    body.add(metrics_frame, weight=1)
    metrics_tree = _make_kv_tree(metrics_frame, height=16, col0_width=160, col1_width=200)

    log_frame = tk.LabelFrame(body, text="Trade log", padx=4, pady=4)
    body.add(log_frame, weight=2)
    log_cols = ("Entry", "Exit_Date", "Lot", "Instrument", "Entry_Px", "Exit_Px", "Amount", "PL_Rs", "PL_%", "Reason")
    _log_col_widths = {
        "Entry": 72,
        "Exit_Date": 72,
        "Lot": 44,
        "Instrument": 120,
        "Entry_Px": 72,
        "Exit_Px": 72,
        "Amount": 80,
        "PL_Rs": 72,
        "PL_%": 56,
        "Reason": 280,
    }
    log_table = tk.Frame(log_frame)
    log_table.pack(fill=tk.BOTH, expand=True)
    log_tree = ttk.Treeview(log_table, columns=log_cols, show="headings", height=14, selectmode="browse")
    _log_headers = {
        "Entry": "Entry",
        "Exit_Date": "Exit",
        "Lot": "Lot",
        "Instrument": "Instrument",
        "Entry_Px": "Entry ₹",
        "Exit_Px": "Exit / Mark",
        "Amount": "Amount",
        "PL_Rs": "P/L ₹",
        "PL_%": "P/L %",
        "Reason": "Reason",
    }
    for col in log_cols:
        log_tree.heading(col, text=_log_headers[col])
        w = _log_col_widths.get(col, 72)
        log_tree.column(
            col,
            width=w,
            minwidth=120 if col == "Reason" else w,
            stretch=(col in ("Instrument", "Reason")),
            anchor="w" if col in ("Instrument", "Reason") else "e",
        )
    log_v_scroll = ttk.Scrollbar(log_table, orient=tk.VERTICAL, command=log_tree.yview)
    log_h_scroll = ttk.Scrollbar(log_table, orient=tk.HORIZONTAL, command=log_tree.xview)
    log_tree.configure(yscrollcommand=log_v_scroll.set, xscrollcommand=log_h_scroll.set)
    log_tree.grid(row=0, column=0, sticky="nsew")
    log_v_scroll.grid(row=0, column=1, sticky="ns")
    log_h_scroll.grid(row=1, column=0, sticky="ew")
    log_table.grid_rowconfigure(0, weight=1)
    log_table.grid_columnconfigure(0, weight=1)

    reason_detail = tk.Text(
        log_frame,
        height=3,
        wrap=tk.WORD,
        state=tk.DISABLED,
        font=("Segoe UI", 9),
        relief=tk.GROOVE,
        padx=6,
        pady=4,
    )
    reason_detail.pack(fill=tk.X, pady=(6, 0))

    def _set_reason_detail(text: str) -> None:
        reason_detail.config(state=tk.NORMAL)
        reason_detail.delete("1.0", tk.END)
        reason_detail.insert(tk.END, text or "—")
        reason_detail.config(state=tk.DISABLED)

    def _on_log_select(_event=None) -> None:
        sel = log_tree.selection()
        if not sel:
            _set_reason_detail("Select a row to see the full reason.")
            return
        values = log_tree.item(sel[0], "values")
        _set_reason_detail(str(values[-1]) if values else "—")

    log_tree.bind("<<TreeviewSelect>>", _on_log_select)
    _set_reason_detail("Select a row to see the full reason.")

    load_btn = ttk.Button(btn_row, text="Load Data")
    load_btn.pack(side=tk.LEFT, padx=4)
    cancel_btn = ttk.Button(btn_row, text="Cancel", state=tk.DISABLED)
    cancel_btn.pack(side=tk.LEFT, padx=4)
    run_all_btn = ttk.Button(btn_row, text="Run All", state=tk.DISABLED)
    run_all_btn.pack(side=tk.LEFT, padx=4)
    reset_btn = ttk.Button(btn_row, text="Reset Run", state=tk.DISABLED)
    reset_btn.pack(side=tk.LEFT, padx=4)

    install_copy_support(win)

    def _parse_capital() -> float:
        return float(capital_var.get().replace(",", "").strip())

    def _parse_float(var: tk.StringVar, name: str, *, min_val: float = 0.1) -> float:
        try:
            val = float(var.get().strip())
        except ValueError as exc:
            raise ValueError(f"Enter a valid {name}.") from exc
        if val < min_val:
            raise ValueError(f"{name} must be at least {min_val}.")
        return val

    def _parse_lots() -> int:
        try:
            val = int(lots_var.get().strip())
        except ValueError as exc:
            raise ValueError("Enter a valid lot count (N).") from exc
        if val < 1:
            raise ValueError("Lots (N) must be at least 1.")
        return val

    def _selected_index_id() -> str:
        label = index_var.get().strip()
        index_id = _index_label_to_id.get(label)
        if not index_id:
            raise ValueError("Select an index.")
        return index_id

    def _selected_trade_etf() -> str:
        label = etf_var.get().strip()
        ticker = _etf_label_to_ticker.get(label)
        if not ticker:
            raise ValueError("Select an ETF for the chosen index.")
        return resolve_trade_etf(_selected_index_id(), ticker)

    def _build_config() -> StaggeredDipBacktestConfig:
        return StaggeredDipBacktestConfig(
            backtest_start=normalize_backtest_date(start_var.get()),
            backtest_end=normalize_backtest_date(end_var.get()),
            num_lots=_parse_lots(),
            profit_pct=_parse_float(profit_var, "Profit X%"),
            dip_pct=_parse_float(dip_var, "Dip Y%"),
            initial_capital=_parse_capital(),
            index_id=_selected_index_id(),
            trade_etf=_selected_trade_etf(),
        )

    def _set_busy(busy: bool) -> None:
        nonlocal _busy
        _busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        load_btn.config(state=state)
        cancel_btn.config(state=tk.NORMAL if busy else tk.DISABLED)
        run_all_btn.config(state=state)
        reset_btn.config(state=state)
        win.config(cursor="watch" if busy else "")
        if not busy:
            _refresh_action_buttons()

    def _refresh_action_buttons() -> None:
        can_run = not _busy and engine is not None and engine.loaded
        reset_btn.config(state=tk.NORMAL if can_run else tk.DISABLED)
        run_all_btn.config(
            state=tk.NORMAL if can_run and engine is not None and not engine.finished else tk.DISABLED
        )

    def _update_equity_chart() -> None:
        chart_fig.clear()
        ax = chart_fig.add_subplot(111)
        if engine is None or engine.trades_df.empty:
            ax.set_title("Run backtest to see equity curve")
            ax.axis("off")
        else:
            df = engine.trades_df
            dates = pd.to_datetime(df["Bar_Date"])
            ax.plot(dates, df["Portfolio_Value"], label="Strategy", color="#1565C0", linewidth=2)
            ax.plot(
                dates,
                df["Bench_Value"],
                label=f"{df.attrs.get('index_label', 'Index')} buy & hold",
                color="#757575",
                linewidth=1.5,
            )
            ax.set_title(
                f"{df.attrs.get('index_label')} · N={df.attrs.get('num_lots')} · "
                f"X={df.attrs.get('profit_pct')}% · Y={df.attrs.get('dip_pct')}%"
            )
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            chart_fig.autofmt_xdate()
        chart_canvas.draw_idle()

    def _update_metrics() -> None:
        if engine is None or engine.trades_df.empty:
            _fill_kv_tree(metrics_tree, None, empty="Click Run All after Load Data.")
            return
        rows = {str(k): str(v) for k, v in compute_metrics(engine.trades_df, _parse_capital()).items()}
        _fill_kv_tree(metrics_tree, rows, empty="Click Run All after Load Data.")

    def _refresh_log_table() -> None:
        for item in log_tree.get_children():
            log_tree.delete(item)
        if engine is None or engine.trade_log_df.empty:
            _set_reason_detail("Select a row to see the full reason.")
            return
        first_iid: str | None = None
        for _, row in engine.trade_log_df.iterrows():
            pl_rs = row.get("PL_Amt")
            pl_pct = row.get("PL_%")
            exit_dt = row.get("Exit_Date")
            exit_px = row.get("Exit")
            iid = log_tree.insert(
                "",
                tk.END,
                values=(
                    rrg_format_date(row["Date"]),
                    rrg_format_date(exit_dt) if exit_dt is not None and pd.notna(exit_dt) else "—",
                    row["Lot"],
                    row["Instrument"],
                    f"{float(row['Entry']):,.2f}" if row.get("Entry") is not None else "—",
                    f"{float(exit_px):,.2f}" if exit_px is not None and pd.notna(exit_px) else "—",
                    f"{float(row['Amount']):,.0f}" if row.get("Amount") is not None else "—",
                    f"{float(pl_rs):+,.0f}" if pl_rs is not None and pd.notna(pl_rs) else "—",
                    f"{float(pl_pct):+.2f}" if pl_pct is not None and pd.notna(pl_pct) else "—",
                    row.get("Reason", ""),
                ),
            )
            if first_iid is None:
                first_iid = iid
        if first_iid:
            log_tree.selection_set(first_iid)
            log_tree.focus(first_iid)
            _on_log_select()

    def _after_run() -> None:
        _update_equity_chart()
        _update_metrics()
        _refresh_log_table()
        _refresh_action_buttons()
        if engine is not None:
            status_var.set(
                f"Done — {engine.total_periods} days, "
                f"final {engine._portfolio_value:,.0f}"
            )

    def _run_worker(*, load_only: bool, run_all: bool) -> None:
        nonlocal engine, _cancel_event
        try:
            cfg = _build_config()
            if engine is None or load_only:
                engine = StaggeredDipBacktestEngine(
                    cfg,
                    progress_cb=lambda msg: win.after(0, status_var.set, msg),
                    cancel_check=lambda: bool(_cancel_event and _cancel_event.is_set()),
                )
                engine.load_data()
                win.after(0, lambda: status_var.set("Data loaded. Click Run All."))
            else:
                engine.apply_run_context(cfg)
            if run_all:
                engine.run_all()
                win.after(0, _after_run)
            else:
                win.after(0, _refresh_action_buttons)
        except BacktestCancelled:
            win.after(0, lambda: status_var.set("Cancelled."))
        except Exception as exc:
            win.after(0, lambda: messagebox.showerror("Error", str(exc), parent=win))
        finally:
            win.after(0, lambda: _set_busy(False))

    def _start_task(*, load_only: bool = False, run_all: bool = False) -> None:
        nonlocal _cancel_event
        if _busy:
            return
        try:
            _build_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=win)
            return
        _cancel_event = threading.Event()
        _set_busy(True)
        threading.Thread(
            target=_run_worker,
            kwargs={"load_only": load_only, "run_all": run_all},
            daemon=True,
        ).start()

    def _on_load() -> None:
        _start_task(load_only=True, run_all=False)

    def _on_run_all() -> None:
        _start_task(load_only=False, run_all=True)

    def _on_reset() -> None:
        if engine is None or not engine.loaded:
            return
        try:
            engine.apply_run_context(_build_config())
            _update_equity_chart()
            _update_metrics()
            _refresh_log_table()
            status_var.set("Run reset. Click Run All.")
            _refresh_action_buttons()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=win)

    def _on_cancel() -> None:
        if _cancel_event is not None:
            _cancel_event.set()

    load_btn.config(command=_on_load)
    run_all_btn.config(command=_on_run_all)
    reset_btn.config(command=_on_reset)
    cancel_btn.config(command=_on_cancel)

    return win


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    open_staggered_dip_backtest(root)
    root.mainloop()


if __name__ == "__main__":
    main()
