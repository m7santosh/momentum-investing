"""Tkinter backtest UI for Nifty index technical-indicator signals."""

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

from backtest.index.backtest_nifty_candlestick import (  # noqa: E402
    CANDLE_MODE_LABELS,
    CANDLE_MODES,
    NiftyCandleBacktestConfig,
    NiftyCandleBacktestEngine,
    _format_display_date,
    compute_metrics,
    normalize_backtest_date,
)
from momentum.backtest_cancel import BacktestCancelled  # noqa: E402
from momentum.index.candle_signals import TIMEFRAME_LABELS  # noqa: E402
from momentum.index.index_indicators import (  # noqa: E402
    DEFAULT_INDICATOR,
    DEFAULT_INDICATOR_PERIOD,
    DEFAULT_SUPERTREND_ATR,
    DEFAULT_SUPERTREND_MULTIPLIER,
    INDICATOR_LABELS,
    resolve_indicator,
)
from momentum.index.nifty_indices import (  # noqa: E402
    DEFAULT_BENCHMARK_KEY,
    DEFAULT_SELECTED_INDEX_IDS,
    NIFTY_BENCHMARKS,
    NIFTY_INDICES,
)
from momentum.index.candle_plot import CandleChartHover, plot_index_with_indicator  # noqa: E402
from momentum.rrg_backtest_ui import backtest_drawdown_pct_series  # noqa: E402
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


def _fill_kv_tree(tree: ttk.Treeview, rows: dict[str, str] | None, *, empty: str) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if not rows:
        tree.insert("", tk.END, values=("—", empty))
        return
    for key, value in rows.items():
        tree.insert("", tk.END, values=(key, value))


def open_nifty_candlestick_backtest(
    parent: tk.Misc,
    *,
    initial_start: str | None = None,
    initial_end: str | None = None,
    initial_mode: str = "candlestick",
) -> tk.Toplevel:
    win = tk.Toplevel(parent)
    win.title("Nifty Index Backtest")
    win.geometry("1140x880")
    win.minsize(900, 640)
    win.lift()

    engine: NiftyCandleBacktestEngine | None = None
    compare_engine: NiftyCandleBacktestEngine | None = None
    _busy = False
    _load_run_all_after = False
    _cancel_event: threading.Event | None = None

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
                entry.focus_set()
                entry.selection_range(0, tk.END)

        entry.bind("<FocusOut>", _on_leave)

    _register_date_entry(start_entry, start_var)
    _register_date_entry(end_entry, end_var)

    tk.Label(params, text="Chart:").grid(row=0, column=4, sticky="w")
    mode_var = tk.StringVar(value=CANDLE_MODE_LABELS.get(initial_mode, "Candlestick"))
    candle_combo = ttk.Combobox(
        params,
        textvariable=mode_var,
        values=[CANDLE_MODE_LABELS[m] for m in CANDLE_MODES],
        width=14,
        state="readonly",
    )
    candle_combo.grid(row=0, column=5, padx=(4, 16))

    tk.Label(params, text="Capital:").grid(row=0, column=6, sticky="w")
    capital_var = tk.StringVar(value="500000")
    tk.Entry(params, textvariable=capital_var, width=10).grid(row=0, column=7, padx=4)

    tk.Label(params, text="Timeframe:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    timeframe_var = tk.StringVar(value="Daily")
    ttk.Combobox(
        params,
        textvariable=timeframe_var,
        values=tuple(TIMEFRAME_LABELS.values()),
        width=10,
        state="readonly",
    ).grid(row=1, column=1, padx=(4, 16), pady=(8, 0), sticky="w")

    tk.Label(params, text="Indicator:").grid(row=1, column=2, sticky="w", pady=(8, 0))
    indicator_var = tk.StringVar(value=INDICATOR_LABELS[DEFAULT_INDICATOR])
    indicator_combo = ttk.Combobox(
        params,
        textvariable=indicator_var,
        values=tuple(INDICATOR_LABELS.values()),
        width=14,
        state="readonly",
    )
    indicator_combo.grid(row=1, column=3, padx=(4, 16), pady=(8, 0), sticky="w")

    period_label = tk.Label(params, text="Period:")
    period_label.grid(row=1, column=4, sticky="w", pady=(8, 0))
    period_var = tk.StringVar(value=str(DEFAULT_INDICATOR_PERIOD))
    period_spin = ttk.Spinbox(params, from_=2, to=200, width=5, textvariable=period_var)
    period_spin.grid(row=1, column=5, padx=(4, 8), pady=(8, 0), sticky="w")

    st_mult_label = tk.Label(params, text="Mult:")
    st_mult_var = tk.StringVar(value=str(DEFAULT_SUPERTREND_MULTIPLIER))
    st_mult_entry = tk.Entry(params, textvariable=st_mult_var, width=5)
    st_mult_label.grid(row=1, column=6, sticky="w", pady=(8, 0))
    st_mult_entry.grid(row=1, column=7, padx=(4, 16), pady=(8, 0), sticky="w")

    tk.Label(params, text="Benchmark:").grid(row=2, column=0, sticky="w", pady=(8, 0))
    _bench_labels = [b.label for b in NIFTY_BENCHMARKS.values()]
    benchmark_var = tk.StringVar(value=NIFTY_BENCHMARKS[DEFAULT_BENCHMARK_KEY].label)
    benchmark_combo = ttk.Combobox(
        params,
        textvariable=benchmark_var,
        values=_bench_labels,
        width=14,
        state="readonly",
    )
    benchmark_combo.grid(row=2, column=1, padx=(4, 16), pady=(8, 0), sticky="w")

    index_row = tk.Frame(win, padx=10)
    index_row.pack(fill=tk.X, pady=(0, 4))

    tk.Label(index_row, text="Indices:").pack(side=tk.LEFT, anchor="n", pady=(4, 0))
    index_list_frame = tk.Frame(index_row)
    index_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 8))

    _index_labels = sorted(i.label for i in NIFTY_INDICES)
    _index_label_to_id = {i.label: i.index_id for i in NIFTY_INDICES}

    index_listbox = tk.Listbox(
        index_list_frame,
        selectmode=tk.EXTENDED,
        height=5,
        exportselection=False,
    )
    index_scroll = ttk.Scrollbar(index_list_frame, orient=tk.VERTICAL, command=index_listbox.yview)
    index_listbox.configure(yscrollcommand=index_scroll.set)
    index_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
    index_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    for label in _index_labels:
        index_listbox.insert(tk.END, label)

    _default_label_set = {
        i.label for i in NIFTY_INDICES if i.index_id in DEFAULT_SELECTED_INDEX_IDS
    }
    for pos, label in enumerate(_index_labels):
        if label in _default_label_set:
            index_listbox.selection_set(pos)

    index_btn_frame = tk.Frame(index_row)
    index_btn_frame.pack(side=tk.LEFT, anchor="n")

    def _select_all_indices() -> None:
        index_listbox.selection_set(0, tk.END)

    def _clear_indices() -> None:
        index_listbox.selection_clear(0, tk.END)

    ttk.Button(index_btn_frame, text="Select all", command=_select_all_indices).pack(fill=tk.X, pady=(0, 4))
    ttk.Button(index_btn_frame, text="Clear", command=_clear_indices).pack(fill=tk.X)
    tk.Label(
        index_row,
        text="Multi-select for backtest; click one index to view its chart",
        fg="#666",
    ).pack(side=tk.LEFT, anchor="n", pady=(4, 0))

    btn_row = tk.Frame(win, pady=6)
    btn_row.pack(fill=tk.X, padx=8)

    status_var = tk.StringVar(value="Set dates and Load Data.")
    tk.Label(win, textvariable=status_var, anchor="w", fg="#333", padx=10).pack(fill=tk.X)

    main_pane = ttk.PanedWindow(win, orient=tk.VERTICAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    chart_frame = tk.LabelFrame(main_pane, text="Index chart", padx=4, pady=4)
    main_pane.add(chart_frame, weight=3)

    chart_detail_var = tk.StringVar(value=CandleChartHover.DEFAULT_DETAIL)
    tk.Label(
        chart_frame,
        textvariable=chart_detail_var,
        anchor="w",
        justify=tk.LEFT,
        fg="#333",
        bg="#f5f5f5",
        padx=6,
        pady=4,
        wraplength=1050,
    ).pack(fill=tk.X, pady=(0, 4))

    chart_fig = Figure(figsize=(9, 3.8), dpi=100)
    chart_canvas = FigureCanvasTkAgg(chart_fig, master=chart_frame)
    chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    candle_hover = CandleChartHover(chart_canvas, set_detail=chart_detail_var.set)

    body = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
    main_pane.add(body, weight=2)

    metrics_frame = tk.LabelFrame(body, text="Summary metrics", padx=4, pady=4)
    body.add(metrics_frame, weight=1)
    metrics_tree = _make_kv_tree(metrics_frame, height=14, col0_width=150, col1_width=200)

    log_frame = tk.LabelFrame(body, text="Index log", padx=4, pady=4)
    body.add(log_frame, weight=2)
    log_cols = ("Bar", "Date", "Held", "Entries", "Exits", "Port%", "Bench%", "DD%", "Value")
    log_tree = ttk.Treeview(log_frame, columns=log_cols, show="headings", height=14, selectmode="browse")
    for col in log_cols:
        log_tree.heading(col, text=col)
        log_tree.column(col, width=88 if col != "Date" else 96, stretch=True)
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_tree.yview)
    log_tree.configure(yscrollcommand=log_scroll.set)
    log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    load_btn = ttk.Button(btn_row, text="Load Data")
    load_btn.pack(side=tk.LEFT, padx=4)
    cancel_btn = ttk.Button(btn_row, text="Cancel", state=tk.DISABLED)
    cancel_btn.pack(side=tk.LEFT, padx=4)
    run_all_btn = ttk.Button(btn_row, text="Run All", state=tk.DISABLED)
    run_all_btn.pack(side=tk.LEFT, padx=4)
    reset_btn = ttk.Button(btn_row, text="Reset Run", state=tk.DISABLED)
    reset_btn.pack(side=tk.LEFT, padx=4)

    install_copy_support(win)

    _last_indicator_key: list[str] = [DEFAULT_INDICATOR]

    def _indicator_key() -> str:
        label = indicator_var.get().strip()
        for key, text in INDICATOR_LABELS.items():
            if text == label:
                return key
        return resolve_indicator(label)

    def _parse_period() -> int | None:
        raw = period_var.get().strip()
        if not raw:
            return None
        try:
            val = int(raw)
        except ValueError:
            return None
        return val if 2 <= val <= 200 else None

    def _require_period() -> int:
        val = _parse_period()
        if val is None:
            raise ValueError("Enter a valid period (2–200).")
        return val

    def _parse_supertrend_multiplier() -> float | None:
        raw = st_mult_var.get().strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError:
            return None
        return val if val > 0 else None

    def _require_supertrend_multiplier() -> float:
        val = _parse_supertrend_multiplier()
        if val is None:
            raise ValueError("Enter a valid Supertrend multiplier (> 0).")
        return val

    def _update_indicator_controls(_event=None) -> None:
        ind = _indicator_key()
        is_candle = ind == "candle"
        is_st = ind == "supertrend"

        if is_st and _last_indicator_key[0] != "supertrend":
            period_var.set(str(DEFAULT_SUPERTREND_ATR))
            st_mult_var.set(str(DEFAULT_SUPERTREND_MULTIPLIER))
        elif not is_st and _last_indicator_key[0] == "supertrend":
            period_var.set(str(DEFAULT_INDICATOR_PERIOD))
        _last_indicator_key[0] = ind

        chart_values = [CANDLE_MODE_LABELS[m] for m in CANDLE_MODES]
        if is_candle:
            chart_values = chart_values + ["Compare Both"]
        candle_combo.config(values=chart_values, state="readonly")
        if not is_candle and mode_var.get().strip() == "Compare Both":
            mode_var.set(CANDLE_MODE_LABELS["candlestick"])

        period_label.config(text="ATR:" if is_st else "Period:")
        if is_candle:
            period_spin.config(state="disabled")
        else:
            period_spin.config(state="normal")

        if is_st:
            st_mult_label.grid()
            st_mult_entry.grid()
        else:
            st_mult_label.grid_remove()
            st_mult_entry.grid_remove()

    def _on_indicator_changed(_event=None) -> None:
        _update_indicator_controls()
        _on_chart_controls_changed()

    indicator_combo.bind("<<ComboboxSelected>>", _on_indicator_changed)

    def _mode_key() -> str:
        label = mode_var.get().strip()
        if label == "Compare Both":
            return "compare"
        for key, text in CANDLE_MODE_LABELS.items():
            if text == label:
                return key
        return "candlestick"

    def _chart_candle_mode() -> str:
        """How price bars are drawn on the chart (independent of signal indicator)."""
        mode = _mode_key()
        return "candlestick" if mode == "compare" else mode

    def _chart_index_label() -> str | None:
        sel = index_listbox.curselection()
        if not sel:
            return None
        return _index_labels[sel[-1]]

    def _timeframe_key() -> str:
        label = timeframe_var.get().strip()
        for key, text in TIMEFRAME_LABELS.items():
            if text == label:
                return key
        return "day"

    def _parse_capital() -> float:
        return float(capital_var.get().replace(",", "").strip())

    def _benchmark_key() -> str:
        label = benchmark_var.get().strip()
        for bench in NIFTY_BENCHMARKS.values():
            if bench.label == label:
                return bench.key
        return DEFAULT_BENCHMARK_KEY

    def _selected_index_ids() -> tuple[str, ...]:
        sel = index_listbox.curselection()
        if not sel:
            raise ValueError("Select at least one index.")
        return tuple(_index_label_to_id[_index_labels[i]] for i in sel)

    def _build_config(candle_mode: str) -> NiftyCandleBacktestConfig:
        ind = _indicator_key()
        return NiftyCandleBacktestConfig(
            backtest_start=normalize_backtest_date(start_var.get()),
            backtest_end=normalize_backtest_date(end_var.get()),
            candle_mode=candle_mode,  # type: ignore[arg-type]
            selected_index_ids=_selected_index_ids(),
            indicator=ind,  # type: ignore[arg-type]
            indicator_period=_require_period(),
            supertrend_multiplier=(
                _require_supertrend_multiplier()
                if ind == "supertrend"
                else DEFAULT_SUPERTREND_MULTIPLIER
            ),
            initial_capital=_parse_capital(),
            timeframe=_timeframe_key(),  # type: ignore[arg-type]
            benchmark_key=_benchmark_key(),
        )

    def _active_engine() -> NiftyCandleBacktestEngine | None:
        return engine

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
        eng = _active_engine()
        can_run = not _busy and eng is not None and eng.loaded
        reset_btn.config(state=tk.NORMAL if can_run else tk.DISABLED)
        run_all_btn.config(state=tk.NORMAL if can_run and not eng.finished else tk.DISABLED)

    def _update_metrics() -> None:
        eng = _active_engine()
        if eng is None or eng.trades_df.empty:
            _fill_kv_tree(metrics_tree, None, empty="Click Run All after Load Data.")
            return
        cap = _parse_capital()
        rows: dict[str, str] = {}
        m = compute_metrics(eng.trades_df, cap)
        for k, v in m.items():
            rows[str(k)] = str(v)
        if compare_engine is not None and not compare_engine.trades_df.empty:
            m2 = compute_metrics(compare_engine.trades_df, cap)
            rows["--- Heikin Ashi ---"] = ""
            for k, v in m2.items():
                if k in ("Benchmark", "Period", "Timeframe", "Indices", "Index_List", "Indicator", "Candle_Mode"):
                    continue
                rows[f"HA {k}"] = str(v)
        _fill_kv_tree(metrics_tree, rows, empty="Click Run All after Load Data.")

    def _ohlc_for_chart_label(label: str):
        eng = _active_engine()
        if eng is None or not eng.loaded:
            return None
        for idx in eng._universe:
            if idx.label == label:
                return eng.ohlc_for_chart(idx.yahoo_ticker)
        return None

    def _chart_display_range() -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Use current UI dates for chart slicing (matches Start/End fields)."""
        try:
            return (
                pd.Timestamp(normalize_backtest_date(start_var.get())),
                pd.Timestamp(normalize_backtest_date(end_var.get())),
            )
        except ValueError:
            eng = _active_engine()
            if eng is not None and eng.loaded:
                return eng.chart_display_range()
            return None, None

    def _update_index_chart() -> None:
        label = _chart_index_label()
        eng = _active_engine()
        ohlc = _ohlc_for_chart_label(label) if label and eng is not None and eng.loaded else None
        ind = _indicator_key()
        period = _parse_period()
        if period is None:
            return
        if ind == "supertrend":
            st_mult = _parse_supertrend_multiplier()
            if st_mult is None:
                return
        else:
            st_mult = DEFAULT_SUPERTREND_MULTIPLIER
        tf_label = timeframe_var.get()
        signal_candle_mode = _chart_candle_mode()
        if ind == "candle":
            mode = _mode_key()
            signal_candle_mode = "candlestick" if mode == "compare" else mode

        hover = plot_index_with_indicator(
            chart_fig,
            ohlc if ohlc is not None else pd.DataFrame(),
            index_label=label or "Index",
            indicator=ind,  # type: ignore[arg-type]
            candle_mode=signal_candle_mode,  # type: ignore[arg-type]
            chart_candle_mode=_chart_candle_mode(),  # type: ignore[arg-type]
            period=period,
            supertrend_multiplier=st_mult,
            timeframe=_timeframe_key(),  # type: ignore[arg-type]
            timeframe_label=tf_label,
            mark_signals=eng is not None and eng.loaded,
            display_start=_chart_display_range()[0] if eng is not None and eng.loaded else None,
            display_end=_chart_display_range()[1] if eng is not None and eng.loaded else None,
        )
        candle_hover.update(hover)
        chart_canvas.draw_idle()

    def _on_chart_controls_changed(_event=None) -> None:
        _update_index_chart()

    def _refresh_log_table() -> None:
        for item in log_tree.get_children():
            log_tree.delete(item)
        eng = _active_engine()
        if eng is None or eng.trades_df.empty:
            return
        dd_vals = backtest_drawdown_pct_series(eng.trades_df["Port_Return"]).values
        for pos, (_, row) in enumerate(eng.trades_df.iterrows()):
            dd_pct = float(dd_vals[pos]) if pos < len(dd_vals) else 0.0
            log_tree.insert(
                "",
                tk.END,
                iid=str(row["Period"]),
                values=(
                    row["Period"],
                    rrg_format_date(row["Bar_Date"]),
                    row.get("Held", "—"),
                    row.get("New_Entries", "—"),
                    row.get("Signal_Exits", "—"),
                    f"{row['Port_Return'] * 100:+.2f}",
                    f"{row['Bench_Return'] * 100:+.2f}",
                    f"{dd_pct:+.2f}",
                    f"{row['Portfolio_Value']:,.0f}",
                ),
            )
        log_tree.yview_moveto(1.0)

    def _after_load(*, ran_backtest: bool = False) -> None:
        _update_index_chart()
        _update_metrics()
        _refresh_log_table()
        _refresh_action_buttons()
        eng = _active_engine()
        if eng is not None and eng.loaded:
            tf = timeframe_var.get()
            n_bars = eng.total_periods
            if ran_backtest and not eng.trades_df.empty:
                status_var.set(f"Done — {n_bars} {tf.lower()} bar(s), backtest complete.")
            elif n_bars == 0:
                status_var.set("No bars in date range — check start/end dates.")
            else:
                msg = f"Loaded {n_bars} {tf.lower()} bar(s)"
                if eng.data_starts_after_backtest() and eng.first_bar_date is not None:
                    msg += (
                        f" — NSE data from {_format_display_date(eng.first_bar_date)} "
                        f"(requested {start_var.get()})"
                    )
                msg += ". Chart updated — click Run All for backtest."
                status_var.set(msg)

    def _load_worker() -> None:
        nonlocal engine, compare_engine, _cancel_event
        try:
            mode = _mode_key()
            primary = "candlestick" if mode == "compare" else mode
            engine = NiftyCandleBacktestEngine(
                _build_config(primary),
                progress_cb=lambda msg: win.after(0, status_var.set, msg),
                cancel_check=(lambda: _cancel_event.is_set() if _cancel_event else False),
            )
            engine.load_data()
            compare_engine = None
            if mode == "compare" and _indicator_key() == "candle":
                win.after(0, status_var.set, "Compare: loading Heikin Ashi engine...")
                compare_engine = NiftyCandleBacktestEngine(
                    _build_config("heikin_ashi"),
                    progress_cb=lambda msg: win.after(0, status_var.set, msg),
                    cancel_check=(lambda: _cancel_event.is_set() if _cancel_event else False),
                )
                compare_engine.load_data()
            if _load_run_all_after:
                engine.run_all()
                if compare_engine is not None:
                    compare_engine.run_all()
            win.after(0, lambda: _after_load(ran_backtest=_load_run_all_after))
        except BacktestCancelled:
            win.after(0, lambda: status_var.set("Cancelled."))
        except Exception as exc:
            win.after(0, lambda: messagebox.showerror("Load failed", str(exc), parent=win))
            win.after(0, lambda: status_var.set(f"Error: {exc}"))
        finally:
            win.after(0, lambda: _set_busy(False))

    def _on_load(*, run_all: bool = False) -> None:
        nonlocal _load_run_all_after, _cancel_event
        if _busy:
            return
        try:
            _selected_index_ids()
            _require_period()
            if _indicator_key() == "supertrend":
                _require_supertrend_multiplier()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc), parent=win)
            return
        _load_run_all_after = run_all
        _cancel_event = threading.Event()
        _set_busy(True)
        status_var.set("Loading market data...")
        threading.Thread(target=_load_worker, daemon=True).start()

    def _on_cancel() -> None:
        if _cancel_event:
            _cancel_event.set()
            status_var.set("Cancelling...")

    def _on_run_all() -> None:
        eng = _active_engine()
        if _busy or eng is None or not eng.loaded:
            return
        _set_busy(True)
        status_var.set("Running backtest...")

        def _worker() -> None:
            try:
                eng.run_all()
                if _mode_key() == "compare" and compare_engine is not None and _indicator_key() == "candle":
                    compare_engine.run_all()
                win.after(0, lambda: status_var.set("Done."))
                win.after(0, lambda: _after_load(ran_backtest=True))
            except Exception as exc:
                win.after(0, lambda: messagebox.showerror("Run failed", str(exc), parent=win))
            finally:
                win.after(0, lambda: _set_busy(False))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_reset() -> None:
        eng = _active_engine()
        if _busy or eng is None or not eng.loaded:
            return
        eng.reset_run()
        if compare_engine is not None:
            compare_engine.reset_run()
        _update_index_chart()
        _update_metrics()
        _refresh_log_table()
        _refresh_action_buttons()
        status_var.set("Run reset.")

    index_listbox.bind("<<ListboxSelect>>", _on_chart_controls_changed)
    timeframe_var.trace_add("write", lambda *_: _on_chart_controls_changed())
    period_var.trace_add("write", lambda *_: _on_chart_controls_changed())
    st_mult_var.trace_add("write", lambda *_: _on_chart_controls_changed())
    mode_var.trace_add("write", lambda *_: _on_chart_controls_changed())

    plot_index_with_indicator(
        chart_fig,
        pd.DataFrame(),
        index_label="Index",
        indicator="sma",
        chart_candle_mode="candlestick",
        timeframe_label="Daily",
        mark_signals=False,
    )
    candle_hover.update(None)
    chart_canvas.draw_idle()
    _update_indicator_controls()

    load_btn.config(command=lambda: _on_load(run_all=False))
    cancel_btn.config(command=_on_cancel)
    run_all_btn.config(command=_on_run_all)
    reset_btn.config(command=_on_reset)

    return win


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    open_nifty_candlestick_backtest(root)
    root.mainloop()


if __name__ == "__main__":
    main()
