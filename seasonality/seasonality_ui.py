from __future__ import annotations

import sys
import threading
import tkinter as tk
from datetime import date, timedelta
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seasonality.seasonality_simple import (
    MONTH_NAMES,
    analyze_index,
    calculate_monthly_returns,
    fetch_index_ohlc_history,
)
from seasonality import seasonality_backtest as backtest_module
from utils.nse_bhavcopy import list_nse_index_names


DEFAULT_INDEX_OPTIONS = [
    "All major indices",
    "Nifty 50",
    "Nifty Bank",
    "Nifty IT",
    "Nifty Pharma",
    "Nifty Auto",
    "Nifty Energy",
    "Nifty FMCG",
    "Nifty Metal",
    "Nifty Realty",
]


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _format_rate(value: float) -> str:
    return f"{value:.1f}%"


def _filter_full_years(monthly_returns: "pd.DataFrame", current_year_allowed: bool = True) -> "pd.DataFrame":
    if monthly_returns is None or monthly_returns.empty:
        return monthly_returns
    counts = monthly_returns.groupby('Year')['Month'].nunique()
    full_years = set(counts[counts == 12].index)
    if current_year_allowed:
        current_year = date.today().year
        if current_year in counts.index:
            full_years.add(current_year)
    if not full_years:
        return monthly_returns.iloc[0:0]
    return monthly_returns[monthly_returns['Year'].isin(full_years)]


def _make_tree(parent: tk.Misc, headings: tuple[str, ...], widths: tuple[int, ...]) -> ttk.Treeview:
    cols = [f"c{i}" for i, _ in enumerate(headings)]
    tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
    for col_id, heading, width in zip(cols, headings, widths):
        anchor = "w" if heading in ("Index", "Month", "Best Month") else "e"
        tree.heading(col_id, text=heading)
        tree.column(col_id, width=width, minwidth=width, stretch=False, anchor=anchor)

    yscroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
    xscroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    return tree


def _fill_tree(tree: ttk.Treeview, rows: list[tuple]) -> None:
    for item in tree.get_children():
        tree.delete(item)
    for row in rows:
        tree.insert("", tk.END, values=row)


def _get_default_indices() -> list[str]:
    try:
        indices = list_nse_index_names()
    except Exception:
        return DEFAULT_INDEX_OPTIONS[1:]

    filtered = [
        idx for idx in indices
        if any(key in idx for key in ["Nifty", "Sensex", "Bank", "IT", "Pharma", "Auto", "Energy", "FMCG", "Metal", "Realty"])
    ]
    return filtered[:20] if filtered else indices[:20]


class SeasonalityUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NSE Seasonality Analysis")
        self.geometry("980x720")
        self.minsize(900, 620)
        self.resizable(True, True)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        title = tk.Label(
            self,
            text="NSE Monthly Seasonality",
            font=("Segoe UI", 16, "bold"),
            pady=12,
        )
        title.grid(row=0, column=0, sticky="ew")

        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))

        self.screen_frame = ttk.Frame(notebook)
        self.backtest_frame = ttk.Frame(notebook)
        notebook.add(self.screen_frame, text="Screen")
        notebook.add(self.backtest_frame, text="Backtest")

        self._build_screen_tab()
        self._build_backtest_tab()
        self._build_output_area()

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 8))

        self._load_index_options()
        self.after(100, self._load_default_screen)

    def _build_screen_tab(self) -> None:
        self.screen_frame.columnconfigure(1, weight=1)

        ttk.Label(self.screen_frame, text="Years:").grid(row=0, column=0, sticky="e", padx=6, pady=8)
        self.screen_years_var = tk.StringVar(value="10")
        ttk.Entry(self.screen_frame, textvariable=self.screen_years_var, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=8)

        ttk.Label(self.screen_frame, text="Index:").grid(row=1, column=0, sticky="e", padx=6, pady=8)
        self.screen_index_var = tk.StringVar(value="Nifty 50")
        self.screen_index_combo = ttk.Combobox(
            self.screen_frame,
            textvariable=self.screen_index_var,
            values=["All major indices"],
            state="readonly",
            width=48,
        )
        self.screen_index_combo.bind('<<ComboboxSelected>>', self._on_index_selected)
        self.screen_index_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=8)

        ttk.Button(self.screen_frame, text="Run screen analysis", command=self._on_screen_run).grid(row=2, column=0, columnspan=2, pady=12)

    def _build_backtest_tab(self) -> None:
        self.backtest_frame.columnconfigure(1, weight=1)

        ttk.Label(self.backtest_frame, text="Years:").grid(row=0, column=0, sticky="e", padx=6, pady=8)
        self.backtest_years_var = tk.StringVar(value="10")
        ttk.Entry(self.backtest_frame, textvariable=self.backtest_years_var, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=8)

        ttk.Label(self.backtest_frame, text="Top N:").grid(row=1, column=0, sticky="e", padx=6, pady=8)
        self.backtest_top_n_var = tk.StringVar(value="15")
        ttk.Entry(self.backtest_frame, textvariable=self.backtest_top_n_var, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=8)

        ttk.Button(self.backtest_frame, text="Run backtest analysis", command=self._on_backtest_run).grid(row=2, column=0, columnspan=2, pady=12)

    def _build_output_area(self) -> None:
        output_frame = ttk.Frame(self)
        output_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 4))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.tree_container = ttk.Frame(output_frame)
        self.tree_container.grid(row=0, column=0, sticky="nsew")
        self.tree_container.columnconfigure(0, weight=1)
        self.tree_container.rowconfigure(0, weight=1)
        self.result_tree: ttk.Treeview | None = None

        self.log_text = tk.Text(output_frame, wrap="word", height=8, state="disabled", bg="#1e1e1e", fg="#dcdcdc", insertbackground="#ffffff")
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        output_frame.rowconfigure(1, weight=0)

        log_scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

    def _load_index_options(self) -> None:
        def load() -> None:
            try:
                options = ["All major indices"] + list_nse_index_names()
            except Exception:
                options = DEFAULT_INDEX_OPTIONS
            self.after(0, lambda: self.screen_index_combo.configure(values=options))

        threading.Thread(target=load, daemon=True).start()

    def _load_default_screen(self) -> None:
        self.screen_index_var.set("Nifty 50")
        self._on_screen_run()

    def _on_index_selected(self, event: tk.Event) -> None:
        selection = self.screen_index_var.get()
        self.screen_years_var.set("10")
        self._append_log(f"Selected index: {selection}\n")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.screen_index_combo.configure(state="readonly" if enabled else "disabled")
        for child in self.screen_frame.winfo_children() + self.backtest_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.configure(state=state)

    def _on_screen_run(self) -> None:
        self._run_analysis("screen")

    def _on_backtest_run(self) -> None:
        self._run_analysis("backtest")

    def _run_analysis(self, mode: str) -> None:
        if mode == "screen":
            years = self.screen_years_var.get().strip() or "5"
            index_selection = self.screen_index_var.get().strip() or "All major indices"
            top_n = None
        else:
            years = self.backtest_years_var.get().strip() or "5"
            index_selection = None
            top_n = self.backtest_top_n_var.get().strip() or "15"

        if not years.isdigit():
            messagebox.showerror("Seasonality UI", "Years must be a number.")
            return

        if mode == "backtest" and top_n and not top_n.isdigit():
            messagebox.showerror("Seasonality UI", "Top N must be a number.")
            return

        self._set_controls_enabled(False)
        self._clear_results()
        self._append_log(f"Running {mode} analysis...\n")
        self.status_var.set("Running analysis...")

        thread = threading.Thread(
            target=self._analysis_thread,
            args=(mode, int(years), index_selection, int(top_n) if top_n else None),
            daemon=True,
        )
        thread.start()

    def _analysis_thread(self, mode: str, years: int, index_selection: str | None, top_n: int | None) -> None:
        try:
            if mode == "screen":
                rows, headings, widths = self._run_screen_analysis(years, index_selection)
            else:
                rows, headings, widths = self._run_backtest_analysis(years, top_n)

            self.after(0, lambda: self._display_results(rows, headings, widths))
            self.after(0, lambda: self._append_log("Analysis complete.\n"))
        except Exception as exc:
            self.after(0, lambda: self._append_log(f"Error: {exc}\n"))
        finally:
            self.after(0, self._analysis_complete)

    def _run_screen_analysis(self, years: int, index_selection: str | None) -> tuple[list[tuple], tuple[str, ...], tuple[int, ...]]:
        end_date = date.today()
        # include an extra month of history so the first requested month
        # has a prior-month close available to compute a monthly return
        EXTRA_DAYS_BEFORE = 40
        start_date = end_date - timedelta(days=years * 365 + EXTRA_DAYS_BEFORE)

        month_headers = tuple(MONTH_NAMES)
        if not index_selection or index_selection == "All major indices":
            indices = _get_default_indices()
            self.after(0, lambda: self._append_log(f"Running screen on {len(indices)} indices.\n"))
            aggregated: dict[tuple[int, int], list[float]] = {}

            for idx_name in indices:
                self.after(0, lambda idx=idx_name: self._append_log(f"Loading {idx}...\n"))
                try:
                    df = fetch_index_ohlc_history(idx_name, start_date, end_date, quiet=True)
                except Exception:
                    continue
                monthly = calculate_monthly_returns(df)
                for _, row in monthly.iterrows():
                    key = (int(row["Year"]), int(row["Month"]))
                    aggregated.setdefault(key, []).append(float(row["Return"]))

            month_values: dict[int, list[float]] = {month: [] for month in range(1, 13)}
            year_month_values: dict[int, dict[int, str]] = {}
            for (year, month), returns in aggregated.items():
                month_values[month].extend(returns)
                year_month_values.setdefault(year, {})[month] = _format_pct(np.mean(returns))

            # Iteratively refetch earlier history if the earliest year is incomplete.
            if year_month_values:
                attempts = 0
                while attempts < 3:
                    earliest_year = min(year_month_values.keys())
                    months_present = set(year_month_values[earliest_year].keys())
                    if len(months_present) >= 12:
                        break
                    alt_start = date(earliest_year - 1, 1, 1)
                    aggregated = {}
                    for idx_name in indices:
                        try:
                            df = fetch_index_ohlc_history(idx_name, alt_start, end_date, quiet=True)
                        except Exception:
                            continue
                        monthly = calculate_monthly_returns(df)
                        for _, row in monthly.iterrows():
                            key = (int(row["Year"]), int(row["Month"]))
                            aggregated.setdefault(key, []).append(float(row["Return"]))

                    year_month_values = {}
                    for (year, month), returns in aggregated.items():
                        year_month_values.setdefault(year, {})[month] = _format_pct(np.mean(returns))
                    attempts += 1

            current_year = date.today().year
            complete_years = {
                year
                for year, months in year_month_values.items()
                if len(months) == 12 or year == current_year
            }
            month_values = {month: [] for month in range(1, 13)}
            filtered_year_month_values: dict[int, dict[int, str]] = {}
            for (year, month), returns in aggregated.items():
                if year not in complete_years:
                    continue
                month_values[month].extend(returns)
                filtered_year_month_values.setdefault(year, {})[month] = _format_pct(np.mean(returns))
            year_month_values = filtered_year_month_values

            prob_row = [
                _format_rate(np.mean([1.0 if ret > 0 else 0.0 for ret in month_values[month]]))
                if month_values[month] else "-"
                for month in range(1, 13)
            ]
            avg_row = [
                _format_pct(np.mean(month_values[month]))
                if month_values[month] else "-"
                for month in range(1, 13)
            ]

            rows = [
                ("Probability %", *prob_row),
                ("Average return %", *avg_row),
            ]
            for year in sorted(year_month_values.keys(), reverse=True):
                row_values = [year_month_values[year].get(month, "-") for month in range(1, 13)]
                rows.append((str(year), *row_values))

            headings = ("Year", *month_headers)
            widths = (140, *([90] * 12))
            return rows, headings, widths

        self.after(0, lambda: self._append_log(f"Running screen for {index_selection}...\n"))
        try:
            df = fetch_index_ohlc_history(index_selection, start_date, end_date, quiet=True)
        except Exception:
            df = None

        if df is None or df.empty:
            return [], ("Year", *month_headers), (100, *([90] * 12))

        monthly_returns = calculate_monthly_returns(df)
        # If the earliest year in monthly_returns has missing months, iteratively
        # fetch earlier history (previous years) so we can compute all months.
        if not monthly_returns.empty:
            attempts = 0
            while attempts < 3:
                years_present = sorted(set(monthly_returns['Year']))
                earliest_year = years_present[0]
                months_for_earliest = set(int(m) for m in monthly_returns[monthly_returns['Year'] == earliest_year]['Month'])
                if len(months_for_earliest) >= 12:
                    break
                alt_start = date(earliest_year - 1, 1, 1)
                try:
                    df = fetch_index_ohlc_history(index_selection, alt_start, end_date, quiet=True)
                    monthly_returns = calculate_monthly_returns(df)
                except Exception:
                    break
                attempts += 1
        monthly_returns = _filter_full_years(monthly_returns, current_year_allowed=True)
        month_values: dict[int, list[float]] = {month: [] for month in range(1, 13)}
        year_month_values: dict[int, dict[int, str]] = {}
        for _, row in monthly_returns.iterrows():
            year = int(row["Year"])
            month = int(row["Month"])
            ret = float(row["Return"])
            month_values[month].append(ret)
            year_month_values.setdefault(year, {})[month] = _format_pct(ret)

        prob_row = [
            _format_rate(np.mean([1.0 if ret > 0 else 0.0 for ret in month_values[month]]))
            if month_values[month] else "-"
            for month in range(1, 13)
        ]
        avg_row = [
            _format_pct(np.mean(month_values[month]))
            if month_values[month] else "-"
            for month in range(1, 13)
        ]

        rows = [
            ("Probability %", *prob_row),
            ("Average return %", *avg_row),
        ]
        for year in sorted(year_month_values.keys(), reverse=True):
            row_values = [year_month_values[year].get(month, "-") for month in range(1, 13)]
            rows.append((str(year), *row_values))

        headings = ("Year", *month_headers)
        widths = (140, *([90] * 12))
        return rows, headings, widths

    def _run_backtest_analysis(self, years: int, top_n: int | None) -> tuple[list[tuple], tuple[str, ...], tuple[int, ...]]:
        end_date = date.today()
        # include an extra month of history so the first requested month
        # has a prior-month close available to compute a monthly return
        EXTRA_DAYS_BEFORE = 40
        start_date = end_date - timedelta(days=years * 365 + EXTRA_DAYS_BEFORE)
        self.after(0, lambda: self._append_log("Loading index universe...\n"))

        indices = backtest_module.build_nifty_index_universe()
        rows = []
        for idx in indices:
            self.after(0, lambda idx_name=idx.label: self._append_log(f"Loading {idx_name}...\n"))
            df = backtest_module.get_india_market_data(idx.yahoo_ticker, start_date, end_date)
            if df is None or df.empty:
                continue
            monthly = backtest_module.calculate_monthly_returns(df)
            if monthly.empty:
                continue
            # If earliest year is missing months, try to include earlier history
            # (previous year) so the earliest year has full months available.
            years_present = sorted(set(monthly['Year']))
            if years_present:
                earliest_year = years_present[0]
                months_for_earliest = set(int(m) for m in monthly[monthly['Year'] == earliest_year]['Month'])
                if len(months_for_earliest) < 12:
                    alt_start = date(earliest_year - 1, 1, 1)
                    try:
                        df2 = backtest_module.get_india_market_data(idx.yahoo_ticker, alt_start, end_date)
                        if df2 is not None and not df2.empty:
                            monthly = backtest_module.calculate_monthly_returns(df2)
                    except Exception:
                        pass
            monthly = _filter_full_years(monthly, current_year_allowed=True)
            seasonality = backtest_module.compute_seasonality(monthly, idx.label)
            if not seasonality:
                continue
            returns = [month_data["avg_return"] for month_data in seasonality.values()]
            win_rates = [month_data["win_rate"] for month_data in seasonality.values()]
            if not returns:
                continue
            best_month = max(seasonality.values(), key=lambda m: m["avg_return"])["month_name"]
            rows.append(
                (
                    idx.label,
                    _format_pct(np.mean(returns)),
                    _format_rate(np.mean(win_rates)),
                    best_month,
                    str(len(returns)),
                )
            )

        rows.sort(key=lambda row: float(row[1].strip("%")) if row[1] else 0.0, reverse=True)
        if top_n:
            rows = rows[:top_n]

        headings = ("Index", "Avg Return", "Avg Win Rate", "Best Month", "Samples")
        widths = (220, 120, 120, 120, 100)
        return rows, headings, widths

    def _display_results(self, rows: list[tuple], headings: tuple[str, ...], widths: tuple[int, ...]) -> None:
        if self.result_tree is not None:
            self.result_tree.destroy()

        self.result_tree = _make_tree(self.tree_container, headings, widths)
        _fill_tree(self.result_tree, rows)

    def _analysis_complete(self) -> None:
        self._set_controls_enabled(True)
        self.status_var.set("Analysis complete.")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_results(self) -> None:
        if self.result_tree is not None:
            self.result_tree.destroy()
            self.result_tree = None
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


def main() -> None:
    app = SeasonalityUI()
    app.mainloop()


if __name__ == "__main__":
    main()
