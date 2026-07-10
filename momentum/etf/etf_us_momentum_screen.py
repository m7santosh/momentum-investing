"""
US ETF momentum rankers — on-screen viewer (replaces Excel for daily scan).

Run:
    python momentum/etf/etf_us_momentum_screen.py

Strategies (same logic as legacy Excel scripts):
  - Abs Momentum      → momentum_us_etfs.py
  - RS Blended        → momentum_us_rs_etfs.py
  - RS Adaptive       → momentum_us_rs_etfs_adaptive.py
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from datetime import date

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.etf.etf_momentum_recommendations import (  # noqa: E402
    US_TOP_PICKS,
    recommendations_dataframe,
)
from momentum.etf.etf_momentum_screen_sort import (  # noqa: E402
    TreeSortState,
    apply_tree_sort,
    reset_tree_sort_state,
    update_etf_momentum_tree_headings,
    wire_etf_momentum_tree_sort,
)
from momentum.etf.etf_us_momentum_engine import (  # noqa: E402
    TOP_N,
    UsEtfMomentumSnapshot,
    fetch_us_etf_momentum_snapshot,
)
from momentum.rrg_busy import RrgBusyOverlay  # noqa: E402
from momentum.rrg_core import rrg_config_date_str, rrg_format_date  # noqa: E402
from momentum.rrg_date_picker import attach_rrg_date_picker  # noqa: E402
from momentum.rrg_ui_copy import install_copy_support  # noqa: E402

ABS_COLUMNS = (
    "Position",
    "Symbol",
    "Name",
    "Close",
    "9EMA",
    "Close_Below_9EMA",
    "Above_9EMA_Since",
    "Pct_Above_9EMA",
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
)

ABS_COLUMN_LABELS = {
    "Close": "Price",
    "9EMA": "9 EMA",
    "Close_Below_9EMA": "9 EMA Close",
    "Above_9EMA_Since": "9 EMA Cross Date",
    "Pct_Above_9EMA": "Since 9 EMA Cross %",
}

RS_BLENDED_COLUMNS = (
    "Position",
    "Symbol",
    "Name",
    "Close",
    "9EMA",
    "Close_Below_9EMA",
    "Above_9EMA_Since",
    "Pct_Above_9EMA",
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
    "RS_3M_vs_SP500",
)

RS_ADAPTIVE_COLUMNS = (
    "Position",
    "Symbol",
    "Name",
    "Close",
    "9EMA",
    "Close_Below_9EMA",
    "Above_9EMA_Since",
    "Pct_Above_9EMA",
    "Return_1W",
    "Return_2W",
    "Return_1M",
    "Return_3M",
)

_COLUMN_WIDTHS: dict[str, int] = {
    "Position": 56,
    "Symbol": 72,
    "Name": 240,
    "Close": 72,
    "9EMA": 72,
    "Close_Below_9EMA": 110,
    "Above_9EMA_Since": 108,
    "Pct_Above_9EMA": 118,
    "Return_1W": 76,
    "Return_2W": 76,
    "Return_1M": 76,
    "Return_3M": 76,
    "RS_3M_vs_SP500": 100,
}


def _cell(value: object, heading: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if heading in ("Position", "Symbol", "Name", "Close_Below_9EMA", "Above_9EMA_Since"):
        if heading == "Position" and isinstance(value, (int, float)):
            return str(int(value))
        return str(value)
    if heading.startswith("Rank") or heading == "Final_Rank":
        if isinstance(value, (int, float)):
            if heading == "Final_Rank":
                return f"{float(value):.2f}"
            return str(int(value)) if float(value) == int(value) else f"{float(value):.1f}"
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


def _make_tree(
    parent: tk.Misc,
    headings: tuple[str, ...],
    *,
    labels: dict[str, str] | None = None,
) -> ttk.Treeview:
    col_ids = tuple(f"c{i}" for i in range(len(headings)))
    tree = ttk.Treeview(parent, columns=col_ids, show="headings", selectmode="extended")
    label_map = labels or {}
    for col_id, heading in zip(col_ids, headings, strict=True):
        tree.heading(col_id, text=label_map.get(heading, heading))
        width = _COLUMN_WIDTHS.get(heading, 88)
        anchor = "w" if heading in ("Symbol", "Name", "Close_Below_9EMA", "Above_9EMA_Since") else "e"
        if heading == "Position":
            anchor = "center"
        tree.column(col_id, width=width, stretch=False, minwidth=width, anchor=anchor)

    yscroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
    xscroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    return tree


def _fill_tree(tree: ttk.Treeview, df: pd.DataFrame | None, headings: tuple[str, ...]) -> None:
    for item in tree.get_children():
        tree.delete(item)
    if df is None or df.empty:
        tree.insert("", tk.END, values=tuple("—" for _ in headings))
        return
    for _, row in df.iterrows():
        values = tuple(_cell(row.get(h), h) for h in headings)
        tag = ""
        if "Close_Below_9EMA" in headings:
            flag = row.get("Close_Below_9EMA")
            if flag == "Exit":
                tag = "exit"
            elif flag == "Hold":
                tag = "hold"
        tree.insert("", tk.END, values=values, tags=(tag,) if tag else ())


def _make_picks_panel(parent: tk.Misc) -> scrolledtext.ScrolledText:
    text = scrolledtext.ScrolledText(
        parent,
        wrap=tk.WORD,
        width=44,
        height=28,
        font=("Segoe UI", 9),
        state=tk.DISABLED,
        relief=tk.FLAT,
        padx=6,
        pady=6,
    )
    text.pack(fill=tk.BOTH, expand=True)
    return text


def _fill_picks_panel(
    text: scrolledtext.ScrolledText,
    df: pd.DataFrame | None,
    *,
    include_name: bool = False,
) -> None:
    text.config(state=tk.NORMAL)
    text.delete("1.0", tk.END)
    if df is None:
        text.insert(tk.END, "Refresh to load recommendations.")
    elif df.empty:
        text.insert(tk.END, f"No ETFs in screener top {US_TOP_PICKS} qualify for recommendations.")
    else:
        for _, row in df.iterrows():
            header = f"#{int(row['Rank'])}  {row['Symbol']}"
            if include_name:
                name = str(row.get("Name", "")).strip()
                if name:
                    header += f"  —  {name}"
            text.insert(tk.END, f"{header}\n")
            text.insert(tk.END, f"{row['Reason']}\n\n")
    text.config(state=tk.DISABLED)


class UsEtfMomentumScreenApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("US ETF Momentum")
        self.root.geometry("1680x760")
        self.root.minsize(1200, 560)

        self._snapshot: UsEtfMomentumSnapshot | None = None
        self._busy = False
        self._overlay = RrgBusyOverlay(root, default_message="Fetching ETF data…")
        self._tree_sort_states: dict[int, TreeSortState] = {}
        self._tree_views: list[tuple[ttk.Treeview, tuple[str, ...], dict[str, str]]] = []

        toolbar = tk.Frame(root, padx=10, pady=8)
        toolbar.pack(fill=tk.X)

        self._refresh_btn = ttk.Button(toolbar, text="Refresh", command=self._on_refresh)
        self._refresh_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Backtest", command=self._on_backtest).pack(
            side=tk.LEFT, padx=(0, 16)
        )

        tk.Label(toolbar, text="As of:").pack(side=tk.LEFT)
        date_frame = tk.Frame(toolbar)
        date_frame.pack(side=tk.LEFT, padx=(4, 16))
        self._date_var = tk.StringVar(value=rrg_format_date(date.today()))
        self._date_entry = tk.Entry(date_frame, textvariable=self._date_var, width=11)
        self._date_entry.pack(side=tk.LEFT)
        attach_rrg_date_picker(
            date_frame,
            self._date_entry,
            self._date_var,
            default_date=date.today(),
            max_date=date.today(),
        ).pack(side=tk.LEFT, padx=(2, 0))

        tk.Label(toolbar, text="Filter:").pack(side=tk.LEFT)
        self._filter_var = tk.StringVar()
        filter_entry = tk.Entry(toolbar, textvariable=self._filter_var, width=18)
        filter_entry.pack(side=tk.LEFT, padx=4)
        self._filter_var.trace_add("write", lambda *_: self._apply_filter())

        self._status_var = tk.StringVar(value="Click Refresh to load rankings.")
        status = tk.Label(root, textvariable=self._status_var, anchor="w", padx=10, pady=4)
        status.pack(fill=tk.X)

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left = tk.Frame(body)
        body.add(left, weight=3)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._abs_tree = self._add_tab(
            notebook, f"Abs Momentum (top {TOP_N})", ABS_COLUMNS, labels=ABS_COLUMN_LABELS
        )
        self._rs_tree = self._add_tab(
            notebook, f"RS Blended (top {TOP_N})", RS_BLENDED_COLUMNS, labels=ABS_COLUMN_LABELS
        )
        self._adaptive_tree = self._add_tab(
            notebook, f"RS Adaptive (top {TOP_N})", RS_ADAPTIVE_COLUMNS, labels=ABS_COLUMN_LABELS
        )

        picks_outer = tk.Frame(body, padx=6, pady=6)
        body.add(picks_outer, weight=1)
        picks_title = tk.Frame(picks_outer)
        picks_title.pack(fill=tk.X, pady=(0, 4))
        tk.Label(picks_title, text=f"Top {US_TOP_PICKS} Picks", font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT
        )
        tk.Label(
            picks_title,
            text="Above 9 EMA only",
            font=("Segoe UI", 8),
            fg="#555555",
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._picks_panel = _make_picks_panel(picks_outer)

        for tree in (self._abs_tree, self._rs_tree, self._adaptive_tree):
            tree.tag_configure("exit", foreground="#b71c1c")
            tree.tag_configure("hold", foreground="#1b5e20")

        install_copy_support(root)
        self.root.after(200, self._on_refresh)

    def _add_tab(
        self,
        notebook: ttk.Notebook,
        title: str,
        headings: tuple[str, ...],
        *,
        labels: dict[str, str] | None = None,
    ) -> ttk.Treeview:
        frame = tk.Frame(notebook)
        notebook.add(frame, text=title)
        label_map = labels or {}
        tree = _make_tree(frame, headings, labels=label_map)
        state = TreeSortState()
        self._tree_sort_states[id(tree)] = state
        self._tree_views.append((tree, headings, label_map))
        wire_etf_momentum_tree_sort(
            tree, headings, label_map, state, self._apply_filter
        )
        return tree

    def _on_backtest(self) -> None:
        from momentum.etf_us_momentum_backtest_ui import open_us_etf_momentum_backtest

        open_us_etf_momentum_backtest(self.root)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._refresh_btn.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.root.config(cursor="watch" if busy else "")

    def _on_refresh(self) -> None:
        if self._busy:
            return
        try:
            as_of_iso = rrg_config_date_str(self._date_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid date", str(exc), parent=self.root)
            self._date_entry.focus_set()
            return
        self._set_busy(True)
        self._status_var.set(f"Fetching rankings as of {self._date_var.get()}…")

        def worker() -> UsEtfMomentumSnapshot:
            return fetch_us_etf_momentum_snapshot(as_of_date=as_of_iso)

        def on_done(snap: UsEtfMomentumSnapshot) -> None:
            self._on_load_done(snap, None)

        def on_error(exc: BaseException) -> None:
            self._on_load_done(None, exc if isinstance(exc, Exception) else RuntimeError(str(exc)))

        try:
            snap = self._overlay.run_threaded(
                worker,
                message="Computing ETF momentum rankings…",
            )
        except BaseException as exc:
            self._set_busy(False)
            on_error(exc)
            return

        self._set_busy(False)
        on_done(snap)

    def _on_load_done(self, snap: UsEtfMomentumSnapshot | None, err: Exception | None) -> None:
        self._set_busy(False)
        if err is not None:
            messagebox.showerror("Load failed", str(err), parent=self.root)
            self._status_var.set(f"Load failed: {err}")
            return
        assert snap is not None
        self._snapshot = snap
        for tree, headings, label_map in self._tree_views:
            state = self._tree_sort_states[id(tree)]
            reset_tree_sort_state(state)
            update_etf_momentum_tree_headings(tree, headings, label_map, state)
        self._apply_filter()
        top_adaptive = ""
        if not snap.rs_adaptive.empty:
            row0 = snap.rs_adaptive.iloc[0]
            top_adaptive = f"  Top adaptive: {row0['Symbol']}"
        self._status_var.set(
            f"Run {snap.run_date}  |  Market_Regime={snap.market_regime}  |  "
            f"Abs={len(snap.abs_momentum)}  RS blended={len(snap.rs_blended)}  "
            f"RS adaptive={len(snap.rs_adaptive)} (ranked {snap.etfs_ranked_adaptive})"
            f"{top_adaptive}"
        )

    def _apply_filter(self) -> None:
        snap = self._snapshot
        if snap is None:
            return
        needle = self._filter_var.get().strip().upper()

        def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or not needle:
                return df
            sym_mask = df["Symbol"].astype(str).str.upper().str.contains(needle, na=False)
            if "Name" in df.columns:
                name_mask = df["Name"].astype(str).str.upper().str.contains(needle, na=False)
                mask = sym_mask | name_mask
            else:
                mask = sym_mask
            return df.loc[mask].reset_index(drop=True)

        def _display_df(tree: ttk.Treeview, df: pd.DataFrame, headings: tuple[str, ...]) -> None:
            state = self._tree_sort_states[id(tree)]
            _fill_tree(tree, apply_tree_sort(df, state), headings)

        _display_df(self._abs_tree, _filter_df(snap.abs_momentum), ABS_COLUMNS)
        _display_df(self._rs_tree, _filter_df(snap.rs_blended), RS_BLENDED_COLUMNS)
        _display_df(self._adaptive_tree, _filter_df(snap.rs_adaptive), RS_ADAPTIVE_COLUMNS)

        picks_df = recommendations_dataframe(
            snap.abs_momentum,
            snap.rs_blended,
            snap.rs_adaptive,
            top_n=US_TOP_PICKS,
            include_name=True,
        )
        _fill_picks_panel(self._picks_panel, picks_df, include_name=True)


def main() -> int:
    root = tk.Tk()
    UsEtfMomentumScreenApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
