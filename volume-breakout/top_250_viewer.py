"""
Top 250 Volume Breakout viewer — matches Google Sheet layout.

Run:
    python volume-breakout/top_250_viewer.py
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

_VB_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _VB_DIR.parent
for _path in (_PROJECT_ROOT, _VB_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from momentum.rrg_ui_copy import install_copy_support  # noqa: E402
import top_250_sheet_logic as _logic  # noqa: E402

FINAL_LIST_TURNOVER_COLUMNS = _logic.FINAL_LIST_TURNOVER_COLUMNS
FINAL_LIST_VOLUME_COLUMNS = _logic.FINAL_LIST_VOLUME_COLUMNS
TOP_250_TURNOVER_COLUMNS = _logic.TOP_250_TURNOVER_COLUMNS
TOP_250_VOLUME_COLUMNS = _logic.TOP_250_VOLUME_COLUMNS
Top250Snapshot = _logic.Top250Snapshot
fetch_top250_snapshot = _logic.fetch_top250_snapshot


_COLUMN_WIDTHS: dict[str, int] = {
    "NSE Code": 88,
    "Volume": 100,
    "Turnover": 110,
    "Close Price": 88,
    "Previous Close": 100,
    "CMP": 72,
    "50 DMA": 72,
    "100 DMA": 80,
    "200 DMA": 80,
    "Output": 110,
    "Difference from 200 DMA": 130,
    "CAR": 120,
}


def _cell(value: object, heading: str) -> str:
    if value is None:
        return ""
    if heading in ("NSE Code", "Output", "CAR"):
        return str(value)
    if heading in ("Volume", "Turnover"):
        return str(int(value)) if isinstance(value, (int, float)) else str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


class Top250ViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Volume Breakout")
        self.root.geometry("1280x720")
        self.root.minsize(960, 520)

        self._snapshot: Top250Snapshot | None = None
        self._busy = False

        toolbar = tk.Frame(root, padx=10, pady=8)
        toolbar.pack(fill=tk.X)

        self._refresh_btn = ttk.Button(toolbar, text="Refresh", command=self._on_refresh)
        self._refresh_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Backtest", command=self._on_backtest).pack(
            side=tk.LEFT, padx=(0, 16)
        )

        tk.Label(toolbar, text="Filter:").pack(side=tk.LEFT)
        self._filter_var = tk.StringVar()
        filter_entry = tk.Entry(toolbar, textvariable=self._filter_var, width=18)
        filter_entry.pack(side=tk.LEFT, padx=4)
        self._filter_var.trace_add("write", lambda *_: self._apply_filter())

        self._status_var = tk.StringVar(value="Loading…")
        status = tk.Label(root, textvariable=self._status_var, anchor="w", padx=10, pady=4)
        status.pack(fill=tk.X)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._vol_tree = self._add_tab(notebook, "Top 250 Tickers", TOP_250_VOLUME_COLUMNS)
        self._final_vol_tree = self._add_tab(notebook, "Final List Volume", FINAL_LIST_VOLUME_COLUMNS)
        self._turnover_tree = self._add_tab(notebook, "Top 250 Turnover", TOP_250_TURNOVER_COLUMNS)
        self._final_turnover_tree = self._add_tab(
            notebook, "Final List Turnover", FINAL_LIST_TURNOVER_COLUMNS
        )

        install_copy_support(root)
        self.root.after(100, self._on_refresh)

    def _add_tab(
        self, notebook: ttk.Notebook, title: str, headings: tuple[str, ...]
    ) -> ttk.Treeview:
        frame = tk.Frame(notebook)
        notebook.add(frame, text=title)
        return self._make_tree(frame, headings)

    def _make_tree(self, parent: tk.Misc, headings: tuple[str, ...]) -> ttk.Treeview:
        col_ids = tuple(f"c{i}" for i in range(len(headings)))
        tree = ttk.Treeview(parent, columns=col_ids, show="headings", selectmode="extended")
        for col_id, heading in zip(col_ids, headings, strict=True):
            tree.heading(col_id, text=heading)
            width = _COLUMN_WIDTHS.get(heading, 100)
            anchor = "w" if heading in ("NSE Code", "Output", "CAR") else "e"
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

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._refresh_btn.config(state=tk.DISABLED if busy else tk.NORMAL)

    def _on_backtest(self) -> None:
        from top_250_backtest_ui import open_top250_backtest

        open_top250_backtest(self.root)

    def _on_refresh(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._status_var.set("Fetching bhavcopy and market data (may take a minute)…")

        def worker() -> None:
            err: Exception | None = None
            snap: Top250Snapshot | None = None
            try:
                snap = fetch_top250_snapshot()
            except Exception as exc:
                err = exc
            self.root.after(0, lambda: self._on_load_done(snap, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_done(self, snap: Top250Snapshot | None, err: Exception | None) -> None:
        self._set_busy(False)
        if err is not None:
            messagebox.showerror("Load failed", str(err), parent=self.root)
            self._status_var.set(f"Load failed: {err}")
            return
        assert snap is not None
        self._snapshot = snap
        self._apply_filter()
        self._status_var.set(snap.status)

    def _apply_filter(self) -> None:
        snap = self._snapshot
        if snap is None:
            return
        needle = self._filter_var.get().strip().upper()

        def match(symbol: str) -> bool:
            return not needle or needle in symbol.upper()

        self._fill_tree(
            self._vol_tree,
            [r for r in snap.volume_rows if match(r.nse_code)],
            use_final=False,
        )
        self._fill_tree(
            self._final_vol_tree,
            [r for r in snap.final_volume if match(r.nse_code)],
            use_final=True,
        )
        self._fill_tree(
            self._turnover_tree,
            [r for r in snap.turnover_rows if match(r.nse_code)],
            use_final=False,
        )
        self._fill_tree(
            self._final_turnover_tree,
            [r for r in snap.final_turnover if match(r.nse_code)],
            use_final=True,
        )

    def _fill_tree(self, tree: ttk.Treeview, rows, *, use_final: bool) -> None:
        headings = tuple(tree.heading(c, "text") for c in tree["columns"])
        for item in tree.get_children():
            tree.delete(item)
        for row in rows:
            values = row.final_list_values() if use_final else row.top250_values()
            tree.insert(
                "",
                tk.END,
                values=tuple(_cell(v, h) for v, h in zip(values, headings, strict=True)),
            )


def main() -> int:
    root = tk.Tk()
    Top250ViewerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
