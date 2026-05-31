"""Clipboard + drag-rectangle copy for RRG table grids (Labels/Entry cells)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

_EMPTY_CELL = "\u2014"
_SEL_OUTLINE = "#1565C0"


def _is_copyable(text: str) -> bool:
    return bool(text) and text not in (_EMPTY_CELL, "-")


def _widget_text(widget: tk.Misc) -> str:
    if isinstance(widget, tk.Label):
        return (widget.cget("text") or "").strip()
    if isinstance(widget, tk.Entry):
        try:
            if widget.selection_present():
                return widget.selection_get()
        except tk.TclError:
            pass
        return (widget.get() or "").strip()
    if isinstance(widget, tk.Text):
        try:
            if widget.tag_ranges(tk.SEL):
                return widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass
        return widget.get("1.0", "end-1c").strip()
    if isinstance(widget, ttk.Treeview):
        sel = widget.selection()
        if not sel:
            return ""
        lines: list[str] = []
        cols = [widget.heading(c, "text") for c in widget["columns"]]
        lines.append("\t".join(str(c) for c in cols))
        for iid in sel:
            row = [str(v) for v in widget.item(iid, "values")]
            lines.append("\t".join(row))
        return "\n".join(lines)
    return ""


class TableRegionCopy:
    """Drag-select a rectangle of table cells; Ctrl+C copies as TSV."""

    _by_window: dict[int, TableRegionCopy] = {}

    def __init__(self, window: tk.Misc) -> None:
        self.window = window
        self._grids: list[dict] = []
        self._anchor: tuple[int, int] | None = None
        self._active_grid: dict | None = None
        self._dragging = False
        window.bind("<ButtonRelease-1>", self._on_global_release, add="+")

    @classmethod
    def for_window(cls, window: tk.Misc) -> TableRegionCopy:
        top = window.winfo_toplevel()
        key = id(top)
        if key not in cls._by_window:
            cls._by_window[key] = cls(top)
        return cls._by_window[key]

    @classmethod
    def set_cell_pos(cls, widget: tk.Misc, row: int, col: int) -> None:
        widget._copy_row = row  # noqa: SLF001
        widget._copy_col = col  # noqa: SLF001

    def register_grid(self, cells: list[list[tk.Misc]]) -> dict:
        cell_set: set[tk.Misc] = set()
        grid: dict = {"cells": cells, "selection": set(), "styles": {}, "cell_set": cell_set}
        idx = len(self._grids)
        self._grids.append(grid)
        for r, row in enumerate(cells):
            for c, widget in enumerate(row):
                cell_set.add(widget)
                widget._copy_grid_idx = idx  # noqa: SLF001
                self.set_cell_pos(widget, r, c)
                self._remember_style(grid, widget)
                if isinstance(widget, tk.Entry):
                    widget.bind(
                        "<Shift-Button-1>",
                        lambda e, g=grid: self._on_press_from_event(g, e),
                        add="+",
                    )
                else:
                    widget.bind(
                        "<Button-1>",
                        lambda e, g=grid: self._on_press_from_event(g, e),
                        add="+",
                    )
        return grid

    def sync_styles(self, grid: dict) -> None:
        for row in grid["cells"]:
            for widget in row:
                self._remember_style(grid, widget)

    def _remember_style(self, grid: dict, widget: tk.Misc) -> None:
        try:
            grid["styles"][widget] = (widget.cget("bg"), widget.cget("fg"))
        except tk.TclError:
            pass

    def _pos(self, widget: tk.Misc) -> tuple[int, int] | None:
        r = getattr(widget, "_copy_row", None)
        c = getattr(widget, "_copy_col", None)
        if r is None or c is None:
            return None
        return int(r), int(c)

    def _find_cell(self, widget: tk.Misc | None, grid: dict) -> tk.Misc | None:
        cell_set = grid["cell_set"]
        while widget is not None:
            if widget in cell_set:
                return widget
            try:
                widget = widget.master  # type: ignore[assignment]
            except (AttributeError, tk.TclError):
                break
        return None

    def _on_press_from_event(self, grid: dict, event) -> None:
        pos = self._pos(event.widget)
        if pos is None:
            return
        self._clear_other_grids(grid)
        self._active_grid = grid
        self._anchor = pos
        self._dragging = True
        row, col = pos
        self._select_rect(grid, row, col, row, col)
        self.window.bind("<B1-Motion>", self._on_global_motion, add="+")

    def _on_global_motion(self, event) -> None:
        if not self._dragging or self._anchor is None or self._active_grid is None:
            return
        under = self.window.winfo_containing(event.x_root, event.y_root)
        cell = self._find_cell(under, self._active_grid)
        if cell is None:
            return
        pos = self._pos(cell)
        if pos is None:
            return
        ar, ac = self._anchor
        self._select_rect(self._active_grid, ar, ac, pos[0], pos[1])

    def _on_global_release(self, _event) -> None:
        if self._dragging:
            self._dragging = False
            self._anchor = None
            try:
                self.window.unbind("<B1-Motion>")
            except tk.TclError:
                pass

    def _clear_other_grids(self, keep: dict) -> None:
        for grid in self._grids:
            if grid is not keep:
                self._clear_selection(grid)

    def _clear_selection(self, grid: dict) -> None:
        for widget in list(grid["selection"]):
            bg, fg = grid["styles"].get(widget, (widget.cget("bg"), widget.cget("fg")))
            try:
                widget.config(highlightthickness=0, bg=bg, fg=fg)
            except tk.TclError:
                pass
        grid["selection"].clear()

    def _select_rect(
        self, grid: dict, r0: int, c0: int, r1: int, c1: int
    ) -> None:
        self._clear_selection(grid)
        rmin, rmax = min(r0, r1), max(r0, r1)
        cmin, cmax = min(c0, c1), max(c0, c1)
        for row in grid["cells"]:
            for widget in row:
                pos = self._pos(widget)
                if pos is None:
                    continue
                r, c = pos
                if rmin <= r <= rmax and cmin <= c <= cmax:
                    grid["selection"].add(widget)
                    self._remember_style(grid, widget)
                    try:
                        widget.config(
                            highlightthickness=2,
                            highlightbackground=_SEL_OUTLINE,
                            highlightcolor=_SEL_OUTLINE,
                        )
                    except tk.TclError:
                        pass

    def copy_selection(self) -> str | None:
        for grid in self._grids:
            if not grid["selection"]:
                continue
            by_row: dict[int, list[tuple[int, tk.Misc]]] = {}
            for widget in grid["selection"]:
                pos = self._pos(widget)
                if pos is None:
                    continue
                r, c = pos
                by_row.setdefault(r, []).append((c, widget))
            lines: list[str] = []
            for r in sorted(by_row):
                cells = sorted(by_row[r], key=lambda x: x[0])
                lines.append("\t".join(_widget_text(w) for _, w in cells))
            return "\n".join(lines)
        return None


def configure_readonly_text(text: tk.Text) -> None:
    def block_edit(event):
        if event.state & 0x4 and event.keysym.lower() in ("c", "a", "x"):
            return None
        if event.keysym in (
            "Left",
            "Right",
            "Up",
            "Down",
            "Home",
            "End",
            "Prior",
            "Next",
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Tab",
        ):
            return None
        if event.keysym in ("BackSpace", "Delete", "Return"):
            return "break"
        if event.char and event.char.isprintable():
            return "break"
        return None

    text.bind("<Key>", block_edit, add="+")
    text.configure(exportselection=True)


def install_copy_support(window: tk.Misc) -> None:
    menu = tk.Menu(window, tearoff=0)
    table_copy = TableRegionCopy.for_window(window)

    def copy_to_clipboard(text: str) -> None:
        if not text:
            return
        window.clipboard_clear()
        window.clipboard_append(text)

    def copy_handler(event):
        try:
            if event.widget.winfo_toplevel() != window.winfo_toplevel():
                return
        except tk.TclError:
            return

        if isinstance(event.widget, tk.Text):
            try:
                if event.widget.tag_ranges(tk.SEL):
                    return
            except tk.TclError:
                pass

        if isinstance(event.widget, ttk.Treeview):
            text = _widget_text(event.widget)
            if text:
                copy_to_clipboard(text)
                return "break"

        if isinstance(event.widget, tk.Entry) and not hasattr(event.widget, "_copy_row"):
            try:
                if event.widget.selection_present():
                    return
            except tk.TclError:
                pass

        tsv = table_copy.copy_selection()
        if tsv:
            copy_to_clipboard(tsv)
            return "break"

        text = _widget_text(event.widget)
        if _is_copyable(text):
            copy_to_clipboard(text)
            return "break"
        return None

    def on_right_click(event):
        try:
            if event.widget.winfo_toplevel() != window.winfo_toplevel():
                return
        except tk.TclError:
            return
        tsv = table_copy.copy_selection()
        text = tsv if tsv else _widget_text(event.widget)
        if not _is_copyable(text):
            return
        menu.delete(0, tk.END)
        menu.add_command(label="Copy", command=lambda t=text: copy_to_clipboard(t))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    window.bind_all("<Control-c>", copy_handler, add="+")
    window.bind_all("<Control-C>", copy_handler, add="+")
    window.bind_class("Label", "<Button-3>", on_right_click, add="+")
    window.bind_class("Entry", "<Button-3>", on_right_click, add="+")
    window.bind_class("Text", "<Button-3>", on_right_click, add="+")
    window.bind_class("Treeview", "<Button-3>", on_right_click, add="+")

    window._table_region_copy = table_copy  # noqa: SLF001
