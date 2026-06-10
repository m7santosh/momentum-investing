"""RRG-style date entry with calendar popup (DD-MM-YYYY)."""

from __future__ import annotations

import calendar
from datetime import date
from typing import Callable

import tkinter as tk
from tkinter import messagebox, ttk

from momentum.rrg_core import rrg_format_date, rrg_parse_user_date

_WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class _DatePickerDialog:
    def __init__(
        self,
        parent: tk.Misc,
        *,
        anchor: tk.Misc | None = None,
        initial: date,
        max_date: date,
        on_select: Callable[[date], None],
    ) -> None:
        self._anchor = anchor or parent
        self._initial = initial
        self._year = initial.year
        self._month = initial.month
        self._max_date = max_date
        self._on_select = on_select
        self._day_buttons: list[tk.Button] = []

        root = parent.winfo_toplevel()
        self._top = tk.Toplevel(root)
        self._top.title("Select date")
        self._top.transient(root)
        self._top.resizable(False, False)
        self._top.protocol("WM_DELETE_WINDOW", self._close)

        header = tk.Frame(self._top, padx=8, pady=6)
        header.pack(fill=tk.X)

        ttk.Button(header, text="<", width=3, command=self._prev_month).pack(side=tk.LEFT)
        self._title_var = tk.StringVar(value=self._month_title())
        tk.Label(header, textvariable=self._title_var, width=18, anchor="center").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(header, text=">", width=3, command=self._next_month).pack(side=tk.LEFT)
        ttk.Button(header, text="Today", command=self._pick_today).pack(side=tk.RIGHT)

        self._grid = tk.Frame(self._top, padx=8, pady=4)
        self._grid.pack()
        for col, label in enumerate(_WEEKDAY_LABELS):
            tk.Label(self._grid, text=label, width=4, anchor="center").grid(
                row=0, column=col, padx=1, pady=(0, 4)
            )

        self._render_month()
        self._top.update_idletasks()
        self._top.grab_set()
        self._top.bind("<Escape>", lambda _e: self._close())

        ax = self._anchor.winfo_rootx()
        ay = self._anchor.winfo_rooty() + self._anchor.winfo_height() + 4
        width = max(self._top.winfo_reqwidth(), 260)
        height = self._top.winfo_reqheight()
        self._top.geometry(f"{width}x{height}+{ax}+{ay}")
        self._top.lift()
        self._top.focus_force()

    def _close(self) -> None:
        try:
            self._top.grab_release()
        except tk.TclError:
            pass
        self._top.destroy()

    def _month_title(self) -> str:
        return f"{calendar.month_name[self._month]} {self._year}"

    def _can_select(self, day: date) -> bool:
        return day <= self._max_date

    def _pick(self, day: date) -> None:
        if not self._can_select(day):
            return
        self._on_select(day)
        self._close()

    def _pick_today(self) -> None:
        self._pick(self._max_date)

    def _prev_month(self) -> None:
        if self._month == 1:
            self._month = 12
            self._year -= 1
        else:
            self._month -= 1
        self._render_month()

    def _next_month(self) -> None:
        probe = date(self._year + (1 if self._month == 12 else 0), (self._month % 12) + 1, 1)
        if probe > date(self._max_date.year, self._max_date.month, 1):
            return
        if self._month == 12:
            self._month = 1
            self._year += 1
        else:
            self._month += 1
        self._render_month()

    def _clear_day_buttons(self) -> None:
        for btn in self._day_buttons:
            btn.destroy()
        self._day_buttons.clear()

    def _render_month(self) -> None:
        self._title_var.set(self._month_title())
        self._clear_day_buttons()

        weeks = calendar.monthcalendar(self._year, self._month)
        for row_idx, week in enumerate(weeks, start=1):
            for col_idx, day_num in enumerate(week):
                if day_num == 0:
                    continue
                day = date(self._year, self._month, day_num)
                enabled = self._can_select(day)
                is_today = day == self._max_date
                is_initial = day == self._initial
                btn = tk.Button(
                    self._grid,
                    text=str(day_num),
                    width=3,
                    relief=tk.SUNKEN if is_initial else tk.RAISED,
                    state=tk.NORMAL if enabled else tk.DISABLED,
                    command=lambda d=day: self._pick(d),
                )
                if is_today and enabled:
                    btn.config(font=("TkDefaultFont", 9, "bold"))
                btn.grid(row=row_idx, column=col_idx, padx=1, pady=1)
                self._day_buttons.append(btn)


def register_rrg_date_entry(
    root: tk.Misc,
    entry: tk.Entry,
    var: tk.StringVar,
    *,
    default_date: date,
    max_date: date | None = None,
) -> None:
    """Validate DD-MM-YYYY typing on blur; empty field resets to ``default_date``."""
    max_d = max_date or default_date

    def _allow_char(proposed: str) -> bool:
        if proposed == "":
            return True
        if len(proposed) > 10:
            return False
        return all(c.isdigit() or c == "-" for c in proposed)

    entry.config(validate="key", validatecommand=(root.register(_allow_char), "%P"))

    def _on_leave(_event=None) -> None:
        raw = var.get().strip()
        if not raw:
            var.set(rrg_format_date(default_date))
            return
        try:
            picked = rrg_parse_user_date(raw).date()
            if picked > max_d:
                raise ValueError(f"Date cannot be after {rrg_format_date(max_d)}.")
            var.set(rrg_format_date(picked))
        except ValueError as exc:
            messagebox.showerror("Invalid date", str(exc), parent=root.winfo_toplevel())
            entry.focus_set()
            entry.selection_range(0, tk.END)

    entry.bind("<FocusOut>", _on_leave)


def open_rrg_date_picker(
    parent: tk.Misc,
    var: tk.StringVar,
    *,
    max_date: date,
    default_date: date | None = None,
) -> None:
    """Open calendar popup and write DD-MM-YYYY into ``var``."""
    fallback = default_date or max_date
    raw = var.get().strip()
    try:
        initial = rrg_parse_user_date(raw).date() if raw else fallback
    except ValueError:
        initial = fallback
    if initial > max_date:
        initial = max_date

    def _on_select(day: date) -> None:
        var.set(rrg_format_date(day))

    _DatePickerDialog(
        parent.winfo_toplevel(),
        anchor=parent,
        initial=initial,
        max_date=max_date,
        on_select=_on_select,
    )


def attach_rrg_date_picker(
    parent: tk.Misc,
    entry: tk.Entry,
    var: tk.StringVar,
    *,
    default_date: date,
    max_date: date | None = None,
) -> ttk.Button:
    """Register date validation and add a calendar button beside ``entry``."""
    root = parent.winfo_toplevel()
    max_d = max_date or default_date
    register_rrg_date_entry(root, entry, var, default_date=default_date, max_date=max_d)

    def _open_picker() -> None:
        open_rrg_date_picker(entry, var, max_date=max_d, default_date=default_date)

    entry.bind("<Double-Button-1>", lambda _e: _open_picker())
    btn = ttk.Button(parent, text="Pick", width=4, command=_open_picker)
    return btn
