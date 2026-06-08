"""Animated busy overlay for RRG Tk apps (India / US / stock / backtest)."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TypeVar

import tkinter as tk

T = TypeVar("T")


class _SpinnerCanvas(tk.Canvas):
    """Rotating-dot loader (animates via ``after`` while the event loop runs)."""

    _DOTS = 8
    _INTERVAL_MS = 75

    def __init__(
        self,
        parent: tk.Misc,
        *,
        size: int = 44,
        color: str = "#2563eb",
        bg: str = "#ffffff",
    ) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            bg=bg,
            bd=0,
        )
        self._size = size
        self._cx = size / 2
        self._cy = size / 2
        self._color = color
        self._bg = bg
        self._step = 0
        self._after_id: str | None = None

    def start(self) -> None:
        self._step = 0
        self._tick()

    def stop(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.delete("all")

    def _tick(self) -> None:
        self.delete("all")
        radius = self._size * 0.34
        dot_r = max(2.5, self._size * 0.07)
        for i in range(self._DOTS):
            angle = math.radians((self._step + i) * (360 / self._DOTS) - 90)
            x = self._cx + radius * math.cos(angle)
            y = self._cy + radius * math.sin(angle)
            alpha = 1.0 - (i / self._DOTS) * 0.82
            fill = _blend_hex(self._color, self._bg, 1.0 - alpha)
            self.create_oval(
                x - dot_r,
                y - dot_r,
                x + dot_r,
                y + dot_r,
                fill=fill,
                outline="",
            )
        self._step = (self._step + 1) % self._DOTS
        self._after_id = self.after(self._INTERVAL_MS, self._tick)


def _blend_hex(fg: str, bg: str, t: float) -> str:
    """Blend foreground toward background (t=0 → fg, t=1 → bg)."""
    t = max(0.0, min(1.0, t))
    fg = fg.lstrip("#")
    bg = bg.lstrip("#")
    if len(fg) != 6 or len(bg) != 6:
        return f"#{fg}"
    fr, fg_g, fb = int(fg[0:2], 16), int(fg[2:4], 16), int(fg[4:6], 16)
    br, bg_g, bb = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
    r = int(fr + (br - fr) * t)
    g = int(fg_g + (bg_g - fg_g) * t)
    b = int(fb + (bb - fb) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class RrgBusyOverlay:
    """Modal loading overlay with animated spinner and compact status text."""

    def __init__(self, root: tk.Misc, *, default_message: str = "Loading…") -> None:
        self._root = root
        self._depth = 0
        self._default_message = default_message
        self._frame: tk.Frame | None = None
        self._label: tk.Label | None = None
        self._spinner: _SpinnerCanvas | None = None

    def show(self, message: str | None = None) -> None:
        self._depth += 1
        if self._frame is None:
            self._build()
        self.update_message(message)
        self._root.config(cursor="watch")
        assert self._frame is not None
        self._frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._frame.lift()
        if self._depth == 1:
            assert self._spinner is not None
            self._spinner.start()
            try:
                self._frame.grab_set()
            except tk.TclError:
                pass
        self._root.update_idletasks()

    def update_message(self, message: str | None = None) -> None:
        if self._label is None:
            return
        msg = (message or self._default_message).strip() or self._default_message
        self._label.config(text=msg)

    def hide(self) -> None:
        if self._depth <= 0:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        if self._spinner is not None:
            self._spinner.stop()
        if self._frame is not None:
            try:
                self._frame.grab_release()
            except tk.TclError:
                pass
            self._frame.place_forget()
        self._root.config(cursor="")

    def run(self, message: str, fn: Callable[[], T]) -> T:
        self.show(message)
        try:
            return fn()
        finally:
            self.hide()

    def run_threaded(self, fn: Callable[[], T], message: str | None = None) -> T:
        """
        Run blocking work off the UI thread; spinner keeps animating.

        Safe for network / disk loads. Do not touch Tk widgets inside ``fn``.
        """
        self.show(message)
        result: list[T] = []
        error: list[BaseException] = []
        done = threading.Event()

        def worker() -> None:
            try:
                result.append(fn())
            except BaseException as exc:
                error.append(exc)
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        try:
            while not done.is_set():
                self._root.update()
                done.wait(0.03)
        finally:
            self.hide()
        if error:
            raise error[0]
        return result[0]

    @contextmanager
    def busy(self, message: str | None = None) -> Generator[None, None, None]:
        self.show(message)
        try:
            yield
        finally:
            self.hide()

    def _build(self) -> None:
        self._frame = tk.Frame(self._root, bg="#3a3a3a", highlightthickness=0)
        card = tk.Frame(
            self._frame,
            bg="#ffffff",
            padx=22,
            pady=16,
            highlightbackground="#c8c8c8",
            highlightthickness=1,
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        self._spinner = _SpinnerCanvas(card, size=44, color="#2563eb", bg="#ffffff")
        self._spinner.pack()
        self._label = tk.Label(
            card,
            text=self._default_message,
            font=("Segoe UI", 8),
            bg="#ffffff",
            fg="#666666",
            wraplength=240,
            justify="center",
        )
        self._label.pack(pady=(10, 0))
