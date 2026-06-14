"""Cooperative cancellation for backtest background workers."""

from __future__ import annotations

import threading
from collections.abc import Callable


class BacktestCancelled(Exception):
    """Raised when the user cancels a long-running backtest operation."""


CancelCheck = Callable[[], None]


def cancel_check_from_event(event: threading.Event | None) -> CancelCheck | None:
    if event is None:
        return None

    def _check() -> None:
        if event.is_set():
            raise BacktestCancelled()

    return _check


def check_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()
