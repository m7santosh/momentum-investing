"""Shared RRG (Relative Rotation Graph) indicator math and row metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RrgRow:
    """One RRG universe line."""

    row_id: str
    label: str
    ref_label: str
    kind: str  # "index" | "etf" | "stock"


def compute_rrg_indicators(index_series, benchmark_series, window=14):
    rs = 100 * (index_series / benchmark_series)
    rsr = (
        100
        + (rs - rs.rolling(window=window).mean())
        / rs.rolling(window=window).std(ddof=0)
    ).dropna()
    if len(rsr) < 2:
        return None, None, None
    rsr_roc = 100 * ((rsr / rsr.iloc[0]) - 1)
    rsm = (
        101
        + (
            (rsr_roc - rsr_roc.rolling(window=window).mean())
            / rsr_roc.rolling(window=window).std(ddof=0)
        )
    ).dropna()
    rsr = rsr[rsr.index.isin(rsm.index)]
    rsm = rsm[rsm.index.isin(rsr.index)]
    if len(rsr) < 2:
        return None, None, None
    return rsr, rsr_roc, rsm


def get_status(x, y):
    if x < 100 and y < 100:
        return "lagging"
    if x > 100 and y > 100:
        return "leading"
    if x < 100 and y > 100:
        return "improving"
    if x > 100 and y < 100:
        return "weakening"
    return None


def get_color(x, y):
    status = get_status(x, y)
    if status == "lagging":
        return "red"
    if status == "leading":
        return "green"
    if status == "improving":
        return "blue"
    if status == "weakening":
        return "yellow"
    return "gray"


TAIL_MARKER_SIZE = 22
HEAD_ARROW_SCALE = 14
HOVER_PIXEL_RADIUS = 14
RRG_NAV_WEEKS = 26
