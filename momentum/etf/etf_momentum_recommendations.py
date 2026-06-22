"""Synthesize top ETF picks and reasons from Abs / RS Blended / RS Adaptive screens."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TOP_PICKS = 10


@dataclass(frozen=True)
class EtfPick:
    rank: int
    symbol: str
    reason: str
    score: float


def _position_map(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", "")).strip()
        pos = row.get("Position")
        if sym and pos is not None and not pd.isna(pos):
            out[sym] = float(pos)
    return out


def _row_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df is None or df.empty:
        return {}
    return {str(row["Symbol"]).strip(): row for _, row in df.iterrows() if pd.notna(row.get("Symbol"))}


def _fmt_pct(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return f"{float(value):+.1f}%"


def _build_reason(
    *,
    abs_pos: float | None,
    blend_pos: float | None,
    adapt_pos: float | None,
    row: pd.Series,
) -> str:
    parts: list[str] = []
    if abs_pos is not None:
        parts.append(f"Abs #{int(abs_pos)}")
    if blend_pos is not None:
        parts.append(f"RS Blended #{int(blend_pos)}")
    if adapt_pos is not None:
        parts.append(f"RS Adaptive #{int(adapt_pos)}")

    pct_since = row.get("Pct_Above_9EMA")
    pct_s = _fmt_pct(pct_since)
    if pct_s:
        parts.append(f"above 9 EMA ({pct_s} since cross)")
    else:
        parts.append("above 9 EMA")

    for label, col in (("1W", "Return_1W"), ("1M", "Return_1M"), ("3M", "Return_3M")):
        ret = _fmt_pct(row.get(col))
        if ret:
            parts.append(f"{label} {ret}")

    return " · ".join(parts)


def _score_pick(
    *,
    abs_pos: float | None,
    blend_pos: float | None,
    adapt_pos: float | None,
    row: pd.Series,
) -> float:
    score = 0.0
    screens = sum(p is not None for p in (abs_pos, blend_pos, adapt_pos))
    score += screens * 40.0

    for pos in (abs_pos, blend_pos, adapt_pos):
        if pos is not None:
            score += max(0.0, 35.0 - pos)

    score += 12.0
    pct = row.get("Pct_Above_9EMA")
    if pct is not None and not pd.isna(pct):
        score += min(float(pct), 25.0) * 0.2

    ret_1m = row.get("Return_1M")
    if ret_1m is not None and not pd.isna(ret_1m):
        score += float(ret_1m) * 0.15

    return score


def _scored_candidates(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
) -> list[EtfPick]:
    """All above-9-EMA ETFs scored across Abs / RS Blended / RS Adaptive screens."""
    abs_pos = _position_map(abs_df)
    blend_pos = _position_map(rs_blended_df)
    adapt_pos = _position_map(rs_adaptive_df)

    rows = _row_map(rs_adaptive_df)
    rows.update(_row_map(rs_blended_df))
    rows.update(_row_map(abs_df))

    symbols = set(abs_pos) | set(blend_pos) | set(adapt_pos)
    if not symbols:
        return []

    candidates: list[EtfPick] = []
    for sym in symbols:
        row = rows[sym]
        if row.get("Close_Below_9EMA") != "Hold":
            continue
        a_pos = abs_pos.get(sym)
        b_pos = blend_pos.get(sym)
        d_pos = adapt_pos.get(sym)
        reason = _build_reason(abs_pos=a_pos, blend_pos=b_pos, adapt_pos=d_pos, row=row)
        score = _score_pick(abs_pos=a_pos, blend_pos=b_pos, adapt_pos=d_pos, row=row)
        candidates.append(EtfPick(rank=0, symbol=sym, reason=reason, score=score))

    candidates.sort(key=lambda p: (-p.score, p.symbol))
    return candidates


def recommend_top_etfs(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
    *,
    top_n: int = TOP_PICKS,
) -> list[EtfPick]:
    """Rank ETFs above 9 EMA that appear on any screen; prefer multi-screen leaders."""
    candidates = _scored_candidates(abs_df, rs_blended_df, rs_adaptive_df)
    out: list[EtfPick] = []
    for i, pick in enumerate(candidates[:top_n], start=1):
        out.append(EtfPick(rank=i, symbol=pick.symbol, reason=pick.reason, score=pick.score))
    return out


def recommendation_rank_dataframe(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
) -> pd.DataFrame:
    """Backtest ranking: all scored above-9-EMA candidates with ``Rank_Position``."""
    candidates = _scored_candidates(abs_df, rs_blended_df, rs_adaptive_df)
    if not candidates:
        return pd.DataFrame(columns=["Symbol", "Rank_Position"])
    return pd.DataFrame(
        [
            {"Symbol": pick.symbol, "Rank_Position": i}
            for i, pick in enumerate(candidates, start=1)
        ]
    )


def recommendations_dataframe(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
    *,
    top_n: int = TOP_PICKS,
) -> pd.DataFrame:
    picks = recommend_top_etfs(abs_df, rs_blended_df, rs_adaptive_df, top_n=top_n)
    if not picks:
        return pd.DataFrame(columns=["Rank", "Symbol", "Reason"])
    return pd.DataFrame(
        [{"Rank": p.rank, "Symbol": p.symbol, "Reason": p.reason} for p in picks]
    )
