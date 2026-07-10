"""Synthesize top ETF picks and reasons from Abs / RS Blended / RS Adaptive screens."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TOP_PICKS = 20
US_TOP_PICKS = 30
SCREEN_TOP_N = 15


@dataclass(frozen=True)
class EtfPick:
    rank: int
    symbol: str
    reason: str
    score: float
    name: str | None = None


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


def _top_n_position_map(df: pd.DataFrame, *, top_n: int = SCREEN_TOP_N) -> dict[str, float]:
    """Symbol → Position for ETFs ranked in the top *top_n* on one screen."""
    return {sym: pos for sym, pos in _position_map(df).items() if pos <= top_n}


def _row_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df is None or df.empty:
        return {}
    return {str(row["Symbol"]).strip(): row for _, row in df.iterrows() if pd.notna(row.get("Symbol"))}


def _fmt_pct(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return f"{float(value):+.1f}%"


def _fmt_cross_date(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _row_name(row: pd.Series) -> str | None:
    name = row.get("Name")
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = str(name).strip()
    return text or None


def _build_reason(
    *,
    abs_pos: float | None,
    blend_pos: float | None,
    adapt_pos: float | None,
    screen_count: int,
    row: pd.Series,
) -> str:
    cross = _fmt_cross_date(row.get("Above_9EMA_Since"))
    pct_s = _fmt_pct(row.get("Pct_Above_9EMA"))
    parts: list[str] = []

    if cross and pct_s:
        parts.append(f"9 EMA cross {cross} ({pct_s})")
    elif cross:
        parts.append(f"9 EMA cross {cross}")
    elif pct_s:
        parts.append(f"{pct_s} since 9 EMA cross")

    for label, col in (
        ("1W", "Return_1W"),
        ("2W", "Return_2W"),
        ("1M", "Return_1M"),
    ):
        ret = _fmt_pct(row.get(col))
        if ret:
            parts.append(f"{label} {ret}")

    parts.append(f"{screen_count}/3 screens")
    if abs_pos is not None:
        parts.append(f"Abs #{int(abs_pos)}")
    if blend_pos is not None:
        parts.append(f"RS Blended #{int(blend_pos)}")
    if adapt_pos is not None:
        parts.append(f"RS Adaptive #{int(adapt_pos)}")

    return " · ".join(parts)


def _score_pick(
    *,
    abs_pos: float | None,
    blend_pos: float | None,
    adapt_pos: float | None,
    screen_count: int,
    row: pd.Series,
) -> float:
    """Prefer more screens, stronger ranks, then recent returns."""
    score = screen_count * 50.0

    for pos in (abs_pos, blend_pos, adapt_pos):
        if pos is not None:
            score += max(0.0, 35.0 - pos)

    pct = row.get("Pct_Above_9EMA")
    if pct is not None and not pd.isna(pct):
        score += min(float(pct), 25.0) * 0.15

    for col, weight in (
        ("Return_1W", 0.12),
        ("Return_2W", 0.10),
        ("Return_1M", 0.15),
    ):
        ret = row.get(col)
        if ret is not None and not pd.isna(ret):
            score += float(ret) * weight

    return score


def _scored_candidates(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
    *,
    screen_top_n: int = SCREEN_TOP_N,
) -> list[EtfPick]:
    """ETFs in any screener top *screen_top_n*, above 9 EMA; rank by screen overlap + metrics."""
    abs_top = _top_n_position_map(abs_df, top_n=screen_top_n)
    blend_top = _top_n_position_map(rs_blended_df, top_n=screen_top_n)
    adapt_top = _top_n_position_map(rs_adaptive_df, top_n=screen_top_n)

    symbols = set(abs_top) | set(blend_top) | set(adapt_top)
    if not symbols:
        return []

    rows = _row_map(rs_adaptive_df)
    rows.update(_row_map(rs_blended_df))
    rows.update(_row_map(abs_df))

    candidates: list[EtfPick] = []
    for sym in symbols:
        row = rows[sym]
        if row.get("Close_Below_9EMA") != "Hold":
            continue
        a_pos = abs_top.get(sym)
        b_pos = blend_top.get(sym)
        d_pos = adapt_top.get(sym)
        screen_count = sum(p is not None for p in (a_pos, b_pos, d_pos))
        reason = _build_reason(
            abs_pos=a_pos,
            blend_pos=b_pos,
            adapt_pos=d_pos,
            screen_count=screen_count,
            row=row,
        )
        score = _score_pick(
            abs_pos=a_pos,
            blend_pos=b_pos,
            adapt_pos=d_pos,
            screen_count=screen_count,
            row=row,
        )
        candidates.append(
            EtfPick(
                rank=0,
                symbol=sym,
                reason=reason,
                score=score,
                name=_row_name(row),
            )
        )

    candidates.sort(key=lambda p: (-p.score, p.symbol))
    return candidates


def recommend_top_etfs(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
    *,
    top_n: int = TOP_PICKS,
    screen_top_n: int = SCREEN_TOP_N,
) -> list[EtfPick]:
    """Top ETFs from screener top lists; favor multi-screen overlap, ranks, and returns."""
    effective_screen_top_n = max(screen_top_n, top_n)
    candidates = _scored_candidates(
        abs_df,
        rs_blended_df,
        rs_adaptive_df,
        screen_top_n=effective_screen_top_n,
    )
    out: list[EtfPick] = []
    for i, pick in enumerate(candidates[:top_n], start=1):
        out.append(
            EtfPick(
                rank=i,
                symbol=pick.symbol,
                reason=pick.reason,
                score=pick.score,
                name=pick.name,
            )
        )
    return out


def recommendation_rank_dataframe(
    abs_df: pd.DataFrame,
    rs_blended_df: pd.DataFrame,
    rs_adaptive_df: pd.DataFrame,
    *,
    screen_top_n: int = SCREEN_TOP_N,
) -> pd.DataFrame:
    """Backtest ranking: scored top-screen ETFs above 9 EMA with ``Rank_Position``."""
    candidates = _scored_candidates(
        abs_df, rs_blended_df, rs_adaptive_df, screen_top_n=screen_top_n
    )
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
    screen_top_n: int = SCREEN_TOP_N,
    include_name: bool = False,
) -> pd.DataFrame:
    picks = recommend_top_etfs(
        abs_df,
        rs_blended_df,
        rs_adaptive_df,
        top_n=top_n,
        screen_top_n=screen_top_n,
    )
    if not picks:
        cols = ["Rank", "Symbol", "Reason"]
        if include_name:
            cols.insert(2, "Name")
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    for p in picks:
        row: dict[str, object] = {"Rank": p.rank, "Symbol": p.symbol, "Reason": p.reason}
        if include_name:
            row["Name"] = p.name or ""
        rows.append(row)
    return pd.DataFrame(rows)
