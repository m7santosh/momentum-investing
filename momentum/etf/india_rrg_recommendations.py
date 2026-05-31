"""India NSE ETF RRG swing recommendations — scored for momentum + reliability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from momentum.etf.universes import india as _india_core
from momentum.etf.us_rrg_recommendations import (
    _build_concise_reason,
    _build_reason,
    format_vol_pct,
    parse_rank_delta,
    recommendation_row_bg,
)
from momentum.rrg_core import get_status
from utils.nse_bhavcopy import today_ist

# Cash-equivalent sleeve — not a tactical swing target.
EXCLUDE_REF_ETFS: frozenset[str] = frozenset({"LIQUIDCASE"})

# Near-duplicate tradable symbols (ref ETF column).
_OVERLAP_BUCKETS: tuple[frozenset[str], ...] = (
    frozenset({"MIDCAPETF", "MOMIDMTM", "HDFCSML250"}),
    frozenset({"NIFTYBEES", "NEXT50IETF", "MOVALUE"}),
    frozenset({"MON100", "MAFANG"}),
    frozenset({"HNGSNGBEES", "MAHKTECH"}),
    frozenset({"PSUBNKBEES", "PVTBANIETF"}),
)

CORE_REF_ETFS: frozenset[str] = frozenset(
    t.replace(".NS", "") for t in _india_core.DEFAULT_VISIBLE
)

SATELLITE_VOL_PCT = 28.0
LOW_VOL_PCT = 16.0
HIGH_VOL_PCT = 38.0


@dataclass(frozen=True)
class IndiaEtfRecommendation:
    pick_rank: int
    row_idx: int
    ticker: str
    name: str
    change_pct: float
    rank_delta: str
    vol_pct: float
    quadrant: str
    size_hint: str
    score: float
    reason: str


def _yahoo_symbol(ref: str) -> str:
    text = (ref or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".NS") else f"{text}.NS"


def load_india_etf_vol_pct(
    row_ids: list[str],
    ref_labels: list[str],
    *,
    vol_days: int = 63,
    history_days: int = 120,
) -> dict[str, float]:
    """Ann. vol % keyed by ref ETF symbol (no .NS suffix)."""
    ref_to_yahoo: dict[str, str] = {}
    for row_id, ref in zip(row_ids, ref_labels):
        sym = (ref or row_id).strip().upper()
        if not sym:
            continue
        bare = sym.replace(".NS", "")
        ref_to_yahoo[bare] = _yahoo_symbol(bare)

    yahoo_symbols = sorted(set(ref_to_yahoo.values()))
    if not yahoo_symbols:
        return {}

    end = today_ist()
    start = end - timedelta(days=history_days)
    end_dl = end + timedelta(days=1)
    close_by_yahoo: dict[str, pd.Series] = {}
    batch_size = 25

    for i in range(0, len(yahoo_symbols), batch_size):
        batch = yahoo_symbols[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end_dl,
                group_by="column",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            raw = None
        for sym in batch:
            series = _extract_close(raw, sym)
            if series is not None:
                close_by_yahoo[sym] = series

    out: dict[str, float] = {}
    for bare, ysym in ref_to_yahoo.items():
        close = close_by_yahoo.get(ysym)
        if close is None or len(close) < 10:
            continue
        rets = close.sort_index().pct_change().dropna()
        vol_slice = rets.tail(vol_days)
        if len(vol_slice) < max(10, vol_days // 3):
            continue
        vol_ann = float(vol_slice.std(ddof=0) * np.sqrt(252) * 100)
        if np.isfinite(vol_ann) and vol_ann > 0:
            out[bare] = vol_ann
    return out


def _extract_close(raw, ticker: str) -> pd.Series | None:
    if raw is None or raw.empty:
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close", ticker) in raw.columns:
                close = raw["Close"][ticker]
            elif ("Adj Close", ticker) in raw.columns:
                close = raw["Adj Close"][ticker]
            else:
                return None
        elif "Close" in raw.columns:
            close = raw["Close"].squeeze()
        elif "Adj Close" in raw.columns:
            close = raw["Adj Close"].squeeze()
        else:
            return None
    except (KeyError, TypeError):
        return None
    close = pd.to_numeric(close, errors="coerce").dropna()
    return close if len(close) >= 5 else None


def _bucket_for_ref(ref: str) -> frozenset[str] | None:
    bare = ref.replace(".NS", "").upper()
    for bucket in _OVERLAP_BUCKETS:
        if bare in bucket:
            return bucket
    return None


def _score_candidate(
    *,
    ref: str,
    status: str,
    rsr_val: float,
    rsm_val: float,
    delta_val: int,
    change_pct: float,
    vol: float,
    table_rank: int,
    prev_table_rank: int | None,
) -> tuple[float, str]:
    score = 0.0

    if status == "leading":
        score += 38.0
        if rsr_val > 100 and rsm_val > 100:
            score += 8.0
    else:
        score += 22.0
        if delta_val >= 20:
            score += 6.0

    score += min(float(delta_val), 80.0) * 1.1
    if change_pct > 0:
        score += min(change_pct, 28.0) * 0.55

    bare = ref.replace(".NS", "").upper()
    if bare in CORE_REF_ETFS:
        score += 10.0
    if vol > 0:
        if vol < LOW_VOL_PCT:
            score += 12.0
        elif vol < SATELLITE_VOL_PCT:
            score += 6.0
        elif vol >= HIGH_VOL_PCT:
            score -= 10.0
    if bare in {"GOLDBEES", "SILVERBEES"}:
        score -= 8.0
    if table_rank <= 10:
        score += 4.0

    tags: list[str] = []
    if bare in CORE_REF_ETFS:
        tags.append("Core india.py")
    if bare in {"GOLDBEES", "SILVERBEES"}:
        tags.append("Commodity sleeve")
    if vol >= HIGH_VOL_PCT:
        tags.append("High vol")

    reason = _build_concise_reason(
        status=status,
        rsr_val=rsr_val,
        rsm_val=rsm_val,
        benchmark="Nifty 500",
        delta_val=delta_val,
        table_rank=table_rank,
        prev_table_rank=prev_table_rank,
        change_pct=change_pct,
        tags=tags,
    )
    return score, reason


def recommend_india_etfs(
    *,
    ranked_row_indices: list[int],
    indices: list[str],
    ref_labels: list[str],
    display_labels: list[str],
    vol_by_ref: dict[str, float],
    end_ts,
    rsr_series_by_row: list,
    rsm_series_by_row: list,
    rank_delta_by_row: dict[int, str],
    change_pct_fn,
    series_at_fn,
    curr_ranks: dict[int, int] | None = None,
    prev_ranks: dict[int, int] | None = None,
    limit: int = 7,
) -> list[IndiaEtfRecommendation]:
    """Score eligible India RRG rows; pick diversified top ``limit`` by ref ETF."""
    prev_ranks = prev_ranks or {}
    candidates: list[tuple[float, int, str, str, str, float, str, str, float]] = []

    for table_rank, j in enumerate(ranked_row_indices, start=1):
        ref = (ref_labels[j] or indices[j]).strip().upper().replace(".NS", "")
        if not ref or ref in EXCLUDE_REF_ETFS:
            continue
        delta_text = rank_delta_by_row.get(j, "—")
        delta_val = parse_rank_delta(delta_text)
        if delta_val is None or delta_val <= 0:
            continue
        try:
            rsr_val = float(series_at_fn(rsr_series_by_row[j], end_ts))
            rsm_val = float(series_at_fn(rsm_series_by_row[j], end_ts))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        status = get_status(rsr_val, rsm_val)
        if status not in ("leading", "improving"):
            continue
        chg = change_pct_fn(j)
        if chg == float("-inf"):
            continue
        vol = vol_by_ref.get(ref, 0.0)
        prev_tr = prev_ranks.get(j)
        score, reason = _score_candidate(
            ref=ref,
            status=status,
            rsr_val=rsr_val,
            rsm_val=rsm_val,
            delta_val=delta_val,
            change_pct=chg,
            vol=vol,
            table_rank=table_rank,
            prev_table_rank=prev_tr,
        )
        candidates.append(
            (
                score,
                j,
                reason,
                ref,
                display_labels[j],
                chg,
                delta_text,
                vol,
                status.capitalize(),
            )
        )

    candidates.sort(key=lambda row: (-row[0], row[3]))

    picks: list[IndiaEtfRecommendation] = []
    used_buckets: set[frozenset[str]] = set()

    for score, j, reason, ref, name, chg, delta_text, vol, quadrant in candidates:
        if len(picks) >= limit:
            break
        bucket = _bucket_for_ref(ref)
        if bucket is not None and bucket in used_buckets:
            continue
        size_hint = "Satellite" if vol >= SATELLITE_VOL_PCT else "Core"
        picks.append(
            IndiaEtfRecommendation(
                pick_rank=len(picks) + 1,
                row_idx=j,
                ticker=ref,
                name=name,
                change_pct=chg,
                rank_delta=delta_text,
                vol_pct=vol,
                quadrant=quadrant,
                size_hint=size_hint,
                score=score,
                reason=_build_reason(reason),
            )
        )
        if bucket is not None:
            used_buckets.add(bucket)

    return picks


__all__ = [
    "IndiaEtfRecommendation",
    "format_vol_pct",
    "load_india_etf_vol_pct",
    "recommend_india_etfs",
    "recommendation_row_bg",
]
