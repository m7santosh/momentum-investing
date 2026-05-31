"""US ETF RRG swing recommendations — scored for momentum + reliability."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.etf.universes import us as _us_core
from momentum.rrg_core import RRG_COLOR_IMPROVING, RRG_COLOR_LEADING, get_status

# Near-duplicate products only — countries stay separate (EWY + EWT both allowed).
_OVERLAP_BUCKETS: tuple[frozenset[str], ...] = (
    frozenset({"SOXX", "SMH", "XSD"}),
    frozenset({"XSW", "SKYY", "IGV", "CLOU"}),
    frozenset({"CIBR", "HACK"}),
    frozenset({"IBB", "XBI"}),
    frozenset({"BITO", "IBIT"}),
    frozenset({"TAN", "ICLN", "QCLN", "FAN"}),
    frozenset({"FXI", "MCHI"}),
    frozenset({"KWEB", "CQQQ"}),
    frozenset({"GDX", "GDXJ"}),
    frozenset({"VNQ", "IYR", "XLRE"}),
)

# Smaller / frontier — penalized unless momentum score is very strong.
FRONTIER_TICKERS: frozenset[str] = frozenset(
    {"ARGT", "TUR", "GREK", "FM", "KSA", "ECH", "VNM", "EPOL", "EWW"}
)

CORE_US_TICKERS: frozenset[str] = frozenset(_us_core.tickers)

SATELLITE_VOL_PCT = 35.0
LOW_VOL_PCT = 22.0
HIGH_VOL_PCT = 45.0


@dataclass(frozen=True)
class UsEtfRecommendation:
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


def parse_rank_delta(text: str) -> int | None:
    text = (text or "").strip()
    if text in ("", "—", "-"):
        return None
    if text == "0":
        return 0
    if text.startswith("+"):
        return int(text[1:])
    return int(text)


def _bucket_for(ticker: str) -> frozenset[str] | None:
    for bucket in _OVERLAP_BUCKETS:
        if ticker in bucket:
            return bucket
    return None


def format_vol_pct(vol: float) -> str:
    if vol <= 0 or vol != vol:
        return ""
    return f"{vol:.1f}"


def _score_candidate(
    *,
    ticker: str,
    status: str,
    rsr_val: float,
    rsm_val: float,
    delta_val: int,
    change_pct: float,
    vol: float,
    table_rank: int,
    prev_table_rank: int | None,
) -> tuple[float, list[str]]:
    """Higher = better blend of RRG momentum and tradability."""
    score = 0.0
    reasons: list[str] = []

    if status == "leading":
        score += 38.0
        reasons.append("Leading vs SPY")
        if rsr_val > 100 and rsm_val > 100:
            score += 8.0
            reasons.append("RS ratio & momentum both > 100")
    else:
        score += 22.0
        reasons.append("Improving (early rotation)")
        if delta_val >= 20:
            score += 6.0
            reasons.append("Large Rank Δ while improving")

    # Momentum: rank acceleration weighted heavily
    score += min(float(delta_val), 80.0) * 1.1
    if delta_val >= 40:
        reasons.append(f"Rank Δ +{delta_val} (sharp rise)")
    elif delta_val >= 15:
        reasons.append(f"Rank Δ +{delta_val} (rising)")
    else:
        reasons.append(f"Rank Δ +{delta_val}")

    # Tail change — keep momentum, cap so one outlier does not dominate
    if change_pct > 0:
        score += min(change_pct, 28.0) * 0.55
        if change_pct >= 12.0:
            reasons.append(f"Change {change_pct:.1f}% over tail")

    # Reliability / tradability
    if ticker in CORE_US_TICKERS:
        score += 10.0
        reasons.append("Core us.py universe")

    if vol > 0:
        if vol < LOW_VOL_PCT:
            score += 14.0
            reasons.append(f"Low vol {vol:.0f}% (stable)")
        elif vol < SATELLITE_VOL_PCT:
            score += 7.0
            reasons.append(f"Moderate vol {vol:.0f}%")
        elif vol >= HIGH_VOL_PCT:
            score -= 12.0
            reasons.append(f"High vol {vol:.0f}% (size down)")

    if ticker in FRONTIER_TICKERS:
        score -= 18.0
        reasons.append("Frontier/small market (lower priority)")

    if table_rank <= 10:
        score += 4.0
        if prev_table_rank is None or prev_table_rank > 10:
            reasons.append("New in top 10 vs prior week")
        else:
            reasons.append("Top 10 by Change %")

    return score, reasons


def _build_reason(reasons: list[str], max_len: int = 120) -> str:
    text = "; ".join(reasons)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def recommend_us_etfs(
    *,
    ranked_row_indices: list[int],
    indices: list[str],
    display_labels: list[str],
    vol_by_ticker: dict[str, float],
    end_ts,
    rsr_series_by_row: list,
    rsm_series_by_row: list,
    rank_delta_by_row: dict[int, str],
    change_pct_fn,
    series_at_fn,
    curr_ranks: dict[int, int] | None = None,
    prev_ranks: dict[int, int] | None = None,
    limit: int = 7,
) -> list[UsEtfRecommendation]:
    """Score all eligible rows, then pick diversified top ``limit``."""
    curr_ranks = curr_ranks or {}
    prev_ranks = prev_ranks or {}
    candidates: list[tuple[float, int, list[str], str, float, str, str, float, str]] = []

    for table_rank, j in enumerate(ranked_row_indices, start=1):
        ticker = indices[j]
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
        vol = vol_by_ticker.get(ticker, 0.0)
        prev_tr = prev_ranks.get(j)
        score, reasons = _score_candidate(
            ticker=ticker,
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
                reasons,
                ticker,
                chg,
                delta_text,
                vol,
                status.capitalize(),
            )
        )

    candidates.sort(key=lambda row: (-row[0], row[3]))

    picks: list[UsEtfRecommendation] = []
    used_buckets: set[frozenset[str]] = set()

    for score, j, reasons, ticker, chg, delta_text, vol, quadrant in candidates:
        if len(picks) >= limit:
            break
        bucket = _bucket_for(ticker)
        if bucket is not None and bucket in used_buckets:
            continue
        size_hint = "Satellite" if vol >= SATELLITE_VOL_PCT else "Core"
        picks.append(
            UsEtfRecommendation(
                pick_rank=len(picks) + 1,
                row_idx=j,
                ticker=ticker,
                name=display_labels[j],
                change_pct=chg,
                rank_delta=delta_text,
                vol_pct=vol,
                quadrant=quadrant,
                size_hint=size_hint,
                score=score,
                reason=_build_reason(reasons),
            )
        )
        if bucket is not None:
            used_buckets.add(bucket)

    return picks


def recommendation_row_bg(quadrant: str) -> str:
    q = quadrant.lower()
    if q == "leading":
        return RRG_COLOR_LEADING
    if q == "improving":
        return RRG_COLOR_IMPROVING
    return "#E8E8E8"
