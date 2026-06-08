"""ADV$/vol metrics for the canonical us.py ETF list (no extra discovery tickers)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from momentum.etf.universes import us_universe as us_core
from momentum.etf.universes import us_liquid_candidates as pool
from utils.nse_bhavcopy import today_ist


@dataclass(frozen=True)
class ScreenedEtf:
    ticker: str
    label: str
    category: str
    adv_usd: float
    vol_ann_pct: float
    pinned: bool = False  # core us.py — always in RRG


def _candidate_tickers(categories: set[str]) -> list[str]:
    """Always the canonical ``us.py`` list (category filter kept for CLI compat)."""
    _ = categories
    return list(us_core.TICKERS)


def _extract_close_volume(raw: pd.DataFrame, ticker: str) -> tuple[pd.Series, pd.Series] | None:
    if raw is None or len(raw) == 0:
        return None

    close: pd.Series | None = None
    volume: pd.Series | None = None

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Close" in lvl0:
            close_df = raw["Close"]
        elif "Adj Close" in lvl0:
            close_df = raw["Adj Close"]
        else:
            return None
        if "Volume" not in lvl0:
            return None
        vol_df = raw["Volume"]
        if isinstance(close_df, pd.Series):
            close = close_df.squeeze()
        elif ticker in close_df.columns:
            close = close_df[ticker]
        if isinstance(vol_df, pd.Series):
            volume = vol_df.squeeze()
        elif ticker in vol_df.columns:
            volume = vol_df[ticker]
    else:
        if "Close" in raw.columns:
            close = raw["Close"].squeeze()
        elif "Adj Close" in raw.columns:
            close = raw["Adj Close"].squeeze()
        if "Volume" in raw.columns:
            volume = raw["Volume"].squeeze()

    if close is None or volume is None:
        return None
    close = pd.to_numeric(close, errors="coerce").dropna()
    volume = pd.to_numeric(volume, errors="coerce").reindex(close.index).fillna(0)
    if len(close) < 5:
        return None
    return close, volume


def _metrics_from_series(
    close: pd.Series, volume: pd.Series, *, adv_days: int, vol_days: int
) -> tuple[float, float] | None:
    close = close.sort_index()
    volume = volume.reindex(close.index).fillna(0)
    dollar = close * volume
    adv_slice = dollar.tail(adv_days)
    if len(adv_slice) < max(5, adv_days // 2):
        return None
    adv_usd = float(adv_slice.mean())
    if adv_usd <= 0 or not np.isfinite(adv_usd):
        return None

    rets = close.pct_change().dropna()
    vol_slice = rets.tail(vol_days)
    if len(vol_slice) < max(10, vol_days // 3):
        return None
    vol_ann = float(vol_slice.std(ddof=0) * np.sqrt(252) * 100)
    if not np.isfinite(vol_ann):
        return None
    return adv_usd, vol_ann


def _fetch_metrics(
    tickers: list[str],
    *,
    adv_days: int,
    vol_days: int,
    history_days: int,
    quiet: bool,
) -> dict[str, tuple[float, float]]:
    end = today_ist()
    start = end - timedelta(days=history_days)
    end_dl = end + timedelta(days=1)
    out: dict[str, tuple[float, float]] = {}
    batch_size = 40

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
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
        except Exception as exc:
            if not quiet:
                print(f"  batch download failed: {exc}")
            raw = None

        for ticker in batch:
            extracted = _extract_close_volume(raw, ticker) if raw is not None else None
            if extracted is None:
                try:
                    raw_one = yf.download(
                        ticker,
                        start=start,
                        end=end_dl,
                        auto_adjust=True,
                        progress=False,
                    )
                    extracted = _extract_close_volume(raw_one, ticker)
                except Exception:
                    extracted = None
            if extracted is None:
                continue
            row_metrics = _metrics_from_series(
                extracted[0], extracted[1], adv_days=adv_days, vol_days=vol_days
            )
            if row_metrics is not None:
                out[ticker] = row_metrics
    return out


def screen_us_etfs(
    *,
    categories: set[str] | None = None,
    min_adv_usd: float = 10_000_000,
    vol_percentile: float = 100.0,
    adv_days: int = 20,
    vol_days: int = 63,
    history_days: int = 120,
    always_include: frozenset[str] | None = None,
    quiet: bool = False,
) -> list[ScreenedEtf]:
    """Return all ``us.py`` ETFs with ADV$/vol metrics (optional vol cap).

    ``vol_percentile=100`` disables the vol filter (ADV$ only).
    ``always_include`` defaults to ``us.py`` tickers; every symbol is kept.
    """
    cats = categories or set(pool.ALL_CATEGORIES)
    tickers = _candidate_tickers(cats)
    pinned = always_include if always_include is not None else frozenset(us_core.TICKERS)
    if not tickers:
        return []

    apply_vol = vol_percentile < 100.0
    if not quiet:
        vol_msg = f"vol <= p{vol_percentile:.0f}" if apply_vol else "vol filter off"
        print(
            f"Screening {len(tickers)} US ETFs from us.py "
            f"(min ADV$ {min_adv_usd:,.0f} for metrics only; {vol_msg})..."
        )

    metrics = _fetch_metrics(
        tickers, adv_days=adv_days, vol_days=vol_days, history_days=history_days, quiet=quiet
    )

    adv_pass: list[tuple[str, float, float]] = []
    for ticker, (adv_usd, vol_ann) in metrics.items():
        if ticker in pinned or adv_usd >= min_adv_usd:
            adv_pass.append((ticker, adv_usd, vol_ann))

    # Core us.py tickers always in universe even without Yahoo metrics.
    in_pass = {t for t, _, _ in adv_pass}
    for ticker in pinned:
        if ticker not in in_pass:
            adv_pass.append((ticker, 0.0, float("nan")))

    if not adv_pass:
        if not quiet:
            print("No ETFs in universe.")
        return []

    vol_cutoff: float | None = None
    if apply_vol:
        vols = [v for _, _, v in adv_pass if np.isfinite(v)]
        if vols:
            vol_cutoff = float(np.percentile(vols, vol_percentile))

    passed: list[ScreenedEtf] = []
    for ticker, adv_usd, vol_ann in adv_pass:
        is_pinned = ticker in pinned
        if (
            vol_cutoff is not None
            and np.isfinite(vol_ann)
            and vol_ann > vol_cutoff
            and not is_pinned
        ):
            continue
        passed.append(
            ScreenedEtf(
                ticker=ticker,
                label=pool.ETF_LABELS.get(ticker, ticker),
                category=pool.ETF_CATEGORY.get(ticker, pool.CATEGORY_CORE),
                adv_usd=adv_usd,
                vol_ann_pct=vol_ann if np.isfinite(vol_ann) else 0.0,
                pinned=is_pinned,
            )
        )

    order = {sym: i for i, sym in enumerate(us_core.TICKERS)}
    passed.sort(key=lambda r: order.get(r.ticker, len(order)))

    if not quiet:
        if apply_vol and vol_cutoff is not None:
            print(
                f"  Universe: {len(passed)} ETFs from us.py "
                f"(vol <= {vol_cutoff:.1f}% when filter on)"
            )
        else:
            print(f"  Universe: {len(passed)} ETFs from us.py")

    return passed


def format_screen_table(rows: list[ScreenedEtf]) -> str:
    if not rows:
        return "(none)"
    lines = [
        f"{'':1} {'Ticker':<7} {'Category':<10} {'ADV$M':>8} {'Vol%':>7}  Name",
        "-" * 76,
    ]
    for r in rows:
        pin = "*" if r.pinned else " "
        adv_s = f"{r.adv_usd / 1e6:>8.1f}" if r.adv_usd > 0 else "       -"
        vol_s = f"{r.vol_ann_pct:>7.1f}" if r.vol_ann_pct > 0 else "      -"
        lines.append(
            f"{pin} {r.ticker:<7} {r.category:<10} {adv_s} {vol_s}  {r.label}"
        )
    lines.append("* = us.py canonical list")
    return "\n".join(lines)
