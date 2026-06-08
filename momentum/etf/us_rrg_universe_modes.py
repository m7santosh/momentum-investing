"""US ETF RRG universe modes — both use canonical ``us.py`` list only."""

from __future__ import annotations

import argparse

import pandas as pd

from momentum.etf.universes import us_universe as us_core
from momentum.etf.us_liquid_rrg_config import (
    DEFAULT_MIN_ADV,
    DEFAULT_VOL_PERCENTILE,
    US_ETF_LIQUID_RRG_PERIOD,
    parse_categories,
)
from momentum.etf.us_liquid_screener import screen_us_etfs
from momentum.etf import us_rrg_universe as rrg
from momentum.rrg_app import RrgAppConfig
from momentum.rrg_swing_cheat_sheet import ETF_SWING_CHEAT_SHEET
from utils.yahoo_weekly import load_yahoo_histories

US_UNIVERSE_CORE = "core"
US_UNIVERSE_EXPANDED = "expanded"

def _us_universe_label() -> str:
    return f"us.py ({len(us_core.TICKERS)} ETFs)"


US_UNIVERSE_LABEL: str = _us_universe_label()

# Single list in us.py — no core vs expanded split (expanded kept as CLI/backtest alias).
US_UNIVERSE_CHOICES: tuple[tuple[str, str], ...] = (
    (US_UNIVERSE_CORE, US_UNIVERSE_LABEL),
)

US_UNIVERSE_SWITCHABLE: bool = len(US_UNIVERSE_CHOICES) > 1

US_UNIVERSE_LABELS: dict[str, str] = dict(US_UNIVERSE_CHOICES)
US_UNIVERSE_LABEL_TO_KEY: dict[str, str] = {label: key for key, label in US_UNIVERSE_CHOICES}
US_UNIVERSE_DROPDOWN_VALUES: tuple[str, ...] = tuple(US_UNIVERSE_LABELS.values())


def normalize_us_universe_mode(mode: str) -> str:
    raw = (mode or US_UNIVERSE_CORE).strip().lower()
    if raw in (US_UNIVERSE_CORE, US_UNIVERSE_EXPANDED):
        return raw
    if raw in US_UNIVERSE_LABEL_TO_KEY:
        return US_UNIVERSE_LABEL_TO_KEY[raw]
    return US_UNIVERSE_CORE


def _resolve_etf_ticker(requested: str) -> str | None:
    text = requested.strip().upper()
    if not text:
        return None
    if text in rrg.RRG_ETF_ROW_IDS:
        return text
    for sym, label in rrg.RRG_ETF_LABELS.items():
        if text == sym.upper() or requested.strip().lower() == label.lower():
            return sym
    return None


def _resolve_row_id(requested: str) -> str | None:
    text = requested.strip()
    if not text:
        return None
    if text in rrg.RRG_ROW_BY_ID:
        return text
    for row_id, row in rrg.RRG_ROW_BY_ID.items():
        if text.lower() == row.label.lower():
            return row_id
    return _resolve_etf_ticker(text)


def build_us_core_rrg_config(
    period: str,
    rrg_window: int,
    *,
    us_universe_switchable: bool | None = None,
) -> RrgAppConfig:
    us_core.ensure_loaded()
    if us_universe_switchable is None:
        us_universe_switchable = US_UNIVERSE_SWITCHABLE

    def load_all_histories(
        hist_period: str, min_weekly_points: int, hist_rrg_window: int, freq: str = "week"
    ) -> dict:
        print(f"Loading US ETF EOD (Yahoo Finance) for RRG ({freq})...")
        batch = load_yahoo_histories(
            rrg.RRG_LOAD_YAHOO_TICKERS,
            period=hist_period,
            min_points=min_weekly_points,
            rrg_window=hist_rrg_window,
            freq=freq,
        )
        out = {}
        for ticker in rrg.RRG_ETF_ROW_IDS:
            out[ticker] = batch.get(ticker, pd.Series(dtype=float))
        out[rrg.RRG_BENCHMARK_YAHOO] = batch.get(rrg.RRG_BENCHMARK_YAHOO, pd.Series(dtype=float))
        return out

    def load_row_history(
        row_id: str,
        kind: str,
        hist_period: str,
        min_weekly_points: int,
        hist_rrg_window: int,
        freq: str = "week",
    ):
        return load_yahoo_histories(
            [row_id],
            period=hist_period,
            min_points=min_weekly_points,
            rrg_window=hist_rrg_window,
            freq=freq,
        ).get(row_id, pd.Series(dtype=float))

    def count_summary(kind_list: list[str]) -> str:
        n_etf = sum(1 for k in kind_list if k == "etf")
        return f"{n_etf} US ETFs"

    return RrgAppConfig(
        window_title=f"RRG — US ETFs vs S&P 500 ({period} Yahoo)",
        benchmark_nse=rrg.RRG_BENCHMARK_YAHOO,
        rows=rrg.RRG_ROWS,
        row_by_id=rrg.RRG_ROW_BY_ID,
        default_visible_ids=rrg.RRG_DEFAULT_VISIBLE_IDS,
        ref_column_header="Ticker",
        name_column_header="Name",
        defaults_checkbox_text="Default ETFs",
        hover_ref_prefix="ticker",
        universe_summary=(
            f"RRG universe: {len(rrg.RRG_ETF_ROW_IDS)} US ETFs "
            f"(benchmark {rrg.RRG_BENCHMARK_YAHOO}; edit momentum/etf/universes/us.py)"
        ),
        row_ref_label=rrg.row_ref_label,
        row_display_label=rrg.row_display_label,
        row_kind=rrg.row_kind,
        resolve_row_id=_resolve_row_id,
        load_all_histories=load_all_histories,
        load_row_history=load_row_history,
        count_summary=count_summary,
        analysis_period=period,
        rrg_window=rrg_window,
        default_tail=1,
        top_movers_panel=True,
        top_movers_count=10,
        top_movers_title="Top 10 — Was vs Now",
        side_cheat_sheet=ETF_SWING_CHEAT_SHEET,
        etf_table_extras=True,
        preview_today_picks=True,
        etf_recommend_profile="us",
        pick_strategy="leading_improved",
        etf_recommend_count=7,
        backtest_enabled=True,
        backtest_profile="us",
        backtest_universe_mode=US_UNIVERSE_CORE,
        us_universe_switchable=us_universe_switchable,
    )


def screen_us_expanded_universe(
    *,
    min_adv: float = DEFAULT_MIN_ADV,
    vol_percentile: float = DEFAULT_VOL_PERCENTILE,
    categories: tuple[str, ...] | list[str] = ("all",),
    adv_days: int = 20,
    vol_days: int = 63,
) -> list:
    args = argparse.Namespace(
        min_adv=min_adv,
        vol_percentile=vol_percentile,
        categories=list(categories),
        adv_days=adv_days,
        vol_days=vol_days,
    )
    args.categories = parse_categories(args.categories)
    return screen_us_etfs(
        categories=args.categories,
        min_adv_usd=args.min_adv,
        vol_percentile=args.vol_percentile,
        adv_days=args.adv_days,
        vol_days=args.vol_days,
    )


def build_us_expanded_rrg_config(
    *,
    period: str = US_ETF_LIQUID_RRG_PERIOD,
    rrg_window: int,
    min_adv: float = DEFAULT_MIN_ADV,
    vol_percentile: float = DEFAULT_VOL_PERCENTILE,
    categories: tuple[str, ...] | list[str] = ("all",),
    screened: list | None = None,
    us_universe_switchable: bool | None = None,
) -> RrgAppConfig:
    """Alias of core — same us.py list (``expanded`` kept for CLI compat)."""
    if screened is None:
        screened = screen_us_expanded_universe(
            min_adv=min_adv,
            vol_percentile=vol_percentile,
            categories=categories,
        )
    if not screened:
        raise RuntimeError(
            "Empty US ETF universe — check Yahoo data for us.py tickers."
        )
    from dataclasses import replace

    cfg = build_us_core_rrg_config(
        period, rrg_window, us_universe_switchable=us_universe_switchable
    )
    return replace(
        cfg,
        backtest_universe_mode=US_UNIVERSE_EXPANDED,
        backtest_min_adv=min_adv,
        backtest_vol_percentile=vol_percentile,
        backtest_categories=tuple(
            categories if isinstance(categories, (list, tuple)) else [categories]
        ),
    )


def build_us_rrg_config(
    mode: str,
    *,
    period: str,
    rrg_window: int,
    min_adv: float = DEFAULT_MIN_ADV,
    vol_percentile: float = DEFAULT_VOL_PERCENTILE,
    categories: tuple[str, ...] | list[str] = ("all",),
    us_universe_switchable: bool | None = None,
) -> RrgAppConfig:
    """Build RRG app config — always the canonical us.py ETF list."""
    us_core.ensure_loaded()
    key = normalize_us_universe_mode(mode)
    if key == US_UNIVERSE_EXPANDED:
        return build_us_expanded_rrg_config(
            period=period,
            rrg_window=rrg_window,
            min_adv=min_adv,
            vol_percentile=vol_percentile,
            categories=categories,
            us_universe_switchable=us_universe_switchable,
        )
    return build_us_core_rrg_config(
        period, rrg_window, us_universe_switchable=us_universe_switchable
    )
