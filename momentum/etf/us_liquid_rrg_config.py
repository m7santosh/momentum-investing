"""RRG app config and screening helpers for liquid US ETF universe."""

from __future__ import annotations

import argparse

import pandas as pd

from momentum.etf.universes.us_liquid_candidates import ALL_CATEGORIES
from momentum.etf.universes import us_liquid_candidates as pool
from momentum.etf.us_liquid_screener import screen_us_etfs
from momentum.rrg_app import RrgAppConfig
from momentum.rrg_swing_cheat_sheet import ETF_SWING_CHEAT_SHEET
from utils.yahoo_weekly import load_yahoo_histories

DEFAULT_MIN_ADV = 10_000_000
DEFAULT_VOL_PERCENTILE = 100.0  # 100 = ADV$ only; lower = optional calmer subset
US_ETF_LIQUID_RRG_PERIOD = "3m"


def parse_categories(raw: list[str] | set[str]) -> set[str]:
    items = list(raw)
    if "all" in items:
        return set(ALL_CATEGORIES) | {"all"}
    return set(items)


def run_screen(args: argparse.Namespace):
    cats = args.categories
    if not isinstance(cats, set):
        cats = parse_categories(cats)
    return screen_us_etfs(
        categories=cats,
        min_adv_usd=args.min_adv,
        vol_percentile=args.vol_percentile,
        adv_days=args.adv_days,
        vol_days=args.vol_days,
    )


def build_liquid_rrg_config(
    universe: dict,
    screened_count: int,
    *,
    period: str,
    rrg_window: int,
    min_adv: float,
    vol_percentile: float,
    categories: list[str] | set[str],
) -> RrgAppConfig:
    rows = universe["rows"]
    row_by_id = universe["row_by_id"]
    row_ids = universe["row_ids"]
    labels = universe["labels"]
    benchmark = universe["benchmark"]
    default_visible = universe["default_visible"]
    load_tickers = universe["load_tickers"]

    adv_m = min_adv / 1e6
    cat_list = list(categories)
    cat_text = "all" if "all" in cat_list else ",".join(cat_list)
    vol_note = (
        f"vol <= p{vol_percentile:.0f}"
        if vol_percentile < 100
        else "ADV$ discovery (vol filter off)"
    )
    summary = (
        f"RRG universe: {len(row_ids)} US ETFs from us.py "
        f"(benchmark {benchmark}; ADV metrics: >= ${adv_m:.1f}M, {vol_note})"
    )

    def row_ref_label(row_id: str) -> str:
        row = row_by_id.get(row_id)
        return row.ref_etf if row else ""

    def row_display_label(row_id: str) -> str:
        return labels.get(row_id, row_id)

    def row_kind(row_id: str) -> str:
        row = row_by_id.get(row_id)
        return row.kind if row else "etf"

    def resolve_row_id(requested: str) -> str | None:
        text = requested.strip()
        if not text:
            return None
        if text in row_by_id:
            return text
        upper = text.upper()
        for row_id, row in row_by_id.items():
            if upper == row_id.upper() or text.lower() == row.label.lower():
                return row_id
        return None

    def load_all_histories(
        hist_period: str, min_weekly_points: int, hist_rrg_window: int, freq: str = "week"
    ) -> dict[str, pd.Series]:
        print(f"Loading screened US ETF EOD (Yahoo Finance) for RRG ({freq})...")
        batch = load_yahoo_histories(
            load_tickers,
            period=hist_period,
            min_points=min_weekly_points,
            rrg_window=hist_rrg_window,
            freq=freq,
        )
        out: dict[str, pd.Series] = {}
        for ticker in row_ids:
            out[ticker] = batch.get(ticker, pd.Series(dtype=float))
        out[benchmark] = batch.get(benchmark, pd.Series(dtype=float))
        return out

    def load_row_history(
        row_id: str,
        kind: str,
        hist_period: str,
        min_weekly_points: int,
        hist_rrg_window: int,
        freq: str = "week",
    ) -> pd.Series:
        return load_yahoo_histories(
            [row_id],
            period=hist_period,
            min_points=min_weekly_points,
            rrg_window=hist_rrg_window,
            freq=freq,
        ).get(row_id, pd.Series(dtype=float))

    def count_summary(kind_list: list[str]) -> str:
        n_etf = sum(1 for k in kind_list if k == "etf")
        return f"{n_etf} US ETFs (us.py)"

    return RrgAppConfig(
        window_title=(
            f"RRG — US ETFs (3m · {rrg_window}w · us.py · {len(row_ids)} ETFs)"
        ),
        benchmark_nse=benchmark,
        rows=rows,
        row_by_id=row_by_id,
        default_visible_ids=default_visible,
        ref_column_header="Ticker",
        name_column_header="Name",
        defaults_checkbox_text="Default ETFs",
        hover_ref_prefix="ticker",
        universe_summary=summary,
        row_ref_label=row_ref_label,
        row_display_label=row_display_label,
        row_kind=row_kind,
        resolve_row_id=resolve_row_id,
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
        backtest_universe_mode="core",
        backtest_min_adv=min_adv,
        backtest_vol_percentile=vol_percentile,
        backtest_categories=tuple(
            categories if isinstance(categories, (list, tuple)) else [categories]
        ),
    )
