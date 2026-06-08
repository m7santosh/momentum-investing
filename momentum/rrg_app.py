"""Parameterized RRG (Relative Rotation Graph) Tkinter application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import tkinter as tk
import tkinter.font as tkfont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch
from tkinter import messagebox, ttk

from momentum.rrg_busy import RrgBusyOverlay
from momentum.rrg_portfolio_panel import (
    PORTFOLIO_PANEL_NUM_COLS,
    PORTFOLIO_PANEL_PICK_GRID_ROW,
    PORTFOLIO_PANEL_PICK_ROW,
    PORTFOLIO_PANEL_REBAL_COL,
    PORTFOLIO_PANEL_SMALL_KEYS,
    PORTFOLIO_PANEL_TAG_KEYS,
    PORTFOLIO_PANEL_WAS_COL,
    PORTFOLIO_PANEL_WAS_ROW,
    configure_portfolio_panel_table_columns,
    build_portfolio_panel,
    format_portfolio_cell,
    live_close_for_panel,
    norm_ticker as _norm_ticker,
    pad_rebal_slots,
    portfolio_panel_dates_line,
    portfolio_panel_pick_header,
    portfolio_panel_totals_line,
    portfolio_panel_was_header,
)
from momentum.rrg_ranking import (
    format_change_pct,
    format_rank_delta,
    rank_by_tail_change,
    ranked_row_indices,
    series_at,
    tail_change_pct,
)
from momentum.rrg_ui_copy import TableRegionCopy, install_copy_support
from momentum.rrg_core import (
    HEAD_ARROW_SCALE,
    HOVER_PIXEL_RADIUS,
    RRG_DEFAULT_TAIL,
    RRG_MAX_TAIL,
    RRG_WINDOW_DEFAULT,
    TAIL_MARKER_SIZE,
    compute_rrg_indicators as compute,
    get_color,
    get_chart_color,
    get_status,
    rrg_row_fg_color,
    RRG_CHART_COLOR_IMPROVING,
    RRG_CHART_COLOR_LAGGING,
    RRG_CHART_COLOR_LEADING,
    RRG_CHART_COLOR_WEAKENING,
    RRG_COLOR_NA,
    rrg_effective_window,
    rrg_min_history_bars,
    rrg_nav_bars,
    rrg_normalize_bar_unit,
    rrg_format_date,
    rrg_period_display,
    rrg_period_label,
    rrg_slider_index_bars,
    rrg_warmup_bars,
    rrg_warmup_weeks,
)


@dataclass
class RrgAppConfig:
    """Configuration and callbacks for a single RRG app instance."""

    window_title: str
    benchmark_nse: str
    rows: list
    row_by_id: dict[str, object]
    default_visible_ids: set[str]
    ref_column_header: str
    name_column_header: str
    defaults_checkbox_text: str
    hover_ref_prefix: str
    universe_summary: str
    row_ref_label: Callable[[str], str]
    row_display_label: Callable[[str], str]
    row_kind: Callable[[str], str]
    resolve_row_id: Callable[[str], str | None]
    load_all_histories: Callable[..., dict[str, pd.Series]]
    load_row_history: Callable[..., pd.Series]
    count_summary: Callable[[list[str]], str]
    analysis_period: str = "6m"
    rrg_window: int = RRG_WINDOW_DEFAULT
    top_movers_panel: bool = False
    top_movers_count: int = 7
    top_movers_kind: str | None = None
    top_movers_title: str = "Top movers"
    default_tail: int = RRG_DEFAULT_TAIL
    side_cheat_sheet_title: str = "Swing trading cheat sheet"
    side_cheat_sheet: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    etf_table_extras: bool = False
    preview_today_picks: bool = True  # Week unit: daily preview checkbox (US/India ETF/stock)
    etf_recommend_profile: str = "us"  # "us" | "india" | "stock"
    etf_recommend_count: int = 7
    etf_recommend_title: str = "Recommended Top 7 (weekly swing)"
    pick_strategy: str = "leading_improved"
    hold_until_rank_exit: bool = False
    max_hold_rank: int = 10
    exit_below_9ema: bool = True
    backtest_enabled: bool = False
    backtest_profile: str = "india"  # "india" | "us" | "stock"
    backtest_universe_key: str = "quality"
    backtest_universe_mode: str = "core"  # US: "core" | "expanded" (both use us.py)
    backtest_min_adv: float = 10_000_000.0
    backtest_vol_percentile: float = 100.0
    backtest_categories: tuple[str, ...] = ("all",)
    us_universe_switchable: bool = False


_PORTFOLIO_N_MAX = 25


def run_rrg_app(config: RrgAppConfig) -> None:
    """Build UI, load data, and run the RRG main loop."""
    use_right_extras = config.etf_table_extras and config.top_movers_panel
    period = config.analysis_period
    window = config.rrg_window
    bar_unit = "week"
    nav_bars = rrg_nav_bars(period, bar_unit)
    effective_window = rrg_effective_window(window, bar_unit)
    min_history_bars = rrg_min_history_bars(window, bar_unit)
    tail = config.default_tail
    end_date_idx = None
    start_date, end_date = None, None
    hover_points = []
    _last_hover_idx = None
    _history_cache: dict[str, dict[str, pd.Series]] = {}
    _daily_pick_rrg_cache: tuple | None = None
    indices_data = pd.DataFrame()
    benchmark_data = pd.Series(dtype=float)

    requested_indices = [row.row_id for row in config.rows]
    indices = requested_indices.copy()
    index_metadata: dict[str, list] = {'ref_label': [], 'display': [], 'kind': []}

    for row in config.rows:
        row_id = row.row_id
        index_metadata['ref_label'].append(config.row_ref_label(row_id))
        index_metadata['display'].append(config.row_display_label(row_id))
        index_metadata['kind'].append(config.row_kind(row_id))

    _use_default_indices_on_load = False
    indices_to_show = (
        [n for n in indices if n in config.default_visible_ids]
        if _use_default_indices_on_load
        else indices.copy()
    )

    root = tk.Tk()
    root.withdraw()
    root.title(config.window_title)
    root.geometry(
        '1600x900'
        if use_right_extras
        else ('1400x900' if config.top_movers_panel else '1100x900')
    )
    root.minsize(
        1280 if use_right_extras else (1280 if config.top_movers_panel else 900),
        650,
    )
    root.resizable(True, True)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    _busy = RrgBusyOverlay(root)
    root.deiconify()
    root.update_idletasks()

    def _histories_for_unit(unit: str) -> dict[str, pd.Series]:
        u = rrg_normalize_bar_unit(unit)
        if u not in _history_cache:
            unit_lbl = "daily" if u == "day" else "weekly"
            min_pts = rrg_min_history_bars(window, u)

            def _load() -> dict[str, pd.Series]:
                return config.load_all_histories(period, min_pts, window, freq=u)

            _history_cache[u] = _busy.run_threaded(
                _load, f"Downloading {unit_lbl} price data…"
            )
        return _history_cache[u]

    def _day_unit_slider_calendar() -> pd.DatetimeIndex:
        """Union of all daily bar dates (benchmark, universe, CM/Yahoo)."""
        dates: set[pd.Timestamp] = set()
        bench = benchmark_data.dropna().sort_index()
        if len(bench):
            dates.update(pd.DatetimeIndex(bench.index))
        for col in indices:
            if col in indices_data.columns:
                series = indices_data[col].dropna().sort_index()
                if len(series):
                    dates.update(pd.DatetimeIndex(series.index))
        for series in _etf_daily_close.values():
            if series is not None and len(series):
                dates.update(pd.DatetimeIndex(series.dropna().index))
        if not dates:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(sorted(dates))

    def _build_rrg_date_index():
        """Bar dates for the Date slider (analysis window + tail buffer)."""
        from momentum.rrg_core import rrg_build_slider_date_index

        daily_sources = None
        if bar_unit == "day":
            cal = _day_unit_slider_calendar()
            if len(cal):
                daily_sources = [pd.Series(0.0, index=cal)]
            else:
                daily_sources = [
                    indices_data[col].dropna().sort_index()
                    for col in indices
                    if col in indices_data.columns and indices_data[col].notna().any()
                ]
        return rrg_build_slider_date_index(
            benchmark_data,
            analysis_period=period,
            window=window,
            unit=bar_unit,
            daily_sources=daily_sources,
        )

    def _date_slider_max_idx() -> int:
        """Latest bar on the index (most recent EOD / today)."""
        return len(_rrg_index) - 1

    def _date_slider_min_idx() -> int:
        """Earliest end date: ``nav_bars`` back from latest (still room for tail)."""
        return max(tail, len(_rrg_index) - nav_bars)

    def _date_range_hint_text() -> str:
        if len(_rrg_index) < 2:
            return ''
        return (
            f"{format_date_label(_date_slider_min_idx())} … "
            f"{format_date_label(_date_slider_max_idx())}"
        )

    print(config.universe_summary)
    print(
        f"RRG analysis: {rrg_period_label(period, bar_unit)}, rolling window "
        f"{effective_window} bars ({window}w equivalent) "
        f"(warmup ~{rrg_warmup_bars(window, bar_unit)} bars before slider)."
    )

    def _apply_price_histories(histories: dict[str, pd.Series]) -> None:
        nonlocal indices_data, benchmark_data, indices, indices_to_show
        nonlocal effective_window, min_history_bars, nav_bars

        effective_window = rrg_effective_window(window, bar_unit)
        min_history_bars = rrg_min_history_bars(window, bar_unit)
        nav_bars = rrg_nav_bars(period, bar_unit)
        benchmark_data = histories.get(config.benchmark_nse, pd.Series(dtype=float))
        indices_data = pd.DataFrame(
            {
                row_id: histories.get(row_id, pd.Series(dtype=float))
                for row_id in requested_indices
            }
        )
        available_indices = [
            n
            for n in requested_indices
            if n in indices_data.columns
            and indices_data[n].notna().sum() > effective_window
        ]
        missing = set(requested_indices) - set(available_indices)
        if missing:
            print(f"Skipping rows with insufficient data: {sorted(missing)}")
        indices[:] = available_indices
        indices_to_show[:] = [n for n in indices_to_show if n in indices]
        index_metadata['ref_label'] = [config.row_ref_label(name) for name in indices]
        index_metadata['display'] = [config.row_display_label(name) for name in indices]
        index_metadata['kind'] = [config.row_kind(name) for name in indices]

    _apply_price_histories(_histories_for_unit(bar_unit))

    rs_tickers = []
    rsr_tickers = []
    rsr_roc_tickers = []
    rsm_tickers = []

    for i in range(len(indices)):
        rsr, rsr_roc, rsm = compute(
            indices_data[indices[i]], benchmark_data, effective_window
        )
        if rsr is None:
            continue
        rs_tickers.append(100 * (indices_data[indices[i]] / benchmark_data))
        rsr_tickers.append(rsr)
        rsr_roc_tickers.append(rsr_roc)
        rsm_tickers.append(rsm)

    indices[:] = indices[: len(rsr_tickers)]
    indices_to_show = [n for n in indices_to_show if n in indices]
    index_metadata['ref_label'] = index_metadata['ref_label'][: len(indices)]
    index_metadata['display'] = index_metadata['display'][: len(indices)]
    index_metadata['kind'] = index_metadata['kind'][: len(indices)]

    if not rsr_tickers:
        raise SystemExit(
            "No RRG rows with enough price history. Check data downloads."
        )

    _rrg_index = _build_rrg_date_index()
    _last_nse = pd.Timestamp(_rrg_index[-1]).date() if len(_rrg_index) else None
    _slider_first = pd.Timestamp(_rrg_index[0]).date() if len(_rrg_index) else None
    print(
        f"RRG ready: {rrg_period_display(period)} — date ends "
        f"{pd.Timestamp(_rrg_index[_date_slider_min_idx()]).date() if len(_rrg_index) else 'n/a'}"
        f" .. {_last_nse} ({nav_bars} {bar_unit} bars, default latest), "
        f"{config.count_summary(index_metadata['kind'])}, "
        f"benchmark {config.benchmark_nse}"
    )
    _bench_bars = len(benchmark_data.dropna())
    print(
        f"  Downloaded {_bench_bars} benchmark {bar_unit} bars total "
        f"(includes ~{rrg_warmup_bars(window, bar_unit)} bar warmup before "
        f"slider start {_slider_first})."
    )

    etf_vol_by_row: dict[int, float] = {}
    if config.etf_table_extras and indices:
        def _load_vol() -> dict[int, float]:
            vol: dict[int, float] = {}
            if config.etf_recommend_profile == "india":
                print("Loading 63-day Vol% for India ETF table...")
                from momentum.etf.india_rrg_recommendations import (
                    load_india_etf_vol_pct,
                )

                vol_by_ref = load_india_etf_vol_pct(
                    indices,
                    index_metadata["ref_label"],
                    vol_days=63,
                    history_days=120,
                )
                for j in range(len(indices)):
                    ref = (
                        index_metadata["ref_label"][j] or indices[j]
                    ).upper().replace(".NS", "")
                    vol[j] = vol_by_ref.get(ref, 0.0)
            elif config.etf_recommend_profile == "stock":
                print("Loading 63-day Vol% for stock table...")
                from momentum.stock.stock_rrg_recommendations import (
                    load_stock_vol_pct,
                )

                vol_by_ref = load_stock_vol_pct(
                    indices, vol_days=63, history_days=120
                )
                for j in range(len(indices)):
                    sym = indices[j].upper().replace(".NS", "")
                    vol[j] = vol_by_ref.get(sym, 0.0)
            else:
                print("Loading 63-day Vol% for US ETF table...")
                from momentum.etf.us_liquid_screener import _fetch_metrics

                vol_metrics = _fetch_metrics(
                    list(indices),
                    adv_days=20,
                    vol_days=63,
                    history_days=120,
                    quiet=True,
                )
                for j, sym in enumerate(indices):
                    if sym in vol_metrics:
                        vol[j] = vol_metrics[sym][1]
            return vol

        try:
            etf_vol_by_row.update(
                _busy.run_threaded(_load_vol, "Loading volatility data…")
            )
        except Exception as exc:
            print(f"Vol% load skipped: {exc}")

    _etf_daily_close: dict[str, pd.Series] = {}

    def _load_etf_daily_close_data(*, force: bool = False) -> None:
        """CM/Yahoo daily closes for ETF rows (9 EMA, exit P&L, portfolio panel)."""
        if _etf_daily_close and not force:
            return
        if force:
            _etf_daily_close.clear()
        profile_msgs = {
            "india": "Downloading NSE ETF daily EOD…",
            "stock": "Downloading NSE stock daily EOD…",
            "us": "Downloading US ETF daily EOD (Yahoo)…",
        }
        msg = profile_msgs.get(
            config.etf_recommend_profile, "Downloading daily EOD…"
        )

        def _download_daily() -> dict[str, pd.Series]:
            from datetime import timedelta

            from utils.nse_bhavcopy import today_ist

            out: dict[str, pd.Series] = {}
            end_d = today_ist()
            start_d = (_rrg_index[0].date() if len(_rrg_index) else end_d) - timedelta(
                days=400
            )
            if config.etf_recommend_profile == "india":
                from utils.nse_bhavcopy import load_nse_cm_histories_range

                syms = {
                    (index_metadata["ref_label"][j] or indices[j])
                    .strip()
                    .upper()
                    .replace(".NS", "")
                    for j in range(len(indices))
                }
                syms.discard("")
                print(
                    f"Loading NSE ETF daily CM (bhavcopy) for "
                    f"{len(syms)} symbol(s)..."
                )
                batch = load_nse_cm_histories_range(
                    sorted(syms),
                    start_d,
                    end_d,
                    min_points=5,
                    quiet=True,
                    asset_label="ETF symbol",
                    freq="day",
                )
            elif config.etf_recommend_profile == "stock":
                from utils.nse_bhavcopy import load_nse_cm_histories_range

                syms = {
                    indices[j].strip().upper().replace(".NS", "")
                    for j in range(len(indices))
                }
                syms.discard("")
                print(
                    f"Loading NSE stock daily CM (bhavcopy) for "
                    f"{len(syms)} symbol(s)..."
                )
                batch = load_nse_cm_histories_range(
                    sorted(syms),
                    start_d,
                    end_d,
                    min_points=5,
                    quiet=True,
                    asset_label="equity symbol",
                    freq="day",
                )
            else:
                from utils.yahoo_weekly import load_yahoo_histories_range

                tickers = list(dict.fromkeys([*indices, config.benchmark_nse]))
                print(
                    f"Loading Yahoo daily for {len(tickers)} US ticker(s) "
                    f"(incl. benchmark {config.benchmark_nse})..."
                )
                batch = load_yahoo_histories_range(
                    tickers,
                    start_d,
                    end_d,
                    min_points=5,
                    quiet=True,
                    freq="day",
                )
            for sym, series in batch.items():
                if len(series):
                    out[sym] = series.sort_index()
            return out

        loaded = _busy.run_threaded(_download_daily, msg)
        _etf_daily_close.update(loaded)

    if config.etf_table_extras or config.exit_below_9ema:
        _load_etf_daily_close_data()

    def update_rrg():
        """Recompute RSR/RSM for every row (same length as ``indices``)."""
        nonlocal rs_tickers, rsr_tickers, rsr_roc_tickers, rsm_tickers
        for i in range(len(indices)):
            name = indices[i]
            rsr, rsr_roc, rsm = compute(
                indices_data[name], benchmark_data, effective_window
            )
            if rsr is None:
                continue
            rs_tickers[i] = 100 * (indices_data[name] / benchmark_data)
            rsr_tickers[i] = rsr
            rsr_roc_tickers[i] = rsr_roc
            rsm_tickers[i] = rsm

    main_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
    main_pane.grid(row=0, column=0, sticky='nsew')

    bottom_frame = tk.Frame(main_pane)
    bottom_frame.columnconfigure(0, weight=1)
    bottom_frame.rowconfigure(1, weight=1)
    main_pane.add(bottom_frame, weight=1)

    chart_frame = tk.Frame(main_pane)
    chart_frame.rowconfigure(0, weight=1)
    chart_frame.columnconfigure(0, weight=1)

    fig, ax_rrg = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    ax_rrg.set_title('RRG Indicator')
    ax_rrg.set_xlabel('JdK RS Ratio')
    ax_rrg.set_ylabel('JdK RS Momentum')
    ax_rrg.axhline(y=100, color='k', linestyle='--')
    ax_rrg.axvline(x=100, color='k', linestyle='--')
    ax_rrg.fill_between([94, 100], [94, 94], [100, 100], color=RRG_CHART_COLOR_LAGGING, alpha=0.35)
    ax_rrg.fill_between([100, 106], [94, 94], [100, 100], color=RRG_CHART_COLOR_WEAKENING, alpha=0.35)
    ax_rrg.fill_between([100, 106], [100, 100], [106, 106], color=RRG_CHART_COLOR_LEADING, alpha=0.35)
    ax_rrg.fill_between([94, 100], [100, 100], [106, 106], color=RRG_CHART_COLOR_IMPROVING, alpha=0.35)
    ax_rrg.text(95, 105, 'Improving')
    ax_rrg.text(104, 105, 'Leading')
    ax_rrg.text(104, 95, 'Weakening')
    ax_rrg.text(95, 95, 'Lagging')
    ax_rrg.set_xlim(94, 106)
    ax_rrg.set_ylim(94, 106)
    ax_rrg.set_xticks(range(94, 107))
    ax_rrg.set_yticks(range(94, 107))
    ax_rrg.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax_rrg.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax_rrg.tick_params(labelsize=8)
    ax_rrg.grid(
        True,
        which='major',
        color='#555555',
        linestyle='-',
        linewidth=0.5,
        alpha=0.45,
        zorder=2,
    )
    ax_rrg.grid(
        True,
        which='minor',
        color='#777777',
        linestyle=':',
        linewidth=0.35,
        alpha=0.3,
        zorder=2,
    )

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky='nsew')

    _hover_annot = ax_rrg.annotate(
        '',
        xy=(0, 0),
        xytext=(12, 12),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='gray', alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9,
        visible=False,
        zorder=100,
    )

    def _hide_hover_tooltip():
        nonlocal _last_hover_idx
        if _hover_annot.get_visible():
            _hover_annot.set_visible(False)
            _last_hover_idx = None

    def _format_hover_text(point):
        title = point['index']
        if point.get('ref_label'):
            title = f"{point['index']} ({config.hover_ref_prefix}: {point['ref_label']})"
        lines = [
            title,
            f"Date: {point['date']}",
            f"RS Ratio: {point['rsr']:.2f}  |  RS Momentum: {point['rsm']:.2f}",
            f"Quadrant: {point['status']}",
            f"Price: {point['price']:.2f}",
        ]
        if point['wow_chg'] is not None:
            lines.append(f"WoW: {point['wow_chg']:+.1f}%")
        if point['is_current']:
            lines.append('(current week)')
        return '\n'.join(lines)

    def _append_hover_points(j, filtered_rsr, filtered_rsm):
        row_id = indices[j]
        ref_label = index_metadata['ref_label'][j]
        display = index_metadata['display'][j]
        prices = indices_data[row_id]
        n = len(filtered_rsr)
        for k in range(n):
            date = filtered_rsr.index[k]
            rsr_val = float(filtered_rsr.iloc[k])
            rsm_val = float(filtered_rsm.iloc[k])
            price = float(prices.loc[date])
            wow_chg = None
            if k > 0:
                prev_date = filtered_rsr.index[k - 1]
                prev_price = float(prices.loc[prev_date])
                wow_chg = (price - prev_price) / prev_price * 100
            status = get_status(rsr_val, rsm_val)
            hover_points.append(
                {
                    'x': rsr_val,
                    'y': rsm_val,
                    'index': display,
                    'ref_label': ref_label,
                    'date': rrg_format_date(date),
                    'rsr': rsr_val,
                    'rsm': rsm_val,
                    'status': status.capitalize() if status else '',
                    'price': price,
                    'wow_chg': wow_chg,
                    'is_current': k == n - 1,
                }
            )

    def on_mouse_move(event):
        nonlocal _last_hover_idx

        if event.inaxes != ax_rrg or not hover_points:
            if _hover_annot.get_visible():
                _hide_hover_tooltip()
                canvas.draw_idle()
            return

        best_i = None
        best_dist = HOVER_PIXEL_RADIUS ** 2
        for i, pt in enumerate(hover_points):
            px, py = ax_rrg.transData.transform((pt['x'], pt['y']))
            dist = (px - event.x) ** 2 + (py - event.y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_i = i

        if best_i is None:
            if _hover_annot.get_visible():
                _hide_hover_tooltip()
                canvas.draw_idle()
            return

        if best_i == _last_hover_idx and _hover_annot.get_visible():
            return

        pt = hover_points[best_i]
        _hover_annot.xy = (pt['x'], pt['y'])
        _hover_annot.set_text(_format_hover_text(pt))
        _hover_annot.set_visible(True)
        _last_hover_idx = best_i
        canvas.draw_idle()

    canvas.mpl_connect('motion_notify_event', on_mouse_move)

    controls_frame = tk.Frame(bottom_frame, height=104, padx=8, pady=6)
    controls_frame.grid(row=0, column=0, sticky='ew')
    controls_frame.grid_propagate(False)

    show_rrg_var = tk.BooleanVar(value=False)
    bar_unit_var = tk.StringVar(value="Week")

    _rrg_chart_sash_pos = 420

    def _refresh_rrg_chart_layout(_event=None):
        if show_rrg_var.get():
            canvas.draw_idle()

    def _save_rrg_chart_sash(_event=None):
        nonlocal _rrg_chart_sash_pos
        if not show_rrg_var.get():
            return
        if str(chart_frame) not in main_pane.panes():
            return
        try:
            pos = main_pane.sashpos(0)
            if pos > 80:
                _rrg_chart_sash_pos = pos
        except tk.TclError:
            pass
        _refresh_rrg_chart_layout()

    main_pane.bind('<ButtonRelease-1>', _save_rrg_chart_sash)
    chart_frame.bind('<Configure>', _refresh_rrg_chart_layout)

    def _apply_rrg_chart_visibility():
        nonlocal _rrg_chart_sash_pos
        if show_rrg_var.get():
            if str(chart_frame) not in main_pane.panes():
                main_pane.insert(0, chart_frame, weight=1)
            main_pane.update_idletasks()
            total_h = main_pane.winfo_height()
            if total_h > 1:
                sash = min(
                    max(_rrg_chart_sash_pos, 200),
                    max(total_h - 200, 200),
                )
                main_pane.sashpos(0, sash)
            else:
                main_pane.sashpos(0, _rrg_chart_sash_pos)
            root.after_idle(_refresh_rrg_chart_layout)
        else:
            if str(chart_frame) in main_pane.panes():
                try:
                    pos = main_pane.sashpos(0)
                    if pos > 80:
                        _rrg_chart_sash_pos = pos
                except tk.TclError:
                    pass
                main_pane.forget(chart_frame)
            _hide_hover_tooltip()
        if _sync_side_scroll is not None:
            root.after_idle(_sync_side_scroll)

    def on_show_rrg_toggle():
        _apply_rrg_chart_visibility()
        if show_rrg_var.get():
            redraw_chart()

    default_indices_var = tk.BooleanVar(value=_use_default_indices_on_load)
    select_all_var = tk.BooleanVar(
        value=not _use_default_indices_on_load or len(indices_to_show) == len(indices)
    )
    _select_all_updating = False

    date_max_idx = _date_slider_max_idx()
    date_min_idx = _date_slider_min_idx()
    end_date_idx = date_max_idx
    start_date = _rrg_index[end_date_idx - tail]
    end_date = _rrg_index[end_date_idx]

    def format_date_label(idx):
        return rrg_format_date(_rrg_index[int(idx)])

    def _tail_marker_sizes(n_points: int, *, base: int | None = None) -> list[int]:
        size = base if base is not None else TAIL_MARKER_SIZE
        return [size] * n_points

    def _add_head_arrow(x_vals, y_vals, color: str):
        """Arrow on the last tail segment (direction of movement)."""
        if len(x_vals) < 2:
            return None
        arrow = FancyArrowPatch(
            (float(x_vals[-2]), float(y_vals[-2])),
            (float(x_vals[-1]), float(y_vals[-1])),
            arrowstyle='-|>',
            mutation_scale=HEAD_ARROW_SCALE,
            linewidth=1.8,
            color=color,
            zorder=5,
            shrinkA=0,
            shrinkB=0,
        )
        ax_rrg.add_patch(arrow)
        return arrow

    def on_tail_change(val):
        nonlocal tail, end_date_idx
        new_tail = int(float(val))
        if end_date_idx - new_tail < 0:
            tail_scale.set(tail)
            return
        tail = new_tail
        _clear_pick_cache()
        date_min = _date_slider_min_idx()
        date_max = _date_slider_max_idx()
        date_scale.config(from_=date_min, to=date_max)
        if end_date_idx < date_min:
            end_date_idx = date_min
            date_scale.set(end_date_idx)
        elif end_date_idx > date_max:
            end_date_idx = date_max
            date_scale.set(end_date_idx)
        date_value_label.config(text=format_date_label(end_date_idx))
        date_range_label.config(text=_date_range_hint_text())
        redraw_chart()

    def on_date_change(val):
        nonlocal end_date_idx
        end_date_idx = int(float(val))
        date_value_label.config(text=format_date_label(end_date_idx))
        redraw_chart()

    def _bar_unit_step_label() -> str:
        return "Day" if bar_unit == "day" else "Week"

    def update_nav_button_labels():
        step = _bar_unit_step_label()
        prev_week_button.config(text=f"Previous {step}")
        next_week_button.config(text=f"Next {step}")

    def update_week_step_buttons():
        current_idx = int(date_scale.get())
        date_min = _date_slider_min_idx()
        date_max = _date_slider_max_idx()
        if current_idx <= date_min:
            prev_week_button.state(['disabled'])
        else:
            prev_week_button.state(['!disabled'])
        if current_idx >= date_max:
            next_week_button.state(['disabled'])
        else:
            next_week_button.state(['!disabled'])

    def step_previous_week():
        nonlocal end_date_idx
        current_idx = int(date_scale.get())
        if current_idx <= _date_slider_min_idx():
            return
        end_date_idx = current_idx - 1
        date_scale.set(end_date_idx)
        date_value_label.config(text=format_date_label(end_date_idx))
        redraw_chart()

    def step_next_week():
        nonlocal end_date_idx
        current_idx = int(date_scale.get())
        if current_idx >= _date_slider_max_idx():
            return
        end_date_idx = current_idx + 1
        date_scale.set(end_date_idx)
        date_value_label.config(text=format_date_label(end_date_idx))
        redraw_chart()

    def step_to_latest():
        nonlocal end_date_idx
        end_date_idx = _date_slider_max_idx()
        date_scale.set(end_date_idx)
        date_value_label.config(text=format_date_label(end_date_idx))
        redraw_chart()

    week_nav_frame = tk.Frame(controls_frame)
    week_nav_frame.pack(side=tk.LEFT, padx=(0, 12), anchor='n')
    prev_week_button = ttk.Button(
        week_nav_frame, text='Previous Week', command=step_previous_week
    )
    prev_week_button.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
    next_week_button = ttk.Button(week_nav_frame, text='Next Week', command=step_next_week)
    next_week_button.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
    ttk.Button(week_nav_frame, text='Latest', command=step_to_latest).pack(
        side=tk.TOP, fill=tk.X
    )

    def _sync_select_all_checkbox():
        if not checkbox_vars:
            return
        select_all_var.set(all(checkbox_vars[i].get() for i in range(len(indices))))

    def apply_select_all(select_all: bool):
        nonlocal indices_to_show, _select_all_updating
        _select_all_updating = True
        default_indices_var.set(False)
        indices_to_show = indices.copy() if select_all else []
        for i in range(len(indices)):
            checkbox_vars[i].set(select_all)
        select_all_var.set(select_all)
        _select_all_updating = False
        redraw_chart()

    def on_select_all_toggle():
        if _select_all_updating:
            return
        apply_select_all(select_all_var.get())

    def apply_default_indices_visibility(use_defaults: bool):
        nonlocal indices_to_show
        if use_defaults:
            indices_to_show = [n for n in indices if n in config.default_visible_ids]
        else:
            indices_to_show = indices.copy()
        for i, index_name in enumerate(indices):
            checkbox_vars[i].set(index_name in indices_to_show)
        _sync_select_all_checkbox()
        redraw_chart()

    def on_default_indices_toggle():
        apply_default_indices_visibility(default_indices_var.get())

    default_indices_cb = ttk.Checkbutton(
        controls_frame,
        text=config.defaults_checkbox_text,
        variable=default_indices_var,
        command=on_default_indices_toggle,
    )
    default_indices_cb.pack(side=tk.LEFT, padx=(0, 12))

    show_rrg_cb = ttk.Checkbutton(
        controls_frame,
        text='Show RRG graph',
        variable=show_rrg_var,
        command=on_show_rrg_toggle,
    )
    show_rrg_cb.pack(side=tk.LEFT, padx=(0, 12))

    _pick_holdings_cache: dict[int, list[str]] = {}
    _active_holdings_cache: dict[int, list[str]] = {}
    _week_exits_cache: dict[int, list] = {}
    _mid_week_9ema_cache: dict[int, list] = {}
    _current_pick_row_indices: set[int] = set()
    pick_strategy_var = tk.StringVar()
    hold_until_rank_exit_var = tk.BooleanVar(value=config.hold_until_rank_exit)
    exit_below_9ema_var = tk.BooleanVar(value=config.exit_below_9ema)
    max_hold_rank_var = tk.IntVar(value=config.max_hold_rank)
    portfolio_n_var: tk.IntVar | None = tk.IntVar(value=config.etf_recommend_count)
    pick_auto_show_var = tk.BooleanVar(value=True)
    preview_today_picks_var = tk.BooleanVar(value=False)
    preview_today_cb = None
    _pick_label_to_key: dict[str, str] = {}

    def _portfolio_top_n() -> int:
        if portfolio_n_var is not None:
            try:
                return max(1, min(int(portfolio_n_var.get()), _PORTFOLIO_N_MAX))
            except (tk.TclError, ValueError):
                pass
        return config.etf_recommend_count

    _us_universe_label_to_key: dict[str, str] = {}
    us_universe_var = tk.StringVar()

    if config.us_universe_switchable:
        from momentum.etf.us_rrg_universe_modes import (
            US_UNIVERSE_DROPDOWN_VALUES,
            US_UNIVERSE_LABELS,
        )

        _us_universe_label_to_key = {
            label: key for key, label in US_UNIVERSE_LABELS.items()
        }
        us_universe_var.set(
            US_UNIVERSE_LABELS.get(
                config.backtest_universe_mode, US_UNIVERSE_DROPDOWN_VALUES[0]
            )
        )

    def _us_universe_mode_key() -> str:
        if not config.us_universe_switchable:
            return config.backtest_universe_mode
        return _us_universe_label_to_key.get(
            us_universe_var.get(), config.backtest_universe_mode
        )

    if config.us_universe_switchable:
        universe_row = tk.Frame(controls_frame)
        universe_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        tk.Label(universe_row, text="ETF universe", width=12, anchor="w").pack(
            side=tk.LEFT
        )
        us_universe_combo = ttk.Combobox(
            universe_row,
            textvariable=us_universe_var,
            values=list(US_UNIVERSE_DROPDOWN_VALUES),
            width=42,
            state="readonly",
        )
        us_universe_combo.pack(side=tk.LEFT, padx=(0, 12))
    else:
        us_universe_combo = None

    if config.etf_table_extras:
        if config.etf_recommend_profile == "us":
            from momentum.etf.us_rrg_pick_strategies import (
                PICK_STRATEGIES,
                pick_strategy_subtitle,
            )
        elif config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import (
                PICK_STRATEGIES,
                pick_strategy_subtitle,
            )
        else:
            from momentum.etf.india_rrg_pick_strategies import (
                PICK_STRATEGIES,
                pick_strategy_subtitle,
            )

        _pick_label_to_key = {label: key for key, label in PICK_STRATEGIES.items()}
        pick_strategy_var.set(
            PICK_STRATEGIES.get(
                config.pick_strategy, PICK_STRATEGIES["leading_improved"]
            )
        )

        def _pick_strategy_key() -> str:
            return _pick_label_to_key.get(
                pick_strategy_var.get(), config.pick_strategy
            )

        def _clear_pick_cache() -> None:
            _pick_holdings_cache.clear()
            _active_holdings_cache.clear()
            _week_exits_cache.clear()
            _mid_week_9ema_cache.clear()
            _current_pick_row_indices.clear()

        pick_row = tk.Frame(controls_frame)
        pick_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        tk.Label(pick_row, text="Pick strategy", width=12, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(
            pick_row,
            textvariable=pick_strategy_var,
            values=list(PICK_STRATEGIES.values()),
            width=42,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(0, 12))
        hold_rank_cb = ttk.Checkbutton(
            pick_row,
            text="Hold until rank worse",
            variable=hold_until_rank_exit_var,
        )
        hold_rank_cb.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            pick_row,
            text="Exit below 9 EMA",
            variable=exit_below_9ema_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        max_rank_lbl = tk.Label(pick_row, text="Max hold rank:")
        max_rank_spin = ttk.Spinbox(
            pick_row,
            from_=5,
            to=60,
            width=4,
            textvariable=max_hold_rank_var,
        )
        tk.Label(pick_row, text="Portfolio N:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(
            pick_row,
            from_=1,
            to=_PORTFOLIO_N_MAX,
            width=4,
            textvariable=portfolio_n_var,
        ).pack(side=tk.LEFT, padx=(0, 12))

        def _toggle_max_hold_rank_ui(*_) -> None:
            if hold_until_rank_exit_var.get():
                max_rank_lbl.pack(side=tk.LEFT)
                max_rank_spin.pack(side=tk.LEFT, padx=(4, 12))
            else:
                max_rank_lbl.pack_forget()
                max_rank_spin.pack_forget()

        pick_auto_show_cb = ttk.Checkbutton(
            pick_row,
            text="Auto-show picks on graph",
            variable=pick_auto_show_var,
        )
        pick_auto_show_cb.pack(side=tk.LEFT, padx=(0, 8))
        preview_today_cb = ttk.Checkbutton(
            pick_row,
            text="Preview today's picks",
            variable=preview_today_picks_var,
        )
        preview_today_cb.pack(side=tk.LEFT, padx=(0, 8))
    else:
        def _pick_strategy_key() -> str:
            return "recommend"

        def _clear_pick_cache() -> None:
            pass

        def _toggle_max_hold_rank_ui(*_) -> None:
            pass

        hold_until_rank_exit_var = tk.BooleanVar(value=False)
        exit_below_9ema_var = tk.BooleanVar(value=config.exit_below_9ema)
        max_rank_lbl = None
        max_rank_spin = None
        hold_rank_cb = None
        pick_auto_show_cb = None
        preview_today_cb = None
        portfolio_n_var = None

    def _sync_preview_pick_ui() -> None:
        if preview_today_cb is None:
            return
        if bar_unit == "week" and config.preview_today_picks:
            preview_today_cb.pack(side=tk.LEFT, padx=(0, 8))
        else:
            if preview_today_picks_var.get():
                preview_today_picks_var.set(False)
            preview_today_cb.pack_forget()
        _sync_recommend_panel_visibility()

    if config.backtest_enabled:
        def open_backtest():
            from momentum.rrg_backtest_ui import open_rrg_backtest

            open_rrg_backtest(
                root,
                profile=config.backtest_profile,
                rrg_window=window,
                tail=int(float(tail_scale.get())),
                analysis_period=period,
                top_n=_portfolio_top_n(),
                backtest_extra={
                    **(
                        {
                            "universe_mode": _us_universe_mode_key(),
                            "min_adv_usd": config.backtest_min_adv,
                            "vol_percentile": config.backtest_vol_percentile,
                            "screen_categories": config.backtest_categories,
                        }
                        if config.backtest_profile == "us"
                        else {}
                    ),
                    **(
                        {"universe_key": config.backtest_universe_key}
                        if config.backtest_profile == "stock"
                        else {}
                    ),
                    "pick_strategy": _pick_strategy_key(),
                    "hold_until_rank_exit": bool(hold_until_rank_exit_var.get()),
                    "max_hold_rank": int(max_hold_rank_var.get()),
                    "exit_below_9ema": bool(exit_below_9ema_var.get()),
                }
                if config.etf_table_extras
                else None,
            )

        ttk.Button(
            controls_frame, text='Backtest', command=open_backtest
        ).pack(side=tk.LEFT, padx=(0, 12))

    tail_row = tk.Frame(controls_frame)
    tail_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    tk.Label(tail_row, text='Tail', width=6, anchor='w').pack(side=tk.LEFT)
    tk.Label(tail_row, text='Unit', width=5, anchor='w').pack(side=tk.LEFT, padx=(0, 2))
    bar_unit_combo = ttk.Combobox(
        tail_row,
        textvariable=bar_unit_var,
        values=["Week", "Day"],
        state="readonly",
        width=6,
    )
    bar_unit_combo.pack(side=tk.LEFT, padx=(0, 8))

    def update_tail_step_buttons():
        current = int(float(tail_scale.get()))
        tail_min = int(float(tail_scale.cget('from')))
        tail_max = int(float(tail_scale.cget('to')))
        if current <= tail_min:
            tail_dec_button.state(['disabled'])
        else:
            tail_dec_button.state(['!disabled'])
        if current >= tail_max:
            tail_inc_button.state(['disabled'])
        else:
            tail_inc_button.state(['!disabled'])

    def step_decrease_tail():
        new_tail = int(float(tail_scale.get())) - 1
        if new_tail < int(float(tail_scale.cget('from'))):
            return
        tail_scale.set(new_tail)
        on_tail_change(new_tail)

    def step_increase_tail():
        new_tail = int(float(tail_scale.get())) + 1
        if new_tail > int(float(tail_scale.cget('to'))):
            return
        tail_scale.set(new_tail)
        on_tail_change(new_tail)

    tail_dec_button = ttk.Button(
        tail_row, text='−', width=3, command=step_decrease_tail
    )
    tail_dec_button.pack(side=tk.LEFT, padx=(0, 4))
    tail_scale = tk.Scale(
        tail_row,
        from_=1,
        to=RRG_MAX_TAIL,
        orient=tk.HORIZONTAL,
        showvalue=True,
        resolution=1,
        command=on_tail_change,
    )
    tail_scale.set(tail)
    tail_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    tail_inc_button = ttk.Button(
        tail_row, text='+', width=3, command=step_increase_tail
    )
    tail_inc_button.pack(side=tk.LEFT, padx=(4, 0))

    date_row = tk.Frame(controls_frame)
    date_row.pack(side=tk.TOP, fill=tk.X)
    tk.Label(date_row, text='Date', width=6, anchor='w').pack(side=tk.LEFT)
    date_scale = tk.Scale(
        date_row,
        from_=date_min_idx,
        to=date_max_idx,
        orient=tk.HORIZONTAL,
        showvalue=False,
        resolution=1,
        command=on_date_change,
    )
    date_scale.set(end_date_idx)
    date_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    date_value_label = tk.Label(
        date_row, text=format_date_label(end_date_idx), width=12, anchor='w'
    )
    date_value_label.pack(side=tk.LEFT, padx=(8, 0))
    date_range_label = tk.Label(
        date_row, text=_date_range_hint_text(), anchor='w', fg='gray', font=('Arial', 9)
    )
    date_range_label.pack(side=tk.LEFT, padx=(4, 0))
    preview_status_label = tk.Label(
        date_row,
        text='',
        anchor='w',
        fg='#1b5e20',
        font=('Arial', 9, 'bold'),
    )
    preview_status_label.pack(side=tk.LEFT, padx=(8, 0))

    calc_row = tk.Frame(controls_frame)
    calc_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
    calc_context_label = tk.Label(
        calc_row,
        text='',
        anchor='w',
        fg='#333333',
        font=('Arial', 9),
        wraplength=1100,
        justify=tk.LEFT,
    )
    calc_context_label.pack(side=tk.LEFT, padx=(6, 0))

    def update_calc_context_label():
        """Show Change % / rank window for current Tail, Date, and Unit."""
        if not len(_rrg_index):
            calc_context_label.config(text='')
            return
        end_i = int(date_scale.get())
        tail_n = int(float(tail_scale.get()))
        unit_plural = 'days' if bar_unit == 'day' else 'weeks'
        if _preview_today_enabled():
            ctx = _preview_ranking_context()
            if ctx is not None:
                preview_end, preview_start = ctx[0], ctx[1]
                tail_d = _preview_tail_trading_days()
                calc_context_label.config(
                    text=(
                        f'Preview = Day unit @ {rrg_format_date(preview_end)}: '
                        f'Chg% {rrg_format_date(preview_start)}→'
                        f'{rrg_format_date(preview_end)} (Tail {tail_d})'
                    )
                )
                return
        if end_i < tail_n:
            calc_context_label.config(
                text=(
                    f'Change % start → end: — (tail {tail_n} {unit_plural} '
                    f'exceeds available history at this date)'
                )
            )
            return
        start_ts = format_date_label(end_i - tail_n)
        end_ts = format_date_label(end_i)
        rank_vs = ''
        if end_i > tail_n:
            rank_vs = f'  ·  Rank Δ vs prior bar: {format_date_label(end_i - 1)}'
        calc_context_label.config(
            text=(
                f'Change % start → end: {start_ts} → {end_ts} '
                f'(tail={tail_n} {unit_plural}){rank_vs}'
            )
        )

    table_section = tk.Frame(bottom_frame)
    table_section.grid(row=1, column=0, sticky='nsew', padx=4, pady=(0, 4))
    table_section.rowconfigure(0, weight=1)
    table_section.columnconfigure(0, weight=1)

    tables_row = tk.Frame(table_section)
    tables_row.grid(row=0, column=0, sticky='nsew')
    tables_row.rowconfigure(0, weight=1)
    tables_row.columnconfigure(0, weight=1)

    tables_pane = None
    _side_pane_sash_pos = 0
    _side_pane_auto = True

    movers_panel = None
    movers_title_label = None
    movers_dates_label = None
    movers_totals_label = None
    movers_exits_label = None
    movers_was_header_cells: list[tk.Label] = []
    movers_pick_header_cells: list[tk.Label] = []
    movers_row_widgets: list[dict[str, tk.Label]] = []
    side_panel = None
    side_content = None
    side_canvas = None
    _sync_side_scroll = None

    def _on_side_mousewheel(event):
        if side_canvas is None:
            return 'break'
        if event.delta:
            side_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        elif event.num == 4:
            side_canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            side_canvas.yview_scroll(1, 'units')
        return 'break'

    def _bind_side_mousewheel(*widgets):
        for widget in widgets:
            if widget is None:
                continue
            widget.bind('<MouseWheel>', _on_side_mousewheel)
            widget.bind('<Button-4>', _on_side_mousewheel)
            widget.bind('<Button-5>', _on_side_mousewheel)

    def _build_movers_panel(parent: tk.Frame) -> None:
        nonlocal movers_panel, movers_title_label, movers_dates_label, movers_totals_label
        nonlocal movers_exits_label, movers_was_header_cells, movers_pick_header_cells
        nonlocal movers_was_copy_grid, movers_pick_copy_grid
        movers_panel = tk.Frame(
            parent,
            padx=6,
            pady=4,
            relief=tk.GROOVE,
            borderwidth=1,
        )
        movers_panel.pack(
            side=tk.TOP,
            anchor='nw',
            fill=tk.BOTH if use_right_extras else tk.X,
            expand=bool(use_right_extras),
        )
        movers_title_label = tk.Label(
            movers_panel,
            text=config.top_movers_title,
            font=('Arial', 10, 'bold'),
            anchor='w',
        )
        movers_title_label.pack(fill=tk.X)
        movers_dates_label = tk.Label(
            movers_panel,
            text='',
            font=('Arial', 9),
            anchor='w',
            fg='gray',
            wraplength=860 if use_right_extras else 320,
            justify=tk.LEFT,
        )
        movers_dates_label.pack(fill=tk.X, pady=(0, 2))
        movers_totals_label = tk.Label(
            movers_panel,
            text="",
            font=("Arial", 9, "bold"),
            anchor="w",
            wraplength=860 if use_right_extras else 320,
            justify=tk.LEFT,
        )
        movers_totals_label.pack(fill=tk.X, pady=(0, 4))

        slot_count = max(
            config.top_movers_count, config.etf_recommend_count, _PORTFOLIO_N_MAX
        )
        grids_wrap = tk.Frame(movers_panel)
        grids_wrap.pack(fill=tk.X)

        was_table = tk.Frame(grids_wrap)
        was_table.pack(fill=tk.X)
        configure_portfolio_panel_table_columns(was_table)

        pick_table = tk.Frame(grids_wrap)
        pick_table.pack(fill=tk.X, pady=(6, 0))
        configure_portfolio_panel_table_columns(pick_table)

        movers_was_header_cells.clear()
        movers_pick_header_cells.clear()

        def _panel_cell_font(key: str) -> tuple[str, int]:
            return ('Arial', 8) if key in PORTFOLIO_PANEL_SMALL_KEYS else ('Arial', 9)

        def _panel_cell_fg(key: str) -> str:
            return '#1565C0' if key in PORTFOLIO_PANEL_TAG_KEYS else 'black'

        def _build_header_row(
            table: tk.Frame,
            specs: tuple[tuple[str, str, str, int], ...],
            *,
            out: list[tk.Label],
        ) -> list[tk.Label]:
            row_cells: list[tk.Label] = []
            for col, (_key, header, anchor, _min_px) in enumerate(specs):
                hdr = tk.Label(
                    table,
                    text=header,
                    font=('Arial', 9, 'bold'),
                    anchor=anchor,
                    relief=tk.RIDGE,
                )
                hdr.grid(row=0, column=col, sticky='ew', padx=2, pady=1)
                out.append(hdr)
                row_cells.append(hdr)
            return row_cells

        was_header_row = _build_header_row(
            was_table, PORTFOLIO_PANEL_WAS_ROW, out=movers_was_header_cells
        )
        pick_header_row = _build_header_row(
            pick_table, PORTFOLIO_PANEL_PICK_GRID_ROW, out=movers_pick_header_cells
        )

        movers_body_was_cells: list[list[tk.Label]] = []
        movers_body_pick_cells: list[list[tk.Label]] = []
        for slot in range(slot_count):
            widgets: dict[str, tk.Label] = {}
            was_grid_row = slot + 1
            pick_grid_row = slot + 1
            was_row_cells: list[tk.Label] = []
            pick_row_cells: list[tk.Label] = []

            for col, (key, _header, anchor, _min_px) in enumerate(
                PORTFOLIO_PANEL_WAS_ROW
            ):
                lbl = tk.Label(
                    was_table,
                    font=_panel_cell_font(key),
                    anchor=anchor,
                    relief=tk.RIDGE,
                    fg=_panel_cell_fg(key),
                )
                lbl.grid(row=was_grid_row, column=col, sticky='ew', padx=2, pady=1)
                widgets[key] = lbl
                was_row_cells.append(lbl)

            for col, (key, _header, anchor, _min_px) in enumerate(
                PORTFOLIO_PANEL_PICK_GRID_ROW
            ):
                widget_key = "pick_rank" if key == "rank" else key
                lbl = tk.Label(
                    pick_table,
                    font=_panel_cell_font(key),
                    anchor=anchor,
                    relief=tk.RIDGE,
                    fg=_panel_cell_fg(key),
                )
                lbl.grid(row=pick_grid_row, column=col, sticky='ew', padx=2, pady=1)
                widgets[widget_key] = lbl
                pick_row_cells.append(lbl)

            movers_row_widgets.append(widgets)
            movers_body_was_cells.append(was_row_cells)
            movers_body_pick_cells.append(pick_row_cells)

        tc = TableRegionCopy.for_window(root)
        movers_was_copy_grid = tc.register_grid([was_header_row, *movers_body_was_cells])
        movers_pick_copy_grid = tc.register_grid(
            [pick_header_row, *movers_body_pick_cells]
        )
        movers_exits_label = None

    if use_right_extras:
        tables_pane = ttk.PanedWindow(tables_row, orient=tk.HORIZONTAL)
        tables_pane.grid(row=0, column=0, sticky='nsew')
        side_panel = tk.Frame(tables_pane)
        side_panel.rowconfigure(0, weight=1)
        side_panel.columnconfigure(0, weight=1)

        side_scroll_y = tk.Scrollbar(side_panel, orient=tk.VERTICAL)
        side_scroll_x = tk.Scrollbar(side_panel, orient=tk.HORIZONTAL)
        side_canvas = tk.Canvas(
            side_panel,
            highlightthickness=0,
            xscrollcommand=side_scroll_x.set,
            yscrollcommand=side_scroll_y.set,
        )
        side_scroll_x.config(command=side_canvas.xview)
        side_scroll_y.config(command=side_canvas.yview)
        side_scroll_x.grid(row=1, column=0, sticky='ew')
        side_scroll_y.grid(row=0, column=1, sticky='ns')
        side_canvas.grid(row=0, column=0, sticky='nsew')

        side_content = tk.Frame(side_canvas)
        _side_canvas_win = side_canvas.create_window(
            (0, 0), window=side_content, anchor='nw'
        )

        def _sync_side_scroll(_event=None):
            if side_content is None or side_canvas is None:
                return
            side_content.update_idletasks()
            side_canvas.config(scrollregion=side_canvas.bbox('all'))
            cw = side_canvas.winfo_width()
            req_w = side_content.winfo_reqwidth()
            if cw > 1:
                side_canvas.itemconfigure(_side_canvas_win, width=max(cw, req_w))
            if side_panel is not None:
                panel_w = side_panel.winfo_width()
                if panel_w > 1:
                    wrap = max(panel_w - 24, 200)
                    if movers_dates_label is not None:
                        movers_dates_label.config(wraplength=wrap)
                    if movers_totals_label is not None:
                        movers_totals_label.config(wraplength=wrap)

        def _save_side_pane_sash(_event=None):
            nonlocal _side_pane_sash_pos, _side_pane_auto
            if tables_pane is None:
                return
            try:
                pos = tables_pane.sashpos(0)
                if pos > 80:
                    _side_pane_sash_pos = pos
                    _side_pane_auto = False
            except tk.TclError:
                pass
            _sync_side_scroll()

        side_content.bind('<Configure>', _sync_side_scroll)
        side_canvas.bind('<Configure>', _sync_side_scroll)
        tables_pane.bind('<ButtonRelease-1>', _save_side_pane_sash)
        side_panel.bind('<Configure>', _sync_side_scroll)

        if config.top_movers_panel:
            _build_movers_panel(side_content)
    elif config.top_movers_panel:
        side_panel = tk.Frame(tables_row)
        side_panel.config(width=340)
        side_panel.pack_propagate(False)
        side_panel.grid(row=0, column=1, sticky='ns', padx=(16, 0))
        tables_row.columnconfigure(1, weight=0, minsize=340)
        _build_movers_panel(side_panel)

        if config.side_cheat_sheet:
            cheat_panel = tk.Frame(
                side_panel,
                padx=6,
                pady=4,
                relief=tk.GROOVE,
                borderwidth=1,
            )
            cheat_panel.pack(side=tk.TOP, anchor='nw', fill=tk.X, pady=(8, 0))
            tk.Label(
                cheat_panel,
                text=config.side_cheat_sheet_title,
                font=('Arial', 10, 'bold'),
                anchor='w',
            ).pack(fill=tk.X, pady=(0, 4))
            for section_title, bullets in config.side_cheat_sheet:
                block = tk.Frame(cheat_panel)
                block.pack(fill=tk.X, pady=(0, 8))
                tk.Label(
                    block,
                    text=section_title,
                    font=('Arial', 9, 'bold'),
                    anchor='w',
                    justify=tk.LEFT,
                ).pack(fill=tk.X, anchor='w')
                for line in bullets:
                    tk.Label(
                        block,
                        text=f'• {line}',
                        font=('Arial', 9),
                        anchor='w',
                        fg='#333333',
                        justify=tk.LEFT,
                    ).pack(fill=tk.X, anchor='w', padx=(8, 0), pady=(1, 0))

    recommend_panel = None
    recommend_title_label: tk.Label | None = None
    recommend_dates_label = None
    recommend_rec_canvas: tk.Canvas | None = None
    recommend_row_widgets: list[dict[str, tk.Label]] = []
    recommend_copy_grid: dict | None = None
    recommend_exits_label: tk.Label | None = None
    main_table_copy_grid: dict | None = None
    movers_was_copy_grid: dict | None = None
    movers_pick_copy_grid: dict | None = None

    def _build_recommend_panel(parent: tk.Frame, *, pack_mode: str | None) -> None:
        nonlocal recommend_panel, recommend_title_label, recommend_dates_label
        nonlocal recommend_copy_grid, recommend_exits_label, recommend_rec_canvas
        recommend_panel = tk.Frame(
            parent,
            padx=4,
            pady=2,
            relief=tk.GROOVE,
            borderwidth=1,
        )
        if pack_mode == 'top':
            recommend_panel.pack(
                side=tk.TOP, anchor='nw', fill=tk.X, pady=(8, 0)
            )
        elif pack_mode == 'left':
            recommend_panel.pack(
                side=tk.LEFT, anchor='nw', fill=tk.BOTH, expand=True
            )

        recommend_title_label = tk.Label(
            recommend_panel,
            text="Preview today's Top N (daily)",
            font=('Arial', 10, 'bold'),
            anchor='w',
        )
        recommend_title_label.pack(fill=tk.X)
        recommend_dates_label = tk.Label(
            recommend_panel,
            text="",
            font=('Arial', 8),
            anchor='w',
            fg='gray',
            justify=tk.LEFT,
        )
        recommend_dates_label.pack(fill=tk.X, pady=(0, 2))

        rec_canvas = tk.Canvas(
            recommend_panel,
            highlightthickness=0,
            height=22 * 2 + 4,
        )
        recommend_rec_canvas = rec_canvas
        rec_canvas.pack(side=tk.TOP, fill=tk.X)

        rec_table = tk.Frame(rec_canvas)
        _rec_table_win = rec_canvas.create_window((0, 0), window=rec_table, anchor='nw')

        def _sync_rec_table_width(_event=None):
            rec_table.update_idletasks()
            cw = rec_canvas.winfo_width()
            req_w = rec_table.winfo_reqwidth()
            if cw > 1:
                rec_canvas.itemconfigure(_rec_table_win, width=max(cw, req_w))
            rec_canvas.config(scrollregion=rec_canvas.bbox('all'))

        rec_table.bind('<Configure>', _sync_rec_table_width)
        rec_canvas.bind('<Configure>', _sync_rec_table_width)

        _rec_col_specs: tuple[tuple[str, str, str, int], ...] = (
            ('rank', '#', 'e', 22),
            ('ticker', 'Ticker', 'w', 78),
            ('name', 'Name', 'w', 110),
            ('rank_delta', 'Rank Δ', 'e', 44),
            ('change', 'Chg%', 'e', 48),
        )
        rec_header_cells: list[tk.Label] = []
        for col, (_key, header, anchor, min_px) in enumerate(_rec_col_specs):
            rec_table.columnconfigure(col, minsize=min_px, weight=0)
            hdr = tk.Label(
                rec_table,
                text=header,
                font=('Arial', 9, 'bold'),
                anchor=anchor,
                relief=tk.RIDGE,
            )
            hdr.grid(row=0, column=col, sticky='ew', padx=1, pady=0)
            rec_header_cells.append(hdr)

        rec_body_cells: list[list[tk.Label]] = []
        for slot in range(_PORTFOLIO_N_MAX):
            widgets: dict[str, tk.Label] = {}
            row_cells: list[tk.Label] = []
            for col, (key, _header, anchor, _min_px) in enumerate(_rec_col_specs):
                lbl = tk.Label(
                    rec_table, font=('Arial', 9), anchor=anchor, relief=tk.RIDGE
                )
                widgets[key] = lbl
                row_cells.append(lbl)
            recommend_row_widgets.append(widgets)
            rec_body_cells.append(row_cells)

        tc = TableRegionCopy.for_window(root)
        recommend_copy_grid = tc.register_grid([rec_header_cells, *rec_body_cells])

        recommend_exits_label = None

    if use_right_extras and side_content is not None:
        _bind_side_mousewheel(
            side_panel,
            side_canvas,
            side_content,
            movers_panel,
        )

    scroll_wrap = tk.Frame(tables_pane if use_right_extras else tables_row)
    if use_right_extras and tables_pane is not None and side_panel is not None:
        tables_pane.add(scroll_wrap, weight=0)
        tables_pane.add(side_panel, weight=1)
    else:
        scroll_wrap.grid(row=0, column=0, sticky='nsew')
    scroll_wrap.columnconfigure(0, weight=1)
    scroll_wrap.rowconfigure(1, weight=1)

    def _sync_recommend_panel_visibility() -> None:
        """3rd table: Week + Preview only; hidden on Day unit."""
        if recommend_panel is None:
            return
        show = (
            bar_unit == "week"
            and config.preview_today_picks
            and preview_today_picks_var.get()
        )
        if use_right_extras:
            if show:
                recommend_panel.pack(
                    side=tk.TOP, anchor='nw', fill=tk.X, pady=(4, 0)
                )
            else:
                recommend_panel.pack_forget()
        elif show:
            recommend_panel.grid(
                row=2, column=0, columnspan=2, sticky='ew', padx=4, pady=(4, 2)
            )
        else:
            recommend_panel.grid_remove()

    if config.etf_table_extras and use_right_extras and side_content is not None:
        _build_recommend_panel(side_content, pack_mode='top')
        recommend_panel.pack_forget()
    elif config.etf_table_extras and not use_right_extras:
        _build_recommend_panel(scroll_wrap, pack_mode=None)
        recommend_panel.grid_remove()
        scroll_wrap.rowconfigure(2, weight=0)

    _HEADER_ROW_PX = 40
    header_canvas = tk.Canvas(
        scroll_wrap, height=_HEADER_ROW_PX, highlightthickness=0, borderwidth=0
    )
    header_canvas.grid(row=0, column=0, sticky='ew')
    table_header = tk.Frame(header_canvas)
    _header_canvas_win = header_canvas.create_window((0, 0), window=table_header, anchor='nw')

    header_scroll_gutter = tk.Frame(scroll_wrap, width=18)
    header_scroll_gutter.grid(row=0, column=1, sticky='ns')

    body_wrap = tk.Frame(scroll_wrap)
    body_wrap.grid(row=1, column=0, sticky='nsew')
    body_wrap.columnconfigure(0, weight=1)
    body_wrap.rowconfigure(0, weight=1)

    table_scroll_y = tk.Scrollbar(scroll_wrap, orient=tk.VERTICAL)
    table_scroll_y.grid(row=1, column=1, sticky='ns')

    table_canvas = tk.Canvas(
        body_wrap,
        highlightthickness=0,
        yscrollcommand=table_scroll_y.set,
    )
    table_canvas.grid(row=0, column=0, sticky='nsew')
    table_scroll_y.config(command=table_canvas.yview)

    def _sync_header_scroll_gutter(_event=None):
        table_scroll_y.update_idletasks()
        gutter_w = table_scroll_y.winfo_width()
        if gutter_w > 1:
            header_scroll_gutter.configure(width=gutter_w)

    table_body = tk.Frame(table_canvas)
    _table_canvas_win = table_canvas.create_window((0, 0), window=table_body, anchor='nw')

    _COL_RANK = 0
    _COL_RANK_DELTA = 1
    _COL_REF = 2
    _COL_INDEX = 3
    _COL_PRICE = 4
    _COL_CHANGE = 5
    _etf_extras = config.etf_table_extras
    if _etf_extras:
        _COL_VOL = 6
        _COL_VISIBLE = 7
    else:
        _COL_VOL = None
        _COL_VISIBLE = 6
    _TABLE_HEADERS = [
        'Rank',
        'Rank Δ',
        config.ref_column_header,
        config.name_column_header,
        'Price',
        'Change',
    ]
    if _etf_extras:
        _TABLE_HEADERS.append('Vol%')
    _TABLE_HEADERS.append('Visible')
    _TABLE_CELL_PADX = (2, 1)
    _TABLE_ROW_PADY = 1
    _TABLE_FONT = ('Arial', 10)
    _TABLE_FONT_BOLD = ('Arial', 10, 'bold')
    _TABLE_NEUTRAL_BG = root.cget('bg')
    _VISIBLE_COL = _COL_VISIBLE
    _TABLE_CELL_PAD_PX = 14
    _TABLE_VISIBLE_EXTRA_PX = 40
    _TABLE_ROW_MIN_PX = 24
    _table_col_widths_px: list[int] = []
    _table_font_obj = None
    _table_font_bold_obj = None

    def _get_table_fonts():
        nonlocal _table_font_obj, _table_font_bold_obj
        if _table_font_obj is None:
            _table_font_obj = tkfont.Font(font=_TABLE_FONT)
            _table_font_bold_obj = tkfont.Font(font=_TABLE_FONT_BOLD)
        return _table_font_obj, _table_font_bold_obj

    def _text_px(text, *, bold: bool = False) -> int:
        font, font_bold = _get_table_fonts()
        f = font_bold if bold else font
        return int(f.measure(str(text))) + _TABLE_CELL_PAD_PX

    def _compute_column_widths_px(end_ts, start_ts, rank_delta_texts: list[str]) -> list[int]:
        """Pixel width per column from header text and current table values."""
        rank_w = _text_px(_TABLE_HEADERS[_COL_RANK], bold=True)
        rank_delta_w = _text_px(_TABLE_HEADERS[_COL_RANK_DELTA], bold=True)
        ref_w = _text_px(_TABLE_HEADERS[_COL_REF], bold=True)
        idx_w = _text_px(_TABLE_HEADERS[_COL_INDEX], bold=True)
        price_w = _text_px(_TABLE_HEADERS[_COL_PRICE], bold=True)
        chg_w = _text_px(_TABLE_HEADERS[_COL_CHANGE], bold=True)
        vol_w = 0
        if _etf_extras and _COL_VOL is not None:
            vol_w = _text_px(_TABLE_HEADERS[_COL_VOL], bold=True)
        vis_w = _text_px(_TABLE_HEADERS[_COL_VISIBLE], bold=True) + _TABLE_VISIBLE_EXTRA_PX

        n = len(indices)
        if n:
            rank_w = max(rank_w, _text_px(str(n)))
            for text in rank_delta_texts:
                rank_delta_w = max(rank_delta_w, _text_px(text))
            for j in range(n):
                ref_w = max(ref_w, _text_px(index_metadata['ref_label'][j] or '-'))
                idx_w = max(idx_w, _text_px(index_metadata['display'][j]))
                try:
                    p = round(float(indices_data[indices[j]].loc[end_ts]), 2)
                    price_w = max(price_w, _text_px(p))
                except (KeyError, TypeError, ValueError):
                    pass
                chg = _tail_change_pct(j, start_ts, end_ts)
                if chg != float('-inf'):
                    chg_w = max(chg_w, _text_px(format_change_pct(chg)))
                if _etf_extras and _COL_VOL is not None:
                    from momentum.etf.us_rrg_recommendations import format_vol_pct

                    vol_w = max(
                        vol_w,
                        _text_px(
                            format_vol_pct(etf_vol_by_row.get(j, 0.0))
                            or "Vol%"
                        ),
                    )

        widths = [rank_w, rank_delta_w, ref_w, idx_w, price_w, chg_w]
        if _etf_extras and _COL_VOL is not None:
            widths.append(vol_w)
        widths.append(vis_w)
        return widths

    def _apply_table_column_widths(widths_px: list[int]):
        nonlocal _table_col_widths_px
        _table_col_widths_px = [int(w) for w in widths_px]
        for col, w in enumerate(_table_col_widths_px):
            table_header.columnconfigure(col, minsize=w, weight=0)
            table_body.columnconfigure(col, minsize=w, weight=0)

    def _sync_index_entry_widths():
        if not table_widgets or not _table_col_widths_px:
            return
        font, _ = _get_table_fonts()
        chars = max(
            len(_TABLE_HEADERS[_COL_INDEX]),
            max((len(index_metadata['display'][i]) for i in range(len(indices))), default=0),
        )
        entry_px = int(chars * font.measure('0') + _TABLE_CELL_PAD_PX + 8)
        if entry_px > _table_col_widths_px[_COL_INDEX]:
            _table_col_widths_px[_COL_INDEX] = entry_px
            table_header.columnconfigure(_COL_INDEX, minsize=entry_px)
            table_body.columnconfigure(_COL_INDEX, minsize=entry_px)
        for w in table_widgets:
            w['index_entry'].config(width=chars)

    def _update_table_column_widths(end_ts, start_ts, rank_delta_texts: list[str]):
        if not indices:
            return
        _apply_table_column_widths(
            _compute_column_widths_px(end_ts, start_ts, rank_delta_texts)
        )
        _sync_index_entry_widths()
        for row in range(len(indices)):
            table_body.grid_rowconfigure(row, minsize=_TABLE_ROW_MIN_PX)
        _sync_header_row_height()

    def _table_col_sticky(col: int) -> str:
        return 'nsew'

    def _sync_header_row_height():
        table_header.update_idletasks()
        req_h = max(table_header.winfo_reqheight(), _HEADER_ROW_PX)
        header_canvas.configure(height=req_h)

    def _sync_table_layout(_event=None):
        table_scroll_y.update_idletasks()
        gutter_w = table_scroll_y.winfo_width()
        if gutter_w <= 1:
            gutter_w = 18
        header_scroll_gutter.configure(width=gutter_w)
        table_header.update_idletasks()
        table_body.update_idletasks()
        _sync_header_row_height()
        if _table_col_widths_px:
            total_w = sum(_table_col_widths_px)
        else:
            total_w = max(table_header.winfo_reqwidth(), table_body.winfo_reqwidth(), 1)
        table_canvas.itemconfigure(_table_canvas_win, width=total_w)
        header_canvas.itemconfigure(_header_canvas_win, width=total_w)
        body_h = max(table_body.winfo_reqheight(), 1)
        header_h = max(table_header.winfo_reqheight(), _HEADER_ROW_PX)
        table_canvas.configure(scrollregion=(0, 0, total_w, body_h))
        header_canvas.configure(scrollregion=(0, 0, total_w, header_h))
        table_canvas.xview_moveto(0)
        header_canvas.xview_moveto(0)
        if tables_pane is not None:
            _sync_side_pane_sash()

    def _sync_side_pane_sash(_event=None):
        nonlocal _side_pane_sash_pos
        if tables_pane is None or _sync_side_scroll is None:
            return
        tables_pane.update_idletasks()
        total_w = tables_pane.winfo_width()
        if total_w <= 1:
            return
        min_right = 360
        min_left = 280
        gutter = max(table_scroll_y.winfo_width(), 18)
        pad = 8
        if _table_col_widths_px:
            left_w = sum(_table_col_widths_px) + gutter + pad
        else:
            left_w = max(table_header.winfo_reqwidth(), 1) + gutter + pad
        left_w = max(left_w, min_left)
        if _side_pane_auto:
            sash = min(left_w, total_w - min_right)
        else:
            sash = min(max(_side_pane_sash_pos, min_left), total_w - min_right)
        try:
            if abs(tables_pane.sashpos(0) - sash) > 2:
                tables_pane.sashpos(0, sash)
        except tk.TclError:
            pass
        _sync_side_scroll()

    def _sync_table_scroll_region(_event=None):
        _sync_table_layout()

    def _on_table_mousewheel(event):
        if event.delta:
            table_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        elif event.num == 4:
            table_canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            table_canvas.yview_scroll(1, 'units')

    table_canvas.bind('<Configure>', _sync_table_layout)
    table_body.bind('<Configure>', _sync_table_scroll_region)
    table_header.bind('<Configure>', _sync_table_layout)
    header_canvas.bind('<Configure>', _sync_table_layout)
    table_scroll_y.bind('<Configure>', _sync_header_scroll_gutter)
    tables_row.bind('<Configure>', _sync_table_layout)
    if tables_pane is not None:
        tables_pane.bind('<Configure>', _sync_table_layout)
    for widget in (
        table_canvas,
        header_canvas,
        table_body,
        table_header,
        body_wrap,
        scroll_wrap,
        tables_row,
        table_section,
    ):
        widget.bind('<MouseWheel>', _on_table_mousewheel)
        widget.bind('<Button-4>', _on_table_mousewheel)
        widget.bind('<Button-5>', _on_table_mousewheel)

    for j in range(len(_TABLE_HEADERS)):
        if j == _COL_VISIBLE:
            visible_header = tk.Frame(
                table_header,
                relief=tk.RIDGE,
                bg=_TABLE_NEUTRAL_BG,
                highlightthickness=0,
            )
            visible_header.grid(
                row=0,
                column=j,
                sticky='nsew',
                padx=_TABLE_CELL_PADX,
                pady=_TABLE_ROW_PADY,
            )
            vis_hdr_inner = tk.Frame(visible_header, bg=_TABLE_NEUTRAL_BG)
            vis_hdr_inner.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
            tk.Label(
                vis_hdr_inner,
                text=_TABLE_HEADERS[j],
                anchor='w',
                font=_TABLE_FONT_BOLD,
                bg=_TABLE_NEUTRAL_BG,
            ).pack(side=tk.LEFT)
            select_all_cb = ttk.Checkbutton(vis_hdr_inner, variable=select_all_var)
            select_all_cb.pack(side=tk.LEFT, padx=(4, 0))
        else:
            right_cols = (_COL_RANK, _COL_RANK_DELTA, _COL_PRICE, _COL_CHANGE)
            if _COL_VOL is not None:
                right_cols = (*right_cols, _COL_VOL)
            anchor = 'e' if j in right_cols else 'w'
            tk.Label(
                table_header,
                text=_TABLE_HEADERS[j],
                relief=tk.RIDGE,
                anchor=anchor,
                font=_TABLE_FONT_BOLD,
                bg=_TABLE_NEUTRAL_BG,
            ).grid(
                row=0,
                column=j,
                sticky=_table_col_sticky(j),
                padx=_TABLE_CELL_PADX,
                pady=_TABLE_ROW_PADY,
            )

    table_header.grid_rowconfigure(0, minsize=_HEADER_ROW_PX)

    def update_entry(event):
        nonlocal indices_data, indices, indices_to_show
        row = event.widget._row_idx
        requested = event.widget.get().strip()
        col_name = config.name_column_header
        try:
            row_id = config.resolve_row_id(requested)
            if not row_id:
                raise ValueError(f'unknown {col_name}: {requested!r}')
            kind = config.row_kind(row_id)
            series = config.load_row_history(
                row_id, kind, period, min_history_bars, window, freq=bar_unit
            )
            if len(series) < min_history_bars:
                raise ValueError(f'insufficient {bar_unit} history')
            previous = indices[row]
            if previous in indices_to_show:
                indices_to_show.remove(previous)
            indices[row] = row_id
            indices_data[row_id] = series
            if checkbox_vars[row].get() and row_id not in indices_to_show:
                indices_to_show.append(row_id)
            index_metadata['ref_label'][row] = config.row_ref_label(row_id)
            index_metadata['display'][row] = config.row_display_label(row_id)
            index_metadata['kind'][row] = kind
            table_widgets[row]['ref_label'].config(
                text=index_metadata['ref_label'][row]
            )
            update_rrg()
            redraw_chart()
        except Exception as e:
            print(e)
            entry = event.widget
            row = entry._row_idx
            entry.delete(0, tk.END)
            entry.insert(0, index_metadata['display'][row])

    def on_visibility_toggle(row_idx):
        nonlocal indices_to_show
        if default_indices_var.get():
            default_indices_var.set(False)
        index_name = indices[row_idx]
        if checkbox_vars[row_idx].get():
            if index_name not in indices_to_show:
                indices_to_show.append(index_name)
        else:
            indices_to_show = [n for n in indices_to_show if n != index_name]
        _sync_select_all_checkbox()
        redraw_chart()

    def _tail_change_pct(row_idx: int, start_ts, end_ts):
        """% price change over the visible tail window (for ranking)."""
        index_name = indices[row_idx]
        return tail_change_pct(indices_data[index_name], start_ts, end_ts)

    def _rank_by_row(end_date_idx_local: int) -> dict[int, int]:
        """Row index -> rank (1 = best tail-window change) at the given week index."""
        if end_date_idx_local < tail:
            return {}
        end_ts = _rrg_index[end_date_idx_local]
        start_ts = _rrg_index[end_date_idx_local - tail]

        def change_pct_fn(j: int) -> float:
            return _tail_change_pct(j, start_ts, end_ts)

        return rank_by_tail_change(len(indices), change_pct_fn)

    def _rank_delta_fg(delta_text: str) -> str:
        if delta_text.startswith('+'):
            return '#006400'
        if delta_text.startswith('-'):
            return '#8b0000'
        return 'black'

    def _ranked_rows_at(end_date_idx_local: int) -> list[int]:
        """Same row order as the main table at the given week index."""
        if end_date_idx_local < tail:
            return []
        end_ts = _rrg_index[end_date_idx_local]
        start_ts = _rrg_index[end_date_idx_local - tail]

        def change_pct_fn(j: int) -> float:
            return _tail_change_pct(j, start_ts, end_ts)

        return ranked_row_indices(len(indices), change_pct_fn)

    def _preview_today_enabled() -> bool:
        return (
            bar_unit == "week"
            and config.preview_today_picks
            and preview_today_picks_var.get()
        )

    def _preview_data_stale() -> bool:
        weekly_end = (
            pd.Timestamp(_rrg_index[-1]).normalize().date()
            if len(_rrg_index)
            else None
        )
        max_daily = None
        for series in _etf_daily_close.values():
            if series is not None and len(series):
                d = pd.Timestamp(series.index[-1]).normalize().date()
                if max_daily is None or d > max_daily:
                    max_daily = d
        day_hist = _history_cache.get("day", {})
        bench = day_hist.get(config.benchmark_nse, pd.Series(dtype=float))
        if len(bench):
            d = pd.Timestamp(bench.index[-1]).normalize().date()
            if max_daily is None or d > max_daily:
                max_daily = d
        if not _etf_daily_close and not len(bench):
            return True
        if weekly_end is not None and (max_daily is None or max_daily <= weekly_end):
            return True
        return False

    def _invalidate_daily_pick_rrg() -> None:
        nonlocal _daily_pick_rrg_cache
        _daily_pick_rrg_cache = None

    def _refresh_preview_market_data() -> None:
        """Refresh daily prices for preview (US Yahoo / India & stock NSE CM)."""
        profile = config.etf_recommend_profile
        print(f"RRG: refreshing daily EOD for preview ({profile})...")
        with _busy.busy("Refreshing daily preview data…"):
            _history_cache.pop("day", None)
            _invalidate_daily_pick_rrg()
            _load_etf_daily_close_data(force=True)
            _histories_for_unit("day")

    def _ensure_preview_day_histories() -> bool:
        """Load/refresh daily EOD for preview (US / India ETF / stock)."""
        if _preview_data_stale():
            _refresh_preview_market_data()
        elif "day" not in _history_cache:
            _histories_for_unit("day")
        return len(_preview_daily_calendar()) >= 2

    def _daily_benchmark_calendar() -> pd.DatetimeIndex:
        bench = (
            _histories_for_unit("day")
            .get(config.benchmark_nse, pd.Series(dtype=float))
            .dropna()
            .sort_index()
        )
        return pd.DatetimeIndex(bench.index)

    def _preview_daily_calendar() -> pd.DatetimeIndex:
        """Latest trading dates from benchmark, universe day bars, and CM/Yahoo daily."""
        dates: set[pd.Timestamp] = set()
        day_hist = _history_cache.get("day", {})
        bench = day_hist.get(config.benchmark_nse, pd.Series(dtype=float))
        if len(bench):
            dates.update(pd.DatetimeIndex(bench.dropna().index))
        for name in indices:
            series = day_hist.get(name, pd.Series(dtype=float))
            if len(series):
                dates.update(pd.DatetimeIndex(series.dropna().index))
        for series in _etf_daily_close.values():
            if series is not None and len(series):
                dates.update(pd.DatetimeIndex(series.dropna().index))
        if not dates:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(sorted(dates))

    def _daily_price_for_row(row_idx: int) -> pd.Series:
        """Daily close series for preview ranking (CM/Yahoo preferred for tradables)."""
        name = indices[row_idx]
        if name == config.benchmark_nse:
            bench = _history_cache.get("day", {}).get(
                config.benchmark_nse, pd.Series(dtype=float)
            )
            if len(bench):
                return bench
            if name in _etf_daily_close and len(_etf_daily_close[name]):
                return _etf_daily_close[name]
        if config.etf_recommend_profile == "india":
            ref = (
                (index_metadata["ref_label"][row_idx] or name)
                .strip()
                .upper()
                .replace(".NS", "")
            )
            if ref in _etf_daily_close and len(_etf_daily_close[ref]):
                return _etf_daily_close[ref]
        elif config.etf_recommend_profile == "stock":
            sym = name.strip().upper().replace(".NS", "")
            if sym in _etf_daily_close and len(_etf_daily_close[sym]):
                return _etf_daily_close[sym]
        else:
            if name in _etf_daily_close and len(_etf_daily_close[name]):
                return _etf_daily_close[name]
        return _histories_for_unit("day").get(name, pd.Series(dtype=float))

    def _preview_live_close_map() -> dict[str, float] | None:
        return live_close_for_panel(
            config.etf_recommend_profile,
            etf_daily_close=_etf_daily_close,
            at_latest_bar=True,
        )

    def _preview_row_mark_price(row_idx: int, end_ts: pd.Timestamp) -> float | None:
        series = _daily_price_for_row(row_idx)
        if not len(series):
            return None
        mark = pd.Timestamp(end_ts).normalize()
        last_bar = pd.Timestamp(series.index[-1]).normalize()
        live = _preview_live_close_map()
        if live:
            if config.etf_recommend_profile == "india":
                key = (
                    (index_metadata["ref_label"][row_idx] or indices[row_idx])
                    .strip()
                    .upper()
                    .replace(".NS", "")
                )
            elif config.etf_recommend_profile == "stock":
                key = indices[row_idx].strip().upper().replace(".NS", "")
            else:
                key = indices[row_idx]
            px = live.get(key)
            if px and mark > last_bar:
                return float(px)
        try:
            return float(series_at(series, end_ts))
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def _preview_tail_change_pct(row_idx: int, start_ts, end_ts) -> float:
        series = _daily_price_for_row(row_idx)
        if not len(series):
            return float("-inf")
        try:
            p_start = float(series_at(series, start_ts))
            p_end = _preview_row_mark_price(row_idx, end_ts)
            if p_end is None or p_start == 0:
                return float("-inf")
            return (p_end - p_start) / p_start * 100
        except (KeyError, TypeError, ValueError, IndexError):
            return float("-inf")

    def _update_preview_status_label() -> None:
        if preview_status_label is None:
            return
        if not preview_today_picks_var.get() or bar_unit != "week":
            preview_status_label.config(text='')
            return
        if not _ensure_preview_day_histories():
            preview_status_label.config(
                text='Preview ON — daily data unavailable',
                fg='#b71c1c',
            )
            return
        cal = _preview_daily_calendar()
        if len(cal) < 2:
            preview_status_label.config(
                text='Preview ON — daily data unavailable',
                fg='#b71c1c',
            )
            return
        end_ts = pd.Timestamp(cal[-1])
        slider_ts = pd.Timestamp(_rrg_index[int(date_scale.get())])
        extra = ''
        if end_ts.normalize() > slider_ts.normalize():
            extra = f' (newer than weekly {rrg_format_date(slider_ts)})'
        preview_status_label.config(
            text=f'PREVIEW · daily as of {rrg_format_date(end_ts)}{extra}',
            fg='#1b5e20',
        )

    def _trading_days_back(
        cal: pd.DatetimeIndex, end_ts: pd.Timestamp, n_days: int
    ) -> pd.Timestamp:
        if not len(cal) or n_days < 0:
            return pd.Timestamp(end_ts)
        sub = cal[cal <= pd.Timestamp(end_ts)]
        if not len(sub):
            return pd.Timestamp(end_ts)
        idx = len(sub) - 1 - n_days
        if idx < 0:
            return pd.Timestamp(sub[0])
        return pd.Timestamp(sub[idx])

    def _preview_tail_trading_days() -> int:
        """Match Day unit: Tail 1 = 1 trading day (not week×5)."""
        return max(1, tail)

    def _daily_pick_calendar() -> pd.DatetimeIndex:
        """Same bar dates as Day-unit slider (for preview ↔ day sync)."""
        if bar_unit == "day" and len(_rrg_index):
            return _rrg_index
        if not _ensure_preview_day_histories():
            return pd.DatetimeIndex([])
        from momentum.rrg_core import rrg_build_slider_date_index

        day_hist = _histories_for_unit("day")
        bench = day_hist.get(config.benchmark_nse, pd.Series(dtype=float))
        if not len(bench.dropna()):
            return pd.DatetimeIndex([])
        sources = [
            _daily_price_for_row(j).dropna().sort_index()
            for j in range(len(indices))
        ]
        return rrg_build_slider_date_index(
            bench.dropna().sort_index(),
            analysis_period=period,
            window=window,
            unit="day",
            daily_sources=[s for s in sources if len(s)],
        )

    def _ensure_daily_pick_rrg() -> tuple[list, list] | None:
        """Daily RSR/RSM (Day unit). On week unit builds overlay for preview."""
        nonlocal _daily_pick_rrg_cache
        if bar_unit == "day":
            return rsr_tickers, rsm_tickers
        if not _ensure_preview_day_histories():
            return None
        day_hist = _histories_for_unit("day")
        bench = day_hist.get(config.benchmark_nse, pd.Series(dtype=float))
        if bench is None or not len(bench.dropna()):
            return None
        bench = bench.dropna().sort_index()
        eff = rrg_effective_window(window, "day")
        cal = _daily_pick_calendar()
        stamp = (len(indices), eff, pd.Timestamp(cal[-1]) if len(cal) else None, tail)
        if _daily_pick_rrg_cache is not None and _daily_pick_rrg_cache[0] == stamp:
            return _daily_pick_rrg_cache[1], _daily_pick_rrg_cache[2]
        daily_rsr: list = []
        daily_rsm: list = []
        empty = pd.Series(dtype=float)
        for j in range(len(indices)):
            series = _daily_price_for_row(j).dropna().sort_index()
            if len(series) <= eff:
                daily_rsr.append(empty)
                daily_rsm.append(empty)
                continue
            rsr, _, rsm = compute(series, bench, eff)
            daily_rsr.append(rsr if rsr is not None else empty)
            daily_rsm.append(rsm if rsm is not None else empty)
        _daily_pick_rrg_cache = (stamp, daily_rsr, daily_rsm)
        return daily_rsr, daily_rsm

    def _daily_rsr_rsm_for_panel() -> tuple[list, list]:
        """Daily RSR/RSM for Day unit + preview (India / US / stock)."""
        pack = _ensure_daily_pick_rrg()
        if pack is not None:
            return pack
        return rsr_tickers, rsm_tickers

    def _daily_tail_change_at(row_idx: int, start_ts, end_ts) -> float:
        if bar_unit == "day":
            return _tail_change_pct(row_idx, start_ts, end_ts)
        return _preview_tail_change_pct(row_idx, start_ts, end_ts)

    def _daily_pick_context_at(as_of_ts: pd.Timestamp) -> tuple | None:
        """Ranking context for Day Tail N at ``as_of`` (preview + day unit)."""
        cal = _daily_pick_calendar()
        tail_n = max(1, tail)
        if len(cal) <= tail_n:
            return None
        as_of = pd.Timestamp(as_of_ts).normalize()
        sub = cal[cal <= as_of]
        if len(sub) <= tail_n:
            return None
        end_ts = pd.Timestamp(sub[-1])
        start_ts = _trading_days_back(cal, end_ts, tail_n)
        prev_end_ts = start_ts
        prev_start_ts = _trading_days_back(cal, prev_end_ts, tail_n)

        def chg(j: int, s, e) -> float:
            return _daily_tail_change_at(j, s, e)

        ranked = sorted(
            range(len(indices)),
            key=lambda j: chg(j, start_ts, end_ts),
            reverse=True,
        )
        curr_ranks = rank_by_tail_change(
            len(indices), lambda j: chg(j, start_ts, end_ts)
        )
        prev_ranks = rank_by_tail_change(
            len(indices), lambda j: chg(j, prev_start_ts, prev_end_ts)
        )
        rank_delta_by_row: dict[int, str] = {}
        for j in ranked:
            rank_delta_by_row[j] = format_rank_delta(
                curr_ranks.get(j, len(indices)),
                prev_ranks.get(j),
            )
        return (
            end_ts,
            start_ts,
            ranked,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
        )

    def _daily_prev_holdings_for_pick(end_ts: pd.Timestamp) -> list[str]:
        if not hold_until_rank_exit_var.get() or bar_unit != "day":
            return []
        idx = int(_rrg_index.get_indexer([pd.Timestamp(end_ts).normalize()], method="ffill")[0])
        if idx <= tail or not _portfolio_cache_enabled():
            return []
        if idx - 1 not in _active_holdings_cache:
            _warm_pick_holdings_cache(idx - 1)
        if exit_below_9ema_var.get():
            return list(_active_holdings_cache.get(idx - 1, []))
        return list(_pick_holdings_cache.get(idx - 1, []))

    def _daily_top_n_picks_at(as_of_ts: pd.Timestamp):
        """
        Day-unit Top N at ``as_of`` (strategy + 9 EMA slots).
        Shared by Day slider and Week+Preview 3rd table.
        """
        ctx = _daily_pick_context_at(as_of_ts)
        if ctx is None:
            return None
        end_ts, start_ts, ranked, rank_delta_by_row, curr_ranks, prev_ranks = ctx
        rrg_pack = _ensure_daily_pick_rrg()
        if rrg_pack is None:
            return None
        daily_rsr, daily_rsm = rrg_pack
        picks = _compute_picks_at_week(
            -1,
            ranked,
            end_ts,
            start_ts,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
            prev_holdings=_daily_prev_holdings_for_pick(end_ts),
            write_cache=False,
            change_pct_fn=lambda j: _daily_tail_change_at(j, start_ts, end_ts),
            rsr_series_by_row=daily_rsr,
            rsm_series_by_row=daily_rsm,
        )
        if not picks:
            return []
        rebal_strategy = _rebal_tickers_table_order(picks, ranked)
        if exit_below_9ema_var.get():
            slots, _ = _rebal_slots_after_9ema(rebal_strategy, end_ts)
        else:
            slots = list(rebal_strategy)
        by_ticker = {p.ticker: p for p in picks}
        n_port = _portfolio_top_n()
        out = [by_ticker[t] for t in slots if t and t in by_ticker]
        return out[:n_port]

    def _daily_strategy_shortfall_hint(
        end_ts,
        start_ts,
        ranked: list[int],
        rank_delta_by_row: dict[int, str],
        curr_ranks: dict[int, int],
        prev_ranks: dict[int, int],
        daily_rsr: list,
        daily_rsm: list,
        picked_n: int,
    ) -> str:
        """Why daily preview has fewer than Portfolio N strategy picks."""
        strategy = _pick_strategy_key()
        top_n = _portfolio_top_n()
        max_rank = int(max_hold_rank_var.get())

        def _row_change_pct(j: int) -> float:
            return _daily_tail_change_at(j, start_ts, end_ts)

        if config.etf_recommend_profile == "india":
            from momentum.etf.india_rrg_pick_strategies import (
                IndiaPickContext,
                pick_shortfall_hint,
            )

            vol_by_ref = {
                (index_metadata["ref_label"][j] or indices[j])
                .upper()
                .replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                for j in range(len(indices))
            }
            ctx = IndiaPickContext(
                ranked_row_indices=ranked,
                indices=indices,
                ref_labels=index_metadata["ref_label"],
                display_labels=index_metadata["display"],
                vol_by_ref=vol_by_ref,
                end_ts=end_ts,
                rsr_series_by_row=daily_rsr,
                rsm_series_by_row=daily_rsm,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=_row_change_pct,
                series_at_fn=series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                top_n=top_n,
                prev_holdings=[],
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=max_rank,
            )
            return pick_shortfall_hint(strategy, ctx, picked_n)

        if config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import (
                StockPickContext,
                pick_shortfall_hint,
            )

            vol_by_ref = {
                indices[j].upper().replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                for j in range(len(indices))
            }
            ctx = StockPickContext(
                ranked_row_indices=ranked,
                indices=indices,
                ref_labels=index_metadata["ref_label"],
                display_labels=index_metadata["display"],
                vol_by_ref=vol_by_ref,
                end_ts=end_ts,
                rsr_series_by_row=daily_rsr,
                rsm_series_by_row=daily_rsm,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=_row_change_pct,
                series_at_fn=series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                top_n=top_n,
                prev_holdings=[],
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=max_rank,
                benchmark=config.benchmark_nse,
            )
            return pick_shortfall_hint(strategy, ctx, picked_n)

        from momentum.etf.us_rrg_pick_strategies import (
            UsPickContext,
            pick_shortfall_hint,
        )

        vol_by_ticker = {
            indices[j]: etf_vol_by_row.get(j, 0.0) for j in range(len(indices))
        }
        ctx = UsPickContext(
            ranked_row_indices=ranked,
            indices=indices,
            display_labels=index_metadata["display"],
            vol_by_ticker=vol_by_ticker,
            end_ts=end_ts,
            rsr_series_by_row=daily_rsr,
            rsm_series_by_row=daily_rsm,
            rank_delta_by_row=rank_delta_by_row,
            change_pct_fn=_row_change_pct,
            series_at_fn=series_at,
            curr_ranks=curr_ranks,
            prev_ranks=prev_ranks,
            top_n=top_n,
            prev_holdings=[],
            hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
            max_hold_rank=max_rank,
        )
        return pick_shortfall_hint(strategy, ctx, picked_n)

    def _daily_preview_table_picks_at(
        as_of_ts: pd.Timestamp,
    ) -> tuple[list | None, str]:
        """
        Preview table rows: strategy picks plus 9 EMA skip labels.
        Day unit Top N still uses ``_daily_top_n_picks_at`` (entered slots only).
        """
        from dataclasses import replace

        ctx = _daily_pick_context_at(as_of_ts)
        if ctx is None:
            return None, ""
        end_ts, start_ts, ranked, rank_delta_by_row, curr_ranks, prev_ranks = ctx
        rrg_pack = _ensure_daily_pick_rrg()
        if rrg_pack is None:
            return None, ""
        daily_rsr, daily_rsm = rrg_pack
        strategy_picks = _compute_picks_at_week(
            -1,
            ranked,
            end_ts,
            start_ts,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
            prev_holdings=_daily_prev_holdings_for_pick(end_ts),
            write_cache=False,
            change_pct_fn=lambda j: _daily_tail_change_at(j, start_ts, end_ts),
            rsr_series_by_row=daily_rsr,
            rsm_series_by_row=daily_rsm,
        )
        n_port = _portfolio_top_n()
        ema_lbl = rrg_format_date(end_ts)
        if not strategy_picks:
            hint = _daily_strategy_shortfall_hint(
                end_ts,
                start_ts,
                ranked,
                rank_delta_by_row,
                curr_ranks,
                prev_ranks,
                daily_rsr,
                daily_rsm,
                0,
            )
            return [], hint

        rebal_strategy = _rebal_tickers_table_order(strategy_picks, ranked)
        if exit_below_9ema_var.get():
            slots, dropped = _rebal_slots_after_9ema(rebal_strategy, end_ts)
        else:
            slots = list(rebal_strategy)
            dropped = []
        by_ticker = {p.ticker: p for p in strategy_picks}
        display: list = []
        for i in range(min(n_port, len(rebal_strategy))):
            strat = rebal_strategy[i]
            if strat not in by_ticker:
                continue
            slot = slots[i] if i < len(slots) else ""
            pick = by_ticker[strat]
            if slot:
                display.append(pick)
            elif exit_below_9ema_var.get() and strat in dropped:
                display.append(
                    replace(pick, name=f"{pick.name} · 9 EMA @ {ema_lbl}")
                )
        entered = sum(1 for t in slots[:n_port] if t)
        footnote = ""
        if not display:
            footnote = _daily_strategy_shortfall_hint(
                end_ts,
                start_ts,
                ranked,
                rank_delta_by_row,
                curr_ranks,
                prev_ranks,
                daily_rsr,
                daily_rsm,
                0,
            )
        elif exit_below_9ema_var.get() and entered < len(display):
            skipped = len(display) - entered
            footnote = (
                f"{entered}/{n_port} above 9 EMA — {skipped} skipped @ {ema_lbl}"
            )
        return display[:n_port], footnote

    def _preview_panel_state(
        ctx: tuple | None = None,
    ) -> tuple[list | None, str] | None:
        """Preview table 3 rows and optional footnote."""
        end_ts = ctx[0] if ctx is not None else _preview_latest_daily_ts()
        if end_ts is None:
            return None
        return _daily_preview_table_picks_at(end_ts)

    def _day_tail_pick_context(end_date_idx_local: int) -> tuple | None:
        """Ranking context at a day-bar index (Day unit Tail N)."""
        if end_date_idx_local < tail or end_date_idx_local >= len(_rrg_index):
            return None
        end_ts = _rrg_index[end_date_idx_local]
        start_ts = _rrg_index[end_date_idx_local - tail]
        curr_ranks = _rank_by_row(end_date_idx_local)
        prev_ranks = (
            _rank_by_row(end_date_idx_local - 1)
            if end_date_idx_local > tail
            else {}
        )
        ranked = _ranked_rows_at(end_date_idx_local)
        rank_delta_by_row: dict[int, str] = {}
        for j in ranked:
            rank_delta_by_row[j] = format_rank_delta(
                curr_ranks.get(j, len(indices)),
                prev_ranks.get(j),
            )
        return (
            end_ts,
            start_ts,
            ranked,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
        )

    def _day_tail_picks(end_date_idx_local: int):
        ctx = _day_tail_pick_context(end_date_idx_local)
        if ctx is None:
            return None
        end_ts, start_ts, ranked, rank_delta_by_row, curr_ranks, prev_ranks = ctx
        return _strategy_picks_for_week(
            end_date_idx_local,
            ranked,
            end_ts,
            start_ts,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
        )

    def _preview_latest_daily_ts() -> pd.Timestamp | None:
        cal = _daily_pick_calendar()
        if not len(cal):
            return None
        return pd.Timestamp(cal[-1])

    def _preview_ranking_context() -> tuple | None:
        """Daily rankings as of latest day-bar (same calendar as Day unit)."""
        latest = _preview_latest_daily_ts()
        if latest is None:
            return None
        return _daily_pick_context_at(latest)

    def _preview_today_picks(ctx: tuple | None = None):
        """Preview table rows at latest daily bar (includes 9 EMA skip labels)."""
        state = _preview_panel_state(ctx)
        if state is None:
            return None
        picks, _footnote = state
        return picks

    def _row_index_for_ticker(ticker: str) -> int | None:
        bare = ticker.strip().upper().replace(".NS", "")
        etf_j: int | None = None
        other_j: int | None = None
        for j, row_id in enumerate(indices):
            ref = (index_metadata["ref_label"][j] or row_id).strip().upper().replace(
                ".NS", ""
            )
            if bare not in (ref, row_id.upper().replace(".NS", "")):
                continue
            if index_metadata["kind"][j] == "etf":
                etf_j = j
            elif other_j is None:
                other_j = j
        return etf_j if etf_j is not None else other_j

    def _portfolio_cell_text(ticker: str, week_idx: int) -> str:
        if not ticker:
            return ""
        j = _row_index_for_ticker(ticker)
        if j is None:
            return ticker
        rk = _rank_by_row(week_idx).get(j)
        return format_portfolio_cell(ticker, rk)

    def _rebal_tickers_table_order(picks, ranked: list[int]) -> list[str]:
        """Top N tickers in main-table momentum rank order (same as ★ rows)."""
        by_row = {p.row_idx: p.ticker for p in picks}
        return [by_row[j] for j in ranked if j in by_row]

    def _prior_day_top_n_portfolio(end_date_idx_local: int) -> list[str]:
        """Prior trading day's Top N (Day unit Was column; all profiles)."""
        prev_idx = end_date_idx_local - 1
        if prev_idx < tail or prev_idx < _date_slider_min_idx():
            return []
        _ensure_etf_daily_close()
        picks = _daily_top_n_picks_at(_rrg_index[prev_idx])
        return [p.ticker for p in picks] if picks else []

    def _prior_week_top_n_portfolio(end_date_idx_local: int) -> list[str]:
        """Prior week's Top N list in ★ order (must match that week's Top N column)."""
        prev_idx = end_date_idx_local - 1
        if prev_idx < tail or prev_idx < _date_slider_min_idx():
            return []
        if prev_idx in _pick_holdings_cache:
            return list(_pick_holdings_cache[prev_idx])
        end_ts_p = _rrg_index[prev_idx]
        start_ts_p = _rrg_index[prev_idx - tail]
        curr_p = _rank_by_row(prev_idx)
        prev_r_p = _rank_by_row(prev_idx - 1) if prev_idx > tail else {}
        ranked_p = sorted(
            range(len(indices)),
            key=lambda j: _tail_change_pct(j, start_ts_p, end_ts_p),
            reverse=True,
        )
        rank_delta_p: dict[int, str] = {}
        for j in ranked_p:
            rank_delta_p[j] = format_rank_delta(
                curr_p.get(j, len(indices)), prev_r_p.get(j)
            )
        picks_p = _compute_picks_at_week(
            prev_idx,
            ranked_p,
            end_ts_p,
            start_ts_p,
            rank_delta_p,
            curr_p,
            prev_r_p,
        )
        rebal_strategy_p = _rebal_tickers_table_order(picks_p, ranked_p)
        if exit_below_9ema_var.get():
            rebal_slots_p, _ = _rebal_slots_after_9ema(rebal_strategy_p, end_ts_p)
            return rebal_slots_p
        return rebal_strategy_p

    def _end_prev_week_holdings(prev_date_idx: int | None) -> list[str] | None:
        """Week-end holdings after prior rebalance week (for Now column)."""
        if prev_date_idx is None or prev_date_idx < tail:
            return None
        if prev_date_idx in _active_holdings_cache:
            return list(_active_holdings_cache[prev_date_idx])
        if prev_date_idx in _pick_holdings_cache:
            return [t for t in _pick_holdings_cache[prev_date_idx] if t]
        return None

    def _panel_rebal_idx(end_date_idx_local: int) -> int:
        """Hold-week rebalance bar for portfolio panel (matches backtest Current week)."""
        from momentum.rrg_core import panel_rebal_bar_index

        if end_date_idx_local < 0 or end_date_idx_local >= len(_rrg_index):
            return max(0, end_date_idx_local)
        return panel_rebal_bar_index(
            _rrg_index,
            pd.Timestamp(_rrg_index[end_date_idx_local]),
            tail,
        )

    def _portfolio_panel_week_idx() -> int:
        """
        Weekly bar for Was / Top N tables.

        Week + Preview ON: when daily EOD is newer than the slider, use the
        weekly bar that contains the latest daily date (e.g. 08-06 → 05-06).
        """
        slider_idx = int(date_scale.get())
        if bar_unit != "week" or not _preview_today_enabled():
            return slider_idx
        latest = _preview_latest_daily_ts()
        if latest is None or not len(_rrg_index):
            return slider_idx
        slider_ts = pd.Timestamp(_rrg_index[slider_idx])
        if latest.normalize() <= slider_ts.normalize():
            return slider_idx
        pos = int(_rrg_index.get_indexer([latest.normalize()], method="ffill")[0])
        if pos < 0:
            return slider_idx
        return max(tail, min(pos, len(_rrg_index) - 1))

    def _portfolio_panel_as_of_ts() -> pd.Timestamp:
        """P/L as-of: latest daily when preview is newer than the weekly slider."""
        slider_idx = int(date_scale.get())
        as_of = pd.Timestamp(_rrg_index[slider_idx])
        if bar_unit == "week" and _preview_today_enabled():
            latest = _preview_latest_daily_ts()
            if latest is not None and latest.normalize() > as_of.normalize():
                return pd.Timestamp(latest)
        return as_of

    def _panel_week_picks(end_date_idx_local: int):
        """Strategy picks at hold-week rebalance (same as portfolio Top N panel)."""
        panel_rebal_idx = _panel_rebal_idx(end_date_idx_local)
        panel_rebal_ts = _rrg_index[panel_rebal_idx]
        panel_start_ts = _rrg_index[panel_rebal_idx - tail]
        panel_curr_ranks = _rank_by_row(panel_rebal_idx)
        prev_panel_idx = panel_rebal_idx - 1 if panel_rebal_idx > tail else None
        panel_prev_ranks = (
            _rank_by_row(prev_panel_idx) if prev_panel_idx is not None else {}
        )
        panel_ranked = _ranked_rows_at(panel_rebal_idx)
        panel_rank_delta: dict[int, str] = {}
        for j in panel_ranked:
            panel_rank_delta[j] = format_rank_delta(
                panel_curr_ranks.get(j, len(indices)),
                panel_prev_ranks.get(j),
            )
        if _portfolio_cache_enabled():
            _warm_pick_holdings_cache(panel_rebal_idx)
        return _strategy_picks_for_week(
            panel_rebal_idx,
            panel_ranked,
            panel_rebal_ts,
            panel_start_ts,
            panel_rank_delta,
            panel_curr_ranks,
            panel_prev_ranks,
        )

    def refresh_top_movers_panel(
        now_ranked: list[int] | None = None,
        *,
        picks=None,
        end_ts=None,
        start_ts=None,
        rank_delta_by_row: dict[int, str] | None = None,
        curr_ranks: dict[int, int] | None = None,
        prev_ranks: dict[int, int] | None = None,
    ):
        """Portfolio Was vs Now: prior holdings, rebalance picks, exits with P&L."""
        if not config.top_movers_panel or movers_dates_label is None:
            return
        end_date_idx_local = (
            _portfolio_panel_week_idx() if bar_unit == "week" else int(date_scale.get())
        )
        if bar_unit == "day":
            _ensure_etf_daily_close()
        day_unit_picks = False
        if bar_unit == "day":
            current_idx = end_date_idx_local
            prev_panel_idx = current_idx - 1 if current_idx > tail else None
            panel_rebal_idx = current_idx
            panel_end_idx = current_idx
            panel_rebal_ts = _rrg_index[current_idx]
            panel_end_ts = _rrg_index[current_idx]
            panel_start_ts = (
                _rrg_index[current_idx - tail]
                if current_idx >= tail
                else panel_rebal_ts
            )
            panel_curr_ranks = {}
            panel_prev_ranks = {}
            panel_ranked = []
            panel_rank_delta = {}
            panel_picks = None
            pick_column_picks = []
            pick_column_ranked = []
            pick_column_rebal_ts = panel_rebal_ts
            pick_column_start_ts = panel_start_ts
            pick_column_rank_delta = {}
            pick_column_curr_ranks = {}
            pick_column_prev_ranks = {}
            as_of = _rrg_index[current_idx]
            day_ctx = _daily_pick_context_at(as_of)
            day_picks = _daily_top_n_picks_at(as_of)
            if day_ctx is not None:
                (
                    pick_column_rebal_ts,
                    pick_column_start_ts,
                    pick_column_ranked,
                    pick_column_rank_delta,
                    pick_column_curr_ranks,
                    pick_column_prev_ranks,
                ) = day_ctx
                pick_column_picks = day_picks or []
                day_unit_picks = True
        else:
            panel_rebal_idx = _panel_rebal_idx(end_date_idx_local)
            panel_end_idx = min(panel_rebal_idx + 1, len(_rrg_index) - 1)
            panel_rebal_ts = _rrg_index[panel_rebal_idx]
            panel_end_ts = _rrg_index[panel_end_idx]
            prev_panel_idx = panel_rebal_idx - 1 if panel_rebal_idx > tail else None
            panel_start_ts = _rrg_index[panel_rebal_idx - tail]
            panel_curr_ranks = _rank_by_row(panel_rebal_idx)
            panel_prev_ranks = (
                _rank_by_row(prev_panel_idx) if prev_panel_idx is not None else {}
            )
            panel_ranked = _ranked_rows_at(panel_rebal_idx)
            panel_rank_delta: dict[int, str] = {}
            for j in panel_ranked:
                panel_rank_delta[j] = format_rank_delta(
                    panel_curr_ranks.get(j, len(indices)),
                    panel_prev_ranks.get(j),
                )
            if _portfolio_cache_enabled():
                _warm_pick_holdings_cache(panel_rebal_idx)
            panel_picks = _panel_week_picks(end_date_idx_local)
            pick_column_picks = panel_picks
            pick_column_ranked = panel_ranked
            pick_column_rebal_ts = panel_rebal_ts
            pick_column_start_ts = panel_start_ts
            pick_column_rank_delta = panel_rank_delta
            pick_column_curr_ranks = panel_curr_ranks
            pick_column_prev_ranks = panel_prev_ranks

        if config.etf_recommend_profile == "us":
            from momentum.etf.us_rrg_pick_strategies import pick_strategy_label
        elif config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import pick_strategy_label
        else:
            from momentum.etf.india_rrg_pick_strategies import pick_strategy_label

        if bar_unit == "day":
            prev_portfolio = _prior_day_top_n_portfolio(end_date_idx_local)
        else:
            prev_portfolio = (
                _prior_week_top_n_portfolio(panel_rebal_idx)
                if panel_rebal_idx > tail
                else []
            )
        rebal_strategy = _rebal_tickers_table_order(
            pick_column_picks, pick_column_ranked
        )
        use_weekly_pick_cache = (
            bar_unit != "day"
            and pick_column_picks is panel_picks
            and panel_rebal_idx in _pick_holdings_cache
        )
        if day_unit_picks:
            rebal_tickers = [p.ticker for p in pick_column_picks]
        elif use_weekly_pick_cache:
            rebal_tickers = list(_pick_holdings_cache[panel_rebal_idx])
        elif exit_below_9ema_var.get():
            rebal_tickers, _ = _rebal_slots_after_9ema(
                rebal_strategy, pick_column_rebal_ts
            )
        else:
            rebal_tickers = list(rebal_strategy)
        from momentum.rrg_portfolio_exits import (
            panel_was_out_exits,
        )

        exit_slices: list[tuple] = []
        if bar_unit == "day":
            if panel_rebal_idx >= tail:
                exit_slices.append((panel_rebal_ts, []))
        else:
            if prev_panel_idx is not None and prev_panel_idx >= tail:
                exit_slices.append(
                    (
                        _rrg_index[prev_panel_idx],
                        _week_exits_cache.get(prev_panel_idx, []),
                    )
                )
            exit_slices.append(
                (panel_rebal_ts, _week_exits_cache.get(panel_rebal_idx, []))
            )
        prev_rebal_ts = (
            _rrg_index[prev_panel_idx] if prev_panel_idx is not None else None
        )

        def _daily_for_panel(sym: str) -> pd.Series | None:
            _ensure_etf_daily_close()
            _weekly, daily = _etf_price_series(sym)
            return daily if len(daily) else None

        week_exits = panel_was_out_exits(
            exit_slices,
            panel_end_ts,
            prev_rebal_ts=prev_rebal_ts,
            panel_rebal_ts=panel_rebal_ts,
            prev_holdings=prev_portfolio,
            rebalance_holdings=[t for t in rebal_tickers if t],
            exit_below_9ema=bool(exit_below_9ema_var.get()),
            daily_for_ticker=_daily_for_panel,
        )

        was_label = (
            format_date_label(prev_panel_idx)
            if prev_panel_idx is not None
            else "—"
        )
        if movers_was_header_cells:
            movers_was_header_cells[PORTFOLIO_PANEL_WAS_COL].config(
                text=portfolio_panel_was_header(was_label)
            )
        rebalance_label = (
            rrg_format_date(_rrg_index[end_date_idx_local])
            if bar_unit == "day"
            else format_date_label(panel_rebal_idx)
        )
        pick_header_label = rebalance_label
        if movers_pick_header_cells:
            movers_pick_header_cells[PORTFOLIO_PANEL_REBAL_COL].config(
                text=portfolio_panel_pick_header(pick_header_label)
            )
        subtitle = pick_strategy_subtitle(
            _pick_strategy_key(),
            hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
            max_hold_rank=int(max_hold_rank_var.get()),
            exit_below_9ema=bool(exit_below_9ema_var.get()),
        )
        if movers_title_label is not None:
            movers_title_label.config(
                text=pick_strategy_label(
                    _pick_strategy_key(),
                    hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                    exit_below_9ema=bool(exit_below_9ema_var.get()),
                )
            )
        n_port = _portfolio_top_n()
        was_n = len([t for t in prev_portfolio if t])
        rebal_n = len([t for t in rebal_tickers if t])
        pick_shortfall = ""
        if config.etf_table_extras and rebal_n < n_port:
            rsr_for_hint, rsm_for_hint = (
                _daily_rsr_rsm_for_panel()
                if bar_unit == "day"
                else (rsr_tickers, rsm_tickers)
            )

            def _hint_change_pct(j: int) -> float:
                return (
                    _daily_tail_change_at(
                        j, pick_column_start_ts, pick_column_rebal_ts
                    )
                    if bar_unit == "day"
                    else _tail_change_pct(
                        j, pick_column_start_ts, pick_column_rebal_ts
                    )
                )

            if config.etf_recommend_profile == "india":
                from momentum.etf.india_rrg_pick_strategies import (
                    IndiaPickContext,
                    pick_shortfall_hint,
                )

                vol_by_ref = {
                    (index_metadata["ref_label"][j] or indices[j])
                    .upper()
                    .replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                    for j in range(len(indices))
                }
                pick_shortfall = pick_shortfall_hint(
                    _pick_strategy_key(),
                    IndiaPickContext(
                        ranked_row_indices=pick_column_ranked,
                        indices=indices,
                        ref_labels=index_metadata["ref_label"],
                        display_labels=index_metadata["display"],
                        vol_by_ref=vol_by_ref,
                        end_ts=pick_column_rebal_ts,
                        rsr_series_by_row=rsr_for_hint,
                        rsm_series_by_row=rsm_for_hint,
                        rank_delta_by_row=pick_column_rank_delta,
                        change_pct_fn=_hint_change_pct,
                        series_at_fn=series_at,
                        curr_ranks=pick_column_curr_ranks,
                        prev_ranks=pick_column_prev_ranks,
                        top_n=n_port,
                        prev_holdings=prev_portfolio,
                        hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                        max_hold_rank=int(max_hold_rank_var.get()),
                    ),
                    rebal_n,
                )
            elif config.etf_recommend_profile == "stock":
                from momentum.stock.stock_rrg_pick_strategies import (
                    StockPickContext,
                    pick_shortfall_hint,
                )

                vol_by_ref = {
                    indices[j].upper().replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                    for j in range(len(indices))
                }
                pick_shortfall = pick_shortfall_hint(
                    _pick_strategy_key(),
                    StockPickContext(
                        ranked_row_indices=pick_column_ranked,
                        indices=indices,
                        ref_labels=index_metadata["ref_label"],
                        display_labels=index_metadata["display"],
                        vol_by_ref=vol_by_ref,
                        end_ts=pick_column_rebal_ts,
                        rsr_series_by_row=rsr_for_hint,
                        rsm_series_by_row=rsm_for_hint,
                        rank_delta_by_row=pick_column_rank_delta,
                        change_pct_fn=_hint_change_pct,
                        series_at_fn=series_at,
                        curr_ranks=pick_column_curr_ranks,
                        prev_ranks=pick_column_prev_ranks,
                        top_n=n_port,
                        prev_holdings=prev_portfolio,
                        hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                        max_hold_rank=int(max_hold_rank_var.get()),
                        benchmark=config.benchmark_nse,
                    ),
                    rebal_n,
                )
            else:
                from momentum.etf.us_rrg_pick_strategies import (
                    UsPickContext,
                    pick_shortfall_hint,
                )

                vol_by_ticker = {
                    indices[j]: etf_vol_by_row.get(j, 0.0)
                    for j in range(len(indices))
                }
                pick_shortfall = pick_shortfall_hint(
                    _pick_strategy_key(),
                    UsPickContext(
                        ranked_row_indices=pick_column_ranked,
                        indices=indices,
                        display_labels=index_metadata["display"],
                        vol_by_ticker=vol_by_ticker,
                        end_ts=pick_column_rebal_ts,
                        rsr_series_by_row=rsr_for_hint,
                        rsm_series_by_row=rsm_for_hint,
                        rank_delta_by_row=pick_column_rank_delta,
                        change_pct_fn=_hint_change_pct,
                        series_at_fn=series_at,
                        curr_ranks=pick_column_curr_ranks,
                        prev_ranks=pick_column_prev_ranks,
                        top_n=n_port,
                        prev_holdings=prev_portfolio,
                        hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                        max_hold_rank=int(max_hold_rank_var.get()),
                    ),
                    rebal_n,
                )
        was_week = (
            prev_panel_idx
            if prev_panel_idx is not None
            else (end_date_idx_local - 1 if bar_unit == "day" else panel_rebal_idx)
        )

        def _prices_for_pnl(sym: str) -> tuple[pd.Series, pd.Series | None]:
            weekly, daily = _etf_price_series(sym)
            return weekly, daily if len(daily) else None

        def _weekly_for_pnl(sym: str) -> pd.Series:
            return _prices_for_pnl(sym)[0]

        def _daily_for_pnl(sym: str) -> pd.Series | None:
            return _prices_for_pnl(sym)[1]

        def _was_rank(ticker: str) -> int | None:
            j = _row_index_for_ticker(ticker)
            if j is None:
                return None
            return _rank_by_row(was_week).get(j)

        def _curr_rank(ticker: str) -> int | None:
            j = _row_index_for_ticker(ticker)
            if j is None:
                return None
            ranks = (
                pick_column_curr_ranks if bar_unit == "day" else panel_curr_ranks
            )
            return ranks.get(j)

        as_of_ts = _portfolio_panel_as_of_ts()
        slider_idx = int(date_scale.get())
        live_close = live_close_for_panel(
            config.etf_recommend_profile,
            etf_daily_close=_etf_daily_close,
            at_latest_bar=(
                slider_idx >= _date_slider_max_idx()
                or (
                    bar_unit == "week"
                    and _preview_today_enabled()
                    and _preview_latest_daily_ts() is not None
                )
            ),
        )
        rebal_slots = pad_rebal_slots(rebal_tickers, n_port)

        end_prev_holdings = (
            list(prev_portfolio)
            if bar_unit == "day"
            else _end_prev_week_holdings(prev_panel_idx)
        )

        def _us_rebal_name_detail(ticker: str) -> str:
            j = _row_index_for_ticker(ticker)
            if j is None:
                return ""
            return index_metadata["display"][j] or ""

        panel_rows, panel_totals = build_portfolio_panel(
            prev_portfolio=prev_portfolio,
            rebal_strategy=rebal_strategy,
            rebal_tickers=rebal_slots,
            end_prev_week_holdings=end_prev_holdings,
            panel_exits=week_exits,
            rebalance_ts=panel_rebal_ts,
            prev_rebalance_ts=prev_rebal_ts,
            as_of_ts=as_of_ts,
            weekly_for_ticker=_weekly_for_pnl,
            daily_for_ticker=_daily_for_pnl,
            was_rank_for_ticker=_was_rank,
            curr_rank_for_ticker=_curr_rank,
            exit_below_9ema=bool(exit_below_9ema_var.get()),
            mid_week_9ema=(
                [] if bar_unit == "day" else _mid_week_9ema_cache.get(panel_rebal_idx, [])
            ),
            live_close_by_ticker=live_close,
            portfolio_slots=n_port,
            rebal_detail_for_ticker=(
                _us_rebal_name_detail
                if config.etf_recommend_profile == "us"
                else None
            ),
        )
        panel_mode = "day" if bar_unit == "day" else "week"
        movers_dates_label.config(
            text=portfolio_panel_dates_line(
                rebalance_label=rebalance_label,
                was_n=was_n,
                was_label=was_label,
                rebal_n=rebal_n,
                pick_shortfall=pick_shortfall,
                exit_below_9ema=bool(exit_below_9ema_var.get()),
                subtitle=subtitle,
                exits_through_label=(
                    rebalance_label
                    if bar_unit == "day"
                    else (
                        rrg_format_date(_preview_latest_daily_ts())
                        if _preview_today_enabled()
                        and _preview_latest_daily_ts() is not None
                        else format_date_label(panel_end_idx)
                    )
                ),
                mode=panel_mode,
            )
        )
        if movers_totals_label is not None:
            movers_totals_label.config(
                text=portfolio_panel_totals_line(
                    panel_totals,
                    live_pick=bool(live_close),
                    mode=panel_mode,
                )
            )
        n_slots = len(movers_row_widgets)
        max_rows = max(len(panel_rows), 1)
        for slot in range(n_slots):
            widgets = movers_row_widgets[slot]
            was_grid_row = slot + 1
            pick_grid_row = slot + 1
            if slot < len(panel_rows):
                row = panel_rows[slot]
                was_text = row["was_text"]
                now_text = row["now_text"]
                move = row["move"]
                rebal_text = row["rebal_text"]
                pick_tag = row["pick"]
                was_pnl_text = row.get("was_pnl", "")
                was_entry_text = row.get("was_entry", "")
                was_close_text = row.get("was_close", "")
                pick_pnl_text = row.get("pick_pnl", "")
                pick_entry_text = row.get("pick_entry", "")
                pick_close_text = row.get("pick_close", "")
                mid_9ema_text = row.get("mid_9ema", "")
                now_fg = row.get("now_fg", "black")
                rebal_fg = row.get("rebal_fg", "black")
                mid_fg = row.get("mid_fg", "black")
            else:
                was_text = now_text = move = rebal_text = pick_tag = ""
                was_pnl_text = was_entry_text = was_close_text = ""
                pick_pnl_text = pick_entry_text = pick_close_text = ""
                mid_9ema_text = ""
                now_fg = rebal_fg = mid_fg = "black"
            if slot < max_rows:
                rank_text = str(slot + 1)
                widgets["rank"].config(text=rank_text)
                widgets["pick_rank"].config(text=rank_text)
                widgets["was"].config(text=was_text)
                widgets["now"].config(text=now_text, fg=now_fg)
                widgets["tag"].config(text=move)
                widgets["rebal"].config(text=rebal_text, fg=rebal_fg)
                widgets["pick_tag"].config(text=pick_tag)
                widgets["was_pnl"].config(text=was_pnl_text)
                widgets["was_entry"].config(text=was_entry_text)
                widgets["was_close"].config(text=was_close_text)
                widgets["pick_pnl"].config(text=pick_pnl_text)
                widgets["pick_entry"].config(text=pick_entry_text)
                widgets["pick_close"].config(text=pick_close_text)
                widgets["mid_9ema"].config(text=mid_9ema_text, fg=mid_fg)
                widgets["rank"].grid(
                    row=was_grid_row, column=0, sticky='ew', padx=2, pady=1
                )
                for col in range(1, PORTFOLIO_PANEL_NUM_COLS):
                    key = PORTFOLIO_PANEL_WAS_ROW[col][0]
                    widgets[key].grid(
                        row=was_grid_row, column=col, sticky='ew', padx=2, pady=1
                    )
                widgets["pick_rank"].grid(
                    row=pick_grid_row, column=0, sticky='ew', padx=2, pady=1
                )
                for col in range(1, PORTFOLIO_PANEL_NUM_COLS):
                    key = PORTFOLIO_PANEL_PICK_ROW[col - 1][0]
                    widgets[key].grid(
                        row=pick_grid_row, column=col, sticky='ew', padx=2, pady=1
                    )
            else:
                for key in widgets:
                    widgets[key].config(text="")
                for w in widgets.values():
                    w.grid_remove()

        tc = TableRegionCopy.for_window(root)
        if movers_was_copy_grid is not None:
            tc.sync_styles(movers_was_copy_grid)
        if movers_pick_copy_grid is not None:
            tc.sync_styles(movers_pick_copy_grid)
        if _sync_side_scroll is not None:
            root.after_idle(_sync_side_scroll)

    def _compute_picks_at_week(
        end_date_idx_local: int,
        ranked: list[int],
        end_ts,
        start_ts,
        rank_delta_by_row: dict[int, str],
        curr_ranks: dict[int, int],
        prev_ranks: dict[int, int],
        *,
        prev_holdings: list[str] | None = None,
        write_cache: bool = True,
        change_pct_fn: Callable[[int], float] | None = None,
        rsr_series_by_row: list | None = None,
        rsm_series_by_row: list | None = None,
    ):
        strategy = _pick_strategy_key()
        max_rank = int(max_hold_rank_var.get())
        top_n = _portfolio_top_n()
        holdings = list(prev_holdings or [])
        rsr_rows = rsr_series_by_row if rsr_series_by_row is not None else rsr_tickers
        rsm_rows = rsm_series_by_row if rsm_series_by_row is not None else rsm_tickers

        def _row_change_pct(j: int) -> float:
            if change_pct_fn is not None:
                return change_pct_fn(j)
            return _tail_change_pct(j, start_ts, end_ts)

        if config.etf_recommend_profile == "india":
            from momentum.etf.india_rrg_pick_strategies import (
                IndiaPickContext,
                pick_india_portfolio,
            )

            vol_by_ref = {
                (index_metadata["ref_label"][j] or indices[j])
                .upper()
                .replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                for j in range(len(indices))
            }
            ctx = IndiaPickContext(
                ranked_row_indices=ranked,
                indices=indices,
                ref_labels=index_metadata["ref_label"],
                display_labels=index_metadata["display"],
                vol_by_ref=vol_by_ref,
                end_ts=end_ts,
                rsr_series_by_row=rsr_rows,
                rsm_series_by_row=rsm_rows,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=_row_change_pct,
                series_at_fn=series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                top_n=top_n,
                prev_holdings=holdings,
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=max_rank,
            )
            picks = _filter_picks_by_tradable_ref(
                pick_india_portfolio(strategy, ctx), end_ts
            )

        elif config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import (
                StockPickContext,
                pick_stock_portfolio,
            )

            vol_by_ref = {
                indices[j].upper().replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                for j in range(len(indices))
            }
            ctx = StockPickContext(
                ranked_row_indices=ranked,
                indices=indices,
                ref_labels=index_metadata["ref_label"],
                display_labels=index_metadata["display"],
                vol_by_ref=vol_by_ref,
                end_ts=end_ts,
                rsr_series_by_row=rsr_rows,
                rsm_series_by_row=rsm_rows,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=_row_change_pct,
                series_at_fn=series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                top_n=top_n,
                prev_holdings=holdings,
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=max_rank,
                benchmark=config.benchmark_nse,
            )
            picks = pick_stock_portfolio(strategy, ctx)

        else:
            from momentum.etf.us_rrg_pick_strategies import (
                UsPickContext,
                pick_us_portfolio,
            )

            vol_by_ticker = {
                indices[j]: etf_vol_by_row.get(j, 0.0) for j in range(len(indices))
            }
            ctx = UsPickContext(
                ranked_row_indices=ranked,
                indices=indices,
                display_labels=index_metadata["display"],
                vol_by_ticker=vol_by_ticker,
                end_ts=end_ts,
                rsr_series_by_row=rsr_rows,
                rsm_series_by_row=rsm_rows,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=_row_change_pct,
                series_at_fn=series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                top_n=top_n,
                prev_holdings=holdings,
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=max_rank,
            )
            picks = _filter_picks_by_tradable_ref(
                pick_us_portfolio(strategy, ctx), end_ts
            )

        if write_cache and end_date_idx_local >= 0:
            prev_pick = (
                _prior_week_top_n_portfolio(end_date_idx_local)
                if end_date_idx_local > tail
                else []
            )
            if _portfolio_cache_enabled():
                stale = [k for k in _pick_holdings_cache if k > end_date_idx_local]
                for k in stale:
                    del _pick_holdings_cache[k]
                    _active_holdings_cache.pop(k, None)
                    _week_exits_cache.pop(k, None)
                    _mid_week_9ema_cache.pop(k, None)
            rebal_strategy = _rebal_tickers_table_order(picks, ranked)
            rebal_slots, dropped_pick = _rebal_slots_after_9ema(rebal_strategy, end_ts)
            _pick_holdings_cache[end_date_idx_local] = rebal_slots
            held_enter = [t for t in rebal_slots if t]
            next_ts = (
                _rrg_index[end_date_idx_local + 1]
                if end_date_idx_local + 1 < len(_rrg_index)
                else None
            )
            end_active, dropped_was, mid = _apply_9ema_week_path(
                held_enter, end_ts, next_ts, prev_holdings=prev_pick or None
            )
            _active_holdings_cache[end_date_idx_local] = end_active
            _week_exits_cache[end_date_idx_local] = _build_live_week_exits(
                prev_pick,
                held_enter,
                curr_ranks,
                end_ts,
                _merge_dropped_tickers(dropped_pick, dropped_was),
                mid,
            )
            _mid_week_9ema_cache[end_date_idx_local] = list(mid)
        return picks

    def _ref_price_weekly_for_picks() -> dict[str, pd.Series]:
        """Weekly ref/ticker closes used to validate picks (bhavcopy or Yahoo)."""
        from momentum.rrg_ref_price import weekly_map_from_daily

        if config.etf_recommend_profile == "india":
            _ensure_etf_daily_close()
            return weekly_map_from_daily(_etf_daily_close)
        if config.etf_recommend_profile == "us":
            _ensure_etf_daily_close()
            if _etf_daily_close:
                return weekly_map_from_daily(_etf_daily_close)
            return {
                indices[j]: indices_data[indices[j]]
                for j in range(len(indices))
                if len(indices_data.get(indices[j], pd.Series(dtype=float)))
            }
        return {}

    def _filter_picks_by_tradable_ref(picks, end_ts):
        from momentum.rrg_ref_price import filter_picks_with_ref_price

        weekly = _ref_price_weekly_for_picks()
        if not weekly:
            return picks
        return filter_picks_with_ref_price(picks, weekly, end_ts)

    def _portfolio_cache_enabled() -> bool:
        """Walk-forward portfolio state for Was/Now panel (all pick strategies)."""
        return bool(config.etf_table_extras)

    def _pick_state_cache_needed() -> bool:
        return _portfolio_cache_enabled()

    def _ensure_etf_daily_close() -> None:
        _load_etf_daily_close_data()

    def _rebal_slots_after_9ema(
        rebal_tickers: list[str], end_ts
    ) -> tuple[list[str], list[str]]:
        """Exclude below-9-EMA picks at rebalance (empty slot, not a rebalance target)."""
        if not exit_below_9ema_var.get():
            return list(rebal_tickers), []
        from momentum.rrg_ema_exit import apply_9ema_rebalance_slots

        _ensure_etf_daily_close()
        return apply_9ema_rebalance_slots(
            rebal_tickers,
            _etf_daily_close,
            end_ts,
            enabled=True,
        )

    def _merge_dropped_tickers(*groups: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for group in groups:
            for sym in group:
                bare = _norm_ticker(sym)
                if not bare or bare in seen:
                    continue
                seen.add(bare)
                out.append(sym)
        return out

    def _etf_price_series(ticker: str) -> tuple[pd.Series, pd.Series]:
        """ETF CM bhavcopy (daily + W-FRI); avoid index EOD mistaken for ETF NAV."""
        bare = ticker.strip().upper().replace(".NS", "")
        _ensure_etf_daily_close()
        daily = _etf_daily_close.get(bare, pd.Series(dtype=float))
        if len(daily):
            weekly = daily.resample("W-FRI").last().dropna()
            return weekly, daily.sort_index()
        j = _row_index_for_ticker(ticker)
        if j is not None and index_metadata["kind"][j] == "etf":
            weekly = indices_data[indices[j]]
            return weekly, pd.Series(dtype=float)
        return pd.Series(dtype=float), pd.Series(dtype=float)

    def _weekly_price_for_ticker(ticker: str) -> pd.Series:
        weekly, _daily = _etf_price_series(ticker)
        return weekly

    def _apply_9ema_week_path(
        rebal_tickers: list[str],
        end_ts,
        next_ts,
        *,
        through_ts=None,
        prev_holdings: list[str] | None = None,
    ) -> tuple[list[str], list[str], list]:
        from momentum.rrg_ema_exit import (
            rebalance_9ema_dropped,
            simulate_week_with_9ema_exits,
        )

        holdings = list(rebal_tickers)
        dropped: list[str] = []
        mid_week: list = []
        if not exit_below_9ema_var.get():
            return holdings, dropped, mid_week
        _ensure_etf_daily_close()
        holdings, dropped = rebalance_9ema_dropped(
            holdings, prev_holdings, _etf_daily_close, end_ts
        )
        if next_ts is None or pd.Timestamp(next_ts) <= pd.Timestamp(end_ts):
            return holdings, dropped, mid_week
        weekly = {}
        daily_map = {}
        for t in holdings:
            w, d = _etf_price_series(t)
            weekly[t] = w
            if len(d):
                daily_map[t] = d
        _, holdings, mid_week = simulate_week_with_9ema_exits(
            holdings,
            end_ts,
            next_ts,
            daily_map if daily_map else _etf_daily_close,
            weekly,
            _portfolio_top_n(),
            through_date=through_ts,
        )
        return holdings, dropped, mid_week

    def _ref_to_row_for_exits() -> dict[str, int]:
        if config.etf_recommend_profile == "india":
            from momentum.etf.india_rrg_pick_strategies import ref_to_row_index

            return ref_to_row_index(indices, index_metadata["ref_label"])
        if config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import ref_to_row_index

            return ref_to_row_index(indices)
        return {indices[j].strip().upper(): j for j in range(len(indices))}

    def _build_live_week_exits(
        prev_active: list[str],
        rebalance_holdings: list[str],
        curr_ranks: dict[int, int],
        end_ts,
        dropped_9ema: list[str],
        mid_week_9ema: list,
    ) -> list:
        from momentum.rrg_portfolio_exits import build_week_exits

        return build_week_exits(
            prev_holdings=prev_active,
            rebalance_holdings=rebalance_holdings,
            hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
            curr_ranks=curr_ranks,
            ref_to_j=_ref_to_row_for_exits(),
            max_hold_rank=int(max_hold_rank_var.get()),
            exit_below_9ema=bool(exit_below_9ema_var.get()),
            dropped_9ema_rebal=dropped_9ema,
            mid_week_9ema=mid_week_9ema,
            decision_date=end_ts,
        )

    def _refresh_exits_display(week_idx: int) -> None:
        if recommend_exits_label is None:
            return
        from momentum.rrg_portfolio_exits import format_exits_multiline

        exits = _week_exits_cache.get(week_idx, [])
        show = bool(
            exits
            or hold_until_rank_exit_var.get()
            or exit_below_9ema_var.get()
            or week_idx > _date_slider_min_idx()
        )
        if not show:
            recommend_exits_label.pack_forget()
            return
        title = "Exits this week (rule · when · detail):\n"
        body = (
            format_exits_multiline(exits, rebalance_ts=_rrg_index[week_idx])
            if exits
            else "No exits this week."
        )
        recommend_exits_label.config(text=title + body)
        recommend_exits_label.pack(fill=tk.X, pady=(6, 0))

    def _warm_pick_holdings_cache(target_week_idx: int) -> None:
        if not _pick_state_cache_needed():
            return
        start_idx = max(_date_slider_min_idx(), tail)
        for idx in range(start_idx, target_week_idx + 1):
            if idx in _pick_holdings_cache:
                continue
            prev_active = _active_holdings_cache.get(idx - 1, [])
            end_ts_i = _rrg_index[idx]
            start_ts_i = _rrg_index[idx - tail]
            curr = _rank_by_row(idx)
            prev_r = _rank_by_row(idx - 1) if idx > tail else {}
            ranked_i = sorted(
                range(len(indices)),
                key=lambda j: _tail_change_pct(j, start_ts_i, end_ts_i),
                reverse=True,
            )
            rank_delta_i: dict[int, str] = {}
            for j in ranked_i:
                rank_delta_i[j] = format_rank_delta(
                    curr.get(j, len(indices)), prev_r.get(j)
                )
            picks_i = _compute_picks_at_week(
                idx,
                ranked_i,
                end_ts_i,
                start_ts_i,
                rank_delta_i,
                curr,
                prev_r,
                prev_holdings=prev_active,
            )
            rebal_strategy = _rebal_tickers_table_order(picks_i, ranked_i)
            rebal_slots, dropped_pick = _rebal_slots_after_9ema(
                rebal_strategy, end_ts_i
            )
            _pick_holdings_cache[idx] = rebal_slots
            held_enter = [t for t in rebal_slots if t]
            next_ts = (
                _rrg_index[idx + 1] if idx + 1 < len(_rrg_index) else None
            )
            prev_pick = (
                list(_pick_holdings_cache[idx - 1])
                if idx > tail and idx - 1 in _pick_holdings_cache
                else []
            )
            end_active, dropped_was, mid = _apply_9ema_week_path(
                held_enter,
                end_ts_i,
                next_ts,
                prev_holdings=prev_pick or None,
            )
            _active_holdings_cache[idx] = end_active
            _week_exits_cache[idx] = _build_live_week_exits(
                prev_pick,
                held_enter,
                curr,
                end_ts_i,
                _merge_dropped_tickers(dropped_pick, dropped_was),
                mid,
            )
            _mid_week_9ema_cache[idx] = list(mid)

    def _strategy_picks_for_week(
        end_date_idx_local: int,
        ranked: list[int],
        end_ts,
        start_ts,
        rank_delta_by_row: dict[int, str],
        curr_ranks: dict[int, int],
        prev_ranks: dict[int, int],
    ):
        prev_holdings: list[str] = []
        if _portfolio_cache_enabled() and end_date_idx_local > 0:
            if end_date_idx_local - 1 not in _active_holdings_cache:
                _warm_pick_holdings_cache(end_date_idx_local - 1)
            if hold_until_rank_exit_var.get():
                if exit_below_9ema_var.get():
                    prev_holdings = list(
                        _active_holdings_cache.get(end_date_idx_local - 1, [])
                    )
                else:
                    prev_holdings = list(
                        _pick_holdings_cache.get(end_date_idx_local - 1, [])
                    )
        return _compute_picks_at_week(
            end_date_idx_local,
            ranked,
            end_ts,
            start_ts,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
            prev_holdings=prev_holdings,
        )

    def refresh_recommendations_panel(
        ranked: list[int],
        end_ts,
        start_ts,
        rank_delta_by_row: dict[int, str],
        curr_ranks: dict[int, int] | None = None,
        prev_ranks: dict[int, int] | None = None,
        picks=None,
    ):
        if not config.etf_table_extras or not recommend_row_widgets:
            return
        if not _preview_today_enabled():
            _sync_recommend_panel_visibility()
            return
        _sync_recommend_panel_visibility()
        preview_footnote = ""
        if picks is None:
            state = _preview_panel_state(_preview_ranking_context())
            if state is None:
                picks = []
            else:
                picks, preview_footnote = state
        if picks is None:
            picks = []
        curr_ranks = curr_ranks or {}
        prev_ranks = prev_ranks or {}
        if config.etf_recommend_profile == "india":
            from momentum.etf.india_rrg_recommendations import recommendation_row_bg
        elif config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_recommendations import recommendation_row_bg
        else:
            from momentum.etf.us_rrg_recommendations import recommendation_row_bg
        n_port = _portfolio_top_n()
        if config.etf_recommend_profile == "us":
            from momentum.etf.us_rrg_pick_strategies import pick_strategy_label
        elif config.etf_recommend_profile == "stock":
            from momentum.stock.stock_rrg_pick_strategies import pick_strategy_label
        else:
            from momentum.etf.india_rrg_pick_strategies import pick_strategy_label

        strategy_lbl = pick_strategy_label(
            _pick_strategy_key(),
            hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
            exit_below_9ema=bool(exit_below_9ema_var.get()),
        )
        ctx = _preview_ranking_context()
        if ctx is not None:
            preview_end = rrg_format_date(ctx[0])
            preview_start = rrg_format_date(ctx[1])
            tail_d = _preview_tail_trading_days()
            rebal_lbl = format_date_label(
                _panel_rebal_idx(_portfolio_panel_week_idx())
            )
            if recommend_title_label is not None:
                recommend_title_label.config(
                    text=(
                        f"Preview Top {n_port} = Day unit Top N @ {preview_end} "
                        f"(Tail {tail_d})"
                    )
                )
            dates_txt = (
                f"{strategy_lbl}  ·  Chg% {preview_start}→{preview_end}  ·  "
                f"same as Day slider @ {preview_end}  ·  "
                f"weekly rebalance {rebal_lbl} above"
            )
            if preview_footnote:
                dates_txt = f"{dates_txt}  ·  {preview_footnote}"
            if recommend_dates_label is not None:
                recommend_dates_label.config(text=dates_txt)
        else:
            if recommend_title_label is not None:
                recommend_title_label.config(
                    text=f"Preview Top {n_port} — daily data unavailable"
                )
            if recommend_dates_label is not None:
                recommend_dates_label.config(text=strategy_lbl)

        n_show = min(len(picks), n_port)
        row_h = 22
        for slot, widgets in enumerate(recommend_row_widgets):
            if slot >= n_show:
                for w in widgets.values():
                    w.config(text='', bg=root.cget('bg'), fg='black')
                    w.grid_remove()
                continue
            pick = picks[slot]
            bg = recommendation_row_bg(pick.quadrant)
            fg = rrg_row_fg_color(bg)
            grid_row = slot + 1
            widgets['rank'].config(text=str(pick.pick_rank), bg=bg, fg=fg)
            widgets['ticker'].config(text=pick.ticker, bg=bg, fg=fg)
            widgets['name'].config(text=pick.name, bg=bg, fg=fg)
            rd_fg = _rank_delta_fg(pick.rank_delta)
            widgets['rank_delta'].config(
                text=pick.rank_delta, bg=bg, fg=rd_fg if fg == 'black' else 'white'
            )
            widgets['change'].config(
                text=format_change_pct(pick.change_pct), bg=bg, fg=fg
            )
            for col, key in enumerate(
                ('rank', 'ticker', 'name', 'rank_delta', 'change')
            ):
                widgets[key].grid(
                    row=grid_row, column=col, sticky='ew', padx=1, pady=0
                )
        if recommend_rec_canvas is not None:
            recommend_rec_canvas.config(height=row_h * (n_show + 1) + 4)
        if recommend_copy_grid is not None:
            TableRegionCopy.for_window(root).sync_styles(recommend_copy_grid)
        if _sync_side_scroll is not None:
            root.after_idle(_sync_side_scroll)

    def _hide_stale_table_rows(visible_row_count: int):
        for i, widgets in enumerate(table_widgets):
            if i < visible_row_count:
                continue
            for key in widgets:
                if key == 'visible_cell':
                    continue
                widgets[key].grid_remove()
            widgets['visible_cell'].grid_remove()

    def refresh_table_ranking():
        """Reorder table rows by tail-window performance (best % change first)."""
        if not table_widgets or not len(_rrg_index):
            return

        end_date_idx_local = int(date_scale.get())
        end_ts = _rrg_index[end_date_idx_local]
        start_ts = _rrg_index[end_date_idx_local - tail]
        preview_ctx = _preview_ranking_context() if _preview_today_enabled() else None

        if preview_ctx is not None:
            (
                end_ts,
                start_ts,
                ranked,
                rank_delta_by_row,
                curr_ranks,
                prev_ranks,
            ) = preview_ctx
            rank_delta_texts = [rank_delta_by_row[j] for j in ranked]
        else:
            curr_ranks = _rank_by_row(end_date_idx_local)
            prev_ranks = (
                _rank_by_row(end_date_idx_local - 1)
                if end_date_idx_local > tail
                else {}
            )

            ranked = sorted(
                range(len(indices)),
                key=lambda j: _tail_change_pct(j, start_ts, end_ts),
                reverse=True,
            )

            rank_delta_texts = []
            rank_delta_by_row = {}
            for j in ranked:
                text = format_rank_delta(
                    curr_ranks.get(j, len(indices)),
                    prev_ranks.get(j),
                )
                rank_delta_texts.append(text)
                rank_delta_by_row[j] = text

        panel_week_idx = (
            _portfolio_panel_week_idx() if bar_unit == "week" else end_date_idx_local
        )
        if _portfolio_cache_enabled() and bar_unit != "day":
            _warm_pick_holdings_cache(panel_week_idx)

        # ★ / graph: week = rebalance picks; day = Day-unit Tail N picks.
        if bar_unit == "day":
            rebalance_picks = (
                _daily_top_n_picks_at(_rrg_index[end_date_idx_local]) or []
            )
        elif config.top_movers_panel:
            rebalance_picks = _panel_week_picks(panel_week_idx)
        else:
            rebalance_picks = _strategy_picks_for_week(
                end_date_idx_local,
                ranked,
                end_ts,
                start_ts,
                rank_delta_by_row,
                curr_ranks,
                prev_ranks,
            )
        preview_picks = (
            _preview_today_picks(preview_ctx)
            if _preview_today_enabled()
            else None
        )
        _current_pick_row_indices.clear()
        _current_pick_row_indices.update(p.row_idx for p in rebalance_picks)

        if config.etf_table_extras and pick_auto_show_var.get():
            for j in _current_pick_row_indices:
                if not checkbox_vars[j].get():
                    checkbox_vars[j].set(True)
                    if indices[j] not in indices_to_show:
                        indices_to_show.append(indices[j])

        from momentum.etf.us_rrg_recommendations import format_vol_pct

        for display_row, j in enumerate(ranked):
            w = table_widgets[j]
            index_name = indices[j]
            if preview_ctx is not None:
                chg = _daily_tail_change_at(j, start_ts, end_ts)
                px = _preview_row_mark_price(j, end_ts)
                price = round(px, 2) if px is not None else ''
            else:
                chg = _tail_change_pct(j, start_ts, end_ts)
                try:
                    price = round(series_at(indices_data[index_name], end_ts), 2)
                except (KeyError, TypeError, ValueError, IndexError):
                    price = ''
            try:
                rsr_val = series_at(rsr_tickers[j], end_ts)
                rsm_val = series_at(rsm_tickers[j], end_ts)
                bg_color = get_color(rsr_val, rsm_val)
            except (KeyError, TypeError, ValueError, IndexError):
                bg_color = RRG_COLOR_NA
            fg_color = rrg_row_fg_color(bg_color)

            rank_num = display_row + 1
            rank_delta_text = rank_delta_texts[display_row]
            rank_delta_fg = _rank_delta_fg(rank_delta_text)
            is_pick = j in _current_pick_row_indices
            rank_display = f"★{rank_num}" if is_pick else str(rank_num)
            for col, key in (
                (_COL_RANK, 'rank_label'),
                (_COL_RANK_DELTA, 'rank_delta_label'),
                (_COL_REF, 'ref_label'),
                (_COL_INDEX, 'index_entry'),
                (_COL_PRICE, 'price_label'),
                (_COL_CHANGE, 'chg_label'),
            ):
                w[key].grid(
                    row=display_row,
                    column=col,
                    sticky=_table_col_sticky(col),
                    padx=_TABLE_CELL_PADX,
                    pady=_TABLE_ROW_PADY,
                )
            if _etf_extras and _COL_VOL is not None and 'vol_label' in w:
                w['vol_label'].grid(
                    row=display_row,
                    column=_COL_VOL,
                    sticky=_table_col_sticky(_COL_VOL),
                    padx=_TABLE_CELL_PADX,
                    pady=_TABLE_ROW_PADY,
                )
            w['visible_cell'].grid(
                row=display_row,
                column=_COL_VISIBLE,
                sticky='nsew',
                padx=_TABLE_CELL_PADX,
                pady=_TABLE_ROW_PADY,
            )
            w['rank_label'].config(text=rank_display)
            w['rank_delta_label'].config(
                text=rank_delta_text, fg=rank_delta_fg if fg_color == 'black' else 'white'
            )
            w['ref_label'].config(text=index_metadata['ref_label'][j] or '-')
            w['index_entry'].delete(0, tk.END)
            w['index_entry'].insert(0, index_metadata['display'][j])
            w['price_label'].config(text=price)
            w['chg_label'].config(text=format_change_pct(chg))
            if _etf_extras and 'vol_label' in w:
                w['vol_label'].config(
                    text=format_vol_pct(etf_vol_by_row.get(j, 0.0))
                )
            style_keys = (
                'rank_label',
                'rank_delta_label',
                'ref_label',
                'index_entry',
                'price_label',
                'chg_label',
            )
            if _etf_extras and 'vol_label' in w:
                style_keys = (*style_keys, 'vol_label')
            for key in style_keys:
                w[key].config(bg=bg_color, fg=fg_color)
            if is_pick:
                pick_font = ('Arial', 10, 'bold')
                w['rank_label'].config(font=pick_font)
                w['ref_label'].config(font=pick_font)
            else:
                w['rank_label'].config(font=_TABLE_FONT)
                w['ref_label'].config(font=_TABLE_FONT)
            w['rank_delta_label'].config(
                fg=rank_delta_fg if fg_color == 'black' else 'white'
            )
        if main_table_copy_grid is not None:
            tc = TableRegionCopy.for_window(root)
            for display_row, j in enumerate(ranked):
                for c, key in enumerate(_main_copy_cols):
                    if key in table_widgets[j]:
                        TableRegionCopy.set_cell_pos(
                            table_widgets[j][key], display_row, c
                        )
            tc.sync_styles(main_table_copy_grid)
        _hide_stale_table_rows(len(indices))
        _update_table_column_widths(end_ts, start_ts, rank_delta_texts)
        refresh_top_movers_panel(
            ranked,
            picks=rebalance_picks,
            end_ts=end_ts,
            start_ts=start_ts,
            rank_delta_by_row=rank_delta_by_row,
            curr_ranks=curr_ranks,
            prev_ranks=prev_ranks,
        )
        refresh_recommendations_panel(
            ranked,
            end_ts,
            start_ts,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
            picks=preview_picks,
        )
        _update_preview_status_label()

    checkbox_vars = []
    table_widgets = []
    _main_copy_cols = [
        'rank_label',
        'rank_delta_label',
        'ref_label',
        'index_entry',
        'price_label',
        'chg_label',
    ]
    if _etf_extras:
        _main_copy_cols.append('vol_label')

    main_table_copy_grid = None

    def _build_table_rows() -> None:
        nonlocal main_table_copy_grid, end_date, start_date
        for w in table_widgets:
            for widget in w.values():
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
        for child in table_body.winfo_children():
            child.destroy()
        checkbox_vars.clear()
        table_widgets.clear()

        end_date = _rrg_index[end_date_idx]
        start_date = _rrg_index[end_date_idx - tail]

        for i in range(len(indices)):
            row_id = indices[i]
            display_label = index_metadata['display'][i]
            ref_label = index_metadata['ref_label'][i]
            price = round(float(indices_data[row_id].loc[end_date]), 2)
            chg = (
                float(indices_data[row_id].loc[end_date])
                - float(indices_data[row_id].loc[start_date])
            ) / float(indices_data[row_id].loc[start_date]) * 100
            bg_color = get_color(rsr_tickers[i].iloc[-1], rsm_tickers[i].iloc[-1])
            fg_color = rrg_row_fg_color(bg_color)
            index_var = tk.StringVar(value=display_label)
            rank_label = tk.Label(
                table_body,
                text=i + 1,
                relief=tk.RIDGE,
                anchor='e',
                bg=bg_color,
                fg=fg_color,
                font=_TABLE_FONT,
            )
            rank_delta_label = tk.Label(
                table_body,
                text='—',
                relief=tk.RIDGE,
                anchor='e',
                bg=bg_color,
                fg=fg_color,
                font=_TABLE_FONT,
            )
            ref_label_widget = tk.Label(
                table_body,
                text=ref_label,
                relief=tk.RIDGE,
                anchor='w',
                bg=bg_color,
                fg=fg_color,
                font=_TABLE_FONT,
            )
            index_entry = tk.Entry(
                table_body,
                textvariable=index_var,
                relief=tk.RIDGE,
                bg=bg_color,
                fg=fg_color,
                font=_TABLE_FONT,
            )
            index_entry._row_idx = i
            index_entry.bind('<Return>', update_entry)
            price_label = tk.Label(
                table_body,
                text=price,
                relief=tk.RIDGE,
                anchor='e',
                bg=bg_color,
                fg=fg_color,
                font=_TABLE_FONT,
            )
            chg_label = tk.Label(
                table_body,
                text=format_change_pct(chg),
                relief=tk.RIDGE,
                anchor='e',
                bg=bg_color,
                fg=fg_color,
                font=_TABLE_FONT,
            )
            vol_label = None
            if _etf_extras:
                vol_label = tk.Label(
                    table_body,
                    text='',
                    relief=tk.RIDGE,
                    anchor='e',
                    bg=bg_color,
                    fg=fg_color,
                    font=_TABLE_FONT,
                )
            visible_cell = tk.Frame(
                table_body,
                relief=tk.RIDGE,
                bg=_TABLE_NEUTRAL_BG,
                highlightthickness=0,
            )
            checkbox_var = tk.BooleanVar(value=indices[i] in indices_to_show)
            checkbox_vars.append(checkbox_var)
            checkbox = ttk.Checkbutton(
                visible_cell,
                variable=checkbox_var,
                command=lambda idx=i: on_visibility_toggle(idx),
            )
            checkbox.pack(anchor='center', expand=True)
            row_widgets = {
                'rank_label': rank_label,
                'rank_delta_label': rank_delta_label,
                'ref_label': ref_label_widget,
                'index_entry': index_entry,
                'price_label': price_label,
                'chg_label': chg_label,
                'visible_cell': visible_cell,
            }
            if vol_label is not None:
                row_widgets['vol_label'] = vol_label
            table_widgets.append(row_widgets)

        _tc = TableRegionCopy.for_window(root)
        main_table_copy_grid = _tc.register_grid(
            [[w[k] for k in _main_copy_cols] for w in table_widgets]
        )

        select_all_cb.config(command=on_select_all_toggle)
        refresh_table_ranking()
        _sync_select_all_checkbox()
        root.update_idletasks()
        _sync_header_scroll_gutter()
        _sync_table_layout()
        root.after_idle(_sync_table_layout)

    _build_table_rows()

    scatter_plots = [None] * len(indices)
    line_plots = [None] * len(indices)
    head_arrows = [None] * len(indices)
    annotations = [None] * len(indices)

    def _ensure_plot_slots():
        n = len(indices)
        for bucket in (scatter_plots, line_plots, head_arrows, annotations):
            while len(bucket) < n:
                bucket.append(None)

    def _resolve_end_date_idx(prev_ts, date_min: int, date_max: int) -> int:
        """Map a calendar end timestamp onto the new Date index."""
        if prev_ts is None or not len(_rrg_index):
            return date_max
        pos = _rrg_index.get_indexer([pd.Timestamp(prev_ts)], method="ffill")
        if pos[0] < 0:
            return date_max
        return max(date_min, min(int(pos[0]), date_max))

    def _rebuild_rrg_for_bar_unit(preserve_calendar_end: bool = True) -> None:
        """Reload prices/RRG and refresh Date slider after week/day switch."""
        nonlocal _rrg_index, end_date_idx, effective_window, min_history_bars, nav_bars

        prev_end_ts = None
        if preserve_calendar_end and end_date_idx is not None and len(_rrg_index):
            prev_end_ts = _rrg_index[end_date_idx]

        if bar_unit == "day":
            _history_cache.pop("day", None)
            _invalidate_daily_pick_rrg()
            _load_etf_daily_close_data(force=True)

        _apply_price_histories(_histories_for_unit(bar_unit))
        if bar_unit == "day":
            _invalidate_daily_pick_rrg()
        rs_tickers.clear()
        rsr_tickers.clear()
        rsr_roc_tickers.clear()
        rsm_tickers.clear()
        for i in range(len(indices)):
            name = indices[i]
            rsr, rsr_roc, rsm = compute(
                indices_data[name], benchmark_data, effective_window
            )
            if rsr is None:
                continue
            rs_tickers.append(100 * (indices_data[name] / benchmark_data))
            rsr_tickers.append(rsr)
            rsr_roc_tickers.append(rsr_roc)
            rsm_tickers.append(rsm)
        if not rsr_tickers:
            raise SystemExit(
                f"No RRG rows with enough {bar_unit} history. Check data downloads."
            )
        indices[:] = indices[: len(rsr_tickers)]
        indices_to_show[:] = [n for n in indices_to_show if n in indices]
        index_metadata['ref_label'] = index_metadata['ref_label'][: len(indices)]
        index_metadata['display'] = index_metadata['display'][: len(indices)]
        index_metadata['kind'] = index_metadata['kind'][: len(indices)]

        _rrg_index = _build_rrg_date_index()
        if not len(_rrg_index):
            raise SystemExit("No RRG dates available for the selected bar unit.")

        date_min = _date_slider_min_idx()
        date_max = _date_slider_max_idx()
        date_scale.config(from_=date_min, to=date_max)
        if bar_unit == "day":
            end_date_idx = date_max
        else:
            end_date_idx = _resolve_end_date_idx(prev_end_ts, date_min, date_max)
        date_scale.set(end_date_idx)
        date_value_label.config(text=format_date_label(end_date_idx))
        date_range_label.config(text=_date_range_hint_text())
        update_calc_context_label()
        _ensure_plot_slots()
        slider_last = (
            rrg_format_date(_rrg_index[-1]) if len(_rrg_index) else "n/a"
        )
        print(
            f"RRG bar unit: {bar_unit} — {len(indices)} rows, "
            f"{len(_rrg_index)} dates on slider, end {format_date_label(end_date_idx)} "
            f"(latest bar {slider_last})"
        )

    def _reload_us_universe() -> None:
        nonlocal config, end_date, start_date, _rrg_index, end_date_idx
        from dataclasses import replace

        from momentum.etf.us_rrg_universe_modes import (
            US_UNIVERSE_LABELS,
            build_us_rrg_config,
        )

        if not config.us_universe_switchable:
            return
        prev_mode = config.backtest_universe_mode
        mode = _us_universe_mode_key()
        if mode == prev_mode:
            return

        print(f"RRG: switching US ETF universe to {mode}...")
        try:
            _busy.show(f"Switching to {mode} universe…")
            new_cfg = build_us_rrg_config(
                mode,
                period=period,
                rrg_window=window,
                min_adv=config.backtest_min_adv,
                vol_percentile=config.backtest_vol_percentile,
                categories=config.backtest_categories,
            )
            config = replace(
                new_cfg,
                pick_strategy=_pick_strategy_key(),
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=int(max_hold_rank_var.get()),
                exit_below_9ema=bool(exit_below_9ema_var.get()),
            )
            requested_indices.clear()
            requested_indices.extend(row.row_id for row in config.rows)
            if default_indices_var.get():
                indices_to_show[:] = [
                    n for n in requested_indices if n in config.default_visible_ids
                ]
            else:
                indices_to_show[:] = list(requested_indices)

            _history_cache.clear()
            _etf_daily_close.clear()
            etf_vol_by_row.clear()
            _clear_pick_cache()

            prev_end_ts = (
                _rrg_index[end_date_idx]
                if end_date_idx is not None and len(_rrg_index)
                else None
            )
            was_near_latest = (
                end_date_idx is not None
                and len(_rrg_index) > 0
                and end_date_idx >= len(_rrg_index) - 2
            )

            if bar_unit == "day":
                _load_etf_daily_close_data(force=True)

            _apply_price_histories(_histories_for_unit(bar_unit))
            rs_tickers.clear()
            rsr_tickers.clear()
            rsr_roc_tickers.clear()
            rsm_tickers.clear()
            for i in range(len(indices)):
                name = indices[i]
                rsr, rsr_roc, rsm = compute(
                    indices_data[name], benchmark_data, effective_window
                )
                if rsr is None:
                    continue
                rs_tickers.append(100 * (indices_data[name] / benchmark_data))
                rsr_tickers.append(rsr)
                rsr_roc_tickers.append(rsr_roc)
                rsm_tickers.append(rsm)
            if not rsr_tickers:
                raise RuntimeError(
                    "No RRG rows with enough history in the selected universe."
                )
            indices[:] = indices[: len(rsr_tickers)]
            indices_to_show[:] = [n for n in indices_to_show if n in indices]
            index_metadata['ref_label'] = [
                config.row_ref_label(name) for name in indices
            ]
            index_metadata['display'] = [
                config.row_display_label(name) for name in indices
            ]
            index_metadata['kind'] = [config.row_kind(name) for name in indices]

            _rrg_index = _build_rrg_date_index()
            if not len(_rrg_index):
                raise RuntimeError("No RRG dates after universe switch.")

            date_min = _date_slider_min_idx()
            date_max = _date_slider_max_idx()
            date_scale.config(from_=date_min, to=date_max)
            end_date_idx = (
                date_max
                if bar_unit == "day" and was_near_latest
                else _resolve_end_date_idx(prev_end_ts, date_min, date_max)
            )
            date_scale.set(end_date_idx)
            date_value_label.config(text=format_date_label(end_date_idx))
            date_range_label.config(text=_date_range_hint_text())
            update_calc_context_label()

            if config.etf_table_extras and indices:
                try:
                    from momentum.etf.us_liquid_screener import _fetch_metrics

                    vol_metrics = _fetch_metrics(
                        list(indices),
                        adv_days=20,
                        vol_days=63,
                        history_days=120,
                        quiet=True,
                    )
                    for j, sym in enumerate(indices):
                        if sym in vol_metrics:
                            etf_vol_by_row[j] = vol_metrics[sym][1]
                except Exception as exc:
                    print(f"Vol% load skipped: {exc}")

            _load_etf_daily_close_data(force=True)

            scatter_plots[:] = [None] * len(indices)
            line_plots[:] = [None] * len(indices)
            head_arrows[:] = [None] * len(indices)
            annotations[:] = [None] * len(indices)
            _ensure_plot_slots()
            _build_table_rows()
            root.title(config.window_title)
            us_universe_var.set(US_UNIVERSE_LABELS[mode])
            print(config.universe_summary)
            redraw_chart()
        except Exception as exc:
            us_universe_var.set(US_UNIVERSE_LABELS.get(prev_mode, us_universe_var.get()))
            messagebox.showerror("Universe switch failed", str(exc), parent=root)
        finally:
            _busy.hide()

    def on_bar_unit_change(_event=None):
        nonlocal bar_unit
        selected = rrg_normalize_bar_unit(bar_unit_var.get())
        if selected == bar_unit:
            return
        print(f"RRG: loading {selected} bar data...")
        unit_lbl = "daily" if selected == "day" else "weekly"
        with _busy.busy(f"Loading {unit_lbl} bar data…"):
            bar_unit = selected
            _clear_pick_cache()
            _sync_preview_pick_ui()
            jump_latest = selected == "day"
            _rebuild_rrg_for_bar_unit(preserve_calendar_end=not jump_latest)
            update_nav_button_labels()
            redraw_chart()
            root.update_idletasks()

    bar_unit_combo.bind('<<ComboboxSelected>>', on_bar_unit_change)
    if us_universe_combo is not None:
        us_universe_combo.bind('<<ComboboxSelected>>', lambda _e: _reload_us_universe())
    update_nav_button_labels()

    def remove_ticker_artists(j):
        for artists in (scatter_plots, line_plots, head_arrows, annotations):
            artist = artists[j]
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass
                artists[j] = None

    def redraw_chart():
        update_frame()
        update_week_step_buttons()
        update_tail_step_buttons()
        update_nav_button_labels()
        if root.winfo_exists():
            try:
                canvas.draw_idle()
            except tk.TclError:
                pass

    def update_frame():
        nonlocal start_date, end_date, end_date_idx, hover_points, _last_hover_idx

        if not root.winfo_exists():
            return

        hover_points = []
        _last_hover_idx = None
        _hide_hover_tooltip()

        end_date_idx = int(date_scale.get())
        end_date = _rrg_index[end_date_idx]
        start_date = _rrg_index[end_date_idx - tail]
        update_calc_context_label()

        refresh_table_ranking()

        if not show_rrg_var.get():
            return

        for j in range(len(indices)):
            remove_ticker_artists(j)

            is_pick = j in _current_pick_row_indices
            on_graph = indices[j] in indices_to_show or (
                config.etf_table_extras
                and pick_auto_show_var.get()
                and is_pick
            )
            if not on_graph:
                continue

            filtered_rsr_tickers = rsr_tickers[j].loc[
                (rsr_tickers[j].index >= start_date) & (rsr_tickers[j].index <= end_date)
            ]
            filtered_rsm_tickers = rsm_tickers[j].loc[
                (rsm_tickers[j].index >= start_date) & (rsm_tickers[j].index <= end_date)
            ]
            if filtered_rsr_tickers.empty:
                continue
            _append_hover_points(j, filtered_rsr_tickers, filtered_rsm_tickers)
            plot_color = get_chart_color(
                filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1]
            )
            xs = filtered_rsr_tickers.values
            ys = filtered_rsm_tickers.values
            line_w = 3.0 if is_pick else 1.4
            line_alpha = 0.9 if is_pick else 0.55
            marker_size = TAIL_MARKER_SIZE + 14 if is_pick else TAIL_MARKER_SIZE
            if len(xs) >= 2:
                scatter_plots[j] = ax_rrg.scatter(
                    xs[:-1],
                    ys[:-1],
                    color=plot_color,
                    s=_tail_marker_sizes(len(xs) - 1, base=marker_size),
                    zorder=5 if is_pick else 4,
                    edgecolors='#333' if is_pick else 'none',
                    linewidths=0.8 if is_pick else 0,
                )
                head_arrows[j] = _add_head_arrow(xs, ys, plot_color)
            else:
                scatter_plots[j] = ax_rrg.scatter(
                    xs,
                    ys,
                    color=plot_color,
                    s=_tail_marker_sizes(len(xs), base=marker_size),
                    zorder=5 if is_pick else 4,
                    edgecolors='#333' if is_pick else 'none',
                    linewidths=0.8 if is_pick else 0,
                )
                head_arrows[j] = None
            line_plots[j] = ax_rrg.plot(
                xs, ys, color=plot_color, alpha=line_alpha, linewidth=line_w, zorder=3 if is_pick else 2
            )[0]
            annotations[j] = ax_rrg.annotate(
                index_metadata['display'][j],
                (filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1]),
                fontsize=9 if is_pick else 8,
                color=plot_color,
                fontweight='bold' if is_pick else 'medium',
            )

    def on_close():
        plt.close(fig)
        root.quit()
        root.destroy()

    def _on_preview_toggle(*_) -> None:
        if not config.preview_today_picks:
            return
        if preview_today_picks_var.get() and bar_unit == "week":
            with _busy.busy("Loading preview data…"):
                _clear_pick_cache()
                _sync_recommend_panel_visibility()
                _update_preview_status_label()
                redraw_chart()
        else:
            _clear_pick_cache()
            _sync_recommend_panel_visibility()
            _update_preview_status_label()
            redraw_chart()

    root.protocol('WM_DELETE_WINDOW', on_close)
    install_copy_support(root)
    if config.etf_table_extras:

        def _on_pick_controls_change(*_) -> None:
            _clear_pick_cache()
            _toggle_max_hold_rank_ui()
            redraw_chart()

        pick_strategy_var.trace_add("write", _on_pick_controls_change)
        hold_until_rank_exit_var.trace_add("write", _on_pick_controls_change)
        exit_below_9ema_var.trace_add("write", _on_pick_controls_change)
        max_hold_rank_var.trace_add("write", _on_pick_controls_change)
        if portfolio_n_var is not None:
            portfolio_n_var.trace_add("write", _on_pick_controls_change)
        if pick_auto_show_cb is not None:
            pick_auto_show_cb.config(command=redraw_chart)
        if hold_rank_cb is not None:
            hold_rank_cb.config(command=_on_pick_controls_change)
        _toggle_max_hold_rank_ui()
        _sync_preview_pick_ui()
    if config.preview_today_picks and preview_today_cb is not None:
        preview_today_picks_var.trace_add("write", _on_preview_toggle)
    if _sync_side_scroll is not None:
        root.after_idle(_sync_side_scroll)
    redraw_chart()
    root.update_idletasks()
    root.mainloop()
