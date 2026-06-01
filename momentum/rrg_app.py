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
from tkinter import ttk

from momentum.rrg_portfolio_panel import (
    PORTFOLIO_PANEL_GRID_KEYS,
    PORTFOLIO_PANEL_HEADERS,
    build_portfolio_panel,
    format_portfolio_cell,
    norm_ticker as _norm_ticker,
    portfolio_panel_dates_line,
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
    etf_recommend_profile: str = "us"  # "us" | "india"
    etf_recommend_count: int = 7
    etf_recommend_title: str = "Recommended Top 7 (weekly swing)"
    pick_strategy: str = "recommend"
    hold_until_rank_exit: bool = False
    max_hold_rank: int = 10
    exit_below_9ema: bool = True
    backtest_enabled: bool = False
    backtest_profile: str = "india"  # "india" | "us"
    backtest_universe_mode: str = "expanded"  # US: "core" | "expanded"
    backtest_min_adv: float = 10_000_000.0
    backtest_vol_percentile: float = 100.0
    backtest_categories: tuple[str, ...] = ("all",)


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

    def _histories_for_unit(unit: str) -> dict[str, pd.Series]:
        u = rrg_normalize_bar_unit(unit)
        if u not in _history_cache:
            min_pts = rrg_min_history_bars(window, u)
            _history_cache[u] = config.load_all_histories(
                period, min_pts, window, freq=u
            )
        return _history_cache[u]

    def _build_rrg_date_index():
        """Bar dates for the Date slider (analysis window + tail buffer)."""
        bench = benchmark_data.dropna().sort_index()
        warmup = rrg_warmup_bars(window, bar_unit)
        slider_bars = rrg_slider_index_bars(period, tail=RRG_MAX_TAIL, unit=bar_unit)
        if len(bench) > warmup:
            cal = bench.index[warmup:]
        elif not rsm_tickers:
            return pd.Index([])
        else:
            longest = max(rsm_tickers, key=lambda s: len(s.index))
            cal = longest.index.sort_values()
        if len(cal) > slider_bars:
            cal = cal[-slider_bars:]
        return cal

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
        if config.etf_recommend_profile == "india":
            print("Loading 63-day Vol% for India ETF table...")
            try:
                from momentum.etf.india_rrg_recommendations import load_india_etf_vol_pct

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
                    etf_vol_by_row[j] = vol_by_ref.get(ref, 0.0)
            except Exception as exc:
                print(f"Vol% load skipped: {exc}")
        else:
            print("Loading 63-day Vol% for US ETF table...")
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

    _etf_daily_close: dict[str, pd.Series] = {}

    def _load_etf_daily_close_data() -> None:
        """CM/Yahoo daily closes for ETF rows (9 EMA, exit P&L, portfolio panel)."""
        if _etf_daily_close:
            return
        from datetime import timedelta

        from utils.nse_bhavcopy import today_ist

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
            for sym, series in batch.items():
                if len(series):
                    _etf_daily_close[sym] = series.sort_index()
        else:
            from utils.yahoo_weekly import load_yahoo_histories_range

            tickers = [t for t in indices if t != config.benchmark_nse]
            print(f"Loading daily prices for {len(tickers)} ticker(s)...")
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
                    _etf_daily_close[sym] = series.sort_index()

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

    root = tk.Tk()
    root.withdraw()
    root.title(config.window_title)
    root.geometry('1600x900' if use_right_extras else ('1400x900' if config.top_movers_panel else '1100x900'))
    root.minsize(1280 if use_right_extras else (1280 if config.top_movers_panel else 900), 650)
    root.resizable(True, True)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

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
                    'date': str(date).split(' ')[0],
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
        return str(_rrg_index[int(idx)]).split(' ')[0]

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

    week_nav_frame = tk.Frame(controls_frame)
    week_nav_frame.pack(side=tk.LEFT, padx=(0, 12), anchor='n')
    prev_week_button = ttk.Button(
        week_nav_frame, text='Previous Week', command=step_previous_week
    )
    prev_week_button.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
    next_week_button = ttk.Button(week_nav_frame, text='Next Week', command=step_next_week)
    next_week_button.pack(side=tk.TOP, fill=tk.X)

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
    _pick_label_to_key: dict[str, str] = {}

    def _portfolio_top_n() -> int:
        if portfolio_n_var is not None:
            try:
                return max(1, min(int(portfolio_n_var.get()), _PORTFOLIO_N_MAX))
            except (tk.TclError, ValueError):
                pass
        return config.etf_recommend_count

    if config.etf_table_extras:
        from momentum.etf.india_rrg_pick_strategies import (
            PICK_STRATEGIES,
            pick_strategy_subtitle,
        )

        _pick_label_to_key = {label: key for key, label in PICK_STRATEGIES.items()}
        pick_strategy_var.set(PICK_STRATEGIES.get(config.pick_strategy, PICK_STRATEGIES["recommend"]))

        def _pick_strategy_key() -> str:
            return _pick_label_to_key.get(pick_strategy_var.get(), "recommend")

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
        portfolio_n_var = None

    if config.backtest_enabled:
        def open_backtest():
            from momentum.rrg_backtest_ui import open_rrg_backtest

            open_rrg_backtest(
                root,
                profile=config.backtest_profile,
                rrg_window=window,
                tail=int(float(tail_scale.get())),
                top_n=_portfolio_top_n(),
                backtest_extra={
                    **(
                        {
                            "universe_mode": config.backtest_universe_mode,
                            "universe_row_ids": tuple(indices),
                            "min_adv_usd": config.backtest_min_adv,
                            "vol_percentile": config.backtest_vol_percentile,
                            "screen_categories": config.backtest_categories,
                        }
                        if config.backtest_profile == "us"
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
    movers_exits_label = None
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
        nonlocal movers_panel, movers_title_label, movers_dates_label, movers_exits_label
        nonlocal movers_copy_grid
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
        movers_dates_label.pack(fill=tk.X, pady=(0, 4))

        slot_count = max(
            config.top_movers_count, config.etf_recommend_count, _PORTFOLIO_N_MAX
        )
        movers_table = tk.Frame(movers_panel)
        movers_table.pack(fill=tk.X)
        movers_header_cells: list[tk.Label] = []
        for col, (_key, header, anchor, min_px) in enumerate(PORTFOLIO_PANEL_HEADERS):
            movers_table.columnconfigure(
                col,
                minsize=min_px,
                weight=1 if _key in ("was", "now", "rebal") else 0,
            )
            hdr = tk.Label(
                movers_table,
                text=header,
                font=('Arial', 9, 'bold'),
                anchor=anchor,
                relief=tk.RIDGE,
            )
            hdr.grid(row=0, column=col, sticky='ew', padx=2, pady=1)
            movers_header_cells.append(hdr)

        movers_body_cells: list[list[tk.Label]] = []
        for slot in range(slot_count):
            widgets: dict[str, tk.Label] = {}
            grid_row = slot + 1
            row_cells: list[tk.Label] = []
            for col, (key, _header, anchor, _min_px) in enumerate(PORTFOLIO_PANEL_HEADERS):
                font = ('Arial', 8) if key in ('tag', 'pick_tag', 'mid_9ema') else ('Arial', 9)
                fg = '#1565C0' if key in ('tag', 'pick_tag') else 'black'
                lbl = tk.Label(
                    movers_table,
                    font=font,
                    anchor=anchor,
                    relief=tk.RIDGE,
                    fg=fg,
                )
                lbl.grid(row=grid_row, column=col, sticky='ew', padx=2, pady=1)
                widgets[key] = lbl
                row_cells.append(lbl)
            movers_row_widgets.append(widgets)
            movers_body_cells.append(row_cells)

        movers_copy_grid = TableRegionCopy.for_window(root).register_grid(
            [movers_header_cells, *movers_body_cells]
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
    recommend_dates_label = None
    recommend_row_widgets: list[dict[str, tk.Label]] = []
    recommend_copy_grid: dict | None = None
    recommend_exits_label: tk.Label | None = None
    main_table_copy_grid: dict | None = None
    movers_copy_grid: dict | None = None

    def _build_recommend_panel(parent: tk.Frame, *, pack_mode: str | None) -> None:
        nonlocal recommend_panel, recommend_dates_label, recommend_copy_grid
        nonlocal recommend_exits_label
        recommend_panel = tk.Frame(
            parent,
            padx=6,
            pady=4,
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

        tk.Label(
            recommend_panel,
            text=config.etf_recommend_title,
            font=('Arial', 10, 'bold'),
            anchor='w',
        ).pack(fill=tk.X)
        recommend_dates_label = tk.Label(
            recommend_panel,
            text="Leading/Improving · Rank Δ>0 · momentum+reliability score",
            font=('Arial', 9),
            anchor='w',
            fg='gray',
            justify=tk.LEFT,
        )
        recommend_dates_label.pack(fill=tk.X, pady=(0, 4))

        rec_scroll_x = tk.Scrollbar(recommend_panel, orient=tk.HORIZONTAL)
        rec_canvas = tk.Canvas(
            recommend_panel,
            highlightthickness=0,
            height=24 * (_PORTFOLIO_N_MAX + 1) + 8,
            xscrollcommand=rec_scroll_x.set,
        )
        rec_scroll_x.config(command=rec_canvas.xview)
        rec_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        rec_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
            ('ticker', 'Ticker', 'w', 82),
            ('name', 'Name', 'w', 110),
            ('vol', 'Vol%', 'e', 36),
            ('rank_delta', 'Rank Δ', 'e', 44),
            ('change', 'Chg%', 'e', 48),
            ('quadrant', 'Quad', 'w', 76),
            ('size', 'Size', 'w', 58),
            ('reason', 'Reason', 'w', 380),
        )
        rec_header_cells: list[tk.Label] = []
        for col, (_key, header, anchor, min_px) in enumerate(_rec_col_specs):
            rec_table.columnconfigure(
                col, minsize=min_px, weight=1 if _key == 'reason' else 0
            )
            hdr = tk.Label(
                rec_table,
                text=header,
                font=('Arial', 9, 'bold'),
                anchor=anchor,
                relief=tk.RIDGE,
            )
            hdr.grid(row=0, column=col, sticky='ew', padx=2, pady=1)
            rec_header_cells.append(hdr)

        rec_body_cells: list[list[tk.Label]] = []
        for slot in range(_PORTFOLIO_N_MAX):
            widgets: dict[str, tk.Label] = {}
            grid_row = slot + 1
            row_cells: list[tk.Label] = []
            for col, (key, _header, anchor, _min_px) in enumerate(_rec_col_specs):
                font = ('Arial', 8) if key == 'reason' else ('Arial', 9)
                lbl = tk.Label(
                    rec_table, font=font, anchor=anchor, relief=tk.RIDGE
                )
                lbl.grid(row=grid_row, column=col, sticky='ew', padx=2, pady=1)
                widgets[key] = lbl
                row_cells.append(lbl)
            recommend_row_widgets.append(widgets)
            rec_body_cells.append(row_cells)

        tc = TableRegionCopy.for_window(root)
        recommend_copy_grid = tc.register_grid([rec_header_cells, *rec_body_cells])

        recommend_exits_label = tk.Label(
            recommend_panel,
            text="",
            font=("Arial", 8),
            anchor="nw",
            fg="#5d4037",
            justify=tk.LEFT,
            wraplength=520,
        )

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

    if config.etf_table_extras and not use_right_extras:
        _build_recommend_panel(scroll_wrap, pack_mode=None)
        recommend_panel.grid(
            row=2, column=0, columnspan=2, sticky='ew', padx=4, pady=(4, 2)
        )
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
        if end_date_idx_local <= tail:
            return end_date_idx_local
        end_ts_local = _rrg_index[end_date_idx_local]
        for k in range(end_date_idx_local, tail - 1, -1):
            if k + 1 < len(_rrg_index):
                week_start = _rrg_index[k]
                week_end = _rrg_index[k + 1]
                if week_start <= end_ts_local <= week_end:
                    return k
        return max(tail, end_date_idx_local - 1)

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
        end_date_idx_local = int(date_scale.get())
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

        from momentum.etf.india_rrg_pick_strategies import (
            pick_strategy_label,
            pick_strategy_subtitle,
        )

        prev_portfolio = (
            _prior_week_top_n_portfolio(panel_rebal_idx)
            if panel_rebal_idx > tail
            else []
        )
        rebal_strategy = _rebal_tickers_table_order(panel_picks, panel_ranked)
        if panel_rebal_idx in _pick_holdings_cache:
            rebal_tickers = list(_pick_holdings_cache[panel_rebal_idx])
        elif exit_below_9ema_var.get():
            rebal_tickers, _ = _rebal_slots_after_9ema(rebal_strategy, panel_rebal_ts)
        else:
            rebal_tickers = list(rebal_strategy)
        from momentum.rrg_portfolio_exits import (
            exits_as_of_through_date,
            filter_exits_portfolio_panel,
        )

        exit_slices: list[tuple] = []
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
        week_exits = filter_exits_portfolio_panel(
            exits_as_of_through_date(exit_slices, panel_end_ts),
            prev_holdings=prev_portfolio,
            rebalance_holdings=[t for t in rebal_tickers if t],
        )

        was_label = (
            format_date_label(prev_panel_idx)
            if prev_panel_idx is not None
            else "—"
        )
        rebalance_label = format_date_label(panel_rebal_idx)
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
                        ranked_row_indices=panel_ranked,
                        indices=indices,
                        ref_labels=index_metadata["ref_label"],
                        display_labels=index_metadata["display"],
                        vol_by_ref=vol_by_ref,
                        end_ts=panel_rebal_ts,
                        rsr_series_by_row=rsr_tickers,
                        rsm_series_by_row=rsm_tickers,
                        rank_delta_by_row=panel_rank_delta,
                        change_pct_fn=lambda j: _tail_change_pct(
                            j, panel_start_ts, panel_rebal_ts
                        ),
                        series_at_fn=series_at,
                        curr_ranks=panel_curr_ranks,
                        prev_ranks=panel_prev_ranks,
                        top_n=n_port,
                        prev_holdings=prev_portfolio,
                        hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                        max_hold_rank=int(max_hold_rank_var.get()),
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
                        ranked_row_indices=panel_ranked,
                        indices=indices,
                        display_labels=index_metadata["display"],
                        vol_by_ticker=vol_by_ticker,
                        end_ts=panel_rebal_ts,
                        rsr_series_by_row=rsr_tickers,
                        rsm_series_by_row=rsm_tickers,
                        rank_delta_by_row=panel_rank_delta,
                        change_pct_fn=lambda j: _tail_change_pct(
                            j, panel_start_ts, panel_rebal_ts
                        ),
                        series_at_fn=series_at,
                        curr_ranks=panel_curr_ranks,
                        prev_ranks=panel_prev_ranks,
                        top_n=n_port,
                        prev_holdings=prev_portfolio,
                        hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                        max_hold_rank=int(max_hold_rank_var.get()),
                    ),
                    rebal_n,
                )
        was_week = prev_panel_idx if prev_panel_idx is not None else panel_rebal_idx
        prev_rebal_ts = (
            _rrg_index[prev_panel_idx] if prev_panel_idx is not None else None
        )

        def _prices_for_pnl(sym: str) -> tuple[pd.Series, pd.Series | None]:
            weekly, daily = _etf_price_series(sym)
            return weekly, daily if len(daily) else None

        def _weekly_for_pnl(sym: str) -> pd.Series:
            return _prices_for_pnl(sym)[0]

        def _daily_for_pnl(sym: str) -> pd.Series | None:
            return _prices_for_pnl(sym)[1]

        movers_dates_label.config(
            text=portfolio_panel_dates_line(
                rebalance_label=rebalance_label,
                was_n=was_n,
                was_label=was_label,
                rebal_n=rebal_n,
                pick_shortfall=pick_shortfall,
                exit_below_9ema=bool(exit_below_9ema_var.get()),
                subtitle=subtitle,
                exits_through_label=format_date_label(panel_end_idx),
            )
        )

        def _was_rank(ticker: str) -> int | None:
            j = _row_index_for_ticker(ticker)
            if j is None:
                return None
            return _rank_by_row(was_week).get(j)

        def _curr_rank(ticker: str) -> int | None:
            j = _row_index_for_ticker(ticker)
            if j is None:
                return None
            return panel_curr_ranks.get(j)

        panel_rows = build_portfolio_panel(
            prev_portfolio=prev_portfolio,
            rebal_strategy=rebal_strategy,
            rebal_tickers=rebal_tickers,
            end_prev_week_holdings=_end_prev_week_holdings(prev_panel_idx),
            panel_exits=week_exits,
            rebalance_ts=panel_rebal_ts,
            prev_rebalance_ts=prev_rebal_ts,
            weekly_for_ticker=_weekly_for_pnl,
            daily_for_ticker=_daily_for_pnl,
            was_rank_for_ticker=_was_rank,
            curr_rank_for_ticker=_curr_rank,
            exit_below_9ema=bool(exit_below_9ema_var.get()),
            mid_week_9ema=_mid_week_9ema_cache.get(panel_rebal_idx, []),
        )
        n_slots = len(movers_row_widgets)
        max_rows = max(len(panel_rows), 1)
        for slot in range(n_slots):
            widgets = movers_row_widgets[slot]
            grid_row = slot + 1
            if slot < len(panel_rows):
                row = panel_rows[slot]
                was_text = row["was_text"]
                now_text = row["now_text"]
                move = row["move"]
                rebal_text = row["rebal_text"]
                pick_tag = row["pick"]
                pnl_text = row["pnl"]
                mid_9ema_text = row.get("mid_9ema", "")
                now_fg = row.get("now_fg", "black")
                rebal_fg = row.get("rebal_fg", "black")
                mid_fg = row.get("mid_fg", "black")
            else:
                was_text = now_text = move = rebal_text = pick_tag = pnl_text = ""
                mid_9ema_text = ""
                now_fg = rebal_fg = mid_fg = "black"
            if slot < max_rows:
                widgets["rank"].config(text=str(slot + 1))
                widgets["was"].config(text=was_text)
                widgets["now"].config(text=now_text, fg=now_fg)
                widgets["tag"].config(text=move)
                widgets["rebal"].config(text=rebal_text, fg=rebal_fg)
                widgets["pick_tag"].config(text=pick_tag)
                widgets["pnl"].config(text=pnl_text)
                widgets["mid_9ema"].config(text=mid_9ema_text, fg=mid_fg)
                for col, key in enumerate(PORTFOLIO_PANEL_GRID_KEYS):
                    widgets[key].grid(
                        row=grid_row, column=col, sticky="ew", padx=2, pady=1
                    )
            else:
                for key in widgets:
                    widgets[key].config(text="")
                for w in widgets.values():
                    w.grid_remove()

        if movers_copy_grid is not None:
            TableRegionCopy.for_window(root).sync_styles(movers_copy_grid)
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
    ):
        strategy = _pick_strategy_key()
        max_rank = int(max_hold_rank_var.get())
        top_n = _portfolio_top_n()
        holdings = list(prev_holdings or [])

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
                rsr_series_by_row=rsr_tickers,
                rsm_series_by_row=rsm_tickers,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=lambda j: _tail_change_pct(j, start_ts, end_ts),
                series_at_fn=series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                top_n=top_n,
                prev_holdings=holdings,
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=max_rank,
            )
            return pick_india_portfolio(strategy, ctx)

        from momentum.etf.us_rrg_pick_strategies import UsPickContext, pick_us_portfolio

        vol_by_ticker = {
            indices[j]: etf_vol_by_row.get(j, 0.0) for j in range(len(indices))
        }
        ctx = UsPickContext(
            ranked_row_indices=ranked,
            indices=indices,
            display_labels=index_metadata["display"],
            vol_by_ticker=vol_by_ticker,
            end_ts=end_ts,
            rsr_series_by_row=rsr_tickers,
            rsm_series_by_row=rsm_tickers,
            rank_delta_by_row=rank_delta_by_row,
            change_pct_fn=lambda j: _tail_change_pct(j, start_ts, end_ts),
            series_at_fn=series_at,
            curr_ranks=curr_ranks,
            prev_ranks=prev_ranks,
            top_n=top_n,
            prev_holdings=holdings,
            hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
            max_hold_rank=max_rank,
        )
        return pick_us_portfolio(strategy, ctx)

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
        from momentum.etf.india_rrg_pick_strategies import ref_to_row_index

        if config.etf_recommend_profile == "india":
            return ref_to_row_index(indices, index_metadata["ref_label"])
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
        picks = _compute_picks_at_week(
            end_date_idx_local,
            ranked,
            end_ts,
            start_ts,
            rank_delta_by_row,
            curr_ranks,
            prev_ranks,
            prev_holdings=prev_holdings,
        )
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
        curr_ranks = curr_ranks or {}
        prev_ranks = prev_ranks or {}
        if picks is None:
            picks = _strategy_picks_for_week(
                int(date_scale.get()),
                ranked,
                end_ts,
                start_ts,
                rank_delta_by_row,
                curr_ranks,
                prev_ranks,
            )
        if config.etf_recommend_profile == "india":
            from momentum.etf.india_rrg_recommendations import (
                format_vol_pct,
                recommendation_row_bg,
            )
        else:
            from momentum.etf.us_rrg_recommendations import (
                format_vol_pct,
                recommendation_row_bg,
            )
        if recommend_dates_label is not None:
            from momentum.etf.india_rrg_pick_strategies import pick_strategy_subtitle

            end_l = format_date_label(int(date_scale.get()))
            subtitle = pick_strategy_subtitle(
                _pick_strategy_key(),
                hold_until_rank_exit=bool(hold_until_rank_exit_var.get()),
                max_hold_rank=int(max_hold_rank_var.get()),
                exit_below_9ema=bool(exit_below_9ema_var.get()),
            )
            recommend_dates_label.config(
                text=f"Date: {end_l}  ·  {subtitle}"
            )
        for slot, widgets in enumerate(recommend_row_widgets):
            if slot >= _portfolio_top_n() or slot >= len(picks):
                for w in widgets.values():
                    w.config(text='', bg=root.cget('bg'), fg='black')
                continue
            pick = picks[slot]
            bg = recommendation_row_bg(pick.quadrant)
            fg = rrg_row_fg_color(bg)
            widgets['rank'].config(text=str(pick.pick_rank), bg=bg, fg=fg)
            widgets['ticker'].config(text=pick.ticker, bg=bg, fg=fg)
            widgets['name'].config(text=pick.name, bg=bg, fg=fg)
            widgets['vol'].config(text=format_vol_pct(pick.vol_pct), bg=bg, fg=fg)
            rd_fg = _rank_delta_fg(pick.rank_delta)
            widgets['rank_delta'].config(
                text=pick.rank_delta, bg=bg, fg=rd_fg if fg == 'black' else 'white'
            )
            widgets['change'].config(
                text=format_change_pct(pick.change_pct), bg=bg, fg=fg
            )
            widgets['quadrant'].config(text=pick.quadrant, bg=bg, fg=fg)
            widgets['size'].config(text=pick.size_hint, bg=bg, fg=fg)
            widgets['reason'].config(text=pick.reason, bg=bg, fg=fg)
        if recommend_copy_grid is not None:
            TableRegionCopy.for_window(root).sync_styles(recommend_copy_grid)
        _refresh_exits_display(int(date_scale.get()))
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

        rank_delta_texts: list[str] = []
        rank_delta_by_row: dict[int, str] = {}
        for j in ranked:
            text = format_rank_delta(
                curr_ranks.get(j, len(indices)),
                prev_ranks.get(j),
            )
            rank_delta_texts.append(text)
            rank_delta_by_row[j] = text

        if _portfolio_cache_enabled():
            _warm_pick_holdings_cache(end_date_idx_local)

        # Table ★ marks portfolio Top N picks (rebalance week), not next slider bar.
        if config.top_movers_panel:
            picks = _panel_week_picks(end_date_idx_local)
        else:
            picks = _strategy_picks_for_week(
                end_date_idx_local,
                ranked,
                end_ts,
                start_ts,
                rank_delta_by_row,
                curr_ranks,
                prev_ranks,
            )
        _current_pick_row_indices.clear()
        _current_pick_row_indices.update(p.row_idx for p in picks)

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
            picks=picks,
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
            picks=picks,
        )

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
        end_date_idx = _resolve_end_date_idx(prev_end_ts, date_min, date_max)
        date_scale.set(end_date_idx)
        date_value_label.config(text=format_date_label(end_date_idx))
        date_range_label.config(text=_date_range_hint_text())
        update_calc_context_label()
        _ensure_plot_slots()
        print(
            f"RRG bar unit: {bar_unit} — {len(indices)} rows, "
            f"{len(_rrg_index)} dates on slider, end {format_date_label(end_date_idx)}"
        )

    def on_bar_unit_change(_event=None):
        nonlocal bar_unit
        selected = rrg_normalize_bar_unit(bar_unit_var.get())
        if selected == bar_unit:
            return
        print(f"RRG: loading {selected} bar data...")
        root.config(cursor='watch')
        root.update_idletasks()
        try:
            bar_unit = selected
            _clear_pick_cache()
            _rebuild_rrg_for_bar_unit()
            update_nav_button_labels()
            redraw_chart()
            root.update_idletasks()
        finally:
            root.config(cursor='')

    bar_unit_combo.bind('<<ComboboxSelected>>', on_bar_unit_change)
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
    if _sync_side_scroll is not None:
        root.after_idle(_sync_side_scroll)
    redraw_chart()
    root.update_idletasks()
    root.deiconify()
    root.mainloop()
