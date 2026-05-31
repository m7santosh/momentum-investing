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


def run_rrg_app(config: RrgAppConfig) -> None:
    """Build UI, load data, and run the RRG main loop."""
    recommend_in_side = config.etf_table_extras and config.top_movers_panel
    use_right_extras = recommend_in_side
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
    root.title(config.window_title)
    root.geometry('1600x900' if use_right_extras else ('1400x900' if config.top_movers_panel else '1100x900'))
    root.minsize(1280 if use_right_extras else (1280 if config.top_movers_panel else 900), 650)
    root.resizable(True, True)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)
    root.rowconfigure(2, weight=1)

    chart_frame = tk.Frame(root)
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

    controls_frame = tk.Frame(root, height=104, padx=8, pady=6)
    controls_frame.grid(row=1, column=0, sticky='ew')
    controls_frame.grid_propagate(False)

    show_rrg_var = tk.BooleanVar(value=False)
    bar_unit_var = tk.StringVar(value="Week")

    def _apply_rrg_chart_visibility():
        if show_rrg_var.get():
            chart_frame.grid(row=0, column=0, sticky='nsew')
            root.rowconfigure(0, weight=2, minsize=420)
        else:
            _hide_hover_tooltip()
            chart_frame.grid_remove()
            root.rowconfigure(0, weight=0, minsize=0)
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

    def _tail_marker_sizes(n_points: int) -> list[int]:
        return [TAIL_MARKER_SIZE] * n_points

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

    table_section = tk.Frame(root)
    table_section.grid(row=2, column=0, sticky='nsew', padx=4, pady=(0, 4))
    table_section.rowconfigure(0, weight=1)
    table_section.columnconfigure(0, weight=1)

    tables_row = tk.Frame(table_section)
    tables_row.grid(row=0, column=0, sticky='nsew')
    tables_row.rowconfigure(0, weight=1)
    tables_row.columnconfigure(0, weight=1)

    movers_panel = None
    movers_dates_label = None
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
        nonlocal movers_panel, movers_dates_label
        movers_panel = tk.Frame(
            parent,
            padx=6,
            pady=4,
            relief=tk.GROOVE,
            borderwidth=1,
        )
        movers_panel.pack(side=tk.TOP, anchor='nw', fill=tk.X)
        tk.Label(
            movers_panel,
            text=config.top_movers_title,
            font=('Arial', 10, 'bold'),
            anchor='w',
        ).pack(fill=tk.X)
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
        hdr = tk.Frame(movers_panel)
        hdr.pack(fill=tk.X)
        for col, (text, width, anchor) in enumerate(
            [
                ('#', 2, 'e'),
                ('Was', 18, 'w'),
                ('Now', 18, 'w'),
            ]
        ):
            tk.Label(
                hdr,
                text=text,
                font=('Arial', 9, 'bold'),
                width=width,
                anchor=anchor,
            ).grid(row=0, column=col, sticky='ew')
        hdr.columnconfigure(1, weight=1)
        hdr.columnconfigure(2, weight=1)
        body = tk.Frame(movers_panel)
        body.pack(fill=tk.X)
        for _ in range(config.top_movers_count):
            row = tk.Frame(body)
            row.pack(fill=tk.X, pady=1)
            movers_row_widgets.append(
                {
                    'rank': tk.Label(row, font=('Arial', 9), width=2, anchor='e'),
                    'was': tk.Label(row, font=('Arial', 9), width=18, anchor='w'),
                    'now': tk.Label(row, font=('Arial', 9), width=18, anchor='w'),
                }
            )
            movers_row_widgets[-1]['rank'].grid(row=0, column=0, sticky='e')
            movers_row_widgets[-1]['was'].grid(row=0, column=1, sticky='w')
            movers_row_widgets[-1]['now'].grid(row=0, column=2, sticky='w')
            row.columnconfigure(1, weight=1)
            row.columnconfigure(2, weight=1)

    if use_right_extras:
        tables_row.columnconfigure(1, weight=0, minsize=920)
        side_panel = tk.Frame(tables_row, width=920)
        side_panel.grid(row=0, column=1, sticky='nsew', padx=(12, 0))
        side_panel.grid_propagate(False)
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

        side_content.bind('<Configure>', _sync_side_scroll)
        side_canvas.bind('<Configure>', _sync_side_scroll)

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

    def _build_recommend_panel(parent: tk.Frame, *, pack_mode: str | None) -> None:
        nonlocal recommend_panel, recommend_dates_label
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
            height=24 * (config.etf_recommend_count + 1) + 8,
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
        for col, (_key, header, anchor, min_px) in enumerate(_rec_col_specs):
            rec_table.columnconfigure(
                col, minsize=min_px, weight=1 if _key == 'reason' else 0
            )
            tk.Label(
                rec_table,
                text=header,
                font=('Arial', 9, 'bold'),
                anchor=anchor,
            ).grid(row=0, column=col, sticky='ew', padx=2, pady=1)

        for slot in range(config.etf_recommend_count):
            widgets: dict[str, tk.Label] = {}
            grid_row = slot + 1
            for col, (key, _header, anchor, _min_px) in enumerate(_rec_col_specs):
                font = ('Arial', 8) if key == 'reason' else ('Arial', 9)
                lbl = tk.Label(rec_table, font=font, anchor=anchor)
                lbl.grid(row=grid_row, column=col, sticky='ew', padx=2, pady=1)
                widgets[key] = lbl
            recommend_row_widgets.append(widgets)

    if use_right_extras and side_content is not None:
        _build_recommend_panel(side_content, pack_mode='top')
        _bind_side_mousewheel(
            side_panel,
            side_canvas,
            side_content,
            movers_panel,
            recommend_panel,
        )

    scroll_wrap = tk.Frame(tables_row)
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
                    chg_w = max(chg_w, _text_px(_format_change_pct(chg)))
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

    def _series_at(series: pd.Series, ts) -> float:
        """Close at ``ts``, or last available bar on or before ``ts``."""
        try:
            return float(series.loc[ts])
        except KeyError:
            pos = series.index.get_indexer([pd.Timestamp(ts)], method="ffill")
            if pos[0] < 0:
                raise
            return float(series.iloc[pos[0]])

    def _tail_change_pct(row_idx: int, start_ts, end_ts):
        """% price change over the visible tail window (for ranking)."""
        index_name = indices[row_idx]
        try:
            p_start = _series_at(indices_data[index_name], start_ts)
            p_end = _series_at(indices_data[index_name], end_ts)
            if p_start == 0:
                return float('-inf')
            return (p_end - p_start) / p_start * 100
        except (KeyError, TypeError, ValueError, IndexError):
            return float('-inf')

    def _format_change_pct(chg: float) -> str:
        if chg == float('-inf'):
            return ''
        return f'{round(chg, 2):.2f}'

    def _rank_by_row(end_date_idx_local: int) -> dict[int, int]:
        """Row index -> rank (1 = best tail-window change) at the given week index."""
        if end_date_idx_local < tail:
            return {}
        end_ts = _rrg_index[end_date_idx_local]
        start_ts = _rrg_index[end_date_idx_local - tail]
        ranked = sorted(
            range(len(indices)),
            key=lambda j: _tail_change_pct(j, start_ts, end_ts),
            reverse=True,
        )
        return {j: display_rank + 1 for display_rank, j in enumerate(ranked)}

    def _format_rank_delta(curr_rank: int, prev_rank: int | None) -> str:
        """Change vs prior week (+ = moved up in rank)."""
        if prev_rank is None:
            return '—'
        delta = prev_rank - curr_rank
        if delta == 0:
            return '0'
        if delta > 0:
            return f'+{delta}'
        return str(delta)

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
        return sorted(
            range(len(indices)),
            key=lambda j: _tail_change_pct(j, start_ts, end_ts),
            reverse=True,
        )

    def _movers_cell_text(row_idx: int, table_rank: int | str) -> str:
        ref = (
            index_metadata['ref_label'][row_idx]
            or index_metadata['display'][row_idx]
        )
        return f"{ref} ({table_rank})"

    def refresh_top_movers_panel(now_ranked: list[int] | None = None):
        """Top-N main-table rows: Was = prior week top-N, Now = current top-N."""
        if not config.top_movers_panel or movers_dates_label is None:
            return
        end_date_idx_local = int(date_scale.get())
        prev_date_idx = (
            end_date_idx_local - 1 if end_date_idx_local > tail else None
        )
        curr_ranks = _rank_by_row(end_date_idx_local)
        if now_ranked is None:
            now_ranked = _ranked_rows_at(end_date_idx_local)
        now_top_rows = now_ranked[: config.top_movers_count]
        was_top_rows = (
            _ranked_rows_at(prev_date_idx)[: config.top_movers_count]
            if prev_date_idx is not None
            else []
        )
        now_label = format_date_label(end_date_idx_local)
        was_label = (
            format_date_label(prev_date_idx)
            if prev_date_idx is not None
            else '—'
        )

        movers_dates_label.config(
            text=(
                f"Was: {was_label}   Now: {now_label} — "
                f"Was (rank) = that line today; Now (rank) = selected {bar_unit}"
            )
        )

        for slot, widgets in enumerate(movers_row_widgets):
            was_text = ''
            now_text = ''
            if slot < len(was_top_rows):
                j = was_top_rows[slot]
                was_rank = curr_ranks.get(j)
                was_text = _movers_cell_text(
                    j, was_rank if was_rank is not None else '—'
                )
            if slot < len(now_top_rows):
                j = now_top_rows[slot]
                now_rank = curr_ranks.get(j)
                now_text = _movers_cell_text(
                    j, now_rank if now_rank is not None else '—'
                )
            widgets['rank'].config(text=str(slot + 1))
            widgets['was'].config(text=was_text)
            widgets['now'].config(text=now_text)
        if _sync_side_scroll is not None:
            root.after_idle(_sync_side_scroll)

    def refresh_recommendations_panel(
        ranked: list[int],
        end_ts,
        start_ts,
        rank_delta_by_row: dict[int, str],
        curr_ranks: dict[int, int] | None = None,
        prev_ranks: dict[int, int] | None = None,
    ):
        if not config.etf_table_extras or not recommend_row_widgets:
            return
        if config.etf_recommend_profile == "india":
            from momentum.etf.india_rrg_recommendations import (
                format_vol_pct,
                recommend_india_etfs,
                recommendation_row_bg,
            )

            picks = recommend_india_etfs(
                ranked_row_indices=ranked,
                indices=indices,
                ref_labels=index_metadata["ref_label"],
                display_labels=index_metadata["display"],
                vol_by_ref={
                    (index_metadata["ref_label"][j] or indices[j])
                    .upper()
                    .replace(".NS", ""): etf_vol_by_row.get(j, 0.0)
                    for j in range(len(indices))
                },
                end_ts=end_ts,
                rsr_series_by_row=rsr_tickers,
                rsm_series_by_row=rsm_tickers,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=lambda j: _tail_change_pct(j, start_ts, end_ts),
                series_at_fn=_series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                limit=config.etf_recommend_count,
            )
        else:
            from momentum.etf.us_rrg_recommendations import (
                format_vol_pct,
                recommend_us_etfs,
                recommendation_row_bg,
            )

            vol_by_ticker = {
                indices[j]: etf_vol_by_row.get(j, 0.0) for j in range(len(indices))
            }
            picks = recommend_us_etfs(
                ranked_row_indices=ranked,
                indices=indices,
                display_labels=index_metadata["display"],
                vol_by_ticker=vol_by_ticker,
                end_ts=end_ts,
                rsr_series_by_row=rsr_tickers,
                rsm_series_by_row=rsm_tickers,
                rank_delta_by_row=rank_delta_by_row,
                change_pct_fn=lambda j: _tail_change_pct(j, start_ts, end_ts),
                series_at_fn=_series_at,
                curr_ranks=curr_ranks,
                prev_ranks=prev_ranks,
                limit=config.etf_recommend_count,
            )
        if recommend_dates_label is not None:
            end_l = format_date_label(int(date_scale.get()))
            recommend_dates_label.config(
                text=(
                    f"Date: {end_l}  ·  Ranked by momentum+reliability score "
                    f"(Leading/Improving, Rank Δ>0)"
                )
            )
        for slot, widgets in enumerate(recommend_row_widgets):
            if slot >= len(picks):
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
                text=_format_change_pct(pick.change_pct), bg=bg, fg=fg
            )
            widgets['quadrant'].config(text=pick.quadrant, bg=bg, fg=fg)
            widgets['size'].config(text=pick.size_hint, bg=bg, fg=fg)
            widgets['reason'].config(text=pick.reason, bg=bg, fg=fg)
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
            text = _format_rank_delta(
                curr_ranks.get(j, len(indices)),
                prev_ranks.get(j),
            )
            rank_delta_texts.append(text)
            rank_delta_by_row[j] = text

        from momentum.etf.us_rrg_recommendations import format_vol_pct

        for display_row, j in enumerate(ranked):
            w = table_widgets[j]
            index_name = indices[j]
            chg = _tail_change_pct(j, start_ts, end_ts)
            try:
                price = round(_series_at(indices_data[index_name], end_ts), 2)
            except (KeyError, TypeError, ValueError, IndexError):
                price = ''
            try:
                rsr_val = _series_at(rsr_tickers[j], end_ts)
                rsm_val = _series_at(rsm_tickers[j], end_ts)
                bg_color = get_color(rsr_val, rsm_val)
            except (KeyError, TypeError, ValueError, IndexError):
                bg_color = RRG_COLOR_NA
            fg_color = rrg_row_fg_color(bg_color)

            rank_num = display_row + 1
            rank_delta_text = rank_delta_texts[display_row]
            rank_delta_fg = _rank_delta_fg(rank_delta_text)
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
            w['rank_label'].config(text=rank_num)
            w['rank_delta_label'].config(
                text=rank_delta_text, fg=rank_delta_fg if fg_color == 'black' else 'white'
            )
            w['ref_label'].config(text=index_metadata['ref_label'][j] or '-')
            w['index_entry'].delete(0, tk.END)
            w['index_entry'].insert(0, index_metadata['display'][j])
            w['price_label'].config(text=price)
            w['chg_label'].config(text=_format_change_pct(chg))
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
            w['rank_delta_label'].config(
                fg=rank_delta_fg if fg_color == 'black' else 'white'
            )
        _hide_stale_table_rows(len(indices))
        _update_table_column_widths(end_ts, start_ts, rank_delta_texts)
        refresh_top_movers_panel(ranked)
        refresh_recommendations_panel(
            ranked, end_ts, start_ts, rank_delta_by_row, curr_ranks, prev_ranks
        )

    checkbox_vars = []
    table_widgets = []

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
            text=_format_change_pct(chg),
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

            if indices[j] not in indices_to_show:
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
            if len(xs) >= 2:
                scatter_plots[j] = ax_rrg.scatter(
                    xs[:-1],
                    ys[:-1],
                    color=plot_color,
                    s=_tail_marker_sizes(len(xs) - 1),
                    zorder=4,
                )
                head_arrows[j] = _add_head_arrow(xs, ys, plot_color)
            else:
                scatter_plots[j] = ax_rrg.scatter(
                    xs, ys, color=plot_color, s=_tail_marker_sizes(len(xs)), zorder=4
                )
                head_arrows[j] = None
            line_plots[j] = ax_rrg.plot(
                xs, ys, color=plot_color, alpha=0.55, linewidth=1.4, zorder=2
            )[0]
            annotations[j] = ax_rrg.annotate(
                index_metadata['display'][j],
                (filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1]),
                fontsize=8,
                color=plot_color,
                fontweight='medium',
            )

    def on_close():
        plt.close(fig)
        root.quit()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)
    if _sync_side_scroll is not None:
        root.after_idle(_sync_side_scroll)
    redraw_chart()
    root.mainloop()
