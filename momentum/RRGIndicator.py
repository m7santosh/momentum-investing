import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.nse_bhavcopy import (
    fetch_index_close_all,
    load_nse_index_weekly_histories,
    resolve_index_name,
    today_ist,
)
sys.path.insert(0, str(Path(__file__).resolve().parent / "etf"))
from etf_universe import (
    RRG_BENCHMARK_NSE,
    RRG_DEFAULT_VISIBLE_INDICES,
    RRG_NSE_INDICES,
    index_ref_etf_label,
)


def compute_rrg_indicators(index_series, benchmark_series, window=14):
    rs = 100 * (index_series / benchmark_series)
    rsr = (
        100
        + (rs - rs.rolling(window=window).mean())
        / rs.rolling(window=window).std(ddof=0)
    ).dropna()
    if len(rsr) < 2:
        return None, None, None
    rsr_roc = 100 * ((rsr / rsr.iloc[0]) - 1)
    rsm = (
        101
        + (
            (rsr_roc - rsr_roc.rolling(window=window).mean())
            / rsr_roc.rolling(window=window).std(ddof=0)
        )
    ).dropna()
    rsr = rsr[rsr.index.isin(rsm.index)]
    rsm = rsm[rsm.index.isin(rsr.index)]
    if len(rsr) < 2:
        return None, None, None
    return rsr, rsr_roc, rsm


_TAIL_MARKER_SIZE = 22
_HEAD_ARROW_SCALE = 14
tail = 5
end_date_idx = None  # set after data load to latest week
start_date, end_date = None, None
HOVER_PIXEL_RADIUS = 14
hover_points = []
_last_hover_idx = None

def get_line_points(x, y):
    # Interpolate a smooth curve through the scatter points
    tck, _ = interpolate.splprep([x, y], s=0)
    t = np.linspace(0, 1, 100)
    line_x, line_y = interpolate.splev(t, tck)
    return line_x, line_y

def get_status(x, y):
    if x < 100 and y < 100:
        return 'lagging'
    elif x > 100 and y > 100:
        return 'leading'
    elif x < 100 and y > 100:
        return 'improving'
    elif x > 100 and y < 100:
        return 'weakening'
    
def get_color(x, y):
    if get_status(x, y) == 'lagging':
        return 'red'
    elif get_status(x, y) == 'leading':
        return 'green'
    elif get_status(x, y) == 'improving':
        return 'blue'
    elif get_status(x, y) == 'weakening':
        return 'yellow'
    
period = '1y'
requested_indices = RRG_NSE_INDICES.copy()
indices = requested_indices.copy()
index_metadata = {'ref_etf': [], 'index': []}

for index_name in indices:
    index_metadata['ref_etf'].append(index_ref_etf_label(index_name))
    index_metadata['index'].append(index_name)

_use_default_indices_on_load = False
indices_to_show = (
    [n for n in indices if n in RRG_DEFAULT_VISIBLE_INDICES]
    if _use_default_indices_on_load
    else indices.copy()
)

window = 14
min_weekly_points = window + 2


def _build_rrg_date_index():
    """Week-ending dates where every index has RRG values."""
    if not rsr_tickers:
        return pd.Index([])
    common = rsr_tickers[0].index
    for rsr in rsr_tickers[1:]:
        common = common.intersection(rsr.index)
    common = common.sort_values()
    if len(common) >= 2:
        return common
    return rsr_tickers[0].index.sort_values()


def _resolve_nse_index_name(requested: str) -> str | None:
    """Resolve user input to an exact ``ind_close_all`` index name."""
    text = requested.strip()
    if not text:
        return None
    if text in indices:
        return text
    d = today_ist()
    for _ in range(12):
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        day_map = fetch_index_close_all(d, quiet=True)
        if day_map:
            canonical = resolve_index_name(text, day_map)
            if canonical:
                return canonical
        d -= timedelta(days=1)
    return None


def load_index_history(index_name: str) -> pd.Series:
    """Weekly closes for one NSE index from ``ind_close_all`` archives."""
    return load_nse_index_weekly_histories(
        [index_name], period=period, min_points=min_weekly_points
    ).get(index_name, pd.Series(dtype=float))


print("Loading NSE index EOD (ind_close_all) for RRG...")
weekly_by_index = load_nse_index_weekly_histories(
    list(dict.fromkeys(indices + [RRG_BENCHMARK_NSE])),
    period=period,
    min_points=min_weekly_points,
)
indices_data = pd.DataFrame(
    {name: weekly_by_index.get(name, pd.Series(dtype=float)) for name in indices}
)
benchmark_data = weekly_by_index.get(RRG_BENCHMARK_NSE, pd.Series(dtype=float))

available_indices = [
    n
    for n in indices
    if n in indices_data.columns and indices_data[n].notna().sum() > window
]
missing = set(indices) - set(available_indices)
if missing:
    print(f"Skipping indices with insufficient NSE data: {sorted(missing)}")
indices = available_indices
indices_to_show = [n for n in indices_to_show if n in indices]
aligned_ref = []
aligned_index = []
for name in indices:
    pos = requested_indices.index(name)
    aligned_ref.append(index_metadata['ref_etf'][pos])
    aligned_index.append(index_metadata['index'][pos])
index_metadata['ref_etf'] = aligned_ref
index_metadata['index'] = aligned_index

rs_tickers = []
rsr_tickers = []
rsr_roc_tickers = []
rsm_tickers = []

for i in range(len(indices)):
    rsr, rsr_roc, rsm = compute_rrg_indicators(
        indices_data[indices[i]], benchmark_data, window
    )
    if rsr is None:
        continue
    rs_tickers.append(100 * (indices_data[indices[i]] / benchmark_data))
    rsr_tickers.append(rsr)
    rsr_roc_tickers.append(rsr_roc)
    rsm_tickers.append(rsm)

indices = indices[: len(rsr_tickers)]
indices_to_show = [n for n in indices_to_show if n in indices]
index_metadata['ref_etf'] = index_metadata['ref_etf'][: len(indices)]
index_metadata['index'] = index_metadata['index'][: len(indices)]

if not rsr_tickers:
    raise SystemExit(
        "No NSE indices with enough price history. Check NSE ind_close_all downloads."
    )

_rrg_index = _build_rrg_date_index()
_last_nse = pd.Timestamp(_rrg_index[-1]).date() if len(_rrg_index) else None
print(
    f"RRG: NSE index EOD through {_last_nse} "
    f"({len(_rrg_index)} weeks, benchmark {RRG_BENCHMARK_NSE})"
)


def update_rrg():
    """Recompute RSR/RSM for every row (same length as ``indices``)."""
    global rs_tickers, rsr_tickers, rsr_roc_tickers, rsm_tickers
    for i in range(len(indices)):
        name = indices[i]
        rsr, rsr_roc, rsm = compute_rrg_indicators(
            indices_data[name], benchmark_data, window
        )
        if rsr is None:
            continue
        rs_tickers[i] = 100 * (indices_data[name] / benchmark_data)
        rsr_tickers[i] = rsr
        rsr_roc_tickers[i] = rsr_roc
        rsm_tickers[i] = rsm

root = tk.Tk()
root.title('RRG — NSE Indices (Bhavcopy EOD)')
root.geometry('1100x900')
root.minsize(900, 650)
root.resizable(True, True)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=2)
root.rowconfigure(2, weight=1)

chart_frame = tk.Frame(root)
chart_frame.grid(row=0, column=0, sticky='nsew')
chart_frame.rowconfigure(0, weight=1)
chart_frame.columnconfigure(0, weight=1)

fig, ax_rrg = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
ax_rrg.set_title('RRG Indicator')
ax_rrg.set_xlabel('JdK RS Ratio')
ax_rrg.set_ylabel('JdK RS Momentum')
ax_rrg.axhline(y=100, color='k', linestyle='--')
ax_rrg.axvline(x=100, color='k', linestyle='--')
ax_rrg.fill_between([94, 100], [94, 94], [100, 100], color='red', alpha=0.2)
ax_rrg.fill_between([100, 106], [94, 94], [100, 100], color='yellow', alpha=0.2)
ax_rrg.fill_between([100, 106], [100, 100], [106, 106], color='green', alpha=0.2)
ax_rrg.fill_between([94, 100], [100, 100], [106, 106], color='blue', alpha=0.2)
ax_rrg.text(95, 105, 'Improving')
ax_rrg.text(104, 105, 'Leading')
ax_rrg.text(104, 95, 'Weakening')
ax_rrg.text(95, 95, 'Lagging')
ax_rrg.set_xlim(94, 106)
ax_rrg.set_ylim(94, 106)

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
    global _last_hover_idx
    if _hover_annot.get_visible():
        _hover_annot.set_visible(False)
        _last_hover_idx = None


def _format_hover_text(point):
    title = point['index']
    if point.get('ref_etf'):
        title = f"{point['index']} (ref ETF: {point['ref_etf']})"
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
    index_name = indices[j]
    ref_etf = index_metadata['ref_etf'][j]
    prices = indices_data[index_name]
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
                'index': index_name,
                'ref_etf': ref_etf,
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
    global _last_hover_idx

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

controls_frame = tk.Frame(root, height=88, padx=8, pady=6)
controls_frame.grid(row=1, column=0, sticky='ew')
controls_frame.grid_propagate(False)

default_indices_var = tk.BooleanVar(value=_use_default_indices_on_load)
select_all_var = tk.BooleanVar(
    value=not _use_default_indices_on_load or len(indices_to_show) == len(indices)
)
_select_all_updating = False

date_max_idx = len(_rrg_index) - 1
end_date_idx = date_max_idx
start_date = _rrg_index[end_date_idx - tail]
end_date = _rrg_index[end_date_idx]


def format_date_label(idx):
    return str(_rrg_index[int(idx)]).split(' ')[0]


def _tail_marker_sizes(n_points: int) -> list[int]:
    return [_TAIL_MARKER_SIZE] * n_points


def _add_head_arrow(x_vals, y_vals, color: str):
    """Arrow on the last tail segment (direction of movement)."""
    if len(x_vals) < 2:
        return None
    arrow = FancyArrowPatch(
        (float(x_vals[-2]), float(y_vals[-2])),
        (float(x_vals[-1]), float(y_vals[-1])),
        arrowstyle='-|>',
        mutation_scale=_HEAD_ARROW_SCALE,
        linewidth=1.8,
        color=color,
        zorder=5,
        shrinkA=0,
        shrinkB=0,
    )
    ax_rrg.add_patch(arrow)
    return arrow


def on_tail_change(val):
    global tail, end_date_idx
    new_tail = int(float(val))
    if end_date_idx - new_tail < 0:
        tail_scale.set(tail)
        return
    tail = new_tail
    date_scale.config(from_=tail)
    if end_date_idx < tail:
        end_date_idx = tail
        date_scale.set(end_date_idx)
    date_value_label.config(text=format_date_label(end_date_idx))
    redraw_chart()


def on_date_change(val):
    global end_date_idx
    end_date_idx = int(float(val))
    date_value_label.config(text=format_date_label(end_date_idx))
    redraw_chart()


def update_week_step_buttons():
    current_idx = int(date_scale.get())
    if current_idx <= tail:
        prev_week_button.state(['disabled'])
    else:
        prev_week_button.state(['!disabled'])
    if current_idx >= date_max_idx:
        next_week_button.state(['disabled'])
    else:
        next_week_button.state(['!disabled'])


def step_previous_week():
    global end_date_idx
    current_idx = int(date_scale.get())
    if current_idx <= tail:
        return
    end_date_idx = current_idx - 1
    date_scale.set(end_date_idx)
    date_value_label.config(text=format_date_label(end_date_idx))
    redraw_chart()


def step_next_week():
    global end_date_idx
    current_idx = int(date_scale.get())
    if current_idx >= date_max_idx:
        return
    end_date_idx = current_idx + 1
    date_scale.set(end_date_idx)
    date_value_label.config(text=format_date_label(end_date_idx))
    redraw_chart()


week_nav_frame = tk.Frame(controls_frame)
week_nav_frame.pack(side=tk.LEFT, padx=(0, 12), anchor='n')
prev_week_button = ttk.Button(week_nav_frame, text='Previous Week', command=step_previous_week)
prev_week_button.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
next_week_button = ttk.Button(week_nav_frame, text='Next Week', command=step_next_week)
next_week_button.pack(side=tk.TOP, fill=tk.X)


def _sync_select_all_checkbox():
    if not checkbox_vars:
        return
    select_all_var.set(all(checkbox_vars[i].get() for i in range(len(indices))))


def apply_select_all(select_all: bool):
    global indices_to_show, _select_all_updating
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
    global indices_to_show
    if use_defaults:
        indices_to_show = [n for n in indices if n in RRG_DEFAULT_VISIBLE_INDICES]
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
    text='Default indices',
    variable=default_indices_var,
    command=on_default_indices_toggle,
)
default_indices_cb.pack(side=tk.LEFT, padx=(0, 12))

tail_row = tk.Frame(controls_frame)
tail_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
tk.Label(tail_row, text='Tail', width=6, anchor='w').pack(side=tk.LEFT)
tail_scale = tk.Scale(
    tail_row,
    from_=1,
    to=10,
    orient=tk.HORIZONTAL,
    showvalue=True,
    resolution=1,
    command=on_tail_change,
)
tail_scale.set(tail)
tail_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

date_row = tk.Frame(controls_frame)
date_row.pack(side=tk.TOP, fill=tk.X)
tk.Label(date_row, text='Date', width=6, anchor='w').pack(side=tk.LEFT)
date_scale = tk.Scale(
    date_row,
    from_=tail,
    to=date_max_idx,
    orient=tk.HORIZONTAL,
    showvalue=False,
    resolution=1,
    command=on_date_change,
)
date_scale.set(end_date_idx)
date_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
date_value_label = tk.Label(date_row, text=format_date_label(end_date_idx), width=12, anchor='w')
date_value_label.pack(side=tk.LEFT, padx=(8, 0))

table_section = tk.Frame(root)
table_section.grid(row=2, column=0, sticky='nsew', padx=4, pady=(0, 4))
table_section.columnconfigure(0, weight=1)
table_section.rowconfigure(0, weight=1)

scroll_wrap = tk.Frame(table_section)
scroll_wrap.grid(row=0, column=0, sticky='nsew')
scroll_wrap.columnconfigure(0, weight=1)
scroll_wrap.rowconfigure(1, weight=1)

table_header = tk.Frame(scroll_wrap)
table_header.grid(row=0, column=0, sticky='ew')

header_scroll_gutter = tk.Frame(scroll_wrap, width=18)
header_scroll_gutter.grid(row=0, column=1, sticky='ns')

body_wrap = tk.Frame(scroll_wrap)
body_wrap.grid(row=1, column=0, sticky='nsew')
body_wrap.columnconfigure(0, weight=1)
body_wrap.rowconfigure(0, weight=1)

table_scroll_y = tk.Scrollbar(scroll_wrap, orient=tk.VERTICAL)
table_scroll_y.grid(row=1, column=1, sticky='ns')

table_scroll_x = tk.Scrollbar(scroll_wrap, orient=tk.HORIZONTAL)
table_scroll_x.grid(row=2, column=0, sticky='ew')

table_canvas = tk.Canvas(
    body_wrap,
    highlightthickness=0,
    yscrollcommand=table_scroll_y.set,
    xscrollcommand=table_scroll_x.set,
)
table_canvas.grid(row=0, column=0, sticky='nsew')
table_scroll_y.config(command=table_canvas.yview)
table_scroll_x.config(command=table_canvas.xview)


def _sync_header_scroll_gutter(_event=None):
    table_scroll_y.update_idletasks()
    gutter_w = table_scroll_y.winfo_width()
    if gutter_w > 1:
        header_scroll_gutter.configure(width=gutter_w)

table_body = tk.Frame(table_canvas)
_table_canvas_win = table_canvas.create_window((0, 0), window=table_body, anchor='nw')

# Character widths — header and body use the same values for alignment.
_COL_RANK = 0
_COL_REF = 1
_COL_INDEX = 2
_COL_PRICE = 3
_COL_CHANGE = 4
_COL_VISIBLE = 5
_TABLE_COL_CHARS = [4, 8, 34, 9, 6, 9]
_TABLE_CELL_PADX = (2, 1)
_TABLE_ROW_PADY = 1
_TABLE_FONT = ('Arial', 10)
_TABLE_FONT_BOLD = ('Arial', 10, 'bold')
_TABLE_NEUTRAL_BG = root.cget('bg')
_VISIBLE_COL = _COL_VISIBLE


def _sync_table_layout(_event=None):
    table_scroll_y.update_idletasks()
    gutter_w = table_scroll_y.winfo_width()
    if gutter_w > 1:
        header_scroll_gutter.configure(width=gutter_w)
    table_header.update_idletasks()
    table_body.update_idletasks()
    total_w = max(table_header.winfo_reqwidth(), table_body.winfo_reqwidth(), 1)
    table_canvas.itemconfigure(_table_canvas_win, width=total_w)
    table_canvas.configure(scrollregion=table_canvas.bbox('all'))


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
table_scroll_y.bind('<Configure>', _sync_header_scroll_gutter)
for widget in (table_canvas, table_body, table_header, body_wrap, scroll_wrap, table_section):
    widget.bind('<MouseWheel>', _on_table_mousewheel)
    widget.bind('<Button-4>', _on_table_mousewheel)
    widget.bind('<Button-5>', _on_table_mousewheel)

headers = ['Rank', 'Ref ETF', 'Index', 'Price', 'Change', 'Visible']
for j in range(len(headers)):
    if j == _COL_VISIBLE:
        visible_header = tk.Frame(
            table_header, relief=tk.RIDGE, bg=_TABLE_NEUTRAL_BG, highlightthickness=0
        )
        visible_header.grid(
            row=0,
            column=j,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        tk.Label(
            visible_header,
            text=headers[j],
            width=5,
            anchor='w',
            font=_TABLE_FONT_BOLD,
            bg=_TABLE_NEUTRAL_BG,
        ).pack(side=tk.LEFT)
        select_all_cb = ttk.Checkbutton(visible_header, variable=select_all_var)
        select_all_cb.pack(side=tk.LEFT)
    else:
        anchor = 'e' if j in (_COL_RANK, _COL_PRICE, _COL_CHANGE) else 'w'
        tk.Label(
            table_header,
            text=headers[j],
            width=_TABLE_COL_CHARS[j],
            relief=tk.RIDGE,
            anchor=anchor,
            font=_TABLE_FONT_BOLD,
            bg=_TABLE_NEUTRAL_BG,
        ).grid(
            row=0,
            column=j,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )

def update_entry(event):
    global indices_data, indices, indices_to_show
    row = event.widget._row_idx
    requested = event.widget.get().strip()
    try:
        index_name = _resolve_nse_index_name(requested)
        if not index_name:
            raise ValueError(f'unknown NSE index: {requested!r}')
        series = load_index_history(index_name)
        if len(series) < min_weekly_points:
            raise ValueError('insufficient NSE weekly history')
        previous = indices[row]
        if previous in indices_to_show:
            indices_to_show.remove(previous)
        indices[row] = index_name
        indices_data[index_name] = series
        if checkbox_vars[row].get() and index_name not in indices_to_show:
            indices_to_show.append(index_name)
        index_metadata['ref_etf'][row] = index_ref_etf_label(index_name)
        index_metadata['index'][row] = index_name
        table_widgets[row]['ref_label'].config(text=index_metadata['ref_etf'][row])
        update_rrg()
        redraw_chart()
    except Exception as e:
        print(e)
        entry = event.widget
        row = entry._row_idx
        entry.delete(0, tk.END)
        entry.insert(0, index_metadata['index'][row])

def on_visibility_toggle(row_idx):
    global indices_to_show
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

def on_enter(event):
    ticker_name = event.widget.cget('text')
    event.widget.configure(text=ticker_name)

def on_leave(event):
    event.widget.configure(text='')


def _tail_change_pct(row_idx: int, start_ts, end_ts):
    """% price change over the visible tail window (for ranking)."""
    index_name = indices[row_idx]
    try:
        p_start = float(indices_data[index_name].loc[start_ts])
        p_end = float(indices_data[index_name].loc[end_ts])
        if p_start == 0:
            return float('-inf')
        return (p_end - p_start) / p_start * 100
    except (KeyError, TypeError, ValueError):
        return float('-inf')


def refresh_table_ranking():
    """Reorder table rows by tail-window performance (best % change first)."""
    end_date_idx = int(date_scale.get())
    end_ts = _rrg_index[end_date_idx]
    start_ts = _rrg_index[end_date_idx - tail]

    ranked = sorted(
        range(len(indices)),
        key=lambda j: _tail_change_pct(j, start_ts, end_ts),
        reverse=True,
    )

    for display_row, j in enumerate(ranked):
        w = table_widgets[j]
        index_name = indices[j]
        chg = _tail_change_pct(j, start_ts, end_ts)
        try:
            price = round(float(indices_data[index_name].loc[end_ts]), 2)
        except (KeyError, TypeError, ValueError):
            price = ''
        try:
            bg_color = get_color(
                float(rsr_tickers[j].loc[end_ts]), float(rsm_tickers[j].loc[end_ts])
            )
        except (KeyError, TypeError, ValueError):
            bg_color = 'gray'
        fg_color = 'white' if bg_color in ('red', 'green', 'blue') else 'black'

        rank_num = display_row + 1
        w['rank_label'].grid(
            row=display_row,
            column=_COL_RANK,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        w['ref_label'].grid(
            row=display_row,
            column=_COL_REF,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        w['index_entry'].grid(
            row=display_row,
            column=_COL_INDEX,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        w['price_label'].grid(
            row=display_row,
            column=_COL_PRICE,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        w['chg_label'].grid(
            row=display_row,
            column=_COL_CHANGE,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        w['visible_cell'].grid(
            row=display_row,
            column=_COL_VISIBLE,
            sticky='ew',
            padx=_TABLE_CELL_PADX,
            pady=_TABLE_ROW_PADY,
        )
        w['rank_label'].config(text=rank_num)
        w['price_label'].config(text=price)
        chg_text = round(chg, 1) if chg != float('-inf') else ''
        w['chg_label'].config(text=chg_text)
        for key in ('rank_label', 'ref_label', 'index_entry', 'price_label', 'chg_label'):
            w[key].config(bg=bg_color, fg=fg_color)


checkbox_vars = []
table_widgets = []

for i in range(len(indices)):
    index_name = indices[i]
    ref_etf = index_metadata['ref_etf'][i]
    price = round(float(indices_data[index_name].loc[end_date]), 2)
    chg = round(
        (float(indices_data[index_name].loc[end_date]) - float(indices_data[index_name].loc[start_date]))
        / float(indices_data[index_name].loc[start_date])
        * 100,
        1,
    )
    bg_color = get_color(rsr_tickers[i].iloc[-1], rsm_tickers[i].iloc[-1])
    fg_color = 'white' if bg_color in ('red', 'green', 'blue') else 'black'
    index_var = tk.StringVar(value=index_name)
    rank_label = tk.Label(
        table_body,
        text=i + 1,
        width=_TABLE_COL_CHARS[_COL_RANK],
        relief=tk.RIDGE,
        anchor='e',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    )
    ref_label = tk.Label(
        table_body,
        text=ref_etf,
        width=_TABLE_COL_CHARS[_COL_REF],
        relief=tk.RIDGE,
        anchor='w',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    )
    index_entry = tk.Entry(
        table_body,
        textvariable=index_var,
        width=_TABLE_COL_CHARS[_COL_INDEX],
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
        width=_TABLE_COL_CHARS[_COL_PRICE],
        relief=tk.RIDGE,
        anchor='e',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    )
    chg_label = tk.Label(
        table_body,
        text=chg,
        width=_TABLE_COL_CHARS[_COL_CHANGE],
        relief=tk.RIDGE,
        anchor='e',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    )
    visible_cell = tk.Frame(
        table_body, relief=tk.RIDGE, bg=_TABLE_NEUTRAL_BG, highlightthickness=0
    )
    checkbox_var = tk.BooleanVar(value=indices[i] in indices_to_show)
    checkbox_vars.append(checkbox_var)
    checkbox = ttk.Checkbutton(
        visible_cell,
        variable=checkbox_var,
        command=lambda idx=i: on_visibility_toggle(idx),
    )
    checkbox.pack(side=tk.LEFT)
    table_widgets.append(
        {
            'rank_label': rank_label,
            'ref_label': ref_label,
            'index_entry': index_entry,
            'price_label': price_label,
            'chg_label': chg_label,
            'visible_cell': visible_cell,
        }
    )

select_all_cb.config(command=on_select_all_toggle)
refresh_table_ranking()
_sync_select_all_checkbox()
root.update_idletasks()
_sync_header_scroll_gutter()
_sync_table_layout()


scatter_plots = [None] * len(indices)
line_plots = [None] * len(indices)
head_arrows = [None] * len(indices)
annotations = [None] * len(indices)


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
    if root.winfo_exists():
        try:
            canvas.draw_idle()
        except tk.TclError:
            pass

def update_frame():
    global start_date, end_date, end_date_idx, hover_points, _last_hover_idx

    if not root.winfo_exists():
        return

    hover_points = []
    _last_hover_idx = None
    _hide_hover_tooltip()

    end_date_idx = int(date_scale.get())
    end_date = _rrg_index[end_date_idx]
    start_date = _rrg_index[end_date_idx - tail]

    refresh_table_ranking()

    for j in range(len(indices)):
        remove_ticker_artists(j)

        if indices[j] not in indices_to_show:
            continue

        filtered_rsr_tickers = rsr_tickers[j].loc[
            (rsr_tickers[j].index > start_date) & (rsr_tickers[j].index <= end_date)
        ]
        filtered_rsm_tickers = rsm_tickers[j].loc[
            (rsm_tickers[j].index > start_date) & (rsm_tickers[j].index <= end_date)
        ]
        if filtered_rsr_tickers.empty:
            continue
        _append_hover_points(j, filtered_rsr_tickers, filtered_rsm_tickers)
        color = get_color(filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1])
        xs = filtered_rsr_tickers.values
        ys = filtered_rsm_tickers.values
        if len(xs) >= 2:
            scatter_plots[j] = ax_rrg.scatter(
                xs[:-1],
                ys[:-1],
                color=color,
                s=_tail_marker_sizes(len(xs) - 1),
                zorder=4,
            )
            head_arrows[j] = _add_head_arrow(xs, ys, color)
        else:
            scatter_plots[j] = ax_rrg.scatter(
                xs, ys, color=color, s=_tail_marker_sizes(len(xs)), zorder=4
            )
            head_arrows[j] = None
        line_plots[j] = ax_rrg.plot(xs, ys, color='black', alpha=0.2, zorder=2)[0]
        annotations[j] = ax_rrg.annotate(
            indices[j],
            (filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1]),
            fontsize=8,
        )


def on_close():
    plt.close(fig)
    root.quit()
    root.destroy()


root.protocol('WM_DELETE_WINDOW', on_close)
redraw_chart()
root.mainloop()