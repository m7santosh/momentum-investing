import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
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
    fetch_weekly_close_series,
    load_nse_index_weekly_histories,
)
sys.path.insert(0, str(Path(__file__).resolve().parent / "etf"))
from etf_universe import (
    ETF_TO_NSE_INDEX,
    RRG_BENCHMARK_NSE,
    RRG_DEFAULT_VISIBLE,
    RRG_ETF_TICKERS,
)


def get_close_prices(downloaded):
    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        price_key = 'Close' if 'Close' in level0 else 'Adj Close'
        return downloaded[price_key]
    price_key = 'Close' if 'Close' in downloaded.columns else 'Adj Close'
    return downloaded[price_key].squeeze()


def compute_rrg_indicators(ticker_series, benchmark_series, window=14):
    rs = 100 * (ticker_series / benchmark_series)
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
requested_tickers = RRG_ETF_TICKERS.copy()
tickers = requested_tickers.copy()
tickers_metadata_dict = {'symbol': [], 'name': []}

for etf in tickers:
    nse_index = ETF_TO_NSE_INDEX[etf]
    tickers_metadata_dict['symbol'].append(etf.replace('.NS', ''))
    tickers_metadata_dict['name'].append(f"{etf.replace('.NS', '')} → {nse_index}")

_use_default_indices_on_load = False
tickers_to_show = (
    [t for t in tickers if t in RRG_DEFAULT_VISIBLE]
    if _use_default_indices_on_load
    else tickers.copy()
)

window = 14
min_weekly_points = window + 2

unmapped = sorted(t for t in ETF_TO_NSE_INDEX if not ETF_TO_NSE_INDEX.get(t))
if unmapped:
    print(f"RRG: skipping ETFs without NSE index map: {unmapped}")


def load_price_history(etf_symbol: str) -> pd.Series:
    """Reload one series: mapped NSE index weekly, else Yahoo/bhavcopy fallback."""
    index_name = ETF_TO_NSE_INDEX.get(etf_symbol)
    if index_name:
        weekly = load_nse_index_weekly_histories(
            [index_name], period=period, min_points=min_weekly_points
        ).get(index_name, pd.Series(dtype=float))
        if len(weekly) >= min_weekly_points:
            return weekly
    return fetch_weekly_close_series(etf_symbol, period=period, min_points=min_weekly_points)


print("Loading NSE index EOD (mapped ETFs) for RRG...")
unique_indices = list(dict.fromkeys(ETF_TO_NSE_INDEX[t] for t in tickers))
weekly_by_index = load_nse_index_weekly_histories(
    list(dict.fromkeys(unique_indices + [RRG_BENCHMARK_NSE])),
    period=period,
    min_points=min_weekly_points,
)
ticker_frames = {
    etf: weekly_by_index.get(ETF_TO_NSE_INDEX[etf], pd.Series(dtype=float))
    for etf in tickers
}
tickers_data = pd.DataFrame(ticker_frames)
benchmark_data = weekly_by_index.get(RRG_BENCHMARK_NSE, pd.Series(dtype=float))

available_tickers = [
    t for t in tickers
    if t in tickers_data.columns and tickers_data[t].notna().sum() > window
]
missing = set(tickers) - set(available_tickers)
if missing:
    print(f"Skipping tickers with insufficient data: {sorted(missing)}")
tickers = available_tickers
tickers_to_show = [t for t in tickers_to_show if t in tickers]
aligned_symbols = []
aligned_names = []
for t in tickers:
    idx = requested_tickers.index(t)
    aligned_symbols.append(tickers_metadata_dict['symbol'][idx])
    aligned_names.append(tickers_metadata_dict['name'][idx])
tickers_metadata_dict['symbol'] = aligned_symbols
tickers_metadata_dict['name'] = aligned_names

rs_tickers = []
rsr_tickers = []
rsr_roc_tickers = []
rsm_tickers = []

for i in range(len(tickers)):
    rsr, rsr_roc, rsm = compute_rrg_indicators(
        tickers_data[tickers[i]], benchmark_data, window
    )
    if rsr is None:
        continue
    rs_tickers.append(100 * (tickers_data[tickers[i]] / benchmark_data))
    rsr_tickers.append(rsr)
    rsr_roc_tickers.append(rsr_roc)
    rsm_tickers.append(rsm)

tickers = tickers[: len(rsr_tickers)]
tickers_to_show = [t for t in tickers_to_show if t in tickers]
tickers_metadata_dict['symbol'] = tickers_metadata_dict['symbol'][: len(tickers)]
tickers_metadata_dict['name'] = tickers_metadata_dict['name'][: len(tickers)]

if not rsr_tickers:
    raise SystemExit(
        "No tickers with enough price history. Check symbols or try a longer period."
    )

def update_rrg():
    global rs_tickers, rsr_tickers, rsr_roc_tickers, rsm_tickers
    rs_tickers = []
    rsr_tickers = []
    rsr_roc_tickers = []
    rsm_tickers = []

    for i in range(len(tickers)):
        rsr, rsr_roc, rsm = compute_rrg_indicators(
            tickers_data[tickers[i]], benchmark_data, window
        )
        if rsr is None:
            continue
        rs_tickers.append(100 * (tickers_data[tickers[i]] / benchmark_data))
        rsr_tickers.append(rsr)
        rsr_roc_tickers.append(rsr_roc)
        rsm_tickers.append(rsm)

root = tk.Tk()
root.title('RRG Indicator')
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
    lines = [
        f"{point['ticker']} ({point['name']})",
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
    ticker = tickers[j]
    name = tickers_metadata_dict['name'][j]
    prices = tickers_data[ticker]
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
                'ticker': ticker,
                'name': name,
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
    value=not _use_default_indices_on_load or len(tickers_to_show) == len(tickers)
)
_select_all_updating = False

_rrg_index = rsr_tickers[0].index
date_max_idx = len(_rrg_index) - 1
end_date_idx = date_max_idx
start_date = _rrg_index[end_date_idx - tail]
end_date = _rrg_index[end_date_idx]


def format_date_label(idx):
    return str(rsr_tickers[0].index[int(idx)]).split(' ')[0]


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


def update_next_week_button():
    if int(date_scale.get()) >= date_max_idx:
        next_week_button.state(['disabled'])
    else:
        next_week_button.state(['!disabled'])


def step_next_week():
    global end_date_idx
    current_idx = int(date_scale.get())
    if current_idx >= date_max_idx:
        return
    end_date_idx = current_idx + 1
    date_scale.set(end_date_idx)
    date_value_label.config(text=format_date_label(end_date_idx))
    redraw_chart()


next_week_button = ttk.Button(controls_frame, text='Next Week', width=10, command=step_next_week)
next_week_button.pack(side=tk.LEFT, padx=(0, 12))


def _sync_select_all_checkbox():
    if not checkbox_vars:
        return
    select_all_var.set(all(checkbox_vars[i].get() for i in range(len(tickers))))


def apply_select_all(select_all: bool):
    global tickers_to_show, _select_all_updating
    _select_all_updating = True
    default_indices_var.set(False)
    tickers_to_show = tickers.copy() if select_all else []
    for i in range(len(tickers)):
        checkbox_vars[i].set(select_all)
    select_all_var.set(select_all)
    _select_all_updating = False
    redraw_chart()


def on_select_all_toggle():
    if _select_all_updating:
        return
    apply_select_all(select_all_var.get())


def apply_default_indices_visibility(use_defaults: bool):
    global tickers_to_show
    if use_defaults:
        tickers_to_show = [t for t in tickers if t in RRG_DEFAULT_VISIBLE]
    else:
        tickers_to_show = tickers.copy()
    for i, etf in enumerate(tickers):
        checkbox_vars[i].set(etf in tickers_to_show)
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
_TABLE_COL_CHARS = [9, 34, 9, 6, 9]
_TABLE_CELL_PADX = (2, 1)
_TABLE_ROW_PADY = 1
_TABLE_FONT = ('Arial', 10)
_TABLE_FONT_BOLD = ('Arial', 10, 'bold')
_TABLE_NEUTRAL_BG = root.cget('bg')
_VISIBLE_COL = 4


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

headers = ['Symbol', 'Name', 'Price', 'Change', 'Visible']
for j in range(len(headers)):
    if j == _VISIBLE_COL:
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
        anchor = 'e' if j in (2, 3) else 'w'
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
    global tickers_data
    symbol = event.widget.get().strip()
    if not symbol.endswith('.NS'):
        symbol = f'{symbol}.NS'
    row = int(event.widget.grid_info()['row'])
    try:
        series = load_price_history(symbol)
        if len(series) < min_weekly_points:
            raise ValueError('insufficient weekly history')
        tickers_data[symbol] = series
        try:
            ticker = yf.Ticker(symbol).info
            long_name = ticker.get('longName', symbol)
        except Exception:
            long_name = symbol
        index_name = ETF_TO_NSE_INDEX.get(symbol)
        display_name = (
            f"{symbol.replace('.NS', '')} → {index_name}"
            if index_name
            else long_name
        )
        # If previous symbol is in the ticker to show list, replace it with the new symbol 
        previous_symbol = tickers[row]

        if previous_symbol in tickers_to_show:
            tickers_to_show.remove(previous_symbol)

        tickers[row] = symbol

        if checkbox_vars[row].get() and symbol not in tickers_to_show:
            tickers_to_show.append(symbol)

        # Check if symbol is in the metadata dictionary 
        tickers_metadata_dict['symbol'][row] = symbol.replace('.NS', '')
        tickers_metadata_dict['name'][row] = display_name
        table_body.grid_slaves(row=row, column=1)[0].config(text=display_name)
        update_rrg()
        redraw_chart()
    except Exception as e:
        print(e)
        # Reset the entry to the previous symbol
        entry = event.widget
        row = int(entry.grid_info()['row'])
        entry.delete(0, tk.END)
        entry.insert(0, tickers_metadata_dict['symbol'][row])

def on_visibility_toggle(row_idx):
    global tickers_to_show
    if default_indices_var.get():
        default_indices_var.set(False)
    symbol = tickers[row_idx]
    if checkbox_vars[row_idx].get():
        if symbol not in tickers_to_show:
            tickers_to_show.append(symbol)
    else:
        tickers_to_show = [t for t in tickers_to_show if t != symbol]
    _sync_select_all_checkbox()
    redraw_chart()

def on_enter(event):
    ticker_name = event.widget.cget('text')
    event.widget.configure(text=ticker_name)

def on_leave(event):
    event.widget.configure(text='')

checkbox_vars = []

for i in range(len(tickers)):
    etf = tickers[i]
    symbol = tickers_metadata_dict['symbol'][i]
    name = tickers_metadata_dict['name'][i]
    price = round(tickers_data[etf][end_date], 2)
    chg = round((price - tickers_data[etf][start_date]) / tickers_data[etf][start_date] * 100, 1)
    bg_color = get_color(rsr_tickers[i].iloc[-1], rsm_tickers[i].iloc[-1])
    fg_color = 'white' if bg_color in ['red', 'green'] else 'black'
    symbol_var = tk.StringVar()
    symbol_var.set(symbol)
    entry = tk.Entry(
        table_body,
        textvariable=symbol_var,
        width=_TABLE_COL_CHARS[0],
        relief=tk.RIDGE,
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    )
    entry.grid(row=i, column=0, sticky='ew', padx=_TABLE_CELL_PADX, pady=_TABLE_ROW_PADY)
    entry.bind('<Return>', update_entry)
    tk.Label(
        table_body,
        text=name,
        width=_TABLE_COL_CHARS[1],
        relief=tk.RIDGE,
        anchor='w',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    ).grid(row=i, column=1, sticky='ew', padx=_TABLE_CELL_PADX, pady=_TABLE_ROW_PADY)
    tk.Label(
        table_body,
        text=price,
        width=_TABLE_COL_CHARS[2],
        relief=tk.RIDGE,
        anchor='e',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    ).grid(row=i, column=2, sticky='ew', padx=_TABLE_CELL_PADX, pady=_TABLE_ROW_PADY)
    tk.Label(
        table_body,
        text=chg,
        width=_TABLE_COL_CHARS[3],
        relief=tk.RIDGE,
        anchor='e',
        bg=bg_color,
        fg=fg_color,
        font=_TABLE_FONT,
    ).grid(row=i, column=3, sticky='ew', padx=_TABLE_CELL_PADX, pady=_TABLE_ROW_PADY)
    visible_cell = tk.Frame(
        table_body, relief=tk.RIDGE, bg=_TABLE_NEUTRAL_BG, highlightthickness=0
    )
    visible_cell.grid(row=i, column=4, sticky='ew', padx=_TABLE_CELL_PADX, pady=_TABLE_ROW_PADY)
    checkbox_var = tk.BooleanVar(value=tickers[i] in tickers_to_show)
    checkbox_vars.append(checkbox_var)
    checkbox = ttk.Checkbutton(
        visible_cell,
        variable=checkbox_var,
        command=lambda idx=i: on_visibility_toggle(idx),
    )
    checkbox.pack(side=tk.LEFT)

select_all_cb.config(command=on_select_all_toggle)
_sync_select_all_checkbox()
root.update_idletasks()
_sync_header_scroll_gutter()
_sync_table_layout()


scatter_plots = [None] * len(tickers)
line_plots = [None] * len(tickers)
head_arrows = [None] * len(tickers)
annotations = [None] * len(tickers)


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
    update_next_week_button()
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
    end_date = rsr_tickers[0].index[end_date_idx]
    start_date = rsr_tickers[0].index[end_date_idx - tail]

    for j in range(len(tickers)):
        remove_ticker_artists(j)
        try:
            price = round(tickers_data[tickers[j]][end_date], 2)
            chg = round(
                (price - tickers_data[tickers[j]][start_date])
                / tickers_data[tickers[j]][start_date]
                * 100,
                1,
            )
            table_body.grid_slaves(row=j, column=2)[0].config(text=price)
            table_body.grid_slaves(row=j, column=3)[0].config(text=chg)
            bg_color = get_color(rsr_tickers[j][end_date], rsm_tickers[j][end_date])
            fg_color = 'white' if bg_color in ['red', 'green', 'blue'] else 'black'
            for k in range(4):
                cell = table_body.grid_slaves(row=j, column=k)[0]
                cell.config(bg=bg_color, fg=fg_color)
        except (tk.TclError, IndexError, KeyError):
            pass

        if tickers[j] not in tickers_to_show:
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
            tickers[j],
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