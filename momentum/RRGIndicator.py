import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import interpolate
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


is_playing = False
marker_size = []
tail = 5
end_date_idx = tail
start_date, end_date = None, None

for i in range(tail):
    if i == tail-1:
        marker_size.append(50)
    else:
        marker_size.append(10)

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
    
# Retrieve historical prices 
period = '1y'
requested_tickers = ['^NSEBANK', '^CNXENERGY', '^CNXMETAL', '^CNXAUTO', 'NIFTY_CAPITAL_MKT.NS', '^CNXPHARMA', 'NIFTY_RAILWAYSPSU.NS', 'NIFTY_OIL_AND_GAS.NS', 'NIFTY_CHEMICALS.NS']
tickers = requested_tickers.copy()
tickers_metadata_dict = {
    'symbol': [],
    'name': []
}

for i in range(len(tickers)):
    try:
        info = yf.Ticker(tickers[i]).info
        tickers_metadata_dict['symbol'].append(info.get('symbol', tickers[i]))
        tickers_metadata_dict['name'].append(info.get('longName', tickers[i]))
    except Exception:
        tickers_metadata_dict['symbol'].append(tickers[i])
        tickers_metadata_dict['name'].append(tickers[i])

tickers_to_show = tickers.copy()

benchmark = '^CRSLDX'

tickers_data = get_close_prices(yf.download(tickers, period=period, interval="1wk"))
benchmark_data = get_close_prices(yf.download(benchmark, period=period, interval="1wk")).squeeze()
window = 14

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
root.geometry('1000x650')
root.minsize(800, 500)
root.resizable(True, True)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(2, weight=0)

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

controls_frame = tk.Frame(root, height=88, padx=8, pady=6)
controls_frame.grid(row=1, column=0, sticky='ew')
controls_frame.grid_propagate(False)

date_max_idx = len(rsr_tickers[0]) - 2
end_date_idx = tail
start_date = rsr_tickers[0].index[0]
end_date = rsr_tickers[0].index[end_date_idx]


def format_date_label(idx):
    return str(rsr_tickers[0].index[int(idx)]).split(' ')[0]


def update_marker_sizes():
    global marker_size
    marker_size = [50 if i == tail - 1 else 10 for i in range(tail)]


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
    update_marker_sizes()
    date_value_label.config(text=format_date_label(end_date_idx))
    if not is_playing:
        redraw_chart()


def on_date_change(val):
    global end_date_idx
    end_date_idx = int(float(val))
    date_value_label.config(text=format_date_label(end_date_idx))
    if not is_playing:
        redraw_chart()


def toggle_play():
    global is_playing, _after_id
    is_playing = not is_playing
    play_button.config(text='Pause' if is_playing else 'Play')
    if is_playing:
        schedule_update()
    elif _after_id is not None:
        try:
            root.after_cancel(_after_id)
        except tk.TclError:
            pass
        _after_id = None
        redraw_chart()


play_button = ttk.Button(controls_frame, text='Play', width=10, command=toggle_play)
play_button.pack(side=tk.LEFT, padx=(0, 12))

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

table = tk.Frame(master=root)
table.grid(row=2, column=0, sticky='ew')

headers = ['Symbol', 'Name', 'Price', 'Change', 'Visible']
widths = [20, 40, 20, 20, 10]
for j in range(len(headers)):
    table.grid_columnconfigure(j, weight=widths[j])
    tk.Label(table, text=headers[j], relief=tk.RIDGE, width=widths[j], font=('Arial', 12, 'bold')).grid(
        row=0, column=j, sticky='ew'
    )

def update_entry(event):
    global tickers_data
    symbol = event.widget.get()
    # Check if the symbol exists with yahoo finance 
    try:
        ticker = yf.Ticker(symbol).info
        # Replace in tickers 
        row = event.widget.grid_info()['row']
        # replace dataframe column 
        tickers_data[symbol] = get_close_prices(yf.download(symbol, period=period, interval='1wk')).squeeze()
        # If previous symbol is in the ticker to show list, replace it with the new symbol 
        previous_symbol = tickers[row - 1]

        if previous_symbol in tickers_to_show:
            tickers_to_show.remove(previous_symbol)

        tickers[row - 1] = symbol

        if checkbox_vars[row - 1].get() and symbol not in tickers_to_show:
            tickers_to_show.append(symbol)

        # Check if symbol is in the metadata dictionary 
        if symbol not in tickers_metadata_dict['symbol']:
            # Add the symbol to the metadata dictionary
            tickers_metadata_dict['symbol'][row-1] = symbol
            tickers_metadata_dict['name'][row-1] = ticker['longName']

        # Update the name label 
        table.grid_slaves(row=row, column=1)[0].config(text=ticker['longName'])
        update_rrg()
        redraw_chart()
    except Exception as e:
        print(e)
        # Reset the entry to the previous symbol
        entry = event.widget
        row = entry.grid_info()['row']
        entry.delete(0, tk.END)
        entry.insert(0, tickers_metadata_dict['symbol'][row-1])

def on_visibility_toggle(row_idx):
    global tickers_to_show
    symbol = tickers[row_idx]
    if checkbox_vars[row_idx].get():
        if symbol not in tickers_to_show:
            tickers_to_show.append(symbol)
    else:
        tickers_to_show = [t for t in tickers_to_show if t != symbol]
    redraw_chart()

def on_enter(event):
    ticker_name = event.widget.cget('text')
    event.widget.configure(text=ticker_name)

def on_leave(event):
    event.widget.configure(text='')

checkbox_vars = []

for i in range(len(tickers_to_show)):
    # Ticker symbol 
    symbol = tickers_metadata_dict['symbol'][i]
    # Ticker name
    name = tickers_metadata_dict['name'][i]
    # Ticker price at end date
    price = round(tickers_data[symbol][end_date], 2)
    # Ticker change from start date to end date in percentage
    chg = round((price - tickers_data[symbol][start_date]) / tickers_data[symbol][start_date] * 100, 1)
    bg_color = get_color(rsr_tickers[i].iloc[-1], rsm_tickers[i].iloc[-1])
    fg_color = 'white' if bg_color in ['red', 'green'] else 'black'
    symbol_var = tk.StringVar()
    symbol_var.set(symbol)
    entry = tk.Entry(table, textvariable=symbol_var, relief=tk.RIDGE, width=20, bg=bg_color, fg=fg_color, font=('Arial', 12))
    entry.grid(row=i+1, column=0, sticky='ew')
    entry.bind('<Return>', update_entry)
    tk.Label(table, text=name, relief=tk.RIDGE, width=40, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(
        row=i+1, column=1, sticky='ew'
    )
    tk.Label(table, text=price, relief=tk.RIDGE, width=20, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(
        row=i+1, column=2, sticky='ew'
    )
    tk.Label(table, text=chg, relief=tk.RIDGE, width=20, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(
        row=i+1, column=3, sticky='ew'
    )
    checkbox_var = tk.BooleanVar(value=True)
    checkbox_vars.append(checkbox_var)
    checkbox = ttk.Checkbutton(
        table,
        variable=checkbox_var,
        command=lambda idx=i: on_visibility_toggle(idx),
    )
    checkbox.grid(row=i+1, column=4, sticky='ew')


scatter_plots = [None] * len(tickers)
line_plots = [None] * len(tickers)
annotations = [None] * len(tickers)


def remove_ticker_artists(j):
    for artists in (scatter_plots, line_plots, annotations):
        artist = artists[j]
        if artist is not None:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
            artists[j] = None


def redraw_chart():
    update_frame()
    if root.winfo_exists():
        try:
            canvas.draw_idle()
        except tk.TclError:
            pass

def update_frame():
    global start_date, end_date, end_date_idx

    if not root.winfo_exists():
        return

    if not is_playing:
        end_date_idx = int(date_scale.get())
        end_date = rsr_tickers[0].index[end_date_idx]
        start_date = rsr_tickers[0].index[end_date_idx - tail]
    else:
        start_date += pd.to_timedelta(1, unit='W')
        end_date += pd.to_timedelta(1, unit='W')

    if end_date == rsr_tickers[0].index[-1]:
        start_date = rsr_tickers[0].index[0]
        end_date = start_date + pd.to_timedelta(tail, unit='W')

    for j in range(len(tickers)):
        remove_ticker_artists(j)
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
        color = get_color(filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1])
        scatter_plots[j] = ax_rrg.scatter(
            filtered_rsr_tickers.values, filtered_rsm_tickers.values, color=color, s=marker_size
        )
        line_plots[j] = ax_rrg.plot(
            filtered_rsr_tickers.values, filtered_rsm_tickers.values, color='black', alpha=0.2
        )[0]
        annotations[j] = ax_rrg.annotate(
            tickers[j],
            (filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1]),
            fontsize=8,
        )

        try:
            price = round(tickers_data[tickers[j]][end_date], 2)
            chg = round((price - tickers_data[tickers[j]][start_date]) / tickers_data[tickers[j]][start_date] * 100, 1)
            table.grid_slaves(row=j+1, column=2)[0].config(text=price)
            table.grid_slaves(row=j+1, column=3)[0].config(text=chg)
            bg_color = get_color(rsr_tickers[j][end_date], rsm_tickers[j][end_date])
            fg_color = 'white' if bg_color in ['red', 'green', 'blue'] else 'black'
            for k in range(4):
                table.grid_slaves(row=j+1, column=k)[0].config(bg=bg_color, fg=fg_color)
        except (tk.TclError, IndexError):
            pass

_after_id = None


def schedule_update():
    global _after_id
    if not root.winfo_exists() or not is_playing:
        return
    update_frame()
    try:
        canvas.draw_idle()
    except tk.TclError:
        return
    _after_id = root.after(50, schedule_update)


def on_close():
    global _after_id
    if _after_id is not None:
        try:
            root.after_cancel(_after_id)
        except tk.TclError:
            pass
    plt.close(fig)
    root.quit()
    root.destroy()


root.protocol('WM_DELETE_WINDOW', on_close)
redraw_chart()
root.mainloop()