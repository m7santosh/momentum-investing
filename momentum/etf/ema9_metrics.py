"""9 EMA cross metrics: hold/exit flag, cross date, % gain since cross bar."""

from __future__ import annotations

import pandas as pd

ETF_EMA_9_SPAN = 9


def _most_recent_cross_above_ts(close: pd.Series, ema9: pd.Series) -> pd.Timestamp | None:
    """Return the session when Close last crossed from below 9 EMA to at/above it."""
    above = close >= ema9
    cross_ts: pd.Timestamp | None = None
    for i in range(1, len(above)):
        if bool(above.iloc[i]) and not bool(above.iloc[i - 1]):
            cross_ts = pd.Timestamp(above.index[i])
    if cross_ts is None and bool(above.iloc[-1]) and not (~above).any():
        # Above on every bar in the lookback (no prior below session in window).
        cross_ts = pd.Timestamp(above.index[0])
    return cross_ts


def compute_ema9_metrics(close: pd.Series, *, span: int = ETF_EMA_9_SPAN) -> dict:
    """Compute 9 EMA metrics from a regular Close series (oldest → newest).

    - ``close_below_9ema``: Hold if last Close >= 9 EMA, else Exit.
    - ``above_9ema_since``: YYYY-MM-DD of the most recent cross from below to at/above
      9 EMA (Close on that bar vs 9 EMA on prior bar). If the ETF was above 9 EMA on
      every bar in the lookback, the first bar's date is used.
    - ``pct_since_cross``: (last Close / Close on cross bar - 1) * 100.
    """
    close = close.dropna()
    if close.empty:
        return {
            "last_close": float("nan"),
            "ema9_close": float("nan"),
            "close_below_9ema": "",
            "above_9ema_since": None,
            "pct_since_cross": float("nan"),
        }

    ema9 = close.ewm(span=span, adjust=False).mean()
    last_close = float(close.iloc[-1])
    ema9_close = float(ema9.iloc[-1])
    close_below = "Exit" if last_close < ema9_close else "Hold"

    cross_ts = _most_recent_cross_above_ts(close, ema9)

    above_since: str | None = None
    pct_since = float("nan")
    if cross_ts is not None:
        cross_close = float(close.loc[cross_ts])
        above_since = cross_ts.strftime("%Y-%m-%d")
        if cross_close > 0:
            pct_since = round((last_close / cross_close - 1) * 100, 2)

    return {
        "last_close": round(last_close, 2),
        "ema9_close": round(ema9_close, 2),
        "close_below_9ema": close_below,
        "above_9ema_since": above_since,
        "pct_since_cross": pct_since,
    }
