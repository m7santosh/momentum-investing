"""Hardcoded ETF universes — edit tickers in india.py, us.py, or add a new module."""

from momentum.etf.universes import india, us

BY_KEY = {
    india.KEY: india,
    us.KEY: us,
}

DEFAULT_KEY = india.KEY

__all__ = ["BY_KEY", "DEFAULT_KEY", "india", "us"]
