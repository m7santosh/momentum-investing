"""Hardcoded ETF universes — edit tickers in india.py, us.py, or add a new module."""

from momentum.etf.universes import india, us, us_liquid_candidates, us_universe

BY_KEY = {
    india.KEY: india,
    us.KEY: us,
    us_liquid_candidates.KEY: us_liquid_candidates,
}

DEFAULT_KEY = india.KEY

__all__ = ["BY_KEY", "DEFAULT_KEY", "india", "us", "us_liquid_candidates", "us_universe"]
