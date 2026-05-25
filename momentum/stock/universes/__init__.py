"""Hardcoded stock universes — edit tickers in each module under this package."""

from momentum.stock.universes import bse_largemidcap, nifty_largemidcap, n500, quality

# RRG / CLI lookup by key (add new modules here when you create them)
BY_KEY = {
    quality.KEY: quality,
    n500.KEY: n500,
    bse_largemidcap.KEY: bse_largemidcap,
    nifty_largemidcap.KEY: nifty_largemidcap,
}

DEFAULT_KEY = quality.KEY
ENV_UNIVERSE_KEY = "STOCK_UNIVERSE"

__all__ = [
    "BY_KEY",
    "DEFAULT_KEY",
    "ENV_UNIVERSE_KEY",
    "bse_largemidcap",
    "nifty_largemidcap",
    "n500",
    "quality",
]
