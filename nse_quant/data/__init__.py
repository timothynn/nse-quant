"""Data access layer: sample data, yfinance fetcher, and NSE scraper."""
from nse_quant.data.fetcher import DataFetcher, load_prices, load_universe
from nse_quant.data.fundamentals import load_fundamentals
from nse_quant.data.sample import (
    SAMPLE_FUNDAMENTALS,
    SAMPLE_PRICES_DIR,
    ensure_sample_data,
)

__all__ = [
    "DataFetcher",
    "load_prices",
    "load_universe",
    "load_fundamentals",
    "ensure_sample_data",
    "SAMPLE_FUNDAMENTALS",
    "SAMPLE_PRICES_DIR",
]
