"""Generate deterministic synthetic OHLCV for the NSE universe.

This bundled sample data exists for two reasons:
  1. The app must work offline and in CI where yfinance/NSE are unreachable.
  2. Live NSE data is not always available via free APIs.

The synthetic series are generated with geometric Brownian motion seeded per
ticker so runs are reproducible. Sectors share a common factor to make
correlation-based tests meaningful. This is explicitly *simulated* data —
use the yfinance fetcher for real prices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nse_quant.config import NSE_UNIVERSE, SAMPLE_DATA_DIR

SAMPLE_PRICES_DIR = SAMPLE_DATA_DIR / "prices"
SAMPLE_FUNDAMENTALS = SAMPLE_DATA_DIR / "fundamentals.csv"
SAMPLE_MANIFEST = SAMPLE_DATA_DIR / "manifest.csv"

DEFAULT_START = "2020-01-02"
DEFAULT_END = "2025-12-31"

_SECTOR_PARAMS = {
    "Telecom":      {"drift": 0.10,  "vol": 0.22, "start": 38.0},
    "Banking":      {"drift": 0.08,  "vol": 0.28, "start": 40.0},
    "Consumer":     {"drift": 0.06,  "vol": 0.25, "start": 150.0},
    "Construction": {"drift": 0.03,  "vol": 0.30, "start": 45.0},
    "Energy":       {"drift": 0.05,  "vol": 0.32, "start": 5.0},
    "Insurance":    {"drift": 0.04,  "vol": 0.30, "start": 15.0},
    "Agriculture":  {"drift": 0.05,  "vol": 0.35, "start": 30.0},
    "Media":        {"drift": 0.02,  "vol": 0.34, "start": 20.0},
    "Investment":   {"drift": 0.06,  "vol": 0.31, "start": 10.0},
    "Industrial":   {"drift": 0.04,  "vol": 0.29, "start": 25.0},
}


def _ticker_seed(ticker: str) -> int:
    return int.from_bytes(ticker.encode(), "little") % (2**32 - 1)


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    # NSE is closed on weekends and Kenyan public holidays.
    from nse_quant.config import NSE_HOLIDAYS_STATIC
    idx = pd.bdate_range(start=start, end=end)
    holidays = pd.to_datetime(NSE_HOLIDAYS_STATIC)
    return idx.difference(holidays)


def _simulate_ticker(
    ticker: str, sector: str, dates: pd.DatetimeIndex, sector_factor: np.ndarray
) -> pd.DataFrame:
    params = _SECTOR_PARAMS.get(sector, {"drift": 0.05, "vol": 0.30, "start": 20.0})
    rng = np.random.default_rng(_ticker_seed(ticker))
    n = len(dates)
    dt = 1.0 / 252.0
    idio = rng.normal(0, 1, n)
    # 60% idiosyncratic + 40% sector beta
    shocks = 0.6 * idio + 0.4 * sector_factor
    log_returns = (params["drift"] - 0.5 * params["vol"] ** 2) * dt + params["vol"] * np.sqrt(dt) * shocks
    price = params["start"] * np.exp(np.cumsum(log_returns))
    # Construct OHLC with intraday noise.
    intraday = rng.normal(0, params["vol"] * np.sqrt(dt) * 0.5, n)
    high = price * (1 + np.abs(intraday))
    low = price * (1 - np.abs(intraday))
    open_ = np.concatenate([[price[0]], price[:-1]]) * (1 + rng.normal(0, 0.002, n))
    close = price
    vol = rng.integers(50_000, 5_000_000, n).astype(float)
    # Occasional dividends for banks/telecom/consumer
    dividend = np.zeros(n)
    if sector in {"Banking", "Telecom", "Consumer", "Insurance"}:
        qtr_ix = np.where(pd.Series(dates).dt.is_quarter_end.values)[0]
        # Two payouts per year
        for i, ix in enumerate(qtr_ix):
            if i % 2 == 0:
                dividend[ix] = close[ix] * rng.uniform(0.01, 0.025)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_, 2),
            "high": np.round(np.maximum.reduce([open_, high, close]), 2),
            "low": np.round(np.minimum.reduce([open_, low, close]), 2),
            "close": np.round(close, 2),
            "adj_close": np.round(close, 2),
            "volume": vol.astype(int),
            "dividend": np.round(dividend, 4),
        }
    )


def _simulate_fundamentals(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for ticker, meta in NSE_UNIVERSE.items():
        sector = meta["sector"]
        row = {
            "ticker": ticker,
            "name": meta["name"],
            "sector": sector,
            "pe": float(rng.uniform(4, 22)),
            "pb": float(rng.uniform(0.4, 4.0)),
            "roe": float(rng.uniform(0.03, 0.28)),
            "roic": float(rng.uniform(0.02, 0.22)),
            "debt_to_equity": float(rng.uniform(0.05, 2.0)),
            "dividend_yield": float(rng.uniform(0.0, 0.12)),
            "payout_ratio": float(rng.uniform(0.0, 0.9)),
            "operating_margin": float(rng.uniform(0.02, 0.40)),
            "revenue_growth_3y": float(rng.uniform(-0.1, 0.25)),
            "eps_growth_3y": float(rng.uniform(-0.2, 0.35)),
            "market_cap_kes_bn": float(rng.uniform(2, 500)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def generate_sample_data(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
) -> pd.DataFrame:
    """Generate and persist sample OHLCV + fundamentals. Returns the manifest."""
    SAMPLE_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_MANIFEST.exists() and not force:
        return pd.read_csv(SAMPLE_MANIFEST)

    dates = _business_days(start, end)
    rng_global = np.random.default_rng(42)
    n = len(dates)

    # Sector factor = cumulative return driven by a common latent process.
    sector_factors: dict[str, np.ndarray] = {}
    for sector in _SECTOR_PARAMS:
        sector_factors[sector] = rng_global.normal(0, 1, n)

    manifest_rows = []
    for ticker, meta in NSE_UNIVERSE.items():
        df = _simulate_ticker(ticker, meta["sector"], dates, sector_factors[meta["sector"]])
        path = SAMPLE_PRICES_DIR / f"{ticker.replace('.', '_')}.csv"
        df.to_csv(path, index=False)
        manifest_rows.append(
            {"ticker": ticker, "name": meta["name"], "sector": meta["sector"], "path": str(path.name)}
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SAMPLE_MANIFEST, index=False)

    fundamentals = _simulate_fundamentals(rng_global)
    fundamentals.to_csv(SAMPLE_FUNDAMENTALS, index=False)
    return manifest


def ensure_sample_data() -> Path:
    """Ensure sample data exists on disk. Returns manifest path."""
    if not SAMPLE_MANIFEST.exists():
        generate_sample_data()
    return SAMPLE_MANIFEST


def sample_price_path(ticker: str) -> Path:
    return SAMPLE_PRICES_DIR / f"{ticker.replace('.', '_')}.csv"
