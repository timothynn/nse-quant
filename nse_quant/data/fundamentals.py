"""Fundamentals loader with yfinance + sample CSV fallback."""
from __future__ import annotations

import logging

import pandas as pd

from nse_quant.data.sample import SAMPLE_FUNDAMENTALS, ensure_sample_data

logger = logging.getLogger(__name__)

_FIELDS = {
    "trailingPE": "pe",
    "priceToBook": "pb",
    "returnOnEquity": "roe",
    "debtToEquity": "debt_to_equity",
    "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio",
    "operatingMargins": "operating_margin",
    "revenueGrowth": "revenue_growth_3y",
    "earningsGrowth": "eps_growth_3y",
    "marketCap": "market_cap_kes_bn",
}


def _load_sample_fundamentals() -> pd.DataFrame:
    ensure_sample_data()
    return pd.read_csv(SAMPLE_FUNDAMENTALS)


def _fetch_live_fundamentals(tickers: list[str]) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance info failed for %s: %s", t, exc)
            continue
        if not info:
            continue
        row = {"ticker": t, "name": info.get("longName", t)}
        for src, dst in _FIELDS.items():
            v = info.get(src)
            if v is None:
                continue
            row[dst] = float(v)
        if "market_cap_kes_bn" in row:
            row["market_cap_kes_bn"] /= 1e9
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_fundamentals(
    tickers: list[str] | None = None,
    use_live: bool = False,
) -> pd.DataFrame:
    if use_live and tickers:
        live = _fetch_live_fundamentals(tickers)
        if live is not None and not live.empty:
            return live
    df = _load_sample_fundamentals()
    if tickers:
        df = df[df["ticker"].isin(tickers)].reset_index(drop=True)
    return df
