"""Price data fetcher for NSE tickers.

Resolution order:
  1. In-memory cache
  2. On-disk cache (Parquet/CSV)
  3. yfinance (live)
  4. Bundled sample data (offline fallback)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nse_quant.config import CACHE_DIR, NSE_UNIVERSE
from nse_quant.data.sample import ensure_sample_data, sample_price_path

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def load_universe() -> pd.DataFrame:
    rows = [{"ticker": t, **meta} for t, meta in NSE_UNIVERSE.items()]
    return pd.DataFrame(rows)


@dataclass
class DataFetcher:
    """Unified OHLCV fetcher with live + offline fallback."""

    use_live: bool = False
    cache_dir: Path = CACHE_DIR

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, pd.DataFrame] = {}
        ensure_sample_data()

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.replace('.', '_')}.parquet"

    def _load_sample(self, ticker: str) -> pd.DataFrame:
        path = sample_price_path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"No sample data for {ticker}")
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        df.index.name = "date"
        return df

    def _load_yfinance(self, ticker: str, start: str | None, end: str | None) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed; skipping live fetch")
            return None
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return None
        if raw is None or raw.empty:
            return None
        # Normalize columns across yfinance versions.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.rename(columns={c: c.lower().replace(" ", "_") for c in raw.columns})
        for col in PRICE_COLUMNS:
            if col not in raw.columns:
                raw[col] = pd.NA
        raw["dividend"] = 0.0
        raw.index.name = "date"
        return raw[[*PRICE_COLUMNS, "dividend"]].sort_index()

    def get(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        if not refresh and ticker in self._memory:
            df = self._memory[ticker]
        else:
            cache = self._cache_path(ticker)
            if cache.exists() and not refresh:
                df = pd.read_parquet(cache)
            elif self.use_live:
                live = self._load_yfinance(ticker, start, end)
                if live is not None and not live.empty:
                    df = live
                    try:
                        df.to_parquet(cache)
                    except Exception:  # pyarrow may be absent
                        df.to_csv(cache.with_suffix(".csv"))
                else:
                    df = self._load_sample(ticker)
            else:
                df = self._load_sample(ticker)
            self._memory[ticker] = df
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df.copy()

    def get_many(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        column: str = "close",
    ) -> pd.DataFrame:
        tickers = tickers or list(NSE_UNIVERSE.keys())
        frames = {}
        for t in tickers:
            try:
                df = self.get(t, start=start, end=end)
                frames[t] = df[column]
            except FileNotFoundError:
                logger.debug("Skipping unavailable ticker %s", t)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1).sort_index()
        out.columns.name = "ticker"
        return out

    def returns(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        prices = self.get_many(tickers=tickers, start=start, end=end, column="adj_close")
        return prices.pct_change().dropna(how="all")


def load_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    use_live: bool = False,
) -> pd.DataFrame:
    return DataFetcher(use_live=use_live).get_many(tickers=tickers, start=start, end=end)
