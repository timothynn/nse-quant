"""Lightweight NSE website scraper.

The NSE publishes a public equity statistics page with daily snapshots of
listed equities. This scraper pulls that page and normalizes it to a pandas
DataFrame. It is intentionally forgiving — if the site layout changes, the
caller falls back to yfinance / sample data.

Only HEADLESS requests are made (no browser) to comply with the session
webscraping policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NSE_EQUITY_STATS_URL = "https://www.nse.co.ke/market-statistics/equity-statistics/"
REQUEST_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (compatible; nse-quant/0.1; +https://github.com/timothynn/nse-quant)"
)


@dataclass
class NSEWebScraper:
    url: str = NSE_EQUITY_STATS_URL
    timeout: int = REQUEST_TIMEOUT

    def fetch_equity_snapshot(self) -> pd.DataFrame:
        """Fetch the latest NSE equity snapshot table (best-effort)."""
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
        try:
            resp = requests.get(self.url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("NSE scrape failed: %s", exc)
            return pd.DataFrame()

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if table is None:
            logger.warning("No table found at %s", self.url)
            return pd.DataFrame()

        try:
            dfs = pd.read_html(str(table))
        except ValueError as exc:
            logger.warning("Failed to parse NSE table: %s", exc)
            return pd.DataFrame()
        if not dfs:
            return pd.DataFrame()

        df = dfs[0]
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df
