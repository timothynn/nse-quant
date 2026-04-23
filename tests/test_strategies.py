"""Strategy smoke tests — each strategy must produce a valid weights matrix."""
from __future__ import annotations

import pandas as pd
import pytest

from nse_quant.data import DataFetcher, load_fundamentals, load_universe
from nse_quant.strategies import STRATEGY_REGISTRY


@pytest.fixture(scope="module")
def panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fetcher = DataFetcher(use_live=False)
    universe = load_universe()
    tickers = universe["ticker"].tolist()[:15]
    prices = fetcher.get_many(tickers=tickers, column="adj_close").tail(800)
    return prices, load_fundamentals(), universe


@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY.keys()))
def test_strategy_produces_weights(name: str, panel: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    prices, fundamentals, universe = panel
    strat = STRATEGY_REGISTRY[name]()
    result = strat.generate(prices, fundamentals=fundamentals, universe=universe)
    assert result.weights.index.equals(prices.index)
    assert set(result.weights.columns) == set(prices.columns)
    assert result.weights.abs().sum(axis=1).max() <= 1.5 + 1e-6  # gross bounded
