"""Dividend aristocrats: screen for high, stable dividend yield + payout."""
from __future__ import annotations

from typing import Any

import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult


class DividendAristocratsStrategy(Strategy):
    name = "dividend_aristocrats"

    def __init__(
        self,
        min_yield: float = 0.04,
        max_payout: float = 0.85,
        min_roe: float = 0.08,
        top_k: int = 8,
        max_weight: float = 0.2,
    ) -> None:
        super().__init__(
            min_yield=min_yield, max_payout=max_payout, min_roe=min_roe,
            top_k=top_k, max_weight=max_weight,
        )

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        if fundamentals is None:
            return StrategyResult(
                name=self.name,
                weights=pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
                metadata={"error": "fundamentals required"},
            )
        f = fundamentals.copy()
        screen = (
            (f["dividend_yield"].fillna(0) >= self.params["min_yield"])
            & (f["payout_ratio"].fillna(1) <= self.params["max_payout"])
            & (f["roe"].fillna(0) >= self.params["min_roe"])
        )
        selected = f.loc[screen].sort_values("dividend_yield", ascending=False).head(self.params["top_k"])
        tickers = [t for t in selected["ticker"] if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if not tickers:
            return StrategyResult(name=self.name, weights=weights, metadata={"selected": []})
        per = min(self.params["max_weight"], 1.0 / len(tickers))
        # Hold constant over the whole window (buy-and-hold aristocrats).
        for t in tickers:
            weights[t] = per
        return StrategyResult(name=self.name, weights=weights, metadata={"selected": tickers})
