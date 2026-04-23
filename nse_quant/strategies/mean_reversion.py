"""Mean reversion using RSI + z-score of price vs long MA, fundamentals filter."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nse_quant.indicators.technical import _rsi, _sma
from nse_quant.strategies.base import Strategy, StrategyResult


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        rsi_n: int = 14,
        rsi_lower: float = 30.0,
        rsi_upper: float = 70.0,
        zscore_window: int = 20,
        zscore_entry: float = -1.5,
        zscore_exit: float = 0.0,
        pe_max: float = 18.0,
        top_k: int = 10,
    ) -> None:
        super().__init__(
            rsi_n=rsi_n,
            rsi_lower=rsi_lower,
            rsi_upper=rsi_upper,
            zscore_window=zscore_window,
            zscore_entry=zscore_entry,
            zscore_exit=zscore_exit,
            pe_max=pe_max,
            top_k=top_k,
        )

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        n = self.params["zscore_window"]
        ma = prices.apply(lambda s: _sma(s, n))
        sd = prices.rolling(n).std()
        z = (prices - ma) / sd.replace(0, np.nan)
        rsi_df = prices.apply(lambda s: _rsi(s, self.params["rsi_n"]))

        entry = (z <= self.params["zscore_entry"]) & (rsi_df <= self.params["rsi_lower"])
        exit_ = (z >= self.params["zscore_exit"]) | (rsi_df >= self.params["rsi_upper"])

        # Fundamentals filter: exclude expensive names.
        if fundamentals is not None and "pe" in fundamentals.columns:
            allowed = set(fundamentals.loc[fundamentals["pe"] <= self.params["pe_max"], "ticker"])
        else:
            allowed = set(prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        active: dict[str, bool] = dict.fromkeys(prices.columns, False)
        for dt in prices.index:
            for t in prices.columns:
                if t not in allowed:
                    continue
                if entry.at[dt, t] if (dt in entry.index and t in entry.columns) else False:
                    active[t] = True
                if exit_.at[dt, t] if (dt in exit_.index and t in exit_.columns) else False:
                    active[t] = False
            longs = [t for t, v in active.items() if v]
            if longs:
                # Equal-weight the active names, capped to top_k by most-oversold z.
                if len(longs) > self.params["top_k"]:
                    snap = z.loc[dt, longs]
                    longs = snap.nsmallest(self.params["top_k"]).index.tolist()
                weights.loc[dt, longs] = 1.0 / len(longs)
        return StrategyResult(name=self.name, weights=weights)
