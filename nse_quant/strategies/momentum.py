"""Momentum rotation: long top decile of past-return performers, short bottom."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult


class MomentumRotationStrategy(Strategy):
    name = "momentum_rotation"

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 21,
        top_q: float = 0.2,
        bottom_q: float = 0.2,
        long_only: bool = True,
        rebalance: str = "ME",
    ) -> None:
        super().__init__(
            lookback=lookback, skip=skip, top_q=top_q, bottom_q=bottom_q,
            long_only=long_only, rebalance=rebalance,
        )

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        lookback = self.params["lookback"]
        skip = self.params["skip"]
        top_q = self.params["top_q"]
        bottom_q = self.params["bottom_q"]
        long_only = self.params["long_only"]

        ret = prices.pct_change().fillna(0)
        cum = (1 + ret).rolling(lookback).apply(np.prod, raw=True) - 1
        cum = cum.shift(skip)  # skip the last month to avoid short-term reversal
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        rebalance_dates = prices.resample(self.params["rebalance"]).last().index.intersection(prices.index)
        last_w: pd.Series | None = None
        for dt in prices.index:
            if dt in rebalance_dates:
                snap = cum.loc[dt].dropna()
                if len(snap) < 5:
                    last_w = None
                    continue
                ranks = snap.rank(pct=True)
                longs = snap.index[ranks >= 1 - top_q]
                shorts = snap.index[ranks <= bottom_q]
                w = pd.Series(0.0, index=prices.columns)
                if len(longs):
                    w.loc[longs] = 1.0 / len(longs)
                if not long_only and len(shorts):
                    w.loc[shorts] = -1.0 / len(shorts)
                last_w = w
            if last_w is not None:
                weights.loc[dt] = last_w
        return StrategyResult(name=self.name, weights=weights)
