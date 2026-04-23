"""Volatility arbitrage: long low-volatility, short high-volatility (risk-parity style)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult


class VolatilityArbitrageStrategy(Strategy):
    name = "volatility_arb"

    def __init__(
        self,
        vol_window: int = 60,
        top_q: float = 0.3,
        bottom_q: float = 0.3,
        long_only: bool = True,
        rebalance: str = "W-FRI",
    ) -> None:
        super().__init__(
            vol_window=vol_window, top_q=top_q, bottom_q=bottom_q,
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
        window = self.params["vol_window"]
        ret = prices.pct_change()
        realized_vol = ret.rolling(window).std() * np.sqrt(252)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        rebalance_dates = prices.resample(self.params["rebalance"]).last().index.intersection(prices.index)
        last_w: pd.Series | None = None
        for dt in prices.index:
            if dt in rebalance_dates:
                snap = realized_vol.loc[dt].dropna()
                if len(snap) < 5:
                    last_w = None
                    continue
                ranks = snap.rank(pct=True)
                low_vol = snap.index[ranks <= self.params["bottom_q"]]
                high_vol = snap.index[ranks >= 1 - self.params["top_q"]]
                w = pd.Series(0.0, index=prices.columns)
                if len(low_vol):
                    # Inverse-vol weights within the low-vol bucket
                    iv = 1.0 / snap.loc[low_vol]
                    w.loc[low_vol] = (iv / iv.sum()).values
                if not self.params["long_only"] and len(high_vol):
                    w.loc[high_vol] = -0.5 / len(high_vol)
                    # Re-normalize long side to sum to 1.0
                    long_mask = w > 0
                    w[long_mask] = w[long_mask] / w[long_mask].sum() * 1.0
                last_w = w
            if last_w is not None:
                weights.loc[dt] = last_w
        return StrategyResult(name=self.name, weights=weights)
