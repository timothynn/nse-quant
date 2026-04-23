"""Sector rotation by cross-sectional momentum across NSE sectors."""
from __future__ import annotations

from typing import Any

import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult


class SectorRotationStrategy(Strategy):
    name = "sector_rotation"

    def __init__(
        self,
        lookback: int = 63,
        top_sectors: int = 3,
        rebalance: str = "ME",
        max_sector_weight: float = 0.35,
    ) -> None:
        super().__init__(
            lookback=lookback, top_sectors=top_sectors, rebalance=rebalance,
            max_sector_weight=max_sector_weight,
        )

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        if universe is None or "sector" not in universe.columns:
            return StrategyResult(
                name=self.name,
                weights=pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
                metadata={"error": "universe missing sector column"},
            )
        sector_map = universe.set_index("ticker")["sector"].to_dict()
        lookback = self.params["lookback"]
        top_sectors = self.params["top_sectors"]
        max_w = self.params["max_sector_weight"]

        ret = prices.pct_change().fillna(0)
        # Equal-weighted sector return.
        sector_returns = {}
        for sector in set(sector_map.values()):
            members = [t for t, s in sector_map.items() if s == sector and t in ret.columns]
            if members:
                sector_returns[sector] = ret[members].mean(axis=1)
        sector_ret_df = pd.DataFrame(sector_returns)
        sector_cum = (1 + sector_ret_df).rolling(lookback).apply(lambda x: x.prod(), raw=True) - 1

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        rebalance_dates = prices.resample(self.params["rebalance"]).last().index.intersection(prices.index)
        last_w: pd.Series | None = None
        for dt in prices.index:
            if dt in rebalance_dates and dt in sector_cum.index:
                snap = sector_cum.loc[dt].dropna()
                if snap.empty:
                    last_w = None
                    continue
                winners = snap.nlargest(top_sectors).index.tolist()
                w = pd.Series(0.0, index=prices.columns)
                per_sector = min(max_w, 1.0 / len(winners))
                for sector in winners:
                    members = [t for t, s in sector_map.items() if s == sector and t in w.index]
                    if members:
                        w.loc[members] = per_sector / len(members)
                last_w = w
            if last_w is not None:
                weights.loc[dt] = last_w
        return StrategyResult(name=self.name, weights=weights)
