"""Quality factor: long-only screen on ROIC, margin, leverage."""
from __future__ import annotations

from typing import Any

import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult


class QualityFactorStrategy(Strategy):
    name = "quality_factor"

    def __init__(
        self,
        min_roic: float = 0.10,
        min_op_margin: float = 0.10,
        max_debt_equity: float = 1.2,
        min_eps_growth: float = 0.05,
        top_k: int = 8,
    ) -> None:
        super().__init__(
            min_roic=min_roic, min_op_margin=min_op_margin,
            max_debt_equity=max_debt_equity, min_eps_growth=min_eps_growth,
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
        if fundamentals is None:
            return StrategyResult(
                name=self.name,
                weights=pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
                metadata={"error": "fundamentals required"},
            )
        f = fundamentals.copy()
        screen = (
            (f["roic"].fillna(0) >= self.params["min_roic"])
            & (f["operating_margin"].fillna(0) >= self.params["min_op_margin"])
            & (f["debt_to_equity"].fillna(5) <= self.params["max_debt_equity"])
            & (f["eps_growth_3y"].fillna(-1) >= self.params["min_eps_growth"])
        )
        # Composite quality score (z-score across four fields, sign-adjusted).
        if screen.any():
            pool = f.loc[screen].copy()
            for col in ["roic", "operating_margin", "eps_growth_3y"]:
                sd = pool[col].std(ddof=0)
                sd = sd if sd and not pd.isna(sd) else 1.0
                pool[col + "_z"] = (pool[col] - pool[col].mean()) / sd
            de_sd = pool["debt_to_equity"].std(ddof=0)
            de_sd = de_sd if de_sd and not pd.isna(de_sd) else 1.0
            pool["de_z"] = -(pool["debt_to_equity"] - pool["debt_to_equity"].mean()) / de_sd
            pool["score"] = pool[["roic_z", "operating_margin_z", "eps_growth_3y_z", "de_z"]].mean(axis=1)
            selected = pool.sort_values("score", ascending=False).head(self.params["top_k"])
            tickers = [t for t in selected["ticker"] if t in prices.columns]
        else:
            tickers = []
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if tickers:
            per = 1.0 / len(tickers)
            for t in tickers:
                weights[t] = per
        return StrategyResult(name=self.name, weights=weights, metadata={"selected": tickers})
