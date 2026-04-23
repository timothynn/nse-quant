"""Pairs trading on cointegrated NSE pairs (Engle-Granger)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult


def _engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Return (hedge_ratio, pvalue) for cointegration between y and x."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return float("nan"), float("nan")
    common = pd.concat([y, x], axis=1).dropna()
    if len(common) < 60:
        return float("nan"), float("nan")
    y_, x_ = common.iloc[:, 0], common.iloc[:, 1]
    beta = np.polyfit(x_.values, y_.values, 1)[0]
    try:
        _, pvalue, _ = coint(y_, x_)
    except Exception:  # noqa: BLE001
        pvalue = 1.0
    return float(beta), float(pvalue)


class PairsTradingStrategy(Strategy):
    name = "pairs_trading"

    def __init__(
        self,
        formation_window: int = 252,
        trading_window: int = 126,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        pvalue_max: float = 0.05,
        top_pairs: int = 5,
    ) -> None:
        super().__init__(
            formation_window=formation_window,
            trading_window=trading_window,
            entry_z=entry_z,
            exit_z=exit_z,
            pvalue_max=pvalue_max,
            top_pairs=top_pairs,
        )

    def _select_pairs(self, prices: pd.DataFrame, universe: pd.DataFrame | None) -> list[tuple[str, str, float]]:
        tickers = prices.columns.tolist()
        # Restrict to same-sector candidates when possible.
        pairs_candidates: list[tuple[str, str]] = []
        if universe is not None and "sector" in universe.columns:
            by_sector: dict[str, list[str]] = {}
            for _, row in universe.iterrows():
                if row["ticker"] in tickers:
                    by_sector.setdefault(row["sector"], []).append(row["ticker"])
            for members in by_sector.values():
                for i, a in enumerate(members):
                    for b in members[i + 1:]:
                        pairs_candidates.append((a, b))
        else:
            for i, a in enumerate(tickers):
                for b in tickers[i + 1:]:
                    pairs_candidates.append((a, b))

        results = []
        form = prices.iloc[-self.params["formation_window"]:]
        for a, b in pairs_candidates:
            beta, pvalue = _engle_granger(form[a], form[b])
            if np.isnan(pvalue):
                continue
            if pvalue <= self.params["pvalue_max"]:
                results.append((a, b, beta))
        return results[: self.params["top_pairs"]]

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        if len(prices) < self.params["formation_window"] + self.params["trading_window"]:
            return StrategyResult(
                name=self.name,
                weights=pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
                metadata={"error": "insufficient_history"},
            )

        form_end = self.params["formation_window"]
        formation_prices = prices.iloc[:form_end]
        pairs = self._select_pairs(formation_prices, universe)
        if not pairs:
            return StrategyResult(
                name=self.name,
                weights=pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
                metadata={"pairs": []},
            )

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        per_pair_weight = 1.0 / len(pairs)
        position: dict[tuple[str, str], int] = {p[:2]: 0 for p in pairs}
        for dt in prices.index[form_end:]:
            window = prices.loc[:dt].iloc[-self.params["formation_window"]:]
            for a, b, beta in pairs:
                if a not in window.columns or b not in window.columns:
                    continue
                spread = window[a] - beta * window[b]
                mu, sd = spread.mean(), spread.std()
                if sd == 0 or np.isnan(sd):
                    continue
                z = (spread.iloc[-1] - mu) / sd
                pos = position[(a, b)]
                if pos == 0:
                    if z > self.params["entry_z"]:
                        pos = -1  # short spread: short a, long b
                    elif z < -self.params["entry_z"]:
                        pos = 1   # long spread: long a, short b
                elif abs(z) < self.params["exit_z"]:
                    pos = 0
                position[(a, b)] = pos
                if pos != 0:
                    # Dollar-neutral: 0.5 per leg of the pair
                    w_a = pos * per_pair_weight * 0.5
                    w_b = -pos * per_pair_weight * 0.5 * np.sign(beta)
                    weights.at[dt, a] = weights.at[dt, a] + w_a
                    weights.at[dt, b] = weights.at[dt, b] + w_b

        return StrategyResult(
            name=self.name,
            weights=weights,
            metadata={"pairs": [{"a": a, "b": b, "beta": float(beta)} for a, b, beta in pairs]},
        )
