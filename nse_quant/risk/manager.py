"""Risk management utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nse_quant.backtest.metrics import (
    conditional_var,
    max_drawdown,
    value_at_risk,
)


def kelly_fraction(mean_return: float, variance: float) -> float:
    """Continuous-approximation Kelly fraction for a single risky asset."""
    if variance <= 0:
        return 0.0
    f = mean_return / variance
    # Clip to a prudent half-Kelly cap.
    return float(max(0.0, min(0.5, f)))


def volatility_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """Equal risk contribution (approximation): weight ∝ 1/vol."""
    vol = returns.std(ddof=0)
    if (vol == 0).all():
        return pd.Series(1.0 / len(vol), index=vol.index)
    inv = 1.0 / vol.replace(0, np.nan)
    return (inv / inv.sum()).fillna(0)


def correlation_cluster_weights(returns: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    """Weights that down-weight highly correlated assets via simple clustering."""
    from sklearn.cluster import KMeans  # type: ignore
    if returns.shape[1] <= n_clusters:
        return pd.Series(1.0 / returns.shape[1], index=returns.columns)
    corr = returns.corr().fillna(0).values
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=7)
    labels = km.fit_predict(corr)
    weights = pd.Series(0.0, index=returns.columns)
    per_cluster = 1.0 / n_clusters
    for cid in range(n_clusters):
        members = returns.columns[labels == cid]
        if len(members) == 0:
            continue
        weights.loc[members] = per_cluster / len(members)
    return weights


def stress_test(
    returns: pd.Series,
    scenarios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply parallel return shocks and report resulting portfolio drawdowns."""
    scenarios = scenarios or {
        "crash_2008": -0.40,
        "covid_2020": -0.30,
        "nse_2015_drawdown": -0.25,
        "mild_shock": -0.10,
        "base_case": 0.0,
    }
    rows = []
    for name, shock in scenarios.items():
        shocked = pd.concat([returns, pd.Series([shock])])
        rows.append(
            {
                "scenario": name,
                "shock": shock,
                "max_drawdown": max_drawdown(shocked),
                "var_5": value_at_risk(shocked, 0.05),
                "cvar_5": conditional_var(shocked, 0.05),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class RiskManager:
    max_position_weight: float = 0.15
    max_sector_weight: float = 0.35
    max_portfolio_vol: float = 0.25
    max_drawdown_stop: float = 0.25
    var_alpha: float = 0.05

    def enforce_caps(
        self,
        weights: pd.Series,
        sector_map: dict[str, str] | None = None,
    ) -> pd.Series:
        w = weights.clip(-self.max_position_weight, self.max_position_weight)
        if sector_map is None:
            return w
        by_sector: dict[str, list[str]] = {}
        for t, s in sector_map.items():
            if t in w.index:
                by_sector.setdefault(s, []).append(t)
        for _sector, members in by_sector.items():
            sec_weight = w.loc[members].sum()
            if sec_weight > self.max_sector_weight:
                scale = self.max_sector_weight / sec_weight
                w.loc[members] = w.loc[members] * scale
        return w

    def size_for_vol_target(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        target_vol: float | None = None,
    ) -> pd.Series:
        target = target_vol or self.max_portfolio_vol
        aligned = returns[weights.index].dropna(how="all")
        if aligned.empty:
            return weights
        port_ret = (aligned * weights).sum(axis=1)
        realized = port_ret.std(ddof=0) * np.sqrt(252)
        if realized <= 0:
            return weights
        return weights * (target / realized)

    def evaluate(self, returns: pd.Series) -> dict[str, float]:
        return {
            "var_5": value_at_risk(returns, self.var_alpha),
            "cvar_5": conditional_var(returns, self.var_alpha),
            "max_drawdown": max_drawdown(returns),
        }
