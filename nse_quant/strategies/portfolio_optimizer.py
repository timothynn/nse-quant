"""Portfolio optimization: Markowitz mean-variance + Black-Litterman."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from nse_quant.strategies.base import Strategy, StrategyResult

logger = logging.getLogger(__name__)


def _nearest_psd(mat: np.ndarray) -> np.ndarray:
    sym = (mat + mat.T) / 2
    w, v = np.linalg.eigh(sym)
    w = np.clip(w, 1e-8, None)
    return (v * w) @ v.T


def markowitz_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 2.0,
    max_weight: float = 0.2,
    long_only: bool = True,
) -> pd.Series:
    """Solve max μ'w − (λ/2) w'Σw with box constraints and sum=1."""
    try:
        import cvxpy as cp
    except ImportError:
        logger.info("cvxpy unavailable; using closed-form unconstrained solution")
        return _closed_form_mv(expected_returns, cov, risk_aversion, max_weight, long_only)

    mu = expected_returns.values
    sig = _nearest_psd(cov.values)
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(sig)))
    constraints = [cp.sum(w) == 1, w <= max_weight]
    constraints.append(w >= (0 if long_only else -max_weight))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:  # noqa: BLE001
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:  # noqa: BLE001
            return _closed_form_mv(expected_returns, cov, risk_aversion, max_weight, long_only)
    if w.value is None:
        return _closed_form_mv(expected_returns, cov, risk_aversion, max_weight, long_only)
    return pd.Series(np.clip(w.value, 0 if long_only else -max_weight, max_weight), index=expected_returns.index)


def _closed_form_mv(mu, cov, risk_aversion, max_weight, long_only) -> pd.Series:
    inv = np.linalg.pinv(_nearest_psd(cov.values))
    raw = inv @ mu.values / risk_aversion
    raw = np.clip(raw, 0, max_weight) if long_only else np.clip(raw, -max_weight, max_weight)
    s = raw.sum()
    raw = np.ones_like(raw) / len(raw) if s <= 0 else raw / s
    return pd.Series(raw, index=mu.index)


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | None = None,
    view_confidence: float = 0.1,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Return posterior expected returns after blending market prior with views."""
    sig = cov.values
    pi = risk_aversion * sig @ market_weights.values
    if not views:
        return pd.Series(pi, index=cov.index)
    tickers = list(cov.index)
    rows = []
    q = []
    for t, ret in views.items():
        if t in tickers:
            r = np.zeros(len(tickers))
            r[tickers.index(t)] = 1.0
            rows.append(r)
            q.append(ret)
    if not rows:
        return pd.Series(pi, index=cov.index)
    P = np.array(rows)
    Q = np.array(q)
    omega = np.diag(np.full(len(Q), view_confidence))
    tau_sig = tau * sig
    M = np.linalg.inv(np.linalg.inv(tau_sig) + P.T @ np.linalg.inv(omega) @ P)
    posterior = M @ (np.linalg.inv(tau_sig) @ pi + P.T @ np.linalg.inv(omega) @ Q)
    return pd.Series(posterior, index=cov.index)


class PortfolioOptimizerStrategy(Strategy):
    name = "portfolio_optimizer"

    def __init__(
        self,
        lookback: int = 252,
        rebalance: str = "ME",
        risk_aversion: float = 3.0,
        max_weight: float = 0.2,
        long_only: bool = True,
        use_black_litterman: bool = True,
        views: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            lookback=lookback, rebalance=rebalance, risk_aversion=risk_aversion,
            max_weight=max_weight, long_only=long_only,
            use_black_litterman=use_black_litterman, views=views or {},
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
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(prices) < lookback:
            return StrategyResult(name=self.name, weights=weights)
        ret = prices.pct_change().dropna()
        rebalance_dates = prices.resample(self.params["rebalance"]).last().index.intersection(prices.index)
        last_w: pd.Series | None = None
        for dt in prices.index:
            if dt in rebalance_dates:
                window = ret.loc[:dt].tail(lookback)
                if len(window) < lookback // 2:
                    continue
                mu_hist = window.mean() * 252
                cov = window.cov() * 252
                if self.params["use_black_litterman"]:
                    # Use equal-weight market prior.
                    mw = pd.Series(1.0 / len(mu_hist), index=mu_hist.index)
                    mu_bl = black_litterman(cov, mw, views=self.params["views"])
                    mu_used = 0.5 * mu_hist + 0.5 * mu_bl
                else:
                    mu_used = mu_hist
                last_w = markowitz_weights(
                    mu_used, cov,
                    risk_aversion=self.params["risk_aversion"],
                    max_weight=self.params["max_weight"],
                    long_only=self.params["long_only"],
                )
            if last_w is not None:
                weights.loc[dt, last_w.index] = last_w.values
        return StrategyResult(name=self.name, weights=weights)
