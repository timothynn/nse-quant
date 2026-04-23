"""Risk manager tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nse_quant.data import DataFetcher
from nse_quant.risk import (
    RiskManager,
    correlation_cluster_weights,
    kelly_fraction,
    stress_test,
    volatility_parity_weights,
)


def test_kelly_bounds() -> None:
    assert kelly_fraction(0.0, 0.04) == 0
    f = kelly_fraction(0.05, 0.04)
    assert 0 < f <= 0.5
    # Negative mean returns → zero allocation
    assert kelly_fraction(-0.02, 0.04) == 0


def test_volatility_parity_weights_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.01, (500, 5)), columns=list("ABCDE"))
    w = volatility_parity_weights(ret)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()


def test_correlation_clusters() -> None:
    rng = np.random.default_rng(7)
    ret = pd.DataFrame(rng.normal(0, 0.01, (300, 6)), columns=list("ABCDEF"))
    w = correlation_cluster_weights(ret, n_clusters=3)
    assert abs(w.sum() - 1.0) < 1e-6


def test_stress_test_rows() -> None:
    ret = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    df = stress_test(ret)
    assert {"scenario", "max_drawdown", "var_5", "cvar_5"} <= set(df.columns)
    assert len(df) >= 4


def test_risk_manager_caps() -> None:
    mgr = RiskManager(max_position_weight=0.1, max_sector_weight=0.3)
    weights = pd.Series({"A": 0.5, "B": 0.05, "C": 0.05, "D": 0.4})
    sectors = {"A": "banking", "B": "banking", "C": "telecom", "D": "telecom"}
    capped = mgr.enforce_caps(weights, sectors)
    assert capped.abs().max() <= 0.1 + 1e-9


def test_risk_manager_evaluate() -> None:
    fetcher = DataFetcher(use_live=False)
    prices = fetcher.get("SCOM.NR").tail(500)
    ret = prices["adj_close"].pct_change().dropna()
    out = RiskManager().evaluate(ret)
    assert {"var_5", "cvar_5", "max_drawdown"} == set(out.keys())
