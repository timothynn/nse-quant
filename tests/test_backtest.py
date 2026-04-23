"""Backtester and metrics tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nse_quant.backtest import Backtester, performance_summary
from nse_quant.backtest.metrics import (
    annualized_return,
    annualized_vol,
    conditional_var,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)
from nse_quant.data import DataFetcher
from nse_quant.strategies import MomentumRotationStrategy


def test_performance_metrics_on_flat_series() -> None:
    ret = pd.Series([0.0] * 252)
    m = performance_summary(ret)
    assert m["cagr"] == 0.0
    assert m["volatility"] == 0.0
    assert m["sharpe"] == 0.0


def test_sharpe_positive_on_positive_drift() -> None:
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sharpe_ratio(ret, rf=0) > 0
    assert annualized_return(ret) > 0
    assert annualized_vol(ret) > 0


def test_var_and_cvar_order() -> None:
    ret = pd.Series(np.random.default_rng(0).normal(0, 0.02, 2000))
    var5 = value_at_risk(ret, 0.05)
    cvar5 = conditional_var(ret, 0.05)
    assert cvar5 <= var5  # CVaR is at least as negative as VaR


def test_max_drawdown_bounded() -> None:
    ret = pd.Series(np.linspace(-0.01, 0.01, 200))
    mdd = max_drawdown(ret)
    assert mdd <= 0


def test_backtester_runs_on_momentum() -> None:
    fetcher = DataFetcher(use_live=False)
    prices = fetcher.get_many(column="adj_close").tail(800)
    strat = MomentumRotationStrategy()
    result = strat.generate(prices)
    bt = Backtester().run(result.weights, prices)
    assert bt.equity_curve.iloc[-1] > 0
    assert len(bt.returns) == len(prices)
    assert "sharpe" in bt.metrics
