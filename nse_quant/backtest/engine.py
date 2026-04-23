"""Vectorized backtester with commission, slippage, and risk constraints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nse_quant.backtest.metrics import performance_summary
from nse_quant.config import settings


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Backtester:
    """Daily vectorized backtest.

    The target weights produced by a strategy are interpreted as end-of-day
    target allocations. We execute the next open (approximated by next close
    here) and charge ``commission_bps`` + ``slippage_bps`` on the turnover
    per rebalance step.
    """

    initial_capital: float = settings.initial_capital
    commission_bps: float = settings.commission_bps
    slippage_bps: float = settings.slippage_bps
    max_position_weight: float = settings.max_position_weight
    max_drawdown_stop: float | None = None

    def _apply_constraints(self, w: pd.DataFrame) -> pd.DataFrame:
        w = w.clip(-self.max_position_weight, self.max_position_weight)
        gross = w.abs().sum(axis=1)
        scale = np.where(gross > 1.0, 1.0 / gross.replace(0, 1), 1.0)
        return w.multiply(scale, axis=0)

    def run(self, weights: pd.DataFrame, prices: pd.DataFrame) -> BacktestResult:
        prices = prices.reindex(weights.index).ffill()
        w = self._apply_constraints(weights.fillna(0))
        # Shift weights by 1 to execute next bar (no lookahead)
        w_exec = w.shift(1).fillna(0)
        ret = prices.pct_change().fillna(0)
        port_ret = (w_exec * ret).sum(axis=1)

        # Transaction costs via turnover.
        turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0)
        cost_bps = (self.commission_bps + self.slippage_bps) / 10_000.0
        port_ret = port_ret - turnover * cost_bps

        # Optional drawdown stop.
        if self.max_drawdown_stop is not None:
            equity = (1 + port_ret).cumprod()
            peak = equity.cummax()
            dd = equity / peak - 1
            mask = dd < -self.max_drawdown_stop
            port_ret = port_ret.where(~mask, 0.0)

        equity = self.initial_capital * (1 + port_ret).cumprod()
        trades = self._trade_log(w)
        metrics = performance_summary(port_ret)
        return BacktestResult(
            equity_curve=equity.rename("equity"),
            returns=port_ret.rename("returns"),
            positions=w,
            trades=trades,
            metrics=metrics,
            metadata={"turnover_mean": float(turnover.mean())},
        )

    def _trade_log(self, w: pd.DataFrame) -> pd.DataFrame:
        diff = (w - w.shift(1)).fillna(0)
        rows = []
        for dt, row in diff.iterrows():
            changes = row[row != 0]
            for ticker, delta in changes.items():
                rows.append({"date": dt, "ticker": ticker, "delta_weight": float(delta)})
        return pd.DataFrame(rows)
