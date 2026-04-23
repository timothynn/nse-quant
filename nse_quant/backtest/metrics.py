"""Performance and risk metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1 / years) - 1)


def annualized_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.08) -> float:
    # Kenyan T-bill yields hover around 8–15%; use 8% as a conservative default.
    ann_ret = annualized_return(returns)
    ann_vol = annualized_vol(returns)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - rf) / ann_vol)


def sortino_ratio(returns: pd.Series, rf: float = 0.08) -> float:
    downside = returns[returns < 0]
    dd_vol = downside.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(downside) else 0
    ann_ret = annualized_return(returns)
    if dd_vol == 0:
        return 0.0
    return float((ann_ret - rf) / dd_vol)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns) / mdd)


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    if returns.empty:
        return 0.0
    return float(np.quantile(returns, alpha))


def conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    if returns.empty:
        return 0.0
    v = value_at_risk(returns, alpha)
    tail = returns[returns <= v]
    return float(tail.mean()) if len(tail) else v


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def performance_summary(returns: pd.Series) -> dict[str, float]:
    return {
        "cagr": annualized_return(returns),
        "volatility": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "var_5": value_at_risk(returns, 0.05),
        "cvar_5": conditional_var(returns, 0.05),
        "hit_rate": hit_rate(returns),
        "n_obs": float(len(returns)),
    }
