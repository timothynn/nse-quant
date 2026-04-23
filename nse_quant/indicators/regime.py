"""Market regime detection: Hurst exponent, ADX, volatility clustering."""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class RegimeLabel(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 60) -> float:
    """Estimate the Hurst exponent via R/S-style variance of lagged differences."""
    s = np.asarray(series.dropna(), dtype=float)
    if len(s) < max_lag + 1:
        return float("nan")
    lags = np.arange(min_lag, max_lag)
    tau = [np.sqrt(np.std(np.subtract(s[lag:], s[:-lag]))) for lag in lags]
    tau = np.asarray(tau)
    mask = tau > 0
    if mask.sum() < 2:
        return float("nan")
    poly = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)
    return float(poly[0] * 2.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().rename("adx")


def volatility_regime(close: pd.Series, short: int = 20, long: int = 100) -> pd.Series:
    """Ratio of short- to long-window realized vol; >1 implies clustering."""
    ret = close.pct_change()
    sv = ret.rolling(short).std()
    lv = ret.rolling(long).std()
    return (sv / lv.replace(0, np.nan)).rename("vol_regime")


def classify_regime(df: pd.DataFrame, adx_trend_threshold: float = 25.0) -> pd.Series:
    """Label each bar as trending_up/down, ranging, or volatile."""
    adx_s = adx(df["high"], df["low"], df["close"])
    vol = volatility_regime(df["close"])
    slope = df["close"].pct_change(20)

    labels = pd.Series(index=df.index, dtype="object")
    for dt in df.index:
        a = adx_s.get(dt, np.nan)
        v = vol.get(dt, np.nan)
        s = slope.get(dt, np.nan)
        if pd.isna(a) or pd.isna(v):
            labels.at[dt] = RegimeLabel.RANGING.value
            continue
        if v > 1.5:
            labels.at[dt] = RegimeLabel.VOLATILE.value
        elif a >= adx_trend_threshold and pd.notna(s) and s > 0:
            labels.at[dt] = RegimeLabel.TRENDING_UP.value
        elif a >= adx_trend_threshold and pd.notna(s) and s < 0:
            labels.at[dt] = RegimeLabel.TRENDING_DOWN.value
        else:
            labels.at[dt] = RegimeLabel.RANGING.value
    return labels.rename("regime")
