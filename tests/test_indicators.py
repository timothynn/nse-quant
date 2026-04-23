"""Indicator tests."""
from __future__ import annotations

import pandas as pd

from nse_quant.data import DataFetcher
from nse_quant.indicators import (
    add_all_indicators,
    adx,
    available_indicators,
    classify_regime,
    hurst_exponent,
    volatility_regime,
)


def _sample_ohlcv() -> pd.DataFrame:
    return DataFetcher(use_live=False).get("SCOM.NR").tail(400)


def test_available_indicators_above_50() -> None:
    names = available_indicators()
    assert len(names) >= 50, f"expected 50+, got {len(names)}"


def test_add_all_indicators_shape() -> None:
    df = _sample_ohlcv()
    out = add_all_indicators(df)
    assert out.shape[0] == df.shape[0]
    # Core indicators present
    for col in ["sma_20", "ema_50", "rsi_14", "macd", "bb_upper", "atr_14", "obv"]:
        assert col in out.columns
    # No all-NaN columns
    assert not out.isna().all(axis=0).any()


def test_hurst_in_range() -> None:
    df = _sample_ohlcv()
    h = hurst_exponent(df["close"])
    assert 0.0 < h < 1.5


def test_adx_positive() -> None:
    df = _sample_ohlcv()
    a = adx(df["high"], df["low"], df["close"]).dropna()
    assert (a >= 0).all()
    assert (a <= 100).all()


def test_volatility_regime_ratio() -> None:
    df = _sample_ohlcv()
    v = volatility_regime(df["close"]).dropna()
    assert (v > 0).all()


def test_classify_regime_labels() -> None:
    df = _sample_ohlcv()
    reg = classify_regime(df).dropna()
    assert set(reg.unique()).issubset({"trending_up", "trending_down", "ranging", "volatile"})
