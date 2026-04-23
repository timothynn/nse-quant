"""Data layer tests."""
from __future__ import annotations

import pandas as pd

from nse_quant.data import DataFetcher, load_fundamentals, load_universe
from nse_quant.data.sample import ensure_sample_data, generate_sample_data


def test_universe_shape() -> None:
    df = load_universe()
    assert {"ticker", "name", "sector"} <= set(df.columns)
    assert len(df) >= 25


def test_sample_data_materializes(tmp_path) -> None:
    ensure_sample_data()
    f = DataFetcher(use_live=False)
    df = f.get("SCOM.NR")
    assert isinstance(df.index, pd.DatetimeIndex)
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        assert col in df.columns
    assert len(df) > 200


def test_get_many_returns_matrix() -> None:
    f = DataFetcher(use_live=False)
    panel = f.get_many(["SCOM.NR", "EQTY.NR", "KCB.NR"])
    assert panel.shape[1] == 3
    assert not panel.empty


def test_fundamentals_loader() -> None:
    df = load_fundamentals()
    assert {"ticker", "pe", "pb", "roe", "dividend_yield"} <= set(df.columns)
    assert (df["pe"] > 0).any()


def test_regenerate_force(tmp_path) -> None:
    m1 = generate_sample_data(force=False)
    m2 = generate_sample_data(force=False)
    assert len(m1) == len(m2)
