"""FastAPI smoke tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from nse_quant.api.main import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_universe_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/universe")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 25
    assert {"ticker", "name", "sector"} <= set(data[0].keys())


def test_strategies_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/strategies")
    assert r.status_code == 200
    names = r.json()["strategies"]
    assert "momentum" in names


def test_backtest_endpoint_momentum() -> None:
    client = TestClient(app)
    r = client.post(
        "/backtest",
        json={"strategy": "momentum", "tickers": ["SCOM.NR", "EQTY.NR", "KCB.NR", "EABL.NR", "BAT.NR"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "momentum"
    assert "sharpe" in body["metrics"]
    assert len(body["equity"]) > 100
