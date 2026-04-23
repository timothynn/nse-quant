"""FastAPI service exposing signals, backtests, and portfolio endpoints."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nse_quant.backtest import Backtester
from nse_quant.data import DataFetcher, load_fundamentals, load_universe
from nse_quant.strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(title="NSE Quant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _fetcher() -> DataFetcher:
    return DataFetcher(use_live=False)


class UniverseItem(BaseModel):
    ticker: str
    name: str
    sector: str


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class SignalRow(BaseModel):
    ticker: str
    weight: float


class BacktestRequest(BaseModel):
    strategy: str = Field(..., description="Strategy key, e.g. 'momentum'")
    start: str | None = None
    end: str | None = None
    tickers: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class EquityPoint(BaseModel):
    date: str
    equity: float


class BacktestResponse(BaseModel):
    strategy: str
    metrics: dict[str, float]
    equity: list[EquityPoint]
    top_positions: list[SignalRow]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/universe", response_model=list[UniverseItem])
def universe() -> list[UniverseItem]:
    df = load_universe()
    return [UniverseItem(**row._asdict()) for row in df.itertuples(index=False)]


@app.get("/fundamentals")
def fundamentals() -> list[dict[str, Any]]:
    return load_fundamentals().to_dict(orient="records")


@app.get("/prices/{ticker}", response_model=list[PricePoint])
def prices(ticker: str, start: str | None = None, end: str | None = None) -> list[PricePoint]:
    try:
        df = _fetcher().get(ticker, start=start, end=end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    df = df.reset_index().rename(columns={"index": "date"})
    return [
        PricePoint(
            date=str(row["date"].date()),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in df.iterrows()
    ]


@app.get("/strategies")
def list_strategies() -> dict[str, list[str]]:
    return {"strategies": sorted(STRATEGY_REGISTRY.keys())}


@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    if req.strategy not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")
    fetcher = _fetcher()
    prices_df = fetcher.get_many(tickers=req.tickers, start=req.start, end=req.end, column="adj_close")
    if prices_df.empty:
        raise HTTPException(status_code=400, detail="No price data for requested tickers")
    fundamentals_df = load_fundamentals()
    universe_df = load_universe()
    strategy = STRATEGY_REGISTRY[req.strategy](**req.params)
    result = strategy.generate(
        prices_df, fundamentals=fundamentals_df, universe=universe_df,
    )
    bt = Backtester().run(result.weights, prices_df)
    equity = [
        {"date": str(idx.date()), "equity": float(v)}
        for idx, v in bt.equity_curve.items()
    ]
    # Top positions from the last row of weights
    last_row = result.weights.iloc[-1] if not result.weights.empty else None
    top_positions = []
    if last_row is not None:
        nonzero = last_row[last_row != 0].sort_values(ascending=False).head(10)
        top_positions = [SignalRow(ticker=t, weight=float(w)) for t, w in nonzero.items()]
    return BacktestResponse(
        strategy=req.strategy,
        metrics=bt.metrics,
        equity=equity,
        top_positions=top_positions,
    )


@app.get("/signals/{strategy}")
def signals(strategy: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
    if strategy not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
    fetcher = _fetcher()
    prices_df = fetcher.get_many(start=start, end=end, column="adj_close")
    if prices_df.empty:
        raise HTTPException(status_code=400, detail="No price data available")
    strat = STRATEGY_REGISTRY[strategy]()
    result = strat.generate(prices_df, fundamentals=load_fundamentals(), universe=load_universe())
    if result.weights.empty:
        return {"strategy": strategy, "signals": []}
    last = result.weights.iloc[-1]
    signals = [
        {"ticker": t, "weight": float(w)}
        for t, w in last.items() if abs(w) > 1e-6
    ]
    return {"strategy": strategy, "date": str(result.weights.index[-1].date()), "signals": signals}
