# NSE Quant

Quantitative analysis toolkit for the **Nairobi Securities Exchange**. Pulls
OHLCV data, computes 50+ technical indicators, runs fundamental screens,
executes 10 investment strategies (including ML alpha and Markowitz /
Black-Litterman portfolio optimization), backtests them with transaction
costs, and surfaces the results through a Streamlit dashboard and FastAPI
service.

## What's in the box

| Layer          | Implementation                                                                |
|----------------|-------------------------------------------------------------------------------|
| Data           | yfinance live fetch with Parquet cache + bundled synthetic NSE sample data   |
| Indicators     | 60+ indicators across trend, momentum, volatility, volume, cycle families    |
| Regime         | Hurst exponent, ADX, volatility clustering → trending/ranging/volatile labels |
| Fundamentals   | P/E, P/B, ROE, ROIC, debt ratios, dividend yield (yfinance + CSV fallback)   |
| Strategies     | 10 (momentum, mean reversion, pairs, sector rotation, vol arb, dividend, quality, XGBoost, LSTM-proxy, Markowitz + Black-Litterman) |
| Backtester     | Vectorized daily with commissions, slippage, turnover, drawdown stop         |
| Risk           | VaR, CVaR, Kelly sizing, vol-parity, correlation clustering, stress tests    |
| UX             | Streamlit dashboard + FastAPI service + CLI                                  |

## Install

Requires Python 3.10–3.12.

```bash
git clone https://github.com/timothynn/nse-quant.git
cd nse-quant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[ta]"          # pandas-ta (requires numpy 2+)
pip install -e ".[optimizer]"   # cvxpy for constrained Markowitz
pip install -e ".[lstm]"        # PyTorch LSTM strategy
```

## Run it

```bash
# Generate the bundled synthetic sample data (runs automatically on first use)
nse-quant generate-samples

# Backtest a single strategy
nse-quant backtest momentum

# Print the current signals for a strategy
nse-quant signals quality

# Launch the Streamlit dashboard
streamlit run nse_quant/dashboard/app.py

# Launch the FastAPI service
uvicorn nse_quant.api.main:app --reload
```

### Docker

```bash
docker compose up
# FastAPI on :8000, Streamlit on :8501
```

## API endpoints

| Method | Path                      | Description                                |
|--------|---------------------------|--------------------------------------------|
| GET    | `/health`                 | Liveness check                             |
| GET    | `/universe`               | NSE universe with sector classification    |
| GET    | `/fundamentals`           | Fundamentals panel                         |
| GET    | `/prices/{ticker}`        | OHLCV for a ticker                         |
| GET    | `/strategies`             | Available strategy names                   |
| GET    | `/signals/{strategy}`     | Latest weights for a strategy              |
| POST   | `/backtest`               | Run a backtest, return metrics + equity    |

Example:

```bash
curl -X POST http://localhost:8000/backtest \
     -H 'Content-Type: application/json' \
     -d '{"strategy":"dividend","tickers":["SCOM.NR","EQTY.NR","KCB.NR"]}'
```

## Dashboard

The Streamlit dashboard has four tabs:

- **Overview** — market summary, top movers, sector performance bar chart.
- **Indicators** — candlestick with overlays, RSI/MACD/regime for the selected ticker.
- **Backtest** — pick a strategy, run it on the loaded panel, and see equity curve, signal heatmap, risk metrics, and stress tests.
- **Portfolio** — upload positions or pick from the universe; get a hypothetical equity curve and risk panel.

## Data

The package ships a deterministic **synthetic** NSE dataset covering 30 top
listings across 10 sectors from 2020 through 2025. This exists so the
package works offline and in CI. For real prices, set `use_live=True` on
`DataFetcher` (or pass `--live` to the CLI) — the fetcher will fall back to
yfinance (`SCOM.NR`, `EQTY.NR`, etc.). Note that NSE coverage on yfinance
is patchy; expect gaps and failures for some tickers, with automatic fallback
to the bundled sample.

## Strategies

All strategies implement a common `Strategy.generate(prices, fundamentals=, universe=) -> StrategyResult` interface and produce a `DataFrame` of target weights indexed by date.

| Key                   | Class                           | Core idea                                                  |
|-----------------------|---------------------------------|------------------------------------------------------------|
| `momentum`            | `MomentumRotationStrategy`      | Long top-quintile 6-month return (skip 1M), monthly rebal  |
| `mean_reversion`      | `MeanReversionStrategy`         | RSI + z-score entry, P/E filter                            |
| `pairs`               | `PairsTradingStrategy`          | Engle-Granger cointegration on same-sector pairs           |
| `sector_rotation`     | `SectorRotationStrategy`        | Rotate into top-N sectors by 3-month momentum              |
| `volatility`          | `VolatilityArbitrageStrategy`   | Inverse-vol weighting within low-vol bucket                |
| `dividend`            | `DividendAristocratsStrategy`   | High, stable dividend yield + payout + ROE screen          |
| `quality`             | `QualityFactorStrategy`         | ROIC + margin + leverage composite score                   |
| `xgboost`             | `XGBoostAlphaStrategy`          | Walk-forward XGBoost next-day return prediction            |
| `lstm`                | `LSTMAlphaStrategy`             | Sequence features + MLP (or PyTorch LSTM)                  |
| `portfolio_optimizer` | `PortfolioOptimizerStrategy`    | Markowitz w/ Black-Litterman overlay, monthly rebal        |

## Risk management

`nse_quant.risk.RiskManager` enforces position and sector caps and computes
VaR / CVaR / drawdown. `volatility_parity_weights`, `correlation_cluster_weights`,
`kelly_fraction`, and `stress_test` are exported helpers.

## Testing

```bash
pytest
```

All 36 unit tests must pass. Every strategy is smoke-tested end-to-end on
the sample panel, and every indicator family is verified to produce
non-trivial output.

## Honest caveats

This README deliberately does **not** quote specific Sharpe or CAGR numbers
because:

1. The bundled sample data is *synthetic* (seeded random walks). Backtests
   on it measure code correctness, not predictive power. Quoting returns
   from it would be misleading.
2. Real NSE data from free sources (yfinance) is sparse and often missing
   for smaller names. Real-world performance will depend heavily on which
   tickers actually have clean data.
3. Kenyan equity markets have real microstructure (wide spreads, thin
   liquidity on small caps, frequent halts) that a vectorized daily
   backtester does not capture. Treat results as directional, not tradeable.

Out of scope for this repository (and explicitly dropped from the original
spec as not deliverable in a working state):

- Kafka / Airflow pipelines, Kubernetes deployment, 99% uptime SLOs
- Reinforcement-learning portfolio manager, AR visualization, NFTs
- M-Pesa paper-trading integration, voice commands, SMS alerts
- Tick-level or intraday data (no free source exposes it for NSE)

If you need those, open an issue describing which one matters and I'll add
it as a separate, honestly-scoped module.

## License

MIT. See `LICENSE` (add your own before publishing).
