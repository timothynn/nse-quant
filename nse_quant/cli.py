"""Command-line entrypoint: `nse-quant <command>`."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from nse_quant.backtest import Backtester
from nse_quant.data import DataFetcher, load_fundamentals, load_universe
from nse_quant.data.sample import generate_sample_data
from nse_quant.strategies import STRATEGY_REGISTRY

logger = logging.getLogger("nse_quant")


def _cmd_generate_samples(args: argparse.Namespace) -> int:
    manifest = generate_sample_data(start=args.start, end=args.end, force=args.force)
    print(f"Generated {len(manifest)} tickers → {manifest.shape}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    if args.strategy not in STRATEGY_REGISTRY:
        print(f"Unknown strategy: {args.strategy}. Available: {sorted(STRATEGY_REGISTRY)}")
        return 2
    fetcher = DataFetcher(use_live=args.live)
    tickers = args.tickers.split(",") if args.tickers else None
    prices = fetcher.get_many(tickers=tickers, start=args.start, end=args.end, column="adj_close")
    if prices.empty:
        print("No price data available")
        return 3
    strat = STRATEGY_REGISTRY[args.strategy]()
    result = strat.generate(prices, fundamentals=load_fundamentals(), universe=load_universe())
    bt = Backtester().run(result.weights, prices)
    out = {"strategy": args.strategy, "metrics": bt.metrics}
    print(json.dumps(out, indent=2, default=float))
    if args.output:
        bt.equity_curve.to_csv(args.output)
        print(f"Equity curve written to {args.output}")
    return 0


def _cmd_signals(args: argparse.Namespace) -> int:
    fetcher = DataFetcher(use_live=args.live)
    prices = fetcher.get_many(start=args.start, end=args.end, column="adj_close")
    if prices.empty:
        print("No price data available")
        return 3
    strat = STRATEGY_REGISTRY[args.strategy]()
    result = strat.generate(prices, fundamentals=load_fundamentals(), universe=load_universe())
    if result.weights.empty:
        print("No signals generated")
        return 0
    last = result.weights.iloc[-1]
    nonzero = last[last.abs() > 1e-6].sort_values(ascending=False)
    df = pd.DataFrame({"ticker": nonzero.index, "weight": nonzero.values.round(4)})
    print(df.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="nse-quant")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-samples", help="Generate synthetic sample OHLCV + fundamentals")
    gen.add_argument("--start", default="2020-01-02")
    gen.add_argument("--end", default="2025-12-31")
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(func=_cmd_generate_samples)

    bt = sub.add_parser("backtest", help="Run a backtest for a strategy")
    bt.add_argument("strategy")
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--tickers", default=None, help="Comma-separated list of tickers")
    bt.add_argument("--live", action="store_true", help="Try yfinance live fetch")
    bt.add_argument("--output", default=None)
    bt.set_defaults(func=_cmd_backtest)

    sig = sub.add_parser("signals", help="Print latest signals for a strategy")
    sig.add_argument("strategy")
    sig.add_argument("--start", default=None)
    sig.add_argument("--end", default=None)
    sig.add_argument("--live", action="store_true")
    sig.set_defaults(func=_cmd_signals)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
