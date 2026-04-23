"""XGBoost next-day return prediction strategy."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nse_quant.indicators.technical import _atr, _rsi, _sma
from nse_quant.strategies.base import Strategy, StrategyResult

logger = logging.getLogger(__name__)


def _feature_frame(prices: pd.DataFrame, ticker: str, ohlcv: pd.DataFrame | None = None) -> pd.DataFrame:
    close = prices[ticker]
    ret_1 = close.pct_change()
    feats = pd.DataFrame(index=close.index)
    feats["ret_1"] = ret_1
    feats["ret_5"] = close.pct_change(5)
    feats["ret_21"] = close.pct_change(21)
    feats["ret_63"] = close.pct_change(63)
    feats["vol_21"] = ret_1.rolling(21).std()
    feats["sma_ratio_5_20"] = _sma(close, 5) / _sma(close, 20)
    feats["sma_ratio_20_50"] = _sma(close, 20) / _sma(close, 50)
    feats["rsi_14"] = _rsi(close, 14)
    if ohlcv is not None and {"high", "low", "close"}.issubset(ohlcv.columns):
        feats["atr_14"] = _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14) / close
    feats["target"] = close.pct_change().shift(-1)
    return feats.dropna()


class XGBoostAlphaStrategy(Strategy):
    name = "xgboost_alpha"

    def __init__(
        self,
        train_window: int = 756,
        retrain_every: int = 63,
        long_q: float = 0.2,
        short_q: float = 0.2,
        long_only: bool = True,
        max_weight: float = 0.15,
    ) -> None:
        super().__init__(
            train_window=train_window, retrain_every=retrain_every,
            long_q=long_q, short_q=short_q, long_only=long_only, max_weight=max_weight,
        )

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        ohlcv: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("xgboost unavailable; falling back to ridge")
            from sklearn.linear_model import Ridge as XGBRegressor  # type: ignore

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        tw = self.params["train_window"]
        retrain = self.params["retrain_every"]
        if len(prices) < tw + retrain:
            return StrategyResult(name=self.name, weights=weights, metadata={"error": "insufficient_history"})

        # Build per-ticker feature frames once.
        per_ticker: dict[str, pd.DataFrame] = {}
        for t in prices.columns:
            oh = (ohlcv or {}).get(t)
            try:
                per_ticker[t] = _feature_frame(prices, t, oh)
            except Exception as exc:  # noqa: BLE001
                logger.debug("feature_frame failed for %s: %s", t, exc)

        # Walk forward: retrain every ``retrain`` bars.
        index = prices.index
        models: dict[str, Any] = {}
        last_train_ix = -1
        preds: dict[pd.Timestamp, dict[str, float]] = {}
        for i, dt in enumerate(index):
            if i < tw:
                continue
            need_retrain = (i - last_train_ix) >= retrain or last_train_ix < 0
            if need_retrain:
                for t, fdf in per_ticker.items():
                    train = fdf.loc[:index[i - 1]].tail(tw)
                    if len(train) < 100:
                        continue
                    X = train.drop(columns=["target"]).values
                    y = train["target"].values
                    try:
                        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, n_jobs=1, verbosity=0)
                    except TypeError:
                        model = XGBRegressor()
                    try:
                        model.fit(X, y)
                    except Exception:  # noqa: BLE001
                        continue
                    models[t] = model
                last_train_ix = i

            # Predict for today.
            row_preds = {}
            for t, fdf in per_ticker.items():
                if dt not in fdf.index or t not in models:
                    continue
                X = fdf.drop(columns=["target"]).loc[[dt]].values
                try:
                    row_preds[t] = float(models[t].predict(X)[0])
                except Exception:  # noqa: BLE001
                    continue
            preds[dt] = row_preds

        # Build weights from predictions: top long_q / bottom short_q each day.
        for dt, row in preds.items():
            if not row:
                continue
            s = pd.Series(row)
            ranks = s.rank(pct=True)
            longs = s.index[ranks >= 1 - self.params["long_q"]]
            shorts = s.index[ranks <= self.params["short_q"]]
            w = pd.Series(0.0, index=prices.columns)
            if len(longs):
                per = min(self.params["max_weight"], 1.0 / len(longs))
                w.loc[longs] = per
                w *= 1.0 / w[w > 0].sum()
            if not self.params["long_only"] and len(shorts):
                w.loc[shorts] = -min(self.params["max_weight"], 1.0 / len(shorts))
            weights.loc[dt] = w
        return StrategyResult(name=self.name, weights=weights)
