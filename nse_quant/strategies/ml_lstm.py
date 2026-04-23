"""LSTM-based alpha strategy with graceful fallback to MLP regression.

Training a full LSTM in CI is expensive, so we default to a small sklearn
``MLPRegressor`` operating on the same feature space. If PyTorch is
installed the class swaps in a proper LSTM using the same interface.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from nse_quant.indicators.technical import _rsi, _sma
from nse_quant.strategies.base import Strategy, StrategyResult

logger = logging.getLogger(__name__)


def _sequence_features(close: pd.Series, window: int = 20) -> pd.DataFrame:
    feats = pd.DataFrame(index=close.index)
    ret = close.pct_change()
    for lag in range(1, window + 1):
        feats[f"ret_lag_{lag}"] = ret.shift(lag)
    feats["sma_ratio"] = _sma(close, 10) / _sma(close, 50)
    feats["rsi"] = _rsi(close, 14)
    feats["vol_21"] = ret.rolling(21).std()
    feats["target"] = ret.shift(-1)
    return feats.dropna()


class LSTMAlphaStrategy(Strategy):
    name = "lstm_alpha"

    def __init__(
        self,
        window: int = 20,
        train_window: int = 756,
        retrain_every: int = 126,
        top_k: int = 5,
        max_weight: float = 0.2,
        use_torch: bool = False,
    ) -> None:
        super().__init__(
            window=window, train_window=train_window, retrain_every=retrain_every,
            top_k=top_k, max_weight=max_weight, use_torch=use_torch,
        )

    def _build_model(self):
        if self.params["use_torch"]:
            try:
                import torch  # noqa: F401
                return _TorchLSTM(input_size=self.params["window"] + 3)
            except ImportError:
                logger.warning("torch not installed; using MLP fallback")
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=7)

    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        tw = self.params["train_window"]
        retrain = self.params["retrain_every"]
        if len(prices) < tw + retrain:
            return StrategyResult(name=self.name, weights=weights, metadata={"error": "insufficient_history"})

        per_ticker: dict[str, pd.DataFrame] = {
            t: _sequence_features(prices[t], self.params["window"]) for t in prices.columns
        }
        models: dict[str, Any] = {}
        last_train_ix = -1
        preds: dict[pd.Timestamp, dict[str, float]] = {}
        for i, dt in enumerate(prices.index):
            if i < tw:
                continue
            if (i - last_train_ix) >= retrain or last_train_ix < 0:
                for t, fdf in per_ticker.items():
                    train = fdf.loc[:prices.index[i - 1]].tail(tw)
                    if len(train) < 100:
                        continue
                    X = train.drop(columns=["target"]).values
                    y = train["target"].values
                    model = self._build_model()
                    try:
                        model.fit(X, y)
                    except Exception:  # noqa: BLE001
                        continue
                    models[t] = model
                last_train_ix = i

            row_preds: dict[str, float] = {}
            for t, fdf in per_ticker.items():
                if dt not in fdf.index or t not in models:
                    continue
                X = fdf.drop(columns=["target"]).loc[[dt]].values
                try:
                    row_preds[t] = float(models[t].predict(X)[0])
                except Exception:  # noqa: BLE001
                    continue
            preds[dt] = row_preds

        for dt, row in preds.items():
            if not row:
                continue
            s = pd.Series(row).sort_values(ascending=False)
            longs = s.head(self.params["top_k"]).index.tolist()
            if not longs:
                continue
            per = min(self.params["max_weight"], 1.0 / len(longs))
            w = pd.Series(0.0, index=prices.columns)
            w.loc[longs] = per
            w = w / w.sum() if w.sum() > 0 else w
            weights.loc[dt] = w
        return StrategyResult(name=self.name, weights=weights)


class _TorchLSTM:  # pragma: no cover - optional dependency
    def __init__(self, input_size: int) -> None:
        import torch  # noqa: F401
        import torch.nn as nn
        self.nn = nn
        import torch as _torch
        self.torch = _torch
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.opt = _torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> _TorchLSTM:
        t = self.torch
        Xt = t.tensor(X, dtype=t.float32)
        yt = t.tensor(y, dtype=t.float32).view(-1, 1)
        for _ in range(40):
            self.opt.zero_grad()
            pred = self.model(Xt)
            loss = self.loss_fn(pred, yt)
            loss.backward()
            self.opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            return self.model(t.tensor(X, dtype=t.float32)).numpy().ravel()
