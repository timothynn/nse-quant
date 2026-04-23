"""Base types for strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """A per-asset position signal in [-1, 1] (short to long)."""

    ticker: str
    weight: float
    reason: str = ""


@dataclass
class StrategyResult:
    """Output of a strategy run over a price panel.

    ``weights`` is a DataFrame indexed by date with columns per ticker and
    values in [-1, 1] representing target portfolio weights. The backtester
    will normalize gross exposure to 1 and apply leverage/risk rules.
    """

    name: str
    weights: pd.DataFrame
    signals: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base for all strategies."""

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def generate(
        self,
        prices: pd.DataFrame,
        *,
        fundamentals: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        """Produce target weights."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}
