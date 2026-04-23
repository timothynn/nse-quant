"""Investment strategies."""
from nse_quant.strategies.base import Signal, Strategy, StrategyResult
from nse_quant.strategies.dividend import DividendAristocratsStrategy
from nse_quant.strategies.mean_reversion import MeanReversionStrategy
from nse_quant.strategies.ml_lstm import LSTMAlphaStrategy
from nse_quant.strategies.ml_xgboost import XGBoostAlphaStrategy
from nse_quant.strategies.momentum import MomentumRotationStrategy
from nse_quant.strategies.pairs import PairsTradingStrategy
from nse_quant.strategies.portfolio_optimizer import PortfolioOptimizerStrategy
from nse_quant.strategies.quality import QualityFactorStrategy
from nse_quant.strategies.sector_rotation import SectorRotationStrategy
from nse_quant.strategies.volatility import VolatilityArbitrageStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumRotationStrategy,
    "mean_reversion": MeanReversionStrategy,
    "pairs": PairsTradingStrategy,
    "sector_rotation": SectorRotationStrategy,
    "volatility": VolatilityArbitrageStrategy,
    "dividend": DividendAristocratsStrategy,
    "quality": QualityFactorStrategy,
    "xgboost": XGBoostAlphaStrategy,
    "lstm": LSTMAlphaStrategy,
    "portfolio_optimizer": PortfolioOptimizerStrategy,
}

__all__ = [
    "Signal",
    "Strategy",
    "StrategyResult",
    "STRATEGY_REGISTRY",
    "MomentumRotationStrategy",
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "SectorRotationStrategy",
    "VolatilityArbitrageStrategy",
    "DividendAristocratsStrategy",
    "QualityFactorStrategy",
    "XGBoostAlphaStrategy",
    "LSTMAlphaStrategy",
    "PortfolioOptimizerStrategy",
]
