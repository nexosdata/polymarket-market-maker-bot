"""Polymarket & Kalshi Avellaneda-Stoikov Market Maker Bot.

Institutional-grade quantitative market making algorithms adapted for
binary prediction markets ($0.00 - $1.00 bounded outcomes).
"""

from market_maker.models import (
    ExecutionReport,
    OrderBookSnapshot,
    PriceLevel,
    PricingMode,
    QuotePair,
    StrategyConfig,
    Trade,
)
from market_maker.orderbook import Level2OrderBook
from market_maker.strategy import AvellanedaStoikovStrategy
from market_maker.execution import SimulatedExecutionEngine

__version__ = "1.0.0"
__all__ = [
    "AvellanedaStoikovStrategy",
    "Level2OrderBook",
    "SimulatedExecutionEngine",
    "StrategyConfig",
    "PricingMode",
    "PriceLevel",
    "OrderBookSnapshot",
    "QuotePair",
    "Trade",
    "ExecutionReport",
]
