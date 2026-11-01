"""Data structures and domain models for prediction market market making."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import time


class PricingMode(str, Enum):
    """Pricing adjustment model for bounded binary prediction market outcomes."""
    LOGIT = "logit"
    BERNOULLI = "bernoulli"
    LINEAR_CLAMPED = "linear_clamped"


@dataclass(frozen=True)
class PriceLevel:
    """Individual price level in a Level 2 orderbook."""
    price: float
    size: float

    def __post_init__(self) -> None:
        if self.price < 0.0:
            raise ValueError(f"Price cannot be negative: {self.price}")
        if self.size < 0.0:
            raise ValueError(f"Size cannot be negative: {self.size}")


@dataclass
class OrderBookSnapshot:
    """Normalized snapshot of a Level 2 prediction market orderbook."""
    market_id: str
    venue: str
    timestamp: float
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb if bb is not None else ba

    @property
    def spread(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None


@dataclass
class StrategyConfig:
    """Configuration parameters for the Avellaneda-Stoikov market maker."""
    gamma: float = 0.1
    sigma: float = 0.25
    kappa: float = 2.0
    time_horizon: float = 1.0
    tick_size: float = 0.001
    target_spread_bps: float = 100.0
    min_spread: float = 0.004
    max_inventory: float = 1000.0
    pricing_mode: PricingMode = PricingMode.LOGIT
    ofi_alpha: float = 0.03
    base_quote_size: float = 100.0
    adverse_selection_threshold: float = 0.02
    min_price_bound: float = 0.001
    max_price_bound: float = 0.999

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError(f"Gamma (risk aversion) must be positive, got {self.gamma}")
        if self.sigma <= 0.0:
            raise ValueError(f"Sigma (volatility) must be positive, got {self.sigma}")
        if self.kappa <= 0.0:
            raise ValueError(f"Kappa (order arrival intensity) must be positive, got {self.kappa}")
        if self.max_inventory <= 0.0:
            raise ValueError(f"Max inventory must be positive, got {self.max_inventory}")
        if isinstance(self.pricing_mode, str):
            self.pricing_mode = PricingMode(self.pricing_mode.lower())


@dataclass
class QuotePair:
    """Calculated optimal quotes to post into the market."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    reservation_price: float
    spread: float
    skew: float
    inventory: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trade:
    """Record of an executed fill in the market."""
    trade_id: str
    market_id: str
    side: str  # 'BUY' or 'SELL'
    price: float
    size: float
    timestamp: float = field(default_factory=time.time)
    is_maker: bool = True


@dataclass
class ExecutionReport:
    """Performance and risk metrics snapshot from the execution engine."""
    timestamp: float
    cash_balance: float
    inventory: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    mark_price: float
    num_trades: int
    inventory_variance: float
    adverse_selection_cancels: int
    active_bids: int = 0
    active_asks: int = 0
