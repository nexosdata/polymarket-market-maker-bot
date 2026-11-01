"""Avellaneda-Stoikov market-making formula adapted for binary prediction markets ($0.00 - $1.00 bounded outcomes).

Calculates reservation (indifference) price, optimal bid/ask spread, and inventory skew penalty (gamma).
Supports Bernoulli outcome variance, logit-space indifference pricing, and Order Flow Imbalance (OFI).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

from market_maker.models import PricingMode, QuotePair, StrategyConfig


class AvellanedaStoikovStrategy:
    """Quantitative market making strategy for binary prediction markets (Polymarket / Kalshi).

    Adapts the classical Avellaneda-Stoikov (2008) framework to the bounded domain [0.0, 1.0]
    of binary prediction markets where contracts settle at either $1.00 (YES) or $0.00 (NO).
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or StrategyConfig()

    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-4) -> float:
        """Compute logit(p) with numerical stabilization to avoid inf/nan."""
        clamped = max(eps, min(1.0 - eps, p))
        return math.log(clamped / (1.0 - clamped))

    @staticmethod
    def _safe_logistic(x: float) -> float:
        """Compute standard logistic sigmoid function 1 / (1 + exp(-x))."""
        if x > 35.0:
            return 1.0 - 1e-6
        if x < -35.0:
            return 1e-6
        return 1.0 / (1.0 + math.exp(-x))

    def calculate_bernoulli_variance(self, p: float) -> float:
        """Instantaneous variance for a binary contract with probability p.

        For a binary Bernoulli variable, Var(X) = p * (1 - p).
        Scaled by annual base volatility sigma^2.
        """
        clamped_p = max(0.001, min(0.999, p))
        return (self.config.sigma ** 2) * clamped_p * (1.0 - clamped_p)

    def calculate_reservation_price(
        self,
        mid_price: float,
        inventory: float,
        time_left: float,
        ofi: float = 0.0,
    ) -> float:
        """Calculate the indifference (reservation) price r(s, q, t).

        In prediction markets:
        - When inventory q > 0 (long YES), reservation price drops below mid to encourage
          selling and penalize buying.
        - When inventory q < 0 (short YES / long NO), reservation price shifts above mid
          to encourage buying YES back.
        - Positive OFI (Order Flow Imbalance) indicates net buying pressure, shifting r upwards.

        Parameters
        ----------
        mid_price : float
            Current market mid-price in (0.0, 1.0).
        inventory : float
            Current contract inventory q (positive = long YES, negative = short YES).
        time_left : float
            Normalized time remaining tau = (T - t), where tau in (0, T].
        ofi : float
            Normalized order flow imbalance in [-1.0, 1.0].
        """
        tau = max(1e-4, time_left)
        s = max(self.config.min_price_bound, min(self.config.max_price_bound, mid_price))

        # Micro-price drift adjustment from Order Flow Imbalance
        drift = self.config.ofi_alpha * ofi

        if self.config.pricing_mode == PricingMode.LOGIT:
            # Theoretical gold standard: project into unbounded log-odds space
            # logit(r) = logit(s) - q_normalized * gamma * sigma^2 * tau + drift
            norm_q = inventory / self.config.max_inventory
            log_odds = self._safe_logit(s)
            skew_shift = norm_q * self.config.gamma * (self.config.sigma ** 2) * tau
            logit_r = log_odds - skew_shift + (drift * 4.0)
            reservation_p = self._safe_logistic(logit_r)

        elif self.config.pricing_mode == PricingMode.BERNOULLI:
            # Quadratic Bernoulli variance scaling: sigma^2(s) = sigma^2 * s * (1 - s)
            var_s = self.calculate_bernoulli_variance(s)
            norm_q = inventory / self.config.max_inventory
            inventory_penalty = norm_q * self.config.gamma * var_s * tau
            reservation_p = s - inventory_penalty + drift

        else:  # LINEAR_CLAMPED
            norm_q = inventory / self.config.max_inventory
            inventory_penalty = norm_q * self.config.gamma * (self.config.sigma ** 2) * tau
            reservation_p = s - inventory_penalty + drift

        # Strictly enforce bounds [min_price_bound, max_price_bound]
        return max(self.config.min_price_bound, min(self.config.max_price_bound, reservation_p))

    def calculate_optimal_spread(
        self,
        gamma: Optional[float] = None,
        kappa: Optional[float] = None,
        time_left: float = 1.0,
    ) -> float:
        """Calculate the theoretical Avellaneda-Stoikov spread:

        S* = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / (kappa * scaling))

        Combined with minimum spread constraints and target spread bps.
        Monotonically increases with risk aversion gamma.
        Monotonically decreases with market liquidity intensity kappa.
        """
        g = gamma if gamma is not None else self.config.gamma
        k = kappa if kappa is not None else self.config.kappa
        tau = max(1e-4, time_left)

        # Base volatility risk premium
        risk_premium = g * (self.config.sigma ** 2) * tau

        # Liquidity compensation scaled for probability space [0, 1]
        # Prediction market prices trade in cents, so order arrival intensity
        # scales with tick/bps density.
        kappa_scaled = max(0.1, k * 50.0)
        liquidity_spread = (2.0 / max(1e-4, g)) * math.log(1.0 + (g / kappa_scaled))

        as_spread = risk_premium + liquidity_spread

        # Target spread from basis points (1 bp = 0.0001, on $1.00 nominal contract)
        bps_spread = self.config.target_spread_bps * 0.0001

        # Enforce minimum market constraints
        spread = max(as_spread, bps_spread, self.config.min_spread)

        # Spread cannot exceed 0.50 (binary contracts trade between 0.0 and 1.0)
        return min(0.50, max(self.config.tick_size * 2.0, spread))

    def calculate_inventory_penalty(self, inventory: float, time_left: float, mid_price: float = 0.5) -> float:
        """Return the exact price deviation penalty induced by current inventory.

        penalty = mid_price - reservation_price
        Positive when inventory > 0 (shading downwards).
        Negative when inventory < 0 (shading upwards).
        """
        r = self.calculate_reservation_price(mid_price, inventory, time_left, ofi=0.0)
        return mid_price - r

    def round_to_tick(self, price: float) -> float:
        """Round price to the configured minimum tick size."""
        tick = self.config.tick_size
        return round(round(price / tick) * tick, 6)

    def calculate_quotes(
        self,
        mid_price: float,
        inventory: float,
        time_left: float = 1.0,
        ofi: float = 0.0,
    ) -> QuotePair:
        """Compute optimal two-sided quotes (bid/ask) given the market state.

        Guarantees:
        - Strict ordering: 0.0 < bid_price < ask_price < 1.0
        - Reservation price shading according to risk aversion gamma & inventory q
        - Size tapering as inventory approaches max_inventory
        - Minimum spread adherence

        Parameters
        ----------
        mid_price : float
            Current reference mid-price in (0.0, 1.0).
        inventory : float
            Current contract position q in [-max_inv, max_inv].
        time_left : float
            Normalized time to horizon tau in (0, 1].
        ofi : float
            Normalized Order Flow Imbalance in [-1.0, 1.0].
        """
        tick = self.config.tick_size
        s = max(self.config.min_price_bound, min(self.config.max_price_bound, mid_price))
        r = self.calculate_reservation_price(s, inventory, time_left, ofi=ofi)
        optimal_spread = self.calculate_optimal_spread()

        half_spread = optimal_spread / 2.0

        raw_bid = r - half_spread
        raw_ask = r + half_spread

        # Snap to tick size
        bid_price = self.round_to_tick(raw_bid)
        ask_price = self.round_to_tick(raw_ask)

        # Enforce market structure invariants:
        # 1. Ask must be at least 1 tick greater than bid
        if ask_price <= bid_price:
            ask_price = self.round_to_tick(bid_price + tick)

        # 2. Both quotes must respect boundary conditions
        if bid_price < self.config.min_price_bound:
            bid_price = self.round_to_tick(self.config.min_price_bound)
            if ask_price <= bid_price:
                ask_price = self.round_to_tick(bid_price + tick)

        if ask_price > self.config.max_price_bound:
            ask_price = self.round_to_tick(self.config.max_price_bound)
            if bid_price >= ask_price:
                bid_price = self.round_to_tick(ask_price - tick)

        # 3. Calculate dynamic quoting sizes based on inventory headroom
        max_q = self.config.max_inventory
        base_size = self.config.base_quote_size

        # As inventory gets positive, reduce bid size (tapering). Stop at max_q.
        if inventory >= max_q:
            bid_size = 0.0
        elif inventory > 0:
            bid_size = round(base_size * (1.0 - (inventory / max_q)), 2)
        else:
            # Inventory is negative (short), we want to buy back: quote full or bonus size
            multiplier = min(1.5, 1.0 + abs(inventory) / (2.0 * max_q))
            bid_size = round(base_size * multiplier, 2)

        # As inventory gets negative, reduce ask size. Stop at -max_q.
        if inventory <= -max_q:
            ask_size = 0.0
        elif inventory < 0:
            ask_size = round(base_size * (1.0 - (abs(inventory) / max_q)), 2)
        else:
            # Inventory is positive, we want to unload: quote full or bonus size
            multiplier = min(1.5, 1.0 + inventory / (2.0 * max_q))
            ask_size = round(base_size * multiplier, 2)

        realized_spread = round(ask_price - bid_price, 6)
        skew = round(r - s, 6)

        return QuotePair(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=round(r, 6),
            spread=realized_spread,
            skew=skew,
            inventory=inventory,
        )
