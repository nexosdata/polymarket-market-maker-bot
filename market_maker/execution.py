"""Simulated paper-trading execution engine for prediction market market making.

Simulates limit order placement, cancellation, matching against incoming order flow,
toxic adverse selection detection, and real-time PnL/inventory variance accounting.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from typing import Dict, List, Optional, Tuple

from market_maker.models import ExecutionReport, QuotePair, Trade


class SimulatedExecutionEngine:
    """Institutional-grade paper trading and backtesting execution simulator."""

    def __init__(
        self,
        market_id: str = "default_market",
        initial_cash: float = 10_000.0,
        initial_inventory: float = 0.0,
        fee_rate_bps: float = 0.0,  # Polymarket CLOB has 0% maker fees, Kalshi ~1-2c
        adverse_selection_threshold: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        self.market_id = market_id
        self.initial_cash = initial_cash
        self.cash: float = initial_cash
        self.inventory: float = initial_inventory
        self.realized_pnl: float = 0.0
        self.fee_rate_bps = fee_rate_bps
        self.adverse_selection_threshold = adverse_selection_threshold

        # Active resting orders
        self.active_bid: Optional[Tuple[float, float]] = None  # (price, size)
        self.active_ask: Optional[Tuple[float, float]] = None  # (price, size)

        # Performance history
        self.trades: List[Trade] = []
        self.inventory_history: List[float] = [initial_inventory]
        self.mid_history: List[float] = []
        self.adverse_selection_cancels: int = 0
        self.last_mark_price: float = 0.50

        # Cost basis tracking (FIFO / average)
        self.avg_entry_price: float = 0.0

        if seed is not None:
            random.seed(seed)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def inventory_variance(self) -> float:
        """Calculate variance of inventory holdings over the trading session."""
        if len(self.inventory_history) < 2:
            return 0.0
        mean_inv = sum(self.inventory_history) / len(self.inventory_history)
        var = sum((x - mean_inv) ** 2 for x in self.inventory_history) / len(self.inventory_history)
        return round(var, 4)

    def calculate_unrealized_pnl(self, mark_price: float) -> float:
        """Mark-to-market unrealized PnL."""
        if self.inventory == 0.0:
            return 0.0
        return (mark_price - self.avg_entry_price) * self.inventory

    def get_equity(self, mark_price: float) -> float:
        """Total portfolio equity = cash + inventory * mark_price."""
        return self.cash + (self.inventory * mark_price)

    def get_total_pnl(self, mark_price: float) -> float:
        """Total PnL from start = Current Equity - Initial Cash."""
        return self.get_equity(mark_price) - self.initial_cash

    def update_quotes(self, quotes: QuotePair, current_mid: float) -> Dict[str, Any]:
        """Update active resting limit orders and check for adverse selection.

        If current mid-price jumps adversely relative to resting orders, trigger
        safety cancellation before replacement.
        """
        self.last_mark_price = current_mid
        cancellations = 0

        # 1. Adverse selection check on existing resting orders
        if self.active_bid is not None:
            bid_price, _ = self.active_bid
            # If mid drops below our resting bid price or sharply towards it:
            if current_mid <= bid_price - self.adverse_selection_threshold:
                self.active_bid = None
                self.adverse_selection_cancels += 1
                cancellations += 1

        if self.active_ask is not None:
            ask_price, _ = self.active_ask
            # If mid rises above our resting ask price or sharply towards it:
            if current_mid >= ask_price + self.adverse_selection_threshold:
                self.active_ask = None
                self.adverse_selection_cancels += 1
                cancellations += 1

        # 2. Place new resting quotes
        if quotes.bid_size > 0.0 and quotes.bid_price < current_mid:
            self.active_bid = (quotes.bid_price, quotes.bid_size)
        else:
            self.active_bid = None

        if quotes.ask_size > 0.0 and quotes.ask_price > current_mid:
            self.active_ask = (quotes.ask_price, quotes.ask_size)
        else:
            self.active_ask = None

        return {
            "adverse_cancels": cancellations,
            "resting_bid": self.active_bid,
            "resting_ask": self.active_ask,
        }

    def process_external_trade(self, side: str, price: float, size: float) -> List[Trade]:
        """Check if incoming aggressive external trade matches our resting limit orders.

        - If external trade is a SELL, it hits bids: if trade price <= our bid, we buy!
        - If external trade is a BUY, it lifts asks: if trade price >= our ask, we sell!
        """
        fills: List[Trade] = []
        side_norm = side.upper()

        if side_norm in ("SELL", "TAKER_SELL") and self.active_bid is not None:
            bid_p, bid_sz = self.active_bid
            if price <= bid_p:
                # Fill occurs at our bid price
                filled_sz = min(bid_sz, size)
                trade = self._execute_fill(side="BUY", price=bid_p, size=filled_sz)
                fills.append(trade)
                remaining = bid_sz - filled_sz
                self.active_bid = (bid_p, remaining) if remaining > 0.0 else None

        elif side_norm in ("BUY", "TAKER_BUY") and self.active_ask is not None:
            ask_p, ask_sz = self.active_ask
            if price >= ask_p:
                # Fill occurs at our ask price
                filled_sz = min(ask_sz, size)
                trade = self._execute_fill(side="SELL", price=ask_p, size=filled_sz)
                fills.append(trade)
                remaining = ask_sz - filled_sz
                self.active_ask = (ask_p, remaining) if remaining > 0.0 else None

        return fills

    def simulate_order_flow_step(
        self,
        current_mid: float,
        intensity_kappa: float = 2.0,
        dt: float = 1.0,
    ) -> List[Trade]:
        """Stochastically simulate order flow matching using Avellaneda-Stoikov arrival intensities.

        Arrival rate lambda(delta) = kappa_scale * exp(-kappa * delta)
        where delta = |quote_price - mid_price|.
        """
        self.last_mark_price = current_mid
        self.mid_history.append(current_mid)
        fills: List[Trade] = []

        # Check bid fill
        if self.active_bid is not None:
            bid_p, bid_sz = self.active_bid
            delta_b = max(0.001, current_mid - bid_p)
            lambda_b = math.exp(-intensity_kappa * delta_b * 10.0)  # scale for 0-1 price space
            prob_bid_fill = 1.0 - math.exp(-lambda_b * dt)

            if random.random() < prob_bid_fill:
                fill_size = min(bid_sz, round(random.uniform(0.2, 1.0) * bid_sz, 2))
                trade = self._execute_fill(side="BUY", price=bid_p, size=fill_size)
                fills.append(trade)
                rem = bid_sz - fill_size
                self.active_bid = (bid_p, rem) if rem > 1.0 else None

        # Check ask fill
        if self.active_ask is not None:
            ask_p, ask_sz = self.active_ask
            delta_a = max(0.001, ask_p - current_mid)
            lambda_a = math.exp(-intensity_kappa * delta_a * 10.0)
            prob_ask_fill = 1.0 - math.exp(-lambda_a * dt)

            if random.random() < prob_ask_fill:
                fill_size = min(ask_sz, round(random.uniform(0.2, 1.0) * ask_sz, 2))
                trade = self._execute_fill(side="SELL", price=ask_p, size=fill_size)
                fills.append(trade)
                rem = ask_sz - fill_size
                self.active_ask = (ask_p, rem) if rem > 1.0 else None

        return fills

    def _execute_fill(self, side: str, price: float, size: float) -> Trade:
        """Internal accounting of cash, inventory, realized PnL, and cost basis."""
        trade_id = f"trd_{uuid.uuid4().hex[:8]}"
        trade = Trade(
            trade_id=trade_id,
            market_id=self.market_id,
            side=side,
            price=round(price, 6),
            size=round(size, 2),
            timestamp=time.time(),
            is_maker=True,
        )

        cost = price * size
        fee = cost * (self.fee_rate_bps * 0.0001)

        if side == "BUY":
            self.cash -= (cost + fee)
            # Update weighted average cost basis
            if self.inventory >= 0:
                total_shares = self.inventory + size
                if total_shares > 0:
                    self.avg_entry_price = (
                        (self.inventory * self.avg_entry_price) + cost
                    ) / total_shares
            else:
                # Covering short position
                covered_shares = min(abs(self.inventory), size)
                realized = (self.avg_entry_price - price) * covered_shares
                self.realized_pnl += realized

            self.inventory += size

        else:  # SELL
            self.cash += (cost - fee)
            if self.inventory > 0:
                closed_shares = min(self.inventory, size)
                realized = (price - self.avg_entry_price) * closed_shares
                self.realized_pnl += realized
            else:
                # Entering or adding to short position
                total_short = abs(self.inventory) + size
                if total_short > 0:
                    self.avg_entry_price = (
                        (abs(self.inventory) * self.avg_entry_price) + cost
                    ) / total_short

            self.inventory -= size

        self.inventory = round(self.inventory, 4)
        self.inventory_history.append(self.inventory)
        self.trades.append(trade)

        return trade

    def get_report(self, mark_price: Optional[float] = None) -> ExecutionReport:
        """Generate structured execution report."""
        mp = mark_price or self.last_mark_price
        unrealized = self.calculate_unrealized_pnl(mp)
        total_pnl = self.realized_pnl + unrealized

        return ExecutionReport(
            timestamp=time.time(),
            cash_balance=round(self.cash, 4),
            inventory=round(self.inventory, 2),
            realized_pnl=round(self.realized_pnl, 4),
            unrealized_pnl=round(unrealized, 4),
            total_pnl=round(total_pnl, 4),
            mark_price=round(mp, 4),
            num_trades=len(self.trades),
            inventory_variance=self.inventory_variance,
            adverse_selection_cancels=self.adverse_selection_cancels,
            active_bids=1 if self.active_bid is not None else 0,
            active_asks=1 if self.active_ask is not None else 0,
        )
