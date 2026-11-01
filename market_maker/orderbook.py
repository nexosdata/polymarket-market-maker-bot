"""Orderbook modeling, Level 2 ingestion, and Order Flow Imbalance (OFI) calculations.

Supports normalized NaxosData feeds, Polymarket CLOB JSON, and Kalshi market data.
Computes microstructural metrics:
- Top-of-book and multi-level weighted Order Book Imbalance (OBI)
- Discrete Order Flow Imbalance (OFI) via the Cont-Kukanov-Stoikov (2014) formulation
- Micro-price and effective spread
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from market_maker.models import OrderBookSnapshot, PriceLevel


class Level2OrderBook:
    """Institutional Level 2 orderbook maintaining sorted bid and ask ladders."""

    def __init__(self, market_id: str = "default_market", venue: str = "polymarket") -> None:
        self.market_id = market_id
        self.venue = venue
        self.timestamp: float = time.time()
        # Price -> Size mapping
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}

        # Prior top of book state for Order Flow Imbalance (OFI) tracking
        self._prev_best_bid_price: Optional[float] = None
        self._prev_best_bid_size: float = 0.0
        self._prev_best_ask_price: Optional[float] = None
        self._prev_best_ask_size: float = 0.0

        # Rolling OFI accumulator
        self._cumulative_ofi: float = 0.0
        self._last_ofi_delta: float = 0.0

    @property
    def bids(self) -> List[PriceLevel]:
        """Descending order of bids (highest price first)."""
        sorted_prices = sorted(self._bids.keys(), reverse=True)
        return [PriceLevel(p, self._bids[p]) for p in sorted_prices if self._bids[p] > 0.0]

    @property
    def asks(self) -> List[PriceLevel]:
        """Ascending order of asks (lowest price first)."""
        sorted_prices = sorted(self._asks.keys())
        return [PriceLevel(p, self._asks[p]) for p in sorted_prices if self._asks[p] > 0.0]

    @property
    def best_bid(self) -> Optional[float]:
        valid_bids = [p for p, sz in self._bids.items() if sz > 0.0]
        return max(valid_bids) if valid_bids else None

    @property
    def best_ask(self) -> Optional[float]:
        valid_asks = [p for p, sz in self._asks.items() if sz > 0.0]
        return min(valid_asks) if valid_asks else None

    @property
    def best_bid_size(self) -> float:
        bb = self.best_bid
        return self._bids.get(bb, 0.0) if bb is not None else 0.0

    @property
    def best_ask_size(self) -> float:
        ba = self.best_ask
        return self._asks.get(ba, 0.0) if ba is not None else 0.0

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return round((bb + ba) / 2.0, 6)
        return bb if bb is not None else ba

    @property
    def spread(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return round(ba - bb, 6)
        return None

    def clear(self) -> None:
        """Clear book levels."""
        self._bids.clear()
        self._asks.clear()

    def update_level(self, side: str, price: float, size: float) -> None:
        """Update or delete a price level (size=0 deletes the level)."""
        side_norm = side.upper()
        p = round(float(price), 6)
        s = float(size)

        if side_norm in ("BID", "BUY", "BIDS"):
            if s <= 0.0:
                self._bids.pop(p, None)
            else:
                self._bids[p] = s
        elif side_norm in ("ASK", "SELL", "ASKS"):
            if s <= 0.0:
                self._asks.pop(p, None)
            else:
                self._asks[p] = s
        else:
            raise ValueError(f"Unknown side: {side}")

    def load_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Replace internal state with a clean snapshot and update OFI metrics."""
        self.market_id = snapshot.market_id
        self.venue = snapshot.venue
        self.timestamp = snapshot.timestamp

        self._bids.clear()
        self._asks.clear()

        for level in snapshot.bids:
            if level.size > 0.0:
                self._bids[round(level.price, 6)] = level.size

        for level in snapshot.asks:
            if level.size > 0.0:
                self._asks[round(level.price, 6)] = level.size

        self._update_ofi()

    def _update_ofi(self) -> float:
        """Calculate Order Flow Imbalance (OFI) based on top-of-book transitions.

        Cont, Kukanov & Stoikov (2014) definition:
        Delta W_bid =
            v_t^b           if b_t > b_{t-1}
            v_t^b - v_{t-1}^b if b_t == b_{t-1}
            -v_{t-1}^b      if b_t < b_{t-1}

        Delta W_ask =
            -v_{t-1}^a      if a_t > a_{t-1}
            v_t^a - v_{t-1}^a if a_t == a_{t-1}
            v_t^a           if a_t < a_{t-1}

        OFI = Delta W_bid - Delta W_ask
        """
        curr_b = self.best_bid
        curr_b_sz = self.best_bid_size
        curr_a = self.best_ask
        curr_a_sz = self.best_ask_size

        if self._prev_best_bid_price is None or self._prev_best_ask_price is None:
            # Baseline snapshot: establish baseline
            self._prev_best_bid_price = curr_b
            self._prev_best_bid_size = curr_b_sz
            self._prev_best_ask_price = curr_a
            self._prev_best_ask_size = curr_a_sz
            self._last_ofi_delta = 0.0
            return 0.0

        if curr_b is None or curr_a is None:
            return 0.0

        # Calculate bid side flow
        if curr_b > self._prev_best_bid_price:
            delta_bid = curr_b_sz
        elif math.isclose(curr_b, self._prev_best_bid_price, abs_tol=1e-7):
            delta_bid = curr_b_sz - self._prev_best_bid_size
        else:
            delta_bid = -self._prev_best_bid_size

        # Calculate ask side flow
        if curr_a > self._prev_best_ask_price:
            delta_ask = -self._prev_best_ask_size
        elif math.isclose(curr_a, self._prev_best_ask_price, abs_tol=1e-7):
            delta_ask = curr_a_sz - self._prev_best_ask_size
        else:
            delta_ask = curr_a_sz

        ofi_delta = delta_bid - delta_ask
        self._last_ofi_delta = ofi_delta
        self._cumulative_ofi += ofi_delta

        # Update cache for next iteration
        self._prev_best_bid_price = curr_b
        self._prev_best_bid_size = curr_b_sz
        self._prev_best_ask_price = curr_a
        self._prev_best_ask_size = curr_a_sz

        return ofi_delta

    @property
    def last_ofi(self) -> float:
        """Most recent step OFI."""
        return self._last_ofi_delta

    @property
    def cumulative_ofi(self) -> float:
        """Cumulative OFI accumulated over session."""
        return self._cumulative_ofi

    def get_order_book_imbalance(self) -> float:
        """Top-of-book imbalance I = (V_bid - V_ask) / (V_bid + V_ask) in [-1.0, 1.0]."""
        bb_sz = self.best_bid_size
        ba_sz = self.best_ask_size
        total = bb_sz + ba_sz
        if total <= 0.0:
            return 0.0
        return (bb_sz - ba_sz) / total

    def get_weighted_depth_imbalance(self, depth: int = 5, decay: float = 0.5) -> float:
        """Calculate exponential volume-weighted book imbalance across top K levels.

        Parameters
        ----------
        depth : int
            Number of price levels to inspect on both bid and ask sides.
        decay : float
            Decay weight per level distance (e.g. 0.5 => level 0 weight 1.0, level 1 weight 0.5).
        """
        bids = self.bids[:depth]
        asks = self.asks[:depth]

        weighted_bid_vol = 0.0
        for i, level in enumerate(bids):
            weight = math.exp(-decay * i)
            weighted_bid_vol += level.size * weight

        weighted_ask_vol = 0.0
        for i, level in enumerate(asks):
            weight = math.exp(-decay * i)
            weighted_ask_vol += level.size * weight

        total_vol = weighted_bid_vol + weighted_ask_vol
        if total_vol <= 0.0:
            return 0.0

        return (weighted_bid_vol - weighted_ask_vol) / total_vol

    def get_micro_price(self) -> Optional[float]:
        """Micro-price incorporating top-of-book volume imbalance:

        P_micro = (V_bid * P_ask + V_ask * P_bid) / (V_bid + V_ask)
        """
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return self.mid_price

        vb = self.best_bid_size
        va = self.best_ask_size
        total_vol = vb + va
        if total_vol <= 0.0:
            return (bb + ba) / 2.0

        return round((vb * ba + va * bb) / total_vol, 6)

    def get_normalized_ofi(self, normalizing_volume: Optional[float] = None) -> float:
        """Return normalized OFI bounded in [-1.0, 1.0]."""
        denom = normalizing_volume or (self.best_bid_size + self.best_ask_size)
        if denom <= 0.0:
            return 0.0
        ratio = self._last_ofi_delta / denom
        return max(-1.0, min(1.0, ratio))

    # Ingestion parsers

    @classmethod
    def from_naxos_snapshot(cls, data: Dict[str, Any]) -> Level2OrderBook:
        """Parse NaxosData normalized L2 market snapshot."""
        market_id = str(data.get("market_id", data.get("id", "unknown")))
        venue = str(data.get("venue", "polymarket"))
        book = cls(market_id=market_id, venue=venue)

        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        ts = float(data.get("timestamp", time.time()))

        parsed_bids: List[PriceLevel] = []
        for item in bids_raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed_bids.append(PriceLevel(float(item[0]), float(item[1])))
            elif isinstance(item, dict):
                parsed_bids.append(PriceLevel(float(item["price"]), float(item["size"])))

        parsed_asks: List[PriceLevel] = []
        for item in asks_raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed_asks.append(PriceLevel(float(item[0]), float(item[1])))
            elif isinstance(item, dict):
                parsed_asks.append(PriceLevel(float(item["price"]), float(item["size"])))

        snapshot = OrderBookSnapshot(
            market_id=market_id,
            venue=venue,
            timestamp=ts,
            bids=parsed_bids,
            asks=parsed_asks,
        )
        book.load_snapshot(snapshot)
        return book

    @classmethod
    def from_polymarket_clob(cls, data: Dict[str, Any], market_id: str = "polymarket_clob") -> Level2OrderBook:
        """Parse Polymarket CLOB book structure."""
        book = cls(market_id=market_id, venue="polymarket")
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        parsed_bids: List[PriceLevel] = []
        for b in bids_raw:
            p = float(b["price"]) if isinstance(b, dict) else float(b[0])
            s = float(b["size"]) if isinstance(b, dict) else float(b[1])
            parsed_bids.append(PriceLevel(p, s))

        parsed_asks: List[PriceLevel] = []
        for a in asks_raw:
            p = float(a["price"]) if isinstance(a, dict) else float(a[0])
            s = float(a["size"]) if isinstance(a, dict) else float(a[1])
            parsed_asks.append(PriceLevel(p, s))

        snapshot = OrderBookSnapshot(
            market_id=market_id,
            venue="polymarket",
            timestamp=float(data.get("timestamp", time.time())),
            bids=parsed_bids,
            asks=parsed_asks,
        )
        book.load_snapshot(snapshot)
        return book

    @classmethod
    def from_kalshi(cls, data: Dict[str, Any], market_id: Optional[str] = None) -> Level2OrderBook:
        """Parse Kalshi orderbook structure (prices often reported in cents 0-100)."""
        ticker = market_id or data.get("ticker", "kalshi_market")
        book = cls(market_id=ticker, venue="kalshi")

        ob = data.get("orderbook", data)
        raw_bids = ob.get("yes", ob.get("bids", []))
        raw_asks = ob.get("no", ob.get("asks", []))

        parsed_bids: List[PriceLevel] = []
        for b in raw_bids:
            p = float(b[0]) if isinstance(b, (list, tuple)) else float(b["price"])
            s = float(b[1]) if isinstance(b, (list, tuple)) else float(b["size"])
            # Normalize cents to dollars if needed
            if p > 1.0:
                p = p / 100.0
            parsed_bids.append(PriceLevel(p, s))

        parsed_asks: List[PriceLevel] = []
        for a in raw_asks:
            p = float(a[0]) if isinstance(a, (list, tuple)) else float(a["price"])
            s = float(a[1]) if isinstance(a, (list, tuple)) else float(a["size"])
            if p > 1.0:
                p = p / 100.0
            parsed_asks.append(PriceLevel(p, s))

        snapshot = OrderBookSnapshot(
            market_id=ticker,
            venue="kalshi",
            timestamp=float(data.get("timestamp", time.time())),
            bids=parsed_bids,
            asks=parsed_asks,
        )
        book.load_snapshot(snapshot)
        return book
