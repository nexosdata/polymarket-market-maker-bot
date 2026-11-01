"""Unit tests for the Avellaneda-Stoikov Prediction Market Maker Bot."""

import math
import unittest
from typing import List

from market_maker.execution import SimulatedExecutionEngine
from market_maker.models import (
    ExecutionReport,
    OrderBookSnapshot,
    PriceLevel,
    PricingMode,
    QuotePair,
    StrategyConfig,
)
from market_maker.orderbook import Level2OrderBook
from market_maker.strategy import AvellanedaStoikovStrategy


class TestAvellanedaStoikovStrategy(unittest.TestCase):
    """Test mathematical formulas and pricing logic for prediction markets."""

    def setUp(self) -> None:
        self.config = StrategyConfig(
            gamma=0.2,
            sigma=0.3,
            kappa=2.5,
            target_spread_bps=100.0,
            max_inventory=500.0,
            pricing_mode=PricingMode.LOGIT,
            tick_size=0.001,
        )
        self.strategy = AvellanedaStoikovStrategy(self.config)

    def test_reservation_price_zero_inventory(self) -> None:
        """With zero inventory and neutral OFI, reservation price must equal mid price."""
        mid = 0.50
        r = self.strategy.calculate_reservation_price(mid_price=mid, inventory=0.0, time_left=1.0)
        self.assertAlmostEqual(r, mid, places=3)

    def test_inventory_skew_direction(self) -> None:
        """Long inventory (q > 0) must lower reservation price; short inventory (q < 0) must raise it."""
        mid = 0.55
        r_neutral = self.strategy.calculate_reservation_price(mid, inventory=0.0, time_left=1.0)
        r_long = self.strategy.calculate_reservation_price(mid, inventory=200.0, time_left=1.0)
        r_short = self.strategy.calculate_reservation_price(mid, inventory=-200.0, time_left=1.0)

        # Long YES inventory -> lower reservation price to encourage selling
        self.assertLess(r_long, r_neutral)
        # Short YES inventory -> higher reservation price to encourage buying
        self.assertGreater(r_short, r_neutral)

    def test_gamma_sensitivity(self) -> None:
        """Higher risk aversion gamma must produce a larger inventory penalty."""
        config_low_g = StrategyConfig(gamma=0.05, max_inventory=500.0)
        config_high_g = StrategyConfig(gamma=0.50, max_inventory=500.0)

        strat_low = AvellanedaStoikovStrategy(config_low_g)
        strat_high = AvellanedaStoikovStrategy(config_high_g)

        mid = 0.50
        inv = 250.0

        penalty_low = strat_low.calculate_inventory_penalty(inv, time_left=1.0, mid_price=mid)
        penalty_high = strat_high.calculate_inventory_penalty(inv, time_left=1.0, mid_price=mid)

        self.assertGreater(penalty_high, penalty_low)

    def test_optimal_spread_properties(self) -> None:
        """Optimal spread should widen with higher gamma and narrow with higher kappa."""
        spread_base = self.strategy.calculate_optimal_spread(gamma=0.1, kappa=2.0)
        spread_high_gamma = self.strategy.calculate_optimal_spread(gamma=0.5, kappa=2.0)
        spread_high_kappa = self.strategy.calculate_optimal_spread(gamma=0.1, kappa=10.0)

        self.assertGreater(spread_high_gamma, spread_base)
        self.assertLess(spread_high_kappa, spread_base)

    def test_bounded_probability_extreme_inventory(self) -> None:
        """Extreme inventories (+10,000 or -10,000) must never breach [0.001, 0.999] in all modes."""
        modes = [PricingMode.LOGIT, PricingMode.BERNOULLI, PricingMode.LINEAR_CLAMPED]
        for mode in modes:
            cfg = StrategyConfig(pricing_mode=mode, max_inventory=500.0, gamma=1.0)
            strat = AvellanedaStoikovStrategy(cfg)

            for extreme_inv in [-10_000.0, -1000.0, 1000.0, 10_000.0]:
                for mid in [0.01, 0.50, 0.99]:
                    r = strat.calculate_reservation_price(mid, extreme_inv, time_left=1.0)
                    self.assertGreaterEqual(r, cfg.min_price_bound, f"Failed for mode={mode}, inv={extreme_inv}")
                    self.assertLessEqual(r, cfg.max_price_bound, f"Failed for mode={mode}, inv={extreme_inv}")

    def test_quote_generation_invariants(self) -> None:
        """Quotes must satisfy: 0 < bid_price < ask_price < 1.0, with non-negative sizes."""
        quotes = self.strategy.calculate_quotes(mid_price=0.50, inventory=0.0)

        self.assertGreater(quotes.bid_price, 0.0)
        self.assertLess(quotes.bid_price, quotes.ask_price)
        self.assertLess(quotes.ask_price, 1.0)
        self.assertGreater(quotes.bid_size, 0.0)
        self.assertGreater(quotes.ask_size, 0.0)
        self.assertGreaterEqual(quotes.spread, self.config.tick_size)

    def test_max_inventory_tapering(self) -> None:
        """When at max inventory (+max_q), bid size must be 0; when at -max_q, ask size must be 0."""
        quotes_max_long = self.strategy.calculate_quotes(mid_price=0.50, inventory=500.0)
        self.assertEqual(quotes_max_long.bid_size, 0.0)
        self.assertGreater(quotes_max_long.ask_size, 0.0)

        quotes_max_short = self.strategy.calculate_quotes(mid_price=0.50, inventory=-500.0)
        self.assertEqual(quotes_max_short.ask_size, 0.0)
        self.assertGreater(quotes_max_short.bid_size, 0.0)


class TestLevel2OrderBook(unittest.TestCase):
    """Test Level 2 orderbook ingestion, imbalance metrics, and OFI tracking."""

    def setUp(self) -> None:
        self.book = Level2OrderBook(market_id="test_market", venue="polymarket")

    def test_naxos_snapshot_ingestion(self) -> None:
        """Ingesting NaxosData normalized snapshot must correctly sort and calculate mid/spread."""
        snapshot_payload = {
            "market_id": "0xabc123",
            "venue": "polymarket",
            "timestamp": 1714000000.0,
            "bids": [[0.51, 1500.0], [0.50, 3000.0], [0.49, 5000.0]],
            "asks": [[0.53, 1200.0], [0.54, 4000.0], [0.55, 8000.0]],
        }
        book = Level2OrderBook.from_naxos_snapshot(snapshot_payload)

        self.assertEqual(book.best_bid, 0.51)
        self.assertEqual(book.best_ask, 0.53)
        self.assertAlmostEqual(book.mid_price, 0.52, places=4)
        self.assertAlmostEqual(book.spread, 0.02, places=4)
        self.assertEqual(book.best_bid_size, 1500.0)
        self.assertEqual(book.best_ask_size, 1200.0)

    def test_polymarket_clob_ingestion(self) -> None:
        """Ingesting Polymarket CLOB dict format."""
        clob_payload = {
            "bids": [{"price": "0.45", "size": "2000"}, {"price": "0.44", "size": "3000"}],
            "asks": [{"price": "0.47", "size": "1800"}, {"price": "0.48", "size": "2500"}],
        }
        book = Level2OrderBook.from_polymarket_clob(clob_payload)
        self.assertEqual(book.best_bid, 0.45)
        self.assertEqual(book.best_ask, 0.47)
        self.assertAlmostEqual(book.mid_price, 0.46, places=4)

    def test_kalshi_cents_normalization(self) -> None:
        """Kalshi orderbooks quoting in cents (e.g. 52c, 54c) must normalize to $0.52, $0.54."""
        kalshi_payload = {
            "ticker": "KXPRES-TEST",
            "orderbook": {
                "yes": [[52, 250], [51, 600]],
                "no": [[54, 300], [55, 700]],
            },
        }
        book = Level2OrderBook.from_kalshi(kalshi_payload)
        self.assertEqual(book.best_bid, 0.52)
        self.assertEqual(book.best_ask, 0.54)
        self.assertAlmostEqual(book.mid_price, 0.53, places=4)

    def test_order_book_imbalance_obi(self) -> None:
        """Top of book imbalance I = (V_bid - V_ask) / (V_bid + V_ask)."""
        snap = OrderBookSnapshot(
            market_id="test",
            venue="polymarket",
            timestamp=1.0,
            bids=[PriceLevel(0.50, 3000.0)],
            asks=[PriceLevel(0.52, 1000.0)],
        )
        self.book.load_snapshot(snap)
        # (3000 - 1000) / 4000 = 0.50
        self.assertAlmostEqual(self.book.get_order_book_imbalance(), 0.50, places=4)

    def test_weighted_depth_imbalance(self) -> None:
        """Multi-level weighted imbalance must yield bounded result in [-1.0, 1.0]."""
        snap = OrderBookSnapshot(
            market_id="test",
            venue="polymarket",
            timestamp=1.0,
            bids=[PriceLevel(0.50, 1000.0), PriceLevel(0.49, 2000.0)],
            asks=[PriceLevel(0.52, 1000.0), PriceLevel(0.53, 500.0)],
        )
        self.book.load_snapshot(snap)
        w_imb = self.book.get_weighted_depth_imbalance(depth=2)
        self.assertGreaterEqual(w_imb, -1.0)
        self.assertLessEqual(w_imb, 1.0)
        self.assertGreater(w_imb, 0.0)  # More bid depth than ask depth

    def test_order_flow_imbalance_ofi_dynamics(self) -> None:
        """Verify Cont-Kukanov-Stoikov (2014) OFI logic across book updates."""
        # Initial snapshot
        snap1 = OrderBookSnapshot(
            market_id="test",
            venue="polymarket",
            timestamp=1.0,
            bids=[PriceLevel(0.50, 1000.0)],
            asks=[PriceLevel(0.52, 1000.0)],
        )
        self.book.load_snapshot(snap1)
        self.assertEqual(self.book.last_ofi, 0.0)

        # Step 2: Bid price jumps up from 0.50 to 0.51 with size 800 (aggressive buying)
        snap2 = OrderBookSnapshot(
            market_id="test",
            venue="polymarket",
            timestamp=2.0,
            bids=[PriceLevel(0.51, 800.0)],
            asks=[PriceLevel(0.52, 1000.0)],
        )
        self.book.load_snapshot(snap2)
        # Delta W_bid = +800, Delta W_ask = 0 => OFI = +800
        self.assertEqual(self.book.last_ofi, 800.0)

        # Step 3: Ask price drops down from 0.52 to 0.515 with size 600 (aggressive selling)
        snap3 = OrderBookSnapshot(
            market_id="test",
            venue="polymarket",
            timestamp=3.0,
            bids=[PriceLevel(0.51, 800.0)],
            asks=[PriceLevel(0.515, 600.0)],
        )
        self.book.load_snapshot(snap3)
        # Delta W_bid = 0, Delta W_ask = +600 => OFI = 0 - (+600) = -600
        self.assertEqual(self.book.last_ofi, -600.0)


class TestSimulatedExecutionEngine(unittest.TestCase):
    """Test simulated execution, fill matching, PnL tracking, and adverse selection."""

    def setUp(self) -> None:
        self.engine = SimulatedExecutionEngine(
            market_id="test_market",
            initial_cash=10_000.0,
            adverse_selection_threshold=0.02,
            seed=42,
        )

    def test_initial_state(self) -> None:
        """Engine begins with correct initial balance, zero inventory, and zero PnL."""
        self.assertEqual(self.engine.cash, 10_000.0)
        self.assertEqual(self.engine.inventory, 0.0)
        self.assertEqual(self.engine.realized_pnl, 0.0)
        self.assertEqual(self.engine.num_trades, 0)

    def test_resting_order_placement(self) -> None:
        """Updating quotes stores resting bids and asks in active state."""
        quotes = QuotePair(
            bid_price=0.49,
            ask_price=0.51,
            bid_size=100.0,
            ask_size=100.0,
            reservation_price=0.50,
            spread=0.02,
            skew=0.0,
            inventory=0.0,
        )
        res = self.engine.update_quotes(quotes, current_mid=0.50)
        self.assertEqual(self.engine.active_bid, (0.49, 100.0))
        self.assertEqual(self.engine.active_ask, (0.51, 100.0))
        self.assertEqual(res["adverse_cancels"], 0)

    def test_external_trade_matching_and_pnl(self) -> None:
        """External taker sell matching our resting bid -> BUY fill, inventory increases."""
        quotes = QuotePair(
            bid_price=0.49,
            ask_price=0.51,
            bid_size=100.0,
            ask_size=100.0,
            reservation_price=0.50,
            spread=0.02,
            skew=0.0,
            inventory=0.0,
        )
        self.engine.update_quotes(quotes, current_mid=0.50)

        # Incoming taker SELL at 0.49 for 60 contracts hits our bid
        fills = self.engine.process_external_trade(side="SELL", price=0.49, size=60.0)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].side, "BUY")
        self.assertEqual(fills[0].size, 60.0)
        self.assertEqual(fills[0].price, 0.49)

        # Balance check: cash spent = 60 * 0.49 = 29.40
        self.assertAlmostEqual(self.engine.cash, 10_000.0 - 29.40, places=3)
        self.assertEqual(self.engine.inventory, 60.0)
        self.assertEqual(self.engine.active_bid, (0.49, 40.0))  # 40 contracts remaining

        # Now an incoming taker BUY at 0.51 for 60 contracts lifts our ask
        fills2 = self.engine.process_external_trade(side="BUY", price=0.51, size=60.0)
        self.assertEqual(len(fills2), 1)
        self.assertEqual(fills2[0].side, "SELL")

        # Inventory is back to 0. Realized profit = 60 * (0.51 - 0.49) = $1.20
        self.assertEqual(self.engine.inventory, 0.0)
        self.assertAlmostEqual(self.engine.realized_pnl, 1.20, places=3)
        self.assertAlmostEqual(self.engine.get_total_pnl(mark_price=0.50), 1.20, places=3)

    def test_adverse_selection_cancellation(self) -> None:
        """Adverse mid-price movement must cancel resting limit orders."""
        quotes = QuotePair(
            bid_price=0.49,
            ask_price=0.53,
            bid_size=100.0,
            ask_size=100.0,
            reservation_price=0.51,
            spread=0.04,
            skew=0.0,
            inventory=0.0,
        )
        self.engine.update_quotes(quotes, current_mid=0.51)
        self.assertIsNotNone(self.engine.active_bid)

        # Mid crashes down to 0.46 (more than adverse_selection_threshold=0.02 below bid=0.49)
        new_quotes = QuotePair(
            bid_price=0.44,
            ask_price=0.48,
            bid_size=100.0,
            ask_size=100.0,
            reservation_price=0.46,
            spread=0.04,
            skew=0.0,
            inventory=0.0,
        )
        res = self.engine.update_quotes(new_quotes, current_mid=0.46)
        self.assertGreater(self.engine.adverse_selection_cancels, 0)

    def test_inventory_variance_calculation(self) -> None:
        """Inventory variance tracking across multiple inventory transitions."""
        # Baseline with single point has 0 variance
        self.assertEqual(self.engine.inventory_variance, 0.0)

        # Perform a few manual fills
        self.engine._execute_fill("BUY", 0.50, 100.0)
        self.engine._execute_fill("BUY", 0.50, 100.0)
        self.engine._execute_fill("SELL", 0.52, 200.0)

        var = self.engine.inventory_variance
        self.assertGreater(var, 0.0)

    def test_execution_report_generation(self) -> None:
        """Structured report contains all required risk and PnL fields."""
        report = self.engine.get_report(mark_price=0.50)
        self.assertIsInstance(report, ExecutionReport)
        self.assertEqual(report.cash_balance, 10_000.0)
        self.assertEqual(report.inventory, 0.0)
        self.assertEqual(report.total_pnl, 0.0)

    def test_short_selling_and_covering(self) -> None:
        """Selling from 0 enters short inventory; subsequent buy covers position and realizes profit."""
        # Sell 100 contracts at 0.55
        self.engine._execute_fill("SELL", 0.55, 100.0)
        self.assertEqual(self.engine.inventory, -100.0)
        self.assertEqual(self.engine.realized_pnl, 0.0)
        # Mark price at 0.55 means unrealized = 0
        self.assertAlmostEqual(self.engine.calculate_unrealized_pnl(0.55), 0.0)
        # Mark price drops to 0.50 means unrealized profit = (-100) * (0.50 - 0.55) = +$5.00
        self.assertAlmostEqual(self.engine.calculate_unrealized_pnl(0.50), 5.0)

        # Cover short by buying 100 at 0.50
        self.engine._execute_fill("BUY", 0.50, 100.0)
        self.assertEqual(self.engine.inventory, 0.0)
        self.assertAlmostEqual(self.engine.realized_pnl, 5.0)


class TestStrategyAdvanced(unittest.TestCase):
    """Test OFI drift, alternative pricing modes, and boundary edge cases."""

    def test_ofi_drift_reservation_shift(self) -> None:
        """Positive Order Flow Imbalance shifts reservation price up; negative shifts it down."""
        cfg = StrategyConfig(ofi_alpha=0.05, max_inventory=500.0)
        strat = AvellanedaStoikovStrategy(cfg)
        mid = 0.50

        r_neutral = strat.calculate_reservation_price(mid, inventory=0.0, time_left=1.0, ofi=0.0)
        r_positive_ofi = strat.calculate_reservation_price(mid, inventory=0.0, time_left=1.0, ofi=0.8)
        r_negative_ofi = strat.calculate_reservation_price(mid, inventory=0.0, time_left=1.0, ofi=-0.8)

        self.assertGreater(r_positive_ofi, r_neutral)
        self.assertLess(r_negative_ofi, r_neutral)

    def test_bernoulli_variance_peak_at_fifty_percent(self) -> None:
        """Bernoulli variance p*(1-p) peaks at p=0.50 and approaches 0 near edges."""
        strat = AvellanedaStoikovStrategy(StrategyConfig(sigma=1.0))
        var_50 = strat.calculate_bernoulli_variance(0.50)
        var_10 = strat.calculate_bernoulli_variance(0.10)
        var_90 = strat.calculate_bernoulli_variance(0.90)

        self.assertAlmostEqual(var_50, 0.25, places=3)
        self.assertAlmostEqual(var_10, 0.09, places=3)
        self.assertAlmostEqual(var_90, 0.09, places=3)
        self.assertGreater(var_50, var_10)

    def test_linear_clamped_mode(self) -> None:
        """Linear clamped mode respects boundary limits."""
        cfg = StrategyConfig(pricing_mode=PricingMode.LINEAR_CLAMPED, max_inventory=100.0, gamma=2.0)
        strat = AvellanedaStoikovStrategy(cfg)
        r_huge_long = strat.calculate_reservation_price(0.10, inventory=5000.0, time_left=1.0)
        self.assertGreaterEqual(r_huge_long, cfg.min_price_bound)


class TestCliExecution(unittest.TestCase):
    """Test CLI demo simulation runs without errors."""

    def test_cli_demo_runs_cleanly(self) -> None:
        import io
        import contextlib
        from market_maker.cli import main

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            main(["--demo", "--sim-steps", "3", "--gamma", "0.2"])

        output = stdout_buf.getvalue()
        self.assertIn("NAXOSDATA QUANTITATIVE MARKET MAKER", output)
        self.assertIn("FINAL PERFORMANCE SUMMARY", output)


if __name__ == "__main__":
    unittest.main()
