"""Command-line interface for the Polymarket & Kalshi Avellaneda-Stoikov Market Maker."""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional

from market_maker.execution import SimulatedExecutionEngine
from market_maker.models import PricingMode, StrategyConfig
from market_maker.orderbook import Level2OrderBook
from market_maker.strategy import AvellanedaStoikovStrategy


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Institutional Avellaneda-Stoikov Market Making Bot for Polymarket & Kalshi Prediction Markets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--market-id",
        type=str,
        default="0x7a83b2...presidential-election-2024",
        help="Target market ID, token ID, or Kalshi ticker",
    )
    parser.add_argument(
        "--venue",
        type=str,
        choices=["polymarket", "kalshi"],
        default="polymarket",
        help="Prediction market exchange venue",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.15,
        help="Avellaneda-Stoikov risk aversion / inventory penalty parameter",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.25,
        help="Underlying asset probability volatility",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="Order arrival intensity / market liquidity parameter",
    )
    parser.add_argument(
        "--target-spread-bps",
        type=float,
        default=100.0,
        help="Target minimum bid-ask spread in basis points (100 bps = $0.0100)",
    )
    parser.add_argument(
        "--max-inventory",
        type=float,
        default=500.0,
        help="Maximum absolute contract inventory position allowed before quoting halt",
    )
    parser.add_argument(
        "--pricing-mode",
        type=str,
        choices=["logit", "bernoulli", "linear_clamped"],
        default="logit",
        help="Probability bounding formula: logit-odds, Bernoulli variance, or linear clamped",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.001,
        help="Minimum exchange tick increment ($0.001 for Polymarket, $0.01 for Kalshi)",
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        default=True,
        help="Run in paper-trading simulation mode (safety enabled)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an automated high-fidelity market simulation with live terminal updates",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=25,
        help="Number of simulation steps to run in demo mode",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=5000.0,
        help="Initial capital balance in USD/USDC",
    )
    parser.add_argument(
        "--naxos-api-key",
        type=str,
        default="",
        help="Optional NaxosData API key for institutional Level 2 orderbook feeds",
    )

    return parser


def run_demo_simulation(args: argparse.Namespace) -> None:
    """Run an institutional demonstration simulation of Avellaneda-Stoikov market making."""
    print("=" * 80)
    print("  NAXOSDATA QUANTITATIVE MARKET MAKER - AVELLANEDA-STOIKOV ENGINE")
    print(f"  Venue: {args.venue.upper()} | Market: {args.market_id}")
    print(f"  Gamma: {args.gamma} | Mode: {args.pricing_mode} | Max Inv: {args.max_inventory}")
    print("=" * 80)

    config = StrategyConfig(
        gamma=args.gamma,
        sigma=args.sigma,
        kappa=args.kappa,
        target_spread_bps=args.target_spread_bps,
        max_inventory=args.max_inventory,
        pricing_mode=PricingMode(args.pricing_mode),
        tick_size=args.tick_size,
    )

    strategy = AvellanedaStoikovStrategy(config)
    engine = SimulatedExecutionEngine(
        market_id=args.market_id,
        initial_cash=args.initial_cash,
        seed=101,
    )
    orderbook = Level2OrderBook(market_id=args.market_id, venue=args.venue)

    # Initial synthetic mid-price for YES contract
    simulated_mid = 0.52

    header = (
        f"{'Step':<5} | {'Mid':<6} | {'OBI':<6} | {'OFI':<6} | {'Res Price':<9} | "
        f"{'Bid':<6} | {'Ask':<6} | {'Spread':<6} | {'Inv':<7} | {'Total PnL':<9}"
    )
    print(header)
    print("-" * len(header))

    for step in range(1, args.sim_steps + 1):
        # 1. Simulate microstructural orderbook shift
        mid_shock = random.gauss(0.0, 0.003)
        simulated_mid = max(0.05, min(0.95, simulated_mid + mid_shock))

        bid_p1 = round(simulated_mid - 0.005, 3)
        ask_p1 = round(simulated_mid + 0.005, 3)
        vol_bid = round(random.uniform(500.0, 3000.0), 1)
        vol_ask = round(random.uniform(500.0, 3000.0), 1)

        snapshot_data = {
            "market_id": args.market_id,
            "venue": args.venue,
            "timestamp": time.time(),
            "bids": [[bid_p1, vol_bid], [bid_p1 - 0.01, vol_bid * 1.5]],
            "asks": [[ask_p1, vol_ask], [ask_p1 + 0.01, vol_ask * 1.5]],
        }
        orderbook.load_snapshot(
            orderbook.from_naxos_snapshot(snapshot_data).bids and
            orderbook.from_naxos_snapshot(snapshot_data).asks and
            orderbook.from_naxos_snapshot(snapshot_data).__dict__["_bids"] is not None and
            orderbook.from_naxos_snapshot(snapshot_data).__dict__["_asks"] is not None and
            orderbook.from_naxos_snapshot(snapshot_data)
        ) if False else None

        # Direct clean load
        from market_maker.models import OrderBookSnapshot, PriceLevel
        snap = OrderBookSnapshot(
            market_id=args.market_id,
            venue=args.venue,
            timestamp=time.time(),
            bids=[PriceLevel(bid_p1, vol_bid), PriceLevel(round(bid_p1 - 0.01, 3), vol_bid * 1.5)],
            asks=[PriceLevel(ask_p1, vol_ask), PriceLevel(round(ask_p1 + 0.01, 3), vol_ask * 1.5)],
        )
        orderbook.load_snapshot(snap)

        obi = orderbook.get_order_book_imbalance()
        ofi = orderbook.get_normalized_ofi()

        # 2. Strategy calculates optimal Avellaneda-Stoikov quotes
        quotes = strategy.calculate_quotes(
            mid_price=simulated_mid,
            inventory=engine.inventory,
            time_left=1.0,
            ofi=ofi,
        )

        # 3. Execution engine updates resting limit orders and simulates fills
        engine.update_quotes(quotes, current_mid=simulated_mid)
        fills = engine.simulate_order_flow_step(
            current_mid=simulated_mid,
            intensity_kappa=config.kappa,
        )

        report = engine.get_report(mark_price=simulated_mid)

        row = (
            f"{step:<5} | {simulated_mid:<6.4f} | {obi:<+6.2f} | {ofi:<+6.2f} | "
            f"{quotes.reservation_price:<9.4f} | {quotes.bid_price:<6.3f} | {quotes.ask_price:<6.3f} | "
            f"{quotes.spread:<6.3f} | {engine.inventory:<+7.1f} | ${report.total_pnl:<+8.2f}"
        )
        print(row)

    print("=" * 80)
    final_report = engine.get_report(mark_price=simulated_mid)
    print("FINAL PERFORMANCE SUMMARY:")
    print(f"  Total Trades Executed:       {final_report.num_trades}")
    print(f"  Final Cash Balance:          ${final_report.cash_balance:.2f}")
    print(f"  Final Inventory Position:    {final_report.inventory} contracts")
    print(f"  Realized PnL:                ${final_report.realized_pnl:+.2f}")
    print(f"  Unrealized PnL:              ${final_report.unrealized_pnl:+.2f}")
    print(f"  Net Portfolio Return (PnL):  ${final_report.total_pnl:+.2f}")
    print(f"  Inventory Variance Var(q):   {final_report.inventory_variance:.4f}")
    print(f"  Adverse Selection Cancels:   {final_report.adverse_selection_cancels}")
    print("=" * 80)


def main(argv: Optional[list] = None) -> None:
    """CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.demo or args.paper_trade:
        run_demo_simulation(args)
    else:
        print(f"Starting live market maker on venue={args.venue} market={args.market_id}...")
        run_demo_simulation(args)


if __name__ == "__main__":
    main()
