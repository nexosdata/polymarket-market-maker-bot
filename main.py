#!/usr/bin/env python3
"""Main entry point for Polymarket & Kalshi Market Maker Bot."""

import sys
from market_maker.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
