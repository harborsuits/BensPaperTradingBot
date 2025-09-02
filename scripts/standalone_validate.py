#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Validation Script for Straddle/Strangle Strategy
This script bypasses the strategy registry entirely
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Define minimal strategy enums directly here - no imports
class StrategyType(Enum):
    VOLATILITY = "volatility"

class AssetClass(Enum):
    OPTIONS = "options"

class TimeFrame(Enum):
    SWING = "swing"

class MarketRegime(Enum):
    VOLATILE = "volatile"

# Mock market data
class MockMarketData:
    def __init__(self):
        self.data = {
            'SPY': {'price': 450.0, 'volatility': 20.0},
            'AAPL': {'price': 175.0, 'volatility': 25.0}
        }
    
    def get_price(self, symbol):
        return self.data.get(symbol, {}).get('price', 0)
    
    def get_volatility(self, symbol):
        return self.data.get(symbol, {}).get('volatility', 0)

# Mock option chains
class MockOptionChains:
    def __init__(self):
        self.chains = {
            'SPY': [
                {'strike': 445, 'type': 'call', 'expiration': '2023-12-15', 'price': 12.5},
                {'strike': 445, 'type': 'put', 'expiration': '2023-12-15', 'price': 10.2},
                {'strike': 455, 'type': 'call', 'expiration': '2023-12-15', 'price': 8.3},
                {'strike': 455, 'type': 'put', 'expiration': '2023-12-15', 'price': 15.7}
            ]
        }
    
    def get_chain(self, symbol):
        return self.chains.get(symbol, [])

def main():
    logger.info("----- STANDALONE STRADDLE/STRANGLE STRATEGY VALIDATION -----")
    
    try:
        # Direct import of the strategy class
        from trading_bot.strategies.options.volatility_spreads.straddle_strangle_strategy import StraddleStrangleStrategy
        logger.info("✓ Successfully imported StraddleStrangleStrategy")
        
        # Create strategy instance with minimal configuration
        config = {
            'symbols': ['SPY', 'AAPL'],
            'vix_threshold': 20,
            'min_volatility': 15,
            'strategy_type': StrategyType.VOLATILITY,
            'asset_class': AssetClass.OPTIONS,
            'timeframe': TimeFrame.SWING,
            'preferred_regime': MarketRegime.VOLATILE
        }
        
        strategy = StraddleStrangleStrategy(config=config)
        logger.info("✓ Successfully instantiated StraddleStrangleStrategy")
        
        # Create mock data
        market_data = MockMarketData()
        option_chains = MockOptionChains()
        
        # Test signal generation
        signals = strategy.generate_signals(market_data, option_chains)
        if signals:
            logger.info(f"✓ Strategy generated {len(signals)} trading signals")
            for i, signal in enumerate(signals):
                logger.info(f"  Signal {i+1}: {signal}")
        else:
            logger.info("✓ Strategy generated no signals (this might be expected)")
        
        logger.info("✓ Strategy validation successful")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import Error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Validation Error: {e}")
        logger.exception("Details:")
        return False
    finally:
        logger.info("----- VALIDATION COMPLETE -----")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
