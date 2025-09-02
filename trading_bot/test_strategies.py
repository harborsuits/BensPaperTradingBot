#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Testing Script

This script tests the proper registration and functionality of trading strategies in BensBot.
"""

import os
import sys
import logging
import inspect
from pprint import pprint

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
try:
    from trading_bot.strategies.factory.strategy_registry import (
        StrategyRegistry, AssetClass, StrategyType, MarketRegime, TimeFrame
    )
    print("✅ Successfully imported StrategyRegistry")
except ImportError as e:
    print(f"❌ Error importing StrategyRegistry: {e}")
    sys.exit(1)

# Import our new strategies
try:
    # Force import of our new strategies to ensure they're registered
    import trading_bot.strategies.options.vertical_spreads.bull_call_spread_strategy_new
    import trading_bot.strategies.options.income_strategies.covered_call_strategy_new
    print("✅ Successfully imported strategy modules")
except ImportError as e:
    print(f"❌ Error importing strategy modules: {e}")

def test_strategies():
    """Test the registration and functionality of trading strategies."""
    print("\n=== TESTING STRATEGY REGISTRATION ===\n")
    
    # Get all registered strategies
    strategies = StrategyRegistry._strategies
    if not strategies:
        print("❌ No strategies found in the registry")
        return
        
    print(f"Found {len(strategies)} registered strategies:")
    for name in strategies:
        print(f"  - {name}")
    
    # Check for our new strategies specifically
    bull_call_name = "BullCallSpreadStrategy"
    covered_call_name = "CoveredCallStrategy"
    
    if bull_call_name in strategies:
        print(f"\n✅ {bull_call_name} is properly registered")
        bull_call_metadata = StrategyRegistry.get_strategy_metadata(bull_call_name)
        print("  Metadata:")
        for key, value in bull_call_metadata.items():
            print(f"    {key}: {value}")
    else:
        print(f"\n❌ {bull_call_name} is NOT registered")
    
    if covered_call_name in strategies:
        print(f"\n✅ {covered_call_name} is properly registered")
        covered_call_metadata = StrategyRegistry.get_strategy_metadata(covered_call_name)
        print("  Metadata:")
        for key, value in covered_call_metadata.items():
            print(f"    {key}: {value}")
    else:
        print(f"\n❌ {covered_call_name} is NOT registered")
    
    # Check filtering by asset class
    print("\n=== TESTING FILTERS ===\n")
    options_strategies = StrategyRegistry.get_strategies_by_asset_class(AssetClass.OPTIONS)
    print(f"Options strategies: {options_strategies}")
    
    # Check filtering by strategy type
    income_strategies = StrategyRegistry.get_strategies_by_type(StrategyType.INCOME)
    print(f"Income strategies: {income_strategies}")
    
    momentum_strategies = StrategyRegistry.get_strategies_by_type(StrategyType.MOMENTUM)
    print(f"Momentum strategies: {momentum_strategies}")
    
    # Check filtering by market regime
    trending_strategies = StrategyRegistry.get_strategies_by_market_regime(MarketRegime.TRENDING)
    print(f"Trending market strategies: {trending_strategies}")
    
    ranging_strategies = StrategyRegistry.get_strategies_by_market_regime(MarketRegime.RANGING)
    print(f"Ranging market strategies: {ranging_strategies}")

if __name__ == "__main__":
    test_strategies()
