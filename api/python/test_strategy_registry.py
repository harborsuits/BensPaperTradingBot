#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that strategies are properly registered with the strategy registry.
"""

import logging
import sys
import os
from pprint import pprint

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our registry
try:
    from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
    logger.info("Successfully imported the strategy registry")
except ImportError as e:
    logger.error(f"Error importing strategy registry: {e}")
    sys.exit(1)

# Import our strategy templates
try:
    from trading_bot.strategy_templates import (
        StrategyTemplate, 
        OptionsStrategyTemplate, 
        ForexStrategyTemplate, 
        StocksStrategyTemplate
    )
    logger.info("Successfully imported strategy templates")
except ImportError as e:
    logger.error(f"Error importing strategy templates: {e}")
    sys.exit(1)

# Import our implemented strategies
try:
    # Import the bull call spread strategy
    from trading_bot.strategies.options.vertical_spreads.bull_call_spread_strategy_new import BullCallSpreadStrategy
    logger.info("Successfully imported BullCallSpreadStrategy")
    
    # Import the covered call strategy
    from trading_bot.strategies.options.income_strategies.covered_call_strategy_new import CoveredCallStrategy
    logger.info("Successfully imported CoveredCallStrategy")
    
except ImportError as e:
    logger.error(f"Error importing implemented strategies: {e}")
    sys.exit(1)

def main():
    """Main function to test the strategy registry."""
    logger.info("Testing strategy registry...")
    
    # Get all registered strategies
    all_strategies = StrategyRegistry.get_all_strategies()
    logger.info(f"Found {len(all_strategies)} total registered strategies")
    
    # Check if our new strategies are registered
    bull_call_spread_found = False
    covered_call_found = False
    
    for name, strategy_class in all_strategies.items():
        if "bull_call_spread" in name.lower():
            bull_call_spread_found = True
            logger.info(f"Found Bull Call Spread strategy: {name}")
            
        if "covered_call" in name.lower():
            covered_call_found = True
            logger.info(f"Found Covered Call strategy: {name}")
    
    # Report on findings
    if bull_call_spread_found:
        logger.info("✅ Bull Call Spread strategy is properly registered")
    else:
        logger.error("❌ Bull Call Spread strategy is NOT properly registered")
        
    if covered_call_found:
        logger.info("✅ Covered Call strategy is properly registered")
    else:
        logger.error("❌ Covered Call strategy is NOT properly registered")
    
    # Get strategies by asset class
    options_strategies = StrategyRegistry.get_strategies_by_asset_class("options")
    logger.info(f"Found {len(options_strategies)} options strategies")
    
    # Try to create an instance of each strategy
    try:
        bull_call = BullCallSpreadStrategy(strategy_id="test_bull_call", name="Test Bull Call")
        logger.info(f"✅ Successfully created Bull Call Spread strategy instance: {bull_call.name}")
        
        # Print some details about the strategy
        logger.info(f"Bull Call Spread parameters: {bull_call.parameters}")
    except Exception as e:
        logger.error(f"❌ Error creating Bull Call Spread strategy instance: {e}")
    
    try:
        covered_call = CoveredCallStrategy(strategy_id="test_covered_call", name="Test Covered Call")
        logger.info(f"✅ Successfully created Covered Call strategy instance: {covered_call.name}")
        
        # Print some details about the strategy
        logger.info(f"Covered Call parameters: {covered_call.parameters}")
    except Exception as e:
        logger.error(f"❌ Error creating Covered Call strategy instance: {e}")
    
    logger.info("Strategy registry test completed")

if __name__ == "__main__":
    main()
