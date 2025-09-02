#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Strategy Registration Check

This script performs a basic check of the strategy registry without requiring
all dependencies to be installed.
"""

import sys
import os
import logging
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_registry_module():
    """Test the registry module imports correctly."""
    try:
        from trading_bot.strategies_new.factory import registry
        logger.info("✓ Successfully imported registry module")
        return registry
    except ImportError as e:
        logger.error(f"✗ Failed to import registry module: {str(e)}")
        return None

def check_strategy_modules():
    """Check that strategy modules can be imported."""
    strategy_modules = [
        'trading_bot.strategies_new.stocks.gap.gap_trading_strategy',
        'trading_bot.strategies_new.stocks.event.news_sentiment_strategy',
        'trading_bot.strategies_new.stocks.event.earnings_announcement_strategy',
        'trading_bot.strategies_new.stocks.volume.volume_surge_strategy',
        'trading_bot.strategies_new.stocks.short.short_selling_strategy',
        'trading_bot.strategies_new.stocks.sector.sector_rotation_strategy',
    ]
    
    success_count = 0
    
    for module_path in strategy_modules:
        try:
            module = importlib.import_module(module_path)
            logger.info(f"✓ Successfully imported {module_path}")
            success_count += 1
        except ImportError as e:
            logger.error(f"✗ Failed to import {module_path}: {str(e)}")
    
    logger.info(f"Successfully imported {success_count} out of {len(strategy_modules)} modules")
    return success_count == len(strategy_modules)

def check_strategy_registration(registry):
    """Check that strategies are properly registered."""
    if not registry:
        return False
    
    # Import the modules to trigger registration
    modules_to_import = [
        'trading_bot.strategies_new.stocks.gap.gap_trading_strategy',
        'trading_bot.strategies_new.stocks.event.news_sentiment_strategy',
        'trading_bot.strategies_new.stocks.event.earnings_announcement_strategy',
        'trading_bot.strategies_new.stocks.volume.volume_surge_strategy',
        'trading_bot.strategies_new.stocks.short.short_selling_strategy',
        'trading_bot.strategies_new.stocks.sector.sector_rotation_strategy',
    ]
    
    for module_path in modules_to_import:
        try:
            importlib.import_module(module_path)
        except ImportError:
            pass  # Already logged in check_strategy_modules
    
    # Get registered strategies
    all_strategies = registry.get_registered_strategies()
    
    # Print total count
    strategy_count = len(all_strategies)
    logger.info(f"Total registered strategies: {strategy_count}")
    
    # Print all registered strategies
    logger.info("Registered strategies:")
    for name, info in all_strategies.items():
        logger.info(f"  - {name} ({info.get('market_type', 'unknown')}/{info.get('strategy_type', 'unknown')})")
        logger.info(f"    Description: {info.get('description', 'No description')[:100]}...")
    
    return strategy_count > 0

if __name__ == "__main__":
    logger.info("Performing simple registration check...")
    
    # Check registry module
    registry = check_registry_module()
    
    # Check strategy modules
    modules_ok = check_strategy_modules()
    
    # Check registration
    if registry:
        registration_ok = check_strategy_registration(registry)
        if registration_ok:
            logger.info("✓ Strategy registration check passed!")
        else:
            logger.error("✗ Strategy registration check failed: No strategies registered")
    else:
        logger.error("✗ Cannot check registration without registry module")
    
    # Overall status
    if registry and modules_ok and registration_ok:
        logger.info("All checks passed successfully!")
    else:
        logger.error("Some checks failed. See log for details.")
