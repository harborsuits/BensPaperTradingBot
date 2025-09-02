#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Registration Test Script

This script verifies that all strategies are properly registered in the system
and can be discovered and instantiated through the factory mechanism.
"""

import sys
import os
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the necessary modules
from trading_bot.core.session import Session
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.strategy_factory import StrategyFactory
from trading_bot.strategies_new.factory.registry import get_registered_strategies, get_strategies_by_market_type

def test_strategy_registration():
    """Test that all strategies are properly registered."""
    # Create a factory instance
    factory = StrategyFactory()
    
    # Get all registered strategies
    all_strategies = get_registered_strategies()
    
    # Print total count
    logger.info(f"Total registered strategies: {len(all_strategies)}")
    
    # Get equity strategies
    equity_strategies = get_strategies_by_market_type("stocks")
    logger.info(f"Equity strategies: {len(equity_strategies)}")
    
    # Print all registered equity strategies
    logger.info("Registered equity strategies:")
    for name, info in equity_strategies.items():
        logger.info(f"  - {name} ({info.get('strategy_type', 'unknown')})")
        logger.info(f"    Description: {info.get('description', 'No description')}")
        logger.info(f"    Timeframes: {info.get('timeframes', [])}")
        logger.info(f"    Parameters: {len(info.get('parameters', {}))}")
        
    # Create a test session
    test_session = Session(symbol="AAPL", timeframe="1d")
    
    # Try to instantiate each equity strategy
    logger.info("Attempting to instantiate equity strategies:")
    for name in equity_strategies.keys():
        try:
            # Create a simple data pipeline
            data_pipeline = DataPipeline(symbol=test_session.symbol)
            
            # Create strategy
            strategy = factory.create_strategy(
                strategy_name=name,
                session=test_session,
                data_pipeline=data_pipeline
            )
            
            logger.info(f"  ✓ Successfully instantiated {name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to instantiate {name}: {str(e)}")
    
    return all_strategies

def test_strategy_discovery():
    """Test the strategy discovery mechanism."""
    # Create a factory instance (triggers discovery)
    factory = StrategyFactory()
    
    # Get available strategies by market type and timeframe
    logger.info("Getting available strategies for stocks/1d:")
    available = factory.get_available_strategies(
        market_type="stocks",
        timeframe="1d"
    )
    
    logger.info(f"Found {len(available)} available strategies for stocks/1d")
    
    # Print strategy names
    for name in available.keys():
        logger.info(f"  - {name}")
    
    return available

if __name__ == "__main__":
    logger.info("Testing strategy registration and discovery...")
    
    # Test registration
    all_strategies = test_strategy_registration()
    
    # Test discovery mechanism
    available_strategies = test_strategy_discovery()
    
    logger.info("Tests completed!")
