#!/usr/bin/env python3
"""
Example of how to use the Adaptive Scheduler with your trading bot.

This example demonstrates how to:
1. Initialize the scheduler
2. Configure API limits
3. Add symbols with appropriate tiers
4. Register custom tasks and event handlers
5. Start and stop the scheduler
"""

import argparse
import logging
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.core.adaptive_scheduler import (
    AdaptiveScheduler, SymbolTier, APIProvider, DataType, MarketHours
)
from trading_bot.core.adaptive_scheduler_factory import create_scheduler
from trading_bot.core.constants import EventType, AssetType
from trading_bot.brokers.broker_factory import load_from_config_file
from trading_bot.core.enhanced_strategy_manager_impl import EnhancedStrategyManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_config(config_file: str) -> Dict:
    """Load API configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading API config: {str(e)}")
        return {}

def load_watchlist(watchlist_file: str) -> Dict[str, List[str]]:
    """Load watchlists from a JSON file."""
    try:
        with open(watchlist_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading watchlist: {str(e)}")
        return {}

def setup_scheduler(broker_config: str, api_config: str, watchlist_file: str) -> AdaptiveScheduler:
    """Set up and configure the adaptive scheduler."""
    
    # Load broker manager
    broker_manager = load_from_config_file(broker_config)
    
    # Load strategy manager (replace with your actual initialization)
    strategy_manager = EnhancedStrategyManager(broker_manager=broker_manager)
    
    # Load API configuration
    api_limits = load_api_config(api_config)
    
    # Create scheduler
    scheduler = create_scheduler(
        broker_manager=broker_manager,
        strategy_manager=strategy_manager,
        api_config=api_limits
    )
    
    # Load watchlists and add symbols
    watchlists = load_watchlist(watchlist_file)
    _add_symbols_from_watchlists(scheduler, watchlists)
    
    # Register custom event handlers
    _register_custom_handlers(scheduler)
    
    return scheduler

def _add_symbols_from_watchlists(scheduler: AdaptiveScheduler, watchlists: Dict[str, List[str]]) -> None:
    """Add symbols from watchlists with appropriate tiers."""
    
    # Top priority symbols (active trades, major indices)
    if 'tier_1' in watchlists:
        for symbol in watchlists['tier_1']:
            scheduler.add_symbol(symbol, SymbolTier.TIER_1)
            logger.info(f"Added Tier 1 symbol: {symbol}")
    
    # Medium priority symbols (watchlist)
    if 'tier_2' in watchlists:
        for symbol in watchlists['tier_2']:
            scheduler.add_symbol(symbol, SymbolTier.TIER_2)
            logger.info(f"Added Tier 2 symbol: {symbol}")
    
    # Low priority symbols (background monitoring)
    if 'tier_3' in watchlists:
        for symbol in watchlists['tier_3']:
            scheduler.add_symbol(symbol, SymbolTier.TIER_3)
            logger.info(f"Added Tier 3 symbol: {symbol}")

def _register_custom_handlers(scheduler: AdaptiveScheduler) -> None:
    """Register custom event handlers for various events."""
    
    # Example: handler for high volume alerts
    scheduler.register_event_handler(
        EventType.VOLUME_ALERT,
        lambda data: _handle_volume_alert(scheduler, data)
    )
    
    # Example: handler for significant price moves
    scheduler.register_event_handler(
        EventType.PRICE_ALERT,
        lambda data: _handle_price_alert(scheduler, data)
    )

def _handle_volume_alert(scheduler: AdaptiveScheduler, data: Dict[str, Any]) -> None:
    """Handler for volume alerts - promotes symbol to higher tier."""
    symbol = data.get('symbol')
    if symbol:
        scheduler.symbol_rotator.promote_symbol(symbol)
        logger.info(f"Promoted {symbol} due to high volume alert")

def _handle_price_alert(scheduler: AdaptiveScheduler, data: Dict[str, Any]) -> None:
    """Handler for price alerts - promotes symbol to higher tier."""
    symbol = data.get('symbol')
    if symbol:
        scheduler.symbol_rotator.promote_symbol(symbol)
        logger.info(f"Promoted {symbol} due to price alert")

def custom_news_task(scheduler: AdaptiveScheduler) -> None:
    """Example custom task for fetching and processing news."""
    # Get high-priority symbols
    symbols = scheduler.symbol_rotator.get_next_batch(SymbolTier.TIER_1)
    
    if symbols:
        logger.info(f"Fetching news for {len(symbols)} high-priority symbols")
        # Actual implementation would call your news API

def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description='Adaptive Scheduler Example')
    parser.add_argument('--broker-config', type=str, required=True, help='Path to broker configuration')
    parser.add_argument('--api-config', type=str, required=True, help='Path to API configuration')
    parser.add_argument('--watchlist', type=str, required=True, help='Path to watchlist file')
    parser.add_argument('--duration', type=int, default=60, help='How long to run (seconds)')
    
    args = parser.parse_args()
    
    # Set up scheduler
    scheduler = setup_scheduler(
        broker_config=args.broker_config,
        api_config=args.api_config,
        watchlist_file=args.watchlist
    )
    
    # Add a custom task
    scheduler.add_scheduled_task(
        name="custom_news_task",
        callback=lambda: custom_news_task(scheduler),
        interval_seconds={
            MarketHours.MARKET_HOURS: 120.0,  # Every 2 minutes during market hours
            MarketHours.PRE_MARKET: 300.0,    # Every 5 minutes pre-market
            MarketHours.POST_MARKET: 600.0,   # Every 10 minutes post-market
            MarketHours.OVERNIGHT: 1800.0,    # Every 30 minutes overnight
            MarketHours.WEEKEND: 3600.0,      # Every hour on weekends
        },
        data_type=DataType.NEWS,
        provider=APIProvider.MARKETAUX
    )
    
    try:
        # Start the scheduler
        scheduler.start()
        logger.info(f"Scheduler started, running for {args.duration} seconds")
        
        # Run for the specified duration
        for _ in range(args.duration):
            # Get and print status every second
            if _ % 10 == 0:  # Print every 10 seconds
                status = scheduler.get_status()
                logger.info(f"Scheduler status: {status}")
            time.sleep(1)
    
    finally:
        # Always stop the scheduler
        scheduler.stop()
        logger.info("Scheduler stopped")

if __name__ == "__main__":
    main()
