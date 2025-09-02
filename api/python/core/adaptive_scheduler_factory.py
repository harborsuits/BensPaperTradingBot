#!/usr/bin/env python3
"""
Adaptive Scheduler Factory and Integration Utilities

This module provides factory functions and integration utilities for the
AdaptiveScheduler, making it easy to set up optimized data fetching
and event processing for various asset classes and market conditions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable

from trading_bot.core.adaptive_scheduler import (
    AdaptiveScheduler, 
    SymbolTier, 
    APIProvider, 
    DataType, 
    MarketHours
)
from trading_bot.core.constants import EventType, AssetType
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

logger = logging.getLogger(__name__)

def create_scheduler(
    broker_manager: MultiBrokerManager,
    strategy_manager: Any = None,
    api_config: Optional[Dict[str, Any]] = None
) -> AdaptiveScheduler:
    """
    Create and configure an adaptive scheduler with default settings.
    
    Args:
        broker_manager: The multi-broker manager instance
        strategy_manager: The strategy manager instance
        api_config: Optional API configuration overrides
        
    Returns:
        A configured AdaptiveScheduler instance
    """
    scheduler = AdaptiveScheduler(
        strategy_manager=strategy_manager,
        broker_manager=broker_manager
    )
    
    # Override API limits if provided
    if api_config:
        _configure_api_limits(scheduler, api_config)
    
    # Register default tasks
    _register_default_tasks(scheduler, broker_manager)
    
    return scheduler

def _configure_api_limits(scheduler: AdaptiveScheduler, api_config: Dict[str, Any]) -> None:
    """Configure API rate limits based on provided configuration."""
    if 'alpaca' in api_config:
        alpaca_config = api_config['alpaca']
        if 'calls_per_second' in alpaca_config:
            scheduler.api_limits[APIProvider.ALPACA].calls_per_second = alpaca_config['calls_per_second']
        if 'calls_per_minute' in alpaca_config:
            scheduler.api_limits[APIProvider.ALPACA].calls_per_minute = alpaca_config['calls_per_minute']
            
    if 'tradier' in api_config:
        tradier_config = api_config['tradier']
        if 'calls_per_second' in tradier_config:
            scheduler.api_limits[APIProvider.TRADIER].calls_per_second = tradier_config['calls_per_second']
        if 'calls_per_minute' in tradier_config:
            scheduler.api_limits[APIProvider.TRADIER].calls_per_minute = tradier_config['calls_per_minute']
    
    if 'marketaux' in api_config:
        marketaux_config = api_config['marketaux']
        if 'calls_per_day' in marketaux_config:
            scheduler.api_limits[APIProvider.MARKETAUX].calls_per_day = marketaux_config['calls_per_day']
    
    # Add other providers as needed

def _register_default_tasks(
    scheduler: AdaptiveScheduler, 
    broker_manager: MultiBrokerManager
) -> None:
    """Register default scheduled tasks for data fetching and processing."""
    
    # Task to fetch price data for active symbols
    scheduler.add_scheduled_task(
        name="fetch_active_prices",
        callback=_create_price_fetch_callback(broker_manager),
        interval_seconds={
            MarketHours.MARKET_HOURS: 5.0,    # Every 5 seconds during market hours
            MarketHours.PRE_MARKET: 15.0,     # Every 15 seconds pre-market
            MarketHours.POST_MARKET: 30.0,    # Every 30 seconds post-market
            MarketHours.OVERNIGHT: 300.0,     # Every 5 minutes overnight
            MarketHours.WEEKEND: 1800.0,      # Every 30 minutes on weekends
        },
        data_type=DataType.PRICE,
        provider=APIProvider.ALPACA
    )
    
    # Task to fetch quotes for active symbols
    scheduler.add_scheduled_task(
        name="fetch_active_quotes",
        callback=_create_quote_fetch_callback(broker_manager),
        interval_seconds={
            MarketHours.MARKET_HOURS: 10.0,    # Every 10 seconds during market hours
            MarketHours.PRE_MARKET: 30.0,      # Every 30 seconds pre-market
            MarketHours.POST_MARKET: 60.0,     # Every minute post-market
            MarketHours.OVERNIGHT: 600.0,      # Every 10 minutes overnight
            MarketHours.WEEKEND: 3600.0,       # Every hour on weekends
        },
        data_type=DataType.QUOTE,
        provider=APIProvider.TRADIER
    )
    
    # Task to fetch news for active symbols
    scheduler.add_scheduled_task(
        name="fetch_news",
        callback=_create_news_fetch_callback(broker_manager),
        interval_seconds={
            MarketHours.MARKET_HOURS: 300.0,    # Every 5 minutes during market hours
            MarketHours.PRE_MARKET: 300.0,      # Every 5 minutes pre-market
            MarketHours.POST_MARKET: 600.0,     # Every 10 minutes post-market
            MarketHours.OVERNIGHT: 1800.0,      # Every 30 minutes overnight
            MarketHours.WEEKEND: 3600.0,        # Every hour on weekends
        },
        data_type=DataType.NEWS,
        provider=APIProvider.MARKETAUX
    )
    
    # Add more default tasks as needed
    
def _create_price_fetch_callback(broker_manager: MultiBrokerManager) -> Callable:
    """Create a callback for fetching price data."""
    def fetch_price_data():
        try:
            # Get the next batch of symbols to fetch from rotator
            symbols = scheduler.symbol_rotator.get_next_batch()
            if not symbols:
                return
                
            # Batch symbols by broker for efficient fetching
            symbols_by_broker = _group_symbols_by_broker(broker_manager, symbols)
            
            # Fetch prices from each broker
            for broker_id, broker_symbols in symbols_by_broker.items():
                if not broker_symbols:
                    continue
                    
                try:
                    broker = broker_manager.get_broker(broker_id)
                    if broker and broker.is_connected():
                        # Batch fetch prices - implementation depends on broker interface
                        prices = broker.get_quotes(broker_symbols)
                        
                        # Process results (implementation will vary)
                        if prices:
                            logger.debug(f"Fetched prices for {len(prices)} symbols from {broker_id}")
                except Exception as e:
                    logger.error(f"Error fetching prices from {broker_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in price fetch task: {str(e)}")
    
    return fetch_price_data

def _create_quote_fetch_callback(broker_manager: MultiBrokerManager) -> Callable:
    """Create a callback for fetching quote data."""
    def fetch_quote_data():
        try:
            # Similar to price fetch, but for detailed quotes
            # Implementation details will depend on your broker interface
            pass
        except Exception as e:
            logger.error(f"Error in quote fetch task: {str(e)}")
    
    return fetch_quote_data

def _create_news_fetch_callback(broker_manager: MultiBrokerManager) -> Callable:
    """Create a callback for fetching news data."""
    def fetch_news_data():
        try:
            # Get active symbols, prioritize Tier 1
            symbols = scheduler.symbol_rotator.get_next_batch(SymbolTier.TIER_1)
            if not symbols:
                return
                
            # Implementation will depend on your news API integration
            # This is a placeholder for the actual implementation
            pass
        except Exception as e:
            logger.error(f"Error in news fetch task: {str(e)}")
    
    return fetch_news_data

def _group_symbols_by_broker(
    broker_manager: MultiBrokerManager, 
    symbols: List[str]
) -> Dict[str, List[str]]:
    """
    Group symbols by the broker that should handle them.
    
    Args:
        broker_manager: The multi-broker manager instance
        symbols: List of symbol strings
        
    Returns:
        Dict mapping broker IDs to lists of symbols
    """
    symbols_by_broker = {}
    
    for symbol in symbols:
        # Determine asset class (simplified example)
        asset_class = _infer_asset_class(symbol)
        
        # Get appropriate broker for this asset class
        broker_id = broker_manager.get_preferred_broker_for_asset(asset_class)
        
        if broker_id not in symbols_by_broker:
            symbols_by_broker[broker_id] = []
            
        symbols_by_broker[broker_id].append(symbol)
    
    return symbols_by_broker

def _infer_asset_class(symbol: str) -> AssetType:
    """
    Infer the asset class from the symbol format.
    This is a simple heuristic and may need to be customized.
    
    Args:
        symbol: The symbol string
        
    Returns:
        The inferred AssetType
    """
    if '/' in symbol:
        return AssetType.FOREX
    elif symbol.endswith('-USD') or symbol.endswith('USDT'):
        return AssetType.CRYPTO
    elif len(symbol) <= 5 and symbol.isupper():
        return AssetType.STOCK
    else:
        return AssetType.STOCK  # Default to stock
