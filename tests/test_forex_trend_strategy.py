#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Forex Trend-Following Strategy.
This verifies that the strategy generates signals and integrates with the event system.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.strategies.forex.trend_following_strategy import ForexTrendFollowingStrategy
from trading_bot.strategies.strategy_template import SignalType, MarketRegime
from trading_bot.event_system import EventBus, EventType
from trading_bot.event_system.event_types import Event
from trading_bot.event_system.event_bus import EventHandler

# Create test forex data
def create_test_data(symbol, n_days=100):
    """Create sample forex data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days)
    
    # Create a dataframe with forex OHLCV data
    if 'uptrend' in symbol.lower():
        # Generate an uptrend
        close = np.linspace(1.0, 1.5, n_days) + np.random.normal(0, 0.01, n_days)
    elif 'downtrend' in symbol.lower():
        # Generate a downtrend
        close = np.linspace(1.5, 1.0, n_days) + np.random.normal(0, 0.01, n_days)
    else:
        # Generate a sideways market with some noise
        close = np.ones(n_days) * 1.2 + np.random.normal(0, 0.02, n_days)
    
    # Create OHLCV data
    high = close + np.random.uniform(0.005, 0.02, n_days)
    low = close - np.random.uniform(0.005, 0.02, n_days)
    open_price = close.copy()
    open_price[1:] = close[:-1]  # Shift close prices by 1 to get open
    volume = np.random.uniform(1000, 5000, n_days)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

# Event listener for testing
class TestEventListener:
    def __init__(self):
        self.events = []
        self.event_bus = EventBus()
        
        # Register handler for strategy update events using the correct method
        handler = EventHandler(
            callback=self.on_strategy_update,
            event_type=EventType.SIGNAL_GENERATED,  # Using SIGNAL_GENERATED as a suitable event type
            name="StrategySignalListener"
        )
        self.event_bus.register_handler(handler)
        
        # Start event processing
        self.event_bus.start()
        
    def on_strategy_update(self, event: Event):
        """Handle strategy update events."""
        event_data = event.data
        self.events.append(event_data)
        logger.info(f"Received strategy event: {event_data['strategy_name']} with {event_data['trend_count']} trends")

def test_forex_trend_strategy():
    """Test the Forex Trend-Following Strategy."""
    logger.info("Starting Forex Trend-Following Strategy test")
    
    # Create the strategy
    strategy = ForexTrendFollowingStrategy()
    
    # Register event listener
    event_listener = TestEventListener()
    
    # Create test data for different market conditions
    test_data = {
        'EURUSD_uptrend': create_test_data('EURUSD_uptrend'),
        'GBPUSD_downtrend': create_test_data('GBPUSD_downtrend'),
        'USDJPY_sideways': create_test_data('USDJPY_sideways')
    }
    
    # Generate signals
    current_time = datetime.now()
    signals = strategy.generate_signals(test_data, current_time)
    
    # Display results
    logger.info(f"Generated {len(signals)} trading signals:")
    for symbol, signal in signals.items():
        logger.info(f"  {symbol}: {signal.signal_type.name} with confidence {signal.confidence:.2f}")
    
    # Check regime compatibility
    regimes = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.CONSOLIDATION,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY
    ]
    
    logger.info("\nMarket regime compatibility scores:")
    for regime in regimes:
        score = strategy.get_compatibility_score(regime)
        logger.info(f"  {regime.name}: {score:.2f}")
        
        # Test parameter optimization
        optimized_params = strategy.optimize_for_regime(regime)
        logger.info(f"  Optimized for {regime.name}: MA periods {optimized_params['fast_ma_period']}/{optimized_params['slow_ma_period']}, " 
                   f"ADX threshold {optimized_params['adx_threshold']}")
    
    # Check if events were emitted correctly
    logger.info(f"\nEvent system received {len(event_listener.events)} events")
    
    logger.info("Test completed!")
    
if __name__ == "__main__":
    test_forex_trend_strategy()
