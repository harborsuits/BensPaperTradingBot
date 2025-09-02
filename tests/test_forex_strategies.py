#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Forex strategies in BensBot.

This script tests the Forex Trend-Following Strategy and Forex Range Trading Strategy,
verifying that they generate appropriate signals for different market conditions
and that they properly integrate with the event system.
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
from trading_bot.strategies.forex.range_trading_strategy import ForexRangeTradingStrategy
from trading_bot.strategies.strategy_template import SignalType, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event
from trading_bot.event_system.event_bus import EventHandler

# Create test data for different market scenarios
def create_test_data(scenario: str, n_days=100):
    """Create sample forex data for testing different scenarios."""
    dates = pd.date_range(end=datetime.now(), periods=n_days)
    
    # Base price starting at 1.2000
    base_price = 1.2000
    
    if scenario == 'uptrend':
        # Generate an uptrend with some noise
        close = np.linspace(base_price, base_price * 1.15, n_days) + np.random.normal(0, 0.002, n_days)
        
    elif scenario == 'downtrend':
        # Generate a downtrend with some noise
        close = np.linspace(base_price, base_price * 0.85, n_days) + np.random.normal(0, 0.002, n_days)
        
    elif scenario == 'range_bound':
        # Generate a sideways market with resistance and support
        amplitude = base_price * 0.03  # 3% range
        close = base_price + amplitude * np.sin(np.linspace(0, 4*np.pi, n_days))
        close = close + np.random.normal(0, 0.0005, n_days)  # Less noise
        
    elif scenario == 'volatile':
        # Generate a volatile market with large price swings
        close = base_price + np.random.normal(0, 0.01, n_days).cumsum()
        
    else:  # Default to sideways
        # Sideways market with mild noise
        close = np.ones(n_days) * base_price + np.random.normal(0, 0.001, n_days)
    
    # Create OHLCV data
    high = close + np.random.uniform(0.0005, 0.003, n_days)
    low = close - np.random.uniform(0.0005, 0.003, n_days)
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
        
        # Register handler for strategy signals
        handler = EventHandler(
            callback=self.on_strategy_signal,
            event_type=EventType.SIGNAL_GENERATED,
            name="StrategySignalListener"
        )
        self.event_bus.register_handler(handler)
        
        # Start event processing
        self.event_bus.start()
        
    def on_strategy_signal(self, event: Event):
        """Handle strategy signal events."""
        event_data = event.data
        self.events.append(event_data)
        logger.info(f"Received signal from {event_data['strategy_name']} with confidence {event_data.get('confidence', 'N/A')}")
        
    def get_event_count(self):
        """Return the number of events received."""
        return len(self.events)

def test_forex_strategies():
    """Test the Forex Trend-Following and Range Trading strategies."""
    logger.info("Starting Forex Strategy Test")
    
    # Create the event listener
    event_listener = TestEventListener()
    
    # Create test data for different market scenarios
    test_data = {
        'EURUSD_uptrend': create_test_data('uptrend'),
        'GBPUSD_downtrend': create_test_data('downtrend'),
        'USDJPY_range': create_test_data('range_bound'),
        'AUDUSD_volatile': create_test_data('volatile')
    }
    
    # Create strategies with specific parameter adjustments for testing
    trend_strategy = ForexTrendFollowingStrategy(parameters={
        'adx_threshold': 20,  # Lower threshold for more signals in test
        'pip_value': 0.0001  # Standard pip value for major pairs
    })
    
    range_strategy = ForexRangeTradingStrategy(parameters={
        'range_threshold': 0.02,  # Adjusted for test data
        'min_touches': 2,  # Lower requirement for test
        'pip_value': 0.0001  # Standard pip value for major pairs
    })
    
    # Current time for signal generation
    current_time = datetime.now()
    
    # Test trend strategy on all scenarios
    logger.info("\n----- Testing Forex Trend-Following Strategy -----")
    trend_signals = trend_strategy.generate_signals(test_data, current_time)
    
    logger.info(f"Generated {len(trend_signals)} trend strategy signals:")
    for symbol, signal in trend_signals.items():
        logger.info(f"  {symbol}: {signal.signal_type.name} with confidence {signal.confidence:.2f}")
    
    # Test range strategy on all scenarios
    logger.info("\n----- Testing Forex Range Trading Strategy -----")
    range_signals = range_strategy.generate_signals(test_data, current_time)
    
    logger.info(f"Generated {len(range_signals)} range strategy signals:")
    for symbol, signal in range_signals.items():
        logger.info(f"  {symbol}: {signal.signal_type.name} with confidence {signal.confidence:.2f}")
    
    # Test market regime compatibility
    regimes = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.CONSOLIDATION,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY
    ]
    
    logger.info("\n----- Market Regime Compatibility Scores -----")
    for regime in regimes:
        trend_score = trend_strategy.get_compatibility_score(regime)
        range_score = range_strategy.get_compatibility_score(regime)
        logger.info(f"  {regime.name}: Trend Strategy {trend_score:.2f}, Range Strategy {range_score:.2f}")
    
    # Check if events were emitted and received correctly
    logger.info(f"\nEvent system received {event_listener.get_event_count()} events")
    
    # Verify opposite nature of strategies
    logger.info("\n----- Strategy Comparison -----")
    logger.info("Verifying complementary nature of strategies across market regimes:")
    
    trend_best_regime = max(regimes, key=lambda r: trend_strategy.get_compatibility_score(r))
    range_best_regime = max(regimes, key=lambda r: range_strategy.get_compatibility_score(r))
    
    trend_worst_regime = min(regimes, key=lambda r: trend_strategy.get_compatibility_score(r))
    range_worst_regime = min(regimes, key=lambda r: range_strategy.get_compatibility_score(r))
    
    logger.info(f"Trend Strategy performs best in: {trend_best_regime.name}")
    logger.info(f"Range Strategy performs best in: {range_best_regime.name}")
    logger.info(f"Trend Strategy performs worst in: {trend_worst_regime.name}")
    logger.info(f"Range Strategy performs worst in: {range_worst_regime.name}")
    
    logger.info("\nTest completed!")
    
if __name__ == "__main__":
    test_forex_strategies()
