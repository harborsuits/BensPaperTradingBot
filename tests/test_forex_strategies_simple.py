#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Test script for Forex strategies

This script focuses only on testing the core functionality of the forex strategies
without relying on the full trading system infrastructure.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules directly - avoiding the full dependency tree
from trading_bot.strategies.strategy_template import Signal, SignalType, MarketRegime
from trading_bot.event_system.event_types import EventType, Event

# Direct imports of our strategies
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_bot/strategies/forex'))
from trend_following_strategy import ForexTrendFollowingStrategy
from range_trading_strategy import ForexRangeTradingStrategy

# Mock EventBus for testing
class MockEventBus:
    def __init__(self):
        self.published_events = []
        
    def publish(self, event: Event):
        logger.info(f"Event published: {event.event_type} from {event.source}")
        self.published_events.append(event)
        
    def get_event_count(self):
        return len(self.published_events)

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

def test_forex_strategies():
    """Test the Forex Trend-Following and Range Trading strategies."""
    logger.info("Starting Simplified Forex Strategy Test")
    
    # Create a mock event bus
    mock_event_bus = MockEventBus()
    
    # Create test data for different market scenarios
    test_data = {
        'EURUSD': create_test_data('uptrend'),
        'GBPUSD': create_test_data('downtrend'),
        'USDJPY': create_test_data('range_bound'),
        'AUDUSD': create_test_data('volatile')
    }
    
    # Create strategies with specific parameter adjustments for testing
    trend_strategy = ForexTrendFollowingStrategy(
        name="Forex Trend-Following Test Strategy",
        parameters={
            'adx_threshold': 20,         # Lower threshold for more signals in test
            'pip_value': 0.0001,         # Standard pip value for major pairs
            'fast_ma_type': 'EMA',
            'fast_ma_period': 10,
            'slow_ma_type': 'SMA',
            'slow_ma_period': 30,
            'adx_period': 14
        }
    )
    
    range_strategy = ForexRangeTradingStrategy(
        name="Forex Range Trading Test Strategy",
        parameters={
            'range_threshold': 0.02,     # Adjusted for test data
            'min_touches': 2,            # Lower requirement for test
            'pip_value': 0.0001,         # Standard pip value for major pairs
            'bb_period': 20,
            'rsi_period': 14
        }
    )
    
    # Assign the mock event bus to the strategies
    trend_strategy.event_bus = mock_event_bus
    range_strategy.event_bus = mock_event_bus
    
    # Current time for signal generation
    current_time = datetime.now()
    
    # Test trend strategy on all scenarios
    logger.info("\n----- Testing Forex Trend-Following Strategy -----")
    try:
        trend_signals = trend_strategy.generate_signals(test_data, current_time)
        
        logger.info(f"Generated {len(trend_signals)} trend strategy signals:")
        for symbol, signal in trend_signals.items():
            if signal:
                logger.info(f"  {symbol}: {signal.signal_type.name} with confidence {signal.confidence:.2f}")
            else:
                logger.info(f"  {symbol}: No signal generated")
    except Exception as e:
        logger.error(f"Error testing trend strategy: {str(e)}")
    
    # Test range strategy on all scenarios
    logger.info("\n----- Testing Forex Range Trading Strategy -----")
    try:
        range_signals = range_strategy.generate_signals(test_data, current_time)
        
        logger.info(f"Generated {len(range_signals)} range strategy signals:")
        for symbol, signal in range_signals.items():
            if signal:
                logger.info(f"  {symbol}: {signal.signal_type.name} with confidence {signal.confidence:.2f}")
            else:
                logger.info(f"  {symbol}: No signal generated")
    except Exception as e:
        logger.error(f"Error testing range strategy: {str(e)}")
    
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
    
    # Check if events were published
    logger.info(f"\nEvent system received {mock_event_bus.get_event_count()} events")
    
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
