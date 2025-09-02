#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Forex Strategies

This module contains comprehensive unit tests for all forex strategies:
- Forex Trend-Following Strategy
- Forex Range Trading Strategy
- Forex Breakout Strategy

It tests signal generation, market regime compatibility, and event emission.
"""

import unittest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy classes
from trading_bot.strategies.forex.trend_following_strategy import ForexTrendFollowingStrategy
from trading_bot.strategies.forex.range_trading_strategy import ForexRangeTradingStrategy
from trading_bot.strategies.forex.breakout_strategy import ForexBreakoutStrategy
from trading_bot.strategies.strategy_template import Signal, SignalType, MarketRegime
from trading_bot.event_system.event_types import EventType, Event

# Mock EventBus for testing
class MockEventBus:
    def __init__(self):
        self.published_events = []
        
    def publish(self, event: Event):
        self.published_events.append(event)
        
    def get_event_count(self):
        return len(self.published_events)
    
    def get_published_events(self):
        return self.published_events

# Test data generator
def create_test_data(scenario: str, n_days=100):
    """Create synthetic forex data for testing different market scenarios."""
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
        
    elif scenario == 'breakout':
        # Generate a consolidation followed by a breakout
        consolidation_days = int(n_days * 0.7)  # 70% consolidation, 30% breakout
        breakout_days = n_days - consolidation_days
        
        # Consolidation period (small range with noise)
        consolidation = base_price + np.random.normal(0, 0.002, consolidation_days)
        
        # Breakout period (strong directional move)
        breakout = np.linspace(base_price, base_price * 1.08, breakout_days)
        
        # Combine both periods
        close = np.concatenate([consolidation, breakout])
        
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

class TestForexTrendFollowingStrategy(unittest.TestCase):
    """Test cases for the Forex Trend-Following Strategy."""
    
    def setUp(self):
        """Set up the test environment before each test method."""
        self.mock_event_bus = MockEventBus()
        
        # Create strategy with test parameters
        self.strategy = ForexTrendFollowingStrategy(
            name="Test Trend Strategy",
            parameters={
                'fast_ma_period': 10,
                'slow_ma_period': 30,
                'adx_period': 14,
                'adx_threshold': 20,
                'pip_value': 0.0001
            }
        )
        self.strategy.event_bus = self.mock_event_bus
        
        # Create test data for different market scenarios
        self.test_data = {
            'EURUSD': create_test_data('uptrend'),
            'GBPUSD': create_test_data('downtrend'),
            'USDJPY': create_test_data('range_bound'),
            'AUDUSD': create_test_data('volatile')
        }
        
        self.current_time = datetime.now()
    
    def test_signal_generation(self):
        """Test that the strategy generates appropriate signals for trending markets."""
        signals = self.strategy.generate_signals(self.test_data, self.current_time)
        
        # We should get signals for trending pairs (EURUSD uptrend, GBPUSD downtrend)
        self.assertIn('EURUSD', signals, "Should generate signal for uptrending pair")
        self.assertIn('GBPUSD', signals, "Should generate signal for downtrending pair")
        
        # Check signal types
        if 'EURUSD' in signals and signals['EURUSD'] is not None:
            self.assertEqual(signals['EURUSD'].signal_type, SignalType.LONG, 
                            "Should generate LONG signal for uptrend")
        
        if 'GBPUSD' in signals and signals['GBPUSD'] is not None:
            self.assertEqual(signals['GBPUSD'].signal_type, SignalType.SHORT, 
                            "Should generate SHORT signal for downtrend")
    
    def test_market_regime_compatibility(self):
        """Test that the strategy has appropriate compatibility scores for different regimes."""
        # Trend-following should work best in trending markets
        bull_score = self.strategy.get_compatibility_score(MarketRegime.BULL_TREND)
        bear_score = self.strategy.get_compatibility_score(MarketRegime.BEAR_TREND)
        consolidation_score = self.strategy.get_compatibility_score(MarketRegime.CONSOLIDATION)
        
        self.assertGreater(bull_score, 0.7, "Should have high compatibility with bull trends")
        self.assertGreater(bear_score, 0.7, "Should have high compatibility with bear trends")
        self.assertLess(consolidation_score, bull_score, 
                      "Should have lower compatibility with consolidation than trends")
    
    def test_event_emission(self):
        """Test that the strategy emits events when signals are generated."""
        # Generate signals to trigger events
        self.strategy.generate_signals(self.test_data, self.current_time)
        
        # Check if events were published
        self.assertGreater(self.mock_event_bus.get_event_count(), 0, 
                          "Should emit events for signal generation")
        
        # Verify event properties
        if self.mock_event_bus.get_event_count() > 0:
            event = self.mock_event_bus.published_events[0]
            self.assertEqual(event.event_type, EventType.SIGNAL_GENERATED, 
                           "Should emit SIGNAL_GENERATED event")
            self.assertEqual(event.source, "Test Trend Strategy", 
                           "Event source should match strategy name")
            self.assertIn('strategy_name', event.data, 
                         "Event data should include strategy name")
    
    def test_regime_optimization(self):
        """Test that the strategy optimizes parameters for different market regimes."""
        # Get optimized parameters for different regimes
        bull_params = self.strategy.optimize_for_regime(MarketRegime.BULL_TREND)
        bear_params = self.strategy.optimize_for_regime(MarketRegime.BEAR_TREND)
        volatile_params = self.strategy.optimize_for_regime(MarketRegime.HIGH_VOLATILITY)
        
        # Parameters should be different for different regimes
        self.assertNotEqual(bull_params['adx_threshold'], volatile_params['adx_threshold'],
                           "ADX threshold should adapt to market regime")

class TestForexRangeStrategy(unittest.TestCase):
    """Test cases for the Forex Range Trading Strategy."""
    
    def setUp(self):
        """Set up the test environment before each test method."""
        self.mock_event_bus = MockEventBus()
        
        # Create strategy with test parameters
        self.strategy = ForexRangeTradingStrategy(
            name="Test Range Strategy",
            parameters={
                'range_threshold': 0.02,
                'min_touches': 2,
                'bb_period': 20,
                'rsi_period': 14,
                'pip_value': 0.0001
            }
        )
        self.strategy.event_bus = self.mock_event_bus
        
        # Create test data for different market scenarios
        self.test_data = {
            'EURUSD': create_test_data('uptrend'),
            'GBPUSD': create_test_data('downtrend'),
            'USDJPY': create_test_data('range_bound'),
            'AUDUSD': create_test_data('volatile')
        }
        
        self.current_time = datetime.now()
    
    def test_signal_generation(self):
        """Test that the strategy generates appropriate signals for range-bound markets."""
        signals = self.strategy.generate_signals(self.test_data, self.current_time)
        
        # We should get signals for range-bound pairs (USDJPY)
        self.assertIn('USDJPY', signals, "Should generate signal for range-bound pair")
        
        # We should get fewer or no signals for trending pairs
        if 'EURUSD' in signals and signals['EURUSD'] is not None:
            self.assertLess(signals['EURUSD'].confidence, 0.8, 
                           "Should generate lower confidence signals for trending markets")
    
    def test_market_regime_compatibility(self):
        """Test that the strategy has appropriate compatibility scores for different regimes."""
        # Range trading should work best in consolidation
        consolidation_score = self.strategy.get_compatibility_score(MarketRegime.CONSOLIDATION)
        low_vol_score = self.strategy.get_compatibility_score(MarketRegime.LOW_VOLATILITY)
        bull_score = self.strategy.get_compatibility_score(MarketRegime.BULL_TREND)
        
        self.assertGreater(consolidation_score, 0.7, 
                          "Should have high compatibility with consolidation")
        self.assertGreater(low_vol_score, 0.7, 
                          "Should have high compatibility with low volatility")
        self.assertLess(bull_score, consolidation_score, 
                       "Should have lower compatibility with trends than consolidation")
    
    def test_event_emission(self):
        """Test that the strategy emits events when signals are generated."""
        # Generate signals to trigger events
        self.strategy.generate_signals(self.test_data, self.current_time)
        
        # Check if events were published
        self.assertGreater(self.mock_event_bus.get_event_count(), 0, 
                          "Should emit events for signal generation")
        
        # Verify event properties
        if self.mock_event_bus.get_event_count() > 0:
            event = self.mock_event_bus.published_events[0]
            self.assertEqual(event.event_type, EventType.SIGNAL_GENERATED, 
                           "Should emit SIGNAL_GENERATED event")
            self.assertEqual(event.source, "Test Range Strategy", 
                           "Event source should match strategy name")
    
    def test_regime_optimization(self):
        """Test that the strategy optimizes parameters for different market regimes."""
        # Get optimized parameters for different regimes
        consolidation_params = self.strategy.optimize_for_regime(MarketRegime.CONSOLIDATION)
        low_vol_params = self.strategy.optimize_for_regime(MarketRegime.LOW_VOLATILITY)
        high_vol_params = self.strategy.optimize_for_regime(MarketRegime.HIGH_VOLATILITY)
        
        # Parameters should be different for different regimes
        self.assertNotEqual(consolidation_params['range_threshold'], high_vol_params['range_threshold'],
                           "Range threshold should adapt to market regime")

class TestForexBreakoutStrategy(unittest.TestCase):
    """Test cases for the Forex Breakout Strategy."""
    
    def setUp(self):
        """Set up the test environment before each test method."""
        self.mock_event_bus = MockEventBus()
        
        # Create strategy with test parameters
        self.strategy = ForexBreakoutStrategy(
            name="Test Breakout Strategy",
            parameters={
                'breakout_threshold': 0.01,
                'donchian_period': 20,
                'atr_period': 14,
                'pip_value': 0.0001
            }
        )
        self.strategy.event_bus = self.mock_event_bus
        
        # Create test data for different market scenarios
        self.test_data = {
            'EURUSD': create_test_data('uptrend'),
            'GBPUSD': create_test_data('downtrend'),
            'USDJPY': create_test_data('range_bound'),
            'AUDUSD': create_test_data('breakout')  # Special breakout scenario
        }
        
        self.current_time = datetime.now()
    
    def test_signal_generation(self):
        """Test that the strategy generates appropriate signals for breakout scenarios."""
        signals = self.strategy.generate_signals(self.test_data, self.current_time)
        
        # We should get signals for breakout pairs (AUDUSD)
        self.assertIn('AUDUSD', signals, "Should generate signal for breakout pattern")
        
        if 'AUDUSD' in signals and signals['AUDUSD'] is not None:
            self.assertEqual(signals['AUDUSD'].signal_type, SignalType.LONG, 
                            "Should generate LONG signal for upward breakout")
    
    def test_market_regime_compatibility(self):
        """Test that the strategy has appropriate compatibility scores for different regimes."""
        # Breakout strategies should work best in high volatility
        high_vol_score = self.strategy.get_compatibility_score(MarketRegime.HIGH_VOLATILITY)
        bull_score = self.strategy.get_compatibility_score(MarketRegime.BULL_TREND)
        consolidation_score = self.strategy.get_compatibility_score(MarketRegime.CONSOLIDATION)
        
        self.assertGreater(high_vol_score, 0.8, 
                          "Should have high compatibility with high volatility")
        self.assertGreater(bull_score, 0.5, 
                          "Should have moderate compatibility with trends")
        self.assertLess(consolidation_score, high_vol_score, 
                       "Should have lower compatibility with consolidation than high volatility")
    
    def test_event_emission(self):
        """Test that the strategy emits events when signals are generated."""
        # Generate signals to trigger events
        self.strategy.generate_signals(self.test_data, self.current_time)
        
        # Check if events were published
        self.assertGreater(self.mock_event_bus.get_event_count(), 0, 
                          "Should emit events for signal generation")
        
        # Verify event properties
        if self.mock_event_bus.get_event_count() > 0:
            event = self.mock_event_bus.published_events[0]
            self.assertEqual(event.event_type, EventType.SIGNAL_GENERATED, 
                           "Should emit SIGNAL_GENERATED event")
            self.assertEqual(event.source, "Test Breakout Strategy", 
                           "Event source should match strategy name")
    
    def test_regime_optimization(self):
        """Test that the strategy optimizes parameters for different market regimes."""
        # Get optimized parameters for different regimes
        high_vol_params = self.strategy.optimize_for_regime(MarketRegime.HIGH_VOLATILITY)
        bull_params = self.strategy.optimize_for_regime(MarketRegime.BULL_TREND)
        consolidation_params = self.strategy.optimize_for_regime(MarketRegime.CONSOLIDATION)
        
        # Parameters should be different for different regimes
        self.assertNotEqual(high_vol_params['breakout_threshold'], consolidation_params['breakout_threshold'],
                           "Breakout threshold should adapt to market regime")
        self.assertNotEqual(high_vol_params['confirmation_candles'], bull_params['confirmation_candles'],
                           "Confirmation requirements should adapt to market regime")

class TestCombinedStrategies(unittest.TestCase):
    """Test cases for combining multiple forex strategies."""
    
    def setUp(self):
        """Set up the test environment before each test method."""
        self.mock_event_bus = MockEventBus()
        
        # Create all strategies with the same event bus
        self.trend_strategy = ForexTrendFollowingStrategy(name="Trend Strategy")
        self.range_strategy = ForexRangeTradingStrategy(name="Range Strategy")
        self.breakout_strategy = ForexBreakoutStrategy(name="Breakout Strategy")
        
        self.trend_strategy.event_bus = self.mock_event_bus
        self.range_strategy.event_bus = self.mock_event_bus
        self.breakout_strategy.event_bus = self.mock_event_bus
        
        # Create diverse test data
        self.test_data = {
            'EURUSD': create_test_data('uptrend'),
            'GBPUSD': create_test_data('downtrend'),
            'USDJPY': create_test_data('range_bound'),
            'AUDUSD': create_test_data('breakout'),
            'USDCAD': create_test_data('volatile')
        }
        
        self.current_time = datetime.now()
    
    def test_complementary_signals(self):
        """Test that strategies generate complementary signals for different market conditions."""
        # Generate signals from all strategies
        trend_signals = self.trend_strategy.generate_signals(self.test_data, self.current_time)
        range_signals = self.range_strategy.generate_signals(self.test_data, self.current_time)
        breakout_signals = self.breakout_strategy.generate_signals(self.test_data, self.current_time)
        
        # Each strategy should generate signals for its optimal market conditions
        trend_signal_pairs = set([k for k in trend_signals if trend_signals[k] is not None])
        range_signal_pairs = set([k for k in range_signals if range_signals[k] is not None])
        breakout_signal_pairs = set([k for k in breakout_signals if breakout_signals[k] is not None])
        
        # We should see different strategies favoring different pairs
        self.assertNotEqual(trend_signal_pairs, range_signal_pairs, 
                           "Trend and range strategies should favor different pairs")
    
    def test_event_integration(self):
        """Test that all strategies publish events to the same event bus."""
        # Generate signals from all strategies
        self.trend_strategy.generate_signals(self.test_data, self.current_time)
        self.range_strategy.generate_signals(self.test_data, self.current_time)
        self.breakout_strategy.generate_signals(self.test_data, self.current_time)
        
        # Count events by strategy source
        event_sources = [event.source for event in self.mock_event_bus.published_events]
        trend_events = event_sources.count("Trend Strategy")
        range_events = event_sources.count("Range Strategy")
        breakout_events = event_sources.count("Breakout Strategy")
        
        # All strategies should have published events
        self.assertGreater(trend_events, 0, "Trend strategy should emit events")
        self.assertGreater(range_events, 0, "Range strategy should emit events")
        self.assertGreater(breakout_events, 0, "Breakout strategy should emit events")
    
    def test_regime_compatibility_distribution(self):
        """Test that strategies have different optimal market regimes."""
        regimes = [
            MarketRegime.BULL_TREND,
            MarketRegime.BEAR_TREND,
            MarketRegime.CONSOLIDATION,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY
        ]
        
        # Find best regime for each strategy
        trend_best = max(regimes, key=lambda r: self.trend_strategy.get_compatibility_score(r))
        range_best = max(regimes, key=lambda r: self.range_strategy.get_compatibility_score(r))
        breakout_best = max(regimes, key=lambda r: self.breakout_strategy.get_compatibility_score(r))
        
        # Strategies should have different optimal regimes
        self.assertNotEqual(trend_best, range_best, 
                           "Trend and range strategies should have different optimal regimes")
        
        # At least one strategy should prefer each major regime type
        trend_regimes = [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]
        sideways_regimes = [MarketRegime.CONSOLIDATION, MarketRegime.LOW_VOLATILITY]
        volatile_regimes = [MarketRegime.HIGH_VOLATILITY]
        
        self.assertTrue(
            trend_best in trend_regimes or 
            range_best in sideways_regimes or 
            breakout_best in volatile_regimes,
            "Strategies should cover different market regime types"
        )

if __name__ == "__main__":
    unittest.main()
