#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the StrategyRotator class.
"""

import unittest
import tempfile
import os
import shutil
import numpy as np
from unittest import mock
from datetime import datetime

from trading_bot.strategy.base.strategy import Strategy
from trading_bot.strategy.rotator.strategy_rotator import StrategyRotator
from trading_bot.common.market_types import MarketRegime, MarketRegimeEvent

class TestStrategyRotator(unittest.TestCase):
    """Test case for the StrategyRotator class."""

    def setUp(self):
        """Set up test case."""
        # Create temp directory for data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock strategies
        self.strategy1 = mock.MagicMock(spec=Strategy)
        self.strategy1.name = "Strategy1"
        self.strategy1.enabled = True
        self.strategy1.generate_signal.return_value = 0.5
        self.strategy1.get_average_performance.return_value = 0.1
        self.strategy1.performance_history = []
        
        self.strategy2 = mock.MagicMock(spec=Strategy)
        self.strategy2.name = "Strategy2"
        self.strategy2.enabled = True
        self.strategy2.generate_signal.return_value = -0.3
        self.strategy2.get_average_performance.return_value = 0.05
        self.strategy2.performance_history = []
        
        # Create rotator with mock strategies
        self.rotator = StrategyRotator(
            strategies=[self.strategy1, self.strategy2],
            data_dir=self.temp_dir,
            performance_window=5
        )

    def tearDown(self):
        """Clean up after tests."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test rotator initialization."""
        self.assertEqual(len(self.rotator.strategies), 2)
        self.assertEqual(self.rotator.strategies[0].name, "Strategy1")
        self.assertEqual(self.rotator.strategies[1].name, "Strategy2")
        self.assertEqual(self.rotator.current_regime, MarketRegime.UNKNOWN)
        self.assertEqual(self.rotator.performance_window, 5)
        self.assertTrue(self.rotator.regime_adaptation)

    def test_add_strategy(self):
        """Test adding a strategy."""
        # Create new mock strategy
        strategy3 = mock.MagicMock(spec=Strategy)
        strategy3.name = "Strategy3"
        strategy3.enabled = True
        
        # Add strategy
        self.rotator.add_strategy(strategy3)
        
        # Check strategy was added
        self.assertEqual(len(self.rotator.strategies), 3)
        self.assertEqual(self.rotator.strategies[2].name, "Strategy3")
        self.assertIn("Strategy3", self.rotator.strategy_weights)
        
        # Check existing strategy is replaced
        strategy1_new = mock.MagicMock(spec=Strategy)
        strategy1_new.name = "Strategy1"
        self.rotator.add_strategy(strategy1_new)
        self.assertEqual(len(self.rotator.strategies), 3)  # Count should remain the same
        self.assertEqual(id(self.rotator.strategies_by_name["Strategy1"]), id(strategy1_new))

    def test_remove_strategy(self):
        """Test removing a strategy."""
        # Remove strategy
        result = self.rotator.remove_strategy("Strategy1")
        
        # Check strategy was removed
        self.assertTrue(result)
        self.assertEqual(len(self.rotator.strategies), 1)
        self.assertEqual(self.rotator.strategies[0].name, "Strategy2")
        self.assertNotIn("Strategy1", self.rotator.strategy_weights)
        
        # Check removing non-existent strategy
        result = self.rotator.remove_strategy("NonExistentStrategy")
        self.assertFalse(result)

    def test_update_market_regime(self):
        """Test updating market regime."""
        # Test with enum
        self.rotator.update_market_regime(MarketRegime.BULL, 0.8)
        self.assertEqual(self.rotator.current_regime, MarketRegime.BULL)
        
        # Test with string
        self.rotator.update_market_regime("BEAR", 0.7)
        self.assertEqual(self.rotator.current_regime, MarketRegime.BEAR)
        
        # Test invalid string
        with mock.patch("logging.Logger.error") as mock_error:
            self.rotator.update_market_regime("INVALID_REGIME", 0.5)
            mock_error.assert_called_once()
            self.assertEqual(self.rotator.current_regime, MarketRegime.BEAR)  # Should not change

    def test_handle_market_regime_event(self):
        """Test handling market regime event."""
        # Mock the MarketRegimeEvent class to avoid initialization issues
        with mock.patch('trading_bot.common.market_types.MarketRegimeEvent') as mock_event_class:
            # Create mock event
            event = mock.MagicMock()
            event.new_regime = MarketRegime.BULL
            event.confidence = 0.9
            mock_event_class.return_value = event
            
            # Handle event
            with mock.patch.object(self.rotator, 'update_market_regime') as mock_update:
                self.rotator.handle_market_regime_event(event)
                mock_update.assert_called_once_with(MarketRegime.BULL, 0.9)

    def test_adapt_weights_to_regime(self):
        """Test adapting weights to regime."""
        # Set initial weights
        self.rotator.strategy_weights = {"Strategy1": 0.5, "Strategy2": 0.5}
        
        # Update to BULL regime
        self.rotator.current_regime = MarketRegime.BULL
        self.rotator._adapt_weights_to_regime(confidence=1.0)
        
        # Check weights were updated
        # Bull weights from default config should be different from 0.5/0.5
        bull_weights = self.rotator.config["regime_weights"]["BULL"]
        self.assertAlmostEqual(
            self.rotator.strategy_weights["Strategy1"], 
            bull_weights.get("Strategy1", 0.5)
        )
        
        # Test with unknown regime
        with mock.patch("logging.Logger.warning") as mock_warning:
            # Create custom regime not in config
            self.rotator.current_regime = mock.MagicMock()
            self.rotator.current_regime.name = "CUSTOM_REGIME"
            self.rotator._adapt_weights_to_regime()
            mock_warning.assert_called_once()

    def test_update_strategy_performance(self):
        """Test updating strategy performance."""
        # Update performance
        performance_data = {
            "Strategy1": 0.15,
            "Strategy2": 0.05,
            "NonExistentStrategy": 0.1  # Should be ignored
        }
        
        self.rotator.update_strategy_performance(performance_data)
        
        # Check strategies were updated
        self.strategy1.update_performance.assert_called_once_with(0.15)
        self.strategy2.update_performance.assert_called_once_with(0.05)
        
        # Check performance history was updated
        self.assertEqual(len(self.rotator.performance_history), 1)
        self.assertEqual(self.rotator.performance_history[0]["data"], performance_data)

    def test_generate_signals(self):
        """Test generating signals."""
        # Generate signals
        market_data = {"prices": [100, 101, 102, 103]}
        signals = self.rotator.generate_signals(market_data)
        
        # Check signals
        self.assertEqual(signals["Strategy1"], 0.5)
        self.assertEqual(signals["Strategy2"], -0.3)
        
        # Check strategy generate_signal was called
        self.strategy1.generate_signal.assert_called_once_with(market_data)
        self.strategy2.generate_signal.assert_called_once_with(market_data)
        
        # Check signals history was updated
        self.assertEqual(len(self.rotator.signals_history), 1)
        
        # Test with disabled strategy
        self.strategy1.enabled = False
        signals = self.rotator.generate_signals(market_data)
        self.assertEqual(signals["Strategy1"], 0.0)  # Should return 0 for disabled
        
        # Test with exception in generate_signal
        self.strategy2.generate_signal.side_effect = Exception("Test exception")
        with mock.patch("logging.Logger.error") as mock_error:
            signals = self.rotator.generate_signals(market_data)
            mock_error.assert_called_once()
            self.assertEqual(signals["Strategy2"], 0.0)  # Should return 0 on exception

    def test_get_combined_signal(self):
        """Test getting combined signal."""
        # Set weights
        self.rotator.strategy_weights = {"Strategy1": 0.7, "Strategy2": 0.3}
        
        # Get combined signal with market data
        market_data = {"prices": [100, 101, 102, 103]}
        combined = self.rotator.get_combined_signal(market_data)
        
        # Expected: 0.5 * 0.7 + (-0.3) * 0.3 = 0.35 - 0.09 = 0.26
        self.assertAlmostEqual(combined, 0.26, places=5)
        
        # Get combined signal without market data (should use last signals)
        self.strategy1.last_signal = 0.8
        self.strategy2.last_signal = 0.2
        combined = self.rotator.get_combined_signal()
        
        # Expected: 0.8 * 0.7 + 0.2 * 0.3 = 0.56 + 0.06 = 0.62
        self.assertAlmostEqual(combined, 0.62, places=5)

    def test_normalize_weights(self):
        """Test normalizing weights."""
        # Set weights
        self.rotator.strategy_weights = {"Strategy1": 2.0, "Strategy2": 3.0}
        
        # Normalize
        self.rotator._normalize_weights()
        
        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(self.rotator.strategy_weights.values()), 1.0, places=5)
        self.assertAlmostEqual(self.rotator.strategy_weights["Strategy1"], 0.4, places=5)
        self.assertAlmostEqual(self.rotator.strategy_weights["Strategy2"], 0.6, places=5)

    def test_reset(self):
        """Test resetting the rotator."""
        # Add some history
        self.rotator.signals_history = [{"test": "data"}]
        self.rotator.performance_history = [{"test": "data"}]
        self.rotator.current_regime = MarketRegime.BULL
        
        # Reset
        self.rotator.reset()
        
        # Check state was reset
        self.assertEqual(self.rotator.signals_history, [])
        self.assertEqual(self.rotator.performance_history, [])
        self.assertEqual(self.rotator.current_regime, MarketRegime.UNKNOWN)
        
        # Check strategies were reset
        self.strategy1.reset.assert_called_once()
        self.strategy2.reset.assert_called_once()

    def test_save_load_state(self):
        """Test saving and loading state."""
        # Setup state
        self.rotator.current_regime = MarketRegime.BULL
        self.rotator.strategy_weights = {"Strategy1": 0.7, "Strategy2": 0.3}
        self.rotator.signals_history = [{"timestamp": datetime.now().isoformat(), "signals": {"Strategy1": 0.5}}]
        self.rotator.performance_history = [{"timestamp": datetime.now().isoformat(), "data": {"Strategy1": 0.1}}]
        
        # Mock strategy to_dict
        self.strategy1.to_dict.return_value = {
            "name": "Strategy1",
            "enabled": True,
            "performance_history": [{"timestamp": "2023-01-01T12:00:00", "performance": 0.1}],
            "last_signal": 0.5,
            "last_update_time": datetime.now().isoformat(),
            "config": {}
        }
        
        self.strategy2.to_dict.return_value = {
            "name": "Strategy2",
            "enabled": True,
            "performance_history": [],
            "last_signal": -0.3,
            "last_update_time": None,
            "config": {}
        }
        
        # Save state
        self.rotator.save_state()
        
        # Create new rotator and load state
        new_rotator = StrategyRotator(
            strategies=[self.strategy1, self.strategy2],
            data_dir=self.temp_dir
        )
        
        # Should load state from file
        result = new_rotator.load_state()
        self.assertTrue(result)
        self.assertEqual(new_rotator.current_regime, MarketRegime.BULL)
        self.assertEqual(new_rotator.strategy_weights, {"Strategy1": 0.7, "Strategy2": 0.3})
        self.assertEqual(len(new_rotator.signals_history), 1)
        self.assertEqual(len(new_rotator.performance_history), 1)


if __name__ == "__main__":
    unittest.main() 