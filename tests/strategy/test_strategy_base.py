#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Strategy base class.
"""

import unittest
from datetime import datetime
from trading_bot.strategy.base.strategy import Strategy

class TestStrategyBase(unittest.TestCase):
    """Test case for the Strategy base class."""

    def setUp(self):
        """Set up test case."""
        self.strategy = Strategy("TestStrategy", {"param1": 10, "param2": 20})

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "TestStrategy")
        self.assertEqual(self.strategy.config, {"param1": 10, "param2": 20})
        self.assertTrue(self.strategy.enabled)
        self.assertEqual(self.strategy.performance_history, [])
        self.assertEqual(self.strategy.last_signal, 0.0)
        self.assertIsNone(self.strategy.last_update_time)

    def test_generate_signal(self):
        """Test generate_signal method."""
        # Base implementation should return 0.0
        signal = self.strategy.generate_signal({"prices": [100, 101, 102]})
        self.assertEqual(signal, 0.0)

    def test_update_performance(self):
        """Test update_performance method."""
        self.strategy.update_performance(0.05)
        self.assertEqual(len(self.strategy.performance_history), 1)
        self.assertEqual(self.strategy.performance_history[0]["performance"], 0.05)
        self.assertIn("timestamp", self.strategy.performance_history[0])

    def test_get_average_performance(self):
        """Test get_average_performance method."""
        # Test with no performance history
        avg = self.strategy.get_average_performance()
        self.assertEqual(avg, 0.0)

        # Add some performance data
        self.strategy.update_performance(0.1)
        self.strategy.update_performance(0.2)
        self.strategy.update_performance(0.3)

        # Test average calculation
        avg = self.strategy.get_average_performance()
        self.assertAlmostEqual(avg, 0.2, places=5)

        # Test with window parameter
        avg = self.strategy.get_average_performance(window=2)
        self.assertAlmostEqual(avg, 0.25, places=5)  # Average of 0.2 and 0.3

    def test_reset(self):
        """Test reset method."""
        # Add some data
        self.strategy.update_performance(0.1)
        self.strategy.last_signal = 0.5
        self.strategy.last_update_time = datetime.now()

        # Reset
        self.strategy.reset()

        # Verify reset
        self.assertEqual(self.strategy.performance_history, [])
        self.assertEqual(self.strategy.last_signal, 0.0)
        self.assertIsNone(self.strategy.last_update_time)

    def test_to_dict(self):
        """Test to_dict method."""
        # Add some data
        self.strategy.update_performance(0.1)
        self.strategy.last_signal = 0.5
        self.strategy.last_update_time = datetime.now()

        # Convert to dict
        strategy_dict = self.strategy.to_dict()

        # Verify dict
        self.assertEqual(strategy_dict["name"], "TestStrategy")
        self.assertEqual(strategy_dict["config"], {"param1": 10, "param2": 20})
        self.assertTrue(strategy_dict["enabled"])
        self.assertEqual(len(strategy_dict["performance_history"]), 1)
        self.assertEqual(strategy_dict["last_signal"], 0.5)
        self.assertIsNotNone(strategy_dict["last_update_time"])

    def test_from_dict(self):
        """Test from_dict method."""
        # Create strategy dict
        strategy_dict = {
            "name": "FromDictStrategy",
            "config": {"param3": 30, "param4": 40},
            "enabled": False,
            "performance_history": [
                {"timestamp": "2023-01-01T12:00:00", "performance": 0.05},
                {"timestamp": "2023-01-02T12:00:00", "performance": 0.1}
            ],
            "last_signal": -0.3,
            "last_update_time": "2023-01-02T12:00:00"
        }

        # Create strategy from dict
        strategy = Strategy.from_dict(strategy_dict)

        # Verify strategy
        self.assertEqual(strategy.name, "FromDictStrategy")
        self.assertEqual(strategy.config, {"param3": 30, "param4": 40})
        self.assertFalse(strategy.enabled)
        self.assertEqual(len(strategy.performance_history), 2)
        self.assertEqual(strategy.last_signal, -0.3)
        self.assertEqual(strategy.last_update_time.isoformat(), "2023-01-02T12:00:00")

if __name__ == "__main__":
    unittest.main() 