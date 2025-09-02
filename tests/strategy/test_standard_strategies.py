#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the standard strategy implementations.
"""

import unittest
import numpy as np
from datetime import datetime

from trading_bot.strategy.implementations.standard_strategies import (
    MomentumStrategy,
    TrendFollowingStrategy,
    MeanReversionStrategy
)

class TestMomentumStrategy(unittest.TestCase):
    """Test case for the MomentumStrategy class."""

    def setUp(self):
        """Set up test case."""
        self.strategy = MomentumStrategy(
            "TestMomentum", 
            {"fast_period": 3, "slow_period": 10}
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "TestMomentum")
        self.assertEqual(self.strategy.config, {"fast_period": 3, "slow_period": 10})
        self.assertTrue(self.strategy.enabled)

    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Empty prices
        signal = self.strategy.generate_signal({"prices": []})
        self.assertEqual(signal, 0.0)

        # Single price
        signal = self.strategy.generate_signal({"prices": [100]})
        self.assertEqual(signal, 0.0)

        # Not enough for slow period
        signal = self.strategy.generate_signal({"prices": [100, 101, 102]})
        self.assertEqual(signal, 0.0)

    def test_generate_signal_uptrend(self):
        """Test signal generation in uptrend."""
        # Create uptrend: 100, 101, 102, ... 110
        prices = [100 + i for i in range(11)]
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be positive in uptrend
        self.assertGreater(signal, 0)
        self.assertLessEqual(signal, 1.0)

    def test_generate_signal_downtrend(self):
        """Test signal generation in downtrend."""
        # Create downtrend: 110, 109, 108, ... 100
        prices = [110 - i for i in range(11)]
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be negative in downtrend
        self.assertLess(signal, 0)
        self.assertGreaterEqual(signal, -1.0)

    def test_generate_signal_flat(self):
        """Test signal generation in flat market."""
        # Create flat: 100, 100, 100, ... 100
        prices = [100] * 11
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be close to zero in flat market
        self.assertAlmostEqual(signal, 0.0, places=5)


class TestTrendFollowingStrategy(unittest.TestCase):
    """Test case for the TrendFollowingStrategy class."""

    def setUp(self):
        """Set up test case."""
        self.strategy = TrendFollowingStrategy(
            "TestTrend", 
            {"short_ma_period": 5, "long_ma_period": 15}
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "TestTrend")
        self.assertEqual(self.strategy.config, {"short_ma_period": 5, "long_ma_period": 15})
        self.assertTrue(self.strategy.enabled)

    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Empty prices
        signal = self.strategy.generate_signal({"prices": []})
        self.assertEqual(signal, 0.0)

        # Single price
        signal = self.strategy.generate_signal({"prices": [100]})
        self.assertEqual(signal, 0.0)

        # Not enough for long period
        signal = self.strategy.generate_signal({"prices": [100 + i for i in range(10)]})
        self.assertEqual(signal, 0.0)

    def test_generate_signal_uptrend(self):
        """Test signal generation in uptrend."""
        # Create uptrend
        prices = [100]
        for i in range(1, 20):
            prices.append(prices[-1] * (1 + 0.01))  # 1% increase each period
            
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be positive in uptrend
        self.assertGreater(signal, 0)
        self.assertLessEqual(signal, 1.0)

    def test_generate_signal_downtrend(self):
        """Test signal generation in downtrend."""
        # Create downtrend
        prices = [100]
        for i in range(1, 20):
            prices.append(prices[-1] * (1 - 0.01))  # 1% decrease each period
            
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be negative in downtrend
        self.assertLess(signal, 0)
        self.assertGreaterEqual(signal, -1.0)

    def test_generate_signal_flat(self):
        """Test signal generation in flat market."""
        # Create flat
        prices = [100] * 20
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be close to zero in flat market
        self.assertAlmostEqual(signal, 0.0, places=5)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test case for the MeanReversionStrategy class."""

    def setUp(self):
        """Set up test case."""
        self.strategy = MeanReversionStrategy(
            "TestMeanReversion", 
            {"period": 10, "std_dev_factor": 1.5}
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "TestMeanReversion")
        self.assertEqual(self.strategy.config, {"period": 10, "std_dev_factor": 1.5})
        self.assertTrue(self.strategy.enabled)

    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Empty prices
        signal = self.strategy.generate_signal({"prices": []})
        self.assertEqual(signal, 0.0)

        # Single price
        signal = self.strategy.generate_signal({"prices": [100]})
        self.assertEqual(signal, 0.0)

        # Not enough for period
        signal = self.strategy.generate_signal({"prices": [100 + i for i in range(5)]})
        self.assertEqual(signal, 0.0)

    def test_generate_signal_overbought(self):
        """Test signal generation in overbought conditions."""
        # Create prices with last price significantly above mean
        prices = [100] * 9  # First 9 prices are 100
        prices.append(120)  # Last price is 120 (significantly above mean)
            
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be negative when overbought (price above mean)
        self.assertLess(signal, 0)
        self.assertGreaterEqual(signal, -1.0)

    def test_generate_signal_oversold(self):
        """Test signal generation in oversold conditions."""
        # Create prices with last price significantly below mean
        prices = [100] * 9  # First 9 prices are 100
        prices.append(80)   # Last price is 80 (significantly below mean)
            
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be positive when oversold (price below mean)
        self.assertGreater(signal, 0)
        self.assertLessEqual(signal, 1.0)

    def test_generate_signal_neutral(self):
        """Test signal generation in neutral conditions."""
        # Create prices with last price at mean
        prices = [100] * 10  # All prices are 100
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should be close to zero when price is at mean
        self.assertAlmostEqual(signal, 0.0, places=5)

    def test_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        # Create prices with no variation (std dev = 0)
        prices = [100] * 10
        signal = self.strategy.generate_signal({"prices": prices})
        
        # Should handle zero std dev without error
        self.assertEqual(signal, 0.0)


if __name__ == "__main__":
    unittest.main() 