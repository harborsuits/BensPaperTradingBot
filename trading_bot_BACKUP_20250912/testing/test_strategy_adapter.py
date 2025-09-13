#!/usr/bin/env python3
"""
Unit Tests for Strategy Adapter

This module contains tests for the strategy adapter implementation,
ensuring it properly wraps different strategy implementations with a
standardized interface.
"""

import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Add project root to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components to test
from trading_bot.strategies.strategy_adapter import StrategyAdapter, create_strategy_adapter
from trading_bot.strategies.strategy_template import AbstractStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBaseStrategy:
    """Mock strategy without standard interface methods."""
    
    def __init__(self):
        self.name = "MockBaseStrategy"
        self.parameters = {
            "param1": 10,
            "param2": "value"
        }
    
    def custom_signal_method(self, market_data):
        """Non-standard method for signal generation."""
        return [{"symbol": "AAPL", "direction": "BUY", "confidence": 0.8}]
    
    def custom_position_method(self, signals, account_balance):
        """Non-standard method for position sizing."""
        return [{"symbol": "AAPL", "quantity": 10, "price": 150.0}]
    
    def custom_trade_manager(self, open_positions, market_data):
        """Non-standard method for trade management."""
        return [{"position_id": "123", "action": "CLOSE"}]
    
    def get_parameter_space(self):
        """Return parameter space for optimization."""
        return {
            "param1": (5, 20, 1),
            "param2": ["value1", "value2", "value3"]
        }

class MockStandardStrategy:
    """Mock strategy that already implements the standard interface."""
    
    def __init__(self):
        self.name = "MockStandardStrategy"
        self.parameters = {
            "threshold": 0.5,
            "window": 20
        }
    
    def generate_signals(self, market_data):
        """Standard method for signal generation."""
        return [{"symbol": "MSFT", "direction": "SELL", "confidence": 0.7}]
    
    def size_position(self, signals, account_balance):
        """Standard method for position sizing."""
        return [{"symbol": "MSFT", "quantity": 5, "price": 250.0}]
    
    def manage_open_trades(self, open_positions, market_data):
        """Standard method for trade management."""
        return [{"position_id": "456", "action": "ADJUST", "new_stop_loss": 240.0}]
    
    def get_parameter_space(self):
        """Return parameter space for optimization."""
        return {
            "threshold": (0.1, 0.9, 0.1),
            "window": (5, 50, 5)
        }

class StrategyAdapterTests(unittest.TestCase):
    """Test cases for StrategyAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_strategy = MockBaseStrategy()
        self.standard_strategy = MockStandardStrategy()
        
        # Mock market data and account info
        self.market_data = {
            "AAPL": {"price": 150.0, "volume": 1000000},
            "MSFT": {"price": 250.0, "volume": 800000}
        }
        self.account_balance = 100000.0
        self.open_positions = [
            {"position_id": "123", "symbol": "AAPL", "quantity": 10, "entry_price": 145.0},
            {"position_id": "456", "symbol": "MSFT", "quantity": 5, "entry_price": 255.0}
        ]
    
    def test_create_adapter_for_base_strategy(self):
        """Test creating an adapter for a non-standard strategy."""
        # Create adapter
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Verify it's an instance of StrategyAdapter
        self.assertIsInstance(adapter, StrategyAdapter)
        
        # Check that standard methods are now available
        self.assertTrue(hasattr(adapter, 'generate_signals'))
        self.assertTrue(hasattr(adapter, 'size_position'))
        self.assertTrue(hasattr(adapter, 'manage_open_trades'))
        
        # Check that original parameters are preserved
        self.assertEqual(adapter.parameters, self.base_strategy.parameters)
    
    def test_create_adapter_for_standard_strategy(self):
        """Test creating an adapter for a strategy that already has standard methods."""
        # Create adapter
        adapter = create_strategy_adapter(self.standard_strategy)
        
        # Verify type
        self.assertIsInstance(adapter, StrategyAdapter)
        
        # Check parameters
        self.assertEqual(adapter.parameters, self.standard_strategy.parameters)
    
    def test_generate_signals_mapping(self):
        """Test mapping of custom signal method to standard interface."""
        # Create adapter for non-standard strategy
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Mock the custom method
        self.base_strategy.custom_signal_method = MagicMock(return_value=[
            {"symbol": "AAPL", "direction": "BUY", "confidence": 0.8}
        ])
        
        # Call through adapter
        signals = adapter.generate_signals(self.market_data)
        
        # Verify custom method was called
        self.base_strategy.custom_signal_method.assert_called_once_with(self.market_data)
        
        # Verify signals were passed through
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["symbol"], "AAPL")
    
    def test_size_position_mapping(self):
        """Test mapping of custom position sizing method to standard interface."""
        # Create adapter for non-standard strategy
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Mock the custom method
        self.base_strategy.custom_position_method = MagicMock(return_value=[
            {"symbol": "AAPL", "quantity": 10, "price": 150.0}
        ])
        
        # Sample signals
        signals = [{"symbol": "AAPL", "direction": "BUY", "confidence": 0.8}]
        
        # Call through adapter
        positions = adapter.size_position(signals, self.account_balance)
        
        # Verify custom method was called
        self.base_strategy.custom_position_method.assert_called_once_with(signals, self.account_balance)
        
        # Verify positions were passed through
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["quantity"], 10)
    
    def test_manage_open_trades_mapping(self):
        """Test mapping of custom trade management method to standard interface."""
        # Create adapter for non-standard strategy
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Mock the custom method
        self.base_strategy.custom_trade_manager = MagicMock(return_value=[
            {"position_id": "123", "action": "CLOSE"}
        ])
        
        # Call through adapter
        actions = adapter.manage_open_trades(self.open_positions, self.market_data)
        
        # Verify custom method was called
        self.base_strategy.custom_trade_manager.assert_called_once_with(self.open_positions, self.market_data)
        
        # Verify actions were passed through
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "CLOSE")
    
    def test_attribute_passthrough(self):
        """Test that attributes are passed through to the wrapped strategy."""
        # Create adapter
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Verify name attribute is accessible
        self.assertEqual(adapter.name, "MockBaseStrategy")
        
        # Set a new attribute on the base strategy
        self.base_strategy.new_attr = "test_value"
        
        # Verify it's accessible through the adapter
        self.assertEqual(adapter.new_attr, "test_value")
    
    def test_parameter_space_passthrough(self):
        """Test that parameter space is correctly passed through."""
        # Create adapter
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Get parameter space through adapter
        param_space = adapter.get_parameter_space()
        
        # Verify it matches the base strategy
        self.assertEqual(param_space, self.base_strategy.get_parameter_space())
        self.assertEqual(param_space["param1"], (5, 20, 1))

    def test_direct_method_forwarding(self):
        """Test that methods are directly forwarded for standard strategies."""
        # Create adapter for standard strategy
        adapter = create_strategy_adapter(self.standard_strategy)
        
        # Mock the standard methods
        self.standard_strategy.generate_signals = MagicMock(return_value=[
            {"symbol": "MSFT", "direction": "SELL", "confidence": 0.7}
        ])
        
        # Call through adapter
        signals = adapter.generate_signals(self.market_data)
        
        # Verify method was called on the standard strategy
        self.standard_strategy.generate_signals.assert_called_once_with(self.market_data)
        
        # Verify results match
        self.assertEqual(signals[0]["symbol"], "MSFT")

    def test_adapter_error_handling(self):
        """Test that the adapter properly handles errors in wrapped methods."""
        # Create adapter
        adapter = create_strategy_adapter(self.base_strategy)
        
        # Make the custom method raise an exception
        self.base_strategy.custom_signal_method = MagicMock(side_effect=Exception("Test error"))
        
        # Call through adapter - should not raise but return empty list
        signals = adapter.generate_signals(self.market_data)
        
        # Verify empty result
        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
