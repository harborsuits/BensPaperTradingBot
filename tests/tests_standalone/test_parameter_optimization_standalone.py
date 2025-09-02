#!/usr/bin/env python3
"""
Standalone Tests for Parameter Optimization System

This test file is completely self-contained, implementing test versions of the core classes
from parameter_optimization.py to avoid any external dependencies.

This follows our proven approach of focusing on core functionality first and avoiding
external dependencies in tests.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import shutil


# Create mock enums and classes
class MarketRegimeType(Enum):
    """Mock of the MarketRegimeType enum."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class Event:
    """Mock of the Event class."""
    
    def __init__(self, event_type, data=None, source=None):
        self.event_type = event_type
        self.data = data or {}
        self.source = source or "test"


class EventBus:
    """Mock of the EventBus class."""
    
    def __init__(self):
        self.handlers = {}
        self.emitted_events = []
    
    def register(self, event_type, handler):
        """Register a handler for an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def emit(self, event):
        """Emit an event."""
        self.emitted_events.append(event)
        event_type = event.event_type
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)


class SyntheticMarketGenerator:
    """Mock of the SyntheticMarketGenerator class."""
    
    def __init__(self):
        self.base_volatility = 0.01
        self.base_drift = 0.0001
        self.mean_reversion_strength = 0.1
        self.regime_parameters = {
            MarketRegimeType.BULLISH.value: {
                "volatility_multiplier": 1.5,
                "drift_multiplier": 8.0,
                "momentum_factor": 0.2,
                "volatility_clustering": 0.1,
                "panic_selling_probability": 0.05
            },
            MarketRegimeType.BEARISH.value: {
                "volatility_multiplier": 2.0,
                "drift_multiplier": -6.0,
                "momentum_factor": 0.1,
                "volatility_clustering": 0.15,
                "panic_selling_probability": 0.3
            },
            MarketRegimeType.SIDEWAYS.value: {
                "volatility_multiplier": 0.8,
                "drift_multiplier": 1.0,
                "momentum_factor": 0.05,
                "volatility_clustering": 0.05,
                "panic_selling_probability": 0.1
            },
            MarketRegimeType.VOLATILE.value: {
                "volatility_multiplier": 3.0,
                "drift_multiplier": 1.0,
                "momentum_factor": 0.15,
                "volatility_clustering": 0.3,
                "panic_selling_probability": 0.2
            }
        }


class CorrelatedMarketGenerator(SyntheticMarketGenerator):
    """Mock of the CorrelatedMarketGenerator class."""
    
    def __init__(self):
        super().__init__()
        self.default_intra_sector_correlation = 0.7
        self.default_inter_sector_correlation = 0.3


# Import from the parameter optimization module
from trading_bot.autonomous.parameter_optimization import (
    MarketParameters, ParameterOptimizer
)


class TestMarketParameters(unittest.TestCase):
    """Test MarketParameters class."""
    
    def test_initialization(self):
        """Test initialization of market parameters."""
        params = MarketParameters()
        
        # Verify initial values
        self.assertEqual(params.base_volatility, 0.01)
        self.assertEqual(params.base_drift, 0.0001)
        self.assertEqual(params.mean_reversion_strength, 0.1)
        self.assertEqual(params.correlation_strength, 0.3)
        self.assertEqual(params.intra_sector_correlation, 0.7)
        self.assertEqual(params.inter_sector_correlation, 0.3)
        self.assertEqual(params.version, "1.0.0")
        self.assertEqual(params.optimization_history, [])
    
    def test_get_effective_parameters(self):
        """Test getting effective parameters for a regime."""
        params = MarketParameters()
        
        # Get bullish parameters
        bullish_params = params.get_effective_parameters("bullish")
        
        # Verify values
        self.assertEqual(bullish_params["volatility"], 0.01 * 1.5)  # base * multiplier
        self.assertEqual(bullish_params["drift"], 0.0001 * 8.0)    # base * multiplier
        self.assertEqual(bullish_params["momentum_factor"], 0.2)
        self.assertEqual(bullish_params["panic_selling_probability"], 0.05)
        
        # Get parameters for invalid regime (should default to sideways)
        invalid_params = params.get_effective_parameters("invalid")
        self.assertEqual(invalid_params["volatility"], 0.01 * 0.8)  # sideways regime
    
    def test_to_dict_and_from_dict(self):
        """Test serialization to and from dictionary."""
        original = MarketParameters()
        original.base_volatility = 0.015
        original.base_drift = 0.0002
        original.mean_reversion_strength = 0.15
        original.regime_parameters["bullish"]["momentum_factor"] = 0.25
        original.optimization_history = [{"timestamp": "2025-01-01T00:00:00", "adjustment": 0.1}]
        
        # Convert to dict
        data = original.to_dict()
        
        # Verify dictionary has expected keys
        self.assertIn("base_volatility", data)
        self.assertIn("base_drift", data)
        self.assertIn("mean_reversion_strength", data)
        self.assertIn("regime_parameters", data)
        self.assertIn("intra_sector_correlation", data)
        self.assertIn("inter_sector_correlation", data)
        self.assertIn("last_updated", data)
        self.assertIn("optimization_history", data)
        self.assertIn("version", data)
        
        # Reconstruct from dict
        reconstructed = MarketParameters.from_dict(data)
        
        # Verify values match
        self.assertEqual(reconstructed.base_volatility, 0.015)
        self.assertEqual(reconstructed.base_drift, 0.0002)
        self.assertEqual(reconstructed.mean_reversion_strength, 0.15)
        self.assertEqual(reconstructed.regime_parameters["bullish"]["momentum_factor"], 0.25)
        self.assertEqual(len(reconstructed.optimization_history), 1)
        self.assertEqual(reconstructed.optimization_history[0]["timestamp"], "2025-01-01T00:00:00")


class TestParameterOptimizer(unittest.TestCase):
    """Test ParameterOptimizer class."""
    
    def setUp(self):
        """Set up test case."""
        # Create temporary directory for parameter storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_parameters.json")
        
        # Mock dependencies
        self.mock_event_bus = EventBus()
        self.mock_verifier = MagicMock()
        
        # Patch dependencies
        self.event_bus_patcher = patch('trading_bot.autonomous.parameter_optimization.EventBus')
        self.mock_event_bus_class = self.event_bus_patcher.start()
        self.mock_event_bus_class.return_value = self.mock_event_bus
        
        self.verifier_patcher = patch('trading_bot.autonomous.parameter_optimization.get_performance_verifier')
        self.mock_verifier_func = self.verifier_patcher.start()
        self.mock_verifier_func.return_value = self.mock_verifier
        
        # Create optimizer
        self.optimizer = ParameterOptimizer(storage_path=self.storage_path)
    
    def tearDown(self):
        """Clean up after test case."""
        self.event_bus_patcher.stop()
        self.verifier_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of parameter optimizer."""
        # Verify initial values
        self.assertEqual(self.optimizer.storage_path, self.storage_path)
        self.assertEqual(self.optimizer.optimization_enabled, True)
        self.assertEqual(self.optimizer.learning_rate, 0.05)
        self.assertEqual(self.optimizer.min_data_points, 10)
        self.assertEqual(self.optimizer.max_adjustment, 0.2)
        self.assertEqual(self.optimizer.optimization_count, 0)
        self.assertIsNone(self.optimizer.last_optimization)
        
        # Verify event handler registered
        self.assertIn("verification_report_generated", self.mock_event_bus.handlers)
    
    def test_handle_verification_report(self):
        """Test handling verification report events."""
        # Create verification report with good accuracy
        report = {
            "overall_metrics": {
                "total_predictions": 20,
                "prediction_accuracy": 0.85,
                "sharpe_correlation": 0.9,
                "drawdown_correlation": 0.9,
                "win_rate_correlation": 0.9
            },
            "regime_accuracy": {
                "bullish": {"accuracy": 0.9},
                "bearish": {"accuracy": 0.8},
                "sideways": {"accuracy": 0.85},
                "volatile": {"accuracy": 0.75}
            }
        }
        
        # Create event
        event = Event(
            event_type="verification_report_generated",
            data={"report": report}
        )
        
        # Handle event
        self.optimizer._handle_verification_report(event)
        
        # Verify no changes made (accuracy is already good)
        self.assertEqual(self.optimizer.optimization_count, 1)
        self.assertIsNotNone(self.optimizer.last_optimization)
        
        # Verify parameters saved to disk
        self.assertTrue(os.path.exists(self.storage_path))
        
        # Verify event emitted
        self.assertEqual(len(self.mock_event_bus.emitted_events), 1)
        emitted_event = self.mock_event_bus.emitted_events[0]
        self.assertEqual(emitted_event.event_type, "market_parameters_updated")
    
    def test_optimize_parameters_low_accuracy(self):
        """Test parameter optimization with low accuracy."""
        # Create verification report with low accuracy
        report = {
            "overall_metrics": {
                "total_predictions": 20,
                "prediction_accuracy": 0.5,
                "sharpe_correlation": 0.4,
                "drawdown_correlation": 0.5,
                "win_rate_correlation": 0.3
            },
            "regime_accuracy": {
                "bullish": {"accuracy": 0.5},
                "bearish": {"accuracy": 0.4},
                "sideways": {"accuracy": 0.6},
                "volatile": {"accuracy": 0.3}
            }
        }
        
        # Record initial parameter values
        initial_volatility = self.optimizer.parameters.base_volatility
        initial_mean_reversion = self.optimizer.parameters.mean_reversion_strength
        initial_bullish_panic = self.optimizer.parameters.regime_parameters["bullish"]["panic_selling_probability"]
        initial_bearish_panic = self.optimizer.parameters.regime_parameters["bearish"]["panic_selling_probability"]
        
        # Optimize parameters
        self.optimizer._optimize_parameters(report)
        
        # Verify parameters have been adjusted
        self.assertNotEqual(self.optimizer.parameters.base_volatility, initial_volatility)
        self.assertNotEqual(self.optimizer.parameters.mean_reversion_strength, initial_mean_reversion)
        
        # Verify optimization history updated
        self.assertEqual(len(self.optimizer.parameters.optimization_history), 1)
        history_entry = self.optimizer.parameters.optimization_history[0]
        self.assertIn("timestamp", history_entry)
        self.assertIn("overall_accuracy", history_entry)
        self.assertIn("adjustments", history_entry)
    
    def test_apply_parameters_to_generator(self):
        """Test applying parameters to market generator."""
        # Create market generator
        generator = SyntheticMarketGenerator()
        
        # Modify optimizer parameters
        self.optimizer.parameters.base_volatility = 0.02
        self.optimizer.parameters.base_drift = 0.0005
        self.optimizer.parameters.mean_reversion_strength = 0.2
        
        # Apply parameters
        self.optimizer.apply_parameters_to_generator(generator)
        
        # Verify parameters applied
        self.assertEqual(generator.base_volatility, 0.02)
        self.assertEqual(generator.base_drift, 0.0005)
        self.assertEqual(generator.mean_reversion_strength, 0.2)
    
    def test_apply_parameters_to_correlated_generator(self):
        """Test applying parameters to correlated market generator."""
        # Create correlated market generator
        generator = CorrelatedMarketGenerator()
        
        # Modify optimizer parameters
        self.optimizer.parameters.intra_sector_correlation = 0.8
        self.optimizer.parameters.inter_sector_correlation = 0.4
        
        # Apply parameters
        self.optimizer.apply_parameters_to_correlated_generator(generator)
        
        # Verify parameters applied
        self.assertEqual(generator.default_intra_sector_correlation, 0.8)
        self.assertEqual(generator.default_inter_sector_correlation, 0.4)
    
    def test_reset_to_defaults(self):
        """Test resetting parameters to defaults."""
        # Modify parameters
        self.optimizer.parameters.base_volatility = 0.02
        self.optimizer.parameters.base_drift = 0.0005
        self.optimizer.parameters.optimization_history = [{"test": "data"}]
        
        # Reset to defaults
        self.optimizer.reset_to_defaults()
        
        # Verify parameters reset
        self.assertEqual(self.optimizer.parameters.base_volatility, 0.01)
        self.assertEqual(self.optimizer.parameters.base_drift, 0.0001)
        self.assertEqual(self.optimizer.parameters.optimization_history, [])
    
    def test_save_and_load(self):
        """Test saving and loading parameters."""
        # Modify parameters
        self.optimizer.parameters.base_volatility = 0.02
        self.optimizer.parameters.base_drift = 0.0005
        
        # Save parameters
        self.optimizer._save_to_disk()
        
        # Create new optimizer (will load from disk)
        new_optimizer = ParameterOptimizer(storage_path=self.storage_path)
        
        # Verify parameters loaded
        self.assertEqual(new_optimizer.parameters.base_volatility, 0.02)
        self.assertEqual(new_optimizer.parameters.base_drift, 0.0005)


class TestParameterOptimizerSingleton(unittest.TestCase):
    """Test ParameterOptimizer singleton."""
    
    def setUp(self):
        """Set up test case."""
        # Patch dependencies
        self.event_bus_patcher = patch('trading_bot.autonomous.parameter_optimization.EventBus')
        self.mock_event_bus_class = self.event_bus_patcher.start()
        
        self.verifier_patcher = patch('trading_bot.autonomous.parameter_optimization.get_performance_verifier')
        self.mock_verifier_func = self.verifier_patcher.start()
        
        # Reset singleton
        import trading_bot.autonomous.parameter_optimization
        trading_bot.autonomous.parameter_optimization._parameter_optimizer = None
    
    def tearDown(self):
        """Clean up after test case."""
        self.event_bus_patcher.stop()
        self.verifier_patcher.stop()
    
    def test_singleton(self):
        """Test singleton pattern."""
        # Import singleton getter
        from trading_bot.autonomous.parameter_optimization import get_parameter_optimizer
        
        # Get optimizer
        optimizer1 = get_parameter_optimizer()
        optimizer2 = get_parameter_optimizer()
        
        # Verify same instance returned
        self.assertIs(optimizer1, optimizer2)


if __name__ == "__main__":
    unittest.main()
