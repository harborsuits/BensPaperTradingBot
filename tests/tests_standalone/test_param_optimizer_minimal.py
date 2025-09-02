#!/usr/bin/env python3
"""
Minimal Standalone Tests for Parameter Optimization System

This test is completely self-contained and doesn't rely on any external imports.
It directly implements test versions of all necessary classes.
"""

import unittest
import json
from datetime import datetime
from enum import Enum
import tempfile
import os


# Mock classes and enums
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


class MarketParameters:
    """Container for synthetic market generation parameters."""
    
    def __init__(self):
        """Initialize with default parameters."""
        # Base parameters - used across all regimes
        self.base_volatility = 0.01
        self.base_drift = 0.0001
        self.mean_reversion_strength = 0.1
        self.correlation_strength = 0.3
        
        # Regime-specific parameters
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
        
        # Sector correlation parameters
        self.intra_sector_correlation = 0.7
        self.inter_sector_correlation = 0.3
        
        # Optimization metadata
        self.last_updated = datetime.utcnow().isoformat()
        self.optimization_history = []
        self.version = "1.0.0"
    
    def get_effective_parameters(self, regime):
        """Get effective parameters for a specific regime."""
        if regime not in self.regime_parameters:
            regime = MarketRegimeType.SIDEWAYS.value  # Default to sideways
        
        regime_params = self.regime_parameters[regime]
        
        return {
            "volatility": self.base_volatility * regime_params["volatility_multiplier"],
            "drift": self.base_drift * regime_params["drift_multiplier"],
            "momentum_factor": regime_params["momentum_factor"],
            "volatility_clustering": regime_params["volatility_clustering"],
            "panic_selling_probability": regime_params["panic_selling_probability"],
            "mean_reversion_strength": self.mean_reversion_strength,
            "correlation_strength": self.correlation_strength,
            "intra_sector_correlation": self.intra_sector_correlation,
            "inter_sector_correlation": self.inter_sector_correlation
        }
    
    def to_dict(self):
        """Convert parameters to dictionary for serialization."""
        return {
            "base_volatility": self.base_volatility,
            "base_drift": self.base_drift,
            "mean_reversion_strength": self.mean_reversion_strength,
            "correlation_strength": self.correlation_strength,
            "regime_parameters": self.regime_parameters,
            "intra_sector_correlation": self.intra_sector_correlation,
            "inter_sector_correlation": self.inter_sector_correlation,
            "last_updated": self.last_updated,
            "optimization_history": self.optimization_history,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create parameters from dictionary."""
        params = cls()
        
        # Base parameters
        params.base_volatility = data.get("base_volatility", params.base_volatility)
        params.base_drift = data.get("base_drift", params.base_drift)
        params.mean_reversion_strength = data.get("mean_reversion_strength", params.mean_reversion_strength)
        params.correlation_strength = data.get("correlation_strength", params.correlation_strength)
        
        # Regime-specific parameters
        regime_params = data.get("regime_parameters", {})
        for regime, regime_data in regime_params.items():
            if regime in params.regime_parameters:
                params.regime_parameters[regime].update(regime_data)
        
        # Sector correlation parameters
        params.intra_sector_correlation = data.get("intra_sector_correlation", params.intra_sector_correlation)
        params.inter_sector_correlation = data.get("inter_sector_correlation", params.inter_sector_correlation)
        
        # Metadata
        params.last_updated = data.get("last_updated", params.last_updated)
        params.optimization_history = data.get("optimization_history", params.optimization_history)
        params.version = data.get("version", params.version)
        
        return params


class ParameterOptimizer:
    """Adjusts synthetic market parameters based on verification results."""
    
    def __init__(self, storage_path=None):
        """Initialize the parameter optimizer."""
        # Set default storage path
        self.storage_path = storage_path or "/tmp/test_parameters.json"
        
        # Core components
        self.event_bus = EventBus()
        
        # Market parameters
        self.parameters = MarketParameters()
        
        # Optimization settings
        self.optimization_enabled = True
        self.learning_rate = 0.05
        self.min_data_points = 10
        self.max_adjustment = 0.2
        
        # Optimization counter
        self.optimization_count = 0
        self.last_optimization = None
        
        # Register for verification events
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register for relevant events."""
        # Listen for verification report generation
        self.event_bus.register(
            "verification_report_generated",
            self._handle_verification_report
        )
    
    def _handle_verification_report(self, event):
        """Handle verification report events."""
        if not self.optimization_enabled:
            return
        
        report = event.data.get("report")
        if not report:
            return
        
        # Check if we have enough data for optimization
        metrics = report.get("overall_metrics", {})
        if metrics.get("total_predictions", 0) < self.min_data_points:
            return
        
        # Optimize parameters based on report
        self._optimize_parameters(report)
        
        # Emit event for parameter update
        self.event_bus.emit(
            Event(
                event_type="market_parameters_updated",
                data={
                    "optimization_count": self.optimization_count,
                    "parameters": self.parameters.to_dict()
                },
                source="parameter_optimizer"
            )
        )
    
    def _optimize_parameters(self, report):
        """Optimize parameters based on verification report."""
        # Get overall metrics
        metrics = report.get("overall_metrics", {})
        prediction_accuracy = metrics.get("prediction_accuracy", 0.0)
        sharpe_correlation = metrics.get("sharpe_correlation", 0.0)
        drawdown_correlation = metrics.get("drawdown_correlation", 0.0)
        
        # Get regime-specific accuracy
        regime_accuracy = report.get("regime_accuracy", {})
        
        # Record optimization attempt
        optimization_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_accuracy": prediction_accuracy,
            "adjustments": {}
        }
        
        # Adjust base parameters based on overall metrics
        adjustments = {}
        
        # If sharpe correlation is low, adjust volatility
        if sharpe_correlation < 0.7:
            adjustment_factor = (0.7 - sharpe_correlation) * self.learning_rate
            adjustment_factor = min(adjustment_factor, self.max_adjustment)
            
            # Apply adjustment (simple example)
            self.parameters.base_volatility *= (1.0 + adjustment_factor)
            
            optimization_record["adjustments"]["base_volatility"] = {
                "before": self.parameters.base_volatility / (1.0 + adjustment_factor),
                "after": self.parameters.base_volatility
            }
        
        # Update metadata
        self.parameters.last_updated = datetime.utcnow().isoformat()
        self.parameters.optimization_history.append(optimization_record)
        
        # Keep only last 50 optimization records
        if len(self.parameters.optimization_history) > 50:
            self.parameters.optimization_history = self.parameters.optimization_history[-50:]
        
        # Increment optimization counter
        self.optimization_count += 1
        self.last_optimization = datetime.utcnow()


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
        self.assertIn("regime_parameters", data)
        
        # Reconstruct from dict
        reconstructed = MarketParameters.from_dict(data)
        
        # Verify values match
        self.assertEqual(reconstructed.base_volatility, 0.015)
        self.assertEqual(reconstructed.base_drift, 0.0002)
        self.assertEqual(reconstructed.mean_reversion_strength, 0.15)
        self.assertEqual(reconstructed.regime_parameters["bullish"]["momentum_factor"], 0.25)
        self.assertEqual(len(reconstructed.optimization_history), 1)


class TestParameterOptimizer(unittest.TestCase):
    """Test ParameterOptimizer class."""
    
    def setUp(self):
        """Set up test case."""
        # Create temporary directory for parameter storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_parameters.json")
        
        # Create optimizer
        self.optimizer = ParameterOptimizer(storage_path=self.storage_path)
    
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
        self.assertIn("verification_report_generated", self.optimizer.event_bus.handlers)
    
    def test_handle_verification_report(self):
        """Test handling verification report events."""
        # Create verification report with good accuracy
        report = {
            "overall_metrics": {
                "total_predictions": 20,
                "prediction_accuracy": 0.85,
                "sharpe_correlation": 0.9,
                "drawdown_correlation": 0.9
            },
            "regime_accuracy": {
                "bullish": {"accuracy": 0.9},
                "bearish": {"accuracy": 0.8}
            }
        }
        
        # Create event
        event = Event(
            event_type="verification_report_generated",
            data={"report": report}
        )
        
        # Handle event
        self.optimizer._handle_verification_report(event)
        
        # Verify optimization count increased
        self.assertEqual(self.optimizer.optimization_count, 1)
        self.assertIsNotNone(self.optimizer.last_optimization)
        
        # Verify event emitted
        self.assertEqual(len(self.optimizer.event_bus.emitted_events), 1)
        emitted_event = self.optimizer.event_bus.emitted_events[0]
        self.assertEqual(emitted_event.event_type, "market_parameters_updated")
    
    def test_optimize_parameters_low_accuracy(self):
        """Test parameter optimization with low accuracy."""
        # Create verification report with low accuracy
        report = {
            "overall_metrics": {
                "total_predictions": 20,
                "prediction_accuracy": 0.5,
                "sharpe_correlation": 0.4,
                "drawdown_correlation": 0.5
            },
            "regime_accuracy": {
                "bullish": {"accuracy": 0.5},
                "bearish": {"accuracy": 0.4}
            }
        }
        
        # Record initial parameter values
        initial_volatility = self.optimizer.parameters.base_volatility
        
        # Optimize parameters
        self.optimizer._optimize_parameters(report)
        
        # Verify parameters have been adjusted
        self.assertNotEqual(self.optimizer.parameters.base_volatility, initial_volatility)
        
        # Verify optimization history updated
        self.assertEqual(len(self.optimizer.parameters.optimization_history), 1)
        history_entry = self.optimizer.parameters.optimization_history[0]
        self.assertIn("timestamp", history_entry)
        self.assertIn("overall_accuracy", history_entry)
        self.assertIn("adjustments", history_entry)


if __name__ == "__main__":
    unittest.main()
