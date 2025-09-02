#!/usr/bin/env python3
"""
Standalone Test Suite for Performance Verification

This standalone test verifies that the performance verification system works correctly
without requiring external dependencies. It uses mocks to simulate the interaction
with other components of the trading system.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
import json
from datetime import datetime, timedelta
from enum import Enum
import tempfile

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create mock classes to avoid importing real ones
class EventType(Enum):
    """Mock of the EventType enum."""
    APPROVAL_REQUEST_CREATED = "approval_request_created"
    APPROVAL_REQUEST_APPROVED = "approval_request_approved"
    APPROVAL_REQUEST_REJECTED = "approval_request_rejected"

class MarketRegimeType(Enum):
    """Mock of the MarketRegimeType enum."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class ApprovalStatus(Enum):
    """Mock of the ApprovalStatus enum."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Event:
    """Mock of the Event class."""
    def __init__(self, event_type, data=None, source=None):
        self.event_type = event_type
        self.data = data or {}
        self.source = source

# Import the module directly to avoid dependencies
performance_verification_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'trading_bot', 
    'autonomous', 
    'performance_verification.py'
)

# Patch modules before importing
with patch.dict('sys.modules', {
    'trading_bot': MagicMock(),
    'trading_bot.event_system': MagicMock(),
    'trading_bot.event_system.EventBus': MagicMock(),
    'trading_bot.event_system.Event': Event,
    'trading_bot.event_system.EventType': EventType,
    'trading_bot.autonomous': MagicMock(),
    'trading_bot.autonomous.approval_workflow': MagicMock(),
    'trading_bot.autonomous.approval_workflow.get_approval_workflow_manager': MagicMock(),
    'trading_bot.autonomous.approval_workflow.ApprovalStatus': ApprovalStatus,
    'trading_bot.autonomous.approval_workflow.ApprovalRequest': MagicMock(),
    'trading_bot.autonomous.ab_testing_core': MagicMock(),
    'trading_bot.autonomous.ab_testing_manager': MagicMock(),
    'trading_bot.autonomous.ab_testing_manager.get_ab_test_manager': MagicMock(),
    'trading_bot.autonomous.synthetic_market_generator': MagicMock(),
    'trading_bot.autonomous.synthetic_market_generator.MarketRegimeType': MarketRegimeType,
    'trading_bot.autonomous.synthetic_testing_integration': MagicMock(),
    'trading_bot.autonomous.synthetic_testing_integration.get_synthetic_testing_integration': MagicMock(),
    'numpy': MagicMock(),
    'pandas': MagicMock()
}):
    # Now import the actual module
    import importlib.util
    spec = importlib.util.spec_from_file_location("performance_verification", performance_verification_path)
    performance_verification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(performance_verification)


class TestVerificationMetrics(unittest.TestCase):
    """Test the VerificationMetrics class."""
    
    def test_initialization(self):
        """Test initialization of verification metrics."""
        metrics = performance_verification.VerificationMetrics()
        
        # Verify initial values
        self.assertEqual(metrics.correct_predictions, 0)
        self.assertEqual(metrics.total_predictions, 0)
        self.assertEqual(metrics.sharpe_correlation, 0.0)
        self.assertEqual(metrics.drawdown_correlation, 0.0)
        self.assertEqual(metrics.win_rate_correlation, 0.0)
        self.assertEqual(metrics.regime_detection_accuracy, 0.0)
        self.assertEqual(metrics.regime_prediction_accuracy, 0.0)
        self.assertEqual(metrics.verification_timestamps, [])
        self.assertEqual(metrics.accuracy_over_time, [])
        self.assertEqual(metrics.market_condition_mapping, {})
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy calculation."""
        metrics = performance_verification.VerificationMetrics()
        
        # Test with no predictions
        self.assertEqual(metrics.prediction_accuracy, 0.0)
        
        # Test with predictions
        metrics.correct_predictions = 7
        metrics.total_predictions = 10
        self.assertEqual(metrics.prediction_accuracy, 0.7)
    
    def test_to_dict_and_from_dict(self):
        """Test serialization to and from dictionary."""
        original = performance_verification.VerificationMetrics()
        original.correct_predictions = 15
        original.total_predictions = 20
        original.sharpe_correlation = 0.8
        original.drawdown_correlation = 0.7
        original.win_rate_correlation = 0.9
        original.verification_timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        original.accuracy_over_time = [0.75, 0.8]
        original.market_condition_mapping = {"bullish": {"count": 5, "correct_predictions": 4}}
        
        # Convert to dict
        data = original.to_dict()
        
        # Verify dictionary has expected keys
        self.assertIn("correct_predictions", data)
        self.assertIn("total_predictions", data)
        self.assertIn("prediction_accuracy", data)
        self.assertIn("sharpe_correlation", data)
        self.assertIn("drawdown_correlation", data)
        self.assertIn("win_rate_correlation", data)
        self.assertIn("verification_timestamps", data)
        self.assertIn("accuracy_over_time", data)
        self.assertIn("market_condition_mapping", data)
        
        # Reconstruct from dict
        reconstructed = performance_verification.VerificationMetrics.from_dict(data)
        
        # Verify values match
        self.assertEqual(reconstructed.correct_predictions, 15)
        self.assertEqual(reconstructed.total_predictions, 20)
        self.assertEqual(reconstructed.sharpe_correlation, 0.8)
        self.assertEqual(reconstructed.drawdown_correlation, 0.7)
        self.assertEqual(reconstructed.win_rate_correlation, 0.9)
        self.assertEqual(len(reconstructed.verification_timestamps), 2)
        self.assertEqual(reconstructed.accuracy_over_time, [0.75, 0.8])
        self.assertEqual(reconstructed.market_condition_mapping, {"bullish": {"count": 5, "correct_predictions": 4}})


class TestStrategyPerformanceRecord(unittest.TestCase):
    """Test the StrategyPerformanceRecord class."""
    
    def test_initialization(self):
        """Test initialization of performance record."""
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        # Verify initial values
        self.assertEqual(record.strategy_id, "test_strategy")
        self.assertEqual(record.version_id, "v1.0")
        self.assertEqual(record.approval_request_id, "req123")
        self.assertEqual(record.test_id, "test123")
        self.assertIsNotNone(record.approval_date)
        self.assertEqual(record.performance_snapshots, [])
        self.assertEqual(record.synthetic_predictions, {})
        self.assertEqual(record.verification_results, {})
        self.assertEqual(record.actual_market_regimes, [])
        self.assertTrue(record.record_id.startswith("test_strategy_v1.0_"))
    
    def test_add_performance_snapshot(self):
        """Test adding performance snapshots."""
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        timestamp = datetime(2025, 1, 1)
        metrics = {"sharpe_ratio": 1.5, "max_drawdown": -0.2}
        
        # Add snapshot without regime
        record.add_performance_snapshot(timestamp, metrics)
        self.assertEqual(len(record.performance_snapshots), 1)
        self.assertEqual(record.performance_snapshots[0]["timestamp"], timestamp.isoformat())
        self.assertEqual(record.performance_snapshots[0]["metrics"], metrics)
        self.assertEqual(record.actual_market_regimes, [])
        
        # Add snapshot with regime
        record.add_performance_snapshot(timestamp, metrics, "bullish")
        self.assertEqual(len(record.performance_snapshots), 2)
        self.assertEqual(record.actual_market_regimes, ["bullish"])
    
    def test_set_synthetic_predictions(self):
        """Test setting synthetic predictions."""
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        predictions = {
            "bullish": {
                "variant_b": {"metrics": {"sharpe_ratio": 1.8}},
                "comparison": {"b_is_better": True}
            }
        }
        
        record.set_synthetic_predictions(predictions)
        self.assertEqual(record.synthetic_predictions, predictions)
    
    def test_to_dict_and_from_dict(self):
        """Test serialization to and from dictionary."""
        original = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123",
            approval_date=datetime(2025, 1, 1)
        )
        
        original.add_performance_snapshot(
            datetime(2025, 1, 2),
            {"sharpe_ratio": 1.5},
            "bullish"
        )
        
        original.set_synthetic_predictions({
            "bullish": {"variant_b": {"metrics": {"sharpe_ratio": 1.8}}}
        })
        
        original.verification_results = {"overall_accuracy": 0.8}
        
        # Convert to dict
        data = original.to_dict()
        
        # Verify dictionary has expected keys
        self.assertIn("strategy_id", data)
        self.assertIn("version_id", data)
        self.assertIn("approval_request_id", data)
        self.assertIn("test_id", data)
        self.assertIn("approval_date", data)
        self.assertIn("record_id", data)
        self.assertIn("performance_snapshots", data)
        self.assertIn("synthetic_predictions", data)
        self.assertIn("verification_results", data)
        self.assertIn("actual_market_regimes", data)
        
        # Reconstruct from dict
        reconstructed = performance_verification.StrategyPerformanceRecord.from_dict(data)
        
        # Verify values match
        self.assertEqual(reconstructed.strategy_id, "test_strategy")
        self.assertEqual(reconstructed.version_id, "v1.0")
        self.assertEqual(reconstructed.approval_request_id, "req123")
        self.assertEqual(reconstructed.test_id, "test123")
        self.assertEqual(reconstructed.approval_date.year, 2025)
        self.assertEqual(reconstructed.approval_date.month, 1)
        self.assertEqual(reconstructed.approval_date.day, 1)
        self.assertEqual(len(reconstructed.performance_snapshots), 1)
        self.assertEqual(len(reconstructed.actual_market_regimes), 1)
        self.assertEqual(reconstructed.actual_market_regimes[0], "bullish")
        self.assertEqual(reconstructed.verification_results, {"overall_accuracy": 0.8})


class TestPerformanceVerifier(unittest.TestCase):
    """Test the PerformanceVerifier class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, "verification_data.json")
        
        # Mock components
        self.event_bus_mock = MagicMock()
        self.approval_manager_mock = MagicMock()
        self.ab_test_manager_mock = MagicMock()
        self.synthetic_integration_mock = MagicMock()
        
        # Create patches
        self.event_bus_patch = patch('trading_bot.event_system.EventBus', return_value=self.event_bus_mock)
        self.approval_manager_patch = patch(
            'trading_bot.autonomous.performance_verification.get_approval_workflow_manager',
            return_value=self.approval_manager_mock
        )
        self.ab_test_manager_patch = patch(
            'trading_bot.autonomous.performance_verification.get_ab_test_manager',
            return_value=self.ab_test_manager_mock
        )
        self.synthetic_integration_patch = patch(
            'trading_bot.autonomous.performance_verification.get_synthetic_testing_integration',
            return_value=self.synthetic_integration_mock
        )
        
        # Start patches
        self.event_bus_patch.start()
        self.approval_manager_patch.start()
        self.ab_test_manager_patch.start()
        self.synthetic_integration_patch.start()
        
        # Create the verifier with a temporary storage path
        self.verifier = performance_verification.PerformanceVerifier(storage_path=self.storage_path)
    
    def tearDown(self):
        """Clean up the test environment."""
        # Stop patches
        self.event_bus_patch.stop()
        self.approval_manager_patch.stop()
        self.ab_test_manager_patch.stop()
        self.synthetic_integration_patch.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of the performance verifier."""
        # Verify components are properly assigned
        self.assertEqual(self.verifier.event_bus, self.event_bus_mock)
        self.assertEqual(self.verifier.approval_manager, self.approval_manager_mock)
        self.assertEqual(self.verifier.ab_test_manager, self.ab_test_manager_mock)
        self.assertEqual(self.verifier.synthetic_integration, self.synthetic_integration_mock)
        
        # Verify storage path is set
        self.assertEqual(self.verifier.storage_path, self.storage_path)
        
        # Verify dictionaries are initialized
        self.assertEqual(self.verifier.performance_records, {})
        self.assertIsInstance(self.verifier.metrics, performance_verification.VerificationMetrics)
    
    def test_event_registration(self):
        """Test that events are properly registered."""
        # Verify event handlers are registered
        expected_registrations = [
            call(EventType.APPROVAL_REQUEST_APPROVED, self.verifier._handle_approval),
            call("strategy_performance_update", self.verifier._handle_performance_update)
        ]
        
        # Check expected calls were made
        self.event_bus_mock.register.assert_has_calls(expected_registrations, any_order=True)
    
    def test_handle_approval(self):
        """Test handling of approval events."""
        # Create mock request and test
        mock_request = MagicMock()
        mock_request.strategy_id = "test_strategy"
        mock_request.version_id = "v1.0"
        mock_request.test_id = "test123"
        mock_request.status = ApprovalStatus.APPROVED
        mock_request.review_date = datetime(2025, 1, 1)
        
        mock_test = MagicMock()
        mock_test.metadata = {
            "synthetic_testing_completed": True,
            "synthetic_testing_results": {
                "bullish": {"variant_b": {"metrics": {"sharpe_ratio": 1.8}}}
            }
        }
        
        # Configure mocks to return our test objects
        self.approval_manager_mock.get_request.return_value = mock_request
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create an approval event
        event = Event(
            event_type=EventType.APPROVAL_REQUEST_APPROVED,
            data={"request_id": "req123"}
        )
        
        # Handle the event
        self.verifier._handle_approval(event)
        
        # Verify request was retrieved
        self.approval_manager_mock.get_request.assert_called_with("req123")
        
        # Verify test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_with("test123")
        
        # Verify a record was created
        self.assertEqual(len(self.verifier.performance_records), 1)
        
        # Get the record (extract the key from the dictionary)
        record_id = next(iter(self.verifier.performance_records.keys()))
        record = self.verifier.performance_records[record_id]
        
        # Verify record properties
        self.assertEqual(record.strategy_id, "test_strategy")
        self.assertEqual(record.version_id, "v1.0")
        self.assertEqual(record.test_id, "test123")
        self.assertEqual(record.approval_date, datetime(2025, 1, 1))
        self.assertEqual(record.synthetic_predictions, mock_test.metadata["synthetic_testing_results"])
    
    def test_handle_performance_update(self):
        """Test handling of performance update events."""
        # Create a performance record
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        # Add synthetic predictions
        record.set_synthetic_predictions({
            "bullish": {
                "variant_b": {
                    "metrics": {
                        "sharpe_ratio": 1.8,
                        "max_drawdown": -0.15,
                        "win_rate": 0.65
                    }
                },
                "comparison": {"b_is_better": True}
            }
        })
        
        # Add record to verifier
        self.verifier.performance_records[record.record_id] = record
        
        # Create a performance update event
        event = Event(
            event_type="strategy_performance_update",
            data={
                "strategy_id": "test_strategy",
                "version_id": "v1.0",
                "metrics": {
                    "sharpe_ratio": 1.7,
                    "max_drawdown": -0.18,
                    "win_rate": 0.62,
                    "volatility": 0.01,
                    "returns_30d": 0.06,
                    "max_drawdown_30d": -0.05
                },
                "timestamp": datetime.utcnow()
            }
        )
        
        # Patch detect_current_market_regime to return a regime
        with patch.object(self.verifier, '_detect_current_market_regime', return_value="bullish"):
            # Handle the event
            self.verifier._handle_performance_update(event)
        
        # Verify snapshot was added
        self.assertEqual(len(record.performance_snapshots), 1)
        self.assertEqual(record.actual_market_regimes, ["bullish"])
    
    def test_detect_current_market_regime(self):
        """Test detection of current market regime."""
        # Test bullish regime
        bullish_metrics = {
            "volatility": 0.02,
            "returns_30d": 0.08,
            "max_drawdown_30d": -0.05
        }
        self.assertEqual(
            self.verifier._detect_current_market_regime(bullish_metrics),
            "bullish"
        )
        
        # Test bearish regime
        bearish_metrics = {
            "volatility": 0.02,
            "returns_30d": -0.08,
            "max_drawdown_30d": -0.15
        }
        self.assertEqual(
            self.verifier._detect_current_market_regime(bearish_metrics),
            "bearish"
        )
        
        # Test volatile regime
        volatile_metrics = {
            "volatility": 0.03,
            "returns_30d": 0.0,
            "max_drawdown_30d": -0.10
        }
        self.assertEqual(
            self.verifier._detect_current_market_regime(volatile_metrics),
            "volatile"
        )
        
        # Test sideways regime
        sideways_metrics = {
            "volatility": 0.01,
            "returns_30d": 0.01,
            "max_drawdown_30d": -0.03
        }
        self.assertEqual(
            self.verifier._detect_current_market_regime(sideways_metrics),
            "sideways"
        )
    
    def test_verify_performance(self):
        """Test verification of performance against predictions."""
        # Create a record with synthetic predictions and performance
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        # Add synthetic predictions
        record.set_synthetic_predictions({
            "bullish": {
                "variant_b": {
                    "metrics": {
                        "sharpe_ratio": 1.8,
                        "max_drawdown": -0.15,
                        "win_rate": 0.65
                    }
                },
                "comparison": {"b_is_better": True}
            }
        })
        
        # Add performance snapshots (need at least 30 for verification)
        for i in range(30):
            record.add_performance_snapshot(
                datetime.utcnow() - timedelta(days=30-i),
                {
                    "sharpe_ratio": 1.7,
                    "max_drawdown": -0.18,
                    "win_rate": 0.62
                },
                "bullish"  # All in bullish regime for simplicity
            )
        
        # Verify performance
        self.verifier._verify_performance(record)
        
        # Check that verification results were set
        self.assertIn("primary_regime", record.verification_results)
        self.assertIn("sharpe_accuracy", record.verification_results)
        self.assertIn("drawdown_accuracy", record.verification_results)
        self.assertIn("win_rate_accuracy", record.verification_results)
        self.assertIn("overall_accuracy", record.verification_results)
        self.assertIn("timestamp", record.verification_results)
        
        # Check that overall accuracy is between 0 and 1
        overall_accuracy = record.verification_results["overall_accuracy"]
        self.assertGreaterEqual(overall_accuracy, 0.0)
        self.assertLessEqual(overall_accuracy, 1.0)
    
    def test_get_verification_recommendations(self):
        """Test getting recommendations for improving synthetic parameters."""
        # Initialize metrics with some values
        self.verifier.metrics.correct_predictions = 7
        self.verifier.metrics.total_predictions = 10
        self.verifier.metrics.sharpe_correlation = 0.6
        self.verifier.metrics.drawdown_correlation = 0.8
        self.verifier.metrics.win_rate_correlation = 0.75
        self.verifier.metrics.market_condition_mapping = {
            "bullish": {"count": 5, "correct_predictions": 4},
            "bearish": {"count": 3, "correct_predictions": 1}
        }
        
        # Get recommendations
        recommendations = self.verifier.get_verification_recommendations()
        
        # Verify structure
        self.assertIn("overall_accuracy", recommendations)
        self.assertIn("regime_specific_adjustments", recommendations)
        self.assertIn("parameter_tuning", recommendations)
        self.assertIn("general_recommendations", recommendations)
        
        # Verify correct values
        self.assertEqual(recommendations["overall_accuracy"], 0.7)
        self.assertIn("bullish", recommendations["regime_specific_adjustments"])
        self.assertIn("bearish", recommendations["regime_specific_adjustments"])
        
        # High accuracy regime should not need adjustment
        bullish_adjustments = recommendations["regime_specific_adjustments"]["bullish"]
        self.assertEqual(bullish_adjustments["current_accuracy"], 0.8)
        
        # Low accuracy regime should need adjustment
        bearish_adjustments = recommendations["regime_specific_adjustments"]["bearish"]
        self.assertEqual(bearish_adjustments["current_accuracy"], 1/3)
        self.assertTrue(len(bearish_adjustments["suggested_adjustments"]) > 0)
        
        # Should suggest adjusting sharpe parameters due to low correlation
        self.assertIn("volatility", recommendations["parameter_tuning"])
        
        # Should not suggest adjusting drawdown parameters due to high correlation
        self.assertNotIn("drawdown_modeling", recommendations["parameter_tuning"])
    
    def test_save_and_load(self):
        """Test saving and loading verification data."""
        # Create a record
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        # Add synthetic predictions
        record.set_synthetic_predictions({
            "bullish": {"variant_b": {"metrics": {"sharpe_ratio": 1.8}}}
        })
        
        # Add to verifier
        self.verifier.performance_records[record.record_id] = record
        
        # Update metrics
        self.verifier.metrics.correct_predictions = 7
        self.verifier.metrics.total_predictions = 10
        
        # Save to disk
        self.verifier._save_to_disk()
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.storage_path))
        
        # Create a new verifier that loads the data
        new_verifier = performance_verification.PerformanceVerifier(storage_path=self.storage_path)
        
        # Verify data was loaded
        self.assertEqual(len(new_verifier.performance_records), 1)
        self.assertEqual(new_verifier.metrics.correct_predictions, 7)
        self.assertEqual(new_verifier.metrics.total_predictions, 10)
    
    def test_generate_verification_report(self):
        """Test generating a verification report."""
        # Set up some metrics and records
        self.verifier.metrics.correct_predictions = 7
        self.verifier.metrics.total_predictions = 10
        self.verifier.metrics.sharpe_correlation = 0.6
        self.verifier.metrics.drawdown_correlation = 0.8
        self.verifier.metrics.win_rate_correlation = 0.75
        self.verifier.metrics.verification_timestamps = [datetime.utcnow()]
        self.verifier.metrics.accuracy_over_time = [0.7]
        self.verifier.metrics.market_condition_mapping = {
            "bullish": {"count": 5, "correct_predictions": 4}
        }
        
        # Create a record with verification results
        record = performance_verification.StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123"
        )
        
        record.verification_results = {
            "primary_regime": "bullish",
            "overall_accuracy": 0.8,
            "sharpe_accuracy": 0.85,
            "drawdown_accuracy": 0.75,
            "win_rate_accuracy": 0.8
        }
        
        # Add to verifier
        self.verifier.performance_records[record.record_id] = record
        
        # Generate report
        report = self.verifier.generate_verification_report()
        
        # Verify structure
        self.assertIn("timestamp", report)
        self.assertIn("overall_metrics", report)
        self.assertIn("regime_accuracy", report)
        self.assertIn("recommendations", report)
        self.assertIn("verification_history", report)
        self.assertIn("strategy_details", report)
        
        # Verify overall metrics
        overall_metrics = report["overall_metrics"]
        self.assertEqual(overall_metrics["prediction_accuracy"], 0.7)
        self.assertEqual(overall_metrics["sharpe_correlation"], 0.6)
        self.assertEqual(overall_metrics["drawdown_correlation"], 0.8)
        self.assertEqual(overall_metrics["win_rate_correlation"], 0.75)
        
        # Verify regime accuracy
        self.assertIn("bullish", report["regime_accuracy"])
        self.assertEqual(report["regime_accuracy"]["bullish"]["accuracy"], 0.8)
        
        # Verify strategy details
        self.assertEqual(len(report["strategy_details"]), 1)
        strategy_details = list(report["strategy_details"].values())[0]
        self.assertEqual(strategy_details["strategy_id"], "test_strategy")
        self.assertEqual(strategy_details["version_id"], "v1.0")
        self.assertEqual(strategy_details["primary_regime"], "bullish")
        self.assertEqual(strategy_details["overall_accuracy"], 0.8)
    
    def test_singleton_accessor(self):
        """Test the singleton accessor function."""
        # Reset singleton for testing
        performance_verification._performance_verifier = None
        
        # Access the singleton
        with patch('trading_bot.autonomous.performance_verification.PerformanceVerifier',
                  return_value=MagicMock()) as mock_init:
            # First call should create a new instance
            instance1 = performance_verification.get_performance_verifier()
            mock_init.assert_called_once()
            
            # Second call should return the same instance
            instance2 = performance_verification.get_performance_verifier()
            # Still only called once
            mock_init.assert_called_once()
            
            # Both variables should reference the same instance
            self.assertEqual(instance1, instance2)


if __name__ == "__main__":
    unittest.main()
