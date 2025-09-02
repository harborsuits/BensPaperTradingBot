#!/usr/bin/env python3
"""
Minimal Core Tests for Performance Verification

This file focuses on testing only the core data structures and logic
of the performance verification system without external dependencies.
It follows our philosophy of building incrementally on working solutions.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
import json
from datetime import datetime, timedelta
from enum import Enum
import tempfile

# Add parent directory to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create bare minimum mocks to run tests
class MarketRegimeType(Enum):
    """Mock of the MarketRegimeType enum."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

# Direct import from our verification module
from trading_bot.autonomous.performance_verification import (
    VerificationMetrics,
    StrategyPerformanceRecord
)


class TestVerificationMetrics(unittest.TestCase):
    """Test the core VerificationMetrics data structure."""
    
    def test_initialization(self):
        """Test initialization of verification metrics."""
        metrics = VerificationMetrics()
        
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
        metrics = VerificationMetrics()
        
        # Test with no predictions
        self.assertEqual(metrics.prediction_accuracy, 0.0)
        
        # Test with predictions
        metrics.correct_predictions = 7
        metrics.total_predictions = 10
        self.assertEqual(metrics.prediction_accuracy, 0.7)
    
    def test_to_dict_and_from_dict(self):
        """Test serialization to and from dictionary."""
        original = VerificationMetrics()
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
        reconstructed = VerificationMetrics.from_dict(data)
        
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
    """Test the core StrategyPerformanceRecord data structure."""
    
    def test_initialization(self):
        """Test initialization of performance record."""
        record = StrategyPerformanceRecord(
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
        record = StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123",
            approval_date=datetime(2025, 1, 1)
        )
        
        timestamp = datetime(2025, 1, 2)
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
        record = StrategyPerformanceRecord(
            strategy_id="test_strategy",
            version_id="v1.0",
            approval_request_id="req123",
            test_id="test123",
            approval_date=datetime(2025, 1, 1)
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
        original = StrategyPerformanceRecord(
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
        reconstructed = StrategyPerformanceRecord.from_dict(data)
        
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


if __name__ == "__main__":
    unittest.main()
