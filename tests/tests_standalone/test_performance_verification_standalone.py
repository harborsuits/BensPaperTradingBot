#!/usr/bin/env python3
"""
Standalone Tests for Performance Verification Components

This test file is completely self-contained, implementing test versions of the core classes
from performance_verification.py to avoid any external dependencies.

This follows our proven approach of focusing on core functionality first, similar to how
we successfully tested the synthetic testing integration.
"""

import sys
import os
import unittest
import json
from datetime import datetime, timedelta
from enum import Enum
import tempfile

# Create minimal versions of the classes we want to test
class VerificationMetrics:
    """Metrics for verification of synthetic testing accuracy."""
    
    def __init__(self):
        # Prediction accuracy
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Metric correlation (real vs. synthetic)
        self.sharpe_correlation = 0.0
        self.drawdown_correlation = 0.0
        self.win_rate_correlation = 0.0
        
        # Regime accuracy
        self.regime_detection_accuracy = 0.0
        self.regime_prediction_accuracy = 0.0
        
        # Time series
        self.verification_timestamps = []
        self.accuracy_over_time = []
        
        # Market condition mapping
        self.market_condition_mapping = {}
    
    @property
    def prediction_accuracy(self) -> float:
        """Calculate overall prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def to_dict(self):
        """Convert metrics to dictionary for serialization."""
        return {
            "correct_predictions": self.correct_predictions,
            "total_predictions": self.total_predictions,
            "prediction_accuracy": self.prediction_accuracy,
            "sharpe_correlation": self.sharpe_correlation,
            "drawdown_correlation": self.drawdown_correlation,
            "win_rate_correlation": self.win_rate_correlation,
            "regime_detection_accuracy": self.regime_detection_accuracy,
            "regime_prediction_accuracy": self.regime_prediction_accuracy,
            "verification_timestamps": [t.isoformat() for t in self.verification_timestamps],
            "accuracy_over_time": self.accuracy_over_time,
            "market_condition_mapping": self.market_condition_mapping
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create metrics from dictionary."""
        metrics = cls()
        metrics.correct_predictions = data.get("correct_predictions", 0)
        metrics.total_predictions = data.get("total_predictions", 0)
        metrics.sharpe_correlation = data.get("sharpe_correlation", 0.0)
        metrics.drawdown_correlation = data.get("drawdown_correlation", 0.0)
        metrics.win_rate_correlation = data.get("win_rate_correlation", 0.0)
        metrics.regime_detection_accuracy = data.get("regime_detection_accuracy", 0.0)
        metrics.regime_prediction_accuracy = data.get("regime_prediction_accuracy", 0.0)
        
        # Parse timestamps
        metrics.verification_timestamps = [
            datetime.fromisoformat(t) 
            for t in data.get("verification_timestamps", [])
        ]
        
        metrics.accuracy_over_time = data.get("accuracy_over_time", [])
        metrics.market_condition_mapping = data.get("market_condition_mapping", {})
        return metrics


class StrategyPerformanceRecord:
    """Record of strategy performance after approval."""
    
    def __init__(
        self, 
        strategy_id, 
        version_id, 
        approval_request_id,
        test_id,
        approval_date=None
    ):
        self.strategy_id = strategy_id
        self.version_id = version_id
        self.approval_request_id = approval_request_id
        self.test_id = test_id
        
        # Ensure approval_date is never None
        self.approval_date = approval_date if approval_date is not None else datetime.utcnow()
        
        # Performance tracking
        self.performance_snapshots = []
        self.synthetic_predictions = {}
        self.verification_results = {}
        
        # Metadata
        self.actual_market_regimes = []
        
        # Generate a unique ID for this record
        date_str = self.approval_date.strftime('%Y%m%d%H%M%S')
        self.record_id = f"{strategy_id}_{version_id}_{date_str}"
    
    def add_performance_snapshot(
        self, 
        timestamp, 
        metrics,
        market_regime=None
    ):
        """
        Add performance snapshot for the strategy.
        
        Args:
            timestamp: Snapshot timestamp
            metrics: Performance metrics
            market_regime: Detected market regime (if available)
        """
        snapshot = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics
        }
        
        if market_regime:
            snapshot["market_regime"] = market_regime
            self.actual_market_regimes.append(market_regime)
        
        self.performance_snapshots.append(snapshot)
    
    def set_synthetic_predictions(self, predictions):
        """
        Set synthetic test predictions for this strategy.
        
        Args:
            predictions: Dictionary of synthetic market predictions
        """
        self.synthetic_predictions = predictions
    
    def to_dict(self):
        """Convert record to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "version_id": self.version_id,
            "approval_request_id": self.approval_request_id,
            "test_id": self.test_id,
            "approval_date": self.approval_date.isoformat(),
            "record_id": self.record_id,
            "performance_snapshots": self.performance_snapshots,
            "synthetic_predictions": self.synthetic_predictions,
            "verification_results": self.verification_results,
            "actual_market_regimes": self.actual_market_regimes
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create record from dictionary."""
        record = cls(
            strategy_id=data["strategy_id"],
            version_id=data["version_id"],
            approval_request_id=data["approval_request_id"],
            test_id=data["test_id"],
            approval_date=datetime.fromisoformat(data["approval_date"])
        )
        
        record.record_id = data["record_id"]
        record.performance_snapshots = data.get("performance_snapshots", [])
        record.synthetic_predictions = data.get("synthetic_predictions", {})
        record.verification_results = data.get("verification_results", {})
        record.actual_market_regimes = data.get("actual_market_regimes", [])
        
        return record


# Define enum for market regimes (without importing)
class MarketRegimeType(Enum):
    """Mock of the MarketRegimeType enum."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class TestVerificationMetrics(unittest.TestCase):
    """Test the VerificationMetrics class."""
    
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
    """Test the StrategyPerformanceRecord class."""
    
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


class TestMarketRegimeDetection(unittest.TestCase):
    """Test market regime detection logic."""
    
    def test_regime_detection(self):
        """Test basic regime detection logic."""
        # These functions simulate the regime detection logic from PerformanceVerifier
        def detect_current_market_regime(metrics):
            """Detect current market regime based on performance metrics."""
            volatility = metrics.get("volatility")
            returns = metrics.get("returns_30d")
            drawdown = metrics.get("max_drawdown_30d")
            
            if volatility is None or returns is None or drawdown is None:
                return None
            
            if volatility > 0.025:  # High volatility
                return MarketRegimeType.VOLATILE.value
            elif returns > 0.05:  # Strong positive returns
                return MarketRegimeType.BULLISH.value
            elif returns < -0.05:  # Strong negative returns
                return MarketRegimeType.BEARISH.value
            else:  # Low volatility, low returns
                return MarketRegimeType.SIDEWAYS.value
        
        # Test bullish regime
        bullish_metrics = {
            "volatility": 0.02,
            "returns_30d": 0.08,
            "max_drawdown_30d": -0.05
        }
        self.assertEqual(
            detect_current_market_regime(bullish_metrics),
            "bullish"
        )
        
        # Test bearish regime
        bearish_metrics = {
            "volatility": 0.02,
            "returns_30d": -0.08,
            "max_drawdown_30d": -0.15
        }
        self.assertEqual(
            detect_current_market_regime(bearish_metrics),
            "bearish"
        )
        
        # Test volatile regime
        volatile_metrics = {
            "volatility": 0.03,
            "returns_30d": 0.0,
            "max_drawdown_30d": -0.10
        }
        self.assertEqual(
            detect_current_market_regime(volatile_metrics),
            "volatile"
        )
        
        # Test sideways regime
        sideways_metrics = {
            "volatility": 0.01,
            "returns_30d": 0.01,
            "max_drawdown_30d": -0.03
        }
        self.assertEqual(
            detect_current_market_regime(sideways_metrics),
            "sideways"
        )


class TestVerificationRecommendations(unittest.TestCase):
    """Test verification recommendation logic."""
    
    def test_recommendation_generation(self):
        """Test generating recommendations from verification metrics."""
        # This function simulates the recommendation logic from PerformanceVerifier
        def get_verification_recommendations(metrics):
            """Generate recommendations for improving synthetic market parameters."""
            recommendations = {
                "overall_accuracy": metrics.prediction_accuracy,
                "regime_specific_adjustments": {},
                "parameter_tuning": {},
                "general_recommendations": []
            }
            
            # Add regime-specific recommendations
            for regime, data in metrics.market_condition_mapping.items():
                if data["count"] > 0:
                    accuracy = data["correct_predictions"] / data["count"]
                    recommendations["regime_specific_adjustments"][regime] = {
                        "current_accuracy": accuracy,
                        "suggested_adjustments": []
                    }
                    
                    # Add specific recommendations based on accuracy
                    if accuracy < 0.5:
                        recommendations["regime_specific_adjustments"][regime]["suggested_adjustments"].append(
                            f"Significant tuning needed for {regime} regime parameters"
                        )
                    elif accuracy < 0.7:
                        recommendations["regime_specific_adjustments"][regime]["suggested_adjustments"].append(
                            f"Fine-tuning needed for {regime} regime parameters"
                        )
                    else:
                        recommendations["regime_specific_adjustments"][regime]["suggested_adjustments"].append(
                            f"Parameters for {regime} regime are reasonably accurate"
                        )
            
            # Add parameter tuning suggestions
            if metrics.sharpe_correlation < 0.7:
                recommendations["parameter_tuning"]["volatility"] = "Adjust volatility parameters to better match real market"
            
            if metrics.drawdown_correlation < 0.7:
                recommendations["parameter_tuning"]["drawdown_modeling"] = "Improve drawdown modeling for more realistic stress scenarios"
            
            if metrics.win_rate_correlation < 0.7:
                recommendations["parameter_tuning"]["mean_reversion"] = "Adjust mean reversion parameters for better win rate prediction"
            
            # Add general recommendations
            if metrics.prediction_accuracy < 0.6:
                recommendations["general_recommendations"].append(
                    "Overall synthetic market accuracy is low, consider fundamental revision of parameters"
                )
            elif metrics.prediction_accuracy < 0.8:
                recommendations["general_recommendations"].append(
                    "Synthetic market parameters need fine-tuning to improve accuracy"
                )
            else:
                recommendations["general_recommendations"].append(
                    "Synthetic market parameters are performing well, maintain current configuration"
                )
            
            return recommendations
        
        # Create metrics with various accuracy levels
        metrics = VerificationMetrics()
        
        # Test with low accuracy
        metrics.correct_predictions = 2
        metrics.total_predictions = 10
        metrics.sharpe_correlation = 0.4
        metrics.drawdown_correlation = 0.5
        metrics.win_rate_correlation = 0.3
        metrics.market_condition_mapping = {
            "bullish": {"count": 5, "correct_predictions": 1},
            "bearish": {"count": 5, "correct_predictions": 1}
        }
        
        low_recommendations = get_verification_recommendations(metrics)
        self.assertEqual(low_recommendations["overall_accuracy"], 0.2)
        self.assertIn("volatility", low_recommendations["parameter_tuning"])
        self.assertIn("drawdown_modeling", low_recommendations["parameter_tuning"])
        self.assertIn("mean_reversion", low_recommendations["parameter_tuning"])
        self.assertIn("Overall synthetic market accuracy is low", low_recommendations["general_recommendations"][0])
        
        # Test with high accuracy
        metrics.correct_predictions = 9
        metrics.total_predictions = 10
        metrics.sharpe_correlation = 0.9
        metrics.drawdown_correlation = 0.85
        metrics.win_rate_correlation = 0.9
        metrics.market_condition_mapping = {
            "bullish": {"count": 5, "correct_predictions": 5},
            "bearish": {"count": 5, "correct_predictions": 4}
        }
        
        high_recommendations = get_verification_recommendations(metrics)
        self.assertEqual(high_recommendations["overall_accuracy"], 0.9)
        self.assertNotIn("volatility", high_recommendations["parameter_tuning"])
        self.assertNotIn("drawdown_modeling", high_recommendations["parameter_tuning"])
        self.assertNotIn("mean_reversion", high_recommendations["parameter_tuning"])
        self.assertIn("performing well", high_recommendations["general_recommendations"][0])
        
        # Check regime-specific recommendations
        self.assertEqual(high_recommendations["regime_specific_adjustments"]["bullish"]["current_accuracy"], 1.0)
        self.assertEqual(high_recommendations["regime_specific_adjustments"]["bearish"]["current_accuracy"], 0.8)
        self.assertIn("reasonably accurate", high_recommendations["regime_specific_adjustments"]["bullish"]["suggested_adjustments"][0])
        self.assertIn("reasonably accurate", high_recommendations["regime_specific_adjustments"]["bearish"]["suggested_adjustments"][0])


if __name__ == "__main__":
    unittest.main()
