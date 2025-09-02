#!/usr/bin/env python3
"""
Performance Verification System

This module tracks post-approval strategy performance and compares it against
synthetic test predictions to verify the accuracy of the testing framework.

It provides:
1. Post-approval strategy performance tracking
2. Comparison between real and synthetic metrics
3. Accuracy measurements for synthetic predictions
4. Feedback mechanisms to improve synthetic market parameters
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from enum import Enum

# Import approval workflow components
from trading_bot.autonomous.approval_workflow import (
    get_approval_workflow_manager, ApprovalStatus, ApprovalRequest
)

# Import A/B testing components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)
from trading_bot.autonomous.ab_testing_manager import (
    get_ab_test_manager
)

# Import synthetic market components
from trading_bot.autonomous.synthetic_market_generator import (
    MarketRegimeType
)
from trading_bot.autonomous.synthetic_testing_integration import (
    get_synthetic_testing_integration
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationMetrics':
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
        strategy_id: str, 
        version_id: str, 
        approval_request_id: str,
        test_id: str,
        approval_date: datetime = None
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
        timestamp: datetime, 
        metrics: Dict[str, Any],
        market_regime: Optional[str] = None
    ) -> None:
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
    
    def set_synthetic_predictions(self, predictions: Dict[str, Any]) -> None:
        """
        Set synthetic test predictions for this strategy.
        
        Args:
            predictions: Dictionary of synthetic market predictions
        """
        self.synthetic_predictions = predictions
    
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyPerformanceRecord':
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


class PerformanceVerifier:
    """
    Verifies the accuracy of synthetic testing by comparing real 
    strategy performance to synthetic predictions.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the performance verifier.
        
        Args:
            storage_path: Path to store verification data
        """
        # Set default storage path
        if storage_path is None:
            home_dir = str(Path.home())
            self.storage_path = os.path.join(home_dir, ".trading_bot", "verification_data.json")
        else:
            self.storage_path = storage_path
        
        # Core components
        self.event_bus = EventBus()
        self.approval_manager = get_approval_workflow_manager()
        self.ab_test_manager = get_ab_test_manager()
        self.synthetic_integration = get_synthetic_testing_integration()
        
        # Performance records
        self.performance_records = {}
        
        # Overall verification metrics
        self.metrics = VerificationMetrics()
        
        # Load existing data
        self._load_from_disk()
        
        # Register for events
        self._register_event_handlers()
        
        logger.info(f"Performance Verifier initialized with storage at {self.storage_path}")
    
    def _register_event_handlers(self) -> None:
        """Register for relevant events."""
        # Track when requests are approved
        self.event_bus.register(
            EventType.APPROVAL_REQUEST_APPROVED,
            self._handle_approval
        )
        
        # Track when new performance data is available
        self.event_bus.register(
            "strategy_performance_update",
            self._handle_performance_update
        )
    
    def _handle_approval(self, event: Event) -> None:
        """
        Handle approval events by starting to track performance.
        
        Args:
            event: The approval event
        """
        # Extract approval data
        request_id = event.data.get("request_id")
        if not request_id:
            return
        
        # Get request details
        request = self.approval_manager.get_request(request_id)
        if not request or request.status != ApprovalStatus.APPROVED:
            return
        
        # Get test details to retrieve synthetic predictions
        test = self.ab_test_manager.get_test(request.test_id)
        if not test:
            return
        
        # Create performance record
        record = StrategyPerformanceRecord(
            strategy_id=request.strategy_id,
            version_id=request.version_id,
            approval_request_id=request_id,
            test_id=request.test_id,
            approval_date=request.review_date if request.review_date else datetime.utcnow()
        )
        
        # Get synthetic predictions if available
        if test.metadata.get("synthetic_testing_completed", False):
            predictions = test.metadata.get("synthetic_testing_results", {})
            record.set_synthetic_predictions(predictions)
        
        # Add to tracking
        self.performance_records[record.record_id] = record
        
        # Save to disk
        self._save_to_disk()
        
        logger.info(f"Started tracking performance for {record.strategy_id} v{record.version_id}")
    
    def _handle_performance_update(self, event: Event) -> None:
        """
        Handle performance update events.
        
        Args:
            event: The performance update event
        """
        # Extract performance data
        strategy_id = event.data.get("strategy_id")
        version_id = event.data.get("version_id")
        metrics = event.data.get("metrics")
        timestamp = event.data.get("timestamp")
        
        if not all([strategy_id, version_id, metrics, timestamp]):
            return
        
        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Find the record for this strategy version
        for record_id, record in self.performance_records.items():
            if record.strategy_id == strategy_id and record.version_id == version_id:
                # Detect current market regime if possible
                market_regime = self._detect_current_market_regime(metrics)
                
                # Add performance snapshot
                record.add_performance_snapshot(
                    timestamp=timestamp,
                    metrics=metrics,
                    market_regime=market_regime
                )
                
                # Verify against synthetic predictions
                if record.synthetic_predictions:
                    self._verify_performance(record)
                
                # Save updated data
                self._save_to_disk()
                
                logger.info(f"Updated performance for {strategy_id} v{version_id}")
                break
    
    def _detect_current_market_regime(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Detect the current market regime based on performance metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Detected market regime or None
        """
        # This is a simplified implementation - in a real system, this would use more
        # sophisticated analysis of market conditions using volatility, trend, etc.
        
        # Extract key metrics that help identify regimes
        volatility = metrics.get("volatility")
        returns = metrics.get("returns_30d")
        drawdown = metrics.get("max_drawdown_30d")
        
        if volatility is None or returns is None or drawdown is None:
            return None
        
        # Simple regime detection based on volatility and returns
        if volatility > 0.025:  # High volatility
            return MarketRegimeType.VOLATILE.value
        elif returns > 0.05:  # Strong positive returns
            return MarketRegimeType.BULLISH.value
        elif returns < -0.05:  # Strong negative returns
            return MarketRegimeType.BEARISH.value
        else:  # Low volatility, low returns
            return MarketRegimeType.SIDEWAYS.value
    
    def _verify_performance(self, record: StrategyPerformanceRecord) -> None:
        """
        Verify strategy performance against synthetic predictions.
        
        Args:
            record: The performance record to verify
        """
        # Need at least 30 days of data for meaningful verification
        if len(record.performance_snapshots) < 30:
            logger.info(f"Not enough data to verify {record.strategy_id} v{record.version_id} yet")
            return
        
        # Extract latest performance metrics for verification
        latest_snapshot = record.performance_snapshots[-1]
        real_metrics = latest_snapshot["metrics"]
        
        # Extract predicted metrics from synthetic testing
        synthetic_metrics = {}
        regime_predictions = {}
        
        # Collect regime-specific predictions
        for regime, data in record.synthetic_predictions.items():
            if "variant_b" in data and "metrics" in data["variant_b"]:
                synthetic_metrics[regime] = data["variant_b"]["metrics"]
                
                if "comparison" in data:
                    regime_predictions[regime] = data["comparison"].get("b_is_better", False)
        
        # Determine which regime's predictions to compare against
        detected_regimes = record.actual_market_regimes[-30:]  # Last 30 days
        primary_regime = max(set(detected_regimes), key=detected_regimes.count)
        
        # Calculate verification results
        verification = {}
        
        # Compare real metrics to predicted metrics for the primary regime
        if primary_regime in synthetic_metrics:
            predicted = synthetic_metrics[primary_regime]
            
            # Calculate metric differences
            sharpe_diff = abs(real_metrics.get("sharpe_ratio", 0) - predicted.get("sharpe_ratio", 0))
            drawdown_diff = abs(real_metrics.get("max_drawdown", 0) - predicted.get("max_drawdown", 0))
            win_rate_diff = abs(real_metrics.get("win_rate", 0) - predicted.get("win_rate", 0))
            
            # Calculate accuracy
            # Lower difference = higher accuracy (1.0 - normalized_diff)
            sharpe_accuracy = max(0, 1.0 - (sharpe_diff / max(abs(real_metrics.get("sharpe_ratio", 1)), 0.001)))
            drawdown_accuracy = max(0, 1.0 - (drawdown_diff / max(abs(real_metrics.get("max_drawdown", 0.1)), 0.001)))
            win_rate_accuracy = max(0, 1.0 - (win_rate_diff / max(real_metrics.get("win_rate", 0.1), 0.001)))
            
            # Overall prediction accuracy
            overall_accuracy = (sharpe_accuracy + drawdown_accuracy + win_rate_accuracy) / 3
            
            # Store verification results
            verification = {
                "primary_regime": primary_regime,
                "sharpe_accuracy": sharpe_accuracy,
                "drawdown_accuracy": drawdown_accuracy,
                "win_rate_accuracy": win_rate_accuracy,
                "overall_accuracy": overall_accuracy,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update global metrics
            self._update_global_metrics(
                real_metrics=real_metrics,
                predicted=predicted,
                primary_regime=primary_regime,
                detected_regimes=detected_regimes,
                verification=verification
            )
        
        # Save verification results
        record.verification_results = verification
        
        logger.info(
            f"Verified {record.strategy_id} v{record.version_id} performance. "
            f"Accuracy: {verification.get('overall_accuracy', 0):.2f}"
        )
    
    def _update_global_metrics(
        self, 
        real_metrics: Dict[str, Any],
        predicted: Dict[str, Any],
        primary_regime: str,
        detected_regimes: List[str],
        verification: Dict[str, Any]
    ) -> None:
        """
        Update global verification metrics.
        
        Args:
            real_metrics: Actual performance metrics
            predicted: Predicted metrics from synthetic testing
            primary_regime: Primary detected regime
            detected_regimes: All detected regimes
            verification: Verification results
        """
        # Update correlations if we have new data
        if 'sharpe_ratio' in real_metrics and 'sharpe_ratio' in predicted:
            # This is a simple update - in a real system, we would track all values
            # and periodically recalculate proper correlations
            new_corr = 0.5 * self.metrics.sharpe_correlation + 0.5 * verification.get("sharpe_accuracy", 0)
            self.metrics.sharpe_correlation = new_corr
        
        if 'max_drawdown' in real_metrics and 'max_drawdown' in predicted:
            new_corr = 0.5 * self.metrics.drawdown_correlation + 0.5 * verification.get("drawdown_accuracy", 0)
            self.metrics.drawdown_correlation = new_corr
        
        if 'win_rate' in real_metrics and 'win_rate' in predicted:
            new_corr = 0.5 * self.metrics.win_rate_correlation + 0.5 * verification.get("win_rate_accuracy", 0)
            self.metrics.win_rate_correlation = new_corr
        
        # Update prediction accuracy time series
        self.metrics.verification_timestamps.append(datetime.utcnow())
        self.metrics.accuracy_over_time.append(verification.get("overall_accuracy", 0))
        
        # Update regime accuracy
        # Track how many predictions were correct for each regime
        self.metrics.total_predictions += 1
        if verification.get("overall_accuracy", 0) > 0.7:
            self.metrics.correct_predictions += 1
        
        # Update market condition mapping
        if primary_regime not in self.metrics.market_condition_mapping:
            self.metrics.market_condition_mapping[primary_regime] = {
                "count": 0,
                "correct_predictions": 0
            }
        
        self.metrics.market_condition_mapping[primary_regime]["count"] += 1
        if verification.get("overall_accuracy", 0) > 0.7:
            self.metrics.market_condition_mapping[primary_regime]["correct_predictions"] += 1
    
    def get_verification_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for improving synthetic market parameters.
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "overall_accuracy": self.metrics.prediction_accuracy,
            "regime_specific_adjustments": {},
            "parameter_tuning": {},
            "general_recommendations": []
        }
        
        # Add regime-specific recommendations
        for regime, data in self.metrics.market_condition_mapping.items():
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
        if self.metrics.sharpe_correlation < 0.7:
            recommendations["parameter_tuning"]["volatility"] = "Adjust volatility parameters to better match real market"
        
        if self.metrics.drawdown_correlation < 0.7:
            recommendations["parameter_tuning"]["drawdown_modeling"] = "Improve drawdown modeling for more realistic stress scenarios"
        
        if self.metrics.win_rate_correlation < 0.7:
            recommendations["parameter_tuning"]["mean_reversion"] = "Adjust mean reversion parameters for better win rate prediction"
        
        # Add general recommendations
        if self.metrics.prediction_accuracy < 0.6:
            recommendations["general_recommendations"].append(
                "Overall synthetic market accuracy is low, consider fundamental revision of parameters"
            )
        elif self.metrics.prediction_accuracy < 0.8:
            recommendations["general_recommendations"].append(
                "Synthetic market parameters need fine-tuning to improve accuracy"
            )
        else:
            recommendations["general_recommendations"].append(
                "Synthetic market parameters are performing well, maintain current configuration"
            )
        
        return recommendations
    
    def _save_to_disk(self) -> None:
        """Save verification data to disk."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "performance_records": {
                    rid: record.to_dict() 
                    for rid, record in self.performance_records.items()
                },
                "metrics": self.metrics.to_dict(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved verification data to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving verification data: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load verification data from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Load performance records
                self.performance_records = {
                    rid: StrategyPerformanceRecord.from_dict(rdata)
                    for rid, rdata in data.get("performance_records", {}).items()
                }
                
                # Load metrics
                if "metrics" in data:
                    self.metrics = VerificationMetrics.from_dict(data["metrics"])
                
                logger.info(f"Loaded verification data from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error loading verification data: {str(e)}")
            # Initialize with empty data
            self.performance_records = {}
            self.metrics = VerificationMetrics()
    
    def generate_verification_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive verification report.
        
        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_metrics": {
                "prediction_accuracy": self.metrics.prediction_accuracy,
                "sharpe_correlation": self.metrics.sharpe_correlation,
                "drawdown_correlation": self.metrics.drawdown_correlation,
                "win_rate_correlation": self.metrics.win_rate_correlation
            },
            "regime_accuracy": {
                regime: {
                    "count": data["count"],
                    "accuracy": data["correct_predictions"] / data["count"] if data["count"] > 0 else 0
                }
                for regime, data in self.metrics.market_condition_mapping.items()
            },
            "recommendations": self.get_verification_recommendations(),
            "verification_history": {
                "timestamps": [t.isoformat() for t in self.metrics.verification_timestamps[-50:]],
                "accuracy": self.metrics.accuracy_over_time[-50:]
            },
            "strategy_details": {}
        }
        
        # Add details for the most recent strategies (up to 10)
        sorted_records = sorted(
            self.performance_records.values(),
            key=lambda r: r.approval_date,
            reverse=True
        )
        
        for record in sorted_records[:10]:
            if record.verification_results:
                report["strategy_details"][record.record_id] = {
                    "strategy_id": record.strategy_id,
                    "version_id": record.version_id,
                    "approval_date": record.approval_date.isoformat(),
                    "primary_regime": record.verification_results.get("primary_regime"),
                    "overall_accuracy": record.verification_results.get("overall_accuracy"),
                    "metric_accuracy": {
                        "sharpe": record.verification_results.get("sharpe_accuracy"),
                        "drawdown": record.verification_results.get("drawdown_accuracy"),
                        "win_rate": record.verification_results.get("win_rate_accuracy")
                    }
                }
        
        return report


# Singleton instance
_performance_verifier = None


def get_performance_verifier() -> PerformanceVerifier:
    """
    Get the singleton instance of the performance verifier.
    
    Returns:
        PerformanceVerifier instance
    """
    global _performance_verifier
    
    if _performance_verifier is None:
        _performance_verifier = PerformanceVerifier()
    
    return _performance_verifier


if __name__ == "__main__":
    # Simple test
    verifier = get_performance_verifier()
    report = verifier.generate_verification_report()
    print(json.dumps(report, indent=2))
