"""
Unit tests for Regime Performance Tracker
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.analytics.market_regime.performance import RegimePerformanceTracker
from trading_bot.analytics.market_regime.detector import MarketRegimeType

class TestRegimePerformanceTracker(unittest.TestCase):
    """Test cases for RegimePerformanceTracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for performance files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.config = {
            "performance_dir": self.temp_dir.name,
            "tracked_metrics": ["win_rate", "profit_factor", "sharpe_ratio", "expectancy", "returns"],
            "max_metric_history": 10,
            "max_time_series": 20
        }
        
        # Create tracker
        self.tracker = RegimePerformanceTracker(self.config)
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test performance data"""
        self.test_metrics = {
            "win_rate": 0.65,
            "profit_factor": 2.1,
            "sharpe_ratio": 1.8,
            "expectancy": 0.5,
            "returns": 0.02
        }
        
        # Create multiple metrics samples
        self.multi_metrics = [
            {
                "win_rate": 0.60,
                "profit_factor": 1.9,
                "sharpe_ratio": 1.5,
                "expectancy": 0.4,
                "returns": 0.015
            },
            {
                "win_rate": 0.65,
                "profit_factor": 2.1,
                "sharpe_ratio": 1.8,
                "expectancy": 0.5,
                "returns": 0.02
            },
            {
                "win_rate": 0.70,
                "profit_factor": 2.3,
                "sharpe_ratio": 2.0,
                "expectancy": 0.6,
                "returns": 0.025
            }
        ]
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.tracker.config, self.config)
        self.assertEqual(self.tracker.performance_dir, self.temp_dir.name)
        self.assertEqual(self.tracker.tracked_metrics, self.config["tracked_metrics"])
    
    def test_update_performance(self):
        """Test updating performance metrics"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Update performance
        self.tracker.update_performance(
            strategy_id, regime_type, self.test_metrics, "AAPL", "1d"
        )
        
        # Verify metrics were stored
        self.assertIn(strategy_id, self.tracker.performance_history)
        self.assertIn(regime_type, self.tracker.performance_history[strategy_id])
        
        # Check each metric
        for metric_name, value in self.test_metrics.items():
            if metric_name in self.tracker.tracked_metrics:
                self.assertIn(metric_name, self.tracker.performance_history[strategy_id][regime_type])
                stored_values = self.tracker.performance_history[strategy_id][regime_type][metric_name]
                self.assertEqual(len(stored_values), 1)
                self.assertEqual(stored_values[0], value)
        
        # Verify time series data was stored
        self.assertIn(strategy_id, self.tracker.time_series_data)
        self.assertIn(regime_type, self.tracker.time_series_data[strategy_id])
        self.assertEqual(len(self.tracker.time_series_data[strategy_id][regime_type]), 1)
        
        ts_entry = self.tracker.time_series_data[strategy_id][regime_type][0]
        self.assertEqual(ts_entry["symbol"], "AAPL")
        self.assertEqual(ts_entry["timeframe"], "1d")
        
        # Verify recent performance was stored
        self.assertIn(strategy_id, self.tracker.recent_performance)
        self.assertIn("AAPL_1d", self.tracker.recent_performance[strategy_id])
        self.assertEqual(len(self.tracker.recent_performance[strategy_id]["AAPL_1d"]), 1)
        
        # Verify file was saved
        expected_file = os.path.join(self.temp_dir.name, f"performance_{strategy_id}.json")
        self.assertTrue(os.path.exists(expected_file))
    
    def test_update_multiple_metrics(self):
        """Test updating multiple sets of metrics"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Update with multiple metrics
        for metrics in self.multi_metrics:
            self.tracker.update_performance(
                strategy_id, regime_type, metrics, "AAPL", "1d"
            )
        
        # Verify all metrics were stored
        for metric_name in self.test_metrics:
            if metric_name in self.tracker.tracked_metrics:
                stored_values = self.tracker.performance_history[strategy_id][regime_type][metric_name]
                self.assertEqual(len(stored_values), len(self.multi_metrics))
                
                # Verify values match
                for i, metrics in enumerate(self.multi_metrics):
                    self.assertEqual(stored_values[i], metrics[metric_name])
        
        # Verify time series has multiple entries
        self.assertEqual(
            len(self.tracker.time_series_data[strategy_id][regime_type]), 
            len(self.multi_metrics)
        )
    
    def test_get_performance_by_regime(self):
        """Test getting performance statistics by regime"""
        strategy_id = "test_strategy"
        
        # Add performance data for multiple regimes
        for regime in [MarketRegimeType.TRENDING_UP, MarketRegimeType.RANGE_BOUND, MarketRegimeType.VOLATILE]:
            for metrics in self.multi_metrics:
                self.tracker.update_performance(
                    strategy_id, regime, metrics, "AAPL", "1d"
                )
        
        # Get performance by regime
        performance = self.tracker.get_performance_by_regime(strategy_id)
        
        # Verify all regimes are included
        self.assertIn(MarketRegimeType.TRENDING_UP, performance)
        self.assertIn(MarketRegimeType.RANGE_BOUND, performance)
        self.assertIn(MarketRegimeType.VOLATILE, performance)
        
        # Check statistics for each regime
        for regime in [MarketRegimeType.TRENDING_UP, MarketRegimeType.RANGE_BOUND, MarketRegimeType.VOLATILE]:
            regime_stats = performance[regime]
            
            # Check basic stats are calculated
            self.assertIn("win_rate_mean", regime_stats)
            self.assertIn("profit_factor_mean", regime_stats)
            self.assertIn("profit_factor_trend", regime_stats)
            self.assertIn("sharpe_ratio_mean", regime_stats)
            
            # Verify mean calculations
            self.assertAlmostEqual(
                regime_stats["win_rate_mean"], 
                np.mean([m["win_rate"] for m in self.multi_metrics])
            )
            
            self.assertAlmostEqual(
                regime_stats["profit_factor_mean"], 
                np.mean([m["profit_factor"] for m in self.multi_metrics])
            )
    
    def test_get_best_strategies_for_regime(self):
        """Test getting best strategies for a regime"""
        # Add performance data for multiple strategies
        strategy_performance = {
            "strategy_1": {"profit_factor": 2.5, "win_rate": 0.65},
            "strategy_2": {"profit_factor": 1.8, "win_rate": 0.55},
            "strategy_3": {"profit_factor": 3.0, "win_rate": 0.70},
            "strategy_4": {"profit_factor": 1.5, "win_rate": 0.50}
        }
        
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Add multiple samples for each strategy
        for _ in range(5):  # Need at least 5 samples
            for strategy_id, metrics in strategy_performance.items():
                self.tracker.update_performance(
                    strategy_id, regime_type, metrics, "AAPL", "1d"
                )
        
        # Get best strategies by profit factor
        best_strategies = self.tracker.get_best_strategies_for_regime(
            regime_type, metric_name="profit_factor", min_sample_size=5
        )
        
        # Verify order is correct (descending by profit factor)
        self.assertEqual(best_strategies[0][0], "strategy_3")  # Highest profit factor (3.0)
        self.assertEqual(best_strategies[1][0], "strategy_1")  # Second highest (2.5)
        self.assertEqual(best_strategies[2][0], "strategy_2")  # Third highest (1.8)
        self.assertEqual(best_strategies[3][0], "strategy_4")  # Lowest (1.5)
    
    def test_get_strategy_regime_ranking(self):
        """Test getting regime ranking for a strategy"""
        strategy_id = "test_strategy"
        
        # Add performance data for multiple regimes
        regime_performance = {
            MarketRegimeType.TRENDING_UP: {"profit_factor": 2.5, "win_rate": 0.65},
            MarketRegimeType.TRENDING_DOWN: {"profit_factor": 2.0, "win_rate": 0.60},
            MarketRegimeType.VOLATILE: {"profit_factor": 1.5, "win_rate": 0.50},
            MarketRegimeType.RANGE_BOUND: {"profit_factor": 3.0, "win_rate": 0.70},
            MarketRegimeType.NORMAL: {"profit_factor": 1.8, "win_rate": 0.55}
        }
        
        # Add data for each regime
        for regime, metrics in regime_performance.items():
            self.tracker.update_performance(
                strategy_id, regime, metrics, "AAPL", "1d"
            )
        
        # Get ranking by profit factor
        ranking = self.tracker.get_strategy_regime_ranking(
            strategy_id, metric_name="profit_factor"
        )
        
        # Verify order is correct (descending by profit factor)
        self.assertEqual(ranking[0][0], MarketRegimeType.RANGE_BOUND)  # Highest (3.0)
        self.assertEqual(ranking[1][0], MarketRegimeType.TRENDING_UP)  # Second (2.5)
        self.assertEqual(ranking[2][0], MarketRegimeType.TRENDING_DOWN)  # Third (2.0)
        self.assertEqual(ranking[3][0], MarketRegimeType.NORMAL)  # Fourth (1.8)
        self.assertEqual(ranking[4][0], MarketRegimeType.VOLATILE)  # Lowest (1.5)
    
    def test_get_performance_time_series(self):
        """Test getting time series data"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Add time series data
        now = datetime.now()
        
        for i in range(5):
            metrics = {
                "profit_factor": 2.0 + (i * 0.1),
                "win_rate": 0.6 + (i * 0.02)
            }
            
            # Manually set timestamp
            timestamp = now - timedelta(days=5-i)
            
            self.tracker.update_performance(
                strategy_id, regime_type, metrics, "AAPL", "1d"
            )
            
            # Manually update timestamp for testing
            self.tracker.time_series_data[strategy_id][regime_type][-1]["timestamp"] = timestamp
        
        # Get time series
        series = self.tracker.get_performance_time_series(
            strategy_id, regime_type, "profit_factor"
        )
        
        # Verify data
        self.assertEqual(len(series), 5)
        
        # Verify values are increasing (we set them to increase)
        for i in range(1, len(series)):
            self.assertGreater(series[i][1], series[i-1][1])
        
        # Test with limit
        limited_series = self.tracker.get_performance_time_series(
            strategy_id, regime_type, "profit_factor", limit=3
        )
        
        # Verify limit is applied
        self.assertEqual(len(limited_series), 3)
        
        # Verify most recent 3 are returned
        self.assertEqual(limited_series[0][1], 2.2)
        self.assertEqual(limited_series[1][1], 2.3)
        self.assertEqual(limited_series[2][1], 2.4)
    
    def test_detect_strategy_drift(self):
        """Test detecting strategy drift"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Case 1: Stable performance
        stable_metrics = [{"profit_factor": 2.0} for _ in range(20)]
        
        self.tracker.performance_history[strategy_id] = {
            regime_type: {"profit_factor": [2.0 for _ in range(20)]}
        }
        
        drift_result = self.tracker.detect_strategy_drift(
            strategy_id, regime_type, "profit_factor", window_size=10
        )
        
        # Verify no drift detected
        self.assertFalse(drift_result["drift_detected"])
        self.assertEqual(drift_result["direction"], "stable")
        
        # Case 2: Improving performance
        self.tracker.performance_history[strategy_id] = {
            regime_type: {"profit_factor": [2.0 for _ in range(10)] + [3.0 for _ in range(10)]}
        }
        
        drift_result = self.tracker.detect_strategy_drift(
            strategy_id, regime_type, "profit_factor", window_size=10
        )
        
        # Verify drift detected
        self.assertTrue(drift_result["drift_detected"])
        self.assertEqual(drift_result["direction"], "improving")
        self.assertGreater(drift_result["percent_change"], 0)
        
        # Case 3: Degrading performance
        self.tracker.performance_history[strategy_id] = {
            regime_type: {"profit_factor": [3.0 for _ in range(10)] + [2.0 for _ in range(10)]}
        }
        
        drift_result = self.tracker.detect_strategy_drift(
            strategy_id, regime_type, "profit_factor", window_size=10
        )
        
        # Verify drift detected
        self.assertTrue(drift_result["drift_detected"])
        self.assertEqual(drift_result["direction"], "degrading")
        self.assertLess(drift_result["percent_change"], 0)
    
    def test_analyze_correlation(self):
        """Test analyzing correlation between strategies"""
        # Setup multiple strategies with correlated and uncorrelated returns
        strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Strategy 1 and 2 will be correlated, 3 will be uncorrelated
        now = datetime.now()
        
        # Create base returns for correlation
        base_returns = np.random.normal(0.01, 0.05, 20)
        
        # Create time series data with known correlation patterns
        for i in range(20):
            timestamp = now - timedelta(days=20-i)
            
            # Strategy 1: Base returns
            metrics_1 = {"returns": base_returns[i]}
            
            # Strategy 2: Highly correlated with strategy 1
            metrics_2 = {"returns": base_returns[i] + np.random.normal(0, 0.01)}
            
            # Strategy 3: Uncorrelated
            metrics_3 = {"returns": np.random.normal(0.01, 0.05)}
            
            # Update for each strategy
            for strategy_id, metrics in zip(strategy_ids, [metrics_1, metrics_2, metrics_3]):
                if strategy_id not in self.tracker.time_series_data:
                    self.tracker.time_series_data[strategy_id] = {}
                if regime_type not in self.tracker.time_series_data[strategy_id]:
                    self.tracker.time_series_data[strategy_id][regime_type] = []
                
                # Add time series entry
                self.tracker.time_series_data[strategy_id][regime_type].append({
                    "timestamp": timestamp,
                    "metrics": metrics
                })
        
        # Analyze correlation
        correlation = self.tracker.analyze_correlation(
            strategy_ids, regime_type, metric_name="returns"
        )
        
        # Verify correlation matrix exists
        self.assertIn("correlation_matrix", correlation)
        self.assertIn("avg_correlations", correlation)
        
        # Verify high correlation between strategy 1 and 2
        self.assertGreater(correlation["correlation_matrix"]["strategy_1"]["strategy_2"], 0.7)
        
        # Verify lower correlation between strategy 1/2 and 3
        self.assertLess(abs(correlation["correlation_matrix"]["strategy_1"]["strategy_3"]), 0.5)
        self.assertLess(abs(correlation["correlation_matrix"]["strategy_2"]["strategy_3"]), 0.5)
    
    def test_get_regime_performance_summary(self):
        """Test getting regime performance summary"""
        strategy_id = "test_strategy"
        
        # Add performance data for multiple regimes
        regime_performance = {
            MarketRegimeType.TRENDING_UP: {"profit_factor": 2.5, "win_rate": 0.65, "expectancy": 0.5},
            MarketRegimeType.VOLATILE: {"profit_factor": 1.5, "win_rate": 0.50, "expectancy": 0.2},
            MarketRegimeType.RANGE_BOUND: {"profit_factor": 3.0, "win_rate": 0.70, "expectancy": 0.7}
        }
        
        # Add data for each regime
        for regime, metrics in regime_performance.items():
            # Add multiple samples to enable trend calculation
            for i in range(5):
                # Slightly vary metrics for trends
                varied_metrics = {k: v * (1 + i*0.02) for k, v in metrics.items()}
                self.tracker.update_performance(
                    strategy_id, regime, varied_metrics, "AAPL", "1d"
                )
        
        # Get performance summary
        summary = self.tracker.get_regime_performance_summary(strategy_id)
        
        # Verify all regimes are included
        for regime in regime_performance:
            self.assertIn(regime, summary)
            
            # Check basic properties
            regime_summary = summary[regime]
            self.assertIn("sample_size", regime_summary)
            self.assertIn("metrics", regime_summary)
            self.assertIn("drift", regime_summary)
            
            # Check metrics exist
            self.assertIn("profit_factor", regime_summary["metrics"])
            self.assertIn("win_rate", regime_summary["metrics"])
            
            # Check each metric has mean, latest, trend
            for metric_name in ["profit_factor", "win_rate"]:
                self.assertIn("mean", regime_summary["metrics"][metric_name])
                self.assertIn("latest", regime_summary["metrics"][metric_name])
                self.assertIn("trend", regime_summary["metrics"][metric_name])
                
                # Verify trend is positive (we set metrics to increase)
                self.assertGreater(regime_summary["metrics"][metric_name]["trend"], 0)

if __name__ == '__main__':
    unittest.main()
