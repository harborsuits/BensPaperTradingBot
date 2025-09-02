"""
Unit tests for Strategy Selector
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from datetime import datetime

from trading_bot.analytics.market_regime.strategy_selector import StrategySelector
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.analytics.market_regime.performance import RegimePerformanceTracker

class TestStrategySelector(unittest.TestCase):
    """Test cases for StrategySelector"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for selector files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock performance tracker
        self.performance_tracker = MagicMock(spec=RegimePerformanceTracker)
        
        self.config = {
            "selector_dir": self.temp_dir.name,
            "scoring_weights": {
                "profit_factor": 0.3,
                "sharpe_ratio": 0.2,
                "win_rate": 0.1,
                "expectancy": 0.2,
                "correlation": 0.1,
                "sample_size": 0.1,
            },
            "min_strategies": 1,
            "max_strategies": 3,
            "min_score_threshold": 0.4,
            "correlation_penalty": 0.5
        }
        
        # Create selector
        self.selector = StrategySelector(self.performance_tracker, self.config)
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test strategy configuration data"""
        self.test_strategies = {
            "strategy_1": {
                "name": "Trend Following",
                "compatible_symbols": ["AAPL", "MSFT", "GOOGL"],
                "compatible_timeframes": ["1d", "4h", "1h"],
                "compatible_regimes": ["trending_up", "trending_down"]
            },
            "strategy_2": {
                "name": "Mean Reversion",
                "compatible_symbols": ["AAPL", "AMZN", "FB"],
                "compatible_timeframes": ["1d", "4h"],
                "compatible_regimes": ["range_bound", "normal"]
            },
            "strategy_3": {
                "name": "Volatility Breakout",
                "compatible_symbols": ["AAPL", "TSLA", "NFLX"],
                "compatible_timeframes": ["1h", "15m"],
                "compatible_regimes": ["volatile", "trending_up"]
            },
            "strategy_4": {
                "name": "Multi-Regime Strategy",
                "compatible_symbols": ["AAPL", "SPY", "QQQ"],
                "compatible_timeframes": ["1d", "4h", "1h", "15m"],
                "compatible_regimes": ["trending_up", "trending_down", "range_bound", "volatile", "normal"]
            }
        }
        
        # Test strategy scores
        self.test_scores = {
            MarketRegimeType.TRENDING_UP: {
                "strategy_1": 0.8,
                "strategy_3": 0.7,
                "strategy_4": 0.6
            },
            MarketRegimeType.RANGE_BOUND: {
                "strategy_2": 0.85,
                "strategy_4": 0.65
            },
            MarketRegimeType.VOLATILE: {
                "strategy_3": 0.9,
                "strategy_4": 0.6
            }
        }
        
        # Test performance data
        self.test_performance = {
            "strategy_1": {
                MarketRegimeType.TRENDING_UP: {
                    "profit_factor_mean": 2.5,
                    "sharpe_ratio_mean": 1.8,
                    "win_rate_mean": 0.65,
                    "expectancy_mean": 0.5,
                    "sample_size": 20
                }
            },
            "strategy_2": {
                MarketRegimeType.RANGE_BOUND: {
                    "profit_factor_mean": 2.8,
                    "sharpe_ratio_mean": 2.0,
                    "win_rate_mean": 0.7,
                    "expectancy_mean": 0.6,
                    "sample_size": 15
                }
            },
            "strategy_3": {
                MarketRegimeType.VOLATILE: {
                    "profit_factor_mean": 3.0,
                    "sharpe_ratio_mean": 1.5,
                    "win_rate_mean": 0.6,
                    "expectancy_mean": 0.7,
                    "sample_size": 10
                }
            },
            "strategy_4": {
                MarketRegimeType.TRENDING_UP: {
                    "profit_factor_mean": 2.0,
                    "sharpe_ratio_mean": 1.5,
                    "win_rate_mean": 0.6,
                    "expectancy_mean": 0.4,
                    "sample_size": 25
                },
                MarketRegimeType.RANGE_BOUND: {
                    "profit_factor_mean": 1.8,
                    "sharpe_ratio_mean": 1.3,
                    "win_rate_mean": 0.55,
                    "expectancy_mean": 0.3,
                    "sample_size": 20
                },
                MarketRegimeType.VOLATILE: {
                    "profit_factor_mean": 1.5,
                    "sharpe_ratio_mean": 1.0,
                    "win_rate_mean": 0.5,
                    "expectancy_mean": 0.2,
                    "sample_size": 15
                }
            }
        }
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.selector.config, self.config)
        self.assertEqual(self.selector.performance_tracker, self.performance_tracker)
        self.assertEqual(self.selector.selector_dir, self.temp_dir.name)
    
    def test_register_strategy(self):
        """Test registering a strategy"""
        strategy_id = "test_strategy"
        strategy_config = {
            "name": "Test Strategy",
            "compatible_symbols": ["AAPL", "MSFT"],
            "compatible_timeframes": ["1d", "4h"]
        }
        
        # Register strategy
        self.selector.register_strategy(strategy_id, strategy_config)
        
        # Verify strategy was registered
        self.assertIn(strategy_id, self.selector.strategy_configs)
        self.assertEqual(self.selector.strategy_configs[strategy_id], strategy_config)
        
        # Verify file was saved
        expected_file = os.path.join(self.temp_dir.name, "strategy_configs.json")
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        with open(expected_file, 'r') as f:
            saved_data = json.load(f)
            self.assertIn(strategy_id, saved_data)
            self.assertEqual(saved_data[strategy_id], strategy_config)
    
    def test_set_timeframe_mapping(self):
        """Test setting timeframe mapping"""
        symbol = "AAPL"
        regime_type = MarketRegimeType.TRENDING_UP
        timeframe = "1h"
        
        # Set mapping
        self.selector.set_timeframe_mapping(symbol, regime_type, timeframe)
        
        # Verify mapping was set
        self.assertIn(symbol, self.selector.timeframe_mappings)
        self.assertIn(regime_type, self.selector.timeframe_mappings[symbol])
        self.assertEqual(self.selector.timeframe_mappings[symbol][regime_type], timeframe)
        
        # Verify file was saved
        expected_file = os.path.join(self.temp_dir.name, "timeframe_mappings.json")
        self.assertTrue(os.path.exists(expected_file))
    
    def test_get_preferred_timeframe(self):
        """Test getting preferred timeframe"""
        symbol = "AAPL"
        regime_type = MarketRegimeType.TRENDING_UP
        timeframe = "1h"
        
        # Set mapping
        self.selector.set_timeframe_mapping(symbol, regime_type, timeframe)
        
        # Get preferred timeframe
        result = self.selector.get_preferred_timeframe(symbol, regime_type)
        
        # Verify result
        self.assertEqual(result, timeframe)
        
        # Test fallback to NORMAL
        result = self.selector.get_preferred_timeframe(symbol, MarketRegimeType.VOLATILE)
        
        # Should fallback to default
        self.assertEqual(result, "1h")  # Default from test parameters
        
        # Test with NORMAL set
        self.selector.set_timeframe_mapping(symbol, MarketRegimeType.NORMAL, "4h")
        
        # Now should fallback to NORMAL
        result = self.selector.get_preferred_timeframe(symbol, MarketRegimeType.VOLATILE)
        self.assertEqual(result, "4h")
    
    def test_update_strategy_scores(self):
        """Test updating strategy scores"""
        # Setup mock performance tracker
        self.performance_tracker.get_performance_by_regime.return_value = {
            MarketRegimeType.TRENDING_UP: self.test_performance["strategy_1"][MarketRegimeType.TRENDING_UP]
        }
        
        # Register strategies
        for strategy_id, config in self.test_strategies.items():
            self.selector.register_strategy(strategy_id, config)
        
        # Update scores for TRENDING_UP
        result = self.selector.update_strategy_scores(MarketRegimeType.TRENDING_UP)
        
        # Verify scores were calculated
        self.assertIn(MarketRegimeType.TRENDING_UP, self.selector.strategy_scores)
        self.assertIn("strategy_1", self.selector.strategy_scores[MarketRegimeType.TRENDING_UP])
        
        # Verify score is between 0 and 1
        score = self.selector.strategy_scores[MarketRegimeType.TRENDING_UP]["strategy_1"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_select_strategies(self):
        """Test selecting strategies"""
        # Register strategies
        for strategy_id, config in self.test_strategies.items():
            self.selector.register_strategy(strategy_id, config)
        
        # Setup mock scores
        self.selector.strategy_scores = self.test_scores
        
        # Select strategies for TRENDING_UP
        result = self.selector.select_strategies("AAPL", MarketRegimeType.TRENDING_UP, "1d")
        
        # Verify the right strategies were selected
        self.assertEqual(len(result), 3)  # max_strategies = 3
        
        # Verify order by score
        self.assertEqual(result[0]["strategy_id"], "strategy_1")
        self.assertEqual(result[1]["strategy_id"], "strategy_3")
        self.assertEqual(result[2]["strategy_id"], "strategy_4")
        
        # Verify active strategies were updated
        self.assertIn("AAPL", self.selector.active_strategies)
        self.assertEqual(self.selector.active_strategies["AAPL"], 
                          ["strategy_1", "strategy_3", "strategy_4"])
        
        # Verify weights were calculated
        self.assertIn("AAPL", self.selector.strategy_weights)
        self.assertEqual(len(self.selector.strategy_weights["AAPL"]), 3)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(self.selector.strategy_weights["AAPL"].values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_select_strategies_with_symbol_filtering(self):
        """Test selecting strategies with symbol filtering"""
        # Register strategies
        for strategy_id, config in self.test_strategies.items():
            self.selector.register_strategy(strategy_id, config)
        
        # Setup mock scores (all high)
        for regime in self.test_scores:
            for strategy_id in self.test_strategies:
                if regime not in self.test_scores:
                    self.test_scores[regime] = {}
                self.test_scores[regime][strategy_id] = 0.8
        
        self.selector.strategy_scores = self.test_scores
        
        # Select strategies for NFLX (only compatible with strategy_3)
        result = self.selector.select_strategies("NFLX", MarketRegimeType.VOLATILE, "1h")
        
        # Verify only compatible strategies were selected
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["strategy_id"], "strategy_3")
    
    def test_select_strategies_with_timeframe_filtering(self):
        """Test selecting strategies with timeframe filtering"""
        # Register strategies
        for strategy_id, config in self.test_strategies.items():
            self.selector.register_strategy(strategy_id, config)
        
        # Setup mock scores (all high)
        for regime in self.test_scores:
            for strategy_id in self.test_strategies:
                if regime not in self.test_scores:
                    self.test_scores[regime] = {}
                self.test_scores[regime][strategy_id] = 0.8
        
        self.selector.strategy_scores = self.test_scores
        
        # Select strategies with 15m timeframe (only compatible with strategy_3 and strategy_4)
        result = self.selector.select_strategies("AAPL", MarketRegimeType.VOLATILE, "15m")
        
        # Verify only compatible strategies were selected
        self.assertEqual(len(result), 2)
        
        # Must include strategy_3 and strategy_4
        strategy_ids = [item["strategy_id"] for item in result]
        self.assertIn("strategy_3", strategy_ids)
        self.assertIn("strategy_4", strategy_ids)
    
    def test_select_strategies_with_regime_filtering(self):
        """Test selecting strategies with regime filtering"""
        # Register strategies
        for strategy_id, config in self.test_strategies.items():
            self.selector.register_strategy(strategy_id, config)
        
        # Setup mock scores (all high)
        for regime in self.test_scores:
            for strategy_id in self.test_strategies:
                if regime not in self.test_scores:
                    self.test_scores[regime] = {}
                self.test_scores[regime][strategy_id] = 0.8
        
        self.selector.strategy_scores = self.test_scores
        
        # Select strategies for RANGE_BOUND (only compatible with strategy_2 and strategy_4)
        result = self.selector.select_strategies("AAPL", MarketRegimeType.RANGE_BOUND, "1d")
        
        # Verify only compatible strategies were selected
        self.assertEqual(len(result), 2)
        
        # Must include strategy_2 and strategy_4
        strategy_ids = [item["strategy_id"] for item in result]
        self.assertIn("strategy_2", strategy_ids)
        self.assertIn("strategy_4", strategy_ids)
    
    def test_calculate_strategy_weights(self):
        """Test calculating strategy weights"""
        # Simple scenario: three strategies with different scores
        selected_strategies = ["strategy_1", "strategy_2", "strategy_3"]
        scores = {
            "strategy_1": 0.8,
            "strategy_2": 0.6,
            "strategy_3": 0.4
        }
        
        # Calculate weights
        weights = self.selector._calculate_strategy_weights(selected_strategies, scores)
        
        # Verify all strategies have weights
        for strategy_id in selected_strategies:
            self.assertIn(strategy_id, weights)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
        # Higher scores should have higher weights
        self.assertGreater(weights["strategy_1"], weights["strategy_2"])
        self.assertGreater(weights["strategy_2"], weights["strategy_3"])
    
    def test_get_active_strategies(self):
        """Test getting active strategies"""
        # Setup active strategies
        self.selector.active_strategies = {
            "AAPL": ["strategy_1", "strategy_3"],
            "MSFT": ["strategy_2", "strategy_4"]
        }
        
        # Get active strategies
        result = self.selector.get_active_strategies("AAPL")
        
        # Verify result
        self.assertEqual(result, ["strategy_1", "strategy_3"])
        
        # Test for non-existent symbol
        result = self.selector.get_active_strategies("UNKNOWN")
        self.assertEqual(result, [])
    
    def test_get_strategy_weight(self):
        """Test getting strategy weight"""
        # Setup strategy weights
        self.selector.strategy_weights = {
            "AAPL": {
                "strategy_1": 0.6,
                "strategy_3": 0.4
            },
            "MSFT": {
                "strategy_2": 0.7,
                "strategy_4": 0.3
            }
        }
        
        # Get strategy weight
        result = self.selector.get_strategy_weight("AAPL", "strategy_1")
        
        # Verify result
        self.assertEqual(result, 0.6)
        
        # Test for non-existent strategy
        result = self.selector.get_strategy_weight("AAPL", "UNKNOWN")
        self.assertEqual(result, 0.0)
        
        # Test for non-existent symbol
        result = self.selector.get_strategy_weight("UNKNOWN", "strategy_1")
        self.assertEqual(result, 0.0)
    
    def test_get_strategy_scores(self):
        """Test getting strategy scores"""
        # Setup strategy scores
        self.selector.strategy_scores = self.test_scores
        
        # Get strategy scores
        result = self.selector.get_strategy_scores(MarketRegimeType.TRENDING_UP)
        
        # Verify result
        self.assertEqual(result, self.test_scores[MarketRegimeType.TRENDING_UP])
        
        # Test for non-existent regime
        result = self.selector.get_strategy_scores(MarketRegimeType.UNKNOWN)
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()
