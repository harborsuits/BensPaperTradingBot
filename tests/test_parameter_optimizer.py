"""
Unit tests for Parameter Optimizer
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from datetime import datetime

from trading_bot.analytics.market_regime.adaptation import ParameterOptimizer
from trading_bot.analytics.market_regime.detector import MarketRegimeType

class TestParameterOptimizer(unittest.TestCase):
    """Test cases for ParameterOptimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for parameter files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.config = {
            "parameter_dir": self.temp_dir.name,
            "transition_smoothing": True,
            "smoothing_factor": 0.3,
            "max_performance_history": 10
        }
        
        # Create optimizer
        self.optimizer = ParameterOptimizer(self.config)
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test parameter data"""
        # Create test parameter sets
        self.test_params = {
            "trending_up": {
                "stop_loss": 2.0,
                "take_profit": 4.0,
                "entry_threshold": 0.5,
                "trail_distance": 1.0
            },
            "trending_down": {
                "stop_loss": 1.5,
                "take_profit": 3.0,
                "entry_threshold": 0.6,
                "trail_distance": 0.8
            },
            "volatile": {
                "stop_loss": 3.0,
                "take_profit": 6.0,
                "entry_threshold": 0.7,
                "trail_distance": 1.5
            },
            "range_bound": {
                "stop_loss": 1.0,
                "take_profit": 2.0,
                "entry_threshold": 0.4,
                "trail_distance": 0.5
            },
            "normal": {
                "stop_loss": 2.0,
                "take_profit": 3.0,
                "entry_threshold": 0.5,
                "trail_distance": 1.0
            }
        }
        
        # Create test performance data
        self.test_performance = {
            "trending_up": {
                "profit_factor": 2.5,
                "sharpe_ratio": 1.8,
                "win_rate": 0.65
            },
            "trending_down": {
                "profit_factor": 2.0,
                "sharpe_ratio": 1.5,
                "win_rate": 0.6
            },
            "volatile": {
                "profit_factor": 1.8,
                "sharpe_ratio": 1.2,
                "win_rate": 0.55
            },
            "range_bound": {
                "profit_factor": 1.5,
                "sharpe_ratio": 1.0,
                "win_rate": 0.5
            },
            "normal": {
                "profit_factor": 1.7,
                "sharpe_ratio": 1.3,
                "win_rate": 0.58
            }
        }
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.optimizer.config, self.config)
        self.assertEqual(self.optimizer.parameter_dir, self.temp_dir.name)
        self.assertEqual(self.optimizer.smoothing_factor, 0.3)
    
    def test_update_optimal_parameters(self):
        """Test updating optimal parameters"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        parameters = self.test_params["trending_up"]
        performance_metrics = self.test_performance["trending_up"]
        
        # Update parameters
        self.optimizer.update_optimal_parameters(
            strategy_id, regime_type, parameters, performance_metrics
        )
        
        # Verify parameters were stored
        self.assertIn(strategy_id, self.optimizer.optimal_parameters)
        self.assertIn(regime_type, self.optimizer.optimal_parameters[strategy_id])
        self.assertEqual(self.optimizer.optimal_parameters[strategy_id][regime_type], parameters)
        
        # Verify performance was stored
        self.assertIn(strategy_id, self.optimizer.parameter_performance)
        self.assertIn(regime_type, self.optimizer.parameter_performance[strategy_id])
        self.assertEqual(len(self.optimizer.parameter_performance[strategy_id][regime_type]), 1)
        stored_perf = self.optimizer.parameter_performance[strategy_id][regime_type][0]
        self.assertEqual(stored_perf["metrics"], performance_metrics)
        
        # Verify file was saved
        expected_file = os.path.join(self.temp_dir.name, f"strategy_{strategy_id}.json")
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        with open(expected_file, 'r') as f:
            saved_data = json.load(f)
            self.assertIn(regime_type.value, saved_data)
            self.assertEqual(saved_data[regime_type.value], parameters)
    
    def test_get_optimal_parameters(self):
        """Test getting optimal parameters"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        parameters = self.test_params["trending_up"]
        
        # First add some parameters
        self.optimizer.update_optimal_parameters(
            strategy_id, regime_type, parameters
        )
        
        # Get parameters
        result = self.optimizer.get_optimal_parameters(
            strategy_id, regime_type, "AAPL", "1d", 0.9
        )
        
        # Verify result
        self.assertEqual(result, parameters)
    
    def test_get_optimal_parameters_fallback(self):
        """Test getting optimal parameters with fallback to NORMAL"""
        strategy_id = "test_strategy"
        
        # Add parameters for NORMAL regime only
        self.optimizer.update_optimal_parameters(
            strategy_id, MarketRegimeType.NORMAL, self.test_params["normal"]
        )
        
        # Try to get parameters for TRENDING_UP
        result = self.optimizer.get_optimal_parameters(
            strategy_id, MarketRegimeType.TRENDING_UP, "AAPL", "1d", 0.9
        )
        
        # Verify fallback to NORMAL
        self.assertEqual(result, self.test_params["normal"])
    
    def test_smoothed_parameters(self):
        """Test parameter smoothing during transitions"""
        strategy_id = "test_strategy"
        symbol = "AAPL"
        timeframe = "1d"
        
        # Add initial parameters (NORMAL)
        self.optimizer.update_optimal_parameters(
            strategy_id, MarketRegimeType.NORMAL, self.test_params["normal"]
        )
        
        # Get parameters for NORMAL
        normal_params = self.optimizer.get_optimal_parameters(
            strategy_id, MarketRegimeType.NORMAL, symbol, timeframe, 0.9
        )
        
        # Add parameters for TRENDING_UP
        self.optimizer.update_optimal_parameters(
            strategy_id, MarketRegimeType.TRENDING_UP, self.test_params["trending_up"]
        )
        
        # Get parameters for TRENDING_UP (first request, should start transition)
        trending_params_1 = self.optimizer.get_optimal_parameters(
            strategy_id, MarketRegimeType.TRENDING_UP, symbol, timeframe, 0.9
        )
        
        # Verify transition has started
        transition_key = f"{strategy_id}_{symbol}_{timeframe}"
        self.assertIn(transition_key, self.optimizer.current_transitions)
        self.assertEqual(
            self.optimizer.current_transitions[transition_key]['regime_type'], 
            MarketRegimeType.TRENDING_UP
        )
        
        # Verify parameters are partially transitioned
        self.assertNotEqual(trending_params_1, self.test_params["normal"])
        self.assertNotEqual(trending_params_1, self.test_params["trending_up"])
        
        # Check specific parameter values
        for param in ["stop_loss", "take_profit", "entry_threshold", "trail_distance"]:
            # In a transition, values should be between source and target
            normal_val = self.test_params["normal"][param]
            trending_val = self.test_params["trending_up"][param]
            current_val = trending_params_1[param]
            
            # Check parameter is between normal and trending values
            if normal_val <= trending_val:
                self.assertGreaterEqual(current_val, normal_val)
                self.assertLessEqual(current_val, trending_val)
            else:
                self.assertLessEqual(current_val, normal_val)
                self.assertGreaterEqual(current_val, trending_val)
        
        # Get parameters again (should continue transition)
        trending_params_2 = self.optimizer.get_optimal_parameters(
            strategy_id, MarketRegimeType.TRENDING_UP, symbol, timeframe, 0.9
        )
        
        # Verify progress has increased
        progress_1 = self.optimizer.current_transitions[transition_key]['progress']
        progress_2 = self.optimizer.current_transitions[transition_key]['progress']
        self.assertGreaterEqual(progress_2, progress_1)
        
        # Make multiple calls to complete transition
        for _ in range(10):
            trending_params_final = self.optimizer.get_optimal_parameters(
                strategy_id, MarketRegimeType.TRENDING_UP, symbol, timeframe, 0.9
            )
        
        # Verify transition has completed
        self.assertNotIn(transition_key, self.optimizer.current_transitions)
        
        # Verify final parameters match target
        self.assertEqual(trending_params_final, self.test_params["trending_up"])
    
    def test_optimize_parameters(self):
        """Test parameter optimization"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Define parameter ranges
        parameter_ranges = {
            "stop_loss": (1.0, 5.0, 0.1),  # (min, max, step)
            "take_profit": (2.0, 10.0, 0.1),
            "entry_threshold": (0.1, 0.9, 0.1)
        }
        
        # Define evaluation function (higher is better)
        def eval_func(params):
            # Simple evaluation that prefers high take_profit/stop_loss ratio
            return params["take_profit"] / params["stop_loss"] * (1 - params["entry_threshold"])
        
        # Run optimization
        optimal_params, score = self.optimizer.optimize_parameters(
            strategy_id, regime_type, parameter_ranges, eval_func,
            {"max_iterations": 10, "population_size": 10}
        )
        
        # Verify result has expected parameters
        self.assertIn("stop_loss", optimal_params)
        self.assertIn("take_profit", optimal_params)
        self.assertIn("entry_threshold", optimal_params)
        
        # Verify parameters are within ranges
        self.assertGreaterEqual(optimal_params["stop_loss"], parameter_ranges["stop_loss"][0])
        self.assertLessEqual(optimal_params["stop_loss"], parameter_ranges["stop_loss"][1])
        
        self.assertGreaterEqual(optimal_params["take_profit"], parameter_ranges["take_profit"][0])
        self.assertLessEqual(optimal_params["take_profit"], parameter_ranges["take_profit"][1])
        
        self.assertGreaterEqual(optimal_params["entry_threshold"], parameter_ranges["entry_threshold"][0])
        self.assertLessEqual(optimal_params["entry_threshold"], parameter_ranges["entry_threshold"][1])
        
        # Verify optimization produces good results
        # The best parameters should have high take_profit, low stop_loss, and low entry_threshold
        self.assertGreaterEqual(optimal_params["take_profit"], parameter_ranges["take_profit"][0] + 
                             (parameter_ranges["take_profit"][1] - parameter_ranges["take_profit"][0]) * 0.5)
        
    def test_clear_transitions(self):
        """Test clearing transitions"""
        # Setup transitions
        strategy_id = "test_strategy"
        self.optimizer.current_transitions = {
            f"{strategy_id}_AAPL_1d": {"regime_type": MarketRegimeType.TRENDING_UP},
            f"{strategy_id}_MSFT_1d": {"regime_type": MarketRegimeType.TRENDING_DOWN},
            "other_strategy_SPY_1d": {"regime_type": MarketRegimeType.VOLATILE}
        }
        
        # Clear transitions for specific strategy
        self.optimizer.clear_transitions(strategy_id)
        
        # Verify only strategy's transitions were cleared
        self.assertNotIn(f"{strategy_id}_AAPL_1d", self.optimizer.current_transitions)
        self.assertNotIn(f"{strategy_id}_MSFT_1d", self.optimizer.current_transitions)
        self.assertIn("other_strategy_SPY_1d", self.optimizer.current_transitions)
        
        # Clear all transitions
        self.optimizer.clear_transitions()
        
        # Verify all transitions were cleared
        self.assertEqual(len(self.optimizer.current_transitions), 0)
    
    def test_learn_from_performance(self):
        """Test learning from performance history"""
        strategy_id = "test_strategy"
        regime_type = MarketRegimeType.TRENDING_UP
        
        # Add performance records
        self.optimizer.parameter_performance[strategy_id] = {
            regime_type: [
                {
                    "parameters": {"stop_loss": 2.0, "take_profit": 4.0},
                    "metrics": {"profit_factor": 1.5}
                },
                {
                    "parameters": {"stop_loss": 1.5, "take_profit": 3.0},
                    "metrics": {"profit_factor": 2.0}  # Better
                },
                {
                    "parameters": {"stop_loss": 2.5, "take_profit": 5.0},
                    "metrics": {"profit_factor": 1.8}
                }
            ]
        }
        
        # Learn from performance
        result = self.optimizer.learn_from_performance(strategy_id, regime_type)
        
        # Verify result
        self.assertTrue(result)
        
        # Verify best parameters were selected (with profit_factor = 2.0)
        optimal_params = self.optimizer.optimal_parameters[strategy_id][regime_type]
        self.assertEqual(optimal_params["stop_loss"], 1.5)
        self.assertEqual(optimal_params["take_profit"], 3.0)

if __name__ == '__main__':
    unittest.main()
