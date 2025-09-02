import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from trading_bot.risk import RiskMonitor

class TestRiskMonitor(unittest.TestCase):
    """Test suite for the RiskMonitor class."""
    
    def setUp(self):
        """Set up test fixtures for RiskMonitor tests."""
        self.config = {
            "anomaly_detection": {
                "z_score_threshold": 3.0,
                "window_size": 20,
                "min_data_points": 5,
                "metrics": ["return", "volatility", "drawdown"]
            },
            "stress_test": {
                "scenarios": {
                    "market_crash": {"return_shock": -0.15, "volatility_multiplier": 2.5, "correlation_multiplier": 1.5},
                    "volatility_spike": {"return_shock": -0.08, "volatility_multiplier": 3.0, "correlation_multiplier": 1.3},
                    "correlation_breakdown": {"return_shock": -0.10, "volatility_multiplier": 2.0, "correlation_multiplier": 2.0}
                },
                "risk_levels": {
                    "low": {"max_drawdown": 0.05, "max_volatility": 0.10},
                    "medium": {"max_drawdown": 0.10, "max_volatility": 0.15},
                    "high": {"max_drawdown": 0.20, "max_volatility": 0.25},
                    "extreme": {"max_drawdown": 0.30, "max_volatility": 0.35}
                }
            }
        }
        
        self.risk_monitor = RiskMonitor(config=self.config)
        
        # Create a test portfolio
        self.portfolio_id = "test_portfolio"
        self.risk_monitor.create_portfolio(self.portfolio_id)
        
        # Add normal return data
        self.base_value = 100000.0
        self.start_date = datetime(2023, 1, 1)
        current_value = self.base_value
        
        # Add 30 days of normal return data
        for i in range(30):
            date = self.start_date + timedelta(days=i)
            # Small normal daily returns
            daily_return = 0.001 if i % 2 == 0 else -0.0005
            current_value *= (1 + daily_return)
            
            self.risk_monitor.update_portfolio(
                portfolio_id=self.portfolio_id,
                value=current_value,
                return_value=daily_return,
                timestamp=date
            )
    
    def test_initialization(self):
        """Test that the RiskMonitor initializes correctly."""
        self.assertEqual(self.risk_monitor.config, self.config)
        self.assertIn(self.portfolio_id, self.risk_monitor.portfolios)
        self.assertEqual(len(self.risk_monitor.portfolios[self.portfolio_id]["returns"]), 30)
    
    def test_anomaly_detection_normal_data(self):
        """Test that no anomalies are detected in normal data."""
        # Check for anomalies in the normal data
        anomalies = self.risk_monitor.detect_anomalies(self.portfolio_id)
        
        # Should be no anomalies in normal data
        self.assertEqual(len(anomalies), 0)
    
    def test_anomaly_detection_return_spike(self):
        """Test that return spikes are detected as anomalies."""
        # Add a large return spike (positive)
        date = self.start_date + timedelta(days=31)
        large_return = 0.10  # 10% daily return
        current_value = self.risk_monitor.portfolios[self.portfolio_id]["values"][-1] * (1 + large_return)
        
        self.risk_monitor.update_portfolio(
            portfolio_id=self.portfolio_id,
            value=current_value,
            return_value=large_return,
            timestamp=date
        )
        
        # Check for anomalies
        anomalies = self.risk_monitor.detect_anomalies(self.portfolio_id)
        
        # Should detect a return anomaly
        self.assertGreater(len(anomalies), 0)
        self.assertEqual(anomalies[0]["type"], "return_anomaly")
        self.assertEqual(anomalies[0]["direction"], "positive")
        
        # Add a large negative return spike
        date = self.start_date + timedelta(days=32)
        large_return = -0.08  # -8% daily return
        current_value = self.risk_monitor.portfolios[self.portfolio_id]["values"][-1] * (1 + large_return)
        
        self.risk_monitor.update_portfolio(
            portfolio_id=self.portfolio_id,
            value=current_value,
            return_value=large_return,
            timestamp=date
        )
        
        # Check for anomalies
        anomalies = self.risk_monitor.detect_anomalies(self.portfolio_id)
        
        # Should detect a negative return anomaly
        self.assertGreater(len(anomalies), 0)
        anomaly_types = [a["type"] for a in anomalies]
        anomaly_directions = [a["direction"] for a in anomalies if a["type"] == "return_anomaly"]
        
        self.assertIn("return_anomaly", anomaly_types)
        self.assertIn("negative", anomaly_directions)
    
    def test_anomaly_detection_volatility_spike(self):
        """Test that volatility spikes are detected as anomalies."""
        # Add a series of highly volatile returns
        current_value = self.risk_monitor.portfolios[self.portfolio_id]["values"][-1]
        
        for i in range(5):
            date = self.start_date + timedelta(days=31+i)
            # Alternating large positive and negative returns
            volatility_return = 0.05 if i % 2 == 0 else -0.05
            current_value *= (1 + volatility_return)
            
            self.risk_monitor.update_portfolio(
                portfolio_id=self.portfolio_id,
                value=current_value,
                return_value=volatility_return,
                timestamp=date
            )
        
        # Check for anomalies
        anomalies = self.risk_monitor.detect_anomalies(self.portfolio_id)
        
        # Should detect a volatility anomaly
        self.assertGreater(len(anomalies), 0)
        anomaly_types = [a["type"] for a in anomalies]
        self.assertIn("volatility_anomaly", anomaly_types)
    
    def test_anomaly_detection_drawdown(self):
        """Test that significant drawdowns are detected as anomalies."""
        # Add a series of negative returns causing a drawdown
        current_value = self.risk_monitor.portfolios[self.portfolio_id]["values"][-1]
        
        for i in range(5):
            date = self.start_date + timedelta(days=31+i)
            # Consecutive negative returns
            drawdown_return = -0.03
            current_value *= (1 + drawdown_return)
            
            self.risk_monitor.update_portfolio(
                portfolio_id=self.portfolio_id,
                value=current_value,
                return_value=drawdown_return,
                timestamp=date
            )
        
        # Check for anomalies
        anomalies = self.risk_monitor.detect_anomalies(self.portfolio_id)
        
        # Should detect a drawdown anomaly
        self.assertGreater(len(anomalies), 0)
        anomaly_types = [a["type"] for a in anomalies]
        self.assertIn("drawdown_anomaly", anomaly_types)
    
    def test_stress_test_single_strategy(self):
        """Test stress testing with a single strategy."""
        # Define a simple strategy allocation
        allocations = {"strategy_a": 100.0}  # 100% in one strategy
        
        # Define simple strategy properties
        strategy_profiles = {
            "strategy_a": {
                "volatility": 0.15,  # 15% annualized
                "expected_return": 0.10,  # 10% expected annual return
                "max_drawdown": -0.20,  # 20% historical max drawdown
                "correlation": 1.0  # Correlation with itself
            }
        }
        
        # Run stress test
        stress_results = self.risk_monitor.run_stress_test(
            self.portfolio_id, 
            allocations,
            strategy_profiles
        )
        
        # Check that results include all scenarios
        self.assertIn("market_crash", stress_results)
        self.assertIn("volatility_spike", stress_results)
        self.assertIn("correlation_breakdown", stress_results)
        
        # Check that projected metrics are reasonable
        self.assertLess(stress_results["market_crash"]["projected_return"], 0)  # Should be negative in crash
        self.assertGreater(stress_results["volatility_spike"]["projected_volatility"], 0.15)  # Should increase volatility
        
        # Check that risk level is determined
        self.assertIn("risk_level", stress_results)
        
        # Check for valid risk level
        valid_risk_levels = ["low", "medium", "high", "extreme"]
        self.assertIn(stress_results["risk_level"], valid_risk_levels)
    
    def test_stress_test_multi_strategy(self):
        """Test stress testing with multiple strategies."""
        # Define a multi-strategy allocation
        allocations = {
            "strategy_a": 40.0,
            "strategy_b": 30.0,
            "strategy_c": 30.0
        }
        
        # Define strategy profiles with different characteristics
        strategy_profiles = {
            "strategy_a": {
                "volatility": 0.20,
                "expected_return": 0.12,
                "max_drawdown": -0.25,
                "correlation": {
                    "strategy_a": 1.0,
                    "strategy_b": 0.6,
                    "strategy_c": 0.3
                }
            },
            "strategy_b": {
                "volatility": 0.15,
                "expected_return": 0.08,
                "max_drawdown": -0.18,
                "correlation": {
                    "strategy_a": 0.6,
                    "strategy_b": 1.0,
                    "strategy_c": 0.4
                }
            },
            "strategy_c": {
                "volatility": 0.12,
                "expected_return": 0.06,
                "max_drawdown": -0.15,
                "correlation": {
                    "strategy_a": 0.3,
                    "strategy_b": 0.4,
                    "strategy_c": 1.0
                }
            }
        }
        
        # Run stress test
        stress_results = self.risk_monitor.run_stress_test(
            self.portfolio_id, 
            allocations,
            strategy_profiles
        )
        
        # Check that diversification benefits are reflected
        # The portfolio drawdown should be less than the weighted average of individual drawdowns
        weighted_avg_drawdown = sum([
            abs(strategy_profiles[s]["max_drawdown"]) * (allocations[s]/100)
            for s in allocations
        ])
        
        # In correlation breakdown, diversification benefit should be reduced
        corr_breakdown_drawdown = abs(stress_results["correlation_breakdown"]["projected_max_drawdown"])
        self.assertLess(corr_breakdown_drawdown, weighted_avg_drawdown * 1.5)  # Still some benefit
        
        # In normal projection, there should be more diversification benefit
        normal_drawdown = abs(stress_results["normal"]["projected_max_drawdown"])
        self.assertLess(normal_drawdown, weighted_avg_drawdown * 0.9)  # Greater benefit
    
    def test_calculate_var(self):
        """Test Value at Risk (VaR) calculation."""
        # Create a portfolio with known returns
        test_portfolio_id = "var_test"
        self.risk_monitor.create_portfolio(test_portfolio_id)
        
        # Add fixed returns with a normal distribution
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.001, 0.02, 100)  # mean=0.1%, std=2%
        
        current_value = 100000.0
        for i, ret in enumerate(returns):
            date = self.start_date + timedelta(days=i)
            current_value *= (1 + ret)
            
            self.risk_monitor.update_portfolio(
                portfolio_id=test_portfolio_id,
                value=current_value,
                return_value=ret,
                timestamp=date
            )
        
        # Calculate VaR at different confidence levels
        var_95 = self.risk_monitor.calculate_var(test_portfolio_id, confidence=0.95)
        var_99 = self.risk_monitor.calculate_var(test_portfolio_id, confidence=0.99)
        
        # 99% VaR should be more severe than 95% VaR
        self.assertLess(var_99, var_95)
        
        # VaR should be negative
        self.assertLess(var_95, 0)
        self.assertLess(var_99, 0)
        
        # Calculate conditional VaR (Expected Shortfall)
        cvar_95 = self.risk_monitor.calculate_cvar(test_portfolio_id, confidence=0.95)
        
        # CVaR should be more severe than VaR
        self.assertLess(cvar_95, var_95)
    
    def test_generate_risk_report(self):
        """Test risk report generation."""
        # Add some additional data to make the report more meaningful
        current_value = self.risk_monitor.portfolios[self.portfolio_id]["values"][-1]
        
        # Add a few volatile days
        for i in range(5):
            date = self.start_date + timedelta(days=31+i)
            volatile_return = 0.03 if i % 2 == 0 else -0.025
            current_value *= (1 + volatile_return)
            
            self.risk_monitor.update_portfolio(
                portfolio_id=self.portfolio_id,
                value=current_value,
                return_value=volatile_return,
                timestamp=date
            )
        
        # Generate risk report
        risk_report = self.risk_monitor.generate_risk_report(self.portfolio_id)
        
        # Check report contents
        self.assertIn("current_metrics", risk_report)
        self.assertIn("historical_metrics", risk_report)
        self.assertIn("anomalies", risk_report)
        
        # Current metrics should include basics
        current_metrics = risk_report["current_metrics"]
        self.assertIn("volatility", current_metrics)
        self.assertIn("drawdown", current_metrics)
        self.assertIn("return", current_metrics)
        
        # Historical metrics should have time series
        historical_metrics = risk_report["historical_metrics"]
        self.assertIn("volatility_series", historical_metrics)
        self.assertIn("drawdown_series", historical_metrics)
        self.assertIn("return_series", historical_metrics)
        
        # Make sure all series have the same length
        vol_len = len(historical_metrics["volatility_series"])
        dd_len = len(historical_metrics["drawdown_series"])
        ret_len = len(historical_metrics["return_series"])
        
        self.assertEqual(vol_len, dd_len)
        self.assertEqual(vol_len, ret_len)

if __name__ == '__main__':
    unittest.main() 