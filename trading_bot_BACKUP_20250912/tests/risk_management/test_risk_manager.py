import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from trading_bot.risk import RiskManager

class TestRiskManager(unittest.TestCase):
    """Test suite for the RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures for RiskManager tests."""
        self.config = {
            "circuit_breakers": {
                "drawdown": {
                    "daily": {"threshold": -0.03, "level": 1},
                    "weekly": {"threshold": -0.05, "level": 2},
                    "monthly": {"threshold": -0.10, "level": 3}
                },
                "volatility": {
                    "threshold": 0.25,  # Annualized
                    "level": 2
                },
                "correlation": {
                    "threshold": 0.80,
                    "level": 2
                },
                "duration": {
                    "level1": 1,  # days
                    "level2": 3,
                    "level3": 5
                }
            },
            "position_sizing": {
                "max_position": 0.50,  # Maximum position size (50%)
                "volatility_scaling": True,
                "target_volatility": 0.15  # Target annualized volatility
            },
            "anomaly_detection": {
                "z_score_threshold": 3.0,
                "window_size": 20
            }
        }
        
        self.risk_manager = RiskManager(config=self.config)
        
        # Initialize portfolio values for testing
        self.base_value = 100000.0
        self.portfolio_values = []
        self.portfolio_dates = []
        self.current_date = datetime(2023, 1, 31)
        
        # Create 30 days of history (with a drawdown in the middle)
        for i in range(30):
            date = self.current_date - timedelta(days=30-i)
            
            # Create a drawdown scenario
            if 10 <= i < 15:
                value = self.base_value * (1 - 0.02 * (i-9))  # 2% daily drop for 5 days
            elif i >= 15:
                value = self.base_value * 0.9 * (1 + 0.01 * (i-14))  # Recovery phase
            else:
                value = self.base_value * (1 + 0.005 * i)  # Growth phase
            
            self.portfolio_values.append(value)
            self.portfolio_dates.append(date)
            
            # Update the risk manager with this value
            self.risk_manager.update_portfolio_value(value, date)
    
    def test_initialization(self):
        """Test that the RiskManager initializes correctly."""
        self.assertEqual(self.risk_manager.config, self.config)
        self.assertFalse(self.risk_manager.circuit_breaker_active)
        self.assertIsNone(self.risk_manager.active_circuit_breaker)
    
    def test_drawdown_calculation(self):
        """Test that drawdown is calculated correctly."""
        # Calculate drawdowns with our own logic
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = np.min((np.array(self.portfolio_values) - peak) / peak)
        
        # Compare with risk manager's calculation
        risk_manager_drawdown = self.risk_manager.calculate_drawdown()
        
        self.assertAlmostEqual(drawdown, risk_manager_drawdown, places=4)
        
        # Additional test with a known scenario
        test_values = [100, 110, 105, 95, 90, 100]
        expected_max_drawdown = (90 - 110) / 110  # -18.18%
        
        # Reset risk manager
        test_risk_manager = RiskManager(config=self.config)
        
        # Add test values
        for i, value in enumerate(test_values):
            test_risk_manager.update_portfolio_value(value, self.current_date - timedelta(days=6-i))
        
        calculated_drawdown = test_risk_manager.calculate_drawdown()
        self.assertAlmostEqual(calculated_drawdown, expected_max_drawdown, places=4)
    
    def test_volatility_calculation(self):
        """Test that volatility is calculated correctly."""
        # Calculate returns
        returns = []
        for i in range(1, len(self.portfolio_values)):
            ret = (self.portfolio_values[i] / self.portfolio_values[i-1]) - 1
            returns.append(ret)
        
        # Calculate annualized volatility
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)  # Standard annualization for daily returns
        
        # Get risk manager's volatility calculation
        risk_manager_vol = self.risk_manager.calculate_volatility()
        
        # Allow for small differences due to window size
        self.assertAlmostEqual(annualized_vol, risk_manager_vol, delta=0.02)
    
    def test_circuit_breaker_activation_drawdown(self):
        """Test that circuit breakers activate based on drawdown."""
        # Create a risk manager with a lower drawdown threshold
        test_config = self.config.copy()
        test_config["circuit_breakers"]["drawdown"]["daily"]["threshold"] = -0.05  # 5% daily
        test_risk_manager = RiskManager(config=test_config)
        
        # Add normal values first
        for i in range(5):
            date = self.current_date - timedelta(days=5-i)
            test_risk_manager.update_portfolio_value(self.base_value * (1 + 0.01 * i), date)
        
        # Verify no circuit breaker is active
        cb_status = test_risk_manager.check_circuit_breakers(self.current_date)
        self.assertFalse(cb_status["active"])
        
        # Add a large daily drawdown
        large_drop_date = self.current_date + timedelta(days=1)
        test_risk_manager.update_portfolio_value(self.base_value * 0.94, large_drop_date)  # 6% drop
        
        # Verify circuit breaker is now active
        cb_status = test_risk_manager.check_circuit_breakers(large_drop_date)
        self.assertTrue(cb_status["active"])
        self.assertEqual(cb_status["level"], 1)  # Should be level 1
        self.assertEqual(cb_status["trigger_cause"], "drawdown")
    
    def test_circuit_breaker_activation_volatility(self):
        """Test that circuit breakers activate based on volatility."""
        # Create a risk manager with a lower volatility threshold
        test_config = self.config.copy()
        test_config["circuit_breakers"]["volatility"]["threshold"] = 0.15  # 15% annualized
        test_risk_manager = RiskManager(config=test_config)
        
        # Add highly volatile values
        base = 100000.0
        for i in range(20):
            date = self.current_date - timedelta(days=20-i)
            # Alternating up and down 3%
            if i % 2 == 0:
                value = base * 1.03
            else:
                value = base * 0.97
            base = value  # Update for next iteration
            test_risk_manager.update_portfolio_value(value, date)
        
        # Verify circuit breaker is active due to high volatility
        cb_status = test_risk_manager.check_circuit_breakers(self.current_date)
        self.assertTrue(cb_status["active"])
        self.assertEqual(cb_status["level"], 2)  # Should be level 2
        self.assertEqual(cb_status["trigger_cause"], "volatility")
    
    def test_circuit_breaker_deactivation(self):
        """Test that circuit breakers deactivate after the specified duration."""
        # Create a risk manager with a short circuit breaker duration
        test_config = self.config.copy()
        test_config["circuit_breakers"]["duration"]["level1"] = 1  # 1 day for level 1
        test_risk_manager = RiskManager(config=test_config)
        
        # Add some normal values
        for i in range(5):
            date = self.current_date - timedelta(days=10-i)
            test_risk_manager.update_portfolio_value(self.base_value, date)
        
        # Add a large daily drawdown to trigger circuit breaker
        drop_date = self.current_date - timedelta(days=4)
        test_risk_manager.update_portfolio_value(self.base_value * 0.94, drop_date)  # 6% drop
        
        # Verify circuit breaker is active on the drop date
        cb_status = test_risk_manager.check_circuit_breakers(drop_date)
        self.assertTrue(cb_status["active"])
        
        # Add a value the next day
        next_date = drop_date + timedelta(days=1)
        test_risk_manager.update_portfolio_value(self.base_value * 0.95, next_date) 
        
        # Circuit breaker should still be active
        cb_status = test_risk_manager.check_circuit_breakers(next_date)
        self.assertTrue(cb_status["active"])
        
        # Add a value after the duration period
        after_duration = drop_date + timedelta(days=2)  # 2 days after the drop
        test_risk_manager.update_portfolio_value(self.base_value * 0.96, after_duration)
        
        # Circuit breaker should now be inactive
        cb_status = test_risk_manager.check_circuit_breakers(after_duration)
        self.assertFalse(cb_status["active"])
    
    def test_position_sizing_adjustments(self):
        """Test that position sizing is adjusted correctly based on volatility."""
        # Create a portfolio with different volatility levels
        low_vol_returns = [0.001] * 20  # Very stable returns
        high_vol_returns = [0.02, -0.018, 0.025, -0.023] * 5  # Volatile returns
        
        # Create risk managers
        low_vol_rm = RiskManager(config=self.config)
        high_vol_rm = RiskManager(config=self.config)
        
        # Initialize with base value
        base = 100000.0
        current = base
        for i, ret in enumerate(low_vol_returns):
            date = self.current_date - timedelta(days=20-i)
            current *= (1 + ret)
            low_vol_rm.update_portfolio_value(current, date)
        
        current = base
        for i, ret in enumerate(high_vol_returns):
            date = self.current_date - timedelta(days=20-i)
            current *= (1 + ret)
            high_vol_rm.update_portfolio_value(current, date)
        
        # Calculate position sizing adjustments
        low_vol_sizing = low_vol_rm.calculate_position_sizing_adjustment(self.base_value)
        high_vol_sizing = high_vol_rm.calculate_position_sizing_adjustment(self.base_value)
        
        # Low volatility should allow larger positions
        self.assertGreater(low_vol_sizing, high_vol_sizing)
        
        # High volatility should limit position size
        self.assertLess(high_vol_sizing, 1.0)  # Should reduce from baseline
    
    def test_max_position_limit(self):
        """Test that position sizing respects maximum position limits."""
        # Create a risk manager with a low max position
        test_config = self.config.copy()
        test_config["position_sizing"]["max_position"] = 0.30  # 30% maximum position
        test_rm = RiskManager(config=test_config)
        
        # Add some low volatility data to encourage higher position sizing
        base = 100000.0
        current = base
        for i in range(20):
            date = self.current_date - timedelta(days=20-i)
            current *= 1.001  # Very stable 0.1% daily returns
            test_rm.update_portfolio_value(current, date)
        
        # Calculate position sizing for a single position
        position_size = test_rm.calculate_max_position_size(base, "test_strategy")
        
        # Position size should not exceed the maximum
        self.assertLessEqual(position_size, 0.30 * base)
    
    def test_leverage_recommendations(self):
        """Test that leverage recommendations adjust with market conditions."""
        # Create two different market scenarios
        calm_rm = RiskManager(config=self.config)
        volatile_rm = RiskManager(config=self.config)
        
        # Add calm market data
        base = 100000.0
        current = base
        for i in range(20):
            date = self.current_date - timedelta(days=20-i)
            current *= 1.002  # Stable positive returns
            calm_rm.update_portfolio_value(current, date)
        
        # Add volatile market data with drawdown
        current = base
        for i in range(20):
            date = self.current_date - timedelta(days=20-i)
            # Create a volatile pattern with drawdown
            if i < 10:
                ret = 0.005 if i % 2 == 0 else -0.003
            else:
                ret = -0.01 if i < 15 else 0.003
            current *= (1 + ret)
            volatile_rm.update_portfolio_value(current, date)
        
        # Get leverage recommendations
        calm_leverage = calm_rm.recommend_leverage()
        volatile_leverage = volatile_rm.recommend_leverage()
        
        # Calm markets should allow more leverage
        self.assertGreater(calm_leverage, volatile_leverage)
        
        # Volatile markets should reduce leverage below 1.0
        self.assertLess(volatile_leverage, 1.0)

if __name__ == '__main__':
    unittest.main() 