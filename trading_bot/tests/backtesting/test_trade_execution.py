import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy

from trading_bot.backtesting.unified_backtester import UnifiedBacktester

class TestTradeExecution(unittest.TestCase):
    """Test suite for trade execution functionality in the backtester."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize a backtester for trade tests
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 31)
        self.initial_capital = 100000.0
        self.strategies = ["trend_following", "momentum", "mean_reversion"]
        
        # Create the backtester with a higher cost for testing
        self.backtester = UnifiedBacktester(
            initial_capital=self.initial_capital,
            strategies=self.strategies,
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d"),
            trading_cost_pct=0.2,  # 0.2% cost for clear impact
            min_trade_value=100.0,  # $100 minimum trade value
            debug_mode=True
        )
        
        # Initialize portfolio and positions for testing
        self.current_capital = self.initial_capital
        self.current_positions = {
            "trend_following": 40000.0,
            "momentum": 30000.0,
            "mean_reversion": 30000.0
        }
        
        # Make sure the trades list is empty
        self.backtester.trades = []
        self.backtester.total_costs = 0.0
    
    def test_basic_trade_execution(self):
        """Test execution of basic trades with costs."""
        # Define new target allocations
        new_allocations = {
            "trend_following": 30.0,  # 30% (from 40%)
            "momentum": 40.0,        # 40% (from 30%)
            "mean_reversion": 30.0   # 30% (unchanged)
        }
        
        # Copy positions for tracking
        positions_before = deepcopy(self.current_positions)
        total_before = sum(positions_before.values())
        
        # Execute the trades
        date_str = "2023-01-15"
        self.backtester._execute_trades(
            date_str,
            new_allocations,
            self.current_capital,
            self.current_positions
        )
        
        # Check that positions were updated
        self.assertLess(
            self.current_positions["trend_following"],
            positions_before["trend_following"],
            "Trend following position should decrease"
        )
        self.assertGreater(
            self.current_positions["momentum"],
            positions_before["momentum"],
            "Momentum position should increase"
        )
        
        # Verify trading costs were applied
        total_after = sum(self.current_positions.values())
        self.assertLess(total_after, total_before, "Total value should decrease due to costs")
        self.assertGreater(self.backtester.total_costs, 0, "Trading costs should be recorded")
        
        # Verify trades were recorded
        self.assertEqual(len(self.backtester.trades), 2, "Two trades should be recorded")
        
        # Specific check for trades
        for trade in self.backtester.trades:
            self.assertEqual(trade["date"], date_str, "Trade date should match")
            
            if trade["strategy"] == "trend_following":
                self.assertEqual(trade["direction"], "sell", "Trend following trade should be a sell")
            elif trade["strategy"] == "momentum":
                self.assertEqual(trade["direction"], "buy", "Momentum trade should be a buy")
    
    def test_cost_calculation(self):
        """Test that trading costs are calculated correctly."""
        # Define a large reallocation for clear cost impact
        new_allocations = {
            "trend_following": 0.0,   # 0% (sell all)
            "momentum": 80.0,         # 80% (large increase)
            "mean_reversion": 20.0    # 20% (small decrease)
        }
        
        # Execute the trades
        self.backtester._execute_trades(
            "2023-01-15",
            new_allocations,
            self.current_capital,
            self.current_positions
        )
        
        # Calculate expected costs manually
        expected_tf_cost = 40000.0 * 0.002  # Sell 40K at 0.2%
        expected_mom_cost = (80000.0 - 30000.0) * 0.002  # Buy 50K more at 0.2% 
        expected_mr_cost = (30000.0 - 20000.0) * 0.002  # Sell 10K at 0.2%
        expected_total_cost = expected_tf_cost + expected_mom_cost + expected_mr_cost
        
        # Verify cost calculations
        self.assertAlmostEqual(
            self.backtester.total_costs,
            expected_total_cost,
            delta=0.1,
            msg="Total trading costs should match expected value"
        )
        
        # Check individual trade costs
        tf_trade = next(t for t in self.backtester.trades if t["strategy"] == "trend_following")
        self.assertAlmostEqual(
            tf_trade["cost"],
            expected_tf_cost,
            delta=0.1,
            msg="Trend following trade cost should match expected"
        )
    
    def test_minimum_trade_threshold(self):
        """Test that small trades below the minimum threshold are not executed."""
        # Define allocations with a small change
        new_allocations = {
            "trend_following": 39.5,  # 39.5% (small change)
            "momentum": 30.5,        # 30.5% (small change)
            "mean_reversion": 30.0   # 30% (unchanged)
        }
        
        # Copy positions for comparison
        positions_before = deepcopy(self.current_positions)
        
        # Execute the trades
        self.backtester._execute_trades(
            "2023-01-15",
            new_allocations,
            self.current_capital,
            self.current_positions
        )
        
        # Verify no trades were executed
        self.assertEqual(len(self.backtester.trades), 0, "No trades should be executed below threshold")
        
        # Positions should remain unchanged
        for strategy in self.strategies:
            self.assertEqual(
                self.current_positions[strategy],
                positions_before[strategy],
                f"{strategy} position should remain unchanged"
            )
        
        # No costs incurred
        self.assertEqual(self.backtester.total_costs, 0.0, "No costs should be incurred")
    
    def test_cash_allocation(self):
        """Test that cash allocations are handled correctly."""
        # Define allocations with cash component
        new_allocations = {
            "trend_following": 20.0,
            "momentum": 20.0,
            "mean_reversion": 20.0,
            "cash": 40.0  # 40% cash
        }
        
        # Execute the trades
        self.backtester._execute_trades(
            "2023-01-15",
            new_allocations,
            self.current_capital,
            self.current_positions
        )
        
        # Check that positions were adjusted correctly
        self.assertAlmostEqual(
            self.current_positions["trend_following"],
            self.current_capital * 0.2 * (1 - 0.002),  # 20% minus cost
            delta=1.0,  # Allow small difference due to rounding
            msg="Trend following should be 20% of capital"
        )
        
        # Total allocation should now be 60% of capital plus cash
        total_strategy_value = sum(self.current_positions.values())
        self.assertAlmostEqual(
            total_strategy_value,
            self.current_capital * 0.6,  # 60% of capital
            delta=self.current_capital * 0.01,  # Allow 1% difference due to costs
            msg="Total allocation should be 60% of capital"
        )
    
    def test_trade_logging(self):
        """Test that trades are properly logged with all required information."""
        # Define new allocations
        new_allocations = {
            "trend_following": 10.0,
            "momentum": 70.0,
            "mean_reversion": 20.0
        }
        
        # Execute the trades
        date_str = "2023-01-15"
        self.backtester._execute_trades(
            date_str,
            new_allocations,
            self.current_capital,
            self.current_positions
        )
        
        # Verify all trades have required fields
        required_fields = [
            "date", "strategy", "direction", "value", 
            "cost", "pre_trade_position", "target_position"
        ]
        
        for trade in self.backtester.trades:
            for field in required_fields:
                self.assertIn(
                    field, 
                    trade, 
                    f"Trade should contain {field} field"
                )
            
            # Check specific fields
            self.assertEqual(trade["date"], date_str, "Trade date should match")
            self.assertIn(trade["strategy"], self.strategies, "Trade strategy should be valid")
            self.assertIn(trade["direction"], ["buy", "sell"], "Trade direction should be buy or sell")
            self.assertGreater(trade["value"], 0, "Trade value should be positive")
            self.assertGreater(trade["cost"], 0, "Trade cost should be positive")
    
    def test_position_adjustment(self):
        """Test that positions are correctly adjusted after trades."""
        # Define drastic allocation change
        new_allocations = {
            "trend_following": 80.0,  # 80% (huge increase)
            "momentum": 10.0,        # 10% (big decrease)
            "mean_reversion": 10.0   # 10% (big decrease)
        }
        
        # Calculate expected positions
        expected_tf = self.current_capital * 0.8
        expected_mom = self.current_capital * 0.1
        expected_mr = self.current_capital * 0.1
        
        # Execute the trades
        self.backtester._execute_trades(
            "2023-01-15",
            new_allocations,
            self.current_capital,
            self.current_positions
        )
        
        # Verify positions match expected (accounting for costs)
        # Since TF is a buy, its final position will be expected minus cost
        tf_cost = (expected_tf - 40000.0) * 0.002  # Cost on the additional amount
        self.assertAlmostEqual(
            self.current_positions["trend_following"],
            expected_tf - tf_cost,
            delta=100.0,  # Allow small difference
            msg="Trend following position should match expected minus costs"
        )
        
        # Since MOM is a sell, final position will be exactly the target
        self.assertAlmostEqual(
            self.current_positions["momentum"],
            expected_mom,
            delta=100.0,
            msg="Momentum position should match expected target"
        )
        
        # Since MR is a sell, final position will be exactly the target
        self.assertAlmostEqual(
            self.current_positions["mean_reversion"],
            expected_mr,
            delta=100.0,
            msg="Mean reversion position should match expected target"
        )
    
    def test_multiple_trade_execution(self):
        """Test execution of multiple trade sequences."""
        # First reallocation
        allocations1 = {
            "trend_following": 50.0,
            "momentum": 30.0,
            "mean_reversion": 20.0
        }
        
        # Execute first set of trades
        self.backtester._execute_trades(
            "2023-01-10",
            allocations1,
            self.current_capital,
            self.current_positions
        )
        
        # Record intermediate state
        intermediate_positions = deepcopy(self.current_positions)
        intermediate_costs = self.backtester.total_costs
        
        # Second reallocation
        allocations2 = {
            "trend_following": 30.0,
            "momentum": 50.0,
            "mean_reversion": 20.0
        }
        
        # Execute second set of trades
        self.backtester._execute_trades(
            "2023-01-15",
            allocations2,
            self.current_capital,
            self.current_positions
        )
        
        # Verify cumulative results
        self.assertGreater(
            len(self.backtester.trades), 
            2,
            "Should have multiple trades recorded"
        )
        
        self.assertGreater(
            self.backtester.total_costs,
            intermediate_costs,
            "Costs should increase with more trades"
        )
        
        # TF should now be lower than intermediate
        self.assertLess(
            self.current_positions["trend_following"],
            intermediate_positions["trend_following"],
            "Trend following should decrease in second trade"
        )
        
        # MOM should now be higher than intermediate
        self.assertGreater(
            self.current_positions["momentum"],
            intermediate_positions["momentum"],
            "Momentum should increase in second trade"
        )

if __name__ == '__main__':
    unittest.main() 