import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from trading_bot.backtesting.unified_backtester import UnifiedBacktester

class TestUnifiedBacktester(unittest.TestCase):
    """Test suite for the UnifiedBacktester class."""
    
    def setUp(self):
        """Set up a test backtester instance with mock data."""
        self.strategies = ["trend_following", "momentum", "mean_reversion"]
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 31)
        
        # Create a test instance with mock data
        self.backtester = UnifiedBacktester(
            initial_capital=100000.0,
            strategies=self.strategies,
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d"),
            rebalance_frequency="weekly",
            use_mock=True,
            debug_mode=True
        )
        
        # Override data generation methods to use our test data
        self.backtester._generate_mock_strategy_data = self._mock_strategy_data
        self.backtester._generate_mock_market_data = self._mock_market_data
        self.backtester._generate_mock_regime_data = self._mock_regime_data
        
        # Load mock data
        self.backtester.load_strategy_data()
        self.backtester.load_market_data()
        self.backtester.load_regime_data()
        
    def _mock_strategy_data(self, strategy_name):
        """Generate consistent mock strategy data for testing."""
        dates = pd.date_range(self.start_date.date(), self.end_date.date(), freq='B')
        returns = []
        
        # Generate predictable returns for each strategy
        if strategy_name == "trend_following":
            returns = [0.001 if i % 3 == 0 else -0.0005 for i in range(len(dates))]
        elif strategy_name == "momentum":
            returns = [0.002 if i % 5 == 0 else 0.0003 for i in range(len(dates))]
        elif strategy_name == "mean_reversion":
            returns = [-0.001 if i % 4 == 0 else 0.001 for i in range(len(dates))]
        
        # Create DataFrame with returns and equity curve
        df = pd.DataFrame(index=dates, data={'return': returns})
        df['equity_curve'] = (1 + df['return']).cumprod()
        return df
    
    def _mock_market_data(self):
        """Generate mock market data for testing."""
        dates = pd.date_range(self.start_date.date(), self.end_date.date(), freq='B')
        
        # Create mock VIX values with a volatility spike
        vix_values = np.ones(len(dates)) * 15  # Base VIX level
        vix_values[5:10] = 25  # Volatility spike
        
        # Create mock index values with a trend
        index_values = np.linspace(100, 105, len(dates))
        
        return pd.DataFrame(
            index=dates,
            data={
                'vix': vix_values,
                'index_value': index_values
            }
        )
    
    def _mock_regime_data(self):
        """Generate mock regime data for testing."""
        dates = pd.date_range(self.start_date.date(), self.end_date.date(), freq='B')
        
        # Create different market regimes
        regimes = ['neutral'] * len(dates)
        
        # Add some regime changes
        regimes[0:5] = ['bullish'] * 5
        regimes[5:10] = ['volatile'] * 5
        regimes[10:15] = ['bearish'] * 5
        
        return pd.DataFrame(
            index=dates,
            data={'regime': regimes}
        )
    
    def test_initialization(self):
        """Test that the backtester initializes correctly."""
        self.assertEqual(self.backtester.initial_capital, 100000.0)
        self.assertEqual(self.backtester.strategies, self.strategies)
        self.assertEqual(self.backtester.rebalance_frequency, "weekly")
        self.assertTrue(self.backtester.debug_mode)
        
        # Check that initial allocations are created
        self.assertEqual(len(self.backtester.initial_allocations), len(self.strategies))
        total_allocation = sum(self.backtester.initial_allocations.values())
        self.assertAlmostEqual(total_allocation, 100.0)  # Should sum to 100%
    
    def test_load_strategy_data(self):
        """Test that strategy data is loaded correctly."""
        # Data should be loaded in setUp
        for strategy in self.strategies:
            self.assertIn(strategy, self.backtester.strategy_data)
            self.assertIsInstance(self.backtester.strategy_data[strategy], pd.DataFrame)
            self.assertIn('return', self.backtester.strategy_data[strategy].columns)
            self.assertIn('equity_curve', self.backtester.strategy_data[strategy].columns)
    
    def test_market_context_calculation(self):
        """Test that market context is calculated correctly."""
        test_date = datetime(2023, 1, 9)  # During volatile regime
        context = self.backtester._get_historical_market_context(test_date)
        
        self.assertEqual(context['date'], test_date)
        self.assertEqual(context['market_regime'], 'volatile')
        self.assertGreater(context['volatility'], 0.5)  # VIX is 25 during this period
        
        # Test normal regime
        test_date = datetime(2023, 1, 20)  # During neutral regime
        context = self.backtester._get_historical_market_context(test_date)
        self.assertEqual(context['market_regime'], 'neutral')
    
    def test_rebalance_detection(self):
        """Test that rebalance days are detected correctly."""
        # Monday (should rebalance)
        monday = datetime(2023, 1, 2)
        self.assertTrue(self.backtester._should_rebalance(monday))
        
        # Tuesday (should not rebalance)
        tuesday = datetime(2023, 1, 3)
        self.assertFalse(self.backtester._should_rebalance(tuesday))
        
        # Test with daily rebalancing
        self.backtester.rebalance_frequency = "daily"
        self.assertTrue(self.backtester._should_rebalance(tuesday))
        
        # Test with monthly rebalancing
        self.backtester.rebalance_frequency = "monthly"
        first_of_month = datetime(2023, 1, 1)
        middle_of_month = datetime(2023, 1, 15)
        self.assertTrue(self.backtester._should_rebalance(first_of_month))
        self.assertFalse(self.backtester._should_rebalance(middle_of_month))
    
    def test_execute_trades(self):
        """Test that trades are executed correctly."""
        # Setup current positions and allocations
        current_capital = 100000.0
        current_positions = {
            'trend_following': 40000.0,
            'momentum': 30000.0,
            'mean_reversion': 30000.0,
        }
        
        # Define new allocations (significant change)
        new_allocations = {
            'trend_following': 20.0,  # 20% (from 40%)
            'momentum': 50.0,        # 50% (from 30%)
            'mean_reversion': 30.0   # 30% (unchanged)
        }
        
        # Execute trades
        date_str = "2023-01-10"
        self.backtester._execute_trades(date_str, new_allocations, current_capital, current_positions)
        
        # Check that positions were updated
        self.assertAlmostEqual(current_positions['trend_following'], 20000.0 * (1 - 0.001), places=1)  # Minus trading cost
        self.assertAlmostEqual(current_positions['momentum'], 50000.0 * (1 - 0.001), places=1)  # Minus trading cost
        self.assertAlmostEqual(current_positions['mean_reversion'], 30000.0, places=1)  # Unchanged
        
        # Check that trades were recorded
        self.assertEqual(len(self.backtester.trades), 2)  # Two trades executed
        self.assertEqual(self.backtester.trades[0]['direction'], 'sell')  # Reduced trend_following
        self.assertEqual(self.backtester.trades[1]['direction'], 'buy')   # Increased momentum
        
        # Test trade cost calculation
        total_costs = self.backtester.total_costs
        self.assertGreater(total_costs, 0)
        
        # Test minimum trade threshold
        small_change_allocations = {
            'trend_following': 20.1,  # 0.1% change
            'momentum': 49.9,         # 0.1% change
            'mean_reversion': 30.0     # No change
        }
        
        # Save the current trade count
        previous_trade_count = len(self.backtester.trades)
        
        # Execute trades with small changes
        self.backtester._execute_trades(date_str, small_change_allocations, current_capital, current_positions)
        
        # No new trades should be executed due to min_trade_value
        self.assertEqual(len(self.backtester.trades), previous_trade_count)
    
    def test_volatility_adjustment(self):
        """Test that volatility adjustments are calculated correctly."""
        # Setup market context with different volatility levels
        low_vol_context = {"volatility": 0.2, "market_regime": "neutral"}
        normal_vol_context = {"volatility": 0.4, "market_regime": "neutral"}
        high_vol_context = {"volatility": 0.8, "market_regime": "volatile"}
        
        # Get adjustments
        low_adj = self.backtester._calculate_volatility_adjustment(low_vol_context)
        normal_adj = self.backtester._calculate_volatility_adjustment(normal_vol_context)
        high_adj = self.backtester._calculate_volatility_adjustment(high_vol_context)
        
        # Lower volatility should allow more aggressive sizing
        self.assertGreater(low_adj, normal_adj)
        
        # Higher volatility should reduce position sizing
        self.assertLess(high_adj, normal_adj)
        
        # Volatile regimes should further reduce position sizing
        high_vol_neutral_context = {"volatility": 0.8, "market_regime": "neutral"}
        high_vol_neutral_adj = self.backtester._calculate_volatility_adjustment(high_vol_neutral_context)
        self.assertLess(high_adj, high_vol_neutral_adj)  # 'volatile' regime reduces more than 'neutral'
    
    def test_circuit_breaker_limits(self):
        """Test that circuit breaker limits are applied correctly."""
        # Setup allocations
        current_allocations = {
            'trend_following': 40.0,
            'momentum': 30.0,
            'mean_reversion': 30.0
        }
        
        # Target allocations with large changes
        target_allocations = {
            'trend_following': 10.0,  # 30% decrease
            'momentum': 60.0,        # 30% increase
            'mean_reversion': 30.0   # unchanged
        }
        
        # Setup circuit breaker status
        circuit_breaker_level1 = {
            'active': True,
            'level': 1,  # 15% maximum change
            'trigger_cause': 'Volatility spike'
        }
        
        circuit_breaker_level3 = {
            'active': True,
            'level': 3,  # 5% maximum change
            'trigger_cause': 'Severe market dislocation'
        }
        
        # Mock the portfolio history
        self.backtester.portfolio_history = [{
            'capital': 100000.0,
            'positions': {
                'trend_following': 40000.0,
                'momentum': 30000.0,
                'mean_reversion': 30000.0
            }
        }]
        
        # Apply level 1 circuit breaker
        modified_allocations = self.backtester._apply_circuit_breaker_limits(
            target_allocations.copy(), circuit_breaker_level1
        )
        
        # Changes should be limited to 15%
        self.assertGreaterEqual(modified_allocations['trend_following'], 25.0)  # Max 15% decrease
        self.assertLessEqual(modified_allocations['momentum'], 45.0)  # Max 15% increase
        
        # Apply level 3 circuit breaker
        modified_allocations = self.backtester._apply_circuit_breaker_limits(
            target_allocations.copy(), circuit_breaker_level3
        )
        
        # Changes should be limited to 5%
        self.assertGreaterEqual(modified_allocations['trend_following'], 35.0)  # Max 5% decrease
        self.assertLessEqual(modified_allocations['momentum'], 35.0)  # Max 5% increase
        
        # Check allocation sum
        allocation_sum = sum(modified_allocations.values())
        self.assertAlmostEqual(allocation_sum, 100.0, places=1)  # Should still sum to 100%
    
    def test_emergency_risk_controls(self):
        """Test that emergency risk controls work correctly."""
        # Setup portfolio history with some volatile returns for one strategy
        self.backtester.portfolio_history = []
        
        # Create 10 days of history
        current_date = self.start_date
        for i in range(10):
            # Add daily entry
            positions = {
                'trend_following': 40000.0 * (1 - 0.05 * i if i > 5 else 1),  # Declining value for trend_following
                'momentum': 30000.0 * (1 + 0.01 * i),  # Slightly increasing
                'mean_reversion': 30000.0 * (1 + 0.005 * i)  # Slightly increasing
            }
            
            capital = sum(positions.values())
            self.backtester.portfolio_history.append({
                'date': current_date + timedelta(days=i),
                'capital': capital,
                'positions': positions.copy(),
                'daily_return': 0.0 if i == 0 else (capital / prev_capital - 1)
            })
            prev_capital = capital
        
        # Current positions and capital (last day)
        current_positions = self.backtester.portfolio_history[-1]['positions'].copy()
        current_capital = self.backtester.portfolio_history[-1]['capital']
        
        # Apply emergency risk controls
        self.backtester._apply_emergency_risk_controls(current_date, current_positions, current_capital)
        
        # Check that trend_following allocation was reduced
        trend_following_alloc_before = 40000.0 * (1 - 0.05 * 9) / current_capital * 100
        trend_following_alloc_after = current_positions['trend_following'] / sum(current_positions.values()) * 100
        
        self.assertLess(trend_following_alloc_after, trend_following_alloc_before)
        
        # Check that debug data was recorded
        self.assertEqual(len(self.backtester.debug_data), 1)
        self.assertEqual(self.backtester.debug_data[0]['type'], 'emergency_risk_control')
    
    def test_stress_test_adjustment(self):
        """Test that stress test adjustments are calculated correctly."""
        # Various stress test results
        low_risk = {'risk_level': 'low', 'projected_max_drawdown': -5.0}
        medium_risk = {'risk_level': 'medium', 'projected_max_drawdown': -10.0}
        high_risk = {'risk_level': 'high', 'projected_max_drawdown': -20.0}
        extreme_risk = {'risk_level': 'extreme', 'projected_max_drawdown': -30.0}
        
        # Calculate adjustments
        low_adj = self.backtester._get_stress_test_adjustment(low_risk)
        medium_adj = self.backtester._get_stress_test_adjustment(medium_risk)
        high_adj = self.backtester._get_stress_test_adjustment(high_risk)
        extreme_adj = self.backtester._get_stress_test_adjustment(extreme_risk)
        
        # Check that higher risk levels result in more reduction
        self.assertGreater(low_adj, medium_adj)
        self.assertGreater(medium_adj, high_adj)
        self.assertGreater(high_adj, extreme_adj)
        
        # Check that large projected drawdowns lead to further reductions
        large_drawdown = {'risk_level': 'high', 'projected_max_drawdown': -25.0}
        large_dd_adj = self.backtester._get_stress_test_adjustment(large_drawdown)
        self.assertLess(large_dd_adj, high_adj)
    
    @patch('trading_bot.backtesting.unified_backtester.UnifiedBacktester.step')
    def test_run_backtest(self, mock_step):
        """Test the overall backtest execution."""
        # Setup mocks
        mock_step.return_value = None
        
        # Run the backtest
        results = self.backtester.run_backtest()
        
        # Check that step was called for each business day
        business_days = pd.date_range(
            start=self.start_date + timedelta(days=1),
            end=self.end_date,
            freq='B'
        )
        self.assertEqual(mock_step.call_count, len(business_days))
        
        # Check that results contain expected metrics
        self.assertIn('final_capital', results)
        self.assertIn('total_return_pct', results)
        self.assertIn('annual_return_pct', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown_pct', results)
        self.assertIn('performance_report', results)
    
    def test_risk_scenario_simulation(self):
        """Test the risk scenario simulation."""
        # Run a backtest first to create portfolio_df
        with patch('trading_bot.backtesting.unified_backtester.UnifiedBacktester.step'):
            self.backtester.run_backtest()
            
            # Create a mock portfolio_df
            dates = pd.date_range(self.start_date.date(), self.end_date.date(), freq='B')
            portfolio_values = 100000 * (1 + np.linspace(0, 0.1, len(dates)))
            self.backtester.portfolio_df = pd.DataFrame(
                index=dates,
                data={'portfolio_value': portfolio_values}
            )
        
        # Run the market crash scenario
        scenario_results = self.backtester.simulate_risk_scenario(
            scenario_type='market_crash',
            duration_days=5,
            severity=0.8
        )
        
        # Check that scenario results are returned
        self.assertIn('scenario_type', scenario_results)
        self.assertEqual(scenario_results['scenario_type'], 'market_crash')
        self.assertIn('severity', scenario_results)
        self.assertIn('risk_responses', scenario_results)
        
        # Check that performance metrics are included
        self.assertIn('performance', scenario_results)
        self.assertIn('scenario_return', scenario_results['performance'])
        self.assertIn('scenario_max_drawdown', scenario_results['performance'])

    def test_process_backtest_results(self):
        """Test processing of backtest results."""
        # Setup sample portfolio history
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        self.backtester.portfolio_history = []
        
        # Create portfolio history with some returns
        initial_capital = 100000.0
        current_capital = initial_capital
        for i, date in enumerate(dates):
            daily_return = 0.002 if i % 3 == 0 else -0.001  # Some up and down days
            current_capital *= (1 + daily_return)
            positions = {
                'trend_following': current_capital * 0.4,
                'momentum': current_capital * 0.3,
                'mean_reversion': current_capital * 0.3
            }
            
            self.backtester.portfolio_history.append({
                'date': date,
                'capital': current_capital,
                'positions': positions.copy(),
                'daily_return': daily_return
            })
        
        # Process results
        self.backtester._process_backtest_results()
        
        # Check that portfolio_df was created
        self.assertTrue(hasattr(self.backtester, 'portfolio_df'))
        self.assertIsInstance(self.backtester.portfolio_df, pd.DataFrame)
        self.assertEqual(len(self.backtester.portfolio_df), len(dates))
        
        # Check that portfolio values and returns are calculated
        self.assertIn('portfolio_value', self.backtester.portfolio_df.columns)
        self.assertIn('daily_return', self.backtester.portfolio_df.columns)
        self.assertIn('cumulative_return', self.backtester.portfolio_df.columns)
        
        # Check benchmark comparison if available
        if not self.backtester.benchmark_data.empty:
            self.assertIn('benchmark_value', self.backtester.portfolio_df.columns)
            self.assertIn('benchmark_return', self.backtester.portfolio_df.columns)
            
    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics."""
        # Setup sample portfolio history and process it
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        self.backtester.portfolio_history = []
        
        # Create portfolio history with a pattern of returns
        initial_capital = 100000.0
        current_capital = initial_capital
        
        for i, date in enumerate(dates):
            # Create a pattern with some volatility and drawdown
            if i < 5:
                daily_return = 0.005  # Initial growth
            elif i < 10:
                daily_return = -0.008  # Drawdown
            else:
                daily_return = 0.003  # Recovery
                
            current_capital *= (1 + daily_return)
            positions = {
                'trend_following': current_capital * 0.4,
                'momentum': current_capital * 0.3,
                'mean_reversion': current_capital * 0.3
            }
            
            self.backtester.portfolio_history.append({
                'date': date,
                'capital': current_capital,
                'positions': positions.copy(),
                'daily_return': daily_return
            })
        
        # Process results to create portfolio_df
        self.backtester._process_backtest_results()
        
        # Calculate performance metrics
        metrics = self.backtester._calculate_performance_metrics()
        
        # Check that basic metrics are calculated
        self.assertIn('initial_capital', metrics)
        self.assertIn('final_capital', metrics)
        self.assertIn('total_return_pct', metrics)
        self.assertIn('annual_return_pct', metrics)
        self.assertIn('volatility_pct', metrics)
        self.assertIn('max_drawdown_pct', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('win_rate_pct', metrics)
        
        # Check that drawdown metrics are calculated
        self.assertIn('drawdowns', metrics)
        self.assertIn('drawdown_series', metrics['drawdowns'])
        
        # Check that calculated values are reasonable
        self.assertGreater(metrics['final_capital'], 0)
        self.assertLess(metrics['max_drawdown_pct'], 0)  # Drawdowns are negative
        self.assertGreater(metrics['volatility_pct'], 0)
        
        # Final capital should match last portfolio history entry
        self.assertAlmostEqual(
            metrics['final_capital'], 
            self.backtester.portfolio_history[-1]['capital'],
            delta=0.01
        )
        
if __name__ == '__main__':
    unittest.main() 