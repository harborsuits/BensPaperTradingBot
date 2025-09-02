"""
Tests for the Adaptive Strategy Controller that integrates performance tracking,
market regime detection, and risk management.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil
import sys
import logging
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.analytics.market_regime_detector import MarketRegime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestAdaptiveStrategyController(unittest.TestCase):
    """Test cases for the AdaptiveStrategyController"""

    def setUp(self):
        """Set up test environment"""
        # Create temp directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        
        # Configuration with test directories
        self.config = {
            'initial_equity': 10000.0,
            'allocation_frequency': 'daily',
            'parameter_update_frequency': 'daily',
            'performance_tracker': {
                'data_dir': os.path.join(self.test_data_dir, 'performance'),
                'auto_save': False
            },
            'market_regime_detector': {
                'data_dir': os.path.join(self.test_data_dir, 'regimes'),
                'auto_save': False
            },
            'snowball_allocator': {
                'rebalance_frequency': 'daily',
                'snowball_reinvestment_ratio': 0.5,
                'min_weight': 0.05,
                'max_weight': 0.9,
                'normalization_method': 'simple'
            }
        }
        
        # Create controller
        self.controller = AdaptiveStrategyController(config=self.config)
        
        # Register test strategies
        self.test_strategies = {
            'trend_strategy': {
                'name': 'Trend Following Strategy',
                'description': 'Follows market trends',
                'category': 'trend_following',
                'symbols': ['SPY', 'QQQ'],
                'timeframes': ['1d'],
                'parameters': {
                    'ma_fast': 20,
                    'ma_slow': 50,
                    'trailing_stop_pct': 2.0,
                    'profit_target_pct': 3.0
                }
            },
            'mean_reversion': {
                'name': 'Mean Reversion Strategy',
                'description': 'Trades mean reversion',
                'category': 'mean_reversion',
                'symbols': ['SPY', 'IWM'],
                'timeframes': ['1d'],
                'parameters': {
                    'entry_threshold': 2.0,
                    'profit_target_pct': 1.5,
                    'stop_loss_pct': 1.0
                }
            },
            'breakout_strategy': {
                'name': 'Breakout Strategy',
                'description': 'Trades breakouts',
                'category': 'breakout',
                'symbols': ['QQQ', 'IWM'],
                'timeframes': ['1d'],
                'parameters': {
                    'breakout_threshold': 2.0,
                    'confirmation_period': 3,
                    'trailing_stop_pct': 2.0
                }
            }
        }
        
        # Register strategies with controller
        for strategy_id, metadata in self.test_strategies.items():
            self.controller.register_strategy(strategy_id, metadata)
            
        # Create sample market data
        self.create_sample_market_data()

    def tearDown(self):
        """Clean up after tests"""
        # Remove temp directory
        shutil.rmtree(self.test_data_dir)

    def create_sample_market_data(self):
        """Create sample market data for testing"""
        # Create sample data for SPY
        # Date range for 100 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=100)
        dates = [start_date + timedelta(days=i) for i in range(101)]
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        
        # Create trending up market
        close_trending_up = [100.0]
        for i in range(1, 101):
            # Add some randomness to an uptrend
            close_trending_up.append(close_trending_up[-1] * (1 + np.random.normal(0.001, 0.01)))
            
        self.spy_trending_up = pd.DataFrame({
            'date': date_strs,
            'open': close_trending_up,
            'high': [p * 1.01 for p in close_trending_up],
            'low': [p * 0.99 for p in close_trending_up],
            'close': close_trending_up,
            'volume': [1000000 + np.random.randint(0, 500000) for _ in range(101)]
        })
        
        # Create ranging market
        close_ranging = [100.0]
        for i in range(1, 101):
            # Mean-reverting pattern around 100
            close_ranging.append(100.0 + np.random.normal(0, 2.0))
            
        self.spy_ranging = pd.DataFrame({
            'date': date_strs,
            'open': close_ranging,
            'high': [p * 1.01 for p in close_ranging],
            'low': [p * 0.99 for p in close_ranging],
            'close': close_ranging,
            'volume': [1000000 + np.random.randint(0, 500000) for _ in range(101)]
        })
        
        # Create volatile market
        close_volatile = [100.0]
        for i in range(1, 101):
            # High volatility moves
            close_volatile.append(close_volatile[-1] * (1 + np.random.normal(0, 0.03)))
            
        self.spy_volatile = pd.DataFrame({
            'date': date_strs,
            'open': close_volatile,
            'high': [p * 1.03 for p in close_volatile],
            'low': [p * 0.97 for p in close_volatile],
            'close': close_volatile,
            'volume': [1500000 + np.random.randint(0, 1000000) for _ in range(101)]
        })

    def test_strategy_registration(self):
        """Test strategy registration and retrieval"""
        # Check if strategies were registered
        strategy_statuses = self.controller.get_all_strategy_statuses()
        
        # Should have 3 strategies
        self.assertEqual(len(strategy_statuses), 3)
        
        # All strategies should be active
        for strategy_id, status in strategy_statuses.items():
            self.assertTrue(status['active'])
            
        # Test deregistration
        result = self.controller.deregister_strategy('trend_strategy')
        self.assertTrue(result)
        
        # Should have 2 strategies now
        strategy_statuses = self.controller.get_all_strategy_statuses()
        self.assertEqual(len(strategy_statuses), 2)
        
        # Test deregistering non-existent strategy
        result = self.controller.deregister_strategy('non_existent')
        self.assertFalse(result)

    def test_market_data_update(self):
        """Test updating market data and regime detection"""
        # Update with trending data
        self.controller.update_market_data('SPY', self.spy_trending_up)
        
        # Get regimes
        regimes = self.controller.get_market_regimes()
        
        # Should have SPY regime
        self.assertIn('SPY', regimes)
        
        # Check suitability scores
        suitability = self.controller.market_regime_detector.get_strategy_suitability('SPY')
        
        # In trending market, trend following should be most suitable
        self.assertGreater(suitability.get('trend_following', 0), 
                           suitability.get('mean_reversion', 0))

    def test_record_trade_results(self):
        """Test recording trade results and performance tracking"""
        # Create sample trade
        trade_data = {
            'entry_time': '2023-01-01T10:00:00',
            'exit_time': '2023-01-01T16:00:00',
            'symbol': 'SPY',
            'direction': 'long',
            'entry_price': 100.0,
            'exit_price': 102.0,
            'quantity': 10,
            'pnl': 20.0,
            'pnl_pct': 0.02,
            'fees': 1.0,
            'slippage': 0.05
        }
        
        # Record trade for trend strategy
        metrics = self.controller.record_trade_result('trend_strategy', trade_data)
        
        # Should have metrics
        self.assertTrue(metrics)
        
        # Win rate should be 1.0 (100%)
        self.assertEqual(metrics.get('win_rate', 0), 1.0)
        
        # Record a losing trade
        trade_data_loss = {
            'entry_time': '2023-01-02T10:00:00',
            'exit_time': '2023-01-02T16:00:00',
            'symbol': 'SPY',
            'direction': 'long',
            'entry_price': 102.0,
            'exit_price': 101.0,
            'quantity': 10,
            'pnl': -10.0,
            'pnl_pct': -0.01,
            'fees': 1.0,
            'slippage': 0.05
        }
        
        # Record trade
        metrics = self.controller.record_trade_result('trend_strategy', trade_data_loss)
        
        # Win rate should be 0.5 (50%)
        self.assertEqual(metrics.get('win_rate', 0), 0.5)
        
        # Test strategy status
        status = self.controller.get_strategy_status('trend_strategy')
        
        # Should have performance metrics
        self.assertIn('performance', status)
        self.assertEqual(status['performance']['total_trades'], 2)

    def test_allocations(self):
        """Test strategy allocations"""
        # Record trades to generate performance data
        # Trend strategy has good performance
        for i in range(5):
            self.controller.record_trade_result('trend_strategy', {
                'entry_time': f'2023-01-0{i+1}T10:00:00',
                'exit_time': f'2023-01-0{i+1}T16:00:00',
                'symbol': 'SPY',
                'direction': 'long',
                'entry_price': 100.0,
                'exit_price': 102.0,
                'quantity': 10,
                'pnl': 20.0,
                'pnl_pct': 0.02,
                'fees': 1.0
            })
        
        # Mean reversion has poor performance
        for i in range(5):
            self.controller.record_trade_result('mean_reversion', {
                'entry_time': f'2023-01-0{i+1}T10:00:00',
                'exit_time': f'2023-01-0{i+1}T16:00:00',
                'symbol': 'SPY',
                'direction': 'long',
                'entry_price': 100.0,
                'exit_price': 99.0,
                'quantity': 10,
                'pnl': -10.0,
                'pnl_pct': -0.01,
                'fees': 1.0
            })
        
        # Force allocation update
        self.controller._update_allocations()
        
        # Get allocations
        allocations = self.controller.get_all_allocations()
        
        # Trend strategy should have higher allocation than mean reversion
        self.assertGreater(allocations.get('trend_strategy', 0), 
                           allocations.get('mean_reversion', 0))
        
        # Test allocation override
        self.controller.set_allocation_override('mean_reversion', 0.5)
        
        # Get allocation after override
        allocation = self.controller.get_strategy_allocation('mean_reversion')
        
        # Should be 0.5
        self.assertEqual(allocation, 0.5)
        
        # Clear override
        self.controller.clear_allocation_override('mean_reversion')
        
        # Get allocation after clearing override
        allocation = self.controller.get_strategy_allocation('mean_reversion')
        
        # Should not be 0.5 anymore
        self.assertNotEqual(allocation, 0.5)

    def test_position_sizing(self):
        """Test position sizing calculations"""
        # Set up market regime
        self.controller.update_market_data('SPY', self.spy_trending_up)
        
        # Record some trades for snowball allocation
        for i in range(3):
            self.controller.record_trade_result('trend_strategy', {
                'entry_time': f'2023-01-0{i+1}T10:00:00',
                'exit_time': f'2023-01-0{i+1}T16:00:00',
                'symbol': 'SPY',
                'direction': 'long',
                'entry_price': 100.0,
                'exit_price': 102.0,
                'quantity': 10,
                'pnl': 20.0,
                'pnl_pct': 0.02,
                'fees': 1.0
            })
        
        # Update allocations
        self.controller._update_allocations()
        
        # Calculate position size
        position_info = self.controller.get_position_size(
            strategy_id='trend_strategy',
            symbol='SPY',
            entry_price=100.0,
            stop_loss=98.0
        )
        
        # Should have size information
        self.assertIn('size', position_info)
        self.assertIn('notional', position_info)
        self.assertIn('allocation', position_info)
        
        # Position should be positive
        self.assertGreater(position_info['size'], 0)
        
        # Notional should be price * size
        self.assertAlmostEqual(position_info['notional'], 
                              position_info['size'] * 100.0)

    def test_parameter_adjustments(self):
        """Test parameter adjustments based on market regime"""
        # Set up trending market regime
        self.controller.update_market_data('SPY', self.spy_trending_up)
        
        # Get parameters for trend strategy
        trend_params = self.controller.get_strategy_parameters(
            strategy_id='trend_strategy',
            symbol='SPY'
        )
        
        # Store original trailing stop
        original_stop = trend_params.get('trailing_stop_pct', 0)
        
        # Set up ranging market regime
        self.controller.update_market_data('SPY', self.spy_ranging)
        
        # Get parameters after regime change
        trend_params_ranging = self.controller.get_strategy_parameters(
            strategy_id='trend_strategy',
            symbol='SPY'
        )
        
        # Parameters should be different in different regimes
        self.assertNotEqual(original_stop, 
                           trend_params_ranging.get('trailing_stop_pct', 0))
        
        # Test parameter override
        self.controller.set_parameter_override('trend_strategy', {
            'trailing_stop_pct': 5.0
        })
        
        # Get parameters after override
        trend_params_override = self.controller.get_strategy_parameters(
            strategy_id='trend_strategy',
            symbol='SPY'
        )
        
        # Should have override value
        self.assertEqual(trend_params_override.get('trailing_stop_pct', 0), 5.0)
        
        # Clear override
        self.controller.clear_parameter_override('trend_strategy', ['trailing_stop_pct'])
        
        # Get parameters after clearing override
        trend_params_after = self.controller.get_strategy_parameters(
            strategy_id='trend_strategy',
            symbol='SPY'
        )
        
        # Should not have override value anymore
        self.assertNotEqual(trend_params_after.get('trailing_stop_pct', 0), 5.0)

    def test_emergency_brake(self):
        """Test emergency brake integration"""
        # Record consecutive losing trades to trigger emergency brake
        for i in range(4):  # Default max_consecutive_losses is 3
            self.controller.record_trade_result('trend_strategy', {
                'entry_time': f'2023-01-0{i+1}T10:00:00',
                'exit_time': f'2023-01-0{i+1}T16:00:00',
                'symbol': 'SPY',
                'direction': 'long',
                'entry_price': 100.0,
                'exit_price': 98.0,
                'quantity': 10,
                'pnl': -20.0,
                'pnl_pct': -0.02,
                'fees': 1.0
            })
        
        # Check if strategy is paused
        status = self.controller.get_strategy_status('trend_strategy')
        
        # Should be paused by emergency brake
        self.assertTrue(status['emergency_brake_active'])
        self.assertFalse(status['active'])
        
        # Attempt to get position size for paused strategy
        position_info = self.controller.get_position_size(
            strategy_id='trend_strategy',
            symbol='SPY',
            entry_price=100.0
        )
        
        # Size should be 0 for paused strategy
        self.assertEqual(position_info['size'], 0)
        
        # Manually resume strategy
        result = self.controller.resume_strategy('trend_strategy')
        
        # Should not be able to resume due to emergency brake
        self.assertFalse(result)
        
        # Reset emergency brake in controller
        self.controller.emergency_brake.reset_strategy('trend_strategy')
        
        # Now should be able to resume
        result = self.controller.resume_strategy('trend_strategy')
        self.assertTrue(result)
        
        # Check status after resume
        status = self.controller.get_strategy_status('trend_strategy')
        self.assertTrue(status['active'])
        self.assertFalse(status['emergency_brake_active'])

    def test_market_regime_detection(self):
        """Test market regime detection functionality"""
        # Test trending market
        self.controller.update_market_data('SPY', self.spy_trending_up)
        regimes = self.controller.get_market_regimes()
        self.assertIn('SPY', regimes)
        
        # Test ranging market
        self.controller.update_market_data('SPY', self.spy_ranging)
        regimes = self.controller.get_market_regimes()
        self.assertIn('SPY', regimes)
        
        # Test volatile market
        self.controller.update_market_data('SPY', self.spy_volatile)
        regimes = self.controller.get_market_regimes()
        self.assertIn('SPY', regimes)
        
        # Get strategy suitability scores
        suitability = self.controller.market_regime_detector.get_strategy_suitability('SPY')
        
        # Should have scores for different strategy types
        self.assertIn('trend_following', suitability)
        self.assertIn('mean_reversion', suitability)
        self.assertIn('breakout', suitability)
        self.assertIn('volatility', suitability)
        
        # Test optimal parameters
        params = self.controller.market_regime_detector.get_optimal_parameters(
            symbol='SPY',
            strategy_type='trend_following'
        )
        
        # Should have recommended parameters
        self.assertTrue(params)


if __name__ == '__main__':
    unittest.main()
