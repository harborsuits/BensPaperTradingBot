"""
Unit tests for the ExitStrategyManager.

These tests verify the functionality of the ExitStrategyManager, including:
- Basic exit strategy creation (stop loss, take profit, trailing stop)
- Market condition adaptation and volatility adjustment
- Exit condition checking and execution
- Thread safety and correct cleanup
"""

import unittest
from unittest.mock import MagicMock, patch
import time
import threading
from datetime import datetime, timedelta

# Import modules to test
from trading_bot.strategy.exit_manager import ExitStrategyManager, ExitType, ExitStatus
from trading_bot.position.position_manager import PositionManager
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.event_system import EventBus, Event


class TestExitStrategyManager(unittest.TestCase):
    """Test suite for ExitStrategyManager."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create mocks
        self.position_manager = MagicMock(spec=PositionManager)
        self.broker_manager = MagicMock(spec=MultiBrokerManager)
        self.market_data_service = MagicMock()
        self.event_bus = MagicMock(spec=EventBus)
        
        # Create sample position data
        self.sample_position = {
            'position_id': 'test_position_1',
            'symbol': 'AAPL',
            'direction': 'long',
            'quantity': 100,
            'entry_price': 150.0,
            'entry_date': datetime.now().isoformat(),
            'broker_id': 'test_broker'
        }
        
        # Setup position manager to return our test position
        self.position_manager.get_position.return_value = self.sample_position
        self.position_manager.get_all_positions.return_value = {'test_position_1': self.sample_position}
        
        # Setup market data service to return a price
        self.market_data_service.get_price.return_value = 155.0
        
        # Create the exit strategy manager with our mocks
        self.exit_manager = ExitStrategyManager(
            position_manager=self.position_manager,
            broker_manager=self.broker_manager,
            market_data_service=self.market_data_service,
            event_bus=self.event_bus
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Make sure monitoring is stopped
        self.exit_manager.stop_monitoring()
    
    def test_add_stop_loss(self):
        """Test adding a stop loss to a position."""
        # Add a stop loss at a specific price
        result = self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=145.0
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertIn('test_position_1', self.exit_manager.stop_losses)
        self.assertEqual(self.exit_manager.stop_losses['test_position_1']['price'], 145.0)
        self.assertEqual(self.exit_manager.stop_losses['test_position_1']['status'], ExitStatus.ACTIVE)
        self.assertEqual(self.exit_manager.stop_losses['test_position_1']['type'], ExitType.STOP_LOSS)
    
    def test_add_take_profit(self):
        """Test adding a take profit to a position."""
        # Add a take profit at a specific price
        result = self.exit_manager.add_take_profit(
            position_id='test_position_1',
            price=160.0
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertIn('test_position_1', self.exit_manager.take_profits)
        self.assertEqual(self.exit_manager.take_profits['test_position_1']['price'], 160.0)
        self.assertEqual(self.exit_manager.take_profits['test_position_1']['status'], ExitStatus.ACTIVE)
        self.assertEqual(self.exit_manager.take_profits['test_position_1']['type'], ExitType.TAKE_PROFIT)
    
    def test_add_trailing_stop(self):
        """Test adding a trailing stop to a position."""
        # Add a trailing stop with a percentage
        result = self.exit_manager.add_trailing_stop(
            position_id='test_position_1',
            trail_percent=2.0
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertIn('test_position_1', self.exit_manager.trailing_stops)
        self.assertEqual(self.exit_manager.trailing_stops['test_position_1']['status'], ExitStatus.ACTIVE)
        self.assertEqual(self.exit_manager.trailing_stops['test_position_1']['type'], ExitType.TRAILING_STOP)
        
        # Verify trail amount calculation (2% of entry price)
        expected_trail = 150.0 * 0.02  # 2% of 150.0
        self.assertEqual(self.exit_manager.trailing_stops['test_position_1']['trail_amount'], expected_trail)
    
    def test_update_stop_loss(self):
        """Test updating an existing stop loss."""
        # First, add a stop loss
        self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=145.0
        )
        
        # Then update it
        result = self.exit_manager.update_stop_loss(
            position_id='test_position_1',
            new_price=147.0
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertEqual(self.exit_manager.stop_losses['test_position_1']['price'], 147.0)
    
    def test_move_stop_to_breakeven(self):
        """Test moving a stop loss to breakeven."""
        # First, add a stop loss
        self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=145.0
        )
        
        # Then move to breakeven with a 0.1% buffer
        result = self.exit_manager.move_stop_to_breakeven(
            position_id='test_position_1',
            buffer_percent=0.1
        )
        
        # Verify results
        self.assertTrue(result)
        expected_price = 150.0 * 0.999  # Entry price with 0.1% buffer below
        self.assertAlmostEqual(self.exit_manager.stop_losses['test_position_1']['price'], expected_price)
    
    def test_add_scale_out_strategy(self):
        """Test adding a scale-out strategy."""
        # Add a stop loss first for R-multiple calculation
        self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=145.0
        )
        
        # Add a scale-out strategy
        result = self.exit_manager.add_scale_out_strategy(
            position_id='test_position_1',
            levels=[25, 50, 75, 100]
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertIn('test_position_1', self.exit_manager.scale_outs)
        
        # Check that we have the right number of levels
        scale_out = self.exit_manager.scale_outs['test_position_1']
        self.assertEqual(len(scale_out['levels']), 4)
        
        # Check the first level's quantity (25% of position)
        self.assertEqual(scale_out['levels'][0]['quantity'], 25.0)
    
    def test_add_time_exit(self):
        """Test adding a time-based exit."""
        # Add a time exit for 30 minutes from now
        result = self.exit_manager.add_time_exit(
            position_id='test_position_1',
            duration=30
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertIn('test_position_1', self.exit_manager.time_exits)
        self.assertEqual(self.exit_manager.time_exits['test_position_1']['status'], ExitStatus.ACTIVE)
        self.assertEqual(self.exit_manager.time_exits['test_position_1']['type'], ExitType.TIME_EXIT)
    
    @patch('trading_bot.strategy.exit_manager.time.sleep')
    def test_monitoring_loop(self, mock_sleep):
        """Test the monitoring loop that checks exit conditions."""
        # Setup mock to avoid actual sleeping in tests
        mock_sleep.return_value = None
        
        # Add a stop loss that should be hit
        self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=160.0  # Above current price of 155
        )
        
        # Setup a mock to intercept the execute_exit method
        self.exit_manager._execute_exit = MagicMock(return_value=True)
        
        # Modify our test position to be a short position
        self.sample_position['direction'] = 'short'
        
        # Start monitoring
        self.exit_manager.start_monitoring()
        
        # Give the thread time to check conditions
        time.sleep(0.1)
        
        # Stop monitoring
        self.exit_manager.stop_monitoring()
        
        # Verify that execute_exit was called with the right parameters
        self.exit_manager._execute_exit.assert_called_with(
            'test_position_1',
            ExitType.STOP_LOSS,
            155.0
        )
    
    def test_adapt_to_market_regime(self):
        """Test that exit strategies adapt to different market regimes."""
        # Add a stop loss and trailing stop first
        self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=145.0
        )
        
        self.exit_manager.add_trailing_stop(
            position_id='test_position_1',
            trail_percent=2.0
        )
        
        # Adapt to trending market
        result = self.exit_manager.adapt_to_market_regime(
            position_id='test_position_1',
            regime='trending'
        )
        
        # Verify that the trailing stop was widened for trending regime
        self.assertTrue(result)
        expected_trail = 150.0 * 0.02 * 3.0  # 2% * 3 = 6% of entry
        self.assertEqual(self.exit_manager.trailing_stops['test_position_1']['trail_amount'], expected_trail)
    
    def test_adjust_exits_for_volatility(self):
        """Test that exits adjust for market volatility."""
        # Add a stop loss and trailing stop first
        self.exit_manager.add_stop_loss(
            position_id='test_position_1',
            price=145.0
        )
        
        self.exit_manager.add_trailing_stop(
            position_id='test_position_1',
            trail_percent=2.0
        )
        
        # Setup ATR value in market data service
        self.market_data_service.get_indicator.return_value = 3.0  # ATR of 3 points
        
        # Adjust for volatility
        result = self.exit_manager.adjust_exits_for_volatility(
            position_id='test_position_1'
        )
        
        # Verify that the trailing stop was adjusted based on ATR
        self.assertTrue(result)
        self.assertEqual(self.exit_manager.trailing_stops['test_position_1']['trail_amount'], 3.0)
    
    def test_add_chandelier_exit(self):
        """Test adding a chandelier exit strategy."""
        # Setup required mocks for historical data
        self.market_data_service.get_indicator.return_value = 3.0  # ATR of 3 points
        self.market_data_service.get_historic_ohlc.return_value = [
            {'high': 155.0, 'low': 145.0},
            {'high': 158.0, 'low': 148.0},
            {'high': 160.0, 'low': 150.0}  # This should be the extreme
        ]
        
        # Add a chandelier exit
        result = self.exit_manager.add_chandelier_exit(
            position_id='test_position_1',
            periods=3,
            multiplier=2.0
        )
        
        # Verify results
        self.assertTrue(result)
        
        # The key is a special chandelier key
        chandelier_key = 'test_position_1-chandelier'
        self.assertIn(chandelier_key, self.exit_manager.trailing_stops)
        
        # Verify that reference price is highest high (160.0)
        self.assertEqual(self.exit_manager.trailing_stops[chandelier_key]['reference_price'], 160.0)
        
        # Verify stop price calculation (highest high - 2*ATR)
        expected_stop = 160.0 - (3.0 * 2.0)
        self.assertEqual(self.exit_manager.trailing_stops[chandelier_key]['stop_price'], expected_stop)
        self.assertEqual(self.exit_manager.trailing_stops[chandelier_key]['type'], ExitType.CHANDELIER_EXIT)


if __name__ == '__main__':
    unittest.main()
