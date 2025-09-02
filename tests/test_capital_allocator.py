"""
Unit tests for the CapitalAllocator.

These tests verify the functionality of the CapitalAllocator, including:
- Allocation methods (performance-based, risk parity, Kelly, etc.)
- Position sizing and adjustment
- Strategy capital allocation
- Drawdown protection
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime

# Import modules to test
from trading_bot.portfolio.capital_allocator import CapitalAllocator, AllocationMethod
from trading_bot.accounting.performance_metrics import PerformanceMetrics
from trading_bot.position.position_manager import PositionManager
from trading_bot.accounting.pnl_calculator import PnLCalculator
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager


class TestCapitalAllocator(unittest.TestCase):
    """Test suite for CapitalAllocator."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create mocks
        self.performance_metrics = MagicMock(spec=PerformanceMetrics)
        self.position_manager = MagicMock(spec=PositionManager)
        self.pnl_calculator = MagicMock(spec=PnLCalculator)
        self.broker_manager = MagicMock(spec=MultiBrokerManager)
        
        # Create test configuration
        self.config = {
            'initial_capital': 100000.0,
            'allocation_method': AllocationMethod.PERFORMANCE,
            'strategy_ids': ['strategy1', 'strategy2', 'strategy3'],
            'min_allocation_pct': 10.0,
            'max_allocation_pct': 40.0,
            'reserved_capital_pct': 10.0,
            'risk_per_trade_pct': 2.0,
            'allow_fractional_shares': True
        }
        
        # Setup mock strategy performance data
        self.performance_metrics.get_strategy_ids.return_value = self.config['strategy_ids']
        self.performance_metrics.get_win_rate.return_value = 0.6
        self.performance_metrics.get_profit_factor.return_value = 1.5
        self.performance_metrics.get_average_win.return_value = 500.0
        self.performance_metrics.get_average_loss.return_value = -300.0
        self.performance_metrics.get_max_drawdown_percentage.return_value = 15.0
        self.performance_metrics.get_current_drawdown_percentage.return_value = 5.0
        self.performance_metrics.get_daily_returns.return_value = [0.01, -0.005, 0.02, 0.01, -0.01]
        
        # Setup mock positions data
        self.position_manager.internal_positions = {
            'pos1': {'position_id': 'pos1', 'strategy_id': 'strategy1', 'symbol': 'AAPL', 'quantity': 100, 'entry_price': 150.0, 'current_price': 155.0, 'direction': 'long'},
            'pos2': {'position_id': 'pos2', 'strategy_id': 'strategy2', 'symbol': 'MSFT', 'quantity': 50, 'entry_price': 250.0, 'current_price': 255.0, 'direction': 'long'}
        }
        
        # Create the capital allocator with our mocks
        self.capital_allocator = CapitalAllocator(
            performance_metrics=self.performance_metrics,
            position_manager=self.position_manager,
            pnl_calculator=self.pnl_calculator,
            broker_manager=self.broker_manager,
            config=self.config
        )
    
    def test_initialize_allocations(self):
        """Test initialization of strategy allocations."""
        # Check that all strategies are allocated initially
        for strategy_id in self.config['strategy_ids']:
            self.assertIn(strategy_id, self.capital_allocator.strategy_allocations)
        
        # Since no predefined allocations were given, they should be equal
        equal_allocation = 100.0 / len(self.config['strategy_ids'])
        for allocation in self.capital_allocator.strategy_allocations.values():
            self.assertAlmostEqual(allocation, equal_allocation)
    
    def test_allocation_methods(self):
        """Test different allocation methods."""
        # Test equal allocation
        self.capital_allocator._allocate_equal()
        equal_allocation = 100.0 / len(self.config['strategy_ids'])
        for allocation in self.capital_allocator.strategy_allocations.values():
            self.assertAlmostEqual(allocation, equal_allocation)
        
        # Test performance-based allocation
        self.capital_allocator._allocate_by_performance()
        # All mock strategies have same metrics, so should remain equal
        for allocation in self.capital_allocator.strategy_allocations.values():
            self.assertAlmostEqual(allocation, equal_allocation)
        
        # Test with different performances
        self.performance_metrics.get_profit_factor.side_effect = lambda sid: 2.0 if sid == 'strategy1' else 1.0
        self.capital_allocator._allocate_by_performance()
        # Strategy1 should have higher allocation
        self.assertGreater(
            self.capital_allocator.strategy_allocations['strategy1'],
            self.capital_allocator.strategy_allocations['strategy2']
        )
    
    def test_drawdown_protection(self):
        """Test drawdown protection functionality."""
        # Set up initial equal allocations
        self.capital_allocator._allocate_equal()
        initial_allocation = self.capital_allocator.strategy_allocations['strategy1']
        
        # Set a high drawdown for strategy1
        self.capital_allocator.max_drawdown_pct = 10.0  # Trigger at 10% drawdown
        self.capital_allocator.strategy_risk_metrics = {
            'strategy1': {'current_drawdown': 20.0},  # Beyond threshold
            'strategy2': {'current_drawdown': 5.0},   # Below threshold
            'strategy3': {'current_drawdown': 3.0}    # Below threshold
        }
        
        # Apply drawdown protection
        self.capital_allocator._apply_drawdown_protection()
        
        # Strategy1 should have reduced allocation
        self.assertLess(
            self.capital_allocator.strategy_allocations['strategy1'],
            initial_allocation
        )
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Setup equal allocations
        self.capital_allocator._allocate_equal()
        
        # Calculate position size
        position_info = self.capital_allocator.calculate_position_size(
            strategy_id='strategy1',
            symbol='GOOG',
            entry_price=2500.0,
            stop_price=2400.0
        )
        
        # Check that size is calculated
        self.assertGreater(position_info['size'], 0)
        
        # Size should be limited by risk
        max_risk_amount = self.config['initial_capital'] * (self.config['risk_per_trade_pct'] / 100.0)
        max_size_by_risk = max_risk_amount / 100.0  # Risk per share is 100 (2500-2400)
        self.assertLessEqual(position_info['size'], max_size_by_risk)
        
        # Check zero allocation case
        self.capital_allocator.strategy_allocations['strategy1'] = 0.0
        no_allocation_info = self.capital_allocator.calculate_position_size(
            strategy_id='strategy1',
            symbol='GOOG',
            entry_price=2500.0
        )
        self.assertEqual(no_allocation_info['size'], 0)
    
    def test_adjust_position_for_correlation(self):
        """Test position adjustment for correlation."""
        # Test with a new symbol (no correlation adjustment)
        adjusted_size = self.capital_allocator.adjust_position_for_correlation(
            strategy_id='strategy1',
            symbol='GOOG',  # Not in current positions
            base_size=10.0
        )
        self.assertEqual(adjusted_size, 10.0)  # No change
        
        # Test with existing symbol
        adjusted_size = self.capital_allocator.adjust_position_for_correlation(
            strategy_id='strategy1',
            symbol='AAPL',  # Already in portfolio
            base_size=10.0
        )
        self.assertLess(adjusted_size, 10.0)  # Should be reduced
    
    def test_get_strategy_allocation(self):
        """Test getting allocation details for a strategy."""
        # Setup initial allocations
        self.capital_allocator._allocate_equal()
        
        # Get allocation for strategy1
        allocation_info = self.capital_allocator.get_strategy_allocation('strategy1')
        
        # Check basic information
        self.assertEqual(allocation_info['strategy_id'], 'strategy1')
        self.assertEqual(
            allocation_info['allocation_pct'],
            self.capital_allocator.strategy_allocations['strategy1']
        )
        
        # Should include active positions count
        self.assertEqual(allocation_info['active_positions'], 1)  # One position in our mock
    
    def test_allocation_limits(self):
        """Test enforcement of allocation limits."""
        # Create extreme allocations
        self.capital_allocator.strategy_allocations = {
            'strategy1': 5.0,   # Below min
            'strategy2': 60.0,  # Above max
            'strategy3': 35.0   # Within limits
        }
        
        # Apply limits
        self.capital_allocator._enforce_allocation_limits()
        
        # Check limits are enforced
        self.assertGreaterEqual(
            self.capital_allocator.strategy_allocations['strategy1'],
            self.config['min_allocation_pct']
        )
        self.assertLessEqual(
            self.capital_allocator.strategy_allocations['strategy2'],
            self.config['max_allocation_pct']
        )
        
        # Check sum is still 100%
        total = sum(self.capital_allocator.strategy_allocations.values())
        self.assertAlmostEqual(total, 100.0)


if __name__ == '__main__':
    unittest.main()
