#!/usr/bin/env python
"""
Risk Management Integration Test

This test validates the risk management components including:
1. Enhanced Position Sizing
2. Volatility-Adjusted Stops
3. Dynamic Capital Allocation
4. Overall Enhanced Risk Manager integration
"""

import unittest
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.risk.enhanced_risk_manager import EnhancedRiskManager, create_enhanced_risk_manager
from trading_bot.risk.enhanced_position_sizing import EnhancedPositionSizer
from trading_bot.risk.volatility_adjusted_stops import VolatilityAdjustedStops
from trading_bot.risk.dynamic_capital_allocator import DynamicCapitalAllocator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskEventTracker:
    """Tracks risk-related events for test validation"""
    
    def __init__(self):
        self.events = []
        self.event_types_received = set()
        self.events_by_type = {}
        self.strategies_affected = set()
        self.symbols_affected = set()
        
    def handle_event(self, event: Event):
        """Process and record an event"""
        self.events.append(event)
        self.event_types_received.add(event.event_type)
        
        # Track by type
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event)
        
        # Track strategies and symbols
        if 'strategy' in event.data or 'strategy_id' in event.data:
            strategy = event.data.get('strategy') or event.data.get('strategy_id')
            if strategy:
                self.strategies_affected.add(strategy)
        
        if 'symbol' in event.data:
            self.symbols_affected.add(event.data['symbol'])


class RiskManagementTest(unittest.TestCase):
    """Tests the risk management components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        # Create test event bus
        cls.event_bus = EventBus()
        
        # Create event tracker
        cls.tracker = RiskEventTracker()
        cls.event_bus.subscribe_all(cls.tracker.handle_event)
        
        # Test account parameters
        cls.initial_equity = 100000.0
        cls.target_equity = 300000.0
        
        # Test trading parameters
        cls.test_strategy = "TestTrendStrategy"
        cls.test_symbol = "AAPL"
        cls.test_entry_price = 185.50
        cls.test_market_data = {
            "close": 185.50,
            "high": 187.25,
            "low": 184.00,
            "open": 184.75,
            "volume": 15000000,
            "atr": 3.5,
            "regime": "trending"
        }
    
    def setUp(self):
        """Set up for each test"""
        # Reset event tracker
        self.tracker.events = []
        self.tracker.event_types_received = set()
        self.tracker.events_by_type = {}
        self.tracker.strategies_affected = set()
        self.tracker.symbols_affected = set()
    
    def test_1_enhanced_position_sizer(self):
        """Test enhanced position sizing component"""
        # Create position sizer
        position_sizer = EnhancedPositionSizer(
            event_bus=self.event_bus,
            config={
                "use_kelly": True,
                "kelly_fraction": 0.3,
                "default_risk_per_trade": 0.02,
                "vol_measure": "atr"
            }
        )
        
        # Add test strategy metrics
        position_sizer.strategy_metrics[self.test_strategy] = {
            "trades": 25,
            "wins": 15,
            "total_pnl": 5000.0,
            "win_pnl": 8000.0,
            "loss_pnl": -3000.0
        }
        
        # Calculate position size
        position_info = position_sizer.calculate_position_size(
            account_value=self.initial_equity,
            symbol=self.test_symbol,
            strategy=self.test_strategy,
            entry_price=self.test_entry_price,
            market_data=self.test_market_data
        )
        
        # Validate results
        self.assertIsNotNone(position_info)
        self.assertIn('position_size', position_info)
        self.assertIn('position_units', position_info)
        self.assertIn('risk_amount', position_info)
        self.assertIn('risk_percent', position_info)
        self.assertIn('factors', position_info)
        
        # Check that position size is reasonable
        risk_amount = position_info['risk_amount']
        self.assertGreater(risk_amount, 0)
        self.assertLess(risk_amount, self.initial_equity * 0.05)  # Should not risk more than 5%
        
        # Verify that Kelly adjustment was applied
        self.assertIn('kelly', position_info['factors'])
        
        # Log sizing info for review
        logger.info(f"Position sizing results:")
        logger.info(f"  Position size: ${position_info['position_size']:.2f}")
        logger.info(f"  Position units: {position_info['position_units']:.2f} shares")
        logger.info(f"  Risk amount: ${position_info['risk_amount']:.2f}")
        logger.info(f"  Risk percent: {position_info['risk_percent']:.2%}")
        
        # Test with stop price
        stop_price = self.test_entry_price - self.test_market_data['atr'] * 2
        position_info_with_stop = position_sizer.calculate_position_size(
            account_value=self.initial_equity,
            symbol=self.test_symbol,
            strategy=self.test_strategy,
            entry_price=self.test_entry_price,
            stop_price=stop_price,
            market_data=self.test_market_data
        )
        
        # Verify stop-based sizing
        self.assertGreater(position_info_with_stop['position_size'], 0)
        logger.info(f"Position size with stop: ${position_info_with_stop['position_size']:.2f}")
    
    def test_2_volatility_adjusted_stops(self):
        """Test volatility-adjusted stops component"""
        # Create stop manager
        stop_manager = VolatilityAdjustedStops(
            event_bus=self.event_bus,
            config={
                "default_atr_multiple": 2.0,
                "trailing_stop_activation_pct": 0.01
            }
        )
        
        # Register a stop
        trade_id = "test_trade_123"
        stop_info = stop_manager.register_stop(
            trade_id=trade_id,
            symbol=self.test_symbol,
            strategy=self.test_strategy,
            entry_price=self.test_entry_price,
            direction="long",
            market_data=self.test_market_data,
            stop_type="atr"
        )
        
        # Validate stop
        self.assertIsNotNone(stop_info)
        self.assertEqual(stop_info['trade_id'], trade_id)
        self.assertEqual(stop_info['symbol'], self.test_symbol)
        self.assertEqual(stop_info['strategy'], self.test_strategy)
        self.assertEqual(stop_info['entry_price'], self.test_entry_price)
        
        # Check stop price calculation
        atr = self.test_market_data['atr']
        expected_stop_distance = atr * 2.0  # default_atr_multiple
        expected_stop_price = self.test_entry_price - expected_stop_distance
        
        self.assertAlmostEqual(
            stop_info['initial_stop_price'], 
            expected_stop_price, 
            delta=0.01
        )
        
        logger.info(f"Stop calculation results:")
        logger.info(f"  Entry price: ${self.test_entry_price:.2f}")
        logger.info(f"  Initial stop price: ${stop_info['initial_stop_price']:.2f}")
        logger.info(f"  Stop distance: ${expected_stop_distance:.2f}")
        
        # Test stop update for price movement
        current_price = self.test_entry_price + 5.0  # Move price up
        high_price = current_price + 0.5
        
        # Simulate price update
        stop_manager._update_stop_for_price(
            trade_id=trade_id,
            price=current_price,
            high=high_price,
            low=current_price - 0.5,
            timestamp=datetime.now()
        )
        
        # Get updated stop
        updated_stop_price = stop_manager.get_stop_price(trade_id)
        
        # Verify trailing stop behavior
        if stop_manager.trailing_stop_activation_pct * self.test_entry_price <= 5.0:
            # If price moved enough to activate trailing stop
            self.assertNotEqual(
                updated_stop_price,
                stop_info['initial_stop_price'],
                "Trailing stop should have activated and moved up"
            )
            logger.info(f"  Updated trailing stop price: ${updated_stop_price:.2f}")
        
        # Test stop triggered check
        stop_triggered = stop_manager.check_stop_triggered(trade_id, updated_stop_price - 0.01)
        self.assertTrue(stop_triggered, "Stop should trigger when price moves below stop price")
        
        stop_not_triggered = stop_manager.check_stop_triggered(trade_id, updated_stop_price + 0.01)
        self.assertFalse(stop_not_triggered, "Stop should not trigger when price is above stop price")
    
    def test_3_dynamic_capital_allocator(self):
        """Test dynamic capital allocator component"""
        # Create capital allocator
        capital_allocator = DynamicCapitalAllocator(
            event_bus=self.event_bus,
            config={
                "initial_capital": self.initial_equity,
                "target_capital": self.target_equity,
                "use_snowball": True,
                "snowball_threshold": 0.1
            }
        )
        
        # Add strategies
        strategies = {
            "TrendStrategy": "trend_following",
            "MeanReversionStrategy": "mean_reversion",
            "BreakoutStrategy": "breakout"
        }
        
        # Add each strategy
        for strategy_id, strategy_type in strategies.items():
            # Publish strategy added event
            self.event_bus.publish(Event(
                event_type=EventType.STRATEGY_ADDED,
                data={
                    'strategy_id': strategy_id,
                    'strategy_type': strategy_type
                }
            ))
        
        # Add some strategy performance
        trend_performance = {
            'profit_factor': 2.2,
            'win_rate': 0.65,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.08,
            'volatility': 0.12
        }
        
        mean_reversion_performance = {
            'profit_factor': 1.7,
            'win_rate': 0.55,
            'sharpe_ratio': 1.4,
            'max_drawdown': 0.12,
            'volatility': 0.18
        }
        
        breakout_performance = {
            'profit_factor': 1.1,
            'win_rate': 0.45,
            'sharpe_ratio': 0.9,
            'max_drawdown': 0.15,
            'volatility': 0.22
        }
        
        # Publish performance events
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_PERFORMANCE_UPDATED,
            data={
                'strategy_id': 'TrendStrategy',
                'metrics': trend_performance,
                'timestamp': datetime.now()
            }
        ))
        
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_PERFORMANCE_UPDATED,
            data={
                'strategy_id': 'MeanReversionStrategy',
                'metrics': mean_reversion_performance,
                'timestamp': datetime.now()
            }
        ))
        
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_PERFORMANCE_UPDATED,
            data={
                'strategy_id': 'BreakoutStrategy',
                'metrics': breakout_performance,
                'timestamp': datetime.now()
            }
        ))
        
        # Wait for events to be processed
        time.sleep(0.5)
        
        # Publish market regime changes
        self.event_bus.publish(Event(
            event_type=EventType.MARKET_REGIME_CHANGED,
            data={
                'symbol': 'SPY',
                'current_regime': 'trending',
                'previous_regime': 'ranging'
            }
        ))
        
        # Wait for events to be processed
        time.sleep(0.5)
        
        # Get allocations
        allocations = capital_allocator.get_all_allocations()
        
        # Validate allocations
        self.assertIsNotNone(allocations)
        self.assertGreaterEqual(len(allocations), 3)
        
        # Expect trend strategy to have highest allocation in trending regime
        trend_allocation = allocations.get('TrendStrategy', 0)
        mean_rev_allocation = allocations.get('MeanReversionStrategy', 0)
        
        logger.info(f"Strategy allocations in trending regime:")
        for strategy, allocation in allocations.items():
            logger.info(f"  {strategy}: {allocation:.2%}")
        
        # Test equity update with profit
        new_equity = self.initial_equity * 1.15  # 15% profit
        
        self.event_bus.publish(Event(
            event_type=EventType.ACCOUNT_EQUITY_UPDATED,
            data={
                'equity': new_equity,
                'previous_equity': self.initial_equity
            }
        ))
        
        # Wait for events to be processed
        time.sleep(0.5)
        
        # Get updated allocations after profit
        new_allocations = capital_allocator.get_all_allocations()
        
        # If snowball active, better strategies should get more allocation
        if new_equity / self.initial_equity > capital_allocator.snowball_threshold + 1.0:
            logger.info(f"Allocations after profit (snowball strategy):")
            for strategy, allocation in new_allocations.items():
                logger.info(f"  {strategy}: {allocation:.2%}")
    
    def test_4_enhanced_risk_manager_integration(self):
        """Test the integrated enhanced risk manager"""
        # Create enhanced risk manager
        risk_manager = create_enhanced_risk_manager(
            initial_equity=self.initial_equity,
            target_equity=self.target_equity,
            event_bus=self.event_bus,
            config={
                'adaptive_risk': {
                    'initial_base_risk_per_trade': 0.02,
                    'conservative_base_risk_per_trade': 0.005
                },
                'position_sizing': {
                    'use_kelly': True,
                    'kelly_fraction': 0.3,
                    'vol_measure': 'atr'
                },
                'stops': {
                    'default_atr_multiple': 2.0,
                    'trailing_stop_activation_pct': 0.01
                },
                'allocation': {
                    'use_snowball': True,
                    'snowball_threshold': 0.1
                }
            }
        )
        
        # Test the full flow: position sizing + stop placement for a trade
        position_info = risk_manager.calculate_position_size(
            strategy_id=self.test_strategy,
            symbol=self.test_symbol,
            entry_price=self.test_entry_price,
            market_data=self.test_market_data
        )
        
        # Create a trade ID
        trade_id = f"{self.test_strategy}_{self.test_symbol}_{int(time.time())}"
        
        # Calculate stop price
        stop_info = risk_manager.calculate_stop_price(
            trade_id=trade_id,
            strategy_id=self.test_strategy,
            symbol=self.test_symbol,
            entry_price=self.test_entry_price,
            direction="long",
            market_data=self.test_market_data
        )
        
        # Validate integrated results
        self.assertIsNotNone(position_info)
        self.assertIn('position_size', position_info)
        self.assertIn('risk_amount', position_info)
        
        self.assertIsNotNone(stop_info)
        self.assertIn('initial_stop_price', stop_info)
        
        # Check consistency between position sizing and stop placement
        stop_distance = self.test_entry_price - stop_info['initial_stop_price']
        self.assertGreater(stop_distance, 0, "Stop should be below entry for long trades")
        
        logger.info(f"Integrated risk management results:")
        logger.info(f"  Position size: ${position_info['position_size']:.2f}")
        logger.info(f"  Risk amount: ${position_info['risk_amount']:.2f}")
        logger.info(f"  Risk percent: {position_info['risk_percent']:.2%}")
        logger.info(f"  Stop price: ${stop_info['initial_stop_price']:.2f}")
        logger.info(f"  Stop distance: ${stop_distance:.2f}")
        
        # Check if risk amount is consistent with stop distance
        position_units = position_info.get('position_units', position_info['position_size'] / self.test_entry_price)
        expected_risk = position_units * stop_distance
        actual_risk = position_info['risk_amount']
        
        # Risks should be roughly similar, allowing for rounding differences
        self.assertAlmostEqual(
            expected_risk,
            actual_risk,
            delta=actual_risk * 0.2,  # Allow 20% tolerance
            msg="Risk amount should be consistent with stop distance"
        )
        
        # Test strategy allocation
        risk_manager.get_strategy_allocation(self.test_strategy)
        allocations = risk_manager.get_all_strategy_allocations()
        
        # Get current risk parameters
        risk_params = risk_manager.get_current_risk_parameters()
        self.assertIsNotNone(risk_params)
        
        logger.info(f"Current risk parameters:")
        for param, value in risk_params.items():
            logger.info(f"  {param}: {value}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        cls.event_bus.clear_subscribers()


if __name__ == '__main__':
    unittest.main()
