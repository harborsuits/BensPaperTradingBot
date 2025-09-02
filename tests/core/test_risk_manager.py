#!/usr/bin/env python3
"""
Unit tests for risk management system.

These tests verify:
1. Margin status monitoring
2. Margin call handling and de-leveraging
3. Circuit breaker functionality
4. Trading pause/resume mechanism
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from trading_bot.core.risk_manager import RiskManager, PortfolioCircuitBreaker
from trading_bot.core.events import (
    MarginCall, MarginCallWarning, TradingPaused, TradingResumed,
    ForcedExitOrder, PortfolioEquityUpdate, VolatilityUpdate
)
from trading_bot.event_system.event_bus import EventBus


class TestRiskManager(unittest.TestCase):
    """Test suite for the RiskManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock trading system/orchestrator
        self.trading_system = MagicMock()
        self.orchestrator = MagicMock()
        self.trading_system.orchestrator = self.orchestrator
        self.trading_system.service_registry = MagicMock()
        
        # Create mock broker manager and position manager
        self.broker_manager = MagicMock()
        self.position_manager = MagicMock()
        
        # Register mocks in service registry
        self.trading_system.service_registry.get.side_effect = lambda key, default=None: {
            "broker_manager": self.broker_manager,
            "position_manager": self.position_manager
        }.get(key, default)
        
        # Create mock alert system
        self.alert_system = MagicMock()
        
        # Create event bus
        self.event_bus = EventBus()
        
        # Create risk configuration
        self.risk_config = {
            "risk": {
                "max_margin_ratio": 0.5,
                "margin_call_buffer": 0.05,
                "margin_check_interval": 1,  # Short interval for testing
                "breakers": {
                    "intraday_drawdown_threshold": 0.05,
                    "overall_drawdown_threshold": 0.10,
                    "volatility_threshold": 0.025
                },
                "broker_specific": {
                    "tradier": {
                        "max_margin_ratio": 0.45
                    }
                },
                "forced_deleveraging": {
                    "max_positions_per_step": 2
                }
            }
        }
        
        # Create risk manager
        self.risk_manager = RiskManager(
            trading_system=self.trading_system,
            config=self.risk_config,
            event_bus=self.event_bus,
            alert_system=self.alert_system
        )
        
        # Set up the circuit breaker after risk manager
        self.circuit_breaker = PortfolioCircuitBreaker(
            trading_system=self.trading_system,
            config=self.risk_config,
            event_bus=self.event_bus,
            alert_system=self.alert_system
        )
        
        # Create test brokers
        self.tradier_broker = MagicMock()
        self.alpaca_broker = MagicMock()
        
        # Set up broker manager to return our test brokers
        self.broker_manager.get_all_brokers.return_value = {
            "tradier": self.tradier_broker,
            "alpaca": self.alpaca_broker
        }
        
        # Create test positions for position manager
        self.test_positions = [
            MagicMock(symbol="AAPL", quantity=100, avg_price=150.0, unrealized_pnl=500.0),
            MagicMock(symbol="MSFT", quantity=50, avg_price=250.0, unrealized_pnl=-200.0),
            MagicMock(symbol="GOOGL", quantity=25, avg_price=2000.0, unrealized_pnl=1000.0)
        ]
        
        # Set up position manager to return our test positions
        self.position_manager.get_positions_by_broker.return_value = self.test_positions
    
    def tearDown(self):
        """Clean up after tests"""
        # Shutdown the risk manager to stop any threads
        if hasattr(self, 'risk_manager'):
            self.risk_manager.shutdown()
    
    def test_margin_status_retrieval(self):
        """Test retrieval of margin status from brokers"""
        # Set up mock margin status returns
        self.tradier_broker.get_margin_status.return_value = {
            "account_id": "tradier_demo",
            "cash": 10000.0,
            "margin_used": 20000.0,
            "buying_power": 40000.0,
            "maintenance_requirement": 50000.0
        }
        
        self.alpaca_broker.get_margin_status.return_value = {
            "account_id": "alpaca_demo",
            "cash": 15000.0,
            "margin_used": 30000.0,
            "buying_power": 60000.0,
            "maintenance_requirement": 75000.0
        }
        
        # Call the method under test
        result = self.risk_manager.get_margin_status()
        
        # Assert that we got status for both brokers
        self.assertEqual(len(result), 2)
        self.assertIn("tradier", result)
        self.assertIn("alpaca", result)
        
        # Verify margin status structure
        tradier_status = result["tradier"]
        self.assertEqual(tradier_status["account_id"], "tradier_demo")
        self.assertEqual(tradier_status["cash"], 10000.0)
        self.assertEqual(tradier_status["margin_used"], 20000.0)
        
        # Verify that the broker methods were called
        self.tradier_broker.get_margin_status.assert_called_once()
        self.alpaca_broker.get_margin_status.assert_called_once()
    
    def test_margin_call_warning(self):
        """Test margin call warning threshold detection"""
        # Set up event capture
        captured_events = []
        def capture_warning(event):
            captured_events.append(event)
        
        self.event_bus.on(MarginCallWarning, capture_warning)
        
        # Set up margin status to trigger warning (at buffer zone)
        # Tradier config has max_ratio of 0.45, buffer is 0.05, so warning at 0.40-0.45
        self.tradier_broker.get_margin_status.return_value = {
            "account_id": "tradier_demo",
            "cash": 10000.0,
            "margin_used": 20000.0,  # 0.4 ratio to maintenance
            "buying_power": 40000.0,
            "maintenance_requirement": 50000.0
        }
        
        # Call the check method
        self.risk_manager._check_margin()
        
        # Verify warning was triggered
        self.assertEqual(len(captured_events), 1)
        warning = captured_events[0]
        self.assertEqual(warning.broker, "tradier")
        self.assertAlmostEqual(warning.ratio, 0.4)
        
        # Verify alert was sent
        self.alert_system.send_alert.assert_called()
    
    def test_margin_call_trigger(self):
        """Test margin call threshold detection and handling"""
        # Set up event capture
        captured_margin_calls = []
        captured_forced_exits = []
        
        def capture_margin_call(event):
            captured_margin_calls.append(event)
        
        def capture_forced_exit(event):
            captured_forced_exits.append(event)
        
        self.event_bus.on(MarginCall, capture_margin_call)
        self.event_bus.on(ForcedExitOrder, capture_forced_exit)
        
        # Set up margin status to trigger margin call
        # Tradier config has max_ratio of 0.45, so margin call at > 0.45
        self.tradier_broker.get_margin_status.return_value = {
            "account_id": "tradier_demo",
            "cash": 10000.0,
            "margin_used": 25000.0,  # 0.5 ratio to maintenance (exceeds 0.45 threshold)
            "buying_power": 40000.0,
            "maintenance_requirement": 50000.0
        }
        
        # Call the check method
        self.risk_manager._check_margin()
        
        # Verify margin call was triggered
        self.assertEqual(len(captured_margin_calls), 1)
        margin_call = captured_margin_calls[0]
        self.assertEqual(margin_call.broker, "tradier")
        self.assertAlmostEqual(margin_call.ratio, 0.5)
        
        # Verify forced exits were emitted
        # We configured max_positions_per_step as 2, so only the 2 largest positions should be exited
        self.assertEqual(len(captured_forced_exits), 2)
        
        # The largest positions are GOOGL and AAPL by notional value
        symbols = [exit_order.symbol for exit_order in captured_forced_exits]
        self.assertIn("GOOGL", symbols)  # GOOGL: 25 * 2000 = 50,000
        self.assertIn("AAPL", symbols)   # AAPL: 100 * 150 = 15,000
        self.assertNotIn("MSFT", symbols)  # MSFT: 50 * 250 = 12,500
        
        # Verify critical alert was sent
        self.alert_system.send_alert.assert_called()
    
    def test_circuit_breaker_drawdown(self):
        """Test portfolio drawdown circuit breaker"""
        # Set up event capture
        captured_paused_events = []
        
        def capture_pause(event):
            captured_paused_events.append(event)
        
        self.event_bus.on(TradingPaused, capture_pause)
        
        # Create drawdown equity update (10% drawdown, exceeds overall threshold)
        equity_update = PortfolioEquityUpdate(
            equity=90000.0,
            peak_equity=100000.0,
            drawdown=0.10  # 10% drawdown
        )
        
        # Emit the equity update
        self.event_bus.emit(equity_update)
        
        # Verify that trading was paused
        self.orchestrator.pause_trading.assert_called_once()
        
        # Verify that pause event was emitted
        self.assertEqual(len(captured_paused_events), 1)
        
        # Reset for next test
        self.orchestrator.reset_mock()
        captured_paused_events.clear()
        
        # Test circuit breaker reset and resume
        captured_resumed_events = []
        
        def capture_resume(event):
            captured_resumed_events.append(event)
        
        self.event_bus.on(TradingResumed, capture_resume)
        
        # Reset the breaker
        self.circuit_breaker.reset_breaker()
        
        # Verify that trading was resumed
        self.orchestrator.resume_trading.assert_called_once()
        
        # Verify that resume event was emitted
        self.assertEqual(len(captured_resumed_events), 1)
    
    def test_volatility_circuit_breaker(self):
        """Test volatility circuit breaker"""
        # Set up event capture
        captured_paused_events = []
        
        def capture_pause(event):
            captured_paused_events.append(event)
        
        self.event_bus.on(TradingPaused, capture_pause)
        
        # Create high volatility update (3% volatility, exceeds 2.5% threshold)
        volatility_update = VolatilityUpdate(
            symbol=None,  # Portfolio-level volatility
            timeframe="1D",
            realized_vol=0.03  # 3% volatility
        )
        
        # Emit the volatility update
        self.event_bus.emit(volatility_update)
        
        # Verify that trading was paused
        self.orchestrator.pause_trading.assert_called_once()
        
        # Verify that pause event was emitted
        self.assertEqual(len(captured_paused_events), 1)
        pause_event = captured_paused_events[0]
        self.assertIn("volatility", pause_event.reason)


if __name__ == '__main__':
    unittest.main()
