"""
Kill Switch Test Suite

This script tests the emergency controls functionality, particularly the kill switch
and circuit breakers that protect the system from excessive losses.

Usage:
    python -m tests.validation.kill_switch_test
"""
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType, TradingMode
from trading_bot.risk.emergency_controls import EmergencyControls


class KillSwitchTest(unittest.TestCase):
    """Tests for the EmergencyControls class, focusing on kill switch functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.event_bus = get_global_event_bus()
        
        # Configure emergency controls for testing
        self.emergency_controls = EmergencyControls(
            max_daily_loss_pct=0.05,    # 5% max daily loss
            max_position_pct=0.20,      # 20% max position size
            max_strategy_loss_pct=0.03, # 3% max strategy loss
            max_drawdown_pct=0.10,      # 10% max drawdown
            circuit_breaker_window=10,  # 10 seconds for circuit breaker window
            auto_enable=True
        )
        
        # Track events published by the emergency controls
        self.kill_switch_events = []
        self.mode_change_events = []
        self.strategy_stop_events = []
        
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.KILL_SWITCH_ACTIVATED, self._on_kill_switch)
        self.event_bus.subscribe(EventType.MODE_CHANGED, self._on_mode_changed)
        self.event_bus.subscribe(EventType.STRATEGY_STOPPED, self._on_strategy_stopped)
        
        # Initialize account value
        self._set_account_value(100000.0)
    
    def _on_kill_switch(self, event):
        """Handler for kill switch activation events."""
        self.kill_switch_events.append(event.data)
    
    def _on_mode_changed(self, event):
        """Handler for trading mode change events."""
        self.mode_change_events.append(event.data)
    
    def _on_strategy_stopped(self, event):
        """Handler for strategy stopped events."""
        self.strategy_stop_events.append(event.data)
    
    def _set_account_value(self, value):
        """Set the account value for testing."""
        self.event_bus.create_and_publish(
            event_type=EventType.CAPITAL_ADJUSTED,
            data={"new_capital": value},
            source="kill_switch_test"
        )
        
        # Let event processing complete
        time.sleep(0.1)
        
        # Verify account value was set
        self.assertEqual(self.emergency_controls.account_value, value)
    
    def test_manual_kill_switch(self):
        """Test manual activation of the kill switch."""
        # Verify kill switch is initially disabled
        self.assertFalse(self.emergency_controls.kill_switch_activated)
        
        # Manually activate kill switch
        reason = "Manual test activation"
        self.emergency_controls.activate_kill_switch(reason=reason)
        
        # Verify kill switch is now active
        self.assertTrue(self.emergency_controls.kill_switch_activated)
        
        # Verify events were published
        self.assertEqual(len(self.kill_switch_events), 1)
        self.assertEqual(self.kill_switch_events[0].get("reason"), reason)
        
        # Verify trading mode was changed to STOPPED
        self.assertEqual(len(self.mode_change_events), 1)
        self.assertEqual(self.mode_change_events[0].get("new_mode"), TradingMode.STOPPED)
        
        # Manually deactivate kill switch
        self.emergency_controls.deactivate_kill_switch()
        
        # Verify kill switch is now inactive
        self.assertFalse(self.emergency_controls.kill_switch_activated)
    
    def test_daily_loss_limit(self):
        """Test kill switch activation due to exceeding daily loss limit."""
        # Reset kill switch and account state
        self.emergency_controls.deactivate_kill_switch()
        self.emergency_controls.reset_daily_metrics()
        self._set_account_value(100000.0)
        self.kill_switch_events = []
        
        # Verify kill switch is inactive
        self.assertFalse(self.emergency_controls.kill_switch_activated)
        
        # Execute trades with small losses (below threshold)
        for i in range(3):
            self.event_bus.create_and_publish(
                event_type=EventType.TRADE_EXECUTED,
                data={
                    "symbol": "AAPL",
                    "quantity": 10,
                    "price": 150.0,
                    "pnl": -500.0,  # $500 loss per trade
                    "timestamp": datetime.now().isoformat()
                },
                source="kill_switch_test"
            )
        
        # Verify kill switch is still inactive
        self.assertFalse(self.emergency_controls.kill_switch_activated)
        
        # Execute large loss trade that exceeds daily loss threshold
        self.event_bus.create_and_publish(
            event_type=EventType.TRADE_EXECUTED,
            data={
                "symbol": "AAPL",
                "quantity": 100,
                "price": 150.0,
                "pnl": -4000.0,  # $4000 loss, pushing total to $5500 (>5%)
                "timestamp": datetime.now().isoformat()
            },
            source="kill_switch_test"
        )
        
        # Verify kill switch was activated
        self.assertTrue(self.emergency_controls.kill_switch_activated)
        
        # Verify appropriate events were published
        self.assertGreaterEqual(len(self.kill_switch_events), 1)
        latest_event = self.kill_switch_events[-1]
        self.assertIn("loss limit", latest_event.get("reason", "").lower())
    
    def test_circuit_breaker_order_rejection(self):
        """Test circuit breaker triggering due to high order rejection rate."""
        # Reset kill switch and account state
        self.emergency_controls.deactivate_kill_switch()
        self.emergency_controls.reset_daily_metrics()
        self.kill_switch_events = []
        
        # Verify kill switch is inactive
        self.assertFalse(self.emergency_controls.kill_switch_activated)
        
        # Submit several successful orders
        for i in range(5):
            self.event_bus.create_and_publish(
                event_type=EventType.ORDER_CREATED,
                data={
                    "order_id": f"test-{i}",
                    "symbol": "MSFT",
                    "quantity": 10,
                    "price": 250.0,
                    "timestamp": datetime.now().isoformat()
                },
                source="kill_switch_test"
            )
        
        # Submit several rejected orders
        for i in range(10):
            self.event_bus.create_and_publish(
                event_type=EventType.ORDER_REJECTED,
                data={
                    "order_id": f"test-rejected-{i}",
                    "symbol": "MSFT",
                    "reason": "Insufficient funds",
                    "timestamp": datetime.now().isoformat()
                },
                source="kill_switch_test"
            )
        
        # Verify kill switch was activated
        self.assertTrue(self.emergency_controls.kill_switch_activated)
        
        # Verify appropriate events were published
        self.assertGreaterEqual(len(self.kill_switch_events), 1)
        latest_event = self.kill_switch_events[-1]
        self.assertIn("circuit breaker", latest_event.get("reason", "").lower())
    
    def test_position_limits(self):
        """Test position limit enforcement."""
        # Reset kill switch and account state
        self.emergency_controls.deactivate_kill_switch()
        self._set_account_value(100000.0)
        
        # Set position limit for a symbol
        symbol = "GOOGL"
        max_size = 50
        self.emergency_controls.set_position_limit(symbol, max_size)
        
        # Verify position within limit is allowed
        self.assertTrue(self.emergency_controls.check_position_limit(
            symbol=symbol,
            new_size=40,
            price=150.0
        ))
        
        # Verify position exceeding limit is rejected
        self.assertFalse(self.emergency_controls.check_position_limit(
            symbol=symbol,
            new_size=60,
            price=150.0
        ))
        
        # Verify position exceeding account percentage is rejected
        expensive_symbol = "AMZN"
        expensive_price = 3000.0
        large_size = 10  # 10 * $3000 = $30,000 (30% of account)
        
        self.assertFalse(self.emergency_controls.check_position_limit(
            symbol=expensive_symbol,
            new_size=large_size,
            price=expensive_price
        ))
    
    def test_drawdown_protection(self):
        """Test kill switch activation due to excessive drawdown."""
        # Reset kill switch and account state
        self.emergency_controls.deactivate_kill_switch()
        self._set_account_value(100000.0)
        self.kill_switch_events = []
        
        # Verify kill switch is inactive
        self.assertFalse(self.emergency_controls.kill_switch_activated)
        
        # Set high watermark
        self.emergency_controls.high_watermark = 100000.0
        
        # Trigger drawdown alert (not exceeding threshold)
        self.event_bus.create_and_publish(
            event_type=EventType.DRAWDOWN_ALERT,
            data={
                "drawdown_pct": 0.05,  # 5% drawdown
                "timestamp": datetime.now().isoformat()
            },
            source="kill_switch_test"
        )
        
        # Verify kill switch is still inactive
        self.assertFalse(self.emergency_controls.kill_switch_activated)
        
        # Trigger drawdown threshold exceeded
        self.event_bus.create_and_publish(
            event_type=EventType.DRAWDOWN_THRESHOLD_EXCEEDED,
            data={
                "drawdown_pct": 0.15,  # 15% drawdown (exceeds 10% threshold)
                "threshold": 0.10,
                "timestamp": datetime.now().isoformat()
            },
            source="kill_switch_test"
        )
        
        # Verify kill switch was activated
        self.assertTrue(self.emergency_controls.kill_switch_activated)
        
        # Verify appropriate events were published
        self.assertGreaterEqual(len(self.kill_switch_events), 1)
    
    def test_strategy_specific_limits(self):
        """Test per-strategy risk limits and disabled strategies."""
        # Reset kill switch and account state
        self.emergency_controls.deactivate_kill_switch()
        self.emergency_controls.reset_daily_metrics()
        self.strategy_stop_events = []
        
        # Initialize strategy status
        strategy_id = "test_strategy"
        self.emergency_controls.enable_strategy(strategy_id)
        
        # Verify strategy is enabled
        self.assertTrue(self.emergency_controls.is_strategy_enabled(strategy_id))
        
        # Disable strategy
        reason = "Testing strategy disabling"
        self.emergency_controls.disable_strategy(strategy_id, reason=reason)
        
        # Verify strategy is now disabled
        self.assertFalse(self.emergency_controls.is_strategy_enabled(strategy_id))
        
        # Verify appropriate events were published
        self.assertEqual(len(self.strategy_stop_events), 1)
        self.assertEqual(self.strategy_stop_events[0].get("strategy_id"), strategy_id)
        self.assertEqual(self.strategy_stop_events[0].get("reason"), reason)

    def test_enable_disable_controls(self):
        """Test enabling and disabling emergency controls."""
        # Test disabling emergency controls
        self.emergency_controls.disable()
        self.assertFalse(self.emergency_controls.enabled)
        
        # When disabled, position checks should always pass
        symbol = "TSLA"
        self.assertTrue(self.emergency_controls.check_position_limit(
            symbol=symbol,
            new_size=1000,  # Unreasonably large position
            price=200.0
        ))
        
        # Re-enable controls
        self.emergency_controls.enable()
        self.assertTrue(self.emergency_controls.enabled)
        
        # Now position checks should enforce limits
        self.assertFalse(self.emergency_controls.check_position_limit(
            symbol=symbol,
            new_size=1000,  # Should exceed limits
            price=200.0
        ))


def main():
    """Run kill switch tests."""
    unittest.main()


if __name__ == "__main__":
    main()
