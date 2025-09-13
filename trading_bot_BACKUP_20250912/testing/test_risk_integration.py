#!/usr/bin/env python3
"""
Unit tests for the Risk Integration module for the Autonomous Trading Engine.

This test suite ensures that the autonomous engine properly integrates with risk 
management, including position sizing, circuit breakers, and risk allocation.
"""

import unittest
import os
import sys
import logging
from datetime import datetime
from unittest.mock import Mock, patch
import json
from typing import Dict, List, Any

# Configure path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components to test
from trading_bot.autonomous.risk_integration import AutonomousRiskManager, get_autonomous_risk_manager
from trading_bot.autonomous.autonomous_engine import AutonomousEngine, StrategyCandidate
from trading_bot.risk.risk_manager import RiskManager, RiskLevel, StopLossType
from trading_bot.event_system import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRiskIntegration(unittest.TestCase):
    """Test suite for Autonomous Risk Manager integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temp directory for test data
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "test_data", 
                                         "risk_integration")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create test risk config
        self.risk_config = {
            "default_risk_level": "MEDIUM",
            "position_sizing": {
                "max_position_size_pct": 5.0,
                "default_risk_per_trade_pct": 1.0
            },
            "stop_loss": {
                "default_type": "VOLATILITY",
                "atr_multiplier": 2.0
            },
            "circuit_breakers": {
                "portfolio_drawdown_pct": 10.0,
                "daily_loss_pct": 3.0
            }
        }
        
        # Mock the risk manager
        self.mock_risk_manager = Mock(spec=RiskManager)
        self.mock_risk_manager.portfolio_value = 100000.0
        self.mock_risk_manager.get_risk_metrics.return_value = {
            "current_drawdown_pct": 2.5,
            "daily_profit_loss_pct": -1.2,
            "total_portfolio_risk": 45.0,
            "open_positions": 5
        }
        
        # Mock the autonomous engine
        self.mock_engine = Mock(spec=AutonomousEngine)
        
        # Create test strategy candidate
        self.test_candidate = StrategyCandidate(
            strategy_id="test_strategy_001",
            strategy_type="iron_condor",
            symbols=["SPY"],
            universe="options",
            parameters={"delta": 0.3, "dte": 45}
        )
        self.test_candidate.returns = 15.2
        self.test_candidate.sharpe_ratio = 1.8
        self.test_candidate.drawdown = 12.4
        self.test_candidate.win_rate = 68.0
        self.test_candidate.profit_factor = 1.9
        self.test_candidate.trades_count = 25
        self.test_candidate.status = "backtested"
        self.test_candidate.meets_criteria = True
        
        # Mock engine get_top_candidates to return our test candidate
        self.mock_engine.get_top_candidates.return_value = [self.test_candidate]
        
        # Create a real event bus for testing events
        self.event_bus = EventBus()
        self.event_tracker = EventTracker(self.event_bus)
        
        # Create the autonomous risk manager with mocks
        with patch('trading_bot.autonomous.risk_integration.RiskManager') as mock_risk_class:
            mock_risk_class.return_value = self.mock_risk_manager
            self.risk_manager = AutonomousRiskManager(
                risk_config=self.risk_config,
                data_dir=self.test_data_dir
            )
            self.risk_manager.event_bus = self.event_bus
            self.risk_manager.engine = self.mock_engine
        
        # Start event tracking
        self.event_tracker.start_tracking()
    
    def tearDown(self):
        """Clean up test artifacts."""
        # Clean up test data directory
        if os.path.exists(os.path.join(self.test_data_dir, "autonomous_risk_state.json")):
            os.remove(os.path.join(self.test_data_dir, "autonomous_risk_state.json"))
        
        # Stop event tracking
        self.event_tracker.stop_tracking()
    
    def test_init(self):
        """Test initialization."""
        self.assertIsNotNone(self.risk_manager)
        self.assertEqual(self.risk_manager.data_dir, self.test_data_dir)
        self.assertDictEqual(self.risk_manager.circuit_breakers, {
            "portfolio_drawdown": 15.0,
            "strategy_drawdown": 25.0,
            "daily_loss": 5.0,
            "trade_frequency": 20,
            "correlation_threshold": 0.7
        })
    
    def test_connect_engine(self):
        """Test connecting to the autonomous engine."""
        # Create a new manager instance
        with patch('trading_bot.autonomous.risk_integration.RiskManager'):
            risk_manager = AutonomousRiskManager(data_dir=self.test_data_dir)
            
            # Connect to engine
            engine = Mock(spec=AutonomousEngine)
            risk_manager.connect_engine(engine)
            
            # Verify connection
            self.assertEqual(risk_manager.engine, engine)
    
    def test_deploy_strategy(self):
        """Test deploying a strategy with risk controls."""
        # Deploy the test strategy
        result = self.risk_manager.deploy_strategy(
            strategy_id="test_strategy_001",
            allocation_percentage=7.5,
            risk_level=RiskLevel.MEDIUM,
            stop_loss_type=StopLossType.VOLATILITY
        )
        
        # Verify result
        self.assertTrue(result)
        
        # Verify engine deploy was called
        self.mock_engine.deploy_strategy.assert_called_once_with("test_strategy_001")
        
        # Verify deployment data was stored
        self.assertIn("test_strategy_001", self.risk_manager.deployed_strategies)
        self.assertEqual(
            self.risk_manager.strategy_allocations["test_strategy_001"], 
            7.5
        )
        
        # Verify event was emitted
        events = self.event_tracker.get_events_by_type(EventType.STRATEGY_DEPLOYED_WITH_RISK)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["strategy_id"], "test_strategy_001")
        self.assertEqual(events[0].data["allocation_percentage"], 7.5)
    
    def test_deploy_strategy_with_maximum_allocation(self):
        """Test deploying a strategy with excessive allocation that gets capped."""
        # Deploy with excessive allocation percentage
        result = self.risk_manager.deploy_strategy(
            strategy_id="test_strategy_001",
            allocation_percentage=30.0  # Should be capped at 20%
        )
        
        # Verify result and allocation capping
        self.assertTrue(result)
        self.assertEqual(
            self.risk_manager.deployed_strategies["test_strategy_001"]["risk_params"]["allocation_percentage"],
            20.0  # Capped value
        )
    
    def test_adjust_allocation(self):
        """Test adjusting allocation for a deployed strategy."""
        # First deploy a strategy
        self.risk_manager.deploy_strategy(strategy_id="test_strategy_001")
        
        # Now adjust its allocation
        result = self.risk_manager.adjust_allocation("test_strategy_001", 12.5)
        
        # Verify result
        self.assertTrue(result)
        self.assertEqual(
            self.risk_manager.strategy_allocations["test_strategy_001"], 
            12.5
        )
        
        # Try adjusting a non-existent strategy
        result = self.risk_manager.adjust_allocation("nonexistent_strategy", 5.0)
        self.assertFalse(result)
    
    def test_set_circuit_breakers(self):
        """Test setting circuit breaker thresholds."""
        new_breakers = {
            "portfolio_drawdown": 10.0,
            "daily_loss": 3.0,
            "invalid_key": 100  # Should be ignored
        }
        
        self.risk_manager.set_circuit_breakers(new_breakers)
        
        # Verify valid keys were updated
        self.assertEqual(self.risk_manager.circuit_breakers["portfolio_drawdown"], 10.0)
        self.assertEqual(self.risk_manager.circuit_breakers["daily_loss"], 3.0)
        
        # Verify invalid key was ignored and other keys weren't changed
        self.assertNotIn("invalid_key", self.risk_manager.circuit_breakers)
        self.assertEqual(self.risk_manager.circuit_breakers["strategy_drawdown"], 25.0)
    
    def test_check_circuit_breakers(self):
        """Test circuit breaker checks."""
        # Setup risk metrics that should trigger breakers
        self.mock_risk_manager.get_risk_metrics.return_value = {
            "current_drawdown_pct": 20.0,  # Exceeds portfolio_drawdown threshold
            "daily_profit_loss_pct": -8.0,  # Exceeds daily_loss threshold
            "total_portfolio_risk": 85.0
        }
        
        # Add strategy with excessive drawdown
        self.risk_manager.risk_metrics["test_strategy_001"] = {
            "current_drawdown": 30.0,  # Exceeds strategy_drawdown threshold
            "trade_count_today": 5,
            "position_count": 2,
            "largest_position_size": 15000.0,
            "daily_profit_loss": -2500.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Check if breakers should trigger
        should_halt, reasons = self.risk_manager.check_circuit_breakers({})
        
        # Verify results
        self.assertTrue(should_halt)
        self.assertEqual(len(reasons), 3)  # Should have 3 reasons
        
        # Verify event was emitted
        events = self.event_tracker.get_events_by_type(EventType.CIRCUIT_BREAKER_TRIGGERED)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0].data["reasons"]), 3)
    
    def test_pause_and_resume_strategy(self):
        """Test pausing and resuming a strategy."""
        # First deploy a strategy
        self.risk_manager.deploy_strategy(strategy_id="test_strategy_001")
        
        # Pause the strategy
        result = self.risk_manager.pause_strategy(
            strategy_id="test_strategy_001", 
            reason="Test pause"
        )
        
        # Verify pause
        self.assertTrue(result)
        self.assertEqual(
            self.risk_manager.deployed_strategies["test_strategy_001"]["status"],
            "paused"
        )
        self.assertEqual(
            self.risk_manager.deployed_strategies["test_strategy_001"]["pause_reason"],
            "Test pause"
        )
        
        # Verify event was emitted
        events = self.event_tracker.get_events_by_type(EventType.STRATEGY_PAUSED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["strategy_id"], "test_strategy_001")
        self.assertEqual(events[0].data["reason"], "Test pause")
        
        # Resume the strategy
        result = self.risk_manager.resume_strategy(strategy_id="test_strategy_001")
        
        # Verify resume
        self.assertTrue(result)
        self.assertEqual(
            self.risk_manager.deployed_strategies["test_strategy_001"]["status"],
            "active"
        )
        self.assertNotIn("pause_reason", self.risk_manager.deployed_strategies["test_strategy_001"])
        
        # Verify event was emitted
        events = self.event_tracker.get_events_by_type(EventType.STRATEGY_RESUMED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["strategy_id"], "test_strategy_001")
    
    def test_calculate_position_size(self):
        """Test calculating risk-adjusted position size."""
        # First deploy a strategy
        self.risk_manager.deploy_strategy(
            strategy_id="test_strategy_001",
            allocation_percentage=10.0
        )
        
        # Calculate position size for a trade
        position_size = self.risk_manager.calculate_position_size(
            strategy_id="test_strategy_001",
            symbol="SPY",
            entry_price=400.0,
            stop_price=380.0,
            market_data={}
        )
        
        # Verify position size calculation
        # Expected: (100000 * 0.1) * 0.02 / (0.05) / 400 shares
        expected = (100000 * 0.1 * 0.02) / ((400 - 380) / 400) / 400
        self.assertAlmostEqual(position_size, expected, places=2)
    
    def test_get_risk_report(self):
        """Test getting a comprehensive risk report."""
        # Setup test data
        self.risk_manager.deploy_strategy(strategy_id="test_strategy_001")
        self.risk_manager.deploy_strategy(strategy_id="test_strategy_002")
        
        # Get risk report
        report = self.risk_manager.get_risk_report()
        
        # Verify report structure
        self.assertIn("timestamp", report)
        self.assertIn("overall_metrics", report)
        self.assertIn("strategy_metrics", report)
        self.assertIn("correlations", report)
        self.assertIn("circuit_breakers", report)
        self.assertEqual(report["total_strategies"], 2)
        self.assertEqual(report["active_strategies"], 2)
    
    def test_event_handlers(self):
        """Test that event handlers process events correctly."""
        # Deploy test strategies
        self.risk_manager.deploy_strategy(strategy_id="test_strategy_001")
        
        # Simulate a trade execution event
        trade_event = Event(
            event_type=EventType.TRADE_EXECUTED,
            source="Broker",
            data={
                "strategy_id": "test_strategy_001",
                "symbol": "SPY",
                "quantity": 10,
                "price": 400.0,
                "direction": "buy"
            },
            timestamp=datetime.now()
        )
        self.event_bus.publish(trade_event)
        
        # Verify trade was recorded
        self.assertEqual(
            self.risk_manager.risk_metrics["test_strategy_001"]["trade_count_today"],
            1
        )
        self.assertEqual(
            self.risk_manager.risk_metrics["test_strategy_001"]["position_count"],
            1
        )
        
        # Simulate a position closed event
        position_event = Event(
            event_type=EventType.POSITION_CLOSED,
            source="Broker",
            data={
                "strategy_id": "test_strategy_001",
                "symbol": "SPY",
                "quantity": 10,
                "price": 410.0,
                "profit_loss": 100.0
            },
            timestamp=datetime.now()
        )
        self.event_bus.publish(position_event)
        
        # Verify position closure was recorded
        self.assertEqual(
            self.risk_manager.risk_metrics["test_strategy_001"]["position_count"],
            0  # Should be decremented
        )
        self.assertEqual(
            self.risk_manager.risk_metrics["test_strategy_001"]["daily_profit_loss"],
            100.0
        )
        self.assertEqual(
            self.risk_manager.deployed_strategies["test_strategy_001"]["performance"]["profit_loss"],
            100.0
        )
    
    def test_risk_level_change_handler(self):
        """Test that risk level changes adjust allocations."""
        # Deploy test strategies
        self.risk_manager.deploy_strategy(
            strategy_id="test_strategy_001",
            allocation_percentage=10.0
        )
        self.risk_manager.deploy_strategy(
            strategy_id="test_strategy_002",
            allocation_percentage=10.0
        )
        
        # Simulate a risk level change event
        risk_event = Event(
            event_type=EventType.RISK_LEVEL_CHANGED,
            source="RiskManager",
            data={
                "old_level": RiskLevel.MEDIUM,
                "new_level": RiskLevel.HIGH
            },
            timestamp=datetime.now()
        )
        self.event_bus.publish(risk_event)
        
        # Verify allocations were reduced
        self.assertAlmostEqual(
            self.risk_manager.strategy_allocations["test_strategy_001"],
            7.0,  # 10.0 * 0.7
            places=1
        )
        self.assertAlmostEqual(
            self.risk_manager.strategy_allocations["test_strategy_002"],
            7.0,  # 10.0 * 0.7
            places=1
        )
    
    def test_singleton_pattern(self):
        """Test that the singleton pattern works correctly."""
        # Get two instances
        instance1 = get_autonomous_risk_manager(
            risk_config=self.risk_config,
            data_dir=self.test_data_dir
        )
        instance2 = get_autonomous_risk_manager()
        
        # They should be the same object
        self.assertIs(instance1, instance2)
    
    def test_state_persistence(self):
        """Test saving and loading state."""
        # Deploy a strategy
        self.risk_manager.deploy_strategy(
            strategy_id="test_strategy_001",
            allocation_percentage=10.0
        )
        
        # Save state
        self.risk_manager._save_state()
        
        # Create a new manager instance that should load the saved state
        with patch('trading_bot.autonomous.risk_integration.RiskManager'):
            new_manager = AutonomousRiskManager(data_dir=self.test_data_dir)
            
            # Verify state was loaded
            self.assertIn("test_strategy_001", new_manager.deployed_strategies)
            self.assertEqual(new_manager.strategy_allocations["test_strategy_001"], 10.0)


class EventTracker:
    """Helper class to track events for testing."""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.events = []
        self.is_tracking = False
    
    def _handle_event(self, event_type, data):
        """Event handler to record events."""
        event = Event(event_type=event_type, source=data.get("source"), data=data, timestamp=data.get("timestamp"))
        self.events.append(event)
    
    def start_tracking(self):
        """Start tracking events."""
        if not self.is_tracking:
            # Register for all event types
            for event_type in dir(EventType):
                if event_type.isupper():
                    self.event_bus.register(getattr(EventType, event_type), self._handle_event)
            
            self.is_tracking = True
    
    def stop_tracking(self):
        """Stop tracking events."""
        if self.is_tracking:
            # Unregister for all event types
            for event_type in dir(EventType):
                if event_type.isupper():
                    self.event_bus.unregister(getattr(EventType, event_type), self._handle_event)
            
            self.is_tracking = False
    
    def get_events(self):
        """Get all tracked events."""
        return self.events
    
    def get_events_by_type(self, event_type):
        """Get events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def clear_events(self):
        """Clear tracked events."""
        self.events = []


if __name__ == "__main__":
    unittest.main()
