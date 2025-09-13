#!/usr/bin/env python3
"""
Integration Tests for Autonomous Risk Management

This module tests the integration between the autonomous trading engine
and risk management systems using actual implementation classes.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import unittest

# Configure paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_risk_integration")

# Import components
from trading_bot.autonomous.autonomous_engine import AutonomousEngine, StrategyCandidate
from trading_bot.autonomous.risk_integration import AutonomousRiskManager, get_autonomous_risk_manager
from trading_bot.risk.risk_manager import RiskManager, RiskLevel, StopLossType
from trading_bot.event_system import EventBus, Event, EventType
from trading_bot.testing.market_data_generator import MarketDataGenerator

class AutonomousRiskIntegrationTest(unittest.TestCase):
    """Integration test for autonomous risk management system."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create test data directory
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      "test_data")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create event bus
        self.event_bus = EventBus()
        
        # Create event tracker
        self.event_tracker = EventTracker(self.event_bus)
        self.event_tracker.start_tracking()
        
        # Create market data generator
        self.market_gen = MarketDataGenerator(
            num_stocks=5,
            days=30,
            volatility_range=(0.15, 0.35),
            price_range=(50, 500)
        )
        
        # Generate market data
        self.market_data = self.market_gen.generate_market_data()
        self.options_data = self.market_gen.generate_options_data(self.market_data)
        
        # Create autonomous engine
        self.engine_config = {
            "use_real_data": False,
            "test_mode": True,
            "data_dir": self.test_dir
        }
        self.engine = AutonomousEngine(config=self.engine_config)
        self.engine.event_bus = self.event_bus
        
        # Initialize with test data
        self.engine._market_data = self.market_data
        self.engine._options_data = self.options_data
        
        # Create risk config
        self.risk_config = {
            "portfolio_value": 100000,
            "risk_per_trade_pct": 1.0,
            "default_stop_loss_type": "VOLATILITY",
            "default_risk_level": "MEDIUM",
            "max_allocation_pct": 5.0,
            "max_position_size_pct": 2.0
        }
        
        # Create risk manager
        self.risk_manager = get_autonomous_risk_manager(
            risk_config=self.risk_config,
            data_dir=os.path.join(self.test_dir, "risk")
        )
        self.risk_manager.event_bus = self.event_bus
        self.risk_manager.connect_engine(self.engine)
        
        # Create test strategies
        self._create_test_strategies()
    
    def tearDown(self):
        """Clean up after tests."""
        self.event_tracker.stop_tracking()
    
    def _create_test_strategies(self):
        """Create test strategies for the engine."""
        # Create a few strategy candidates
        symbols = list(self.market_data.keys())[:3]
        
        # Top candidate 1 - Iron Condor
        candidate1 = StrategyCandidate(
            strategy_id="test_iron_condor_001",
            strategy_type="iron_condor",
            symbols=[symbols[0]],
            universe="options",
            parameters={
                "delta": 0.3,
                "dte": 45,
                "take_profit": 50
            }
        )
        candidate1.returns = 15.2
        candidate1.sharpe_ratio = 1.8
        candidate1.drawdown = 12.4
        candidate1.win_rate = 68.0
        candidate1.profit_factor = 1.9
        candidate1.trades_count = 25
        candidate1.status = "backtested"
        candidate1.meets_criteria = True
        
        # Top candidate 2 - Strangle
        candidate2 = StrategyCandidate(
            strategy_id="test_strangle_002",
            strategy_type="strangle",
            symbols=[symbols[1]],
            universe="options",
            parameters={
                "delta": 0.2,
                "dte": 30,
                "take_profit": 60
            }
        )
        candidate2.returns = 18.5
        candidate2.sharpe_ratio = 2.1
        candidate2.drawdown = 10.2
        candidate2.win_rate = 72.0
        candidate2.profit_factor = 2.2
        candidate2.trades_count = 18
        candidate2.status = "backtested"
        candidate2.meets_criteria = True
        
        # Near-miss candidate
        candidate3 = StrategyCandidate(
            strategy_id="test_butterfly_003",
            strategy_type="butterfly",
            symbols=[symbols[2]],
            universe="options",
            parameters={
                "width": 5,
                "dte": 15,
                "take_profit": 40
            }
        )
        candidate3.returns = 8.5
        candidate3.sharpe_ratio = 0.9
        candidate3.drawdown = 18.2
        candidate3.win_rate = 58.0
        candidate3.profit_factor = 1.4
        candidate3.trades_count = 22
        candidate3.status = "backtested"
        candidate3.meets_criteria = False
        
        # Add candidates to engine
        self.engine.strategy_candidates = {
            candidate1.strategy_id: candidate1,
            candidate2.strategy_id: candidate2,
            candidate3.strategy_id: candidate3
        }
        
        # Set top candidates
        self.engine.top_candidates = [candidate1, candidate2]
        
        # Set near-miss candidates
        self.engine.near_miss_candidates = [candidate3]
    
    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.risk_manager)
        self.assertIsNotNone(self.event_bus)
        
        # Verify engine has strategies
        self.assertEqual(len(self.engine.strategy_candidates), 3)
        self.assertEqual(len(self.engine.top_candidates), 2)
        self.assertEqual(len(self.engine.near_miss_candidates), 1)
        
        # Verify risk manager has engine
        self.assertEqual(self.risk_manager.engine, self.engine)
    
    def test_strategy_deployment(self):
        """Test deploying a strategy with risk controls."""
        # Get a strategy to deploy
        strategy = self.engine.top_candidates[0]
        
        # Deploy the strategy
        self.assertTrue(
            self.risk_manager.deploy_strategy(
                strategy_id=strategy.strategy_id,
                allocation_percentage=5.0,
                risk_level=RiskLevel.MEDIUM,
                stop_loss_type=StopLossType.VOLATILITY
            )
        )
        
        # Verify deployment records
        self.assertIn(strategy.strategy_id, self.risk_manager.deployed_strategies)
        self.assertEqual(
            self.risk_manager.strategy_allocations[strategy.strategy_id],
            5.0
        )
        
        # Check event emission
        events = self.event_tracker.get_events_by_type(EventType.STRATEGY_DEPLOYED_WITH_RISK)
        self.assertGreaterEqual(len(events), 1)
        
        last_event = events[-1]
        self.assertEqual(last_event.data["strategy_id"], strategy.strategy_id)
        self.assertEqual(last_event.data["allocation_percentage"], 5.0)
    
    def test_position_sizing(self):
        """Test risk-based position sizing calculation."""
        # Deploy a strategy first
        strategy = self.engine.top_candidates[0]
        self.risk_manager.deploy_strategy(
            strategy_id=strategy.strategy_id,
            allocation_percentage=10.0
        )
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            strategy_id=strategy.strategy_id,
            symbol="AAPL",
            entry_price=175.0,
            stop_price=170.0,
            market_data={"price": 175.0}
        )
        
        # Ensure positive position size
        self.assertGreater(position_size, 0)
        
        # Test with invalid strategy ID
        invalid_position_size = self.risk_manager.calculate_position_size(
            strategy_id="invalid_strategy",
            symbol="AAPL",
            entry_price=175.0,
            stop_price=170.0,
            market_data={"price": 175.0}
        )
        
        self.assertEqual(invalid_position_size, 0)
    
    def test_circuit_breakers(self):
        """Test circuit breaker functionality."""
        # Deploy a strategy
        strategy = self.engine.top_candidates[0]
        self.risk_manager.deploy_strategy(strategy_id=strategy.strategy_id)
        
        # Set up risk metrics to trigger breakers
        self.risk_manager.risk_metrics[strategy.strategy_id] = {
            "current_drawdown": 30.0,  # Exceeds threshold
            "daily_profit_loss": -2000.0,
            "trade_count_today": 5,
            "position_count": 2,
            "largest_position_size": 10000.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Check circuit breakers
        should_halt, reasons = self.risk_manager.check_circuit_breakers({})
        
        # Verify breakers triggered
        self.assertTrue(should_halt)
        self.assertTrue(any("drawdown" in reason.lower() for reason in reasons))
        
        # Check event emission
        events = self.event_tracker.get_events_by_type(EventType.CIRCUIT_BREAKER_TRIGGERED)
        self.assertGreaterEqual(len(events), 1)
    
    def test_pause_resume_strategy(self):
        """Test pausing and resuming a strategy."""
        # Deploy a strategy
        strategy = self.engine.top_candidates[0]
        self.risk_manager.deploy_strategy(strategy_id=strategy.strategy_id)
        
        # Pause the strategy
        pause_reason = "Circuit breaker triggered"
        self.assertTrue(self.risk_manager.pause_strategy(
            strategy_id=strategy.strategy_id,
            reason=pause_reason
        ))
        
        # Verify strategy is paused
        self.assertEqual(
            self.risk_manager.deployed_strategies[strategy.strategy_id]["status"],
            "paused"
        )
        self.assertEqual(
            self.risk_manager.deployed_strategies[strategy.strategy_id]["pause_reason"],
            pause_reason
        )
        
        # Check event emission
        pause_events = self.event_tracker.get_events_by_type(EventType.STRATEGY_PAUSED)
        self.assertGreaterEqual(len(pause_events), 1)
        
        # Resume the strategy
        self.assertTrue(self.risk_manager.resume_strategy(
            strategy_id=strategy.strategy_id
        ))
        
        # Verify strategy is active again
        self.assertEqual(
            self.risk_manager.deployed_strategies[strategy.strategy_id]["status"],
            "active"
        )
        
        # Check event emission
        resume_events = self.event_tracker.get_events_by_type(EventType.STRATEGY_RESUMED)
        self.assertGreaterEqual(len(resume_events), 1)
    
    def test_risk_report_generation(self):
        """Test generating risk reports."""
        # Deploy a few strategies
        for strategy in self.engine.top_candidates:
            self.risk_manager.deploy_strategy(
                strategy_id=strategy.strategy_id,
                allocation_percentage=5.0
            )
        
        # Get risk report
        report = self.risk_manager.get_risk_report()
        
        # Verify report structure
        self.assertIn("timestamp", report)
        self.assertIn("overall_metrics", report)
        self.assertIn("strategy_metrics", report)
        self.assertIn("circuit_breakers", report)
        self.assertEqual(report["total_strategies"], 2)
        self.assertEqual(report["active_strategies"], 2)
        
        # Verify strategy metrics
        for strategy in self.engine.top_candidates:
            self.assertIn(strategy.strategy_id, report["strategy_metrics"])
    
    def test_event_handling(self):
        """Test event handling for trading activities."""
        # Deploy a strategy
        strategy = self.engine.top_candidates[0]
        self.risk_manager.deploy_strategy(strategy_id=strategy.strategy_id)
        
        # Emit trade events
        self.event_bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            source="TestBroker",
            data={
                "strategy_id": strategy.strategy_id,
                "symbol": "AAPL",
                "quantity": 10,
                "price": 175.0,
                "direction": "buy"
            },
            timestamp=datetime.now()
        ))
        
        # Emit position closed event
        self.event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            source="TestBroker",
            data={
                "strategy_id": strategy.strategy_id,
                "symbol": "AAPL",
                "quantity": 10,
                "price": 180.0,
                "profit_loss": 50.0
            },
            timestamp=datetime.now()
        ))
        
        # Verify risk metrics are updated
        self.assertIn(strategy.strategy_id, self.risk_manager.risk_metrics)
        metrics = self.risk_manager.risk_metrics[strategy.strategy_id]
        
        # Position count should be 1 - 1 = 0 (opened one, closed one)
        self.assertEqual(metrics["position_count"], 0)
        
        # Profit/loss should be updated
        self.assertEqual(metrics["daily_profit_loss"], 50.0)
        
        # Performance should be updated
        perf = self.risk_manager.deployed_strategies[strategy.strategy_id]["performance"]
        self.assertEqual(perf["profit_loss"], 50.0)
    
    def test_risk_level_changes(self):
        """Test response to risk level changes."""
        # Deploy strategies
        for i, strategy in enumerate(self.engine.top_candidates):
            self.risk_manager.deploy_strategy(
                strategy_id=strategy.strategy_id,
                allocation_percentage=10.0
            )
        
        # Record initial allocations
        initial_allocations = {
            strategy_id: allocation 
            for strategy_id, allocation in self.risk_manager.strategy_allocations.items()
        }
        
        # Emit risk level change event
        self.event_bus.publish(Event(
            event_type=EventType.RISK_LEVEL_CHANGED,
            source="RiskManager",
            data={
                "old_level": RiskLevel.MEDIUM,
                "new_level": RiskLevel.HIGH
            },
            timestamp=datetime.now()
        ))
        
        # Verify allocations were reduced
        for strategy_id, initial_allocation in initial_allocations.items():
            current_allocation = self.risk_manager.strategy_allocations[strategy_id]
            self.assertLess(current_allocation, initial_allocation)
    
    def test_end_to_end_workflow(self):
        """Test the end-to-end risk management workflow."""
        # Start by deploying a strategy
        strategy = self.engine.top_candidates[0]
        self.risk_manager.deploy_strategy(
            strategy_id=strategy.strategy_id,
            allocation_percentage=5.0
        )
        
        # Calculate position size
        pos_size = self.risk_manager.calculate_position_size(
            strategy_id=strategy.strategy_id,
            symbol="AAPL",
            entry_price=175.0,
            stop_price=170.0,
            market_data={"price": 175.0}
        )
        
        # Simulate opening a position
        self.event_bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            source="TestBroker",
            data={
                "strategy_id": strategy.strategy_id,
                "symbol": "AAPL",
                "quantity": pos_size,
                "price": 175.0,
                "direction": "buy"
            },
            timestamp=datetime.now()
        ))
        
        # Check risk metrics were updated
        self.assertEqual(
            self.risk_manager.risk_metrics[strategy.strategy_id]["position_count"],
            1
        )
        
        # Simulate risk level increase
        self.event_bus.publish(Event(
            event_type=EventType.RISK_LEVEL_CHANGED,
            source="RiskManager",
            data={
                "old_level": RiskLevel.MEDIUM,
                "new_level": RiskLevel.HIGH
            },
            timestamp=datetime.now()
        ))
        
        # Verify allocation was reduced
        self.assertLess(
            self.risk_manager.strategy_allocations[strategy.strategy_id],
            5.0
        )
        
        # Simulate closing a position with profit
        self.event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            source="TestBroker",
            data={
                "strategy_id": strategy.strategy_id,
                "symbol": "AAPL",
                "quantity": pos_size,
                "price": 180.0,
                "profit_loss": 50.0
            },
            timestamp=datetime.now()
        ))
        
        # Check position was closed
        self.assertEqual(
            self.risk_manager.risk_metrics[strategy.strategy_id]["position_count"],
            0
        )
        
        # Check performance was updated
        self.assertEqual(
            self.risk_manager.deployed_strategies[strategy.strategy_id]["performance"]["profit_loss"],
            50.0
        )
        
        # Check win rate was updated
        self.assertGreater(
            self.risk_manager.deployed_strategies[strategy.strategy_id]["performance"]["win_rate"],
            0.0
        )
        
        # Get final risk report
        report = self.risk_manager.get_risk_report()
        
        # Verify report has the expected data
        self.assertEqual(report["total_strategies"], 1)
        self.assertEqual(report["active_strategies"], 1)
        self.assertIn(strategy.strategy_id, report["strategy_metrics"])


class EventTracker:
    """Helper class to track events for testing."""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.events = []
        self.is_tracking = False
    
    def _handle_event(self, event):
        """Event handler to record events."""
        self.events.append(event)
    
    def start_tracking(self):
        """Start tracking events."""
        if not self.is_tracking:
            self.event_bus.subscribe(self._handle_event)
            self.is_tracking = True
    
    def stop_tracking(self):
        """Stop tracking events."""
        if self.is_tracking:
            self.event_bus.unsubscribe(self._handle_event)
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
