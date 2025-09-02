#!/usr/bin/env python3
"""
Integration Tests for Broker Intelligence System

This test suite simulates broker failures and verifies that
the intelligence engine correctly identifies issues, trips
circuit breakers, and generates appropriate recommendations.
"""

import os
import sys
import unittest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from trading_bot.brokers.metrics.base import MetricType, MetricOperation, MetricPeriod
from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.intelligence.broker_advisor import BrokerAdvisor, BrokerSelectionAdvice
from trading_bot.brokers.intelligence.multi_broker_integration import BrokerIntelligenceEngine
from trading_bot.brokers.intelligence.orchestrator_integration import (
    OrchestratorAdvisor,
    MultiBrokerAdvisorIntegration
)
from trading_bot.event_system.event_bus import EventBus
from trading_bot.event_system.event_types import EventType


class MockBrokerInterface:
    """Mock broker interface for testing"""
    
    def __init__(self, broker_id):
        self.broker_id = broker_id
        self.connected = True
        self.error_rate = 0.0
        self.latency = 100  # ms
        self.slippage = 0.01  # %
        self.commission = 1.0  # $
        
        # Track calls for verification
        self.calls = []
    
    def connect(self):
        self.calls.append(("connect", None))
        return self.connected
    
    def disconnect(self):
        self.calls.append(("disconnect", None))
        self.connected = False
    
    def place_order(self, order_params):
        self.calls.append(("place_order", order_params))
        # Simulate failure based on error rate
        import random
        if random.random() < self.error_rate:
            raise Exception("Simulated broker failure")
        return {"order_id": "test_order_123"}
    
    def get_positions(self):
        self.calls.append(("get_positions", None))
        # Simulate failure based on error rate
        import random
        if random.random() < self.error_rate:
            raise Exception("Simulated broker failure")
        return [{"symbol": "AAPL", "quantity": 100}]
    
    # Additional methods to simulate different broker operations
    def get_quote(self, symbol):
        self.calls.append(("get_quote", symbol))
        # Simulate failure based on error rate
        import random
        if random.random() < self.error_rate:
            raise Exception("Simulated broker failure")
        return {"bid": 150.0, "ask": 150.1}


class MockMultiBrokerManager:
    """Mock multi-broker manager for testing"""
    
    def __init__(self):
        self.brokers = {}
        self.primary_broker_id = None
        self.asset_routing = {}
        
    def add_broker(self, broker_id, broker_interface, asset_classes=None):
        self.brokers[broker_id] = {
            "interface": broker_interface,
            "asset_classes": asset_classes or ["equities"]
        }
        if not self.primary_broker_id:
            self.primary_broker_id = broker_id
    
    def set_asset_routing(self, asset_class, broker_id):
        self.asset_routing[asset_class] = broker_id
    
    def get_broker_interface(self, broker_id):
        return self.brokers.get(broker_id, {}).get("interface")
    
    def get_asset_routing(self, asset_class=None):
        if asset_class:
            return self.asset_routing.get(asset_class)
        return self.asset_routing
    
    def get_registered_brokers(self):
        return list(self.brokers.keys())


class TestBrokerIntelligenceIntegration(unittest.TestCase):
    """Integration tests for the broker intelligence system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create event bus
        self.event_bus = EventBus()
        
        # Create metrics manager
        self.metrics_manager = BrokerMetricsManager(self.event_bus)
        
        # Load test configuration
        self.config = {
            "factor_weights": {
                "latency": 0.20,
                "reliability": 0.40,
                "execution_quality": 0.25,
                "cost": 0.15
            },
            "circuit_breaker_thresholds": {
                "error_count": 3,  # Lower for testing
                "error_rate": 0.3,
                "availability_min": 90.0,
                "reset_after_seconds": 5  # Short time for testing
            },
            "failover_threshold": 20.0,
            "health_threshold_normal": 80.0,
            "health_threshold_warning": 60.0
        }
        
        # Create broker advisor
        self.advisor = BrokerAdvisor(
            metrics_manager=self.metrics_manager,
            config=self.config
        )
        
        # Create intelligence engine
        self.intelligence_engine = BrokerIntelligenceEngine(
            event_bus=self.event_bus,
            metrics_manager=self.metrics_manager,
            broker_advisor=self.advisor,
            config=self.config
        )
        
        # Create mock broker manager
        self.broker_manager = MockMultiBrokerManager()
        
        # Create orchestrator advisor
        self.orchestrator_advisor = OrchestratorAdvisor(
            event_bus=self.event_bus,
            broker_advisor=self.advisor
        )
        
        # Create multi-broker integration
        self.broker_integration = MultiBrokerAdvisorIntegration(
            broker_manager=self.broker_manager,
            intelligence_engine=self.intelligence_engine,
            event_bus=self.event_bus
        )
        
        # Register event handlers
        self.captured_events = []
        self.event_bus.subscribe(
            event_type=EventType.BROKER_INTELLIGENCE,
            handler=self._capture_intelligence_event
        )
        self.event_bus.subscribe(
            event_type=EventType.ORCHESTRATOR_ADVISORY,
            handler=self._capture_advisory_event
        )
        
        # Set up test brokers
        self._setup_test_brokers()
    
    def _capture_intelligence_event(self, event):
        """Capture broker intelligence events for inspection"""
        self.captured_events.append({
            "type": "intelligence",
            "data": event
        })
    
    def _capture_advisory_event(self, event):
        """Capture orchestrator advisory events for inspection"""
        self.captured_events.append({
            "type": "advisory",
            "data": event
        })
    
    def _setup_test_brokers(self):
        """Set up test brokers for the integration test"""
        # Create mock broker interfaces
        self.broker_a = MockBrokerInterface("broker_a")
        self.broker_b = MockBrokerInterface("broker_b")
        self.broker_c = MockBrokerInterface("broker_c")
        
        # Register brokers with the manager
        self.broker_manager.add_broker("broker_a", self.broker_a, ["equities", "forex"])
        self.broker_manager.add_broker("broker_b", self.broker_b, ["equities", "options"])
        self.broker_manager.add_broker("broker_c", self.broker_c, ["forex", "crypto"])
        
        # Set initial routing
        self.broker_manager.set_asset_routing("equities", "broker_a")
        self.broker_manager.set_asset_routing("forex", "broker_c")
        self.broker_manager.set_asset_routing("options", "broker_b")
        
        # Register brokers with the intelligence system
        self.broker_integration.register_brokers()
    
    def test_broker_failure_detection(self):
        """Test broker failure detection and circuit breaker triggering"""
        # Start with normal broker operation
        self.assertEqual(len(self.captured_events), 0)
        
        # Simulate a broker failing several operations
        self.broker_a.error_rate = 1.0  # 100% failure rate
        
        # Simulate events that would trigger metrics collection
        for _ in range(5):
            # Simulate order placement failure
            with self.assertRaises(Exception):
                self.broker_a.place_order({"symbol": "AAPL", "quantity": 100})
            
            # Let metrics manager know of the failure
            self.metrics_manager.record_broker_error(
                broker_id="broker_a",
                operation_type="order",
                asset_class="equities",
                error_type="connection_error"
            )
        
        # Manually trigger intelligence check (normally done by event loop)
        self.intelligence_engine.check_broker_health()
        
        # Verify circuit breaker was tripped
        self.assertTrue(self.advisor.is_circuit_breaker_active("broker_a"))
        
        # Verify intelligence events were emitted
        intelligence_events = [e for e in self.captured_events if e["type"] == "intelligence"]
        self.assertGreater(len(intelligence_events), 0)
        
        # Check for circuit breaker event
        circuit_breaker_events = [
            e for e in intelligence_events
            if e["data"].get("event_subtype") == "circuit_breaker_tripped"
        ]
        self.assertGreater(len(circuit_breaker_events), 0)
        
        # Verify the event data
        event_data = circuit_breaker_events[0]["data"]
        self.assertEqual(event_data["broker_id"], "broker_a")
        self.assertIn("reason", event_data)
    
    def test_broker_failover_recommendations(self):
        """Test broker failover recommendations when performance degrades"""
        # Set broker_a (primary) to have worse performance metrics
        self.broker_a.latency = 500  # Higher latency
        self.broker_a.error_rate = 0.2  # 20% error rate
        
        # Set broker_b (backup) to have better performance
        self.broker_b.latency = 100  # Lower latency
        self.broker_b.error_rate = 0.0  # No errors
        
        # Manually record some metrics for both brokers
        # Record metrics for broker_a (poor performance)
        self.metrics_manager.record_broker_latency(
            broker_id="broker_a",
            operation_type="order",
            asset_class="equities",
            latency_ms=500
        )
        self.metrics_manager.record_broker_error(
            broker_id="broker_a",
            operation_type="order",
            asset_class="equities",
            error_type="timeout"
        )
        
        # Record metrics for broker_b (good performance)
        self.metrics_manager.record_broker_latency(
            broker_id="broker_b",
            operation_type="order",
            asset_class="equities",
            latency_ms=100
        )
        
        # Request a selection recommendation
        self.intelligence_engine.emit_broker_selection_event(
            asset_class="equities",
            operation_type="order"
        )
        
        # Verify advisory events were emitted
        advisory_events = [e for e in self.captured_events if e["type"] == "advisory"]
        self.assertGreater(len(advisory_events), 0)
        
        # Check for selection advice
        selection_events = [
            e for e in advisory_events
            if e["data"].get("event_subtype") == "broker_selection_advice"
        ]
        self.assertGreater(len(selection_events), 0)
        
        # Verify the recommendation prioritizes broker_b
        event_data = selection_events[0]["data"]
        advice = BrokerSelectionAdvice.from_dict(event_data["advice"])
        self.assertEqual(advice.asset_class, "equities")
        self.assertEqual(advice.operation_type, "order")
        
        # broker_b should be recommended over broker_a due to better performance
        self.assertEqual(advice.primary_broker_id, "broker_b")
        
        # Verify failover is recommended
        self.assertTrue(advice.is_failover_recommended)
    
    def test_automatic_circuit_breaker_reset(self):
        """Test automatic circuit breaker reset after time period"""
        # Trip circuit breaker on broker_a
        self.advisor.trip_circuit_breaker(
            broker_id="broker_a",
            reason="Test reason",
            reset_after_seconds=2  # Very short for testing
        )
        
        # Verify circuit breaker is active
        self.assertTrue(self.advisor.is_circuit_breaker_active("broker_a"))
        
        # Wait for reset time
        time.sleep(2.5)  # Give extra time for safety
        
        # Verify circuit breaker has automatically reset
        self.assertFalse(self.advisor.is_circuit_breaker_active("broker_a"))
        
        # Manually trigger health check to generate reset event
        self.intelligence_engine.check_broker_health()
        
        # Verify reset event was emitted
        intelligence_events = [e for e in self.captured_events if e["type"] == "intelligence"]
        reset_events = [
            e for e in intelligence_events
            if e["data"].get("event_subtype") == "circuit_breaker_reset"
        ]
        self.assertGreater(len(reset_events), 0)
    
    def test_broker_health_status_changes(self):
        """Test broker health status monitoring and changes"""
        # Start with good metrics
        self.metrics_manager.record_broker_latency(
            broker_id="broker_a",
            operation_type="order",
            asset_class="equities",
            latency_ms=100
        )
        
        # Check initial health status
        self.intelligence_engine.check_broker_health()
        
        # Find initial health event
        intelligence_events = [e for e in self.captured_events if e["type"] == "intelligence"]
        health_events = [
            e for e in intelligence_events
            if e["data"].get("event_subtype") == "broker_health_update"
        ]
        
        if health_events:
            initial_health = health_events[-1]["data"]["health_status"]
            # Should be NORMAL with good metrics
            self.assertEqual(initial_health, "NORMAL")
        
        # Clear events for next test
        self.captured_events.clear()
        
        # Degrade metrics for broker_a
        self.broker_a.latency = 400  # Higher latency
        self.broker_a.error_rate = 0.15  # 15% error rate
        
        # Record degraded metrics
        self.metrics_manager.record_broker_latency(
            broker_id="broker_a",
            operation_type="order",
            asset_class="equities",
            latency_ms=400
        )
        self.metrics_manager.record_broker_error(
            broker_id="broker_a",
            operation_type="order",
            asset_class="equities",
            error_type="timeout"
        )
        
        # Check health again
        self.intelligence_engine.check_broker_health()
        
        # Find updated health event
        intelligence_events = [e for e in self.captured_events if e["type"] == "intelligence"]
        health_events = [
            e for e in intelligence_events
            if e["data"].get("event_subtype") == "broker_health_update"
        ]
        
        if health_events:
            updated_health = health_events[-1]["data"]["health_status"]
            # Should be CAUTION with degraded metrics
            self.assertEqual(updated_health, "CAUTION")


if __name__ == '__main__':
    unittest.main()
