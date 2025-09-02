#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Strategy Intelligence Test Script

This script tests the Strategy Intelligence Recorder with mock data
without relying on other complex components in the system.
"""
import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
import random
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the event bus and strategy intelligence recorder
from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def simulate_events(event_bus, num_events=20, interval=0.5):
    """
    Simulate various strategy intelligence events.
    
    Args:
        event_bus: The event bus to publish events to
        num_events: Number of events to simulate
        interval: Interval between events in seconds
    """
    print(f"\nSimulating {num_events} events with {interval}s interval...")
    
    # Define some sample symbols and strategies
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
    strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "RangeTrading"]
    market_regimes = ["trending", "ranging", "volatile", "low_volatility"]
    
    # Generate random events
    for i in range(num_events):
        # Pick a random event type
        event_types = [
            EventType.MARKET_REGIME_CHANGED,
            EventType.ASSET_CLASS_SELECTED,
            EventType.SYMBOL_SELECTED,
            EventType.STRATEGY_SELECTED,
            EventType.PERFORMANCE_ATTRIBUTED
        ]
        event_type = random.choice(event_types)
        
        # Create event data based on type
        if event_type == EventType.MARKET_REGIME_CHANGED:
            data = {
                "symbol": random.choice(symbols),
                "current_regime": random.choice(market_regimes),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "previous_regime": random.choice(market_regimes),
                "timestamp": datetime.now().isoformat(),
                "trigger": random.choice([
                    "volatility_spike", "trend_reversal", 
                    "consolidation", "breakout"
                ])
            }
        elif event_type == EventType.ASSET_CLASS_SELECTED:
            data = {
                "asset_class": "forex",
                "opportunity_score": round(random.uniform(0.1, 0.9), 2),
                "selection_reason": random.choice([
                    "high_volatility", "strong_trends", 
                    "favorable_correlations", "economic_conditions"
                ]),
                "timestamp": datetime.now().isoformat()
            }
        elif event_type == EventType.SYMBOL_SELECTED:
            data = {
                "symbol": random.choice(symbols),
                "score": round(random.uniform(0.1, 0.9), 2),
                "rank": random.randint(1, 5),
                "selection_criteria": {
                    "volatility": round(random.uniform(0.1, 0.9), 2),
                    "trend_strength": round(random.uniform(0.1, 0.9), 2),
                    "liquidity": round(random.uniform(0.1, 0.9), 2)
                },
                "timestamp": datetime.now().isoformat()
            }
        elif event_type == EventType.STRATEGY_SELECTED:
            data = {
                "strategy_id": random.choice(strategies),
                "market_regime": random.choice(market_regimes),
                "compatibility_score": round(random.uniform(0.1, 0.9), 2),
                "selection_reason": random.choice([
                    "highest_compatibility", "best_historical_performance",
                    "optimal_risk_profile", "adaptive_parameters"
                ]),
                "timestamp": datetime.now().isoformat()
            }
        elif event_type == EventType.PERFORMANCE_ATTRIBUTED:
            data = {
                "strategy_id": random.choice(strategies),
                "symbol": random.choice(symbols),
                "factors": {
                    "timing": round(random.uniform(-0.1, 0.1), 3),
                    "sizing": round(random.uniform(-0.1, 0.1), 3),
                    "execution": round(random.uniform(-0.1, 0.1), 3),
                    "market_conditions": round(random.uniform(-0.1, 0.1), 3)
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Publish the event
        event = Event(
            event_type=event_type,
            data=data,
            source="test_script"
        )
        
        event_bus.publish(event)
        print(f"Published event {i+1}/{num_events}: {event_type}")
        print(f"Data: {json.dumps(data, indent=2)}")
        print("-" * 60)
        
        # Sleep between events
        time.sleep(interval)

def main():
    """Main function to run the test."""
    print_section("SIMPLE STRATEGY INTELLIGENCE TEST")
    
    # Get the global event bus
    event_bus = get_global_event_bus()
    
    # Create a mock persistence class
    class MockPersistence:
        def __init__(self):
            self.storage = {}
            
        def is_connected(self):
            return True
            
        def save_strategy_state(self, strategy_id, state_data):
            self.storage[strategy_id] = state_data
            return True
            
        def load_strategy_state(self, strategy_id):
            return self.storage.get(strategy_id, {})
            
        def insert_document(self, collection, document):
            if collection not in self.storage:
                self.storage[collection] = []
            self.storage[collection].append(document)
            return True
            
        def list_collections(self):
            return list(self.storage.keys())
    
    # Create the persistence and intelligence recorder
    persistence = MockPersistence()
    intelligence_recorder = StrategyIntelligenceRecorder(persistence, event_bus)
    
    print("\nInitialized Strategy Intelligence Recorder")
    
    # Subscribe to all events for monitoring
    def event_listener(event):
        logger.debug(f"Event received: {event.event_type} from {event.source}")
    
    event_bus.subscribe_all(event_listener)
    
    # Option 1: Initialize with mock data
    print("\nInitializing mock data...")
    intelligence_recorder.initialize_mock_data()
    
    # Show mock data
    print_section("MARKET REGIME DATA (MOCK)")
    regime_data = persistence.load_strategy_state("market_regime_detector")
    pprint(regime_data)
    
    print_section("ASSET SELECTION DATA (MOCK)")
    asset_data = persistence.load_strategy_state("market_analysis")
    pprint(asset_data)
    
    # Option 2: Simulate events
    print_section("SIMULATING EVENTS")
    simulate_events(event_bus, num_events=10, interval=0.2)
    
    # Show data after events
    print_section("INTELLIGENCE DATA AFTER EVENTS")
    
    # List all collections
    print("\nCollections in persistence:")
    for collection in persistence.list_collections():
        print(f"- {collection}: {len(persistence.storage.get(collection, []))} items")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
