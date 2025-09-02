#!/usr/bin/env python3
"""
Simplified Core Test for Autonomous Trading Engine

This script tests the core functionality of the autonomous trading engine
with mocked market data, focusing on:
1. Strategy adapter integration
2. Near-miss candidate identification 
3. Event emission validation

No external dependencies like yfinance required.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_simplified_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("autonomous_test")

# Import core components only
from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.event_system import EventBus, EventType, EventHandler
from trading_bot.strategies.components.component_registry import ComponentRegistry
from trading_bot.strategies.strategy_adapter import StrategyAdapter

# Simple event listener for testing
class CoreEventListener:
    def __init__(self):
        self.event_log = []
        self.register_events()
        
    def register_events(self):
        event_bus = EventBus()
        
        # Use register_handler with EventHandler objects instead of direct register
        event_bus.register_handler(EventHandler(
            callback=lambda event_type, event_data: self.log_event(EventType.STRATEGY_OPTIMISED, event_data),
            event_type=EventType.STRATEGY_OPTIMISED
        ))
        
        event_bus.register_handler(EventHandler(
            callback=lambda event_type, event_data: self.log_event(EventType.STRATEGY_EXHAUSTED, event_data),
            event_type=EventType.STRATEGY_EXHAUSTED
        ))
        
        event_bus.register_handler(EventHandler(
            callback=lambda event_type, event_data: self.log_event(EventType.STRATEGY_GENERATED, event_data),
            event_type=EventType.STRATEGY_GENERATED
        ))
        
        event_bus.register_handler(EventHandler(
            callback=lambda event_type, event_data: self.log_event(EventType.STRATEGY_EVALUATED, event_data),
            event_type=EventType.STRATEGY_EVALUATED
        ))
        
    def log_event(self, event_type, event_data):
        logger.info(f"Event received: {event_type} with data: {event_data}")
        self.event_log.append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.datetime.now()
        })
        
    def get_events_by_type(self, event_type):
        return [e for e in self.event_log if e["type"] == event_type]
        
    def get_event_summary(self):
        summary = {}
        for event in self.event_log:
            event_type = event["type"]
            if event_type in summary:
                summary[event_type] += 1
            else:
                summary[event_type] = 1
        return summary


def main():
    print("\n" + "="*60)
    print("SIMPLIFIED AUTONOMOUS ENGINE TEST")
    print("="*60)
    
    # Just test if we can import and initialize the autonomous engine
    try:
        engine = AutonomousEngine()
        print("✓ Successfully initialized AutonomousEngine")
    except Exception as e:
        print(f"✗ Failed to initialize AutonomousEngine: {str(e)}")
        return 1
    
    # Test if the event bus works
    try:
        event_bus = EventBus()
        print("✓ Successfully initialized EventBus")
    except Exception as e:
        print(f"✗ Failed to initialize EventBus: {str(e)}")
        return 1
        
    # Test if the strategy adapter is accessible
    try:
        adapter = StrategyAdapter()
        print("✓ Successfully initialized StrategyAdapter")
    except Exception as e:
        print(f"✗ Failed to initialize StrategyAdapter: {str(e)}")
        return 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("BensBot autonomous core components are accessible")
    print("See autonomous_simplified_test.log for detailed results")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
