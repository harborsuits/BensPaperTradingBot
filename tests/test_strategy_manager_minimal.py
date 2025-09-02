#!/usr/bin/env python3
"""
Minimal test script for the Enhanced Strategy Manager implementation.
This script avoids dependencies on data providers and other complex components.
"""

import json
import logging
import os
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_strategy_manager")

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports of only the components we need
from trading_bot.core.enhanced_strategy_manager_impl import EnhancedStrategyManager
from trading_bot.core.strategy_manager import StrategyPerformanceManager
from trading_bot.core.event_bus import get_global_event_bus, Event, EventBus
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_base import Strategy, StrategyState, StrategyType

# Create a mock strategy for testing
class MockStrategy(Strategy):
    """A minimal mock strategy for testing."""
    
    def __init__(self, strategy_id="mock_strategy", name="Mock Strategy", **kwargs):
        super().__init__(
            strategy_id=strategy_id,
            name=name,
            description="A mock strategy for testing",
            symbols=kwargs.get("symbols", ["MOCK"]),
            asset_type=kwargs.get("asset_type", "stocks"),
            timeframe=kwargs.get("timeframe", "1m"),
            parameters=kwargs.get("parameters", {}),
            risk_limits=kwargs.get("risk_limits", {}),
            broker_id=kwargs.get("broker_id"),
            enabled=kwargs.get("enabled", True)
        )
        self.strategy_type = StrategyType.MOMENTUM
    
    def generate_signal(self, data):
        """Mock signal generation."""
        logger.info(f"Mock strategy {self.strategy_id} generating signal for {data.get('symbol')}")
        # No actual signal generation in this mock
        return None

# Mock broker manager
class MockBrokerManager:
    """A minimal mock broker manager for testing."""
    
    def __init__(self):
        self.connected = False
        self.brokers = {"mock_broker": None}
        self.active_broker_id = "mock_broker"
    
    def connect_all(self):
        """Mock connection."""
        self.connected = True
        logger.info("Mock broker connected")
        return {"mock_broker": True}
    
    def get_all_positions(self):
        """Mock positions."""
        return {"mock_broker": {}}
    
    def get_position(self, symbol):
        """Mock position."""
        return None
    
    def get_quote(self, symbol):
        """Mock quote."""
        return None
    
    def get_all_accounts(self):
        """Mock accounts."""
        return {"mock_broker": {"buying_power": 10000}}
    
    def get_broker_for_asset_type(self, asset_type):
        """Mock broker routing."""
        return "mock_broker"

def main():
    """Main test function."""
    try:
        logger.info("Starting minimal Enhanced Strategy Manager test")
        
        # Initialize event bus
        event_bus = get_global_event_bus()
        
        # Create mock broker manager
        broker_manager = MockBrokerManager()
        
        # Create performance manager
        performance_manager = StrategyPerformanceManager()
        
        # Create Enhanced Strategy Manager
        strategy_manager = EnhancedStrategyManager(
            broker_manager=broker_manager,
            performance_manager=performance_manager,
            config={
                "risk_limits": {
                    "max_position_per_symbol": 0.05,
                    "max_allocation_per_strategy": 0.20,
                    "max_allocation_per_asset_type": 0.50,
                    "max_total_allocation": 0.80,
                    "max_drawdown": 0.10
                }
            }
        )
        
        # Create some mock strategies
        mock_strategies = [
            {
                "strategy_id": "mock_strategy_1",
                "name": "Mock Strategy 1",
                "description": "First mock strategy",
                "symbols": ["AAPL", "MSFT"],
                "asset_type": "stocks",
                "timeframe": "1h",
                "broker_id": "mock_broker",
                "enabled": True,
                "parameters": {"param1": 10, "param2": 20}
            },
            {
                "strategy_id": "mock_strategy_2",
                "name": "Mock Strategy 2",
                "description": "Second mock strategy",
                "symbols": ["BTC-USD"],
                "asset_type": "crypto",
                "timeframe": "4h",
                "broker_id": "mock_broker",
                "enabled": True,
                "parameters": {"param1": 5, "param2": 15}
            }
        ]
        
        # Register the mock strategy class to make it available
        setattr(sys.modules[__name__], "MockStrategy", MockStrategy)
        
        # Update the class_path to use our mock strategy
        for strategy in mock_strategies:
            strategy["class_path"] = "__main__.MockStrategy"
        
        # Load strategies into manager
        logger.info("Loading mock strategies")
        strategy_manager.load_strategies(mock_strategies)
        
        # Check loaded strategies
        logger.info(f"Loaded {len(strategy_manager.strategies)} strategies")
        for strategy_id, strategy in strategy_manager.strategies.items():
            logger.info(f"  - {strategy.name} ({strategy_id}): {strategy.symbols}")
        
        # Start the strategy manager
        logger.info("Starting strategy manager")
        strategy_manager.start_strategies()
        
        # Simulate some market data events
        logger.info("Simulating market data events")
        for symbol in ["AAPL", "MSFT", "BTC-USD"]:
            # Create mock market data
            mock_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "last": 100.0,
                "bid": 99.8,
                "ask": 100.2,
                "volume": 1000
            }
            
            # Publish event
            event_bus.publish(Event(
                event_type=EventType.MARKET_DATA_UPDATE,
                data=mock_data,
                source="test_script"
            ))
            
            logger.info(f"Published market data for {symbol}")
        
        # Let's see if there were any signals
        logger.info(f"Signals generated: {len(strategy_manager.signal_history)}")
        
        # Check active strategies
        active_strategies = strategy_manager.get_active_strategies()
        logger.info(f"Active strategies: {len(active_strategies)}")
        for strategy in active_strategies:
            logger.info(f"  - {strategy['name']} ({strategy['strategy_id']}): {strategy['state']}")
        
        # Stop the strategy manager
        logger.info("Stopping strategy manager")
        strategy_manager.stop_strategies()
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
