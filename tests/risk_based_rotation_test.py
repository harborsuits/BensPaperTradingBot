#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk-Based Strategy Rotation Test

This script demonstrates how the Risk-Based Strategy Rotation system integrates with
the Risk Management Engine to automatically adjust strategy allocations based on
detected risk factors.
"""
import os
import sys
import time
import logging
import random
import json
from datetime import datetime, timedelta
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
from trading_bot.risk.risk_based_strategy_rotation import RiskBasedStrategyRotation

# Use a simplified mock instead of the real risk management engine
class SimpleRiskEngine:
    """A simplified risk engine for testing."""
    
    def __init__(self, config, persistence_manager=None):
        self.config = config or {}
        self.persistence = persistence_manager
        self.event_bus = get_global_event_bus()
        
        # Initialize risk parameters
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.max_position_size = self.config.get('max_position_size', 0.2)
        self.drawdown_threshold = self.config.get('drawdown_threshold', 0.1)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        
        logger.info("Simple Risk Engine initialized")
    
    def register_event_handlers(self):
        """Register for relevant events from the event bus."""
        # Just a placeholder that does nothing for this test
        pass

class MockStrategy:
    """Mock strategy for testing."""
    def __init__(self, strategy_id, name, metadata=None):
        self.id = strategy_id
        self.name = name
        self.metadata = metadata or {}
        self.is_active = False
        
    def __repr__(self):
        return f"Strategy({self.id}, {self.name}, active={self.is_active})"


class MockStrategyManager:
    """Mock strategy manager for testing."""
    def __init__(self):
        self.strategies = {}
        self.active_strategies = {}
        
    def add_strategy(self, strategy):
        """Add a strategy to the manager."""
        self.strategies[strategy.id] = strategy
        
    def get_all_strategies(self):
        """Get all strategies."""
        return self.strategies
        
    def get_active_strategies(self):
        """Get active strategies."""
        return {k: v for k, v in self.strategies.items() if v.is_active}
        
    def get_strategy(self, strategy_id):
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
        
    def promote_strategy(self, strategy_id, reason=None):
        """Promote a strategy to active status."""
        if strategy_id in self.strategies:
            logger.info(f"Promoting strategy {strategy_id} (reason: {reason})")
            self.strategies[strategy_id].is_active = True
            return True
        return False
        
    def demote_strategy(self, strategy_id, reason=None):
        """Demote a strategy from active status."""
        if strategy_id in self.strategies:
            logger.info(f"Demoting strategy {strategy_id} (reason: {reason})")
            self.strategies[strategy_id].is_active = False
            return True
        return False


def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def create_mock_persistence():
    """Create a mock persistence manager for testing."""
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
    
    return MockPersistence()

def create_mock_strategies():
    """Create a set of mock strategies for testing."""
    strategies = [
        # Momentum strategies - higher beta and volatility
        MockStrategy("momentum_1", "Momentum Strategy 1", {
            "strategy_type": "momentum",
            "market_beta": 0.85,
            "volatility": 0.75,
            "correlation_risk": 0.6,
            "sector_bias": 0.4,
            "liquidity_risk": 0.3
        }),
        MockStrategy("momentum_2", "Momentum Strategy 2", {
            "strategy_type": "momentum",
            "market_beta": 0.8,
            "volatility": 0.7,
            "correlation_risk": 0.55,
            "sector_bias": 0.5,
            "liquidity_risk": 0.25
        }),
        
        # Value strategies - moderate beta and lower volatility
        MockStrategy("value_1", "Value Strategy 1", {
            "strategy_type": "value",
            "market_beta": 0.6,
            "volatility": 0.4,
            "correlation_risk": 0.5,
            "sector_bias": 0.6,
            "liquidity_risk": 0.4
        }),
        MockStrategy("value_2", "Value Strategy 2", {
            "strategy_type": "value",
            "market_beta": 0.55,
            "volatility": 0.35,
            "correlation_risk": 0.45,
            "sector_bias": 0.7,
            "liquidity_risk": 0.5
        }),
        
        # Mean reversion strategies - lower beta but can have higher volatility
        MockStrategy("mean_rev_1", "Mean Reversion 1", {
            "strategy_type": "mean_reversion",
            "market_beta": 0.3,
            "volatility": 0.6,
            "correlation_risk": 0.4,
            "sector_bias": 0.5,
            "liquidity_risk": 0.3
        }),
        MockStrategy("mean_rev_2", "Mean Reversion 2", {
            "strategy_type": "mean_reversion",
            "market_beta": 0.25,
            "volatility": 0.55,
            "correlation_risk": 0.35,
            "sector_bias": 0.4,
            "liquidity_risk": 0.35
        }),
        
        # Low volatility strategies
        MockStrategy("low_vol_1", "Low Volatility 1", {
            "strategy_type": "low_volatility",
            "market_beta": 0.2,
            "volatility": 0.2,
            "correlation_risk": 0.3,
            "sector_bias": 0.6,
            "liquidity_risk": 0.5
        }),
        MockStrategy("low_vol_2", "Low Volatility 2", {
            "strategy_type": "low_volatility",
            "market_beta": 0.15,
            "volatility": 0.15,
            "correlation_risk": 0.25,
            "sector_bias": 0.7,
            "liquidity_risk": 0.55
        }),
    ]
    
    return strategies

def main():
    """Main test function."""
    print_section("RISK-BASED STRATEGY ROTATION TEST")
    
    # Create event bus
    event_bus = get_global_event_bus()
    
    # Create mock persistence
    persistence = create_mock_persistence()
    
    # Create strategy intelligence recorder
    recorder = StrategyIntelligenceRecorder(persistence, event_bus)
    
    # Create mock strategy manager
    strategy_manager = MockStrategyManager()
    
    # Create and add mock strategies
    strategies = create_mock_strategies()
    for strategy in strategies:
        strategy_manager.add_strategy(strategy)
    
    # Activate a few strategies initially
    strategy_manager.promote_strategy("momentum_1")
    strategy_manager.promote_strategy("value_1")
    strategy_manager.promote_strategy("mean_rev_1")
    
    # Create risk management engine (simplified mock)
    risk_config = {
        "max_portfolio_risk": 0.05,  # 5% maximum portfolio risk
        "correlation_threshold": 0.7,  # Alert on correlations above 0.7
        "max_position_size": 0.2,  # No position can be > 20% of portfolio
        "drawdown_threshold": 0.1,  # Alert on 10% drawdowns
        "risk_per_trade": 0.01,  # Risk 1% per trade
        "initial_portfolio_value": 100000.0  # Start with $100k
    }
    
    risk_engine = SimpleRiskEngine(risk_config, persistence)
    
    # Create risk-based strategy rotation
    rotation_config = {
        "max_active_strategies": 3,
        "risk_factor_weights": {
            "market_beta": 0.25,
            "volatility": 0.3,
            "correlation": 0.15,
            "sector_exposure": 0.2,
            "liquidity": 0.1
        }
    }
    
    rotation_system = RiskBasedStrategyRotation(
        strategy_manager=strategy_manager,
        event_bus=event_bus,
        config=rotation_config
    )
    
    # Register for events
    risk_engine.register_event_handlers()
    
    # Print initial state
    print("\nInitial active strategies:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 1: Normal Market Regime
    print_section("TEST 1: NORMAL MARKET REGIME")
    
    print("Publishing risk attribution for normal market conditions...")
    event_bus.create_and_publish(
        event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
        data={
            "risk_factors": {
                "market_beta": 0.02,
                "volatility": 0.01,
                "correlation": 0.005,
                "sector_exposure": 0.015,
                "liquidity": 0.005
            },
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after normal market risk attribution:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 2: Volatile Market Regime
    print_section("TEST 2: VOLATILE MARKET REGIME")
    
    print("Publishing market regime change to volatile...")
    event_bus.create_and_publish(
        event_type=EventType.MARKET_REGIME_CHANGED,
        data={
            "symbol": "SPY",
            "current_regime": "volatile",
            "confidence": 0.85,
            "previous_regime": "normal",
            "timestamp": datetime.now().isoformat(),
            "trigger": "volatility_spike"
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("Publishing risk attribution for volatile market...")
    event_bus.create_and_publish(
        event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
        data={
            "risk_factors": {
                "market_beta": 0.04,
                "volatility": 0.03,
                "correlation": 0.015,
                "sector_exposure": 0.02,
                "liquidity": 0.01
            },
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after volatile market regime:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 3: Correlation Risk Alert
    print_section("TEST 3: CORRELATION RISK ALERT")
    
    print("Publishing correlation risk alert...")
    event_bus.create_and_publish(
        event_type=EventType.CORRELATION_RISK_ALERT,
        data={
            "symbols": ["AAPL", "MSFT"],
            "correlation": 0.85,
            "threshold": 0.7,
            "action": "diversify_assets",
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after correlation risk alert:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 4: Drawdown Protection
    print_section("TEST 4: DRAWDOWN PROTECTION (DEFENSIVE ROTATION)")
    
    print("Publishing significant drawdown alert...")
    event_bus.create_and_publish(
        event_type=EventType.DRAWDOWN_THRESHOLD_EXCEEDED,
        data={
            "current_drawdown": 0.15,
            "threshold": 0.10,
            "exceeded": True,
            "severity": 2,
            "action": "reduce_exposure",
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after significant drawdown:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 5: Trending Market Regime
    print_section("TEST 5: TRENDING MARKET REGIME")
    
    print("Publishing market regime change to trending...")
    event_bus.create_and_publish(
        event_type=EventType.MARKET_REGIME_CHANGED,
        data={
            "symbol": "SPY",
            "current_regime": "trending",
            "confidence": 0.8,
            "previous_regime": "volatile",
            "timestamp": datetime.now().isoformat(),
            "trigger": "sustained_momentum"
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("Publishing risk attribution for trending market...")
    event_bus.create_and_publish(
        event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
        data={
            "risk_factors": {
                "market_beta": 0.03,
                "volatility": 0.02,
                "correlation": 0.01,
                "sector_exposure": 0.015,
                "liquidity": 0.005
            },
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after trending market regime:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Summary
    print_section("RISK-BASED ROTATION SUMMARY")
    
    print("\nFinal active strategies:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    print("\nStrategy rotation successfully demonstrated!")

if __name__ == "__main__":
    main()
