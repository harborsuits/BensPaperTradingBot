#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-End Integration Test for BensBot Enhanced Components

This test script validates the integration of all enhanced reliability components:
1. Persistence Layer (MongoDB)
2. Watchdog & Fault Tolerance
3. Dynamic Capital Scaling
4. Strategy Retirement & Promotion
5. Execution Quality Modeling

It focuses particularly on crash recovery and state restoration.

Usage:
    python end_to_end_test.py
"""

import os
import sys
import time
import logging
import random
import json
import signal
import threading
import unittest
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the enhanced components
from trading_bot.data.persistence import PersistenceManager
from trading_bot.core.watchdog import ServiceWatchdog, RecoveryStrategy
from trading_bot.risk.capital_manager import CapitalManager
from trading_bot.core.strategy_manager import StrategyPerformanceManager, StrategyStatus
from trading_bot.execution.execution_model import ExecutionQualityModel
from trading_bot.core.enhanced_integration import (
    initialize_enhanced_components,
    register_core_services,
    integrate_with_trading_system,
    load_saved_states,
    save_states,
    shutdown_components
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    'persistence': {
        'mongodb_uri': "mongodb://localhost:27017/",
        'database': "bensbot_test",
        'auto_connect': True
    },
    'capital': {
        'initial_capital': 100000.0,
        'risk_params': {
            'base_risk_pct': 0.01,
            'max_account_risk_pct': 0.05,
            'volatility_scaling': True,
            'performance_scaling': True,
            'drawdown_scaling': True
        }
    },
    'watchdog': {
        'check_interval': 5  # Shortened for testing
    },
    'strategy_manager': {
        'evaluation_params': {
            'min_trades': 10,
            'evaluation_window': 30,
            'win_rate_threshold': 0.5,
            'profit_factor_threshold': 1.5,
            'sharpe_threshold': 1.0,
            'drawdown_threshold': 0.15
        }
    },
    'execution': {
        'parameters': {
            'volatility_spread_multiplier': 2.0,
            'market_impact_factor': 0.1,
            'latency_ms': 50
        }
    }
}

class MockDataFeed:
    """Mock data feed service for testing"""
    
    def __init__(self, should_fail: bool = False):
        self.connected = True
        self.should_fail = should_fail
        self.failure_count = 0
        self.max_failures = 2
        
    def is_connected(self) -> bool:
        """Check if data feed is connected"""
        if self.should_fail and self.failure_count < self.max_failures:
            self.failure_count += 1
            self.connected = False
            return False
        return self.connected
    
    def reconnect(self) -> bool:
        """Reconnect the data feed"""
        logger.info("Reconnecting data feed...")
        self.connected = True
        return True
    
    def get_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Get mock data for a symbol"""
        if not self.connected:
            raise ConnectionError("Data feed not connected")
        
        # Create mock OHLCV data
        dates = pd.date_range(end=datetime.datetime.now(), periods=count, freq='1min')
        data = {
            'timestamp': dates,
            'open': np.random.normal(100, 5, count),
            'high': np.random.normal(105, 5, count),
            'low': np.random.normal(95, 5, count),
            'close': np.random.normal(100, 5, count),
            'volume': np.random.randint(100, 1000, count)
        }
        return pd.DataFrame(data)

class MockBroker:
    """Mock broker service for testing"""
    
    def __init__(self, should_fail: bool = False):
        self.connected = True
        self.should_fail = should_fail
        self.failure_count = 0
        self.max_failures = 2
        self.orders = []
        self.positions = {}
        self.account_balance = 100000.0
        
    def is_connected(self) -> bool:
        """Check if broker is connected"""
        if self.should_fail and self.failure_count < self.max_failures:
            self.failure_count += 1
            self.connected = False
            return False
        return self.connected
    
    def reconnect(self) -> bool:
        """Reconnect to the broker"""
        logger.info("Reconnecting broker...")
        self.connected = True
        return True
    
    def place_order(self, symbol: str, order_type: str, quantity: float, 
                   price: Optional[float] = None) -> Dict[str, Any]:
        """Place a mock order"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
        
        order_id = f"order_{len(self.orders) + 1}"
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price or 100.0,
            'status': 'filled',
            'timestamp': datetime.datetime.now()
        }
        self.orders.append(order)
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity
        
        return order
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        return self.account_balance
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.positions

class MockTradingEngine:
    """Mock trading engine for testing"""
    
    def __init__(self, should_fail: bool = False):
        self.running = True
        self.should_fail = should_fail
        self.failure_count = 0
        self.max_failures = 2
        self.strategies = {}
        self.persistence_manager = None
        self.capital_manager = None
        self.strategy_manager = None
        self.execution_model = None
        self.state = {
            'running': True,
            'strategies': {},
            'last_update': datetime.datetime.now().isoformat()
        }
    
    def check_health(self) -> bool:
        """Check if trading engine is healthy"""
        if self.should_fail and self.failure_count < self.max_failures:
            self.failure_count += 1
            self.running = False
            return False
        return self.running
    
    def restore_from_saved_state(self) -> bool:
        """Restore trading engine from saved state"""
        logger.info("Restoring trading engine from saved state...")
        if self.persistence_manager:
            state = self.persistence_manager.load_strategy_state('trading_engine')
            if state:
                self.state = state
                logger.info(f"Restored state: {state}")
        self.running = True
        return True
    
    def add_strategy(self, strategy_id: str, parameters: Dict[str, Any]) -> None:
        """Add a strategy to the trading engine"""
        self.strategies[strategy_id] = parameters
        self.state['strategies'][strategy_id] = parameters
    
    def set_persistence_manager(self, persistence_manager: PersistenceManager) -> None:
        """Set the persistence manager"""
        self.persistence_manager = persistence_manager
    
    def set_capital_manager(self, capital_manager: CapitalManager) -> None:
        """Set the capital manager"""
        self.capital_manager = capital_manager
    
    def set_strategy_manager(self, strategy_manager: StrategyPerformanceManager) -> None:
        """Set the strategy manager"""
        self.strategy_manager = strategy_manager
    
    def set_execution_model(self, execution_model: ExecutionQualityModel) -> None:
        """Set the execution model"""
        self.execution_model = execution_model
    
    def save_state(self) -> None:
        """Save trading engine state"""
        if self.persistence_manager:
            self.state['last_update'] = datetime.datetime.now().isoformat()
            self.persistence_manager.save_strategy_state('trading_engine', self.state)

def run_test_scenario_1():
    """
    Test Scenario 1: Basic Initialization and Integration
    
    This scenario tests the basic initialization and integration of all components.
    """
    logger.info("=== Starting Test Scenario 1: Basic Initialization and Integration ===")
    
    # Initialize components
    components = initialize_enhanced_components(TEST_CONFIG)
    
    # Check that all components were initialized
    assert components.get('persistence') is not None, "Persistence manager not initialized"
    assert components.get('watchdog') is not None, "Watchdog not initialized"
    assert components.get('capital_manager') is not None, "Capital manager not initialized"
    assert components.get('strategy_manager') is not None, "Strategy manager not initialized"
    assert components.get('execution_model') is not None, "Execution model not initialized"
    
    # Create mock services
    data_feed = MockDataFeed()
    broker = MockBroker()
    trading_engine = MockTradingEngine()
    
    # Register services with watchdog
    register_core_services(
        components['watchdog'],
        data_feed,
        broker,
        trading_engine
    )
    
    # Integrate with trading engine
    integrate_with_trading_system(
        trading_engine,
        components
    )
    
    # Check that components were set in trading engine
    assert trading_engine.persistence_manager is not None, "Persistence manager not set in trading engine"
    assert trading_engine.capital_manager is not None, "Capital manager not set in trading engine"
    assert trading_engine.strategy_manager is not None, "Strategy manager not set in trading engine"
    assert trading_engine.execution_model is not None, "Execution model not set in trading engine"
    
    # Start the watchdog
    components['watchdog'].start()
    
    # Wait a bit for watchdog to run health checks
    time.sleep(2)
    
    # Stop watchdog and clean up
    components['watchdog'].stop()
    persistence = components['persistence']
    
    # Clean up test database
    if persistence.is_connected():
        persistence.client.drop_database(TEST_CONFIG['persistence']['database'])
    
    logger.info("=== Test Scenario 1 completed successfully ===")

def run_test_scenario_2():
    """
    Test Scenario 2: Crash Recovery
    
    This scenario tests the crash recovery capabilities of the system.
    """
    logger.info("=== Starting Test Scenario 2: Crash Recovery ===")
    
    # Initialize components
    components = initialize_enhanced_components(TEST_CONFIG)
    
    # Create mock services with failures
    data_feed = MockDataFeed(should_fail=True)
    broker = MockBroker(should_fail=True)
    trading_engine = MockTradingEngine(should_fail=True)
    
    # Register services with watchdog
    register_core_services(
        components['watchdog'],
        data_feed,
        broker,
        trading_engine
    )
    
    # Integrate with trading engine
    integrate_with_trading_system(
        trading_engine,
        components
    )
    
    # Add strategies to trading engine and save state
    trading_engine.add_strategy('test_strategy_1', {'param1': 1, 'param2': 2})
    trading_engine.add_strategy('test_strategy_2', {'param1': 3, 'param2': 4})
    trading_engine.save_state()
    
    # Start the watchdog
    components['watchdog'].start()
    
    # Wait for failures and recovery
    logger.info("Waiting for service failures and recovery...")
    time.sleep(30)  # Allow more time for failures and recoveries
    
    # Verify recovery
    assert data_feed.is_connected(), "Data feed not recovered"
    assert broker.is_connected(), "Broker not recovered"
    assert trading_engine.running, "Trading engine not recovered"
    
    # Stop watchdog and clean up
    components['watchdog'].stop()
    persistence = components['persistence']
    
    # Clean up test database
    if persistence.is_connected():
        persistence.client.drop_database(TEST_CONFIG['persistence']['database'])
    
    logger.info("=== Test Scenario 2 completed successfully ===")

if __name__ == "__main__":
    # Run test scenarios
    run_test_scenario_1()
    run_test_scenario_2()
