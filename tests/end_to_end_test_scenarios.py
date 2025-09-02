#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Additional End-to-End Test Scenarios for BensBot Enhanced Components

This script contains additional test scenarios for the enhanced reliability components,
focusing on state persistence, capital management, strategy lifecycle, and execution modeling.

Usage:
    python end_to_end_test_scenarios.py
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

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the enhanced components and test fixtures from the main test file
from end_to_end_test import (
    TEST_CONFIG,
    MockDataFeed,
    MockBroker,
    MockTradingEngine,
    initialize_enhanced_components,
    register_core_services,
    integrate_with_trading_system,
    load_saved_states,
    save_states,
    shutdown_components
)

# Import the enhanced components
from trading_bot.data.persistence import PersistenceManager
from trading_bot.core.watchdog import ServiceWatchdog, RecoveryStrategy
from trading_bot.risk.capital_manager import CapitalManager
from trading_bot.core.strategy_manager import StrategyPerformanceManager, StrategyStatus
from trading_bot.execution.execution_model import ExecutionQualityModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mock_trades(count: int = 100) -> List[Dict[str, Any]]:
    """Generate mock trade data for testing"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    strategies = ['trend_following', 'mean_reversion', 'breakout', 'momentum']
    trades = []
    
    for i in range(count):
        symbol = random.choice(symbols)
        strategy_id = random.choice(strategies)
        win = random.random() > 0.4  # 60% win rate
        entry_price = round(random.uniform(1.0, 1.5), 5)
        exit_price = entry_price * (1 + random.uniform(0.001, 0.01)) if win else entry_price * (1 - random.uniform(0.001, 0.01))
        volume = random.randint(1000, 10000) / 1000
        profit_loss = round((exit_price - entry_price) * volume * 10000, 2)  # Pip value calculation
        
        trade = {
            'trade_id': f"trade_{i}",
            'strategy_id': strategy_id,
            'symbol': symbol,
            'direction': 'long' if random.random() > 0.5 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'volume': volume,
            'profit_loss': profit_loss if win else -abs(profit_loss),
            'win': win,
            'entry_time': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30)),
            'exit_time': datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 29)),
            'slippage': random.uniform(0.1, 2.0),
            'spread': random.uniform(0.5, 3.0),
            'latency': random.uniform(5, 100),
            'fill_quality': random.uniform(80, 99),
            'session': random.choice(['asian', 'european', 'american']),
            'market_impact': random.uniform(0.1, 1.0)
        }
        
        # Ensure exit time is after entry time
        if trade['exit_time'] < trade['entry_time']:
            trade['exit_time'] = trade['entry_time'] + datetime.timedelta(hours=random.randint(1, 24))
        
        trades.append(trade)
    
    return trades

def simulate_strategy_performance(persistence_manager: PersistenceManager):
    """Simulate strategy performance data for testing strategy manager"""
    strategies = ['trend_following', 'mean_reversion', 'breakout', 'momentum']
    metrics = ['win_rate', 'sharpe_ratio', 'profit_factor', 'max_drawdown', 'expectancy']
    
    # Simulate performance metrics for each strategy
    for strategy in strategies:
        for metric in metrics:
            # Generate values based on metric
            if metric == 'win_rate':
                value = random.uniform(0.4, 0.7)  # 40-70% win rate
            elif metric == 'sharpe_ratio':
                value = random.uniform(0.5, 2.5)  # 0.5-2.5 Sharpe
            elif metric == 'profit_factor':
                value = random.uniform(0.8, 2.0)  # 0.8-2.0 profit factor
            elif metric == 'max_drawdown':
                value = random.uniform(0.05, 0.2)  # 5-20% drawdown
            elif metric == 'expectancy':
                value = random.uniform(-0.5, 2.0)  # -0.5 to 2.0 R expectancy
            
            # Save metric
            persistence_manager.save_performance_metric(
                metric_type='strategy_metrics',
                metric_name=metric,
                value=value,
                timestamp=datetime.datetime.now(),
                metadata={
                    'strategy_id': strategy
                }
            )
            
            # Create 30 days of history with some variance
            for day in range(1, 31):
                # Add some random variance to simulate changes over time
                variance = random.uniform(-0.15, 0.15) 
                day_value = max(0, value + variance * value)
                
                # Save historical metric
                persistence_manager.save_performance_metric(
                    metric_type='strategy_metrics',
                    metric_name=metric,
                    value=day_value,
                    timestamp=datetime.datetime.now() - datetime.timedelta(days=day),
                    metadata={
                        'strategy_id': strategy
                    }
                )

def simulate_execution_quality(persistence_manager: PersistenceManager):
    """Simulate execution quality metrics for testing execution model"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    metrics = ['slippage', 'latency', 'effective_spread', 'market_impact']
    sessions = ['asian', 'european', 'american']
    
    # Generate execution metrics for each symbol and session
    for symbol in symbols:
        for session in sessions:
            for metric in metrics:
                # Generate values based on metric
                if metric == 'slippage':
                    base_value = random.uniform(0.2, 2.0)  # 0.2-2.0 pips
                elif metric == 'latency':
                    base_value = random.uniform(20, 150)  # 20-150 ms
                elif metric == 'effective_spread':
                    base_value = random.uniform(0.5, 3.0)  # 0.5-3.0 pips
                elif metric == 'market_impact':
                    base_value = random.uniform(0.1, 1.0)  # 0.1-1.0 pips
                
                # Save current metric
                persistence_manager.save_performance_metric(
                    metric_type='execution_quality',
                    metric_name=metric,
                    value=base_value,
                    timestamp=datetime.datetime.now(),
                    metadata={
                        'symbol': symbol,
                        'session': session
                    }
                )
                
                # Create 30 days of history with some variance
                for day in range(1, 31):
                    # Add some random variance
                    variance = random.uniform(-0.2, 0.2)
                    day_value = max(0, base_value + variance * base_value)
                    
                    # Save historical metric
                    persistence_manager.save_performance_metric(
                        metric_type='execution_quality',
                        metric_name=metric,
                        value=day_value,
                        timestamp=datetime.datetime.now() - datetime.timedelta(days=day),
                        metadata={
                            'symbol': symbol,
                            'session': session
                        }
                    )

def run_test_scenario_3():
    """
    Test Scenario 3: State Persistence and Restoration
    
    This scenario tests the state persistence and restoration capabilities,
    ensuring that system state can be saved and loaded properly.
    """
    logger.info("=== Starting Test Scenario 3: State Persistence and Restoration ===")
    
    # Initialize components
    components = initialize_enhanced_components(TEST_CONFIG)
    persistence = components['persistence']
    capital_manager = components['capital_manager']
    strategy_manager = components['strategy_manager']
    
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
    
    # Save initial states
    trading_engine.add_strategy('trend_following', {'param1': 1, 'param2': 2})
    trading_engine.add_strategy('mean_reversion', {'param1': 3, 'param2': 4})
    trading_engine.save_state()
    
    # Modify capital manager state
    capital_manager.update_capital(110000.0)  # 10% gain
    capital_manager.record_drawdown(0.05)  # 5% drawdown at some point
    capital_manager.save_state()
    
    # Add strategies to strategy manager
    strategy_manager.register_strategy('trend_following', {'min_trades': 20})
    strategy_manager.register_strategy('mean_reversion', {'min_trades': 20})
    strategy_manager.register_strategy('breakout', {'min_trades': 20})
    strategy_manager.register_strategy('momentum', {'min_trades': 20})
    strategy_manager.save_state()
    
    # Save all states
    save_states(components)
    
    logger.info("States saved, now we'll create new instances and restore states")
    
    # Shutdown components
    shutdown_components(components)
    
    # Create new component instances
    new_components = initialize_enhanced_components(TEST_CONFIG)
    
    # Load saved states
    load_success = load_saved_states(new_components)
    assert load_success, "Failed to load saved states"
    
    # Verify state restoration
    new_capital_manager = new_components['capital_manager']
    new_strategy_manager = new_components['strategy_manager']
    
    # Verify capital manager state
    assert new_capital_manager.get_current_capital() == 110000.0, "Capital not restored correctly"
    assert new_capital_manager.get_max_drawdown_pct() == 0.05, "Max drawdown not restored correctly"
    
    # Verify strategy manager state
    assert 'trend_following' in new_strategy_manager.get_strategies(), "Strategy not restored correctly"
    assert 'mean_reversion' in new_strategy_manager.get_strategies(), "Strategy not restored correctly"
    
    # Clean up test database
    if persistence.is_connected():
        persistence._client.drop_database(TEST_CONFIG['persistence']['database'])
    
    logger.info("=== Test Scenario 3 completed successfully ===")

def run_test_scenario_4():
    """
    Test Scenario 4: Capital Management
    
    This scenario tests the dynamic capital management capabilities,
    adjusting position sizes based on account performance and volatility.
    """
    logger.info("=== Starting Test Scenario 4: Capital Management ===")
    
    # Initialize components
    components = initialize_enhanced_components(TEST_CONFIG)
    persistence = components['persistence']
    capital_manager = components['capital_manager']
    
    # Generate and save some mock trades with increasing profitability
    mock_trades = generate_mock_trades(50)
    for trade in mock_trades:
        persistence.save_trade(trade)
    
    # Update capital based on trades
    initial_capital = capital_manager.get_current_capital()
    total_profit = sum(trade['profit_loss'] for trade in mock_trades)
    
    # Update capital and simulate a drawdown and recovery
    capital_manager.update_capital(initial_capital + total_profit)
    capital_manager.record_drawdown(0.1)  # 10% drawdown at some point
    
    # Record streak for scaling
    capital_manager.record_win_streak(5)
    
    # Save state
    capital_manager.save_state()
    
    # Calculate position size recommendations
    symbol = "EURUSD"
    entry_price = 1.1000
    stop_loss_pips = 20.0
    
    # Get position sizing recommendations before and after scaling
    initial_size = capital_manager.calculate_position_size(
        symbol, entry_price, stop_loss_pips
    )
    
    # Apply volatility scaling
    capital_manager.set_volatility_factor(1.5)  # Higher volatility
    
    # Get scaled position size
    scaled_size = capital_manager.calculate_position_size(
        symbol, entry_price, stop_loss_pips
    )
    
    # Verify scaling effect
    assert scaled_size < initial_size, "Volatility scaling not reducing position size"
    
    # Log results
    logger.info(f"Initial position size: {initial_size}")
    logger.info(f"Scaled position size: {scaled_size}")
    logger.info(f"Scaling factor: {scaled_size / initial_size:.2f}")
    
    # Clean up test database
    if persistence.is_connected():
        persistence._client.drop_database(TEST_CONFIG['persistence']['database'])
    
    logger.info("=== Test Scenario 4 completed successfully ===")

def run_test_scenario_5():
    """
    Test Scenario 5: Strategy Lifecycle Management
    
    This scenario tests the strategy performance evaluation and lifecycle management,
    retiring underperforming strategies and promoting successful ones.
    """
    logger.info("=== Starting Test Scenario 5: Strategy Lifecycle Management ===")
    
    # Initialize components
    components = initialize_enhanced_components(TEST_CONFIG)
    persistence = components['persistence']
    strategy_manager = components['strategy_manager']
    
    # Generate mock trade data for different strategies with varying performance
    mock_trades = generate_mock_trades(200)
    for trade in mock_trades:
        persistence.save_trade(trade)
    
    # Generate mock strategy performance metrics
    simulate_strategy_performance(persistence)
    
    # Register strategies with the manager
    strategy_manager.register_strategy('trend_following', {'min_trades': 20})
    strategy_manager.register_strategy('mean_reversion', {'min_trades': 20})
    strategy_manager.register_strategy('breakout', {'min_trades': 20})
    strategy_manager.register_strategy('momentum', {'min_trades': 20})
    
    # Run strategy evaluation
    logger.info("Running strategy evaluation")
    evaluation_results = strategy_manager.evaluate_strategies()
    
    # Verify evaluation results
    assert evaluation_results is not None, "Strategy evaluation returned None"
    assert len(evaluation_results) > 0, "Strategy evaluation returned empty results"
    
    # Log results
    for strategy_id, result in evaluation_results.items():
        logger.info(f"Strategy {strategy_id}: Status {result.get('status', 'UNKNOWN')}")
        logger.info(f"  Metrics: {result.get('metrics', {})}")
    
    # Run strategy ranking
    rankings = strategy_manager.rank_strategies()
    assert rankings is not None, "Strategy ranking returned None"
    
    # Log rankings
    for metric, ranked_strategies in rankings.items():
        logger.info(f"Rankings by {metric}:")
        for rank, (strategy_id, value) in enumerate(ranked_strategies):
            logger.info(f"  #{rank+1}: {strategy_id} = {value}")
    
    # Save state
    strategy_manager.save_state()
    
    # Clean up test database
    if persistence.is_connected():
        persistence._client.drop_database(TEST_CONFIG['persistence']['database'])
    
    logger.info("=== Test Scenario 5 completed successfully ===")

def run_test_scenario_6():
    """
    Test Scenario 6: Execution Quality Modeling
    
    This scenario tests the execution quality modeling capabilities,
    simulating slippage, latency, and market impact.
    """
    logger.info("=== Starting Test Scenario 6: Execution Quality Modeling ===")
    
    # Initialize components
    components = initialize_enhanced_components(TEST_CONFIG)
    persistence = components['persistence']
    execution_model = components['execution_model']
    
    # Generate mock execution quality data
    simulate_execution_quality(persistence)
    
    # Test execution quality modeling for different symbols and scenarios
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    order_sizes = [0.1, 1.0, 5.0, 10.0]  # Lot sizes
    sessions = ['asian', 'european', 'american']
    
    for symbol in symbols:
        logger.info(f"Testing execution model for {symbol}")
        
        for session in sessions:
            logger.info(f"  Session: {session}")
            
            # Set current session
            execution_model.set_current_session(session)
            
            for order_size in order_sizes:
                # Calculate execution parameters
                slippage, spread, impact, latency = execution_model.calculate_execution_parameters(
                    symbol=symbol,
                    order_size=order_size,
                    volatility=0.0005  # 5 pips volatility
                )
                
                logger.info(f"    Order size: {order_size} lots")
                logger.info(f"      Slippage: {slippage:.2f} pips")
                logger.info(f"      Spread: {spread:.2f} pips")
                logger.info(f"      Market Impact: {impact:.2f} pips")
                logger.info(f"      Latency: {latency:.2f} ms")
                
                # Verify that larger orders have higher impact
                if order_size > 1.0:
                    small_order_impact = execution_model.calculate_execution_parameters(
                        symbol=symbol,
                        order_size=0.1,
                        volatility=0.0005
                    )[2]  # Impact is the third returned value
                    
                    assert impact > small_order_impact, "Larger orders should have higher market impact"
    
    # Clean up test database
    if persistence.is_connected():
        persistence._client.drop_database(TEST_CONFIG['persistence']['database'])
    
    logger.info("=== Test Scenario 6 completed successfully ===")

if __name__ == "__main__":
    # Run additional test scenarios
    run_test_scenario_3()
    run_test_scenario_4()
    run_test_scenario_5()
    run_test_scenario_6()
