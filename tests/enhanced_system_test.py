#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced System Test for BensBot
Demonstrates the integration of the new reliability and efficiency enhancements:
1. Persistence Layer (MongoDB)
2. Watchdog & Fault Tolerance
3. Dynamic Capital Scaling
4. Strategy Retirement & Promotion
5. Execution Quality Modeling

This script simulates a trading cycle with multiple strategies and
shows how the enhancements improve resilience and performance.
"""

import logging
import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced components
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from trading_bot.data.persistence import PersistenceManager
    from trading_bot.core.watchdog import ServiceWatchdog, RecoveryStrategy
    from trading_bot.risk.capital_manager import CapitalManager
    from trading_bot.core.strategy_manager import StrategyPerformanceManager
    from trading_bot.execution.execution_model import ExecutionQualityModel
except ImportError as e:
    logger.error(f"Failed to import enhanced components: {e}")
    logger.info("This script tests the newly added reliability and efficiency enhancements.")
    logger.info("If imports fail, ensure you've implemented all the new modules.")
    sys.exit(1)

# Mock classes and functions for testing purposes
class MockStrategy:
    """Mock strategy for testing"""
    
    def __init__(self, strategy_id, win_rate=0.5, risk_params=None):
        self.strategy_id = strategy_id
        self.win_rate = win_rate
        self.active = True
        self.risk_params = risk_params or {'position_size_factor': 1.0}
        self.parameters = {}
        
    def generate_signal(self, symbol):
        """Generate a mock trading signal"""
        if not self.active:
            return None
            
        # Randomize if we generate a signal (70% chance)
        if random.random() > 0.3:
            signal = {
                'strategy_id': self.strategy_id,
                'symbol': symbol,
                'direction': 'BUY' if random.random() > 0.5 else 'SELL',
                'entry_price': 100.0 + random.uniform(-5, 5),
                'stop_loss': 95.0 + random.uniform(-3, 3),
                'target_price': 110.0 + random.uniform(-3, 3),
                'timestamp': datetime.now()
            }
            
            # Calculate prices
            if signal['direction'] == 'BUY':
                signal['stop_loss'] = signal['entry_price'] * 0.95
                signal['target_price'] = signal['entry_price'] * 1.1
            else:
                signal['stop_loss'] = signal['entry_price'] * 1.05
                signal['target_price'] = signal['entry_price'] * 0.9
                
            return signal
            
        return None
        
    def deactivate(self):
        """Deactivate the strategy"""
        self.active = False
        logger.info(f"Strategy {self.strategy_id} deactivated")
        
    def activate(self):
        """Activate the strategy"""
        self.active = True
        logger.info(f"Strategy {self.strategy_id} activated")
        
    def get_performance_metrics(self):
        """Get mock performance metrics"""
        return {
            'win_rate': self.win_rate,
            'profit_factor': 1.2 if self.win_rate > 0.5 else 0.9,
            'sharpe_ratio': 1.0 if self.win_rate > 0.5 else 0.5,
            'expectancy': 0.5 if self.win_rate > 0.5 else -0.1,
            'drawdown': 0.1 if self.win_rate > 0.5 else 0.2,
            'trades': 50,
            'total_profit': 1000 if self.win_rate > 0.5 else -200
        }
        
    def auto_optimize(self):
        """Mock optimization"""
        logger.info(f"Running optimization for strategy {self.strategy_id}")
        # Simulate parameter optimization
        self.parameters = {
            'lookback': random.randint(10, 50),
            'threshold': random.uniform(0.5, 2.0)
        }
        return True

class MockDataFeed:
    """Mock data feed for testing"""
    
    def __init__(self):
        self.running = False
        self.last_update = None
        self.subscribers = []
        self.thread = None
        self.error_rate = 0.05  # 5% chance of error
        
    def start(self):
        """Start the data feed"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Mock data feed started")
        
    def stop(self):
        """Stop the data feed"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Mock data feed stopped")
        
    def _run(self):
        """Main data feed loop"""
        while self.running:
            try:
                # Simulate data update
                self.last_update = datetime.now()
                
                # Randomly generate error
                if random.random() < self.error_rate:
                    raise Exception("Simulated data feed error")
                    
                # Publish to subscribers
                data = {
                    'timestamp': self.last_update,
                    'symbol': 'EURUSD',
                    'bid': 1.1 + random.uniform(-0.01, 0.01),
                    'ask': 1.1 + random.uniform(0, 0.02)
                }
                
                for subscriber in self.subscribers:
                    subscriber(data)
                    
                # Sleep for a bit
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in data feed: {e}")
                time.sleep(5.0)  # Longer sleep on error
                
    def subscribe(self, callback):
        """Subscribe to data updates"""
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            
    def unsubscribe(self, callback):
        """Unsubscribe from data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            
    def is_connected(self):
        """Check if data feed is connected"""
        return self.running and self.last_update is not None
        
    def last_update_within(self, seconds=60):
        """Check if the last update was within the specified seconds"""
        if self.last_update is None:
            return False
            
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < seconds
        
    def restart_subscription(self):
        """Restart the data feed subscription"""
        if self.running:
            self.stop()
            
        # Small delay
        time.sleep(1.0)
        
        # Restart
        self.start()
        return True

class MockBroker:
    """Mock broker for testing"""
    
    def __init__(self):
        self.orders = {}
        self.positions = {}
        self.connected = False
        self.next_order_id = 1000
        self.error_rate = 0.1  # 10% chance of error
        
    def connect(self):
        """Connect to broker"""
        self.connected = True
        logger.info("Connected to mock broker")
        return True
        
    def disconnect(self):
        """Disconnect from broker"""
        self.connected = False
        logger.info("Disconnected from mock broker")
        return True
        
    def is_connected(self):
        """Check if connected to broker"""
        return self.connected
        
    def reconnect(self):
        """Reconnect to broker"""
        self.disconnect()
        time.sleep(1.0)  # Simulate reconnection delay
        return self.connect()
        
    def place_order(self, order_data):
        """Place an order with the broker"""
        # Simulate random errors
        if random.random() < self.error_rate:
            logger.error("Simulated broker order error")
            return None
            
        order_id = self.next_order_id
        self.next_order_id += 1
        
        # Create order record
        order = {
            'order_id': order_id,
            'symbol': order_data.get('symbol', 'UNKNOWN'),
            'direction': order_data.get('direction', 'BUY'),
            'quantity': order_data.get('quantity', 1),
            'order_type': order_data.get('order_type', 'MARKET'),
            'price': order_data.get('price', 100.0),
            'status': 'PENDING',
            'timestamp': datetime.now()
        }
        
        self.orders[order_id] = order
        logger.info(f"Placed order: {order_id} - {order['direction']} {order['quantity']} {order['symbol']}")
        
        # Simulate order execution (immediately for simplicity)
        threading.Timer(0.5, self._execute_order, args=[order_id]).start()
        
        return order_id
        
    def _execute_order(self, order_id):
        """Simulate order execution"""
        if order_id not in self.orders:
            return
            
        order = self.orders[order_id]
        
        # Simulate fill with slippage
        slippage = random.uniform(-0.5, 1.0)
        
        if order['direction'] == 'BUY':
            fill_price = order['price'] + abs(slippage)
        else:
            fill_price = order['price'] - abs(slippage)
            
        # Update order
        order['status'] = 'FILLED'
        order['fill_price'] = fill_price
        order['fill_time'] = datetime.now()
        
        # Create position
        if order['symbol'] not in self.positions:
            self.positions[order['symbol']] = {
                'symbol': order['symbol'],
                'quantity': 0,
                'average_price': 0,
                'unrealized_pnl': 0
            }
            
        position = self.positions[order['symbol']]
        
        # Update position
        if order['direction'] == 'BUY':
            new_quantity = position['quantity'] + order['quantity']
            if new_quantity == 0:
                position['average_price'] = 0
            else:
                position['average_price'] = (
                    (position['quantity'] * position['average_price']) + 
                    (order['quantity'] * fill_price)
                ) / new_quantity
            position['quantity'] = new_quantity
        else:  # SELL
            position['quantity'] -= order['quantity']
            
        logger.info(f"Order {order_id} filled at {fill_price}")
        
    def get_order_status(self, order_id):
        """Get order status"""
        if order_id not in self.orders:
            return None
            
        return self.orders[order_id]['status']
        
    def get_positions(self):
        """Get current positions"""
        return self.positions
        
    def get_account_balance(self):
        """Get account balance"""
        return 100000 + random.uniform(-5000, 5000)

class TradingEngine:
    """Mock trading engine for testing"""
    
    def __init__(self, persistence_manager=None):
        self.strategies = {}
        self.data_feed = None
        self.broker = None
        self.running = False
        self.thread = None
        self.persistence_manager = persistence_manager
        self.orders = {}
        self.last_health_check = datetime.now()
        
    def add_strategy(self, strategy_id, strategy):
        """Add a strategy to the engine"""
        self.strategies[strategy_id] = strategy
        
    def set_data_feed(self, data_feed):
        """Set the data feed"""
        self.data_feed = data_feed
        
    def set_broker(self, broker):
        """Set the broker"""
        self.broker = broker
        
    def start(self):
        """Start the trading engine"""
        if self.running:
            return
            
        if not self.data_feed or not self.broker:
            logger.error("Cannot start trading engine without data feed and broker")
            return False
            
        # Connect data feed and broker
        if not self.data_feed.is_connected():
            self.data_feed.start()
            
        if not self.broker.is_connected():
            self.broker.connect()
            
        # Subscribe to data feed
        self.data_feed.subscribe(self._on_data)
        
        # Start engine thread
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Trading engine started")
        return True
        
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
            
        # Unsubscribe from data feed
        if self.data_feed:
            self.data_feed.unsubscribe(self._on_data)
            
        logger.info("Trading engine stopped")
        
    def _run(self):
        """Main engine loop"""
        while self.running:
            try:
                # Perform background tasks
                self._process_orders()
                
                # Update health check timestamp
                self.last_health_check = datetime.now()
                
                # Sleep briefly
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in trading engine: {e}")
                time.sleep(5.0)
                
    def _on_data(self, data):
        """Handle incoming data"""
        try:
            # Process data
            symbol = data.get('symbol')
            
            # Generate signals from strategies
            for strategy_id, strategy in self.strategies.items():
                signal = strategy.generate_signal(symbol)
                
                if signal:
                    self._process_signal(signal)
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            
    def _process_signal(self, signal):
        """Process a trading signal"""
        # Create order from signal
        order_data = {
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'quantity': 1,  # Simple quantity for testing
            'order_type': 'MARKET',
            'price': signal['entry_price'],
            'strategy_id': signal['strategy_id'],
            'timestamp': datetime.now()
        }
        
        # Place order with broker
        order_id = self.broker.place_order(order_data)
        
        if order_id:
            # Store order
            self.orders[order_id] = {
                'order_data': order_data,
                'signal': signal,
                'status': 'PENDING',
                'created_at': datetime.now()
            }
            
            logger.info(f"Signal from {signal['strategy_id']} converted to order {order_id}")
            
            # Store to persistence if available
            if self.persistence_manager:
                order_record = {
                    'order_id': order_id,
                    **order_data,
                    'status': 'PENDING'
                }
                self.persistence_manager.save_trade(order_record)
                
    def _process_orders(self):
        """Process and update orders"""
        for order_id, order in list(self.orders.items()):
            # Skip non-pending orders
            if order['status'] != 'PENDING':
                continue
                
            # Check status with broker
            status = self.broker.get_order_status(order_id)
            
            if status == 'FILLED':
                # Update order status
                order['status'] = 'FILLED'
                order['filled_at'] = datetime.now()
                
                # Log
                logger.info(f"Order {order_id} filled")
                
                # Update persistence if available
                if self.persistence_manager:
                    self.persistence_manager.update_trade_status(
                        str(order_id), 
                        'FILLED',
                        {'filled_at': datetime.now()}
                    )
                    
                # Simulate trade outcome
                self._simulate_trade_outcome(order_id, order)
                
    def _simulate_trade_outcome(self, order_id, order):
        """Simulate a trade outcome for testing"""
        signal = order['signal']
        strategy_id = signal['strategy_id']
        
        # Get strategy
        if strategy_id not in self.strategies:
            return
            
        strategy = self.strategies[strategy_id]
        
        # Simulate win based on strategy win rate
        is_win = random.random() < strategy.win_rate
        
        # Calculate P&L
        if is_win:
            if signal['direction'] == 'BUY':
                profit_loss = signal['target_price'] - signal['entry_price']
            else:
                profit_loss = signal['entry_price'] - signal['target_price']
        else:
            if signal['direction'] == 'BUY':
                profit_loss = signal['stop_loss'] - signal['entry_price']
            else:
                profit_loss = signal['entry_price'] - signal['stop_loss']
                
        # Create trade result
        trade_result = {
            'order_id': order_id,
            'strategy_id': strategy_id,
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'exit_price': signal['target_price'] if is_win else signal['stop_loss'],
            'profit_loss': profit_loss,
            'win': is_win,
            'timestamp': datetime.now(),
            'status': 'CLOSED'
        }
        
        # Log
        logger.info(f"Trade {order_id} {'won' if is_win else 'lost'} with P&L: {profit_loss:.2f}")
        
        # Store to persistence if available
        if self.persistence_manager:
            self.persistence_manager.save_trade(trade_result)
            
        return trade_result
        
    def check_health(self):
        """Check if trading engine is healthy"""
        if not self.running:
            return False
            
        # Check if health check is recent
        max_time = 60  # 60 seconds
        elapsed = (datetime.now() - self.last_health_check).total_seconds()
        
        return elapsed < max_time
        
    def restore_from_saved_state(self):
        """Restore from saved state"""
        if not self.persistence_manager:
            logger.warning("No persistence manager available for state restoration")
            return False
            
        # Restart the engine
        was_running = self.running
        
        if was_running:
            self.stop()
            
        # Small delay
        time.sleep(2.0)
        
        # Restart
        result = self.start()
        
        logger.info(f"Trading engine restored: {result}")
        return result
    
def run_enhanced_system_test():
    """
    Run a comprehensive test of the enhanced system components
    """
    logger.info("Starting Enhanced System Test")
    logger.info("This test simulates a trading system with the new reliability and efficiency enhancements")
    
    # Initialize components
    try:
        # Create persistence manager (with in-memory mode for testing)
        logger.info("Initializing Persistence Layer")
        persistence = PersistenceManager(
            connection_string="mongodb://localhost:27017/",
            database="bensbot_test",
            auto_connect=False  # Don't actually connect for this test
        )
        
        # Mock successful connection
        persistence.connected = True
        persistence.collections = {
            'trades': None,
            'strategy_states': None,
            'performance': None,
            'logs': None
        }
        
        # Initialize capital manager
        logger.info("Initializing Capital Manager")
        capital_manager = CapitalManager(
            initial_capital=100000.0,
            persistence_manager=persistence
        )
        
        # Initialize strategy manager
        logger.info("Initializing Strategy Performance Manager")
        strategy_manager = StrategyPerformanceManager(
            persistence_manager=persistence
        )
        
        # Initialize execution quality model
        logger.info("Initializing Execution Quality Model")
        execution_model = ExecutionQualityModel()
        
        # Create mock components
        logger.info("Creating Mock Trading Components")
        broker = MockBroker()
        data_feed = MockDataFeed()
        trading_engine = TradingEngine(persistence_manager=persistence)
        
        # Connect components
        trading_engine.set_broker(broker)
        trading_engine.set_data_feed(data_feed)
        
        # Create strategies with varying performance
        strategies = {
            'trend_following': MockStrategy('trend_following', win_rate=0.6),
            'mean_reversion': MockStrategy('mean_reversion', win_rate=0.55),
            'breakout': MockStrategy('breakout', win_rate=0.5),
            'momentum': MockStrategy('momentum', win_rate=0.45),
            'arbitrage': MockStrategy('arbitrage', win_rate=0.4)
        }
        
        # Add strategies to engine
        for strategy_id, strategy in strategies.items():
            trading_engine.add_strategy(strategy_id, strategy)
            strategy_manager.register_strategy(strategy_id, strategy)
            
        # Initialize watchdog
        logger.info("Initializing Service Watchdog")
        watchdog = ServiceWatchdog(persistence_manager=persistence)
        
        # Register services with watchdog
        watchdog.register_service(
            name="data_feed",
            health_check=lambda: data_feed.last_update_within(seconds=10),
            recovery_action=lambda: data_feed.restart_subscription(),
            recovery_strategy=RecoveryStrategy.RESTART,
            max_failures=3
        )
        
        watchdog.register_service(
            name="broker_connection",
            health_check=lambda: broker.is_connected(),
            recovery_action=lambda: broker.reconnect(),
            recovery_strategy=RecoveryStrategy.RECONNECT,
            max_failures=2
        )
        
        watchdog.register_service(
            name="trading_engine",
            health_check=lambda: trading_engine.check_health(),
            recovery_action=lambda: trading_engine.restore_from_saved_state(),
            recovery_strategy=RecoveryStrategy.RELOAD_STATE,
            max_failures=2,
            dependencies=["data_feed", "broker_connection"]
        )
        
        # Start components
        logger.info("Starting Trading System Components")
        broker.connect()
        data_feed.start()
        trading_engine.start()
        watchdog.start()
        
        # Run simulation
        logger.info("Running Trading Simulation")
        
        # Log initial state
        logger.info(f"Initial capital: ${capital_manager.current_capital:.2f}")
        logger.info(f"Active strategies: {len(strategy_manager.get_active_strategies(strategies))}")
        
        # Simulate trading activity
        for i in range(1, 61):
            logger.info(f"Simulation step {i}/60")
            
            # Every 10 steps, evaluate strategies
            if i % 10 == 0:
                logger.info("Evaluating strategy performance")
                results = strategy_manager.evaluate_strategies(strategies)
                strategy_manager.apply_recommendations(strategies, results)
                
                # Log evaluation results
                for strategy_id, result in results.items():
                    logger.info(f"Strategy {strategy_id}: Action={result['action']}, " +
                               f"Status={strategy_manager.get_strategy_status(strategy_id)}")
                
            # Every 5 steps, update capital
            if i % 5 == 0:
                # Simulate account fluctuation
                new_capital = capital_manager.current_capital * (1 + random.uniform(-0.02, 0.03))
                update = capital_manager.update_capital(new_capital)
                logger.info(f"Capital updated to ${new_capital:.2f}")
                
            # Every 20 steps, simulate component failure and recovery
            if i == 20:
                logger.info("Simulating data feed failure")
                # Force data feed to disconnect
                data_feed.error_rate = 1.0  # 100% error rate
                
            if i == 40:
                logger.info("Simulating broker connection failure")
                # Force broker disconnect
                broker.connected = False
                
            # Process several trades per step
            for _ in range(3):
                # Randomly select a strategy
                strategy_id = random.choice(list(strategies.keys()))
                strategy = strategies[strategy_id]
                
                # Skip if not active
                if not strategy.active:
                    continue
                    
                # Generate fake signal
                symbol = random.choice(['EURUSD', 'GBPUSD', 'USDJPY'])
                signal = strategy.generate_signal(symbol)
                
                if signal:
                    # Process signal
                    trading_engine._process_signal(signal)
                    
                    # Simulate trade execution and outcome immediately
                    order_id = max(trading_engine.orders.keys()) if trading_engine.orders else 1000
                    order = trading_engine.orders.get(order_id)
                    
                    if order:
                        # Simulate execution
                        order['status'] = 'FILLED'
                        order['filled_at'] = datetime.now()
                        
                        # Model execution quality
                        execution_result = execution_model.model_order_execution(
                            symbol=signal['symbol'],
                            order_size_usd=10000.0,
                            price=signal['entry_price'],
                            is_buy=signal['direction'] == 'BUY',
                            volatility_percentile=random.random()
                        )
                        
                        # Log execution quality
                        logger.info(f"Order execution modeled: " +
                                  f"Slippage={execution_result['slippage_pips']:.2f} pips, " +
                                  f"Latency={execution_result['execution_time_ms']:.2f}ms")
                        
                        # Simulate outcome
                        trade_result = trading_engine._simulate_trade_outcome(order_id, order)
                        
                        # Record in capital manager
                        if trade_result:
                            capital_manager.record_trade(
                                strategy_id=strategy_id,
                                trade_data={
                                    'symbol': trade_result['symbol'],
                                    'profit_loss': trade_result['profit_loss'],
                                    'win': trade_result['win'],
                                    'timestamp': trade_result['timestamp']
                                }
                            )
                            
                            # Log position sizing factor
                            strategy_factor = capital_manager.strategy_scaling_factors.get(strategy_id, 1.0)
                            logger.info(f"Strategy {strategy_id} scaling factor: {strategy_factor:.2f}x")
                            
                            # Calculate position size for next trade
                            position_size = capital_manager.calculate_position_size(
                                strategy_id=strategy_id,
                                symbol=symbol,
                                entry_price=100.0,
                                stop_loss_price=95.0,
                                target_price=110.0
                            )
                            
                            logger.info(f"Next position size for {strategy_id}: " +
                                      f"{position_size['position_size']:.2f} units, " +
                                      f"Risk: ${position_size['risk_amount']:.2f}")
            
            # Short delay between steps
            time.sleep(0.5)
            
        # Simulation complete
        logger.info("Simulation Complete")
        
        # Log final state
        logger.info(f"Final capital: ${capital_manager.current_capital:.2f}")
        logger.info(f"High water mark: ${capital_manager.high_water_mark:.2f}")
        logger.info(f"Max drawdown: {capital_manager.max_drawdown*100:.2f}%")
        
        # Get strategy performance
        logger.info("Final Strategy Performance:")
        for strategy_id, strategy in strategies.items():
            perf = capital_manager.get_strategy_performance(strategy_id)
            status = strategy_manager.get_strategy_status(strategy_id)
            
            logger.info(f"Strategy {strategy_id}: Status={status}, " +
                      f"Scaling={capital_manager.strategy_scaling_factors.get(strategy_id, 1.0):.2f}x")
            
        # Get system health
        health = watchdog.get_system_health_summary()
        logger.info(f"System Health: {health['overall_status']}")
        
        # Stop components
        logger.info("Stopping Trading System Components")
        watchdog.stop()
        trading_engine.stop()
        data_feed.stop()
        broker.disconnect()
        
        logger.info("Enhanced System Test Completed Successfully")
        return True
    except Exception as e:
        logger.error(f"Enhanced System Test failed: {e}")
        return False

if __name__ == "__main__":
    run_enhanced_system_test()
