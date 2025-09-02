#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Example - Demonstrates how to use the enhanced Event-Driven
Architecture with Message Queues and Channels.

This example shows a practical integration of:
1. Message Queues for decoupling components
2. Channels for real-time data streaming
3. Parallel processing of trading signals
"""

import logging
import threading
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import random
import uuid
from pprint import pprint

# Import our event system components
from trading_bot.event_system.message_queue import (
    MessageQueue, Message, QueueType, QueueBackend
)
from trading_bot.event_system.channel_system import (
    ChannelManager, Channel
)
from trading_bot.event_system.event_types import (
    Event, EventType, MarketDataEvent, SignalEvent, OrderEvent
)
from trading_bot.strategies.base_strategy import SignalType
from trading_bot.trading_modes.base_trading_mode import Order, OrderType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegrationExample")

class TradingSystemExample:
    """
    Example trading system using the enhanced event-driven architecture.
    
    This example demonstrates:
    1. Decoupling of market data, signal generation, trading logic, and execution
    2. Parallel processing of signals from multiple strategies
    3. Real-time responsive messaging between components
    4. Scalable architecture that can be distributed
    """
    
    def __init__(self):
        """Initialize the trading system example"""
        # Create channel manager
        self.channel_manager = ChannelManager()
        
        # Create key message queues
        self.market_data_queue = self.channel_manager.create_queue_channel(
            name="market_data",
            queue_type=QueueType.TOPIC,
            worker_threads=2
        )
        
        self.signal_queue = self.channel_manager.create_queue_channel(
            name="trading_signals",
            queue_type=QueueType.PRIORITY,
            worker_threads=2
        )
        
        self.order_queue = self.channel_manager.create_queue_channel(
            name="orders",
            queue_type=QueueType.DIRECT,
            worker_threads=1  # Orders processed sequentially
        )
        
        # Create real-time channels for analytics and monitoring
        self.price_channel = self.channel_manager.create_channel(
            name="price_updates",
            buffer_size=1000,
            allow_replay=True
        )
        
        self.portfolio_channel = self.channel_manager.create_channel(
            name="portfolio_updates",
            buffer_size=100,
            allow_replay=True
        )
        
        # Thread for running example
        self.running = False
        self.example_thread = None
        
        # Simulated state
        self.portfolio = {
            "cash": 100000.0,
            "positions": {},
            "equity": 100000.0
        }
        
        # Subscribe to relevant queues and channels
        self._setup_subscriptions()
        
        logger.info("Trading system example initialized")
    
    def _setup_subscriptions(self):
        """Set up queue and channel subscriptions"""
        # Strategy component subscribes to market data
        self.market_data_queue.subscribe(
            self._strategy_process_market_data,
            subscriber_id="strategy_processor"
        )
        
        # Trading mode subscribes to signals
        self.signal_queue.subscribe(
            self._trading_mode_process_signal,
            subscriber_id="trading_mode"
        )
        
        # Execution component subscribes to orders
        self.order_queue.subscribe(
            self._execution_process_order,
            subscriber_id="executor"
        )
        
        # Analytics subscribes to price updates
        self.price_channel.subscribe(
            self._analytics_process_price,
            subscriber_id="analytics"
        )
        
        # UI subscribes to portfolio updates
        self.portfolio_channel.subscribe(
            self._ui_process_portfolio,
            subscriber_id="dashboard"
        )
    
    def start_example(self):
        """Start the example"""
        if self.running:
            logger.warning("Example already running")
            return
            
        self.running = True
        
        # Start example thread
        self.example_thread = threading.Thread(
            target=self._run_example,
            name="ExampleRunner",
            daemon=True
        )
        self.example_thread.start()
        
        logger.info("Trading system example started")
    
    def stop_example(self):
        """Stop the example"""
        if not self.running:
            logger.warning("Example not running")
            return
            
        self.running = False
        
        # Wait for thread to finish
        if self.example_thread:
            self.example_thread.join(timeout=5.0)
            
        # Shutdown channel manager
        self.channel_manager.shutdown()
        
        logger.info("Trading system example stopped")
    
    def _run_example(self):
        """Run the example simulation"""
        logger.info("Starting example simulation")
        
        # Symbols to simulate
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # Generate market data
        iteration = 0
        while self.running and iteration < 100:
            # Publish market data for each symbol
            for symbol in symbols:
                # Generate random price data
                price = 100 + random.random() * 10
                volume = random.randint(1000, 10000)
                
                # Create market data
                market_data = {
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "timestamp": datetime.now().isoformat(),
                    "bid": price - 0.1,
                    "ask": price + 0.1
                }
                
                # Publish to market data queue (strategies will consume)
                self.market_data_queue.publish(
                    Message(
                        payload=market_data,
                        source="data_feed",
                        priority=1  # Regular priority
                    )
                )
                
                # Also publish to price channel for real-time updates
                self.price_channel.publish(
                    data=market_data,
                    metadata={"source": "data_feed", "type": "tick"}
                )
            
            # Sleep between updates
            time.sleep(0.5)
            iteration += 1
            
            # Occasionally publish portfolio updates
            if iteration % 5 == 0:
                # Update portfolio value
                total_value = self.portfolio["cash"]
                for symbol, position in self.portfolio["positions"].items():
                    # Use last price (would be more accurate in real system)
                    price = 100 + random.random() * 10
                    total_value += position["quantity"] * price
                
                self.portfolio["equity"] = total_value
                
                # Publish update
                self.portfolio_channel.publish(
                    data=self.portfolio,
                    metadata={"source": "portfolio_manager", "type": "update"}
                )
        
        logger.info("Example simulation finished")
    
    # Simulated component callbacks
    
    def _strategy_process_market_data(self, message: Message):
        """Simulated strategy processing market data to generate signals"""
        # Extract market data
        market_data = message.payload
        symbol = market_data["symbol"]
        price = market_data["price"]
        
        # Simple random signal generation for demo
        if random.random() < 0.1:  # 10% chance of signal
            # Generate random signal
            signal_type = random.choice([SignalType.LONG, SignalType.SHORT])
            strength = random.random()
            
            # Create signal
            signal = {
                "symbol": symbol,
                "price": price,
                "signal_type": signal_type.name,
                "direction": signal_type.value,
                "strength": strength,
                "strategy": f"Strategy_{random.randint(1, 3)}",  # Random strategy
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to signal queue (trading mode will consume)
            self.signal_queue.publish(
                Message(
                    payload=signal,
                    source="strategy",
                    priority=2  # Higher priority than market data
                )
            )
            
            logger.info(f"Strategy generated {signal_type.name} signal for {symbol} at {price:.2f}")
    
    def _trading_mode_process_signal(self, message: Message):
        """Simulated trading mode processing signals to generate orders"""
        # Extract signal
        signal = message.payload
        symbol = signal["symbol"]
        price = signal["price"]
        direction = signal["direction"]
        
        # Check if we want to act on this signal (random for demo)
        if random.random() < 0.8:  # 80% chance of acting on signal
            # Calculate order size (simple for demo)
            size = random.randint(1, 10) * 10  # 10-100 shares
            
            # Create order
            order = {
                "order_id": str(uuid.uuid4()),
                "symbol": symbol,
                "order_type": "market",
                "side": direction,
                "quantity": size,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "signal_id": message.message_id,
                "source": "trading_mode"
            }
            
            # Publish to order queue (execution will consume)
            self.order_queue.publish(
                Message(
                    payload=order,
                    source="trading_mode",
                    priority=3  # Highest priority
                )
            )
            
            side_str = "BUY" if direction == 1 else "SELL"
            logger.info(f"Trading mode created {side_str} order for {size} {symbol} at {price:.2f}")
    
    def _execution_process_order(self, message: Message):
        """Simulated execution processing orders"""
        # Extract order
        order = message.payload
        symbol = order["symbol"]
        side = order["side"]
        quantity = order["quantity"]
        price = order["price"]
        
        # Simulate order execution (random for demo)
        filled_price = price * (1 + (random.random() - 0.5) * 0.01)  # +/- 0.5% slippage
        
        # Update portfolio
        if side == 1:  # BUY
            # Deduct cash
            cost = filled_price * quantity
            self.portfolio["cash"] -= cost
            
            # Add to positions
            if symbol not in self.portfolio["positions"]:
                self.portfolio["positions"][symbol] = {
                    "quantity": 0,
                    "avg_price": 0
                }
                
            position = self.portfolio["positions"][symbol]
            old_qty = position["quantity"]
            old_price = position["avg_price"]
            
            # Calculate new average price
            total_qty = old_qty + quantity
            position["avg_price"] = (old_qty * old_price + quantity * filled_price) / total_qty
            position["quantity"] = total_qty
            
        else:  # SELL
            # Add to cash
            proceeds = filled_price * quantity
            self.portfolio["cash"] += proceeds
            
            # Reduce position
            if symbol in self.portfolio["positions"]:
                position = self.portfolio["positions"][symbol]
                position["quantity"] -= quantity
                
                # Remove if zero
                if position["quantity"] <= 0:
                    del self.portfolio["positions"][symbol]
        
        # Create fill notification
        fill = {
            "order_id": order["order_id"],
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "requested_price": price,
            "filled_price": filled_price,
            "timestamp": datetime.now().isoformat(),
            "commission": filled_price * quantity * 0.001  # 0.1% commission for demo
        }
        
        side_str = "BUY" if side == 1 else "SELL"
        logger.info(f"Executed {side_str} order for {quantity} {symbol} at {filled_price:.2f}")
        
        # Publish portfolio update
        self.portfolio_channel.publish(
            data=self.portfolio,
            metadata={"source": "execution", "type": "fill", "details": fill}
        )
    
    def _analytics_process_price(self, data, metadata):
        """Simulated analytics processing price updates"""
        # In a real system, this would update charts, calculate metrics, etc.
        # Just log for demo
        symbol = data["symbol"]
        price = data["price"]
        
        # Only log occasionally to reduce noise
        if random.random() < 0.1:  # 10% chance
            logger.debug(f"Analytics processed price update for {symbol}: {price:.2f}")
    
    def _ui_process_portfolio(self, data, metadata):
        """Simulated UI processing portfolio updates"""
        # In a real system, this would update UI components
        # Just log for demo
        equity = data["equity"]
        positions = len(data["positions"])
        
        logger.debug(f"UI updated with portfolio value: ${equity:.2f}, positions: {positions}")
    
    def print_stats(self):
        """Print system statistics"""
        stats = self.channel_manager.get_stats()
        
        print("\n----- Trading System Statistics -----")
        print(f"Total channels: {stats['channels']}")
        print(f"Total queue channels: {stats['queue_channels']}")
        print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        
        print("\n----- Queue Statistics -----")
        for name, queue_stats in stats['queue_stats'].items():
            print(f"Queue '{name}':")
            print(f"  Messages in: {queue_stats['messages_in']}")
            print(f"  Messages out: {queue_stats['messages_out']}")
            print(f"  Queue size: {queue_stats.get('queue_size', 'N/A')}")
            print(f"  Subscribers: {queue_stats['subscribers']}")
            
        print("\n----- Channel Statistics -----")
        for name, channel_stats in stats['channel_stats'].items():
            print(f"Channel '{name}':")
            print(f"  Messages published: {channel_stats['messages_published']}")
            print(f"  Messages delivered: {channel_stats['messages_delivered']}")
            print(f"  Subscribers: {channel_stats['subscriber_count']}")
            
        print("\n----- Portfolio Status -----")
        print(f"Cash: ${self.portfolio['cash']:.2f}")
        print(f"Equity: ${self.portfolio['equity']:.2f}")
        print(f"Positions: {len(self.portfolio['positions'])}")
        for symbol, position in self.portfolio['positions'].items():
            print(f"  {symbol}: {position['quantity']} shares @ ${position['avg_price']:.2f}")
        
        print("\n")


# Example usage
if __name__ == "__main__":
    # Create trading system example
    trading_system = TradingSystemExample()
    
    # Start the example
    trading_system.start_example()
    
    try:
        # Run for a while
        for i in range(10):
            time.sleep(1)
            print(f"Running... {i+1}/10")
        
        # Print stats
        trading_system.print_stats()
        
    finally:
        # Stop the example
        trading_system.stop_example()
        print("Example completed")
