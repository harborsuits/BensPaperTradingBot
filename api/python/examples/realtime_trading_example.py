#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing how to use the RealTimeDataConnector and StrategyRotator together
to implement event-driven trading strategies.
"""

import os
import sys
import json
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our modules
from trading_bot.strategy.strategy_rotator import StrategyRotator
from trading_bot.realtime.real_time_data_connector import RealTimeDataConnector
from trading_bot.common.market_types import MarketRegime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("RealtimeTradingExample")

# Mock trading functions
class MockTrader:
    """Simple mock trader for demonstration purposes."""
    
    def __init__(self):
        """Initialize the mock trader."""
        self.positions = {}
        self.orders = []
        self.cash = 100000.0
        self.last_signal = 0.0
        self.position_value = 0.0
        
    async def handle_signal(self, combined_signal: float, signals: Dict[str, float]) -> None:
        """
        Handle a trading signal from the strategy rotator.
        
        Args:
            combined_signal: Combined trading signal (-1.0 to 1.0)
            signals: Individual strategy signals
        """
        # Only trade if signal changed significantly
        if abs(combined_signal - self.last_signal) < 0.1:
            return
            
        # Update last signal
        self.last_signal = combined_signal
        
        # Determine target position based on signal
        # Signal range: -1.0 (fully short) to 1.0 (fully long)
        target_position_pct = combined_signal
        target_position_value = self.cash * target_position_pct
        
        # Determine order size
        order_size = target_position_value - self.position_value
        
        # Trade if order size is significant
        if abs(order_size) > 1000:
            order_type = "BUY" if order_size > 0 else "SELL"
            logger.info(f"TRADE: {order_type} ${abs(order_size):.2f}")
            
            # Record order
            self.orders.append({
                "type": order_type,
                "size": abs(order_size),
                "timestamp": datetime.now().isoformat()
            })
            
            # Update position
            self.position_value += order_size
            
            # Log portfolio status
            total_value = self.cash + self.position_value
            position_pct = self.position_value / total_value * 100
            logger.info(f"Portfolio: ${total_value:.2f} (Position: ${self.position_value:.2f}, {position_pct:.1f}%)")


class MarketDataSimulator:
    """
    Simulates market data for testing when no real data source is available.
    """
    
    def __init__(
        self,
        base_price: float = 100.0,
        volatility: float = 0.002,
        trend: float = 0.0001,
        regime_change_prob: float = 0.001
    ):
        """
        Initialize market data simulator.
        
        Args:
            base_price: Starting price
            volatility: Daily volatility (standard deviation)
            trend: Daily trend (drift)
            regime_change_prob: Probability of regime change per tick
        """
        self.current_price = base_price
        self.volatility = volatility
        self.trend = trend
        self.regime_change_prob = regime_change_prob
        self.current_regime = MarketRegime.UNKNOWN
        self.last_update = datetime.now()
        self.tick_count = 0
        
    async def generate_tick(self) -> Dict[str, Any]:
        """
        Generate a simulated market data tick.
        
        Returns:
            Dict with simulated market data
        """
        import random
        import numpy as np
        
        # Update tick count
        self.tick_count += 1
        
        # Occasionally change market regime
        if random.random() < self.regime_change_prob:
            regimes = list(MarketRegime)
            self.current_regime = random.choice(regimes)
            logger.info(f"Market regime changed to {self.current_regime.name}")
            
            # Return regime update
            return {
                "regime": self.current_regime.name,
                "confidence": random.uniform(0.7, 1.0),
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_update).total_seconds() / 86400  # in days
        self.last_update = now
        
        # Calculate price change
        price_drift = self.trend * time_delta
        price_vol = self.volatility * np.sqrt(time_delta)
        price_change = np.random.normal(price_drift, price_vol)
        
        # Adjust trend and volatility based on regime
        if self.current_regime == MarketRegime.BULL:
            price_change += 0.001
        elif self.current_regime == MarketRegime.BEAR:
            price_change -= 0.001
        elif self.current_regime == MarketRegime.HIGH_VOL:
            price_change *= 2
        
        # Update price
        self.current_price *= (1 + price_change)
        
        # Generate volume
        base_volume = 1000
        volume = int(np.random.gamma(2.0, base_volume / 2))
        
        # Every 100 ticks, generate a performance update
        if self.tick_count % 100 == 0:
            return {
                "performance": {
                    "MomentumStrategy": random.uniform(-0.02, 0.05),
                    "TrendFollowingStrategy": random.uniform(-0.01, 0.03),
                    "MeanReversionStrategy": random.uniform(-0.03, 0.02),
                    "DQNStrategy": random.uniform(-0.02, 0.04),
                    "MetaLearningStrategy": random.uniform(-0.03, 0.06)
                },
                "timestamp": now.isoformat()
            }
        
        # Return price update
        return {
            "price": self.current_price,
            "volume": volume,
            "timestamp": now.isoformat()
        }


async def run_with_simulator(rotator: StrategyRotator, duration: int = 600) -> None:
    """
    Run the strategy rotator with simulated market data.
    
    Args:
        rotator: Strategy rotator instance
        duration: Duration in seconds to run
    """
    # Initialize simulator
    simulator = MarketDataSimulator()
    
    # Create trader
    trader = MockTrader()
    
    # Register signal handler
    rotator.register_signal_handler(trader.handle_signal)
    
    # Start rotator
    await rotator.start()
    
    # Run for specified duration
    logger.info(f"Running simulation for {duration} seconds...")
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < duration:
        # Generate data tick
        data = await simulator.generate_tick()
        
        # Publish to rotator
        await rotator.publish_market_data(data)
        
        # Wait a bit
        await asyncio.sleep(0.1)
    
    # Stop rotator
    await rotator.stop()
    
    logger.info("Simulation completed")


async def run_with_websocket(
    rotator: StrategyRotator,
    websocket_url: str,
    auth_params: Optional[Dict[str, Any]] = None,
    subscriptions: Optional[Dict[str, Any]] = None,
    duration: int = 3600
) -> None:
    """
    Run the strategy rotator with real market data from a WebSocket connection.
    
    Args:
        rotator: Strategy rotator instance
        websocket_url: WebSocket server URL
        auth_params: Authentication parameters 
        subscriptions: Channel subscriptions
        duration: Maximum duration in seconds
    """
    # Create trader
    trader = MockTrader()
    
    # Register signal handler
    rotator.register_signal_handler(trader.handle_signal)
    
    # Start rotator
    await rotator.start()
    
    # Create data connector
    async def on_market_data(data):
        # Transform data if needed
        transformed = transform_exchange_data(data)
        
        # Publish to strategy rotator
        await rotator.publish_market_data(transformed)
    
    connector = RealTimeDataConnector(
        url=websocket_url,
        on_message_callback=on_market_data,
        auth_params=auth_params,
        subscriptions=subscriptions
    )
    
    # Setup graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        loop.create_task(shutdown())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    async def shutdown():
        await rotator.stop()
        connector.stop()
        logger.info("Gracefully shut down")
        loop.stop()
    
    # Run connector
    logger.info(f"Connecting to {websocket_url}")
    connector_task = asyncio.create_task(connector.connect())
    
    # Run for specified duration or until interrupted
    if duration:
        try:
            await asyncio.sleep(duration)
            await shutdown()
        except asyncio.CancelledError:
            pass
    else:
        # Run indefinitely
        await connector_task


def transform_exchange_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform exchange-specific data format to our internal format.
    
    Args:
        data: Exchange-specific data
        
    Returns:
        Dict with transformed data
    """
    # This function would be customized for each exchange/data source
    # Here's a simple example for a generic price feed
    
    # Extract price if available
    if "price" in data:
        return {
            "price": float(data["price"]),
            "timestamp": data.get("timestamp", datetime.now().isoformat())
        }
    
    # Handle other data types
    return data


def get_exchange_config(exchange_name: str) -> Dict[str, Any]:
    """
    Get exchange-specific configuration.
    
    Args:
        exchange_name: Name of exchange
        
    Returns:
        Dict with exchange configuration
    """
    # Load from config or environment variables
    if exchange_name.lower() == "binance":
        return {
            "websocket_url": "wss://stream.binance.com:9443/ws",
            "subscriptions": {
                "BTCUSDT@trade": {}
            }
        }
    elif exchange_name.lower() == "coinbase":
        return {
            "websocket_url": "wss://ws-feed.pro.coinbase.com",
            "subscriptions": {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["ticker"]
            }
        }
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")


async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Realtime Trading Example")
    parser.add_argument("--exchange", type=str, default="simulator",
                        help="Exchange to connect to (or 'simulator')")
    parser.add_argument("--duration", type=int, default=600,
                        help="Duration to run in seconds")
    
    args = parser.parse_args()
    
    # Create strategy rotator
    rotator = StrategyRotator(
        use_event_driven=True,
        message_broker_config={
            "queue_type": "memory",
            "max_queue_size": 10000
        }
    )
    
    # Run with simulator or exchange
    if args.exchange.lower() == "simulator":
        await run_with_simulator(rotator, args.duration)
    else:
        # Get exchange config
        try:
            exchange_config = get_exchange_config(args.exchange)
            
            await run_with_websocket(
                rotator,
                websocket_url=exchange_config["websocket_url"],
                subscriptions=exchange_config.get("subscriptions"),
                auth_params=exchange_config.get("auth_params"),
                duration=args.duration
            )
        except ValueError as e:
            logger.error(str(e))
            return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main()) 