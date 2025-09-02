#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Trading Example

This example demonstrates the paper trading system with both paper and live strategies
running concurrently. It shows how to:
1. Set up PaperBroker and integrate it with real broker interfaces
2. Define strategies with different trading modes (paper/live)
3. Route orders to the appropriate broker based on strategy settings
4. Track paper positions and orders alongside live ones
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.brokers.paper_broker import PaperBroker
from trading_bot.brokers.paper_broker_factory import create_paper_broker, mirror_broker_as_paper
from trading_bot.core.strategy_broker_router import StrategyBrokerRouter
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, OrderSide
from trading_bot.strategies.moving_average_crossover import MovingAverageCrossover
from trading_bot.strategies.rsi_strategy import RSIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_event_listeners(event_bus):
    """Set up event listeners for order events."""
    
    def on_order_placed(event):
        """Handler for order placed events."""
        data = event.data
        mode = data.get("mode", "unknown")
        mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"
        strategy_id = data.get("strategy_id", "unknown")
        order = data.get("order", {})
        
        logger.info(f"{mode_tag} Order placed by strategy {strategy_id}: {order}")
    
    def on_order_filled(event):
        """Handler for order filled events."""
        data = event.data
        broker = data.get("broker", "unknown")
        mode_tag = "[PAPER]" if "paper" in broker else "[LIVE]"
        order_id = data.get("order_id", "unknown")
        symbol = data.get("symbol", "unknown")
        side = data.get("side", "unknown")
        price = data.get("price", 0.0)
        
        logger.info(f"{mode_tag} Order {order_id} filled: {side} {symbol} @ {price}")
    
    # Register event handlers
    event_bus.subscribe(EventType.ORDER_PLACED, on_order_placed)
    event_bus.subscribe(EventType.ORDER_FILLED, on_order_filled)


def create_sample_strategy_configs():
    """Create sample strategy configurations for paper and live trading."""
    return {
        "ma_crossover_paper": {
            "name": "MA Crossover Paper",
            "id": "ma_crossover_paper",
            "mode": "paper",  # Paper trading mode
            "broker": "alpaca_paper",  # Use Alpaca paper broker
            "symbols": ["AAPL", "MSFT", "GOOG"],
            "timeframe": "1h",
            "parameters": {
                "fast_ma": 10,
                "slow_ma": 30,
                "volume_factor": 1.5
            },
            "risk": {
                "max_position_size": 0.05,  # 5% of portfolio
                "stop_loss_pct": 0.02      # 2% stop loss
            }
        },
        "ma_crossover_live": {
            "name": "MA Crossover Live",
            "id": "ma_crossover_live",
            "mode": "live",  # Live trading mode
            "broker": "alpaca",  # Use Alpaca live broker
            "symbols": ["SPY", "QQQ", "IWM"],
            "timeframe": "1d",
            "parameters": {
                "fast_ma": 5,
                "slow_ma": 20,
                "volume_factor": 1.2
            },
            "risk": {
                "max_position_size": 0.03,  # 3% of portfolio
                "stop_loss_pct": 0.015     # 1.5% stop loss
            }
        },
        "rsi_paper": {
            "name": "RSI Paper",
            "id": "rsi_paper",
            "mode": "paper",  # Paper trading mode
            "broker": "tradier_paper",  # Use Tradier paper broker
            "symbols": ["AMZN", "NFLX", "META"],
            "timeframe": "1h",
            "parameters": {
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30
            },
            "risk": {
                "max_position_size": 0.04,  # 4% of portfolio
                "stop_loss_pct": 0.025     # 2.5% stop loss
            }
        }
    }


def setup_broker_manager():
    """Set up the multi-broker manager with paper and live brokers."""
    # Create event bus and register with service registry
    event_bus = EventBus()
    service_registry = ServiceRegistry.get_instance()
    service_registry.register_service(EventBus, event_bus)
    
    # Set up event listeners
    setup_event_listeners(event_bus)
    
    # Create multi-broker manager
    broker_manager = MultiBrokerManager()
    
    # Example: In real implementation, you would add actual brokers
    # For this example, we'll simulate having Alpaca and Tradier brokers
    
    # For demonstration only: Create dummy broker classes
    # In a real implementation, you would use actual broker implementations
    class DummyBroker:
        def __init__(self, name):
            self.name = name
        
        def get_quote(self, symbol):
            # Simulate a quote
            import random
            price = round(random.uniform(100, 500), 2)
            return {
                "symbol": symbol,
                "bid": price - 0.01,
                "ask": price + 0.01,
                "last": price,
                "volume": int(random.uniform(1000, 10000)),
                "timestamp": datetime.now().isoformat()
            }
        
        def get_historical_data(self, symbol, interval, start_date, end_date=None):
            # Simulate historical data
            data = []
            current = start_date
            end = end_date or datetime.now()
            
            while current < end:
                import random
                price = round(random.uniform(100, 500), 2)
                data.append({
                    "timestamp": current.isoformat(),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": int(random.uniform(1000, 10000))
                })
                
                if interval == "1d":
                    current += timedelta(days=1)
                elif interval == "1h":
                    current += timedelta(hours=1)
                else:
                    current += timedelta(minutes=5)
            
            return data
    
    # Create dummy brokers (for demo - in real code, use actual brokers)
    dummy_alpaca = DummyBroker("Alpaca")
    dummy_tradier = DummyBroker("Tradier")
    
    # Register dummy brokers with manager (for demo purposes)
    broker_manager.add_broker("alpaca", dummy_alpaca, None, True)
    broker_manager.add_broker("tradier", dummy_tradier, None, False)
    
    # Create paper brokers that mirror the real ones
    # In a real scenario, these would use the real brokers for market data
    alpaca_paper_config = {
        "name": "AlpacaPaper",
        "id": "alpaca_paper",
        "initial_balance": 100000.0,
        "slippage_model": {
            "type": "fixed",
            "basis_points": 3  # 0.03% slippage
        },
        "commission_model": {
            "type": "per_share",
            "per_share": 0.005,
            "minimum": 1.0
        }
    }
    
    tradier_paper_config = {
        "name": "TradierPaper",
        "id": "tradier_paper",
        "initial_balance": 50000.0,
        "slippage_model": {
            "type": "random",
            "min_basis_points": 1,
            "max_basis_points": 5
        },
        "commission_model": {
            "type": "fixed",
            "per_order": 0.5
        }
    }
    
    # Create paper brokers and add to manager
    alpaca_paper = create_paper_broker(alpaca_paper_config, dummy_alpaca)
    tradier_paper = create_paper_broker(tradier_paper_config, dummy_tradier)
    
    broker_manager.add_broker("alpaca_paper", alpaca_paper, None, False)
    broker_manager.add_broker("tradier_paper", tradier_paper, None, False)
    
    return broker_manager


def setup_strategies(broker_manager, strategy_configs):
    """Set up strategies with the broker router."""
    # Create strategy broker router
    router = StrategyBrokerRouter(
        broker_manager=broker_manager,
        default_paper_broker_id="alpaca_paper",
        default_live_broker_id="alpaca"
    )
    
    # Create strategies from configs
    strategies = {}
    
    for strategy_id, config in strategy_configs.items():
        if "ma_crossover" in strategy_id:
            # Create MA Crossover strategy
            strategy = MovingAverageCrossover(
                strategy_id=config["id"],
                symbols=config["symbols"],
                timeframe=config["timeframe"],
                fast_ma=config["parameters"]["fast_ma"],
                slow_ma=config["parameters"]["slow_ma"]
            )
        elif "rsi" in strategy_id:
            # Create RSI strategy
            strategy = RSIStrategy(
                strategy_id=config["id"],
                symbols=config["symbols"],
                timeframe=config["timeframe"],
                rsi_period=config["parameters"]["rsi_period"],
                overbought=config["parameters"]["overbought"],
                oversold=config["parameters"]["oversold"]
            )
        else:
            logger.warning(f"Unknown strategy type for {strategy_id}, skipping")
            continue
        
        # Register strategy with router
        router.register_strategy_from_config(strategy, config)
        
        # Store in our strategies dict
        strategies[strategy_id] = strategy
    
    return router, strategies


def simulate_trading(router, strategies):
    """Simulate trading activity with both paper and live strategies."""
    logger.info("Starting trading simulation with paper and live strategies")
    
    # Get all strategy IDs
    strategy_ids = router.get_strategy_ids()
    
    # Print info about strategies
    logger.info("Registered strategies:")
    for strategy_id in strategy_ids:
        mode = router.get_strategy_mode(strategy_id)
        broker_id = router.get_strategy_broker_id(strategy_id)
        logger.info(f"  - {strategy_id}: Mode={mode}, Broker={broker_id}")
    
    # Simulate some trading activity
    for strategy_id in strategy_ids:
        strategy = strategies.get(strategy_id)
        if not strategy:
            continue
        
        mode = router.get_strategy_mode(strategy_id)
        mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"
        
        logger.info(f"{mode_tag} Simulating trading for strategy {strategy_id}")
        
        # Get symbols from strategy
        symbols = strategy.get_symbols()
        
        # Place a few test orders
        for symbol in symbols[:2]:  # Limit to first 2 symbols
            # Simulate a buy signal
            logger.info(f"{mode_tag} Strategy {strategy_id} generated BUY signal for {symbol}")
            
            # Place order through router
            order_result = router.place_order_for_strategy(
                strategy_id=strategy_id,
                symbol=symbol,
                side="buy",
                quantity=10,
                order_type="market"
            )
            
            if order_result:
                logger.info(f"{mode_tag} Order placed: {order_result}")
                
                # Get order status
                order_id = order_result.get("order_id")
                if order_id:
                    time.sleep(1)  # Wait for order to process
                    status = router.get_order_status_for_strategy(strategy_id, order_id)
                    logger.info(f"{mode_tag} Order status: {status}")
    
    # Wait for orders to process
    logger.info("Waiting for orders to process...")
    time.sleep(5)
    
    # Get positions for each strategy
    for strategy_id in strategy_ids:
        mode = router.get_strategy_mode(strategy_id)
        mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"
        
        positions = router.get_positions_for_strategy(strategy_id)
        logger.info(f"{mode_tag} Positions for strategy {strategy_id}: {positions}")
    
    logger.info("Trading simulation completed")


def main():
    """Main entry point for paper trading example."""
    try:
        # Set up broker manager with paper and live brokers
        broker_manager = setup_broker_manager()
        
        # Create strategy configurations
        strategy_configs = create_sample_strategy_configs()
        
        # Set up strategies with broker router
        router, strategies = setup_strategies(broker_manager, strategy_configs)
        
        # Simulate trading activity
        simulate_trading(router, strategies)
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error running paper trading example: {e}", exc_info=True)


if __name__ == "__main__":
    main()
