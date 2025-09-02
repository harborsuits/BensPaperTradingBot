#!/usr/bin/env python3
"""
Broker Integration Example

This example demonstrates how to use the brokerage API integration
to connect to brokers, place orders, and monitor account activity.
"""

import os
import sys
import logging
import time
from typing import Dict, Any
import argparse
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_bot.brokers.broker_registry import get_broker_registry
from trading_bot.brokers.brokerage_client import OrderType, OrderSide, TimeInForce, BrokerConnectionStatus
from trading_bot.brokers.order_selector import OrderSelector, MarketCondition, ExecutionSpeed, PriceAggression
from trading_bot.brokers.connection_monitor import ConnectionAlert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('broker_example')

def connection_alert_handler(alert: ConnectionAlert) -> None:
    """Handle connection alerts."""
    if alert.resolved:
        logger.info(f"RESOLVED: {alert.broker_name} - {alert.resolution_message}")
    else:
        logger.warning(f"ALERT: {alert.broker_name} - {alert.message}")

def display_account_info(account_info: Dict[str, Any]) -> None:
    """Display account information."""
    print("\n=== Account Information ===")
    print(f"Cash: ${account_info.get('cash', 0):.2f}")
    print(f"Equity: ${account_info.get('equity', 0):.2f}")
    print(f"Buying Power: ${account_info.get('buying_power', 0):.2f}")
    print(f"Day Trades: {account_info.get('daytrade_count', 0)}")
    print("===========================\n")

def display_positions(positions: list) -> None:
    """Display positions."""
    if not positions:
        print("\nNo open positions")
        return
    
    print("\n=== Open Positions ===")
    for position in positions:
        symbol = position.get('symbol', 'Unknown')
        qty = position.get('quantity', 0)
        entry = position.get('avg_entry_price', 0)
        current = position.get('current_price', 0)
        pnl = position.get('unrealized_pl', 0)
        pnl_pct = position.get('unrealized_plpc', 0) * 100  # Convert to percentage
        
        print(f"{symbol}: {qty} shares @ ${entry:.2f} | Current: ${current:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    print("=====================\n")

def display_orders(orders: list) -> None:
    """Display orders."""
    if not orders:
        print("\nNo open orders")
        return
    
    print("\n=== Open Orders ===")
    for order in orders:
        order_id = order.get('id', 'Unknown')
        symbol = order.get('symbol', 'Unknown')
        side = order.get('side', 'Unknown')
        qty = order.get('quantity', 0)
        filled = order.get('filled_quantity', 0)
        order_type = order.get('type', 'Unknown')
        price = order.get('price', None)
        stop_price = order.get('stop_price', None)
        status = order.get('status', 'Unknown')
        
        price_str = f"@ ${price:.2f}" if price else ""
        stop_str = f"Stop: ${stop_price:.2f}" if stop_price else ""
        
        print(f"{symbol}: {side} {qty} shares ({filled} filled) {order_type} {price_str} {stop_str} | Status: {status} | ID: {order_id}")
    
    print("=================\n")

def main():
    """Main function demonstrating broker integration."""
    parser = argparse.ArgumentParser(description='Broker Integration Example')
    parser.add_argument('--broker', type=str, default='alpaca_paper', help='Broker to use')
    parser.add_argument('--config-dir', type=str, help='Configuration directory')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to use for orders')
    parser.add_argument('--place-order', action='store_true', help='Place a sample order')
    parser.add_argument('--order-qty', type=float, default=1, help='Order quantity')
    
    args = parser.parse_args()
    
    try:
        # Get broker registry
        registry = get_broker_registry(args.config_dir)
        
        # Register alert handler
        registry.register_alert_handler(connection_alert_handler)
        
        # Start connection monitoring
        registry.start_connection_monitoring()
        
        # Get the broker client
        broker = registry.get_broker(args.broker)
        
        if not broker:
            logger.error(f"Broker '{args.broker}' not found")
            print(f"Available brokers: {list(registry.get_all_brokers().keys())}")
            return 1
        
        # Ensure broker is connected
        if broker.connection_status != BrokerConnectionStatus.CONNECTED:
            logger.info(f"Connecting to {args.broker}...")
            broker.connect()
        
        # Create order selector
        order_selector = OrderSelector(broker)
        
        # Get account information
        logger.info("Fetching account information...")
        account_info = broker.get_account_info()
        display_account_info(account_info)
        
        # Get current positions
        logger.info("Fetching positions...")
        positions = broker.get_positions()
        display_positions(positions)
        
        # Get open orders
        logger.info("Fetching open orders...")
        orders = broker.get_orders(status='open')
        display_orders(orders)
        
        # Check if market is open
        logger.info("Checking market status...")
        market_hours = broker.get_market_hours()
        market_open = broker.is_market_open()
        
        print(f"Market is {'OPEN' if market_open else 'CLOSED'}")
        
        if market_hours.get('next_open'):
            next_open = datetime.fromisoformat(market_hours['next_open'].replace('Z', '+00:00'))
            print(f"Next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if market_hours.get('next_close'):
            next_close = datetime.fromisoformat(market_hours['next_close'].replace('Z', '+00:00'))
            print(f"Next market close: {next_close.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Place a sample order if requested
        if args.place_order:
            # Check if we already have a position
            existing_position = None
            for position in positions:
                if position.get('symbol') == args.symbol:
                    existing_position = position
                    break
            
            # Determine order side (sell if we have a position, buy otherwise)
            side = OrderSide.SELL if existing_position else OrderSide.BUY
            quantity = existing_position.get('quantity', 0) if existing_position else args.order_qty
            
            if side == OrderSide.BUY:
                logger.info(f"Placing a BUY order for {quantity} {args.symbol}...")
            else:
                logger.info(f"Placing a SELL order for {quantity} {args.symbol} (closing position)...")
            
            # Get optimal order parameters
            market_condition = order_selector._detect_market_condition()
            print(f"Detected market condition: {market_condition}")
            
            # Base parameters for demonstration
            params = order_selector.get_optimal_order_parameters(
                symbol=args.symbol,
                side=side,
                quantity=quantity,
                market_condition=market_condition,
                execution_speed=ExecutionSpeed.BALANCED,
                price_aggression=PriceAggression.NEUTRAL
            )
            
            print(f"Order parameters: {params}")
            
            # Confirm before placing order
            confirm = input("Place this order? (y/n): ").lower().strip()
            if confirm != 'y':
                print("Order cancelled by user")
                return 0
            
            # Place the order
            order_result = broker.place_order(
                symbol=params['symbol'],
                side=params['side'],
                quantity=params['quantity'],
                order_type=params['order_type'],
                time_in_force=params['time_in_force'],
                limit_price=params.get('limit_price'),
                stop_price=params.get('stop_price')
            )
            
            print(f"Order placed: {order_result}")
            
            # Wait briefly and fetch updated orders
            time.sleep(2)
            logger.info("Fetching updated orders...")
            orders = broker.get_orders(status='open')
            display_orders(orders)
        
        # All done
        print("Broker integration example completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("User interrupted, shutting down...")
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1
    
    finally:
        # Clean up
        if 'registry' in locals():
            registry.stop_connection_monitoring()

if __name__ == '__main__':
    sys.exit(main()) 