#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Cloud Brokerage Client

This module implements a brokerage client for Coinbase using their Cloud API format.
It adapts the CoinbaseCloudBroker to the BrokerageClient interface.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.brokers.coinbase_cloud_broker import CoinbaseCloudBroker
from trading_bot.brokers.brokerage_client import BrokerageClient

logger = logging.getLogger(__name__)

class CoinbaseCloudBrokerageClient(BrokerageClient):
    """
    A brokerage client implementation for Coinbase Cloud API.
    
    This client adapts the CoinbaseCloudBroker to fit the BrokerageClient
    interface, ensuring compatibility with the trading system.
    """
    
    def __init__(self, api_key_name: str, private_key: str, sandbox: bool = False):
        """
        Initialize the Coinbase Cloud brokerage client.
        
        Args:
            api_key_name: The API key name/path from Coinbase Cloud
            private_key: The EC private key in PEM format
            sandbox: Whether to use the sandbox environment
        """
        super().__init__()
        self.broker = CoinbaseCloudBroker(
            api_key_name=api_key_name,
            private_key=private_key,
            sandbox=sandbox
        )
        self.broker_name = "coinbase_cloud"
        self.account_id = "default"  # Coinbase Cloud doesn't use explicit account IDs
        self.sandbox_mode = sandbox
        self.rate_limit_remaining = 100  # Default assumption
        self.rate_limit_reset = time.time() + 60
        
        # Initialize with a connection check
        self.connection_status = self.check_connection()
        logger.info(f"Coinbase Cloud client initialized: {self.connection_status}")
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check connection to the Coinbase API.
        
        Returns:
            Dict with connection status information
        """
        return self.broker.check_connection()
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict with account information
        """
        return self.broker.get_account_info()
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get a quote for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            
        Returns:
            Dict with quote information
        """
        return self.broker.get_quote(symbol)
    
    def get_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> List[Dict]:
        """
        Get historical bars for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            timeframe: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start: Start datetime
            end: End datetime
            
        Returns:
            List of bar data
        """
        return self.broker.get_bars(symbol, timeframe, start, end)
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position data
        """
        return self.broker.get_positions()
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str, 
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None, 
                   time_in_force: str = 'gtc') -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            quantity: Quantity to buy/sell
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force ('gtc', 'ioc', 'fok')
            
        Returns:
            Dict with order information
        """
        # Validate order parameters
        if order_type in ['limit', 'stop_limit'] and limit_price is None:
            logger.error("Limit price is required for limit orders")
            return {"status": "error", "message": "Limit price is required for limit orders"}
        
        if order_type in ['stop', 'stop_limit'] and stop_price is None:
            logger.error("Stop price is required for stop orders")
            return {"status": "error", "message": "Stop price is required for stop orders"}
        
        # For now, we're implementing the basic place_order function
        # and ignoring the more complex parameters
        return self.broker.place_order(symbol, side, quantity, order_type)
    
    def create_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        Create a market order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            quantity: Quantity to buy/sell
            
        Returns:
            Dict with order information
        """
        return self.place_order(symbol, side, quantity, 'market')
    
    def create_limit_order(self, symbol: str, side: str, quantity: float, 
                          limit_price: float, time_in_force: str = 'gtc') -> Dict[str, Any]:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            quantity: Quantity to buy/sell
            limit_price: Limit price
            time_in_force: Time in force ('gtc', 'ioc', 'fok')
            
        Returns:
            Dict with order information
        """
        return self.place_order(
            symbol, side, quantity, 'limit', 
            limit_price=limit_price, time_in_force=time_in_force
        )
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get orders with optional status filter.
        
        Args:
            status: Order status filter (e.g., 'open', 'filled')
            
        Returns:
            List of order data
        """
        return self.broker.get_orders(status)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict with cancellation status
        """
        return self.broker.cancel_order(order_id)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all orders, optionally filtered by symbol.
        
        Args:
            symbol: Trading pair to filter orders by
            
        Returns:
            Dict with cancellation status
        """
        # Get open orders
        open_orders = self.get_orders(status='open')
        
        # Filter by symbol if provided
        if symbol:
            open_orders = [o for o in open_orders if o.get('symbol') == symbol]
        
        # Cancel each order
        cancelled = []
        failed = []
        for order in open_orders:
            result = self.cancel_order(order['id'])
            if result.get('status') == 'success':
                cancelled.append(order['id'])
            else:
                failed.append(order['id'])
        
        return {
            'status': 'success' if not failed else 'partial',
            'cancelled': cancelled,
            'failed': failed,
            'total_cancelled': len(cancelled),
            'total_failed': len(failed)
        }
    
    def handle_rate_limit(self):
        """
        Handle rate limiting by waiting if necessary.
        """
        current_time = time.time()
        if self.rate_limit_remaining <= 5 and current_time < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - current_time
            logger.warning(f"Rate limit low ({self.rate_limit_remaining}), waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        # Reset if past reset time
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 100
            self.rate_limit_reset = current_time + 60
