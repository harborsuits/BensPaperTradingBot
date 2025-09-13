"""
Live Order Executor

This module provides a real-time execution layer that interfaces with broker APIs
for live and paper trading. It supports multiple brokers including Alpaca and Tradier.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class OrderExecutor:
    """Base class for order executors"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the order executor."""
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize execution tracking
        self.orders = []
        self.executions = []
        self.errors = []
    
    def place_order(self, symbol: str, quantity: float, side: str, order_type: str, 
                   price: Optional[float] = None, time_in_force: str = 'day',
                   stop_price: Optional[float] = None, 
                   take_profit_price: Optional[float] = None,
                   strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order for a security.
        
        Args:
            symbol: Security symbol
            quantity: Order quantity
            side: Order side ('buy', 'sell', 'sell_short', 'buy_to_cover')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            price: Limit price (required for limit and stop_limit orders)
            time_in_force: Time in force for the order ('day', 'gtc', 'ioc', 'fok')
            stop_price: Stop price (required for stop and stop_limit orders)
            take_profit_price: Take profit price for OCO orders
            strategy: Optional strategy name for tracking
            
        Returns:
            Order details dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Status dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of open orders.
        
        Returns:
            List of open order dictionaries
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get list of current positions.
        
        Returns:
            List of position dictionaries
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Security symbol
            
        Returns:
            Position dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Account balance dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")


class AlpacaOrderExecutor(OrderExecutor):
    """Order executor using Alpaca API"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 config_path: Optional[str] = None, paper_trading: bool = True):
        """
        Initialize Alpaca order executor.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            config_path: Path to configuration file
            paper_trading: Whether to use paper trading
        """
        super().__init__(config_path)
        
        # Import Alpaca here to avoid dependencies if not used
        import alpaca_trade_api as tradeapi
        
        # Get API credentials from config or parameters
        if not api_key and self.config:
            api_key = self.config.get('alpaca', {}).get('api_key', os.environ.get('ALPACA_API_KEY'))
        else:
            api_key = api_key or os.environ.get('ALPACA_API_KEY')
            
        if not api_secret and self.config:
            api_secret = self.config.get('alpaca', {}).get('api_secret', os.environ.get('ALPACA_API_SECRET'))
        else:
            api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Alpaca API credentials not provided")
        
        # Set base URL based on paper trading setting
        if paper_trading:
            base_url = 'https://paper-api.alpaca.markets'
        else:
            base_url = 'https://api.alpaca.markets'
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(api_key, api_secret, base_url=base_url)
        
        self.paper_trading = paper_trading
        logger.info(f"Alpaca order executor initialized (paper trading: {paper_trading})")
    
    def place_order(self, symbol: str, quantity: float, side: str, order_type: str, 
                   price: Optional[float] = None, time_in_force: str = 'day',
                   stop_price: Optional[float] = None, 
                   take_profit_price: Optional[float] = None,
                   strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order using Alpaca API.
        
        Args:
            symbol: Security symbol
            quantity: Order quantity
            side: Order side ('buy', 'sell', 'sell_short', 'buy_to_cover')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            price: Limit price (required for limit and stop_limit orders)
            time_in_force: Time in force for the order ('day', 'gtc', 'ioc', 'fok')
            stop_price: Stop price (required for stop and stop_limit orders)
            take_profit_price: Take profit price for OCO orders
            strategy: Optional strategy name for tracking
            
        Returns:
            Order details dictionary
        """
        try:
            # Map side to Alpaca format
            alpaca_side = side.lower()
            if alpaca_side == 'sell_short':
                alpaca_side = 'sell'
            elif alpaca_side == 'buy_to_cover':
                alpaca_side = 'buy'
            
            # Map order type to Alpaca format
            alpaca_type = order_type.lower()
            if alpaca_type == 'stop':
                alpaca_type = 'stop'
            elif alpaca_type == 'stop_limit':
                alpaca_type = 'stop_limit'
            
            # Map time in force to Alpaca format
            alpaca_tif = time_in_force.lower()
            if alpaca_tif == 'gtc':
                alpaca_tif = 'gtc'
            elif alpaca_tif == 'ioc':
                alpaca_tif = 'ioc'
            elif alpaca_tif == 'fok':
                alpaca_tif = 'fok'
            else:
                alpaca_tif = 'day'
            
            # Place the order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                type=alpaca_type,
                time_in_force=alpaca_tif,
                limit_price=price,
                stop_price=stop_price,
                take_profit=take_profit_price
            )
            
            # Format response
            order_details = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_quantity': float(order.filled_qty) if order.filled_qty else 0,
                'price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'time_in_force': order.time_in_force,
                'strategy': strategy
            }
            
            # Track the order
            self.orders.append(order_details)
            
            logger.info(f"Order placed: {symbol} {side} {quantity} shares at {price if price else 'market price'}")
            return order_details
            
        except Exception as e:
            error_msg = f"Error placing order for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type,
                'price': price,
                'error': str(e)
            }
            
            self.errors.append(error_details)
            return {'error': error_msg, 'details': error_details}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order using Alpaca API.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Status dictionary
        """
        try:
            # Cancel the order
            self.api.cancel_order(order_id)
            
            # Return success status
            result = {
                'success': True,
                'order_id': order_id,
                'message': f"Order {order_id} successfully cancelled"
            }
            
            logger.info(f"Order cancelled: {order_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error cancelling order {order_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order using Alpaca API.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            # Get order details
            order = self.api.get_order(order_id)
            
            # Format response
            order_details = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_quantity': float(order.filled_qty) if order.filled_qty else 0,
                'price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'time_in_force': order.time_in_force
            }
            
            return order_details
            
        except Exception as e:
            error_msg = f"Error retrieving order status for {order_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg,
                'order_id': order_id
            }
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of open orders using Alpaca API.
        
        Returns:
            List of open order dictionaries
        """
        try:
            # Get all open orders
            orders = self.api.list_orders(status='open')
            
            # Format response
            order_list = []
            for order in orders:
                order_details = {
                    'id': order.id,
                    'client_order_id': order.client_order_id,
                    'symbol': order.symbol,
                    'quantity': float(order.qty),
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'created_at': order.created_at,
                    'submitted_at': order.submitted_at,
                    'price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'time_in_force': order.time_in_force
                }
                order_list.append(order_details)
            
            return order_list
            
        except Exception as e:
            error_msg = f"Error retrieving open orders: {str(e)}"
            logger.error(error_msg)
            
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get list of current positions using Alpaca API.
        
        Returns:
            List of position dictionaries
        """
        try:
            # Get all positions
            positions = self.api.list_positions()
            
            # Format response
            position_list = []
            for position in positions:
                position_details = {
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'side': 'long' if float(position.qty) > 0 else 'short',
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'change_today': float(position.change_today)
                }
                position_list.append(position_details)
            
            return position_list
            
        except Exception as e:
            error_msg = f"Error retrieving positions: {str(e)}"
            logger.error(error_msg)
            
            return []
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol using Alpaca API.
        
        Args:
            symbol: Security symbol
            
        Returns:
            Position dictionary
        """
        try:
            # Get specific position
            position = self.api.get_position(symbol)
            
            # Format response
            position_details = {
                'symbol': position.symbol,
                'quantity': float(position.qty),
                'side': 'long' if float(position.qty) > 0 else 'short',
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'change_today': float(position.change_today)
            }
            
            return position_details
            
        except Exception as e:
            error_msg = f"Error retrieving position for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg,
                'symbol': symbol
            }
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information using Alpaca API.
        
        Returns:
            Account balance dictionary
        """
        try:
            # Get account information
            account = self.api.get_account()
            
            # Format response
            balance = {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': int(account.daytrade_count),
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin)
            }
            
            return balance
            
        except Exception as e:
            error_msg = f"Error retrieving account balance: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg
            }


class TradierOrderExecutor(OrderExecutor):
    """Order executor using Tradier API"""
    
    def __init__(self, api_key: str = None, config_path: Optional[str] = None, sandbox: bool = True):
        """
        Initialize Tradier order executor.
        
        Args:
            api_key: Tradier API key
            config_path: Path to configuration file
            sandbox: Whether to use Tradier sandbox environment
        """
        super().__init__(config_path)
        
        # Get API key from config or parameter
        if not api_key and self.config:
            api_key = self.config.get('tradier', {}).get('api_key', os.environ.get('TRADIER_API_KEY'))
        else:
            api_key = api_key or os.environ.get('TRADIER_API_KEY')
        
        if not api_key:
            raise ValueError("Tradier API key not provided")
        
        # Get account number
        account_number = None
        if self.config:
            account_number = self.config.get('tradier', {}).get('account_number', os.environ.get('TRADIER_ACCOUNT_NUMBER'))
        else:
            account_number = os.environ.get('TRADIER_ACCOUNT_NUMBER')
        
        if not account_number:
            raise ValueError("Tradier account number not provided")
        
        # Set API base URL based on environment
        self.sandbox = sandbox
        if sandbox:
            self.base_url = 'https://sandbox.tradier.com/v1'
        else:
            self.base_url = 'https://api.tradier.com/v1'
        
        # Set headers for API requests
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
        self.account_number = account_number
        logger.info(f"Tradier order executor initialized (sandbox: {sandbox})")
    
    def place_order(self, symbol: str, quantity: float, side: str, order_type: str, 
                   price: Optional[float] = None, time_in_force: str = 'day',
                   stop_price: Optional[float] = None, 
                   take_profit_price: Optional[float] = None,
                   strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order using Tradier API.
        
        Args:
            symbol: Security symbol
            quantity: Order quantity
            side: Order side ('buy', 'sell', 'sell_short', 'buy_to_cover')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            price: Limit price (required for limit and stop_limit orders)
            time_in_force: Time in force for the order ('day', 'gtc', 'ioc', 'fok')
            stop_price: Stop price (required for stop and stop_limit orders)
            take_profit_price: Take profit price for OCO orders
            strategy: Optional strategy name for tracking
            
        Returns:
            Order details dictionary
        """
        try:
            import requests
            
            # Map side to Tradier format
            tradier_side = side.lower()
            if tradier_side not in ['buy', 'sell', 'sell_short', 'buy_to_cover']:
                raise ValueError(f"Invalid order side: {side}")
            
            # Map order type to Tradier format
            tradier_type = order_type.lower()
            if tradier_type not in ['market', 'limit', 'stop', 'stop_limit']:
                raise ValueError(f"Invalid order type: {order_type}")
            
            # Map time in force to Tradier format
            tradier_tif = time_in_force.lower()
            if tradier_tif not in ['day', 'gtc', 'ioc', 'fok']:
                tradier_tif = 'day'
            
            # Build request data
            data = {
                'class': 'equity',
                'symbol': symbol,
                'side': tradier_side,
                'quantity': quantity,
                'type': tradier_type,
                'duration': tradier_tif
            }
            
            # Add price parameters based on order type
            if tradier_type in ['limit', 'stop_limit']:
                if price is None:
                    raise ValueError(f"Price required for {order_type} orders")
                data['price'] = price
            
            if tradier_type in ['stop', 'stop_limit']:
                if stop_price is None:
                    raise ValueError(f"Stop price required for {order_type} orders")
                data['stop'] = stop_price
            
            # Place the order
            url = f"{self.base_url}/accounts/{self.account_number}/orders"
            response = requests.post(url, data=data, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Parse response
            order_data = response.json()
            if 'errors' in order_data:
                raise Exception(f"Order error: {order_data['errors']['error']}")
            
            order_id = order_data.get('order', {}).get('id')
            
            # Get order details
            order_details = self.get_order_status(order_id)
            order_details['strategy'] = strategy
            
            # Track the order
            self.orders.append(order_details)
            
            logger.info(f"Order placed: {symbol} {side} {quantity} shares at {price if price else 'market price'}")
            return order_details
            
        except Exception as e:
            error_msg = f"Error placing order for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type,
                'price': price,
                'error': str(e)
            }
            
            self.errors.append(error_details)
            return {'error': error_msg, 'details': error_details}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order using Tradier API.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Status dictionary
        """
        try:
            import requests
            
            # Send cancel request
            url = f"{self.base_url}/accounts/{self.account_number}/orders/{order_id}"
            response = requests.delete(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Parse response
            result_data = response.json()
            if 'errors' in result_data:
                raise Exception(f"Cancel error: {result_data['errors']['error']}")
            
            # Return success status
            result = {
                'success': True,
                'order_id': order_id,
                'message': f"Order {order_id} successfully cancelled"
            }
            
            logger.info(f"Order cancelled: {order_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error cancelling order {order_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order using Tradier API.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            import requests
            
            # Get order details
            url = f"{self.base_url}/accounts/{self.account_number}/orders/{order_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Parse response
            order_data = response.json()
            if 'errors' in order_data:
                raise Exception(f"Order status error: {order_data['errors']['error']}")
            
            order = order_data.get('order', {})
            
            # Format response
            order_details = {
                'id': order.get('id'),
                'symbol': order.get('symbol'),
                'quantity': float(order.get('quantity', 0)),
                'side': order.get('side'),
                'type': order.get('type'),
                'status': order.get('status'),
                'created_at': order.get('create_date'),
                'price': float(order.get('price', 0)) if order.get('price') else None,
                'stop_price': float(order.get('stop', 0)) if order.get('stop') else None,
                'time_in_force': order.get('duration'),
                'filled_quantity': float(order.get('filled_quantity', 0))
            }
            
            return order_details
            
        except Exception as e:
            error_msg = f"Error retrieving order status for {order_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg,
                'order_id': order_id
            }
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of open orders using Tradier API.
        
        Returns:
            List of open order dictionaries
        """
        try:
            import requests
            
            # Get all orders
            url = f"{self.base_url}/accounts/{self.account_number}/orders"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Parse response
            orders_data = response.json()
            if 'errors' in orders_data:
                raise Exception(f"Orders error: {orders_data['errors']['error']}")
            
            orders = orders_data.get('orders', {}).get('order', [])
            if not isinstance(orders, list):
                orders = [orders]
            
            # Filter for open orders
            open_orders = [order for order in orders if order.get('status') in ['open', 'pending', 'partially_filled']]
            
            # Format response
            order_list = []
            for order in open_orders:
                order_details = {
                    'id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'quantity': float(order.get('quantity', 0)),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'status': order.get('status'),
                    'created_at': order.get('create_date'),
                    'price': float(order.get('price', 0)) if order.get('price') else None,
                    'stop_price': float(order.get('stop', 0)) if order.get('stop') else None,
                    'time_in_force': order.get('duration'),
                    'filled_quantity': float(order.get('filled_quantity', 0))
                }
                order_list.append(order_details)
            
            return order_list
            
        except Exception as e:
            error_msg = f"Error retrieving open orders: {str(e)}"
            logger.error(error_msg)
            
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get list of current positions using Tradier API.
        
        Returns:
            List of position dictionaries
        """
        try:
            import requests
            
            # Get all positions
            url = f"{self.base_url}/accounts/{self.account_number}/positions"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Parse response
            positions_data = response.json()
            if 'errors' in positions_data:
                raise Exception(f"Positions error: {positions_data['errors']['error']}")
            
            positions = positions_data.get('positions', {}).get('position', [])
            if not isinstance(positions, list):
                positions = [positions]
            
            # Format response
            position_list = []
            for position in positions:
                position_details = {
                    'symbol': position.get('symbol'),
                    'quantity': float(position.get('quantity', 0)),
                    'side': 'long' if float(position.get('quantity', 0)) > 0 else 'short',
                    'avg_entry_price': float(position.get('cost_basis', 0)) / float(position.get('quantity', 1)),
                    'current_price': float(position.get('last_price', 0)),
                    'market_value': float(position.get('market_value', 0)),
                    'cost_basis': float(position.get('cost_basis', 0)),
                    'unrealized_pl': float(position.get('gain_loss', 0))
                }
                position_list.append(position_details)
            
            return position_list
            
        except Exception as e:
            error_msg = f"Error retrieving positions: {str(e)}"
            logger.error(error_msg)
            
            return []
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol using Tradier API.
        
        Args:
            symbol: Security symbol
            
        Returns:
            Position dictionary
        """
        try:
            # Get all positions and filter
            positions = self.get_positions()
            
            # Find position for the requested symbol
            for position in positions:
                if position.get('symbol') == symbol:
                    return position
            
            # If not found, return empty position
            return {
                'symbol': symbol,
                'quantity': 0,
                'side': None,
                'avg_entry_price': 0,
                'current_price': 0,
                'market_value': 0,
                'cost_basis': 0,
                'unrealized_pl': 0
            }
            
        except Exception as e:
            error_msg = f"Error retrieving position for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg,
                'symbol': symbol
            }
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information using Tradier API.
        
        Returns:
            Account balance dictionary
        """
        try:
            import requests
            
            # Get account balances
            url = f"{self.base_url}/accounts/{self.account_number}/balances"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Parse response
            balances_data = response.json()
            if 'errors' in balances_data:
                raise Exception(f"Balance error: {balances_data['errors']['error']}")
            
            balances = balances_data.get('balances', {})
            
            # Format response
            balance = {
                'cash': float(balances.get('cash', 0)),
                'portfolio_value': float(balances.get('total_equity', 0)),
                'equity': float(balances.get('equity', 0)),
                'buying_power': float(balances.get('option_buying_power', 0)),
                'maintenance_margin': float(balances.get('maintenance_requirement', 0)),
                'day_trade_buying_power': float(balances.get('day_trade_buying_power', 0))
            }
            
            return balance
            
        except Exception as e:
            error_msg = f"Error retrieving account balance: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg
            }


# Factory function to create an order executor based on configuration
def create_order_executor(
    broker: str,
    config_path: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    paper_trading: bool = True
) -> OrderExecutor:
    """
    Create an order executor for the specified broker.
    
    Args:
        broker: Broker name ('alpaca', 'tradier')
        config_path: Path to configuration file
        api_key: API key for broker
        api_secret: API secret (if required by broker)
        paper_trading: Whether to use paper trading
        
    Returns:
        OrderExecutor instance
    """
    broker = broker.lower()
    
    if broker == 'alpaca':
        return AlpacaOrderExecutor(
            api_key=api_key,
            api_secret=api_secret,
            config_path=config_path,
            paper_trading=paper_trading
        )
    elif broker == 'tradier':
        return TradierOrderExecutor(
            api_key=api_key,
            config_path=config_path,
            sandbox=paper_trading
        )
    else:
        raise ValueError(f"Unsupported broker: {broker}") 