#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Broker Implementation

This module implements a broker interface for Coinbase to allow trading
cryptocurrencies using the Coinbase Advanced API.
"""

import logging
import uuid
import json
import hmac
import hashlib
import time
import requests
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading

from trading_bot.brokers.broker_interface import BrokerInterface, MarketSession
from trading_bot.core.events import Event, EventType
from trading_bot.event_system.event_bus import EventBus

logger = logging.getLogger(__name__)

# Constants
COINBASE_API_URL = "https://api.coinbase.com"
COINBASE_ADVANCED_API_URL = "https://api.exchange.coinbase.com"
SECONDS_IN_DAY = 86400


class CoinbaseBroker(BrokerInterface):
    """
    Coinbase Broker Implementation
    
    Connects to Coinbase's APIs to allow real cryptocurrency trading with
    proper account management and risk controls.
    """
    
    def __init__(self, 
                api_key: str, 
                api_secret: str, 
                passphrase: Optional[str] = None, 
                sandbox: bool = False,
                event_bus: Optional[EventBus] = None):
        """
        Initialize the Coinbase broker.
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            passphrase: Coinbase API passphrase (required for Advanced API)
            sandbox: Whether to use sandbox environment
            event_bus: Event bus for publishing events
        """
        super().__init__(event_bus)
        self.broker_id = 'coinbase'
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._sandbox = sandbox
        
        # Set API URLs based on sandbox mode
        if sandbox:
            self._base_url = "https://api-public.sandbox.exchange.coinbase.com"
        else:
            self._base_url = COINBASE_ADVANCED_API_URL
            
        # Internal state
        self._status = "disconnected"
        self._account_cache = {}
        self._last_account_update = None
        self._last_quote_update = {}
        self._market_data_cache = {}
        self._order_cache = {}
        self._position_cache = []
        
        # Rate limiting
        self._rate_limit_remaining = 3000  # Default limit
        self._rate_limit_reset = time.time() + 3600
        
        # Threading
        self._lock = threading.RLock()
        self._last_connection_check = datetime.now()
        
        # Initialize connection
        self._initialize_connection()
        
        # Set up market data refresh
        self._market_data_thread = threading.Thread(target=self._market_data_refresh_thread, daemon=True)
        self._market_data_thread.start()
    
    def _initialize_connection(self) -> None:
        """Initialize connection to Coinbase API and validate credentials."""
        try:
            # Test API key with a simple request
            response = self._send_request("GET", "/accounts", is_auth=True)
            
            if response.status_code == 200:
                self._status = "connected"
                self.logger.info("Successfully connected to Coinbase API")
            else:
                self._status = "error"
                self.logger.error(f"Failed to connect to Coinbase API: {response.status_code}, {response.text}")
                
        except Exception as e:
            self._status = "error"
            self.logger.error(f"Error connecting to Coinbase API: {str(e)}")
    
    def _market_data_refresh_thread(self) -> None:
        """Background thread to refresh market data periodically."""
        while True:
            try:
                # Refresh position data for active positions
                if self._position_cache:
                    symbols = list(set(position.get('symbol') for position in self._position_cache))
                    for symbol in symbols:
                        self.get_quote(symbol)
                        
                # Sleep between refreshes
                time.sleep(5)  # Refresh every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in market data refresh thread: {str(e)}")
                time.sleep(10)  # Longer wait on error
    
    def _send_request(self, 
                    method: str, 
                    endpoint: str, 
                    params: Optional[Dict[str, Any]] = None, 
                    data: Optional[Dict[str, Any]] = None,
                    is_auth: bool = False) -> requests.Response:
        """
        Send a request to the Coinbase API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            is_auth: Whether to authenticate the request
            
        Returns:
            Response object
        """
        url = self._base_url + endpoint
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        # Add authentication if needed
        if is_auth:
            timestamp = str(int(time.time()))
            message = timestamp + method + endpoint
            
            if params:
                message += '?' + '&'.join(f"{k}={v}" for k, v in params.items())
            
            if data:
                message += json.dumps(data)
                
            # Create signature
            signature = hmac.new(
                base64.b64decode(self._api_secret),
                message.encode('utf-8'),
                hashlib.sha256
            )
            signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
            
            # Add auth headers
            headers.update({
                'CB-ACCESS-KEY': self._api_key,
                'CB-ACCESS-SIGN': signature_b64,
                'CB-ACCESS-TIMESTAMP': timestamp,
            })
            
            if self._passphrase:
                headers['CB-ACCESS-PASSPHRASE'] = self._passphrase
        
        # Send the request
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, params=params, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Update rate limits from headers
            if 'cb-after' in response.headers:
                self._rate_limit_remaining = int(response.headers.get('cb-remaining', 3000))
                self._rate_limit_reset = float(response.headers.get('cb-after', time.time() + 3600))
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error sending request to Coinbase: {str(e)}")
            raise
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Crypto markets operate 24/7, so this always returns True.
        
        Returns:
            bool: Always True for crypto
        """
        return True
    
    def get_next_market_open(self) -> datetime:
        """
        Get the next market open datetime.
        
        Crypto markets operate 24/7, so this returns the current time.
        
        Returns:
            datetime: Current time
        """
        return datetime.now()
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get trading hours information.
        
        Returns:
            Dict: Trading hours information (24/7 for crypto)
        """
        return {
            "is_open": True,
            "open_time": "00:00:00",
            "close_time": "23:59:59",
            "timezone": "UTC",
            "is_24h": True
        }
    
    def get_account_balances(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Dict: Account balance details
        """
        # Check if we need to refresh the cache
        now = datetime.now()
        if (self._last_account_update is None or 
            (now - self._last_account_update).total_seconds() > 60):
            
            try:
                # Get account information
                response = self._send_request("GET", "/accounts", is_auth=True)
                
                if response.status_code == 200:
                    accounts_data = response.json()
                    
                    # Process account data
                    balances = {}
                    total_value_usd = 0.0
                    
                    for account in accounts_data:
                        currency = account.get('currency')
                        balance = float(account.get('balance', 0))
                        hold = float(account.get('hold', 0))
                        available = float(account.get('available', 0))
                        
                        # Get USD value for non-USD currencies
                        usd_value = balance
                        if currency != 'USD':
                            # Get exchange rate
                            try:
                                ticker_response = self._send_request(
                                    "GET", 
                                    f"/products/{currency}-USD/ticker",
                                    is_auth=False
                                )
                                
                                if ticker_response.status_code == 200:
                                    ticker_data = ticker_response.json()
                                    price = float(ticker_data.get('price', 0))
                                    usd_value = balance * price
                            except Exception as e:
                                self.logger.warning(f"Error getting USD value for {currency}: {str(e)}")
                        
                        # Update total value
                        total_value_usd += usd_value
                        
                        # Store balance info
                        balances[currency] = {
                            'balance': balance,
                            'hold': hold,
                            'available': available,
                            'usd_value': usd_value
                        }
                    
                    # Create account summary
                    account_info = {
                        'balances': balances,
                        'total_value_usd': total_value_usd,
                        'timestamp': now.isoformat()
                    }
                    
                    # Update cache
                    with self._lock:
                        self._account_cache = account_info
                        self._last_account_update = now
                        
                    return account_info
                else:
                    self.logger.error(f"Failed to get account balances: {response.status_code}, {response.text}")
                    return self._account_cache
                    
            except Exception as e:
                self.logger.error(f"Error getting account balances: {str(e)}")
                return self._account_cache
        
        # Return cached data
        return self._account_cache
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List[Dict]: List of positions
        """
        # Get account balances first
        account_info = self.get_account_balances()
        balances = account_info.get('balances', {})
        
        # Convert balances to positions format
        positions = []
        
        for currency, balance_info in balances.items():
            balance = balance_info.get('balance', 0)
            
            # Only include non-zero balances
            if balance > 0:
                # For crypto, only the basic quote is available
                symbol = f"{currency}-USD"
                
                # Get current price
                current_price = 1.0  # Default for USD
                if currency != 'USD':
                    quote = self.get_quote(symbol)
                    current_price = quote.get('last', 1.0)
                
                # Calculate position values
                market_value = balance * current_price
                
                position = {
                    'symbol': symbol,
                    'quantity': balance,
                    'average_price': 0.0,  # Not available from Coinbase API
                    'current_price': current_price,
                    'market_value': market_value,
                    'unrealized_pl': 0.0,  # Not available without cost basis
                    'realized_pl': 0.0,    # Not available without transaction history
                    'currency': currency,
                    'position_type': 'crypto'
                }
                
                positions.append(position)
        
        # Update cache
        with self._lock:
            self._position_cache = positions
        
        return positions
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current orders.
        
        Returns:
            List[Dict]: List of orders
        """
        try:
            response = self._send_request("GET", "/orders", is_auth=True)
            
            if response.status_code == 200:
                orders_data = response.json()
                
                # Process orders
                orders = []
                for order_data in orders_data:
                    order = {
                        'order_id': order_data.get('id'),
                        'symbol': order_data.get('product_id'),
                        'side': order_data.get('side'),
                        'quantity': float(order_data.get('size', 0)),
                        'filled_quantity': float(order_data.get('filled_size', 0)),
                        'order_type': order_data.get('type'),
                        'price': float(order_data.get('price', 0)) if order_data.get('price') else None,
                        'stop_price': float(order_data.get('stop_price', 0)) if order_data.get('stop_price') else None,
                        'time_in_force': order_data.get('time_in_force', 'GTC'),
                        'status': order_data.get('status', 'open'),
                        'created_at': order_data.get('created_at'),
                        'updated_at': order_data.get('done_at')
                    }
                    orders.append(order)
                
                # Update cache
                with self._lock:
                    self._order_cache = orders
                
                return orders
            else:
                self.logger.error(f"Failed to get orders: {response.status_code}, {response.text}")
                return self._order_cache
                
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return self._order_cache
    
    def place_equity_order(self, 
                          symbol: str, 
                          quantity: float, 
                          side: str, 
                          order_type: str, 
                          time_in_force: str = 'GTC', 
                          limit_price: float = None, 
                          stop_price: float = None, 
                          expected_price: float = None) -> Dict[str, Any]:
        """
        Place a cryptocurrency order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            quantity: Quantity to trade
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit')
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (for stop orders)
            expected_price: Expected execution price (for analysis)
            
        Returns:
            Dict: Order result with ID and status
        """
        # Validate inputs
        if order_type.lower() == 'limit' and limit_price is None:
            raise ValueError("Limit price is required for limit orders")
        
        # Prepare order data
        order_data = {
            'product_id': symbol,
            'side': side.lower(),
            'type': order_type.lower(),
            'size': str(quantity)
        }
        
        # Add price for limit orders
        if order_type.lower() == 'limit':
            order_data['price'] = str(limit_price)
            order_data['time_in_force'] = time_in_force.upper()
        
        # Add stop price if provided
        if stop_price is not None:
            order_data['stop'] = 'loss' if side.lower() == 'sell' else 'entry'
            order_data['stop_price'] = str(stop_price)
        
        try:
            # Send order request
            response = self._send_request("POST", "/orders", data=order_data, is_auth=True)
            
            if response.status_code == 200:
                order_response = response.json()
                
                # Create standardized order result
                order_result = {
                    'order_id': order_response.get('id'),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'filled_quantity': float(order_response.get('filled_size', 0)),
                    'order_type': order_type,
                    'price': limit_price,
                    'stop_price': stop_price,
                    'time_in_force': time_in_force,
                    'status': order_response.get('status', 'open'),
                    'created_at': order_response.get('created_at'),
                    'message': 'Order submitted successfully'
                }
                
                # Publish event
                if self.event_bus:
                    self.event_bus.publish(Event(
                        type=EventType.ORDER_ACKNOWLEDGED,
                        data={
                            'order_id': order_result['order_id'],
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'broker_id': self.broker_id
                        }
                    ))
                
                return order_result
            else:
                error_msg = f"Failed to place order: {response.status_code}, {response.text}"
                self.logger.error(error_msg)
                
                if self.event_bus:
                    self.event_bus.publish(Event(
                        type=EventType.ORDER_REJECTED,
                        data={
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'reason': error_msg,
                            'broker_id': self.broker_id
                        }
                    ))
                
                return {
                    'order_id': None,
                    'status': 'rejected',
                    'message': error_msg
                }
                
        except Exception as e:
            error_msg = f"Error placing order: {str(e)}"
            self.logger.error(error_msg)
            
            if self.event_bus:
                self.event_bus.publish(Event(
                    type=EventType.ORDER_REJECTED,
                    data={
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'reason': error_msg,
                        'broker_id': self.broker_id
                    }
                ))
            
            return {
                'order_id': None,
                'status': 'rejected',
                'message': error_msg
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Order status details
        """
        try:
            response = self._send_request("GET", f"/orders/{order_id}", is_auth=True)
            
            if response.status_code == 200:
                order_data = response.json()
                
                # Create standardized order status
                order_status = {
                    'order_id': order_data.get('id'),
                    'symbol': order_data.get('product_id'),
                    'side': order_data.get('side'),
                    'quantity': float(order_data.get('size', 0)),
                    'filled_quantity': float(order_data.get('filled_size', 0)),
                    'remaining_quantity': float(order_data.get('size', 0)) - float(order_data.get('filled_size', 0)),
                    'order_type': order_data.get('type'),
                    'price': float(order_data.get('price', 0)) if order_data.get('price') else None,
                    'stop_price': float(order_data.get('stop_price', 0)) if order_data.get('stop_price') else None,
                    'time_in_force': order_data.get('time_in_force', 'GTC'),
                    'status': order_data.get('status', 'open'),
                    'created_at': order_data.get('created_at'),
                    'updated_at': order_data.get('done_at')
                }
                
                return order_status
            else:
                self.logger.error(f"Failed to get order status: {response.status_code}, {response.text}")
                return {
                    'order_id': order_id,
                    'status': 'unknown',
                    'message': f"Failed to get order status: {response.text}"
                }
                
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")
            return {
                'order_id': order_id,
                'status': 'unknown',
                'message': f"Error: {str(e)}"
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Cancellation result
        """
        try:
            response = self._send_request("DELETE", f"/orders/{order_id}", is_auth=True)
            
            if response.status_code == 200:
                # Create standardized cancellation result
                result = {
                    'order_id': order_id,
                    'status': 'cancelled',
                    'message': 'Order cancelled successfully'
                }
                
                # Publish event
                if self.event_bus:
                    self.event_bus.publish(Event(
                        type=EventType.ORDER_CANCELLED,
                        data={
                            'order_id': order_id,
                            'broker_id': self.broker_id
                        }
                    ))
                
                return result
            else:
                error_msg = f"Failed to cancel order: {response.status_code}, {response.text}"
                self.logger.error(error_msg)
                return {
                    'order_id': order_id,
                    'status': 'failed',
                    'message': error_msg
                }
                
        except Exception as e:
            error_msg = f"Error cancelling order: {str(e)}"
            self.logger.error(error_msg)
            return {
                'order_id': order_id,
                'status': 'failed',
                'message': error_msg
            }
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            
        Returns:
            Dict: Quote information
        """
        # Check if we need to refresh the cache
        now = datetime.now()
        last_update = self._last_quote_update.get(symbol)
        
        if last_update is None or (now - last_update).total_seconds() > 5:
            try:
                response = self._send_request("GET", f"/products/{symbol}/ticker", is_auth=False)
                
                if response.status_code == 200:
                    ticker_data = response.json()
                    
                    # Create standardized quote
                    quote = {
                        'symbol': symbol,
                        'bid': float(ticker_data.get('bid', 0)),
                        'ask': float(ticker_data.get('ask', 0)),
                        'last': float(ticker_data.get('price', 0)),
                        'volume': float(ticker_data.get('volume', 0)),
                        'timestamp': ticker_data.get('time')
                    }
                    
                    # Update cache
                    with self._lock:
                        self._market_data_cache[symbol] = quote
                        self._last_quote_update[symbol] = now
                        
                    return quote
                else:
                    self.logger.error(f"Failed to get quote for {symbol}: {response.status_code}, {response.text}")
                    return self._market_data_cache.get(symbol, {
                        'symbol': symbol,
                        'bid': 0,
                        'ask': 0,
                        'last': 0,
                        'volume': 0,
                        'timestamp': now.isoformat()
                    })
                    
            except Exception as e:
                self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
                return self._market_data_cache.get(symbol, {
                    'symbol': symbol,
                    'bid': 0,
                    'ask': 0,
                    'last': 0,
                    'volume': 0,
                    'timestamp': now.isoformat()
                })
        
        # Return cached data
        return self._market_data_cache.get(symbol, {
            'symbol': symbol,
            'bid': 0,
            'ask': 0,
            'last': 0,
            'volume': 0,
            'timestamp': now.isoformat()
        })
    
    def get_historical_data(self, 
                           symbol: str, 
                           interval: str, 
                           start_date: datetime, 
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get historical market data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date
            end_date: End date (defaults to current time)
            
        Returns:
            List[Dict]: Historical data points
        """
        # Set default end date
        if end_date is None:
            end_date = datetime.now()
        
        # Convert interval to Coinbase granularity
        granularity_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '1d': 86400
        }
        
        granularity = granularity_map.get(interval, 3600)
        
        # Calculate time range
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Coinbase has a limit of 300 candles per request
        # We need to make multiple requests for longer periods
        candles = []
        current_start = start_timestamp
        current_end = min(current_start + (granularity * 300), end_timestamp)
        
        while current_start < end_timestamp:
            try:
                params = {
                    'start': datetime.fromtimestamp(current_start).isoformat(),
                    'end': datetime.fromtimestamp(current_end).isoformat(),
                    'granularity': granularity
                }
                
                response = self._send_request("GET", f"/products/{symbol}/candles", params=params, is_auth=False)
                
                if response.status_code == 200:
                    candle_data = response.json()
                    
                    # Process candles (Coinbase returns in reverse order)
                    for candle in candle_data:
                        # Coinbase format: [timestamp, low, high, open, close, volume]
                        timestamp, low, high, open_price, close, volume = candle
                        
                        candles.append({
                            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume
                        })
                else:
                    self.logger.error(f"Failed to get historical data: {response.status_code}, {response.text}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error getting historical data: {str(e)}")
                break
                
            # Move to next time range
            current_start = current_end
            current_end = min(current_start + (granularity * 300), end_timestamp)
            
            # Respect rate limits
            time.sleep(0.5)
        
        # Sort candles by timestamp
        candles.sort(key=lambda x: x['timestamp'])
        
        return candles
    
    def name(self) -> str:
        """Get broker name."""
        if self._sandbox:
            return "Coinbase Pro [SANDBOX]"
        return "Coinbase Pro"
    
    def status(self) -> str:
        """Get broker connection status."""
        return self._status
    
    def supports_extended_hours(self) -> bool:
        """
        Check if broker supports extended hours trading.
        
        Crypto markets operate 24/7, so this always returns True.
        """
        return True
    
    def supports_fractional_shares(self) -> bool:
        """
        Check if broker supports fractional shares.
        
        Cryptocurrencies are divisible, so this returns True.
        """
        return True
    
    def api_calls_remaining(self) -> Optional[int]:
        """Get number of API calls remaining (rate limiting)."""
        return self._rate_limit_remaining
    
    def get_broker_time(self) -> datetime:
        """Get current time from broker's servers."""
        try:
            response = self._send_request("GET", "/time", is_auth=False)
            
            if response.status_code == 200:
                time_data = response.json()
                iso_time = time_data.get('iso')
                return datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
            else:
                return datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error getting broker time: {str(e)}")
            return datetime.now()
    
    def needs_refresh(self) -> bool:
        """Check if broker connection needs refresh."""
        # Check connection every 30 minutes
        now = datetime.now()
        if (now - self._last_connection_check).total_seconds() > 1800:
            self._last_connection_check = now
            return True
        
        return False
    
    def refresh_connection(self) -> bool:
        """Refresh broker connection (re-authenticate)."""
        try:
            self._initialize_connection()
            return self._status == "connected"
        except Exception as e:
            self.logger.error(f"Error refreshing connection: {str(e)}")
            return False
