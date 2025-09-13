#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Broker Client

This module implements a brokerage client adapter for Coinbase,
using the CoinbaseBroker implementation.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .brokerage_client import (
    BrokerageClient, 
    BrokerConnectionStatus,
    OrderType, 
    OrderSide,
    TimeInForce,
    BrokerAPIError,
    BROKER_IMPLEMENTATIONS
)
from .coinbase_broker import CoinbaseBroker

# Configure logging
logger = logging.getLogger(__name__)

class CoinbaseBrokerageClient(BrokerageClient):
    """
    Coinbase adapter that conforms to the BrokerageClient interface.
    
    This adapter wraps the CoinbaseBroker implementation and adapts it
    to the standardized BrokerageClient interface.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the Coinbase broker client.
        
        Args:
            config_path: Path to configuration file (optional)
            **kwargs: Additional configuration options including:
                - api_key: Coinbase API key
                - api_secret: Coinbase API secret
                - passphrase: Coinbase API passphrase (optional)
                - sandbox: Whether to use sandbox environment (default: False)
        """
        super().__init__(config_path, **kwargs)
        
        # Extract config from kwargs or config file
        api_key = kwargs.get('api_key') or self.config.get('api_key')
        api_secret = kwargs.get('api_secret') or self.config.get('api_secret')
        passphrase = kwargs.get('passphrase') or self.config.get('passphrase')
        sandbox = kwargs.get('sandbox', False) or self.config.get('sandbox', False)
        
        if not api_key or not api_secret:
            raise ValueError("Coinbase API key and secret are required")
        
        # Create the underlying broker implementation
        self._broker = CoinbaseBroker(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            sandbox=sandbox
        )
        
        # Set available order types and time in force options
        self.available_order_types = {
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.STOP_LIMIT
        }
        
        self.available_time_in_force = {
            TimeInForce.DAY,
            TimeInForce.GTC,
            TimeInForce.IOC,
            TimeInForce.FOK
        }
        
        # Connect if credentials are available
        if api_key and api_secret:
            try:
                self.connect()
            except Exception as e:
                logger.error(f"Failed to connect to Coinbase: {str(e)}")
                self.connection_status = BrokerConnectionStatus.DISCONNECTED
    
    def connect(self) -> bool:
        """
        Establish connection to the Coinbase API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self._broker._initialize_connection()
            connected = self._broker._status == "connected"
            
            if connected:
                self.connection_status = BrokerConnectionStatus.CONNECTED
                self.connection_errors = []
                logger.info("Connected to Coinbase API")
            else:
                self.connection_status = BrokerConnectionStatus.DISCONNECTED
                logger.error("Failed to connect to Coinbase API")
            
            return connected
            
        except Exception as e:
            self._handle_connection_error(e)
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Coinbase API.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        # Coinbase API is stateless, no need to actually disconnect
        self.connection_status = BrokerConnectionStatus.DISCONNECTED
        logger.info("Disconnected from Coinbase API")
        return True
    
    def check_connection(self) -> BrokerConnectionStatus:
        """
        Check the current connection status.
        
        Returns:
            BrokerConnectionStatus: Current connection status
        """
        try:
            # Use a lightweight API call to check connection
            self._broker.get_broker_time()
            
            if self._broker._status == "connected":
                self.connection_status = BrokerConnectionStatus.CONNECTED
                self.connection_errors = []
            else:
                self.connection_status = BrokerConnectionStatus.DISCONNECTED
                
        except Exception as e:
            self._handle_connection_error(e)
        
        self.last_connection_check = datetime.now()
        return self.connection_status
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            account_info = self._broker.get_account_balances()
            return {
                'account_id': self._broker.broker_id,
                'status': 'active',
                'account_type': 'crypto',
                'balances': account_info.get('balances', {}),
                'total_value': account_info.get('total_value_usd', 0),
                'buying_power': account_info.get('total_value_usd', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        try:
            return self._broker.get_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            status: Filter by order status (optional)
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        try:
            orders = self._broker.get_orders()
            
            # Filter by status if provided
            if status:
                orders = [order for order in orders if order.get('status') == status]
                
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            return self._broker.get_order_status(order_id)
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            return {
                'order_id': order_id,
                'status': 'unknown',
                'error': str(e)
            }
    
    def place_order(self, 
                  symbol: str, 
                  side: Union[OrderSide, str], 
                  quantity: float, 
                  order_type: Union[OrderType, str],
                  limit_price: Optional[float] = None,
                  time_in_force: Union[TimeInForce, str] = TimeInForce.DAY,
                  extended_hours: bool = False,
                  stop_price: Optional[float] = None,
                  trail_price: Optional[float] = None,
                  trail_percent: Optional[float] = None,
                  client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (buy, sell, sell_short, buy_to_cover)
            quantity: Order quantity
            order_type: Type of order (market, limit, stop, etc.)
            limit_price: Limit price (required for limit and stop_limit orders)
            time_in_force: Order lifetime (day, gtc, etc.)
            extended_hours: Whether the order can execute during extended hours
            stop_price: Stop price (required for stop and stop_limit orders)
            trail_price: Trailing stop price
            trail_percent: Trailing stop percentage
            client_order_id: Client-specified order ID
            
        Returns:
            Dict[str, Any]: Order result
        """
        try:
            # Convert enums to strings if needed
            if isinstance(side, OrderSide):
                side = side.value
            
            if isinstance(order_type, OrderType):
                order_type = order_type.value
                
            if isinstance(time_in_force, TimeInForce):
                time_in_force = time_in_force.value
            
            # Validate order type and select appropriate type
            has_limit_price = limit_price is not None
            has_stop_price = stop_price is not None
            selected_order_type = self._select_order_type(
                order_type, has_limit_price, has_stop_price
            ).value
            
            # Place the order with the underlying broker
            order_result = self._broker.place_equity_order(
                symbol=symbol,
                quantity=float(quantity),
                side=side,
                order_type=selected_order_type,
                time_in_force=self._select_time_in_force(time_in_force).value,
                limit_price=limit_price,
                stop_price=stop_price,
                expected_price=limit_price or None
            )
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            
            # Return standardized error response
            return {
                'order_id': None,
                'status': 'rejected',
                'message': f"Error placing order: {str(e)}",
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict[str, Any]: Cancellation status
        """
        try:
            return self._broker.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return {
                'order_id': order_id,
                'status': 'failed',
                'message': f"Error cancelling order: {str(e)}"
            }
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get a real-time quote for a symbol.
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Dict[str, Any]: Quote data
        """
        try:
            return self._broker.get_quote(symbol)
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'last': None,
                'bid': None,
                'ask': None,
                'error': str(e)
            }
    
    def get_bars(self, 
              symbol: str, 
              timeframe: str = '1d', 
              start: Optional[datetime] = None, 
              end: Optional[datetime] = None, 
              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical price bars.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 1d)
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars to return
            
        Returns:
            List[Dict[str, Any]]: List of bars
        """
        try:
            if start is None:
                start = datetime.now() - datetime.timedelta(days=30)
                
            if end is None:
                end = datetime.now()
                
            return self._broker.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                start_date=start,
                end_date=end
            )
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {str(e)}")
            return []
    
    def get_market_hours(self, market: str = "equity") -> Dict[str, Any]:
        """
        Get market hours.
        
        Args:
            market: Market to get hours for (equity, options, etc.)
            
        Returns:
            Dict[str, Any]: Market hours information
        """
        # Crypto markets are always open
        return {
            'market': 'crypto',
            'is_open': True,
            'next_open': datetime.now().isoformat(),
            'next_close': (datetime.now() + datetime.timedelta(days=9999)).isoformat(),
            'session': 'regular',
            'timezone': 'UTC'
        }
    
    def get_calendar(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """
        Get market calendar for date range.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List[Dict[str, Any]]: List of market days
        """
        # Crypto markets are open every day
        calendar = []
        
        current_date = start
        while current_date <= end:
            calendar.append({
                'date': current_date.date().isoformat(),
                'is_open': True,
                'open_time': '00:00',
                'close_time': '23:59',
                'session': 'regular'
            })
            current_date += datetime.timedelta(days=1)
            
        return calendar
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get full ticker information for a symbol.
        
        Args:
            symbol: Symbol to get ticker for
            
        Returns:
            Dict[str, Any]: Ticker information
        """
        try:
            quote = self._broker.get_quote(symbol)
            
            # Get 24h data to calculate additional metrics
            end = datetime.now()
            start = end - datetime.timedelta(days=1)
            
            bars = self._broker.get_historical_data(
                symbol=symbol,
                interval='1h',
                start_date=start,
                end_date=end
            )
            
            # Calculate additional metrics if possible
            high_24h = 0
            low_24h = float('inf')
            volume_24h = 0
            
            for bar in bars:
                high_24h = max(high_24h, bar.get('high', 0))
                low_24h = min(low_24h, bar.get('low', float('inf')))
                volume_24h += bar.get('volume', 0)
            
            if low_24h == float('inf'):
                low_24h = 0
                
            # Construct response
            return {
                'symbol': symbol,
                'last': quote.get('last'),
                'bid': quote.get('bid'),
                'ask': quote.get('ask'),
                'volume': quote.get('volume'),
                'high_24h': high_24h,
                'low_24h': low_24h,
                'volume_24h': volume_24h,
                'timestamp': quote.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e)
            }


# Register this implementation
BROKER_IMPLEMENTATIONS['coinbase'] = CoinbaseBrokerageClient
