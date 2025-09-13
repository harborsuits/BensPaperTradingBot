"""
Alpaca Brokerage API Client

This module implements the BrokerageClient interface for Alpaca API,
providing a standardized way to interact with Alpaca for trading stocks,
options, and crypto.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import threading

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
except ImportError:
    raise ImportError("Please install alpaca-trade-api package to use the Alpaca client")

from .brokerage_client import (
    BrokerageClient, 
    OrderType, 
    OrderSide, 
    TimeInForce, 
    BrokerConnectionStatus,
    BrokerAPIError,
    BrokerAuthError,
    BrokerConnectionError,
    OrderExecutionError
)

# Configure logging
logger = logging.getLogger(__name__)

class AlpacaClient(BrokerageClient):
    """
    Client for Alpaca API that implements the BrokerageClient interface.
    Provides functionality for account management, trading, and market data.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_secret: Optional[str] = None,
                base_url: Optional[str] = None,
                data_url: Optional[str] = None,
                paper_trading: bool = True,
                config_path: Optional[str] = None,
                **kwargs):
        """
        Initialize the Alpaca client.
        
        Args:
            api_key: Alpaca API key (optional if provided in config)
            api_secret: Alpaca API secret (optional if provided in config)
            base_url: Base URL for API (optional, defaults to paper or live URL)
            data_url: Data URL for API (optional)
            paper_trading: Whether to use paper trading (default: True)
            config_path: Path to configuration file (optional)
            **kwargs: Additional parameters
        """
        super().__init__(config_path=config_path, **kwargs)
        
        # Extract credentials from config if not provided directly
        if not api_key and self.config:
            api_key = self.config.get('alpaca', {}).get('api_key')
        if not api_secret and self.config:
            api_secret = self.config.get('alpaca', {}).get('api_secret')
        
        # Try environment variables if still not found
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise BrokerAuthError("Alpaca API credentials not provided", "Alpaca")
        
        # Set API URLs
        self.paper_trading = paper_trading
        if not base_url:
            self.base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        else:
            self.base_url = base_url
            
        self.data_url = data_url or 'https://data.alpaca.markets'
        
        # Set available order types and time in force options
        self.available_order_types = {
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.STOP_LIMIT,
            OrderType.TRAILING_STOP
        }
        
        self.available_time_in_force = {
            TimeInForce.DAY,
            TimeInForce.GTC,
            TimeInForce.IOC,
            TimeInForce.FOK
        }
        
        # Initialize API client
        self.api = None
        self.connection_monitoring_thread = None
        self.connection_monitoring_active = False
        self.connection_check_interval = kwargs.get('connection_check_interval', 60)  # seconds
        
        # Initialize additional state
        self.account_cache = {}
        self.account_cache_expiry = datetime.now()
        self.account_cache_ttl = timedelta(seconds=kwargs.get('account_cache_ttl', 5))
        
        # Connect if requested
        if kwargs.get('auto_connect', True):
            self.connect()
    
    def connect(self) -> bool:
        """
        Establish connection to the Alpaca API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Alpaca API ({self.base_url})")
            self.connection_status = BrokerConnectionStatus.AUTHENTICATING
            
            # Initialize API client
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection by getting account info
            _ = self.api.get_account()
            
            # Connection successful
            self.connection_status = BrokerConnectionStatus.CONNECTED
            self.last_connection_check = datetime.now()
            
            # Start connection monitoring if not already running
            self._start_connection_monitoring()
            
            logger.info("Successfully connected to Alpaca API")
            return True
            
        except APIError as e:
            error_msg = f"Failed to connect to Alpaca API: {str(e)}"
            self.connection_status = BrokerConnectionStatus.DISCONNECTED
            logger.error(error_msg)
            
            if "authentication failed" in str(e).lower():
                raise BrokerAuthError(str(e), "Alpaca")
            else:
                raise BrokerAPIError(str(e), "Alpaca")
            
        except Exception as e:
            error_msg = f"Failed to connect to Alpaca API: {str(e)}"
            self.connection_status = BrokerConnectionStatus.DISCONNECTED
            logger.error(error_msg)
            
            raise BrokerConnectionError(str(e), "Alpaca")
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Alpaca API.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        # Stop connection monitoring
        self._stop_connection_monitoring()
        
        # Clear API client
        self.api = None
        self.connection_status = BrokerConnectionStatus.DISCONNECTED
        
        logger.info("Disconnected from Alpaca API")
        return True
    
    def check_connection(self) -> BrokerConnectionStatus:
        """
        Check current connection status to Alpaca API.
        
        Returns:
            BrokerConnectionStatus: Current connection status
        """
        # If we haven't initialized the API client, we're disconnected
        if self.api is None:
            return BrokerConnectionStatus.DISCONNECTED
        
        try:
            # Test connection by getting account info
            _ = self.api.get_account()
            
            # Connection is good
            self.connection_status = BrokerConnectionStatus.CONNECTED
            self.last_connection_check = datetime.now()
            
            return BrokerConnectionStatus.CONNECTED
            
        except Exception as e:
            error_msg = f"Connection check failed: {str(e)}"
            logger.error(error_msg)
            
            self._handle_connection_error(e)
            return self.connection_status
    
    def _start_connection_monitoring(self) -> None:
        """Start background thread for connection monitoring."""
        if self.connection_monitoring_thread is not None and self.connection_monitoring_thread.is_alive():
            logger.debug("Connection monitoring already running")
            return
        
        self.connection_monitoring_active = True
        self.connection_monitoring_thread = threading.Thread(
            target=self._connection_monitoring_loop,
            daemon=True,
            name="alpaca-connection-monitor"
        )
        self.connection_monitoring_thread.start()
        logger.info("Started Alpaca connection monitoring")
    
    def _stop_connection_monitoring(self) -> None:
        """Stop connection monitoring thread."""
        self.connection_monitoring_active = False
        
        if self.connection_monitoring_thread and self.connection_monitoring_thread.is_alive():
            self.connection_monitoring_thread.join(timeout=2)
            logger.info("Stopped Alpaca connection monitoring")
    
    def _connection_monitoring_loop(self) -> None:
        """Connection monitoring background thread."""
        while self.connection_monitoring_active:
            try:
                # Check if it's time to check the connection
                if (self.last_connection_check is None or 
                    (datetime.now() - self.last_connection_check).total_seconds() >= self.connection_check_interval):
                    
                    status = self.check_connection()
                    logger.debug(f"Alpaca connection status: {status.value}")
                
            except Exception as e:
                logger.error(f"Error in connection monitoring: {str(e)}")
            
            # Sleep before next check
            time.sleep(10)  # Check every 10 seconds if it's time for a full check
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from Alpaca.
        
        Returns:
            Dict[str, Any]: Account information
        """
        # Check cache first
        if (self.account_cache and 
            datetime.now() < self.account_cache_expiry):
            return self.account_cache
        
        try:
            # Get account info
            account = self.api.get_account()
            
            # Format response
            account_info = {
                'account_id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'daytrade_count': int(account.daytrade_count),
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'status': account.status,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'updated_at': account.updated_at,
                'pattern_day_trader': account.pattern_day_trader,
                'multiplier': account.multiplier
            }
            
            # Update cache
            self.account_cache = account_info
            self.account_cache_expiry = datetime.now() + self.account_cache_ttl
            
            return account_info
            
        except APIError as e:
            logger.error(f"Error retrieving account info: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving account info: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from Alpaca.
        
        Returns:
            List[Dict[str, Any]]: List of positions
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
                    'change_today': float(position.change_today),
                    'exchange': position.exchange
                }
                position_list.append(position_details)
            
            return position_list
            
        except APIError as e:
            logger.error(f"Error retrieving positions: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving positions: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders from Alpaca.
        
        Args:
            status: Filter by order status (optional)
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        try:
            # Get orders with optional status filter
            if status:
                orders = self.api.list_orders(status=status)
            else:
                orders = self.api.list_orders()
            
            # Format response
            order_list = []
            for order in orders:
                order_details = {
                    'id': order.id,
                    'client_order_id': order.client_order_id,
                    'symbol': order.symbol,
                    'quantity': float(order.qty),
                    'filled_quantity': float(order.filled_qty) if order.filled_qty else 0,
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'created_at': order.created_at,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'time_in_force': order.time_in_force,
                    'extended_hours': order.extended_hours,
                    'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
                }
                order_list.append(order_details)
            
            return order_list
            
        except APIError as e:
            logger.error(f"Error retrieving orders: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving orders: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order information
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
                'filled_quantity': float(order.filled_qty) if order.filled_qty else 0,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'time_in_force': order.time_in_force,
                'extended_hours': order.extended_hours,
                'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
            return order_details
            
        except APIError as e:
            logger.error(f"Error retrieving order {order_id}: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving order {order_id}: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
    
    def place_order(self, 
                  symbol: str, 
                  side: Union[OrderSide, str], 
                  quantity: float, 
                  order_type: Union[OrderType, str] = OrderType.MARKET,
                  time_in_force: Union[TimeInForce, str] = TimeInForce.DAY,
                  limit_price: Optional[float] = None, 
                  stop_price: Optional[float] = None,
                  trail_price: Optional[float] = None,
                  trail_percent: Optional[float] = None,
                  client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order with Alpaca.
        
        Args:
            symbol: Symbol to trade
            side: Order side (buy, sell, sell_short, buy_to_cover)
            quantity: Order quantity
            order_type: Type of order (market, limit, stop, etc.)
            time_in_force: Time in force for the order (day, gtc, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            trail_price: Trailing amount for trailing stop orders
            trail_percent: Trailing percentage for trailing stop orders
            client_order_id: Client-specified order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Convert enums to strings if needed
            if isinstance(side, OrderSide):
                side = side.value
            
            if isinstance(order_type, OrderType):
                order_type = order_type.value
            
            if isinstance(time_in_force, TimeInForce):
                time_in_force = time_in_force.value
            
            # Select appropriate order type based on available options and parameters
            has_limit = limit_price is not None
            has_stop = stop_price is not None
            has_trail = trail_price is not None or trail_percent is not None
            
            # Handle trailing stop orders
            if order_type == OrderType.TRAILING_STOP.value and has_trail:
                if not (trail_price or trail_percent):
                    raise OrderExecutionError("Either trail_price or trail_percent must be specified for trailing stop orders", "Alpaca")
                
                if trail_price and trail_percent:
                    logger.warning("Both trail_price and trail_percent provided, using trail_price")
                
                try:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        type='market',
                        time_in_force=time_in_force,
                        client_order_id=client_order_id,
                        order_class='oto',
                        stop_loss={
                            'trail_price': trail_price if trail_price else None,
                            'trail_percent': trail_percent if trail_percent else None
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to submit trailing stop order: {str(e)}")
                    # Fall back to regular market order
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        type='market',
                        time_in_force=time_in_force,
                        client_order_id=client_order_id
                    )
            
            # Handle other order types
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    client_order_id=client_order_id
                )
            
            # Format response
            order_details = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty) if order.filled_qty else 0,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'time_in_force': order.time_in_force,
                'extended_hours': order.extended_hours,
                'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
            logger.info(f"Order placed: {symbol} {side} {quantity} @ {order_type}")
            return order_details
            
        except APIError as e:
            logger.error(f"Error placing order: {str(e)}")
            raise OrderExecutionError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error placing order: {str(e)}")
            raise OrderExecutionError(str(e), "Alpaca")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order with Alpaca.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict[str, Any]: Cancellation status
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
            
        except APIError as e:
            if "404" in str(e):
                logger.warning(f"Order {order_id} not found or already cancelled")
                return {
                    'success': False,
                    'order_id': order_id,
                    'message': "Order not found or already cancelled"
                }
            
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise OrderExecutionError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error cancelling order {order_id}: {str(e)}")
            raise OrderExecutionError(str(e), "Alpaca")
    
    def modify_order(self, 
                   order_id: str,
                   quantity: Optional[float] = None,
                   order_type: Optional[Union[OrderType, str]] = None,
                   time_in_force: Optional[Union[TimeInForce, str]] = None,
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing order with Alpaca.
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            order_type: New order type
            time_in_force: New time in force
            limit_price: New limit price
            stop_price: New stop price
            
        Returns:
            Dict[str, Any]: Modified order information
        """
        try:
            # Alpaca doesn't support direct order modification, so we need to:
            # 1. Get the current order
            # 2. Cancel it
            # 3. Place a new order with the updated parameters
            
            # Get current order
            current_order = self.get_order(order_id)
            
            # Cancel the existing order
            cancel_result = self.cancel_order(order_id)
            if not cancel_result['success']:
                return {
                    'success': False,
                    'order_id': order_id,
                    'message': f"Failed to cancel order for modification: {cancel_result['message']}"
                }
            
            # Update parameters with new values or keep current ones
            new_quantity = quantity if quantity is not None else float(current_order['quantity'])
            
            new_order_type = order_type
            if new_order_type is None:
                new_order_type = current_order['type']
            elif isinstance(new_order_type, OrderType):
                new_order_type = new_order_type.value
            
            new_time_in_force = time_in_force
            if new_time_in_force is None:
                new_time_in_force = current_order['time_in_force']
            elif isinstance(new_time_in_force, TimeInForce):
                new_time_in_force = new_time_in_force.value
            
            new_limit_price = limit_price if limit_price is not None else current_order['price']
            new_stop_price = stop_price if stop_price is not None else current_order['stop_price']
            
            # Place new order with updated parameters
            new_order = self.place_order(
                symbol=current_order['symbol'],
                side=current_order['side'],
                quantity=new_quantity,
                order_type=new_order_type,
                time_in_force=new_time_in_force,
                limit_price=new_limit_price,
                stop_price=new_stop_price,
                client_order_id=f"replace_{order_id}"
            )
            
            # Return the new order with additional context
            new_order['replaced_order_id'] = order_id
            new_order['success'] = True
            new_order['message'] = "Order successfully modified"
            
            logger.info(f"Order {order_id} modified, new order ID: {new_order['id']}")
            return new_order
            
        except APIError as e:
            logger.error(f"Error modifying order {order_id}: {str(e)}")
            raise OrderExecutionError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error modifying order {order_id}: {str(e)}")
            raise OrderExecutionError(str(e), "Alpaca")
    
    def get_market_hours(self, market: str = "equity") -> Dict[str, Any]:
        """
        Get market hours information from Alpaca.
        
        Args:
            market: Market to get hours for (equity, options, etc.)
            
        Returns:
            Dict[str, Any]: Market hours information
        """
        try:
            # Get calendar for today
            today = datetime.now().strftime('%Y-%m-%d')
            calendar = self.api.get_calendar(start=today, end=today)
            
            if not calendar:
                # No calendar entry means market is likely closed today
                return {
                    'is_open': False,
                    'next_open': None,
                    'next_close': None,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get clock information
            clock = self.api.get_clock()
            
            # Format response
            market_hours = {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat() if clock.next_open else None,
                'next_close': clock.next_close.isoformat() if clock.next_close else None,
                'timestamp': clock.timestamp.isoformat() if clock.timestamp else None,
                'today_open': calendar[0].open.isoformat() if calendar else None,
                'today_close': calendar[0].close.isoformat() if calendar else None
            }
            
            return market_hours
            
        except APIError as e:
            logger.error(f"Error retrieving market hours: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving market hours: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
    
    def is_market_open(self, market: str = "equity") -> bool:
        """
        Check if a market is currently open.
        
        Args:
            market: Market to check (equity, options, etc.)
            
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Get clock information
            clock = self.api.get_clock()
            
            return clock.is_open
            
        except APIError as e:
            logger.error(f"Error checking if market is open: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")
        
        except Exception as e:
            logger.error(f"Unexpected error checking if market is open: {str(e)}")
            raise BrokerAPIError(str(e), "Alpaca")

# Register the implementation
from .brokerage_client import BROKER_IMPLEMENTATIONS
BROKER_IMPLEMENTATIONS['alpaca'] = AlpacaClient 