"""
Alpaca Broker Adapter

Implements the BrokerInterface for Alpaca Markets API.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import uuid

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, APIError

from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.core.events import (
    OrderAcknowledged, OrderPartialFill, OrderFilled, 
    OrderCancelled, OrderRejected, SlippageMetric
)
from trading_bot.event_system.event_bus import EventBus


class AlpacaCredentials:
    """Credentials for Alpaca API access."""
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Initialize Alpaca credentials.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Whether to use paper trading API (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert credentials to dictionary."""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'paper': self.paper
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlpacaCredentials':
        """Create credentials from dictionary."""
        return cls(
            api_key=data.get('api_key', ''),
            api_secret=data.get('api_secret', ''),
            paper=data.get('paper', True)
        )


class AlpacaAdapter(BrokerInterface):
    """Adapter for Alpaca Markets API."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize Alpaca adapter.
        
        Args:
            event_bus: Optional event bus for publishing trading events
        """
        super().__init__(event_bus)
        self.logger = logging.getLogger(__name__)
        self.broker_id = 'alpaca'
        self.api: Optional[REST] = None
        self.account_id: Optional[str] = None
        self.is_paper: bool = True
        self.last_connection_time: Optional[datetime] = None
        self.connection_timeout: int = 3600  # 1 hour in seconds
    
    def connect(self, credentials) -> bool:
        """
        Connect to Alpaca API.
        
        Args:
            credentials: AlpacaCredentials object or dictionary
            
        Returns:
            bool: True if connection successful
        """
        if isinstance(credentials, dict):
            credentials = AlpacaCredentials.from_dict(credentials)
        
        try:
            self.is_paper = credentials.paper
            base_url = 'https://paper-api.alpaca.markets' if self.is_paper else 'https://api.alpaca.markets'
            
            self.api = tradeapi.REST(
                key_id=credentials.api_key,
                secret_key=credentials.api_secret,
                base_url=base_url,
                api_version='v2'
            )
            
            # Test connection by getting account info
            account = self.api.get_account()
            self.account_id = account.id
            self.last_connection_time = datetime.now()
            
            self.logger.info(f"Connected to Alpaca API ({('Paper' if self.is_paper else 'Live')})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            self.api = None
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Alpaca API.
        
        Returns:
            bool: True if disconnection successful
        """
        self.api = None
        self.account_id = None
        self.last_connection_time = None
        return True
    
    def is_connected(self) -> bool:
        """
        Check if connected to Alpaca API.
        
        Returns:
            bool: True if connected
        """
        if self.api is None or self.account_id is None:
            return False
        
        try:
            # Check if connection is stale
            if self.last_connection_time and (datetime.now() - self.last_connection_time).total_seconds() > self.connection_timeout:
                return False
            
            # Make a lightweight call to verify connection
            self.api.get_clock()
            return True
        except Exception:
            return False
    
    def needs_refresh(self) -> bool:
        """
        Check if connection needs refresh.
        
        Returns:
            bool: True if connection needs refresh
        """
        if not self.is_connected():
            return True
        
        if self.last_connection_time and (datetime.now() - self.last_connection_time).total_seconds() > self.connection_timeout:
            return True
        
        return False
    
    def refresh_connection(self) -> bool:
        """
        Refresh connection.
        
        Returns:
            bool: True if refresh successful
        """
        if self.api is None:
            return False
        
        try:
            # Just make a simple call to verify connection
            self.api.get_clock()
            self.last_connection_time = datetime.now()
            return True
        except Exception:
            return False
    
    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        Returns:
            bool: True if market is open
        """
        if not self.is_connected():
            return False
        
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.error(f"Error checking market status: {str(e)}")
            return False
    
    def get_next_market_open(self) -> datetime:
        """
        Get next market open time.
        
        Returns:
            datetime: Next market open time
        """
        if not self.is_connected():
            return datetime.now() + timedelta(days=1)
        
        try:
            clock = self.api.get_clock()
            if clock.is_open:
                # Get the next trading day
                calendar = self.api.get_calendar(start=datetime.now().date(), end=datetime.now().date() + timedelta(days=7))
                for day in calendar:
                    if day.date > datetime.now().date():
                        return datetime.combine(day.date, day.open)
                
                # Fallback
                return datetime.now() + timedelta(days=1)
            else:
                return clock.next_open.datetime
        except Exception as e:
            self.logger.error(f"Error getting next market open: {str(e)}")
            return datetime.now() + timedelta(days=1)
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get trading hours.
        
        Returns:
            Dict: Trading hours information
        """
        if not self.is_connected():
            return {}
        
        try:
            clock = self.api.get_clock()
            calendar = self.api.get_calendar(start=datetime.now().date(), end=datetime.now().date())
            
            if not calendar:
                return {
                    'is_open': False,
                    'market_open': None,
                    'market_close': None,
                    'next_open': clock.next_open.datetime if hasattr(clock, 'next_open') else None,
                    'next_close': clock.next_close.datetime if hasattr(clock, 'next_close') else None
                }
            
            day = calendar[0]
            
            return {
                'is_open': clock.is_open,
                'market_open': datetime.combine(day.date, day.open),
                'market_close': datetime.combine(day.date, day.close),
                'next_open': clock.next_open.datetime if hasattr(clock, 'next_open') else None,
                'next_close': clock.next_close.datetime if hasattr(clock, 'next_close') else None
            }
        except Exception as e:
            self.logger.error(f"Error getting trading hours: {str(e)}")
            return {}
    
    def get_account_balances(self) -> Dict[str, Any]:
        """
        Get account balances.
        
        Returns:
            Dict: Account balance information
        """
        if not self.is_connected():
            return {}
        
        try:
            account = self.api.get_account()
            
            return {
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin) if hasattr(account, 'initial_margin') else 0.0,
                'maintenance_margin': float(account.maintenance_margin) if hasattr(account, 'maintenance_margin') else 0.0,
                'daytrading_buying_power': float(account.daytrading_buying_power) if hasattr(account, 'daytrading_buying_power') else 0.0,
                'account_id': account.id
            }
        except Exception as e:
            self.logger.error(f"Error getting account balances: {str(e)}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List[Dict]: List of positions
        """
        if not self.is_connected():
            return []
        
        try:
            positions = self.api.list_positions()
            
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'quantity': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc),
                    'side': 'long' if float(pos.qty) > 0 else 'short'
                })
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current orders.
        
        Returns:
            List[Dict]: List of orders
        """
        if not self.is_connected():
            return []
        
        try:
            orders = self.api.list_orders(status='open')
            
            result = []
            for order in orders:
                result.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'quantity': float(order.qty),
                    'filled_quantity': float(order.filled_qty),
                    'side': order.side,
                    'type': order.type,
                    'time_in_force': order.time_in_force,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'status': order.status,
                    'created_at': order.created_at
                })
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def place_equity_order(self, symbol: str, quantity: int, side: str, order_type: str, 
                         time_in_force: str = 'day', limit_price: float = None, 
                         stop_price: float = None, expected_price: float = None) -> Dict[str, Any]:
        """
        Place equity order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Order duration ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            expected_price: Expected execution price (for monitoring slippage)
            
        Returns:
            Dict: Order result
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to Alpaca'}
        
        # Generate a client-side order ID for idempotency
        client_order_id = f"bensbot-{uuid.uuid4().hex[:12]}"
        
        try:
            # Map order parameters to Alpaca API format
            alpaca_order_type = order_type
            if order_type == 'stop':
                alpaca_order_type = 'stop'
            elif order_type == 'stop_limit':
                alpaca_order_type = 'stop_limit'
            
            alpaca_time_in_force = time_in_force
            if time_in_force == 'gtc':
                alpaca_time_in_force = 'gtc'
            elif time_in_force == 'day':
                alpaca_time_in_force = 'day'
            
            # Place the order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=alpaca_order_type,
                time_in_force=alpaca_time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            # Publish order event
            if self.event_bus:
                self.event_bus.publish(OrderAcknowledged(
                    broker_id=self.broker_id,
                    order_id=order.id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    quantity=float(order.qty),
                    side=side,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    timestamp=datetime.now()
                ))
                
                # Track slippage if we have expected price
                if expected_price and order_type == 'market':
                    execution_price = None
                    if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                        execution_price = float(order.filled_avg_price)
                    
                    if execution_price:
                        slippage = abs(execution_price - expected_price) / expected_price
                        self.event_bus.publish(SlippageMetric(
                            broker_id=self.broker_id,
                            order_id=order.id,
                            symbol=symbol,
                            expected_price=expected_price,
                            execution_price=execution_price,
                            slippage=slippage,
                            timestamp=datetime.now()
                        ))
            
            return {
                'success': True,
                'order_id': order.id,
                'client_order_id': client_order_id,
                'status': order.status
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error placing order for {symbol}: {error_msg}")
            
            # Publish rejection event
            if self.event_bus:
                self.event_bus.publish(OrderRejected(
                    broker_id=self.broker_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    order_type=order_type,
                    reason=error_msg,
                    timestamp=datetime.now()
                ))
            
            return {
                'success': False,
                'error': error_msg,
                'client_order_id': client_order_id
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Order status
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to Alpaca'}
        
        try:
            order = self.api.get_order(order_id)
            
            return {
                'success': True,
                'order_id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'filled_at': order.filled_at if hasattr(order, 'filled_at') else None,
                'filled_avg_price': float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else None
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error getting order status for {order_id}: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'order_id': order_id
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Cancellation result
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to Alpaca'}
        
        try:
            self.api.cancel_order(order_id)
            
            # Publish cancellation event
            if self.event_bus:
                self.event_bus.publish(OrderCancelled(
                    broker_id=self.broker_id,
                    order_id=order_id,
                    timestamp=datetime.now()
                ))
            
            return {
                'success': True,
                'order_id': order_id,
                'status': 'cancelled'
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error cancelling order {order_id}: {error_msg}")
            
            # Check if the error is because the order is already cancelled or filled
            if 'Invalid order ID' in error_msg or 'Order not found' in error_msg:
                return {
                    'success': False,
                    'error': 'Order not found',
                    'order_id': order_id
                }
            
            return {
                'success': False,
                'error': error_msg,
                'order_id': order_id
            }
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Quote information
        """
        if not self.is_connected():
            return {}
        
        try:
            # Get latest trade
            last_trade = self.api.get_latest_trade(symbol)
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            
            return {
                'symbol': symbol,
                'bid': float(quote.bp),
                'ask': float(quote.ap),
                'bid_size': int(quote.bs),
                'ask_size': int(quote.as_),
                'last': float(last_trade.p),
                'last_size': int(last_trade.s),
                'last_timestamp': last_trade.t,
                'volume': int(last_trade.v) if hasattr(last_trade, 'v') else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, interval: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get historical market data.
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date
            end_date: End date (default: current time)
            
        Returns:
            List[Dict]: Historical data
        """
        if not self.is_connected():
            return []
        
        if end_date is None:
            end_date = datetime.now()
        
        # Map interval to Alpaca timeframe
        timeframe_map = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1Hour',
            '1d': '1Day'
        }
        
        timeframe = timeframe_map.get(interval, '1Day')
        
        try:
            # Alpaca API requires dates in ISO format
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_str,
                end=end_str,
                adjustment='raw'
            ).df
            
            if bars.empty:
                return []
            
            # Convert to list of dictionaries
            result = []
            for timestamp, bar in bars.iterrows():
                result.append({
                    'timestamp': timestamp.to_pydatetime(),
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': int(bar['volume'])
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []
    
    def name(self) -> str:
        """
        Get broker name.
        
        Returns:
            str: Broker name
        """
        mode = "Paper" if self.is_paper else "Live"
        return f"Alpaca ({mode})"
    
    def status(self) -> str:
        """
        Get broker status.
        
        Returns:
            str: Status ('connected', 'disconnected', 'error')
        """
        if self.is_connected():
            return 'connected'
        elif self.api is not None:
            return 'error'
        else:
            return 'disconnected'
    
    def supports_extended_hours(self) -> bool:
        """
        Check if broker supports extended hours trading.
        
        Returns:
            bool: True if extended hours trading is supported
        """
        return True
    
    def supports_fractional_shares(self) -> bool:
        """
        Check if broker supports fractional shares.
        
        Returns:
            bool: True if fractional shares are supported
        """
        return True
    
    def api_calls_remaining(self) -> Optional[int]:
        """
        Get number of API calls remaining.
        
        Returns:
            Optional[int]: Number of API calls remaining or None if unknown
        """
        # Alpaca doesn't provide an easy way to get this info
        return None
    
    def get_broker_time(self) -> datetime:
        """
        Get broker server time.
        
        Returns:
            datetime: Current time according to broker
        """
        if not self.is_connected():
            return datetime.now()
        
        try:
            clock = self.api.get_clock()
            return clock.timestamp.to_pydatetime()
        except Exception:
            return datetime.now()
    
    def get_margin_status(self) -> Dict[str, Any]:
        """
        Get margin account status.
        
        Returns:
            Dict: Margin status details
        """
        if not self.is_connected():
            return {}
        
        try:
            account = self.api.get_account()
            
            # Calculate margin percentage
            equity = float(account.equity)
            initial_margin = float(account.initial_margin) if hasattr(account, 'initial_margin') else 0.0
            
            # Avoid division by zero
            margin_percentage = 1.0
            if initial_margin > 0:
                margin_percentage = equity / initial_margin
            
            return {
                'account_id': account.id,
                'equity': equity,
                'initial_margin': initial_margin,
                'maintenance_margin': float(account.maintenance_margin) if hasattr(account, 'maintenance_margin') else 0.0,
                'margin_used': initial_margin,
                'margin_available': equity - initial_margin if equity > initial_margin else 0.0,
                'margin_percentage': margin_percentage,
                'buying_power': float(account.buying_power),
                'last_updated': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error getting margin status: {str(e)}")
            return {}
