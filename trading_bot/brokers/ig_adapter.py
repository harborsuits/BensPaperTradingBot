"""
IG Markets API Adapter

This module implements the broker interface for IG Markets, based on
the implementation from ilcardella's TradingBot.
"""

import logging
import json
import time
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, timezone
import pytz

from trading_bot.brokers.broker_interface import BrokerInterface, MarketSession

logger = logging.getLogger(__name__)

class IGAPIError(Exception):
    """Exception raised for IG API errors"""
    pass


class IGAdapter(BrokerInterface):
    """
    IG Markets API adapter implementation
    
    This adapter allows the trading system to interface with IG Markets
    for trade execution, market data, and account management.
    """
    
    def __init__(self, 
                api_key: str, 
                username: str, 
                password: str, 
                demo: bool = True,
                timeout: int = 60,
                auto_refresh: bool = True):
        """
        Initialize the IG adapter
        
        Args:
            api_key: IG API key
            username: IG username
            password: IG password
            demo: Whether to use demo environment
            timeout: API request timeout in seconds
            auto_refresh: Whether to auto-refresh token
        """
        self._api_key = api_key
        self._username = username
        self._password = password
        self._demo = demo
        self._timeout = timeout
        self._auto_refresh = auto_refresh
        
        # API URLs
        self._base_url = "https://demo-api.ig.com/gateway/deal" if demo else "https://api.ig.com/gateway/deal"
        
        # Session variables
        self._headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json; charset=utf-8',
            'X-IG-API-KEY': api_key
        }
        self._session = requests.Session()
        
        # Auth tokens
        self._cst = ""
        self._security_token = ""
        self._token_expiry = datetime.now(timezone.utc)
        self._connected = False
        self._api_calls_left = None
        
        # Connect to API
        self._connect()
    
    def _connect(self) -> bool:
        """
        Connect to IG API
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("Connecting to IG API...")
            
            # Auth request
            auth_data = {
                "identifier": self._username,
                "password": self._password
            }
            
            headers = self._headers.copy()
            headers['Version'] = '2'
            
            response = self._session.post(
                f"{self._base_url}/session",
                data=json.dumps(auth_data),
                headers=headers,
                timeout=self._timeout
            )
            
            if response.status_code != 200:
                logger.error(f"IG authentication failed: {response.status_code}, {response.text}")
                self._connected = False
                return False
            
            # Parse response
            self._cst = response.headers['CST']
            self._security_token = response.headers['X-SECURITY-TOKEN']
            self._headers['CST'] = self._cst
            self._headers['X-SECURITY-TOKEN'] = self._security_token
            
            # Set token expiry (tokens expire after 6 hours)
            self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=6)
            
            # Update API calls remaining
            if 'X-RATE-LIMIT-REMAINING' in response.headers:
                try:
                    self._api_calls_left = int(response.headers['X-RATE-LIMIT-REMAINING'])
                except (ValueError, TypeError):
                    pass
                    
            logger.info("Successfully connected to IG API")
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to IG API: {str(e)}")
            self._connected = False
            return False
    
    def refresh_connection(self) -> bool:
        """
        Refresh IG API token
        
        Returns:
            bool: True if refresh successful
        """
        if not self._connected:
            return self._connect()
            
        try:
            logger.info("Refreshing IG API token...")
            
            headers = self._headers.copy()
            headers['Version'] = '1'
            
            response = self._session.post(
                f"{self._base_url}/session/refresh-token",
                headers=headers,
                timeout=self._timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"Token refresh failed: {response.status_code}, {response.text}")
                # Token might be completely expired, try reconnecting
                return self._connect()
            
            # Update tokens
            self._security_token = response.headers['X-SECURITY-TOKEN']
            self._headers['X-SECURITY-TOKEN'] = self._security_token
            self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=6)
            
            # Update API calls remaining
            if 'X-RATE-LIMIT-REMAINING' in response.headers:
                try:
                    self._api_calls_left = int(response.headers['X-RATE-LIMIT-REMAINING'])
                except (ValueError, TypeError):
                    pass
                    
            logger.info("Successfully refreshed IG API token")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing IG API token: {str(e)}")
            # Try reconnecting
            return self._connect()
    
    def _make_request(self, method: str, endpoint: str, data: Any = None, 
                    version: str = '1', params: Dict = None) -> Dict:
        """
        Make request to IG API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            version: API version
            params: Query parameters
            
        Returns:
            Dict: Response JSON
        """
        if not self._connected:
            if not self._connect():
                raise IGAPIError("Not connected to IG API")
                
        # Check if token needs refresh
        if self.needs_refresh() and self._auto_refresh:
            self.refresh_connection()
            
        # Prepare request
        url = f"{self._base_url}/{endpoint}"
        headers = self._headers.copy()
        headers['Version'] = version
        
        # Make request
        try:
            if method.upper() == 'GET':
                response = self._session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=self._timeout
                )
            elif method.upper() == 'POST':
                response = self._session.post(
                    url, 
                    headers=headers, 
                    data=json.dumps(data) if data else None,
                    params=params,
                    timeout=self._timeout
                )
            elif method.upper() == 'PUT':
                response = self._session.put(
                    url, 
                    headers=headers, 
                    data=json.dumps(data) if data else None,
                    params=params,
                    timeout=self._timeout
                )
            elif method.upper() == 'DELETE':
                response = self._session.delete(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=self._timeout
                )
            else:
                raise IGAPIError(f"Unsupported HTTP method: {method}")
                
            # Update API calls remaining
            if 'X-RATE-LIMIT-REMAINING' in response.headers:
                try:
                    self._api_calls_left = int(response.headers['X-RATE-LIMIT-REMAINING'])
                except (ValueError, TypeError):
                    pass
                    
            # Handle response
            if response.status_code < 200 or response.status_code >= 300:
                raise IGAPIError(f"HTTP error {response.status_code}: {response.text}")
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self._connected = False
            raise IGAPIError(f"Request error: {str(e)}")
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            trading_hours = self.get_trading_hours()
            if not trading_hours:
                return False
                
            return trading_hours.get('marketState', '').lower() == 'open'
            
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
    
    def get_next_market_open(self) -> datetime:
        """
        Get the next market open datetime
        
        Returns:
            datetime: Next market open time
        """
        try:
            # First try to get from IG API
            trading_hours = self.get_trading_hours()
            if trading_hours and 'openTime' in trading_hours:
                return trading_hours['openTime']
                
            # Fallback to MarketSession utility
            now = datetime.now(timezone.utc)
            eastern_tz = pytz.timezone('US/Eastern')
            now_eastern = now.astimezone(eastern_tz)
            
            return MarketSession.get_next_market_open(now_eastern)
            
        except Exception as e:
            logger.error(f"Error getting next market open: {str(e)}")
            # Use MarketSession as fallback
            now = datetime.now(timezone.utc)
            eastern_tz = pytz.timezone('US/Eastern')
            now_eastern = now.astimezone(eastern_tz)
            
            return MarketSession.get_next_market_open(now_eastern)
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get trading hours information
        
        Returns:
            Dict: Trading hours information for the current day
        """
        try:
            # Use SPY as a proxy for market hours
            response = self._make_request(
                'GET',
                'markets/SPY',
                version='1'
            )
            
            if not response or 'instrument' not in response:
                return {}
                
            return {
                'marketState': response['instrument'].get('marketStatus', ''),
                'openTime': datetime.fromisoformat(response['instrument'].get('openTime', '')) if 'openTime' in response['instrument'] else None,
                'closeTime': datetime.fromisoformat(response['instrument'].get('closeTime', '')) if 'closeTime' in response['instrument'] else None
            }
            
        except Exception as e:
            logger.error(f"Error getting trading hours: {str(e)}")
            return {}
    
    def get_account_balances(self) -> Dict[str, Any]:
        """
        Get account balance information
        
        Returns:
            Dict: Account balance details
        """
        try:
            response = self._make_request(
                'GET',
                'accounts',
                version='1'
            )
            
            if not response or 'accounts' not in response:
                return {}
                
            account = response['accounts'][0]  # Use first account
            
            return {
                'accountId': account.get('accountId', ''),
                'accountName': account.get('accountName', ''),
                'balance': account.get('balance', {}).get('balance', 0),
                'equity': account.get('balance', {}).get('available', 0),
                'pnl': account.get('balance', {}).get('profitLoss', 0),
                'currency': account.get('currency', 'USD')
            }
            
        except Exception as e:
            logger.error(f"Error getting account balances: {str(e)}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
            List[Dict]: List of positions
        """
        try:
            response = self._make_request(
                'GET',
                'positions',
                version='2'
            )
            
            if not response or 'positions' not in response:
                return []
                
            positions = []
            for pos in response['positions']:
                position = {
                    'dealId': pos.get('position', {}).get('dealId', ''),
                    'symbol': pos.get('market', {}).get('epic', ''),
                    'marketName': pos.get('market', {}).get('instrumentName', ''),
                    'direction': pos.get('position', {}).get('direction', ''),
                    'size': pos.get('position', {}).get('size', 0),
                    'open_price': pos.get('position', {}).get('openLevel', 0),
                    'current_price': pos.get('market', {}).get('bid', 0) if pos.get('position', {}).get('direction', '') == 'BUY' else pos.get('market', {}).get('offer', 0),
                    'profit_loss': pos.get('position', {}).get('profitLoss', 0),
                    'currency': pos.get('position', {}).get('currency', 'USD'),
                    'created_time': pos.get('position', {}).get('createdDateUTC', '')
                }
                positions.append(position)
                
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current orders
        
        Returns:
            List[Dict]: List of orders
        """
        try:
            response = self._make_request(
                'GET',
                'workingorders',
                version='2'
            )
            
            if not response or 'workingOrders' not in response:
                return []
                
            orders = []
            for order in response['workingOrders']:
                order_data = {
                    'dealId': order.get('workingOrderData', {}).get('dealId', ''),
                    'symbol': order.get('marketData', {}).get('epic', ''),
                    'marketName': order.get('marketData', {}).get('instrumentName', ''),
                    'direction': order.get('workingOrderData', {}).get('direction', ''),
                    'size': order.get('workingOrderData', {}).get('size', 0),
                    'price': order.get('workingOrderData', {}).get('level', 0),
                    'status': order.get('workingOrderData', {}).get('status', ''),
                    'order_type': order.get('workingOrderData', {}).get('type', ''),
                    'time_in_force': order.get('workingOrderData', {}).get('timeInForce', ''),
                    'created_time': order.get('workingOrderData', {}).get('createdDateUTC', '')
                }
                orders.append(order_data)
                
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def place_equity_order(self, 
                          symbol: str, 
                          side: str, 
                          quantity: int, 
                          order_type: str = "market", 
                          duration: str = "day", 
                          price: Optional[float] = None, 
                          stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an equity order
        
        Args:
            symbol: Stock symbol
            side: Order side ('buy' or 'sell')
            quantity: Number of shares
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            duration: Order duration ('day', 'gtc')
            price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            
        Returns:
            Dict: Order result with ID and status
        """
        try:
            # Map order type to IG order type
            ig_order_type = {
                'market': 'MARKET',
                'limit': 'LIMIT',
                'stop': 'STOP',
                'stop_limit': 'STOP_LIMIT'
            }.get(order_type.lower(), 'MARKET')
            
            # Map duration to IG time in force
            ig_time_in_force = {
                'day': 'GOOD_TILL_DATE',
                'gtc': 'GOOD_TILL_CANCELLED'
            }.get(duration.lower(), 'GOOD_TILL_DATE')
            
            # Map side to IG direction
            ig_direction = 'BUY' if side.lower() == 'buy' else 'SELL'
            
            # Create order request
            order_data = {
                'epic': symbol,
                'expiry': '-',
                'direction': ig_direction,
                'size': quantity,
                'orderType': ig_order_type,
                'timeInForce': ig_time_in_force,
                'guaranteedStop': False,
                'forceOpen': True
            }
            
            # Add limit price if needed
            if price is not None and ig_order_type in ['LIMIT', 'STOP_LIMIT']:
                order_data['level'] = price
                
            # Add stop price if needed
            if stop_price is not None and ig_order_type in ['STOP', 'STOP_LIMIT']:
                order_data['stopLevel'] = stop_price
                
            # Set good-till-date if needed
            if ig_time_in_force == 'GOOD_TILL_DATE':
                # Default to end of day
                order_data['goodTillDate'] = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Place the order
            response = self._make_request(
                'POST',
                'workingorders/otc',
                data=order_data,
                version='2'
            )
            
            # Check response
            if not response or 'dealReference' not in response:
                raise IGAPIError(f"Failed to place order: {json.dumps(response)}")
                
            # Get order details
            deal_id = response['dealReference']
            
            # Allow time for the order to process
            time.sleep(1)
            
            # Get order confirmation
            confirmation = self._make_request(
                'GET',
                f'confirms/{deal_id}',
                version='1'
            )
            
            if not confirmation:
                return {
                    'id': deal_id,
                    'status': 'PENDING',
                    'message': 'Order placed, awaiting confirmation'
                }
                
            return {
                'id': confirmation.get('dealId', deal_id),
                'status': confirmation.get('status', 'PENDING'),
                'date': confirmation.get('date', ''),
                'price': confirmation.get('level', price),
                'size': confirmation.get('size', quantity),
                'message': confirmation.get('reason', '')
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise IGAPIError(f"Failed to place order: {str(e)}")
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Order status details
        """
        try:
            # Check current orders
            orders = self.get_orders()
            for order in orders:
                if order.get('dealId') == order_id:
                    return order
                    
            # If not found in active orders, check for confirmation
            try:
                confirmation = self._make_request(
                    'GET',
                    f'confirms/{order_id}',
                    version='1'
                )
                
                if confirmation:
                    return {
                        'dealId': confirmation.get('dealId', order_id),
                        'status': confirmation.get('status', 'UNKNOWN'),
                        'date': confirmation.get('date', ''),
                        'price': confirmation.get('level', 0),
                        'size': confirmation.get('size', 0),
                        'reason': confirmation.get('reason', '')
                    }
            except:
                pass
                
            # If not found, assume it was filled or cancelled
            return {
                'dealId': order_id,
                'status': 'UNKNOWN',
                'message': 'Order not found in active orders'
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return {
                'dealId': order_id,
                'status': 'ERROR',
                'message': f"Error getting order status: {str(e)}"
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Cancellation result
        """
        try:
            # First check if order exists
            order = self.get_order_status(order_id)
            if order.get('status') == 'UNKNOWN':
                return {
                    'success': False,
                    'message': 'Order not found'
                }
                
            # Cancel the order
            response = self._make_request(
                'DELETE',
                f'workingorders/otc/{order_id}',
                version='2'
            )
            
            if not response or 'dealReference' not in response:
                raise IGAPIError(f"Failed to cancel order: {json.dumps(response)}")
                
            # Get cancellation confirmation
            deal_id = response['dealReference']
            
            # Allow time for the cancellation to process
            time.sleep(1)
            
            # Get confirmation
            confirmation = self._make_request(
                'GET',
                f'confirms/{deal_id}',
                version='1'
            )
            
            if not confirmation:
                return {
                    'success': True,
                    'message': 'Cancellation request accepted, awaiting confirmation'
                }
                
            return {
                'success': confirmation.get('status', '') == 'ACCEPTED',
                'message': confirmation.get('reason', ''),
                'dealId': confirmation.get('dealId', deal_id),
                'date': confirmation.get('date', '')
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return {
                'success': False,
                'message': f"Error cancelling order: {str(e)}"
            }
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Quote information
        """
        try:
            response = self._make_request(
                'GET',
                f'markets/{symbol}',
                version='3'
            )
            
            if not response or 'snapshot' not in response:
                return {}
                
            snapshot = response.get('snapshot', {})
            
            return {
                'symbol': symbol,
                'bid': snapshot.get('bid', 0),
                'ask': snapshot.get('offer', 0),
                'last': snapshot.get('bid', 0), # IG doesn't provide last price
                'volume': snapshot.get('marketData', {}).get('volumeQuote', 0),
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {}
    
    def get_historical_data(self, 
                           symbol: str, 
                           interval: str, 
                           start_date: datetime, 
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get historical market data
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date
            end_date: End date (defaults to current time)
            
        Returns:
            List[Dict]: List of OHLCV candles
        """
        try:
            # Map interval to IG resolution
            ig_resolution = {
                '1m': 'MINUTE',
                '5m': 'MINUTE_5',
                '15m': 'MINUTE_15',
                '30m': 'MINUTE_30',
                '1h': 'HOUR',
                '2h': 'HOUR_2',
                '4h': 'HOUR_4',
                '1d': 'DAY'
            }.get(interval.lower(), 'DAY')
            
            # Set end date if not provided
            if end_date is None:
                end_date = datetime.now(timezone.utc)
                
            # Format dates for IG API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get market data
            params = {
                'resolution': ig_resolution,
                'from': start_str,
                'to': end_str,
                'max': 1000
            }
            
            response = self._make_request(
                'GET',
                f'prices/{symbol}',
                params=params,
                version='3'
            )
            
            if not response or 'prices' not in response:
                return []
                
            # Convert prices to OHLCV format
            candles = []
            for price in response['prices']:
                candle = {
                    'datetime': datetime.fromisoformat(price.get('snapshotTimeUTC', '')),
                    'open': price.get('openPrice', {}).get('bid', 0),
                    'high': price.get('highPrice', {}).get('bid', 0),
                    'low': price.get('lowPrice', {}).get('bid', 0),
                    'close': price.get('closePrice', {}).get('bid', 0),
                    'volume': price.get('lastTradedVolume', 0)
                }
                candles.append(candle)
                
            return candles
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []
    
    @property
    def name(self) -> str:
        """
        Get broker name
        
        Returns:
            str: Broker name
        """
        return "IG Markets"
    
    @property
    def status(self) -> str:
        """
        Get broker connection status
        
        Returns:
            str: Status ('connected', 'disconnected', 'error')
        """
        if not self._connected:
            return "disconnected"
            
        if self.needs_refresh() and not self._auto_refresh:
            return "token_expiring"
            
        return "connected"
    
    @property
    def supports_extended_hours(self) -> bool:
        """
        Check if broker supports extended hours trading
        
        Returns:
            bool: True if extended hours trading is supported
        """
        return True
    
    @property
    def supports_fractional_shares(self) -> bool:
        """
        Check if broker supports fractional shares
        
        Returns:
            bool: True if fractional shares are supported
        """
        return False
    
    @property
    def api_calls_remaining(self) -> Optional[int]:
        """
        Get number of API calls remaining (rate limiting)
        
        Returns:
            Optional[int]: Number of calls remaining or None if not applicable
        """
        return self._api_calls_left
    
    def get_broker_time(self) -> datetime:
        """
        Get current time from broker's servers
        
        Returns:
            datetime: Current time according to broker
        """
        try:
            response = self._make_request(
                'GET',
                'time',
                version='1'
            )
            
            if not response or 'time' not in response:
                return datetime.now(timezone.utc)
                
            return datetime.fromisoformat(response['time'])
            
        except Exception as e:
            logger.error(f"Error getting broker time: {str(e)}")
            return datetime.now(timezone.utc)
    
    def needs_refresh(self) -> bool:
        """
        Check if token needs refresh
        
        Returns:
            bool: True if token needs refresh
        """
        # Check if token expires within 30 minutes
        return datetime.now(timezone.utc) + timedelta(minutes=30) > self._token_expiry
