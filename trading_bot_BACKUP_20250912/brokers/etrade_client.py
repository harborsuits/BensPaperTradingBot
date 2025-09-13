"""
E*TRADE Brokerage API Client

This module implements the BrokerInterface for E*TRADE API,
providing integration with E*TRADE for trading stocks, options, and other supported assets.
"""

import os
import logging
import time
import json
import base64
import hashlib
import hmac
import urllib.parse
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import requests
from requests_oauthlib import OAuth1

from trading_bot.models.broker_models import (
    BrokerAccount, Position, Order, Quote, Bar,
    OrderStatus, OrderType, OrderSide, TimeInForce, AssetType
)
from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.brokers.broker_credentials import BrokerCredentials
from trading_bot.exceptions.broker_exceptions import (
    BrokerConnectionError, BrokerAuthenticationError, BrokerOrderError
)

# Set up logging
logger = logging.getLogger(__name__)

class ETradeClient(BrokerInterface):
    """
    E*TRADE broker client implementation.
    
    This class implements the BrokerInterface for E*TRADE, providing access
    to E*TRADE's trading API for stocks, options, and other supported assets.
    """
    
    def __init__(self, 
                 consumer_key: Optional[str] = None,
                 consumer_secret: Optional[str] = None,
                 sandbox_mode: bool = True,
                 cache_ttl: int = 5,
                 default_account_index: int = 0):
        """
        Initialize the E*TRADE client.
        
        Args:
            consumer_key: E*TRADE API consumer key
            consumer_secret: E*TRADE API consumer secret
            sandbox_mode: Use sandbox environment if True, production if False
            cache_ttl: Cache time-to-live in seconds
            default_account_index: Default account index to use if multiple accounts
        """
        # API connection settings
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.sandbox_mode = sandbox_mode
        
        # Set base URLs based on environment
        if sandbox_mode:
            self.base_url = "https://apisb.etrade.com"
        else:
            self.base_url = "https://api.etrade.com"
            
        # Authentication tokens
        self.oauth_token = None
        self.oauth_token_secret = None
        self.oauth_session = None
        self.connected = False
        
        # Account information
        self.available_accounts = []
        self.account_id = None
        self.default_account_index = default_account_index
        
        # Request rate limiting
        self.min_request_interval = 0.2  # seconds
        self.last_request_time = datetime.now() - timedelta(seconds=1)
        
        # Cache settings
        self.cache_ttl = cache_ttl
        self.account_cache = {}
        self.position_cache = {}
        self.order_cache = {}
        self.quote_cache = {}
        self.bars_cache = {}
        self.cache_expiry = {}
        
        # Create a requests session for connection pooling
        self.session = requests.Session()
        
        # Request rate limit tracking
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
    
    def connect(self, credentials: BrokerCredentials) -> bool:
        """
        Connect to E*TRADE API using OAuth1.
        
        Args:
            credentials: BrokerCredentials with OAuth token and secret
            
        Returns:
            bool: True if connection is successful
            
        Raises:
            BrokerAuthenticationError: If authentication fails
        """
        try:
            # Extract credentials
            self.consumer_key = credentials.api_key or self.consumer_key
            self.consumer_secret = credentials.api_secret or self.consumer_secret
            self.oauth_token = credentials.access_token
            self.oauth_token_secret = credentials.access_token_secret
            
            if not self.consumer_key or not self.consumer_secret:
                raise BrokerAuthenticationError("E*TRADE API consumer key and secret are required")
            
            # If we have existing OAuth tokens, try to use them
            if self.oauth_token and self.oauth_token_secret:
                # Set up OAuth1 session
                self.oauth_session = OAuth1(
                    client_key=self.consumer_key,
                    client_secret=self.consumer_secret,
                    resource_owner_key=self.oauth_token,
                    resource_owner_secret=self.oauth_token_secret
                )
                
                # Test authentication by making a simple request
                self.connected = self._test_connection()
                if not self.connected:
                    logger.warning("Existing OAuth tokens are invalid or expired")
                    self.oauth_token = None
                    self.oauth_token_secret = None
                    raise BrokerAuthenticationError("E*TRADE OAuth tokens are invalid or expired. Please re-authorize.")
            else:
                # If we don't have tokens, we need to go through OAuth flow
                logger.warning("OAuth token and secret not provided. Manual OAuth flow required.")
                raise BrokerAuthenticationError(
                    "E*TRADE requires OAuth1 authentication flow. "
                    "Please complete web authorization and provide oauth_token and oauth_token_secret."
                )
            
            # Get available accounts
            if self.connected:
                try:
                    self.available_accounts = self.get_account_info()
                    
                    # Set default account
                    if self.available_accounts:
                        if self.default_account_index < len(self.available_accounts):
                            self.account_id = self.available_accounts[self.default_account_index].account_id
                        else:
                            self.account_id = self.available_accounts[0].account_id
                    
                    return True
                except Exception as e:
                    logger.error(f"Failed to retrieve accounts: {str(e)}")
                    return False
            
            return self.connected
        
        except Exception as e:
            logger.error(f"Failed to connect to E*TRADE: {str(e)}")
            self.connected = False
            raise BrokerAuthenticationError(f"Failed to connect to E*TRADE: {str(e)}")
    
    def disconnect(self) -> bool:
        """
        Disconnect from E*TRADE API.
        
        Returns:
            bool: True if disconnection is successful
        """
        try:
            # Clear authentication tokens
            self.oauth_session = None
            self.connected = False
            
            # Clear caches
            self.account_cache = {}
            self.position_cache = {}
            self.order_cache = {}
            self.quote_cache = {}
            self.bars_cache = {}
            self.cache_expiry = {}
            
            return True
        
        except Exception as e:
            logger.error(f"Error during E*TRADE disconnect: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to E*TRADE API.
        
        Returns:
            bool: True if connected
        """
        # If we haven't established a connection yet, return current status
        if not self.connected or not self.oauth_session:
            return False
            
        # Only test the connection occasionally to avoid excessive API calls
        test_interval = 300  # 5 minutes
        time_since_last_request = (datetime.now() - self.last_request_time).total_seconds()
        
        if time_since_last_request > test_interval:
            self.connected = self._test_connection()
        
        return self.connected
    
    def _test_connection(self) -> bool:
        """
        Test connection to E*TRADE API.
        
        Returns:
            bool: True if connection test succeeds
        """
        try:
            # Make a simple request to test authentication
            url = f"{self.base_url}/v1/accounts/list"
            
            response = self.session.get(
                url,
                auth=self.oauth_session,
                timeout=10
            )
            
            # Update last request time
            self.last_request_time = datetime.now()
            
            # Check if the request was successful
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                logger.warning("E*TRADE authentication failed. Tokens may be expired.")
                return False
            else:
                logger.warning(f"E*TRADE connection test failed with status code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"E*TRADE connection test failed: {str(e)}")
            return False
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Any:
        """
        Make an API request to E*TRADE.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (starting with /)
            params: Query parameters
            data: Request body data
            
        Returns:
            Response data as JSON or dict
            
        Raises:
            BrokerAuthenticationError: If authentication fails
            BrokerConnectionError: If connection fails
            BrokerOrderError: If order-related error occurs
        """
        if not self.oauth_session:
            raise BrokerAuthenticationError("Not authenticated with E*TRADE")
        
        # Rate limiting: ensure we don't send requests too quickly
        elapsed = (datetime.now() - self.last_request_time).total_seconds()
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        # Construct full URL
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Make the request
            if method == 'GET':
                response = self.session.get(url, auth=self.oauth_session, params=params, timeout=30)
            elif method == 'POST':
                headers = {'Content-Type': 'application/json'}
                response = self.session.post(url, auth=self.oauth_session, params=params, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                headers = {'Content-Type': 'application/json'}
                response = self.session.put(url, auth=self.oauth_session, params=params, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = self.session.delete(url, auth=self.oauth_session, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Record last request time
            self.last_request_time = datetime.now()
            
            # Check for errors
            if response.status_code == 401:
                raise BrokerAuthenticationError("E*TRADE authentication failed, token may be expired")
            elif response.status_code == 403:
                raise BrokerAuthenticationError("E*TRADE access forbidden, insufficient permissions")
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded, retrying after {retry_after} seconds")
                time.sleep(retry_after)
                return self._make_request(method, endpoint, params, data)  # Retry
            
            response.raise_for_status()
            
            # Parse response
            if response.content:
                return response.json()
            return {}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"E*TRADE API request failed ({method} {endpoint}): {str(e)}")
            if "401" in str(e):
                raise BrokerAuthenticationError(f"E*TRADE authentication error: {str(e)}")
            elif any(error in str(e) for error in ["403", "404", "400"]):
                raise BrokerOrderError(f"E*TRADE order error: {str(e)}")
            else:
                raise BrokerConnectionError(f"E*TRADE connection error: {str(e)}")
        
        except json.JSONDecodeError as e:
            logger.error(f"E*TRADE API returned invalid JSON: {str(e)}")
            raise BrokerConnectionError(f"E*TRADE returned invalid response: {str(e)}")
    
    def get_account_info(self) -> List[BrokerAccount]:
        """
        Get account information from E*TRADE.
        
        Returns:
            List[BrokerAccount]: List of account information
            
        Raises:
            BrokerConnectionError: If connection is not established
        """
        # Check if we have cached data that's still valid
        cache_key = 'accounts'
        if (cache_key in self.account_cache and 
            cache_key in self.cache_expiry and 
            datetime.now() < self.cache_expiry[cache_key]):
            return self.account_cache[cache_key]
        
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        try:
            # Get accounts list
            response = self._make_request('GET', '/v1/accounts/list')
            
            accounts = []
            if 'AccountListResponse' in response and 'Accounts' in response['AccountListResponse']:
                for acct_data in response['AccountListResponse']['Accounts'].get('Account', []):
                    # Extract the basic account data
                    account_id = acct_data.get('accountId')
                    account_name = acct_data.get('accountName', '')
                    account_type = acct_data.get('accountType', '')
                    account_status = acct_data.get('accountStatus', '')
                    
                    # Skip inactive accounts
                    if account_status.lower() != 'active':
                        logger.info(f"Skipping inactive account {account_id}")
                        continue
                    
                    # Get account balances/details
                    balance_response = self._make_request('GET', f'/v1/accounts/{account_id}/balance')
                    
                    # Extract balances
                    balance_data = {}
                    if 'BalanceResponse' in balance_response:
                        balance_data = balance_response['BalanceResponse']
                    
                    # Create the account object
                    buying_power = 0
                    cash_balance = 0
                    equity = 0
                    
                    if 'Computed' in balance_data:
                        computed = balance_data['Computed']
                        buying_power = float(computed.get('buyingPower', 0))
                        
                    if 'Cash' in balance_data:
                        cash = balance_data['Cash']
                        cash_balance = float(cash.get('fundsForOpenOrdersCash', 0))
                    
                    if 'RealTimeValues' in balance_data:
                        real_time = balance_data['RealTimeValues']
                        equity = float(real_time.get('totalAccountValue', 0))
                    
                    account = BrokerAccount(
                        broker_id=self.get_broker_name(),
                        account_id=account_id,
                        account_number=acct_data.get('accountIdKey', ''),
                        account_type=account_type,
                        buying_power=buying_power,
                        cash_balance=cash_balance,
                        equity=equity
                    )
                    accounts.append(account)
            
            # Update cache
            self.account_cache[cache_key] = accounts
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
            
            self.available_accounts = accounts
            
            return accounts
        
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            raise BrokerConnectionError(f"Failed to get account info: {str(e)}")
    
    def get_positions(self) -> List[Position]:
        """
        Get current positions from E*TRADE.
        
        Returns:
            List[Position]: List of current positions
            
        Raises:
            BrokerConnectionError: If connection is not established
        """
        # Check if we have cached data that's still valid
        cache_key = f'positions_{self.account_id}'
        if (cache_key in self.position_cache and 
            cache_key in self.cache_expiry and 
            datetime.now() < self.cache_expiry[cache_key]):
            return self.position_cache[cache_key]
        
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Get positions data
            response = self._make_request('GET', f'/v1/accounts/{self.account_id}/portfolio')
            
            positions = []
            if 'PortfolioResponse' in response and 'AccountPortfolio' in response['PortfolioResponse']:
                portfolios = response['PortfolioResponse']['AccountPortfolio']
                if not isinstance(portfolios, list):
                    portfolios = [portfolios]
                    
                for portfolio in portfolios:
                    if 'Position' not in portfolio:
                        continue
                        
                    position_list = portfolio['Position']
                    if not isinstance(position_list, list):
                        position_list = [position_list]
                    
                    for pos_data in position_list:
                        # Extract the position data
                        symbol = pos_data.get('Product', {}).get('symbol', '')
                        security_type = pos_data.get('Product', {}).get('securityType', '')
                        quantity = float(pos_data.get('quantityLong', 0)) - float(pos_data.get('quantityShort', 0))
                        price_info = pos_data.get('QuickView', {})
                        
                        # Determine asset type
                        asset_type = AssetType.STOCK
                        if security_type:
                            sec_type = security_type.lower()
                            if 'option' in sec_type:
                                asset_type = AssetType.OPTION
                            elif 'future' in sec_type or 'fut' in sec_type:
                                asset_type = AssetType.FUTURE
                            elif 'forex' in sec_type or 'fx' in sec_type or 'currency' in sec_type:
                                asset_type = AssetType.FOREX
                        
                        # Create position object
                        position = Position(
                            symbol=symbol,
                            quantity=quantity,
                            avg_entry_price=float(pos_data.get('costBasis', 0)) / abs(quantity) if quantity != 0 else 0,
                            current_price=float(price_info.get('lastTrade', 0)),
                            market_value=float(pos_data.get('marketValue', 0)),
                            unrealized_pl=float(pos_data.get('totalGain', 0)),
                            asset_type=asset_type,
                            broker_id=self.get_broker_name(),
                            broker_position_id=pos_data.get('positionId', '')
                        )
                        positions.append(position)
            
            # Update cache
            self.position_cache[cache_key] = positions
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
            
            return positions
        
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            raise BrokerConnectionError(f"Failed to get positions: {str(e)}")
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get orders from E*TRADE.
        
        Args:
            status: Filter orders by status
            
        Returns:
            List[Order]: List of orders
            
        Raises:
            BrokerConnectionError: If connection is not established
        """
        # Check if we have cached data that's still valid and no status filter
        cache_key = f'orders_{self.account_id}'
        if (status is None and
            cache_key in self.order_cache and 
            cache_key in self.cache_expiry and 
            datetime.now() < self.cache_expiry[cache_key]):
            
            return self.order_cache[cache_key]
        
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Map our OrderStatus to E*TRADE status
            params = {}
            if status:
                et_status = None
                if status == OrderStatus.NEW:
                    et_status = 'OPEN'
                elif status == OrderStatus.FILLED:
                    et_status = 'EXECUTED'
                elif status == OrderStatus.PARTIALLY_FILLED:
                    et_status = 'PARTIAL'
                elif status == OrderStatus.CANCELLED:
                    et_status = 'CANCELLED'
                elif status == OrderStatus.REJECTED:
                    et_status = 'REJECTED'
                elif status == OrderStatus.PENDING:
                    et_status = 'PENDING'
                elif status == OrderStatus.EXPIRED:
                    et_status = 'EXPIRED'
                
                if et_status:
                    params['status'] = et_status
            
            # Get orders data
            response = self._make_request('GET', f'/v1/accounts/{self.account_id}/orders')
            
            orders = []
            if 'OrdersResponse' in response and 'Order' in response['OrdersResponse']:
                orders_data = response['OrdersResponse']['Order']
                if not isinstance(orders_data, list):
                    orders_data = [orders_data]
                
                for order_data in orders_data:
                    # Check if order matches status filter
                    if status:
                        et_status = order_data.get('status', '')
                        if et_status != params.get('status'):
                            continue
                    
                    # Extract order information from the order data
                    # E*TRADE orders are complex and can have multiple legs, types, etc.
                    # We'll extract the core information needed for our Order model
                    
                    # Get the main order information
                    order_id = order_data.get('orderId', '')
                    placed_time = order_data.get('placedTime', '')
                    executed_time = order_data.get('executedTime', '')
                    order_status = order_data.get('status', '')
                    
                    # Parse times if available
                    created_at = None
                    updated_at = None
                    
                    if placed_time:
                        try:
                            created_at = datetime.fromisoformat(placed_time.replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                # Try another format if ISO format fails
                                created_at = datetime.strptime(placed_time, '%m/%d/%Y %H:%M:%S')
                            except ValueError:
                                created_at = None
                    
                    if executed_time:
                        try:
                            updated_at = datetime.fromisoformat(executed_time.replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                # Try another format if ISO format fails
                                updated_at = datetime.strptime(executed_time, '%m/%d/%Y %H:%M:%S')
                            except ValueError:
                                updated_at = None
                    
                    # Map E*TRADE status to our OrderStatus
                    order_status_enum = OrderStatus.NEW
                    if order_status == 'EXECUTED':
                        order_status_enum = OrderStatus.FILLED
                    elif order_status == 'PARTIAL':
                        order_status_enum = OrderStatus.PARTIALLY_FILLED
                    elif order_status == 'CANCELLED':
                        order_status_enum = OrderStatus.CANCELLED
                    elif order_status == 'REJECTED':
                        order_status_enum = OrderStatus.REJECTED
                    elif order_status == 'PENDING':
                        order_status_enum = OrderStatus.PENDING
                    elif order_status == 'EXPIRED':
                        order_status_enum = OrderStatus.EXPIRED
                    
                    # E*TRADE orders can have multiple legs, we'll create an Order for each
                    # OrderDetail section contains individual legs
                    if 'OrderDetail' in order_data:
                        order_details = order_data['OrderDetail']
                        if not isinstance(order_details, list):
                            order_details = [order_details]
                            
                        for detail in order_details:
                            # Get instrument details
                            if 'Instrument' not in detail:
                                continue
                                
                            instruments = detail['Instrument']
                            if not isinstance(instruments, list):
                                instruments = [instruments]
                            
                            for instrument in instruments:
                                # Extract instrument details
                                symbol = instrument.get('Product', {}).get('symbol', '')
                                security_type = instrument.get('Product', {}).get('securityType', '')
                                quantity = float(instrument.get('quantity', 0))
                                order_action = instrument.get('orderAction', '')
                                order_term = detail.get('orderTerm', 'GOOD_FOR_DAY')
                                price_type = detail.get('priceType', 'MARKET')
                                limit_price = float(detail.get('limitPrice', 0)) if 'limitPrice' in detail else None
                                stop_price = float(detail.get('stopPrice', 0)) if 'stopPrice' in detail else None
                                executed_quantity = float(instrument.get('filledQuantity', 0))
                                average_execution_price = float(instrument.get('averageExecutionPrice', 0))
                                
                                # Map order_action to OrderSide
                                side = OrderSide.BUY
                                if order_action.upper() in ['SELL', 'SELL_SHORT']:
                                    side = OrderSide.SELL
                                
                                # Map price_type to OrderType
                                order_type = OrderType.MARKET
                                if price_type == 'LIMIT':
                                    order_type = OrderType.LIMIT
                                elif price_type == 'STOP':
                                    order_type = OrderType.STOP
                                elif price_type == 'STOP_LIMIT':
                                    order_type = OrderType.STOP_LIMIT
                                elif 'TRAILING' in price_type:
                                    order_type = OrderType.TRAILING_STOP
                                
                                # Map order_term to TimeInForce
                                time_in_force = TimeInForce.DAY
                                if order_term == 'GOOD_UNTIL_CANCEL':
                                    time_in_force = TimeInForce.GTC
                                elif order_term == 'IMMEDIATE_OR_CANCEL':
                                    time_in_force = TimeInForce.IOC
                                elif order_term == 'FILL_OR_KILL':
                                    time_in_force = TimeInForce.FOK
                                
                                # Determine asset type
                                asset_type = AssetType.STOCK
                                if security_type:
                                    sec_type = security_type.lower()
                                    if 'option' in sec_type:
                                        asset_type = AssetType.OPTION
                                    elif 'future' in sec_type or 'fut' in sec_type:
                                        asset_type = AssetType.FUTURE
                                    elif 'forex' in sec_type or 'fx' in sec_type or 'currency' in sec_type:
                                        asset_type = AssetType.FOREX
                                
                                # Create Order object
                                order = Order(
                                    symbol=symbol,
                                    quantity=quantity,
                                    side=side,
                                    order_type=order_type,
                                    time_in_force=time_in_force,
                                    limit_price=limit_price,
                                    stop_price=stop_price,
                                    client_order_id=order_data.get('clientOrderId', ''),
                                    broker_order_id=order_id,
                                    status=order_status_enum,
                                    filled_quantity=executed_quantity,
                                    filled_avg_price=average_execution_price,
                                    asset_type=asset_type,
                                    broker_id=self.get_broker_name(),
                                    created_at=created_at,
                                    updated_at=updated_at,
                                    notes=f"Preview ID: {order_data.get('previewId', '')}"
                                )
                                orders.append(order)
            
            # Only cache if not filtering by status
            if status is None:
                self.order_cache[cache_key] = orders
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
            
            return orders
        
        except Exception as e:
            logger.error(f"Failed to get orders: {str(e)}")
            raise BrokerConnectionError(f"Failed to get orders: {str(e)}")
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get the status of an order from E*TRADE.
        
        Args:
            order_id: Order ID to get status for
            
        Returns:
            Order: Order with current status
            
        Raises:
            BrokerConnectionError: If connection is not established
            BrokerOrderError: If order status request fails
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Get the order details
            response = self._make_request('GET', f'/v1/accounts/{self.account_id}/orders/{order_id}')
            
            if 'OrdersResponse' not in response or 'Order' not in response['OrdersResponse']:
                raise BrokerOrderError(f"Order {order_id} not found")
            
            order_data = response['OrdersResponse']['Order']
            if isinstance(order_data, list) and len(order_data) > 0:
                order_data = order_data[0]  # Use the first order if multiple are returned
            
            # Extract order information (similar to get_orders method)
            # Get the main order information
            placed_time = order_data.get('placedTime', '')
            executed_time = order_data.get('executedTime', '')
            order_status = order_data.get('status', '')
            
            # Parse times if available
            created_at = None
            updated_at = None
            
            if placed_time:
                try:
                    created_at = datetime.fromisoformat(placed_time.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        created_at = datetime.strptime(placed_time, '%m/%d/%Y %H:%M:%S')
                    except ValueError:
                        created_at = None
            
            if executed_time:
                try:
                    updated_at = datetime.fromisoformat(executed_time.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        updated_at = datetime.strptime(executed_time, '%m/%d/%Y %H:%M:%S')
                    except ValueError:
                        updated_at = None
            
            # Map E*TRADE status to our OrderStatus
            order_status_enum = OrderStatus.NEW
            if order_status == 'EXECUTED':
                order_status_enum = OrderStatus.FILLED
            elif order_status == 'PARTIAL':
                order_status_enum = OrderStatus.PARTIALLY_FILLED
            elif order_status == 'CANCELLED':
                order_status_enum = OrderStatus.CANCELLED
            elif order_status == 'REJECTED':
                order_status_enum = OrderStatus.REJECTED
            elif order_status == 'PENDING':
                order_status_enum = OrderStatus.PENDING
            elif order_status == 'EXPIRED':
                order_status_enum = OrderStatus.EXPIRED
            
            # Get the order details - we'll use the first leg/instrument for simplicity
            if 'OrderDetail' not in order_data:
                raise BrokerOrderError(f"Order {order_id} has no details")
                
            order_detail = order_data['OrderDetail']
            if isinstance(order_detail, list):
                order_detail = order_detail[0]  # Use the first detail if multiple
            
            if 'Instrument' not in order_detail:
                raise BrokerOrderError(f"Order {order_id} has no instruments")
                
            instrument = order_detail['Instrument']
            if isinstance(instrument, list):
                instrument = instrument[0]  # Use the first instrument if multiple
            
            # Extract instrument details
            symbol = instrument.get('Product', {}).get('symbol', '')
            security_type = instrument.get('Product', {}).get('securityType', '')
            quantity = float(instrument.get('quantity', 0))
            order_action = instrument.get('orderAction', '')
            order_term = order_detail.get('orderTerm', 'GOOD_FOR_DAY')
            price_type = order_detail.get('priceType', 'MARKET')
            limit_price = float(order_detail.get('limitPrice', 0)) if 'limitPrice' in order_detail else None
            stop_price = float(order_detail.get('stopPrice', 0)) if 'stopPrice' in order_detail else None
            executed_quantity = float(instrument.get('filledQuantity', 0))
            average_execution_price = float(instrument.get('averageExecutionPrice', 0))
            
            # Map order_action to OrderSide
            side = OrderSide.BUY
            if order_action.upper() in ['SELL', 'SELL_SHORT']:
                side = OrderSide.SELL
            
            # Map price_type to OrderType
            order_type = OrderType.MARKET
            if price_type == 'LIMIT':
                order_type = OrderType.LIMIT
            elif price_type == 'STOP':
                order_type = OrderType.STOP
            elif price_type == 'STOP_LIMIT':
                order_type = OrderType.STOP_LIMIT
            elif 'TRAILING' in price_type:
                order_type = OrderType.TRAILING_STOP
            
            # Map order_term to TimeInForce
            time_in_force = TimeInForce.DAY
            if order_term == 'GOOD_UNTIL_CANCEL':
                time_in_force = TimeInForce.GTC
            elif order_term == 'IMMEDIATE_OR_CANCEL':
                time_in_force = TimeInForce.IOC
            elif order_term == 'FILL_OR_KILL':
                time_in_force = TimeInForce.FOK
            
            # Determine asset type
            asset_type = AssetType.STOCK
            if security_type:
                sec_type = security_type.lower()
                if 'option' in sec_type:
                    asset_type = AssetType.OPTION
                elif 'future' in sec_type or 'fut' in sec_type:
                    asset_type = AssetType.FUTURE
                elif 'forex' in sec_type or 'fx' in sec_type or 'currency' in sec_type:
                    asset_type = AssetType.FOREX
            
            # Create Order object
            order = Order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=order_data.get('clientOrderId', ''),
                broker_order_id=order_id,
                status=order_status_enum,
                filled_quantity=executed_quantity,
                filled_avg_price=average_execution_price,
                asset_type=asset_type,
                broker_id=self.get_broker_name(),
                created_at=created_at,
                updated_at=updated_at,
                notes=f"Preview ID: {order_data.get('previewId', '')}"
            )
            
            return order
        
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            raise BrokerOrderError(f"Failed to get order status for {order_id}: {str(e)}")
    
    def place_order(self, order: Order) -> Order:
        """
        Place an order with E*TRADE.
        
        Args:
            order: Order to place
            
        Returns:
            Order: Updated order with broker order ID and status
            
        Raises:
            BrokerConnectionError: If connection is not established
            BrokerOrderError: If order placement fails
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Map our OrderSide to E*TRADE order action
            order_action = 'BUY'
            if order.side == OrderSide.SELL:
                order_action = 'SELL'
            
            # Map our OrderType to E*TRADE price type
            price_type = 'MARKET'
            if order.order_type == OrderType.LIMIT:
                price_type = 'LIMIT'
            elif order.order_type == OrderType.STOP:
                price_type = 'STOP'
            elif order.order_type == OrderType.STOP_LIMIT:
                price_type = 'STOP_LIMIT'
            elif order.order_type == OrderType.TRAILING_STOP:
                price_type = 'TRAILING_STOP'
            
            # Map our TimeInForce to E*TRADE order term
            order_term = 'GOOD_FOR_DAY'
            if order.time_in_force == TimeInForce.GTC:
                order_term = 'GOOD_UNTIL_CANCEL'
            elif order.time_in_force == TimeInForce.IOC:
                order_term = 'IMMEDIATE_OR_CANCEL'
            elif order.time_in_force == TimeInForce.FOK:
                order_term = 'FILL_OR_KILL'
            
            # Determine security type based on asset type
            security_type = 'EQ'
            if order.asset_type == AssetType.OPTION:
                security_type = 'OPTN'
            elif order.asset_type == AssetType.FUTURE:
                security_type = 'FUT'
            elif order.asset_type == AssetType.FOREX:
                security_type = 'FOREX'
            
            # Create order preview request
            order_request = {
                "PreviewOrderRequest": {
                    "orderType": "EQ",
                    "clientOrderId": order.client_order_id if order.client_order_id else str(uuid.uuid4()),
                    "Order": {
                        "allOrNone": False,
                        "priceType": price_type,
                        "orderTerm": order_term,
                        "marketSession": "REGULAR",
                        "stopPrice": str(order.stop_price) if order.stop_price is not None else None,
                        "limitPrice": str(order.limit_price) if order.limit_price is not None else None,
                        "Instrument": [
                            {
                                "Product": {
                                    "securityType": security_type,
                                    "symbol": order.symbol
                                },
                                "orderAction": order_action,
                                "quantityType": "QUANTITY",
                                "quantity": str(int(order.quantity))
                            }
                        ]
                    }
                }
            }
            
            # Remove None values from the request
            # E*TRADE doesn't like null values in the JSON
            def clean_dict(d):
                if not isinstance(d, dict):
                    return d
                return {k: clean_dict(v) for k, v in d.items() if v is not None}
            
            order_request = clean_dict(order_request)
            
            # Preview the order first (E*TRADE typically requires a preview before placing an order)
            preview_response = self._make_request('POST', f'/v1/accounts/{self.account_id}/orders/preview', data=order_request)
            
            if 'PreviewOrderResponse' not in preview_response:
                raise BrokerOrderError("Failed to preview order: Invalid response")
            
            preview_data = preview_response['PreviewOrderResponse']
            preview_id = preview_data.get('PreviewIds', {}).get('previewId', None)
            
            if not preview_id:
                raise BrokerOrderError("Failed to get order preview ID")
            
            # Now place the actual order using the preview ID
            place_request = {
                "PlaceOrderRequest": {
                    "orderType": "EQ",
                    "clientOrderId": order.client_order_id if order.client_order_id else str(uuid.uuid4()),
                    "PreviewIds": {
                        "previewId": preview_id
                    },
                    "Order": {
                        "allOrNone": False,
                        "priceType": price_type,
                        "orderTerm": order_term,
                        "marketSession": "REGULAR",
                        "stopPrice": str(order.stop_price) if order.stop_price is not None else None,
                        "limitPrice": str(order.limit_price) if order.limit_price is not None else None,
                        "Instrument": [
                            {
                                "Product": {
                                    "securityType": security_type,
                                    "symbol": order.symbol
                                },
                                "orderAction": order_action,
                                "quantityType": "QUANTITY",
                                "quantity": str(int(order.quantity))
                            }
                        ]
                    }
                }
            }
            
            place_request = clean_dict(place_request)
            
            # Place the order
            place_response = self._make_request('POST', f'/v1/accounts/{self.account_id}/orders/place', data=place_request)
            
            if 'PlaceOrderResponse' not in place_response:
                raise BrokerOrderError("Failed to place order: Invalid response")
            
            place_data = place_response['PlaceOrderResponse']
            
            # Extract order ID from response
            order_id = None
            if 'OrderIds' in place_data and 'orderId' in place_data['OrderIds']:
                order_id = place_data['OrderIds']['orderId']
            
            if not order_id:
                raise BrokerOrderError("Order was placed but no order ID was returned")
            
            # Update the order with the broker order ID and status
            order.broker_order_id = order_id
            order.status = OrderStatus.NEW  # Initially set to NEW
            order.created_at = datetime.now()
            order.updated_at = datetime.now()
            
            # Fetch the full order details to get all fields
            try:
                updated_order = self.get_order_status(order_id)
                
                # Copy fields from updated order to our order object
                order.status = updated_order.status
                order.filled_quantity = updated_order.filled_quantity
                order.filled_avg_price = updated_order.filled_avg_price
                order.updated_at = updated_order.updated_at
                order.notes = updated_order.notes
            except Exception as e:
                logger.warning(f"Order was placed but failed to get full order details: {str(e)}")
            
            # Clear the orders cache since we've made a change
            cache_key = f'orders_{self.account_id}'
            if cache_key in self.order_cache:
                del self.order_cache[cache_key]
                del self.cache_expiry[cache_key]
            
            return order
        
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise BrokerOrderError(f"Failed to place order: {str(e)}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with E*TRADE.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if order was cancelled successfully
            
        Raises:
            BrokerConnectionError: If connection is not established
            BrokerOrderError: If order cancellation fails
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Create cancel request
            cancel_request = {
                "CancelOrderRequest": {
                    "orderId": order_id
                }
            }
            
            # Cancel the order
            response = self._make_request('PUT', f'/v1/accounts/{self.account_id}/orders/cancel', data=cancel_request)
            
            if 'CancelOrderResponse' not in response:
                raise BrokerOrderError("Failed to cancel order: Invalid response")
            
            cancel_data = response['CancelOrderResponse']
            
            # Check if the cancellation was successful
            if 'ErrorMessage' in cancel_data:
                raise BrokerOrderError(f"Failed to cancel order: {cancel_data['ErrorMessage']}")
            
            # Clear the orders cache since we've made a change
            cache_key = f'orders_{self.account_id}'
            if cache_key in self.order_cache:
                del self.order_cache[cache_key]
                del self.cache_expiry[cache_key]
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise BrokerOrderError(f"Failed to cancel order {order_id}: {str(e)}")
    
    def get_quote(self, symbol: str, asset_type: AssetType = AssetType.STOCK) -> Quote:
        """
        Get a quote for a symbol from E*TRADE.
        
        Args:
            symbol: Symbol to get quote for
            asset_type: Type of asset (stock, option, futures, forex, crypto)
            
        Returns:
            Quote: Quote information
            
        Raises:
            BrokerConnectionError: If connection is not established
        """
        # Check if we have cached data that's still valid
        cache_key = f'quote_{symbol}_{asset_type.name}'
        if (cache_key in self.quote_cache and 
            cache_key in self.cache_expiry and 
            datetime.now() < self.cache_expiry[cache_key]):
            return self.quote_cache[cache_key]
        
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        try:
            # Build detail flag for desired fields
            detail_flag = 'ALL'  # Get all available data
            
            # Get quote data
            params = {
                'symbol': symbol,
                'detailFlag': detail_flag
            }
            
            response = self._make_request('GET', '/v1/market/quote', params=params)
            
            if 'QuoteResponse' not in response or 'QuoteData' not in response['QuoteResponse']:
                raise BrokerConnectionError(f"No quote data found for {symbol}")
            
            quote_data_list = response['QuoteResponse']['QuoteData']
            if not isinstance(quote_data_list, list):
                quote_data_list = [quote_data_list]
            
            # Find the quote for our symbol
            quote_data = None
            for data in quote_data_list:
                if 'Product' in data and data['Product'].get('symbol') == symbol:
                    quote_data = data
                    break
            
            if not quote_data:
                raise BrokerConnectionError(f"Symbol {symbol} not found in quote response")
            
            # Extract quote details
            product = quote_data.get('Product', {})
            all_data = quote_data.get('All', {})
            
            # Create quote object
            quote = Quote(
                symbol=symbol,
                bid_price=float(all_data.get('bid', 0)),
                ask_price=float(all_data.get('ask', 0)),
                last_price=float(all_data.get('lastTrade', 0)),
                volume=int(all_data.get('totalVolume', 0)),
                timestamp=datetime.now(),  # Use current time as timestamp
                exchange=product.get('exchange', ''),
                asset_type=asset_type,
                open_price=float(all_data.get('open', 0)) if 'open' in all_data else None,
                high_price=float(all_data.get('high52', 0)) if 'high52' in all_data else None,
                low_price=float(all_data.get('low52', 0)) if 'low52' in all_data else None,
                close_price=float(all_data.get('previousClose', 0)) if 'previousClose' in all_data else None,
                description=product.get('securityName', ''),
                bid_size=int(all_data.get('bidSize', 0)) if 'bidSize' in all_data else 0,
                ask_size=int(all_data.get('askSize', 0)) if 'askSize' in all_data else 0,
                change_amount=float(all_data.get('changeClose', 0)) if 'changeClose' in all_data else 0,
                change_percentage=float(all_data.get('changeClosePercentage', 0)) if 'changeClosePercentage' in all_data else 0
            )
            
            # Update cache
            self.quote_cache[cache_key] = quote
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
            
            return quote
        
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {str(e)}")
            raise BrokerConnectionError(f"Failed to get quote for {symbol}: {str(e)}")
    
    def get_bars(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None, limit: Optional[int] = None, asset_type: AssetType = AssetType.STOCK) -> List[Bar]:
        """
        Get historical bars for a symbol from E*TRADE.
        
        Args:
            symbol: Symbol to get bars for
            timeframe: Timeframe (1M, 5M, 15M, 1H, 1D, etc.)
            start: Start date/time
            end: End date/time (default: now)
            limit: Maximum number of bars to return
            asset_type: Type of asset (stock, option, futures, forex, crypto)
            
        Returns:
            List[Bar]: List of bars
            
        Raises:
            BrokerConnectionError: If connection is not established
        """
        # Check if we have cached data that's still valid
        cache_key = f'bars_{symbol}_{timeframe}_{start.isoformat()}_{end.isoformat() if end else "now"}_{asset_type.name}'
        if (cache_key in self.bars_cache and 
            cache_key in self.cache_expiry and 
            datetime.now() < self.cache_expiry[cache_key]):
            return self.bars_cache[cache_key]
        
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to E*TRADE")
        
        try:
            # Set end time to now if not provided
            if not end:
                end = datetime.now()
            
            # Map our timeframe to E*TRADE interval
            # E*TRADE supports: m1, m5, m10, m30, h1, d1, w1
            interval = 'd1'  # Default to daily
            
            if timeframe == '1M':
                interval = 'm1'
            elif timeframe == '5M':
                interval = 'm5'
            elif timeframe == '10M':
                interval = 'm10'
            elif timeframe == '15M':
                # No direct match, use closest (m10)
                interval = 'm10'
            elif timeframe == '30M':
                interval = 'm30'
            elif timeframe == '1H':
                interval = 'h1'
            elif timeframe == '1D':
                interval = 'd1'
            elif timeframe == '1W':
                interval = 'w1'
            
            # Format dates for E*TRADE API
            start_str = start.strftime('%Y%m%d')
            end_str = end.strftime('%Y%m%d')
            
            # Get historical data
            params = {
                'symbol': symbol,
                'interval': interval,
                'startDate': start_str,
                'endDate': end_str
            }
            
            response = self._make_request('GET', '/v1/market/historicaldata', params=params)
            
            if 'HistoricalDataResponse' not in response or 'HistoricalData' not in response['HistoricalDataResponse']:
                return []  # Return empty list if no data
            
            historical_data = response['HistoricalDataResponse']['HistoricalData']
            bars = []
            
            for bar_data in historical_data:
                # E*TRADE timestamp is in format like "2023-04-12T16:00:00-04:00"
                timestamp = bar_data.get('dateTime', None)
                
                # Parse timestamp
                bar_time = None
                if timestamp:
                    try:
                        bar_time = datetime.fromisoformat(timestamp)
                    except ValueError:
                        try:
                            # Try another format if ISO format fails
                            bar_time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')
                        except ValueError:
                            bar_time = None
                
                if not bar_time:
                    continue  # Skip entries without valid timestamp
                
                # Create Bar object
                bar = Bar(
                    symbol=symbol,
                    open_price=float(bar_data.get('open', 0)),
                    high_price=float(bar_data.get('high', 0)),
                    low_price=float(bar_data.get('low', 0)),
                    close_price=float(bar_data.get('close', 0)),
                    volume=int(bar_data.get('volume', 0)),
                    timestamp=bar_time,
                    timeframe=timeframe,
                    asset_type=asset_type
                )
                bars.append(bar)
            
            # Apply limit if specified
            if limit and len(bars) > limit:
                bars = bars[:limit]
            
            # Update cache
            if bars:
                self.bars_cache[cache_key] = bars
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
            
            return bars
        
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {str(e)}")
            raise BrokerConnectionError(f"Failed to get bars for {symbol}: {str(e)}")
    
    def get_broker_name(self) -> str:
        """
        Get the name of the broker.
        
        Returns:
            str: Broker name
        """
        return "E*TRADE"
    
    def get_supported_asset_types(self) -> List[AssetType]:
        """
        Get the asset types supported by this broker.
        
        Returns:
            List[AssetType]: List of supported asset types
        """
        return [AssetType.STOCK, AssetType.OPTION, AssetType.FUTURE, AssetType.FOREX]
