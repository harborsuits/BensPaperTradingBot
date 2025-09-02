"""
TradeStation Brokerage API Client

This module implements the BrokerInterface for TradeStation API,
providing integration with TradeStation for trading stocks, options, and futures.
"""

import os
import logging
import time
import json
import requests
import base64
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import threading
from urllib.parse import urlencode

from .broker_interface import (
    BrokerInterface, BrokerCredentials, BrokerAccount, Position, Order, Quote, Bar,
    AssetType, OrderType, OrderSide, OrderStatus, TimeInForce,
    BrokerAuthenticationError, BrokerConnectionError, BrokerOrderError
)

# Configure logging
logger = logging.getLogger(__name__)


class TradeStationClient(BrokerInterface):
    """Implementation of BrokerInterface for TradeStation API."""
    
    def __init__(self, sandbox: bool = True):
        """
        Initialize the TradeStation client.
        
        Args:
            sandbox: Whether to use TradeStation's sandbox environment
        """
        self.session = requests.Session()
        self.connected = False
        self.credentials = None
        self.sandbox = sandbox
        
        # API endpoints
        self.base_url = "https://sim-api.tradestation.com" if sandbox else "https://api.tradestation.com"
        self.auth_url = "https://signin.tradestation.com/oauth/token"
        
        # Authentication state
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None
        
        # Rate limiting and API state
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        self.last_request_time = datetime.now()
        self.min_request_interval = 0.2  # seconds between requests to avoid overwhelming the API
        
        # Account state
        self.account_id = None  # Selected account ID
        self.available_accounts = []
        
        # Token refresh management
        self.token_refresh_thread = None
        self.token_refresh_active = False
        self.token_refresh_lock = threading.Lock()
        
        # Cache for frequently accessed data
        self.account_cache = {}
        self.position_cache = {}
        self.order_cache = {}
        self.quote_cache = {}
        self.cache_expiry = {}  # Timestamp when each cache expires
        self.cache_ttl = 5  # seconds

    def connect(self, credentials: BrokerCredentials) -> bool:
        """
        Connect to TradeStation API using OAuth2.
        
        Args:
            credentials: Authentication credentials with client_id and client_secret
            
        Returns:
            bool: Success status
            
        Raises:
            BrokerAuthenticationError: If authentication fails
            BrokerConnectionError: If connection fails
        """
        if not credentials.additional_params:
            credentials.additional_params = {}
        
        client_id = credentials.api_key
        client_secret = credentials.api_secret
        
        # Additional authentication parameters
        redirect_uri = credentials.additional_params.get('redirect_uri', 'http://localhost')
        scope = credentials.additional_params.get('scope', 'ReadAccount TradeAccount')
        
        # Handle two authentication flows:
        # 1. With access_token/refresh_token (from previous auth or provided)
        # 2. With authorization code (requires user to have completed web auth flow)
        
        self.credentials = credentials
        
        # If access token provided directly, use it
        if credentials.access_token and credentials.refresh_token:
            self.access_token = credentials.access_token
            self.refresh_token = credentials.refresh_token
            
            # Test the token
            try:
                self._test_connection()
                self.connected = True
                self._start_token_refresh()
                
                # Set account ID if provided
                if credentials.account_id:
                    self.account_id = credentials.account_id
                else:
                    # Fetch accounts and use the first one
                    self._set_default_account()
                
                return True
            except (BrokerAuthenticationError, BrokerConnectionError):
                # Token might be expired, try refreshing it
                try:
                    self._refresh_access_token()
                    self._test_connection()
                    self.connected = True
                    self._start_token_refresh()
                    
                    # Set account ID
                    if credentials.account_id:
                        self.account_id = credentials.account_id
                    else:
                        self._set_default_account()
                        
                    return True
                except Exception as e:
                    logger.error(f"Failed to refresh token: {str(e)}")
                    return False
        
        # If authorization code is provided, exchange it for tokens
        if 'authorization_code' in credentials.additional_params:
            try:
                code = credentials.additional_params['authorization_code']
                
                # Exchange code for token
                token_data = {
                    'grant_type': 'authorization_code',
                    'code': code,
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'redirect_uri': redirect_uri
                }
                
                response = requests.post(self.auth_url, data=token_data)
                if response.status_code != 200:
                    logger.error(f"Authorization code exchange failed: {response.text}")
                    raise BrokerAuthenticationError(f"Failed to exchange authorization code: {response.text}")
                
                token_response = response.json()
                self.access_token = token_response['access_token']
                self.refresh_token = token_response['refresh_token']
                
                # Calculate token expiry time
                expires_in = token_response.get('expires_in', 1800)  # Default to 30 minutes
                self.expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # Refresh 1 minute before expiry
                
                # Test connection
                self._test_connection()
                self.connected = True
                self._start_token_refresh()
                
                # Set account ID
                if credentials.account_id:
                    self.account_id = credentials.account_id
                else:
                    self._set_default_account()
                
                return True
            
            except Exception as e:
                logger.error(f"Authentication failed: {str(e)}")
                raise BrokerAuthenticationError(f"TradeStation authentication failed: {str(e)}")
        
        # If we got here, we don't have enough authentication info
        auth_url = (
            f"https://signin.tradestation.com/authorize?response_type=code&client_id={client_id}&"
            f"redirect_uri={redirect_uri}&scope={scope}"
        )
        
        logger.error(f"TradeStation requires OAuth2 authorization. Visit: {auth_url}")
        raise BrokerAuthenticationError(
            f"TradeStation requires OAuth2 authorization. Visit:\n{auth_url}\n"
            f"After authorization, provide the 'code' parameter from the redirect URL."
        )

    def disconnect(self) -> bool:
        """
        Disconnect from TradeStation API.
        
        Returns:
            bool: Success status
        """
        with self.token_refresh_lock:
            self.token_refresh_active = False
        
        if self.token_refresh_thread and self.token_refresh_thread.is_alive():
            self.token_refresh_thread.join(timeout=5.0)
            self.token_refresh_thread = None
        
        self.access_token = None
        self.refresh_token = None
        self.connected = False
        self.session.close()
        
        logger.info("Disconnected from TradeStation API")
        return True

    def is_connected(self) -> bool:
        """
        Check if connected to TradeStation API.
        
        Returns:
            bool: Connection status
        """
        if not self.connected or not self.access_token:
            return False
        
        # If token is expired or expiring soon and no refresh is in progress
        if self.expires_at and datetime.now() > self.expires_at:
            try:
                self._refresh_access_token()
            except Exception as e:
                logger.error(f"Token refresh failed during connection check: {str(e)}")
                return False
        
        # Test connection if we haven't made a request recently
        if (datetime.now() - self.last_request_time).total_seconds() > 300:  # 5 minutes
            try:
                self._test_connection()
            except Exception:
                return False
        
        return True

    def _test_connection(self) -> bool:
        """
        Test connection to TradeStation API.
        
        Returns:
            bool: Success status
            
        Raises:
            BrokerAuthenticationError: If authentication fails
            BrokerConnectionError: If connection fails
        """
        try:
            # Make a simple API call to verify connection
            response = self._make_request('GET', '/v3/brokerage/accounts')
            
            if not response or 'accounts' not in response:
                raise BrokerConnectionError("TradeStation API response missing account data")
            
            return True
        
        except requests.exceptions.RequestException as e:
            raise BrokerConnectionError(f"TradeStation connection error: {str(e)}")

    def _refresh_access_token(self) -> bool:
        """
        Refresh the OAuth2 access token.
        
        Returns:
            bool: Success status
            
        Raises:
            BrokerAuthenticationError: If token refresh fails
        """
        with self.token_refresh_lock:
            if not self.refresh_token:
                raise BrokerAuthenticationError("No refresh token available")
            
            try:
                # Prepare token refresh request
                token_data = {
                    'grant_type': 'refresh_token',
                    'refresh_token': self.refresh_token,
                    'client_id': self.credentials.api_key,
                    'client_secret': self.credentials.api_secret,
                }
                
                response = requests.post(self.auth_url, data=token_data)
                if response.status_code != 200:
                    logger.error(f"Token refresh failed: {response.text}")
                    raise BrokerAuthenticationError(f"Failed to refresh token: {response.text}")
                
                token_response = response.json()
                self.access_token = token_response['access_token']
                self.refresh_token = token_response['refresh_token']
                
                # Calculate token expiry time
                expires_in = token_response.get('expires_in', 1800)  # Default to 30 minutes
                self.expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # Refresh 1 minute before expiry
                
                logger.info("TradeStation API token refreshed successfully")
                return True
            
            except Exception as e:
                logger.error(f"Token refresh failed: {str(e)}")
                raise BrokerAuthenticationError(f"Failed to refresh token: {str(e)}")

    def _start_token_refresh(self) -> None:
        """Start background thread for token refresh."""
        if self.token_refresh_thread and self.token_refresh_thread.is_alive():
            return  # Already running
        
        with self.token_refresh_lock:
            self.token_refresh_active = True
            self.token_refresh_thread = threading.Thread(
                target=self._token_refresh_loop,
                daemon=True,
                name="TradeStationTokenRefreshThread"
            )
            self.token_refresh_thread.start()
    
    def _token_refresh_loop(self) -> None:
        """Background thread for automatic token refresh."""
        logger.info("TradeStation token refresh thread started")
        
        while self.token_refresh_active:
            try:
                if self.expires_at:
                    # Time until token expires
                    now = datetime.now()
                    time_to_expiry = (self.expires_at - now).total_seconds()
                    
                    if time_to_expiry <= 0:
                        # Token expired, refresh now
                        logger.info("Access token expired, refreshing")
                        self._refresh_access_token()
                    elif time_to_expiry < 300:  # Less than 5 minutes left
                        # Refresh token soon
                        logger.info(f"Access token expiring in {time_to_expiry:.1f} seconds, refreshing")
                        self._refresh_access_token()
                    else:
                        # Calculate sleep time (check every minute or half of time remaining)
                        sleep_time = min(60, time_to_expiry / 2)
                        time.sleep(sleep_time)
                else:
                    # No expiry info, check every 10 minutes
                    time.sleep(600)
            
            except Exception as e:
                logger.error(f"Error in token refresh thread: {str(e)}")
                # Sleep briefly after error to avoid tight loop
                time.sleep(30)
        
        logger.info("TradeStation token refresh thread stopped")

    def _set_default_account(self) -> None:
        """Set the default account ID by fetching accounts."""
        try:
            accounts = self.get_account_info()
            if accounts:
                self.account_id = accounts[0].account_id
                logger.info(f"Set default TradeStation account: {self.account_id}")
            else:
                logger.warning("No TradeStation accounts found")
        except Exception as e:
            logger.error(f"Error setting default account: {str(e)}")

    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Any:
        """
        Make an API request to TradeStation.
        
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
        if not self.access_token:
            raise BrokerAuthenticationError("Not authenticated with TradeStation")
        
        # Rate limiting: ensure we don't send requests too quickly
        elapsed = (datetime.now() - self.last_request_time).total_seconds()
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        # Construct full URL
        url = f"{self.base_url}{endpoint}"
        
        # Set up headers
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            # Make the request
            if method == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, params=params, json=data)
            elif method == 'PUT':
                response = self.session.put(url, headers=headers, params=params, json=data)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Update rate limit info
            self.rate_limit_remaining = response.headers.get('X-Rate-Limit-Remaining')
            self.rate_limit_reset = response.headers.get('X-Rate-Limit-Reset')
            
            # Record last request time
            self.last_request_time = datetime.now()
            
            # Check for errors
            if response.status_code == 401:
                raise BrokerAuthenticationError("TradeStation authentication failed, token may be expired")
            elif response.status_code == 403:
                raise BrokerAuthenticationError("TradeStation access forbidden, insufficient permissions")
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
            logger.error(f"TradeStation API request failed ({method} {endpoint}): {str(e)}")
            if "401" in str(e):
                raise BrokerAuthenticationError(f"TradeStation authentication error: {str(e)}")
            elif any(error in str(e) for error in ["403", "404", "400"]):
                raise BrokerOrderError(f"TradeStation order error: {str(e)}")
            else:
                raise BrokerConnectionError(f"TradeStation connection error: {str(e)}")
        
        except json.JSONDecodeError as e:
            logger.error(f"TradeStation API returned invalid JSON: {str(e)}")
            raise BrokerConnectionError(f"TradeStation returned invalid response: {str(e)}")
            
    def get_account_info(self) -> List[BrokerAccount]:
        """
        Get account information from TradeStation.
        
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
            raise BrokerConnectionError("Not connected to TradeStation")
        
        try:
            # Get accounts data
            response = self._make_request('GET', '/v3/brokerage/accounts')
            
            accounts = []
            for acct_data in response.get('accounts', []):
                # Extract the basic account data
                account_id = acct_data.get('account_id')
                
                # Get detailed account balances
                balances = self._make_request('GET', f'/v3/brokerage/accounts/{account_id}/balances')
                
                # Create the account object
                account = BrokerAccount(
                    broker_id=self.get_broker_name(),
                    account_id=account_id,
                    account_number=acct_data.get('account_number', ''),
                    account_type=acct_data.get('account_type', ''),
                    buying_power=float(balances.get('buying_power', {}).get('amount', 0)),
                    cash_balance=float(balances.get('cash', {}).get('amount', 0)),
                    equity=float(balances.get('equity', {}).get('amount', 0))
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
        Get current positions from TradeStation.
        
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
            raise BrokerConnectionError("Not connected to TradeStation")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Get positions data
            response = self._make_request('GET', f'/v3/brokerage/accounts/{self.account_id}/positions')
            
            positions = []
            for pos_data in response.get('positions', []):
                # Determine asset type
                asset_type = AssetType.STOCK
                if 'security_type' in pos_data:
                    sec_type = pos_data['security_type'].lower()
                    if 'option' in sec_type:
                        asset_type = AssetType.OPTION
                    elif 'future' in sec_type:
                        asset_type = AssetType.FUTURE
                    elif 'forex' in sec_type:
                        asset_type = AssetType.FOREX
                    elif 'crypto' in sec_type:
                        asset_type = AssetType.CRYPTO
                
                # Create position object
                position = Position(
                    symbol=pos_data.get('symbol', ''),
                    quantity=float(pos_data.get('quantity', 0)),
                    avg_entry_price=float(pos_data.get('average_price', 0)),
                    current_price=float(pos_data.get('mark_price', 0)),
                    market_value=float(pos_data.get('market_value', 0)),
                    unrealized_pl=float(pos_data.get('unrealized_pl', 0)),
                    asset_type=asset_type,
                    broker_id=self.get_broker_name(),
                    broker_position_id=pos_data.get('position_id')
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
        Get orders from TradeStation.
        
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
            
            # If there's a status filter, apply it to cached data
            return self.order_cache[cache_key]
        
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to TradeStation")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Prepare filter parameters
            params = {}
            if status:
                # Map our OrderStatus to TradeStation status
                ts_status = None
                if status == OrderStatus.NEW:
                    ts_status = 'OPEN'
                elif status == OrderStatus.FILLED:
                    ts_status = 'FILLED'
                elif status == OrderStatus.PARTIALLY_FILLED:
                    ts_status = 'PARTIAL'
                elif status == OrderStatus.CANCELLED:
                    ts_status = 'CANCELED'
                elif status == OrderStatus.REJECTED:
                    ts_status = 'REJECTED'
                elif status == OrderStatus.PENDING:
                    ts_status = 'PENDING'
                elif status == OrderStatus.EXPIRED:
                    ts_status = 'EXPIRED'
                
                if ts_status:
                    params['status'] = ts_status
            
            # Get orders data
            response = self._make_request('GET', f'/v3/brokerage/accounts/{self.account_id}/orders', params=params)
            
            orders = []
            for order_data in response.get('orders', []):
                # Map TradeStation status to our OrderStatus
                order_status = OrderStatus.NEW
                ts_status = order_data.get('status', '').upper()
                
                if ts_status == 'FILLED':
                    order_status = OrderStatus.FILLED
                elif ts_status == 'PARTIAL':
                    order_status = OrderStatus.PARTIALLY_FILLED
                elif ts_status == 'CANCELED':
                    order_status = OrderStatus.CANCELLED
                elif ts_status == 'REJECTED':
                    order_status = OrderStatus.REJECTED
                elif ts_status == 'PENDING':
                    order_status = OrderStatus.PENDING
                elif ts_status == 'EXPIRED':
                    order_status = OrderStatus.EXPIRED
                
                # Map TradeStation order type to our OrderType
                order_type = OrderType.MARKET
                ts_type = order_data.get('order_type', '').upper()
                
                if ts_type == 'LIMIT':
                    order_type = OrderType.LIMIT
                elif ts_type == 'STOP':
                    order_type = OrderType.STOP
                elif ts_type == 'STOP_LIMIT':
                    order_type = OrderType.STOP_LIMIT
                elif ts_type == 'TRAILING_STOP':
                    order_type = OrderType.TRAILING_STOP
                
                # Map TradeStation side to our OrderSide
                side = OrderSide.BUY
                ts_side = order_data.get('side', '').upper()
                
                if ts_side == 'SELL' or ts_side == 'SHORT':
                    side = OrderSide.SELL
                
                # Map TradeStation time in force to our TimeInForce
                time_in_force = TimeInForce.DAY
                ts_tif = order_data.get('time_in_force', '').upper()
                
                if ts_tif == 'GTC':
                    time_in_force = TimeInForce.GTC
                elif ts_tif == 'IOC':
                    time_in_force = TimeInForce.IOC
                elif ts_tif == 'FOK':
                    time_in_force = TimeInForce.FOK
                
                # Determine asset type
                asset_type = AssetType.STOCK
                if 'security_type' in order_data:
                    sec_type = order_data['security_type'].lower()
                    if 'option' in sec_type:
                        asset_type = AssetType.OPTION
                    elif 'future' in sec_type:
                        asset_type = AssetType.FUTURE
                    elif 'forex' in sec_type:
                        asset_type = AssetType.FOREX
                    elif 'crypto' in sec_type:
                        asset_type = AssetType.CRYPTO
                
                # Parse timestamps
                created_at = None
                updated_at = None
                
                if 'create_time' in order_data:
                    try:
                        created_at = datetime.fromisoformat(order_data['create_time'].replace('Z', '+00:00'))
                    except ValueError:
                        created_at = None
                
                if 'update_time' in order_data:
                    try:
                        updated_at = datetime.fromisoformat(order_data['update_time'].replace('Z', '+00:00'))
                    except ValueError:
                        updated_at = None
                
                # Create order object
                order = Order(
                    symbol=order_data.get('symbol', ''),
                    quantity=float(order_data.get('quantity', 0)),
                    side=side,
                    order_type=order_type,
                    time_in_force=time_in_force,
                    limit_price=float(order_data.get('limit_price', 0)) if 'limit_price' in order_data else None,
                    stop_price=float(order_data.get('stop_price', 0)) if 'stop_price' in order_data else None,
                    client_order_id=order_data.get('client_order_id'),
                    broker_order_id=order_data.get('order_id'),
                    status=order_status,
                    filled_quantity=float(order_data.get('filled_quantity', 0)),
                    filled_avg_price=float(order_data.get('filled_price', 0)) if 'filled_price' in order_data else None,
                    asset_type=asset_type,
                    broker_id=self.get_broker_name(),
                    created_at=created_at,
                    updated_at=updated_at,
                    notes=order_data.get('notes')
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
    
    def place_order(self, order: Order) -> Order:
        """
        Place an order with TradeStation.
        
        Args:
            order: Order to place
            
        Returns:
            Order: Updated order with broker order ID and status
            
        Raises:
            BrokerConnectionError: If connection is not established
            BrokerOrderError: If order placement fails
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to TradeStation")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Map our OrderSide to TradeStation side
            ts_side = 'BUY'
            if order.side == OrderSide.SELL:
                ts_side = 'SELL'
            
            # Map our OrderType to TradeStation order type
            ts_type = 'MARKET'
            if order.order_type == OrderType.LIMIT:
                ts_type = 'LIMIT'
            elif order.order_type == OrderType.STOP:
                ts_type = 'STOP'
            elif order.order_type == OrderType.STOP_LIMIT:
                ts_type = 'STOP_LIMIT'
            elif order.order_type == OrderType.TRAILING_STOP:
                ts_type = 'TRAILING_STOP'
            
            # Map our TimeInForce to TradeStation time in force
            ts_tif = 'DAY'
            if order.time_in_force == TimeInForce.GTC:
                ts_tif = 'GTC'
            elif order.time_in_force == TimeInForce.IOC:
                ts_tif = 'IOC'
            elif order.time_in_force == TimeInForce.FOK:
                ts_tif = 'FOK'
            
            # Determine security type based on asset type
            security_type = 'STOCK'
            if order.asset_type == AssetType.OPTION:
                security_type = 'OPTION'
            elif order.asset_type == AssetType.FUTURE:
                security_type = 'FUTURE'
            elif order.asset_type == AssetType.FOREX:
                security_type = 'FOREX'
            elif order.asset_type == AssetType.CRYPTO:
                security_type = 'CRYPTO'
            
            # Prepare order data
            order_data = {
                'account_id': self.account_id,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'side': ts_side,
                'order_type': ts_type,
                'time_in_force': ts_tif,
                'security_type': security_type
            }
            
            # Add order-type specific parameters
            if order.order_type == OrderType.LIMIT or order.order_type == OrderType.STOP_LIMIT:
                if order.limit_price is not None:
                    order_data['limit_price'] = order.limit_price
                else:
                    raise BrokerOrderError("Limit price is required for LIMIT and STOP_LIMIT orders")
            
            if order.order_type == OrderType.STOP or order.order_type == OrderType.STOP_LIMIT or order.order_type == OrderType.TRAILING_STOP:
                if order.stop_price is not None:
                    order_data['stop_price'] = order.stop_price
                else:
                    raise BrokerOrderError("Stop price is required for STOP, STOP_LIMIT, and TRAILING_STOP orders")
            
            # Add client order ID if provided
            if order.client_order_id:
                order_data['client_order_id'] = order.client_order_id
            
            # Place the order
            response = self._make_request('POST', '/v3/brokerage/orders', data=order_data)
            
            # Check for errors
            if 'errors' in response and response['errors']:
                error_msg = ', '.join([err.get('message', 'Unknown error') for err in response['errors']])
                raise BrokerOrderError(f"Failed to place order: {error_msg}")
            
            # Get the new order ID
            order_id = response.get('order_id')
            if not order_id:
                raise BrokerOrderError("Order was placed but no order ID was returned")
            
            # Update the order with the broker order ID and status
            order.broker_order_id = order_id
            order.status = OrderStatus.NEW  # Initially set to NEW, get actual status in next step
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
        Cancel an order with TradeStation.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if order was cancelled successfully
            
        Raises:
            BrokerConnectionError: If connection is not established
            BrokerOrderError: If order cancellation fails
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to TradeStation")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Cancel the order
            response = self._make_request('DELETE', f'/v3/brokerage/accounts/{self.account_id}/orders/{order_id}')
            
            # Check for errors
            if 'errors' in response and response['errors']:
                error_msg = ', '.join([err.get('message', 'Unknown error') for err in response['errors']])
                raise BrokerOrderError(f"Failed to cancel order: {error_msg}")
            
            # Clear the orders cache since we've made a change
            cache_key = f'orders_{self.account_id}'
            if cache_key in self.order_cache:
                del self.order_cache[cache_key]
                del self.cache_expiry[cache_key]
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise BrokerOrderError(f"Failed to cancel order {order_id}: {str(e)}")
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get the status of an order from TradeStation.
        
        Args:
            order_id: Order ID to get status for
            
        Returns:
            Order: Order with current status
            
        Raises:
            BrokerConnectionError: If connection is not established
            BrokerOrderError: If order status request fails
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to TradeStation")
        
        if not self.account_id:
            if not self.available_accounts:
                self.get_account_info()  # Fetch accounts if we don't have them
            if self.available_accounts:
                self.account_id = self.available_accounts[0].account_id
            else:
                raise BrokerConnectionError("No accounts available")
        
        try:
            # Get order status
            response = self._make_request('GET', f'/v3/brokerage/accounts/{self.account_id}/orders/{order_id}')
            
            # Check if the order exists
            if not response or 'order' not in response:
                raise BrokerOrderError(f"Order {order_id} not found")
            
            order_data = response['order']
            
            # Map TradeStation status to our OrderStatus
            order_status = OrderStatus.NEW
            ts_status = order_data.get('status', '').upper()
            
            if ts_status == 'FILLED':
                order_status = OrderStatus.FILLED
            elif ts_status == 'PARTIAL':
                order_status = OrderStatus.PARTIALLY_FILLED
            elif ts_status == 'CANCELED':
                order_status = OrderStatus.CANCELLED
            elif ts_status == 'REJECTED':
                order_status = OrderStatus.REJECTED
            elif ts_status == 'PENDING':
                order_status = OrderStatus.PENDING
            elif ts_status == 'EXPIRED':
                order_status = OrderStatus.EXPIRED
            
            # Map TradeStation order type to our OrderType
            order_type = OrderType.MARKET
            ts_type = order_data.get('order_type', '').upper()
            
            if ts_type == 'LIMIT':
                order_type = OrderType.LIMIT
            elif ts_type == 'STOP':
                order_type = OrderType.STOP
            elif ts_type == 'STOP_LIMIT':
                order_type = OrderType.STOP_LIMIT
            elif ts_type == 'TRAILING_STOP':
                order_type = OrderType.TRAILING_STOP
            
            # Map TradeStation side to our OrderSide
            side = OrderSide.BUY
            ts_side = order_data.get('side', '').upper()
            
            if ts_side == 'SELL' or ts_side == 'SHORT':
                side = OrderSide.SELL
            
            # Map TradeStation time in force to our TimeInForce
            time_in_force = TimeInForce.DAY
            ts_tif = order_data.get('time_in_force', '').upper()
            
            if ts_tif == 'GTC':
                time_in_force = TimeInForce.GTC
            elif ts_tif == 'IOC':
                time_in_force = TimeInForce.IOC
            elif ts_tif == 'FOK':
                time_in_force = TimeInForce.FOK
            
            # Determine asset type
            asset_type = AssetType.STOCK
            if 'security_type' in order_data:
                sec_type = order_data['security_type'].lower()
                if 'option' in sec_type:
                    asset_type = AssetType.OPTION
                elif 'future' in sec_type:
                    asset_type = AssetType.FUTURE
                elif 'forex' in sec_type:
                    asset_type = AssetType.FOREX
                elif 'crypto' in sec_type:
                    asset_type = AssetType.CRYPTO
            
            # Parse timestamps
            created_at = None
            updated_at = None
            
            if 'create_time' in order_data:
                try:
                    created_at = datetime.fromisoformat(order_data['create_time'].replace('Z', '+00:00'))
                except ValueError:
                    created_at = None
            
            if 'update_time' in order_data:
                try:
                    updated_at = datetime.fromisoformat(order_data['update_time'].replace('Z', '+00:00'))
                except ValueError:
                    updated_at = None
            
            # Create order object
            order = Order(
                symbol=order_data.get('symbol', ''),
                quantity=float(order_data.get('quantity', 0)),
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=float(order_data.get('limit_price', 0)) if 'limit_price' in order_data else None,
                stop_price=float(order_data.get('stop_price', 0)) if 'stop_price' in order_data else None,
                client_order_id=order_data.get('client_order_id'),
                broker_order_id=order_id,
                status=order_status,
                filled_quantity=float(order_data.get('filled_quantity', 0)),
                filled_avg_price=float(order_data.get('filled_price', 0)) if 'filled_price' in order_data else None,
                asset_type=asset_type,
                broker_id=self.get_broker_name(),
                created_at=created_at,
                updated_at=updated_at,
                notes=order_data.get('notes')
            )
            
            return order
        
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            raise BrokerOrderError(f"Failed to get order status for {order_id}: {str(e)}")
    
    def get_quote(self, symbol: str, asset_type: AssetType = AssetType.STOCK) -> Quote:
        """
        Get a quote for a symbol from TradeStation.
        
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
            raise BrokerConnectionError("Not connected to TradeStation")
        
        try:
            # Map our AssetType to TradeStation security type
            ts_asset_type = 'EQUITY'
            if asset_type == AssetType.OPTION:
                ts_asset_type = 'OPTION'
            elif asset_type == AssetType.FUTURE:
                ts_asset_type = 'FUTURE'
            elif asset_type == AssetType.FOREX:
                ts_asset_type = 'FOREX'
            elif asset_type == AssetType.CRYPTO:
                ts_asset_type = 'CRYPTO'
            
            # Get quote data
            params = {
                'symbols': symbol,
                'security_type': ts_asset_type
            }
            response = self._make_request('GET', '/v3/marketdata/quotes', params=params)
            
            # Check if we got a valid quote
            if not response or 'quotes' not in response or not response['quotes']:
                raise BrokerConnectionError(f"No quote data found for {symbol}")
            
            quote_data = response['quotes'][0]  # Get the first quote
            
            # Create quote object
            quote = Quote(
                symbol=symbol,
                bid_price=float(quote_data.get('bid', 0)),
                ask_price=float(quote_data.get('ask', 0)),
                last_price=float(quote_data.get('last', 0)),
                volume=int(quote_data.get('volume', 0)) if 'volume' in quote_data else 0,
                timestamp=datetime.now(),  # Use current time as timestamp
                exchange=quote_data.get('exchange', ''),
                asset_type=asset_type,
                open_price=float(quote_data.get('open', 0)) if 'open' in quote_data else None,
                high_price=float(quote_data.get('high', 0)) if 'high' in quote_data else None,
                low_price=float(quote_data.get('low', 0)) if 'low' in quote_data else None,
                close_price=float(quote_data.get('previous_close', 0)) if 'previous_close' in quote_data else None,
                description=quote_data.get('description', ''),
                bid_size=int(quote_data.get('bid_size', 0)) if 'bid_size' in quote_data else 0,
                ask_size=int(quote_data.get('ask_size', 0)) if 'ask_size' in quote_data else 0,
                change_amount=float(quote_data.get('net_change', 0)) if 'net_change' in quote_data else 0,
                change_percentage=float(quote_data.get('percent_change', 0)) if 'percent_change' in quote_data else 0
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
        Get historical bars for a symbol from TradeStation.
        
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
            raise BrokerConnectionError("Not connected to TradeStation")
        
        try:
            # Set end time to now if not provided
            if not end:
                end = datetime.now()
            
            # Map our timeframe to TradeStation interval
            ts_interval = 'M1'
            if timeframe == '1M':
                ts_interval = 'M1'
            elif timeframe == '5M':
                ts_interval = 'M5'
            elif timeframe == '15M':
                ts_interval = 'M15'
            elif timeframe == '30M':
                ts_interval = 'M30'
            elif timeframe == '1H':
                ts_interval = 'H1'
            elif timeframe == '4H':
                ts_interval = 'H4'
            elif timeframe == '1D':
                ts_interval = 'D1'
            elif timeframe == '1W':
                ts_interval = 'W1'
            else:
                # Try to map custom timeframe (e.g., '5M' -> 'M5')
                if timeframe.endswith('M'):
                    minutes = timeframe[:-1]
                    ts_interval = f'M{minutes}'
                elif timeframe.endswith('H'):
                    hours = timeframe[:-1]
                    ts_interval = f'H{hours}'
                elif timeframe.endswith('D'):
                    days = timeframe[:-1]
                    ts_interval = f'D{days}'
            
            # Map our AssetType to TradeStation security type
            ts_asset_type = 'EQUITY'
            if asset_type == AssetType.OPTION:
                ts_asset_type = 'OPTION'
            elif asset_type == AssetType.FUTURE:
                ts_asset_type = 'FUTURE'
            elif asset_type == AssetType.FOREX:
                ts_asset_type = 'FOREX'
            elif asset_type == AssetType.CRYPTO:
                ts_asset_type = 'CRYPTO'
            
            # Prepare request parameters
            params = {
                'symbol': symbol,
                'interval': ts_interval,
                'start_date': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'end_date': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'security_type': ts_asset_type
            }
            
            if limit:
                params['limit'] = limit
            
            # Get bars data
            response = self._make_request('GET', '/v3/marketdata/barcharts', params=params)
            
            # Check if we got valid data
            if not response or 'bars' not in response:
                return []  # Return empty list if no bars found
            
            bars = []
            for bar_data in response['bars']:
                # Parse timestamp
                bar_time = None
                if 'timestamp' in bar_data:
                    try:
                        bar_time = datetime.fromisoformat(bar_data['timestamp'].replace('Z', '+00:00'))
                    except ValueError:
                        bar_time = None
                
                # Create bar object
                bar = Bar(
                    symbol=symbol,
                    open_price=float(bar_data.get('open', 0)),
                    high_price=float(bar_data.get('high', 0)),
                    low_price=float(bar_data.get('low', 0)),
                    close_price=float(bar_data.get('close', 0)),
                    volume=int(bar_data.get('volume', 0)) if 'volume' in bar_data else 0,
                    timestamp=bar_time if bar_time else datetime.now(),
                    timeframe=timeframe,
                    asset_type=asset_type
                )
                bars.append(bar)
            
            # Update cache (only if not empty)
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
        return "TradeStation"
    
    def get_supported_asset_types(self) -> List[AssetType]:
        """
        Get the asset types supported by this broker.
        
        Returns:
            List[AssetType]: List of supported asset types
        """
        return [AssetType.STOCK, AssetType.OPTION, AssetType.FUTURE, AssetType.FOREX, AssetType.CRYPTO]
