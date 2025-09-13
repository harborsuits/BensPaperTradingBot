#!/usr/bin/env python3
"""
Tradier API Client for the Trading Bot

This module provides a client for interacting with the Tradier API,
which allows for account management, market data retrieval, and order execution.
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
from enum import Enum
from dateutil import parser
from requests.exceptions import RequestException, Timeout
from urllib.parse import urljoin
import pandas as pd

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Enum for order types supported by Tradier"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Enum for order sides"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"

class OrderDuration(Enum):
    """Enum for order durations"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    PRE = "pre"  # Pre-market
    POST = "post"  # Post-market

class OptionType(Enum):
    """Enum for option types"""
    CALL = "call"
    PUT = "put"

class TradierAPIError(Exception):
    """Exception raised for Tradier API errors."""
    
    def __init__(self, status_code: int, message: str, response_data: Any = None):
        self.status_code = status_code
        self.message = message
        self.response_data = response_data
        super().__init__(f"Tradier API Error ({status_code}): {message}")

class TradierClient:
    """
    Client for interacting with the Tradier API for trading and market data.
    
    This client handles authentication, request throttling, error handling, and
    provides methods for common API operations such as:
    - Account information retrieval
    - Market data (quotes, options chains)
    - Order placement and management
    - Position information
    """
    
    # API Endpoints
    SANDBOX_API_BASE_URL = "https://sandbox.tradier.com/v1/"
    LIVE_API_BASE_URL = "https://api.tradier.com/v1/"
    
    # Rate Limiting Constants
    MAX_REQUESTS_PER_SECOND = 2
    
    def __init__(
        self,
        api_key: str,
        account_id: str = None,
        use_sandbox: bool = True,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the Tradier API client.
        
        Args:
            api_key: The Tradier API access token
            account_id: The account ID to use for trading (optional, can be set later)
            use_sandbox: Whether to use the sandbox environment (default: True)
            timeout: Request timeout in seconds (default: 10)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Delay between retries in seconds (default: 1)
        """
        self.api_key = api_key
        self.account_id = account_id
        self.use_sandbox = use_sandbox
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up base URL based on environment
        self.base_url = self.SANDBOX_API_BASE_URL if use_sandbox else self.LIVE_API_BASE_URL
        
        # Rate limiting tracking
        self._last_request_time = 0
        
        logger.info(f"Initialized Tradier {'sandbox' if use_sandbox else 'live'} API client")
        
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers required for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
    def _handle_rate_limiting(self):
        """Handle rate limiting by adding appropriate delays between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        # If we've made a request less than 1 second ago, sleep to respect rate limit
        if time_since_last_request < (1.0 / self.MAX_REQUESTS_PER_SECOND):
            sleep_time = (1.0 / self.MAX_REQUESTS_PER_SECOND) - time_since_last_request
            time.sleep(sleep_time)
            
        self._last_request_time = time.time()
        
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict[str, Any] = None, 
        data: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the Tradier API with retries and error handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint to call
            params: URL parameters for the request
            data: Request body data
            json_data: JSON data for POST/PUT requests
            
        Returns:
            Dictionary containing the API response
            
        Raises:
            TradierAPIError: If the API returns an error
        """
        # Respect rate limits
        self._handle_rate_limiting()
        
        # Merge default headers with any additional headers
        request_headers = self._get_headers()
        
        # Full URL to the API endpoint
        url = urljoin(self.base_url, endpoint)
        
        # For data submission in POST/PUT requests
        if data and (method == "POST" or method == "PUT"):
            request_headers["Content-Type"] = "application/x-www-form-urlencoded"
            
        # Make the request with retries
        attempts = 0
        last_exception = None
        
        while attempts < self.max_retries:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    params=params,
                    data=data,
                    json=json_data,
                    timeout=self.timeout
                )
                
                # Handle HTTP errors
                if response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_message = error_data.get("fault", {}).get("message", "Unknown error")
                    except ValueError:
                        error_message = response.text or "Unknown error"
                        
                    raise TradierAPIError(
                        status_code=response.status_code,
                        message=error_message,
                        response_data=response.text
                    )
                    
                # Parse JSON response
                try:
                    return response.json()
                except ValueError:
                    raise TradierAPIError(
                        status_code=response.status_code,
                        message="Invalid JSON response",
                        response_data=response.text
                    )
                    
            except (RequestException, Timeout) as e:
                attempts += 1
                last_exception = e
                
                if attempts >= self.max_retries:
                    logger.error(f"Maximum retries reached for {url}: {str(e)}")
                    raise TradierAPIError(
                        status_code=500,
                        message=f"Request failed after {self.max_retries} attempts: {str(e)}"
                    )
                    
                logger.warning(f"Request to {url} failed (attempt {attempts}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)
                
        # This should never be reached, but just in case
        raise TradierAPIError(
            status_code=500,
            message=f"Unknown error after {self.max_retries} attempts: {str(last_exception)}"
        )

    # ----- Account Methods -----
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        return self._make_request("GET", "user/profile")
    
    def get_accounts(self) -> Dict[str, Any]:
        """Get a list of all accounts for the user."""
        return self._make_request("GET", "user/accounts")
    
    def get_account_balance(self, account_id: str = None) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Args:
            account_id: The account ID (uses default if not specified)
            
        Returns:
            Dictionary with account balance details
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        return self._make_request("GET", f"accounts/{account_id}/balances")
    
    def get_account_positions(self, account_id: str = None) -> Dict[str, Any]:
        """
        Get current positions for an account.
        
        Args:
            account_id: The account ID (uses default if not specified)
            
        Returns:
            Dictionary with position information
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        return self._make_request("GET", f"accounts/{account_id}/positions")
    
    def get_account_history(self, account_id: str = None) -> Dict[str, Any]:
        """
        Get account activity history.
        
        Args:
            account_id: The account ID (uses default if not specified)
            
        Returns:
            Dictionary with account activity history
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        return self._make_request("GET", f"accounts/{account_id}/history")
    
    def get_account_orders(
        self, 
        account_id: str = None,
        include_tags: bool = True
    ) -> Dict[str, Any]:
        """
        Get orders for an account.
        
        Args:
            account_id: The account ID (uses default if not specified)
            include_tags: Whether to include order tags in the response
            
        Returns:
            Dictionary containing order information
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        params = {"includeTags": "true" if include_tags else "false"}
        return self._make_request("GET", f"accounts/{account_id}/orders", params=params)
    
    def get_order(
        self, 
        order_id: str,
        account_id: str = None,
        include_tags: bool = True
    ) -> Dict[str, Any]:
        """
        Get a specific order by ID.
        
        Args:
            order_id: The order ID to retrieve
            account_id: The account ID (uses default if not specified)
            include_tags: Whether to include order tags in the response
            
        Returns:
            Dictionary containing order details
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        params = {"includeTags": "true" if include_tags else "false"}
        return self._make_request("GET", f"accounts/{account_id}/orders/{order_id}", params=params)
    
    # ----- Trading Methods -----
    
    def place_equity_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        duration: str = "day",
        price: float = None,
        stop: float = None,
        trail: float = None,
        tag: str = None,
        account_id: str = None
    ) -> Dict[str, Any]:
        """
        Place an equity order.
        
        Args:
            symbol: The stock symbol
            side: Order side ('buy' or 'sell')
            quantity: Number of shares
            order_type: Type of order ('market', 'limit', 'stop', 'stop_limit', 'trailing_stop')
            duration: Time the order is in force ('day', 'gtc', 'pre', 'post')
            price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop: Stop price (required for 'stop' and 'stop_limit' orders)
            trail: Trail amount (required for 'trailing_stop' orders)
            tag: Custom tag for the order
            account_id: The account ID (uses default if not specified)
            
        Returns:
            Dictionary with order confirmation details
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        # Validate parameters
        side = side.lower()
        if side not in ["buy", "sell", "buy_to_cover", "sell_short"]:
            raise ValueError("Side must be one of: buy, sell, buy_to_cover, sell_short")
            
        order_type = order_type.lower()
        if order_type not in ["market", "limit", "stop", "stop_limit", "trailing_stop"]:
            raise ValueError("Order type must be one of: market, limit, stop, stop_limit, trailing_stop")
            
        if (order_type in ["limit", "stop_limit"]) and price is None:
            raise ValueError("Price is required for limit and stop_limit orders")
            
        if (order_type in ["stop", "stop_limit"]) and stop is None:
            raise ValueError("Stop price is required for stop and stop_limit orders")
            
        if order_type == "trailing_stop" and trail is None:
            raise ValueError("Trail amount is required for trailing_stop orders")
            
        time_in_force = duration.lower()
        if time_in_force not in ["day", "gtc", "pre", "post"]:
            raise ValueError("Time in force must be one of: day, gtc, pre, post")
            
        # Construct the order data
        data = {
            "class": "equity",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": time_in_force
        }
        
        if price is not None:
            data["price"] = price
            
        if stop is not None:
            data["stop"] = stop
            
        if trail is not None:
            data["trail"] = trail
            
        if tag:
            data["tag"] = tag
            
        return self._make_request("POST", f"accounts/{account_id}/orders", data=data)
    
    def place_option_order(
        self,
        symbol: str,
        option_symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        duration: str = "day",
        price: float = None,
        stop: float = None,
        tag: str = None,
        account_id: str = None
    ) -> Dict[str, Any]:
        """
        Place an option order.
        
        Args:
            symbol: The underlying symbol
            option_symbol: The option symbol
            side: Order side ('buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close')
            quantity: Number of contracts
            order_type: Type of order ('market', 'limit', 'stop', 'stop_limit')
            duration: Time the order is in force ('day', 'gtc')
            price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop: Stop price (required for 'stop' and 'stop_limit' orders)
            tag: Custom tag for the order
            account_id: The account ID (uses default if not specified)
            
        Returns:
            Dictionary with order confirmation details
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        # Validate parameters
        side = side.lower()
        if side not in ["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]:
            raise ValueError("Side must be one of: buy_to_open, buy_to_close, sell_to_open, sell_to_close")
            
        order_type = order_type.lower()
        if order_type not in ["market", "limit", "stop", "stop_limit"]:
            raise ValueError("Order type must be one of: market, limit, stop, stop_limit")
            
        if (order_type in ["limit", "stop_limit"]) and price is None:
            raise ValueError("Price is required for limit and stop_limit orders")
            
        if (order_type in ["stop", "stop_limit"]) and stop is None:
            raise ValueError("Stop price is required for stop and stop_limit orders")
            
        time_in_force = duration.lower()
        if time_in_force not in ["day", "gtc"]:
            raise ValueError("Time in force must be one of: day, gtc")
            
        # Construct the order data
        data = {
            "class": "option",
            "symbol": symbol,
            "option_symbol": option_symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": time_in_force
        }
        
        if price is not None:
            data["price"] = price
            
        if stop is not None:
            data["stop"] = stop
            
        if tag:
            data["tag"] = tag
            
        return self._make_request("POST", f"accounts/{account_id}/orders", data=data)
    
    def cancel_order(self, order_id: str, account_id: str = None) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: The order ID to cancel
            account_id: The account ID (uses default if not specified)
            
        Returns:
            Dictionary with cancellation confirmation
        """
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError("Account ID is required")
            
        return self._make_request("DELETE", f"accounts/{account_id}/orders/{order_id}")
    
    # ----- Market Data Methods -----
    
    def get_quotes(self, symbols: Union[str, List[str]], greeks: bool = False) -> Dict[str, Any]:
        """
        Get quotes for one or more symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            greeks: Whether to include greek values (for options)
            
        Returns:
            Dictionary with quote information
        """
        if isinstance(symbols, list):
            symbols = ",".join(symbols)
            
        params = {"symbols": symbols}
        
        if greeks:
            params["greeks"] = "true"
            
        return self._make_request("GET", "markets/quotes", params=params)
    
    def get_option_chain(
        self, 
        symbol: str, 
        expiration: str, 
        greeks: bool = False
    ) -> Dict[str, Any]:
        """
        Get option chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date (YYYY-MM-DD format)
            greeks: Whether to include greek values
            
        Returns:
            Dictionary with option chain information
        """
        params = {"symbol": symbol, "greeks": "true" if greeks else "false"}
        
        if expiration:
            params["expiration"] = expiration
            
        return self._make_request("GET", "markets/options/chains", params=params)
    
    def get_option_expirations(
        self, 
        symbol: str, 
        include_all_roots: bool = False,
        strikes: bool = False
    ) -> Dict[str, Any]:
        """
        Get option expiration dates for a symbol.
        
        Args:
            symbol: Underlying symbol
            include_all_roots: Include all option roots
            strikes: Include strikes in response
            
        Returns:
            Dictionary with expiration dates
        """
        params = {
            "symbol": symbol, 
            "includeAllRoots": "true" if include_all_roots else "false",
            "strikes": "true" if strikes else "false"
        }
            
        return self._make_request("GET", "markets/options/expirations", params=params)
    
    def get_option_strikes(
        self, 
        symbol: str, 
        expiration: str
    ) -> Dict[str, Any]:
        """
        Get option strike prices for a symbol and expiration.
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date (YYYY-MM-DD format)
            
        Returns:
            Dictionary with strike prices
        """
        params = {"symbol": symbol, "expiration": expiration}
        return self._make_request("GET", "markets/options/strikes", params=params)
    
    def get_historical_quotes(
        self,
        symbol: str,
        interval: str,
        start: str = None,
        end: str = None,
        session_filter: str = None
    ) -> Dict[str, Any]:
        """
        Get historical pricing for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Data interval ('daily', 'weekly', 'monthly', or 'minute')
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            session_filter: Which session(s) to include ('open', 'all')
            
        Returns:
            Dictionary with historical pricing data
        """
        params = {"symbol": symbol, "interval": interval}
        
        if start:
            params["start"] = start
            
        if end:
            params["end"] = end
            
        if session_filter:
            params["session_filter"] = session_filter
            
        return self._make_request("GET", "markets/history", params=params)
    
    def get_time_and_sales(
        self,
        symbol: str,
        interval: str = None,
        start: str = None,
        end: str = None,
        session_filter: str = None
    ) -> Dict[str, Any]:
        """
        Get time and sales data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Data interval (default: '1min')
            start: Start date/time
            end: End date/time
            session_filter: Which session(s) to include ('open', 'all')
            
        Returns:
            Dictionary with time and sales data
        """
        params = {"symbol": symbol}
        
        if interval:
            params["interval"] = interval
            
        if start:
            params["start"] = start
            
        if end:
            params["end"] = end
            
        if session_filter:
            params["session_filter"] = session_filter
            
        return self._make_request("GET", "markets/timesales", params=params)
    
    def get_clock(self) -> Dict[str, Any]:
        """
        Get current market clock information.
        
        Returns:
            Dictionary with market status information
        """
        return self._make_request("GET", "markets/clock")
    
    def get_calendar(self, month: str = None, year: int = None) -> Dict[str, Any]:
        """
        Get market calendar information.
        
        Args:
            month: Calendar month (1-12)
            year: Calendar year
            
        Returns:
            Dictionary with market calendar information
        """
        params = {}
        
        if month:
            params["month"] = month
            
        if year:
            params["year"] = year
            
        return self._make_request("GET", "markets/calendar", params=params)
    
    def search_companies(self, query: str) -> Dict[str, Any]:
        """
        Search for companies by name or symbol.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        params = {"q": query}
        return self._make_request("GET", "markets/search", params=params)
    
    def get_lookup(self, query: str, search_type: str = None) -> Dict[str, Any]:
        """
        Lookup symbols for companies, indexes, or funds.
        
        Args:
            query: Search query
            search_type: Type of search ('stock', 'option', 'index', 'mutual_fund', 'etf')
            
        Returns:
            Dictionary with lookup results
        """
        params = {"q": query}
        
        if search_type:
            params["type"] = search_type
            
        return self._make_request("GET", "markets/lookup", params=params)

    def get_quote_dataframe(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get quotes and return as a pandas DataFrame.
        
        Args:
            symbols: Single symbol or list of symbols
        
        Returns:
            DataFrame containing quote data
        """
        response = self.get_quotes(symbols)
        
        # Extract quote data
        if 'quotes' in response and 'quote' in response['quotes']:
            quotes = response['quotes']['quote']
            
            # If only one quote, wrap it in a list
            if not isinstance(quotes, list):
                quotes = [quotes]
                
            return pd.DataFrame(quotes)
        
        return pd.DataFrame()

    def get_historical_dataframe(self, symbol: str, interval: str = "daily", 
                                start_date: Union[str, date] = None,
                                end_date: Union[str, date] = None) -> pd.DataFrame:
        """
        Get historical quotes and return as a pandas DataFrame.
        
        Args:
            symbol: The symbol to get data for
            interval: Data interval ('daily', 'weekly', 'monthly', or 'minute')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame containing historical data
        """
        response = self.get_historical_quotes(symbol, interval, start_date, end_date)
        
        # Extract historical data
        if 'history' in response and 'day' in response['history']:
            days = response['history']['day']
            df = pd.DataFrame(days)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            return df
        
        return pd.DataFrame()

    def format_positions_dataframe(self) -> pd.DataFrame:
        """
        Get positions and format as a pandas DataFrame.
        
        Returns:
            DataFrame containing position information
        """
        response = self.get_account_positions()
        
        if 'positions' in response and 'position' in response['positions']:
            positions = response['positions']['position']
            
            # If only one position, wrap it in a list
            if not isinstance(positions, list):
                positions = [positions]
                
            df = pd.DataFrame(positions)
            
            # Convert numeric columns
            numeric_cols = ['quantity', 'cost_basis', 'date_acquired']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            return df
        
        return pd.DataFrame()

    def calculate_position_value(self) -> Dict[str, float]:
        """
        Calculate the current value of each position.
        
        Returns:
            Dictionary mapping symbols to current position values
        """
        positions_df = self.format_positions_dataframe()
        
        if positions_df.empty:
            return {}
        
        # Get symbols from positions
        symbols = positions_df['symbol'].tolist()
        
        # Get current quotes
        quotes_df = self.get_quote_dataframe(symbols)
        
        if quotes_df.empty:
            return {}
        
        # Create a lookup dictionary for last prices
        price_lookup = dict(zip(quotes_df['symbol'], quotes_df['last']))
        
        # Calculate position values
        position_values = {}
        for _, position in positions_df.iterrows():
            symbol = position['symbol']
            quantity = position['quantity']
            
            if symbol in price_lookup:
                current_price = price_lookup[symbol]
                position_values[symbol] = quantity * current_price
        
        return position_values

    def calculate_portfolio_stats(self) -> Dict[str, Any]:
        """
        Calculate various portfolio statistics.
        
        Returns:
            Dictionary containing portfolio statistics
        """
        # Get account balances
        balances = self.get_account_balance()
        
        # Get positions
        positions_df = self.format_positions_dataframe()
        
        # Calculate statistics
        stats = {
            "account_value": 0,
            "cash_balance": 0,
            "buying_power": 0,
            "position_count": 0,
            "position_allocation": {},
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "total_pnl": 0
        }
        
        # Extract account value and cash balance
        if 'balances' in balances:
            account_balances = balances['balances']
            stats["account_value"] = account_balances.get('total_equity', 0)
            stats["cash_balance"] = account_balances.get('cash', 0)
            stats["buying_power"] = account_balances.get('buying_power', 0)
            stats["realized_pnl"] = account_balances.get('day_trade_realized_pnl', 0)
        
        # Position statistics
        if not positions_df.empty:
            stats["position_count"] = len(positions_df)
            
            # Get current position values
            position_values = self.calculate_position_value()
            
            # Calculate unrealized P&L and allocations
            total_position_value = sum(position_values.values())
            
            for _, position in positions_df.iterrows():
                symbol = position['symbol']
                cost_basis = position['cost_basis']
                quantity = position['quantity']
                
                if symbol in position_values:
                    current_value = position_values[symbol]
                    unrealized_pnl = current_value - (cost_basis * quantity)
                    
                    # Add to total unrealized P&L
                    stats["unrealized_pnl"] += unrealized_pnl
                    
                    # Calculate allocation percentage
                    if total_position_value > 0:
                        allocation = (current_value / total_position_value) * 100
                    else:
                        allocation = 0
                    
                    stats["position_allocation"][symbol] = {
                        "value": current_value,
                        "allocation_percent": allocation,
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_percent": (unrealized_pnl / (cost_basis * quantity)) * 100 if cost_basis > 0 else 0
                    }
        
        # Calculate total P&L
        stats["total_pnl"] = stats["realized_pnl"] + stats["unrealized_pnl"]
        
        return stats


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get API key from environment variable
    api_key = os.environ.get("TRADIER_API_KEY")
    account_id = os.environ.get("TRADIER_ACCOUNT_ID")
    
    if not api_key:
        print("TRADIER_API_KEY environment variable not set")
        exit(1)
    
    # Create client
    client = TradierClient(api_key=api_key, account_id=account_id, use_sandbox=True)
    
    # Get market status
    market_status = client.get_clock()
    print(f"Market status: {market_status['clock']['state']}")
    
    # Get quotes for symbols
    quotes = client.get_quotes(["SPY", "AAPL", "MSFT"], greeks=True)
    for quote in quotes.get("quotes", {}).get("quote", []):
        print(f"{quote['symbol']}: ${quote['last']} (change: {quote['change']})")
    
    # Only proceed with account operations if account_id is set
    if account_id:
        # Get account balances
        balances = client.get_account_balance()
        print(f"Account balance: ${balances['balances']['total_equity']}")
        
        # Get positions
        positions = client.get_account_positions()
        for position in positions.get("positions", {}).get("position", []):
            print(f"Position: {position['symbol']} - {position['quantity']} shares at ${position['cost_basis']}")
        
        # Get historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        aapl_history = client.get_historical_dataframe("AAPL", start_date=start_date, end_date=end_date)
        print("\nAPPL Historical Data (last 5 days):")
        print(aapl_history.tail(5)[["date", "open", "high", "low", "close", "volume"]])
        
        # Calculate portfolio statistics
        portfolio_stats = client.calculate_portfolio_stats()
        print("\nPortfolio Statistics:")
        print(f"Account Value: ${portfolio_stats['account_value']:.2f}")
        print(f"Cash Balance: ${portfolio_stats['cash_balance']:.2f}")
        print(f"Buying Power: ${portfolio_stats['buying_power']:.2f}")
        print(f"Position Count: {portfolio_stats['position_count']}")
        print(f"Realized P&L: ${portfolio_stats['realized_pnl']:.2f}")
        print(f"Unrealized P&L: ${portfolio_stats['unrealized_pnl']:.2f}")
        print(f"Total P&L: ${portfolio_stats['total_pnl']:.2f}") 