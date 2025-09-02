#!/usr/bin/env python3
"""
Tradier API Client for Trading Bot

This module provides a client for interacting with the Tradier API.
"""

import os
import json
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class TradierAPI:
    """Client for Tradier API to interact with account and market data."""
    
    # API Endpoints
    SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"
    PRODUCTION_BASE_URL = "https://api.tradier.com/v1"
    
    # API Paths
    ACCOUNT_PATH = "/accounts/{account_id}"
    BALANCES_PATH = "/accounts/{account_id}/balances"
    POSITIONS_PATH = "/accounts/{account_id}/positions"
    ORDERS_PATH = "/accounts/{account_id}/orders"
    ORDER_PATH = "/accounts/{account_id}/orders/{order_id}"
    
    QUOTES_PATH = "/markets/quotes"
    OPTIONS_CHAIN_PATH = "/markets/options/chains"
    OPTIONS_EXPIRATIONS_PATH = "/markets/options/expirations"
    OPTIONS_STRIKES_PATH = "/markets/options/strikes"
    HISTORY_PATH = "/markets/history"
    CLOCK_PATH = "/markets/clock"
    CALENDAR_PATH = "/markets/calendar"
    
    # Order Types
    ORDER_TYPES = ["market", "limit", "stop", "stop_limit"]
    
    # Durations
    DURATIONS = ["day", "gtc", "pre", "post"]
    
    def __init__(self, 
                 api_key: str, 
                 account_id: str,
                 use_sandbox: bool = True,
                 timeout: int = 10,
                 max_retries: int = 3,
                 retry_delay: int = 1):
        """
        Initialize Tradier API client.
        
        Args:
            api_key: Tradier API access token
            account_id: Tradier account ID
            use_sandbox: Whether to use sandbox environment (default: True)
            timeout: Request timeout in seconds (default: 10)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Delay between retries in seconds (default: 1)
        """
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = self.SANDBOX_BASE_URL if use_sandbox else self.PRODUCTION_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        })
        
        # Cache for market data
        self.market_data_cache = {}
        self.cache_expiry = {}
        
        logger.info(f"Initialized Tradier API client for account {account_id} with {'sandbox' if use_sandbox else 'production'} environment")
    
    def _make_request(self, method: str, path: str, params: Optional[Dict] = None, 
                      data: Optional[Dict] = None, headers: Optional[Dict] = None,
                      force_json: bool = True) -> Dict:
        """
        Make request to Tradier API with error handling and retries.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API endpoint path
            params: Query parameters (default: None)
            data: Request body data (default: None)
            headers: Additional headers (default: None)
            force_json: Force JSON response (default: True)
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}{path}"
        
        # Merge headers
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        # Add Content-Type header for POST requests
        if method.upper() == "POST" and data:
            request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=request_headers,
                    timeout=self.timeout
                )
                elapsed = time.time() - start_time
                
                # Log request details
                logger.debug(f"{method} {url} completed in {elapsed:.3f}s with status {response.status_code}")
                
                # Handle non-200 responses
                if response.status_code != 200:
                    error_msg = f"Tradier API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "fault" in error_data:
                            error_msg = f"Tradier API error: {error_data['fault']['detail']}"
                    except Exception:
                        error_msg = f"Tradier API error: {response.text}"
                    
                    logger.error(error_msg)
                    
                    # Check if we should retry
                    if attempt < self.max_retries and response.status_code in [429, 500, 502, 503, 504]:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Retrying request in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    
                    # Return error response if we're not retrying
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "error": error_msg
                    }
                
                # Parse response
                if force_json:
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        **response.json()
                    }
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response": response
                }
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout: {method} {url}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retrying request in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return {
                        "success": False, 
                        "error": "Request timeout",
                        "status_code": 0
                    }
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retrying request in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return {
                        "success": False, 
                        "error": str(e),
                        "status_code": 0
                    }
        
        # This should not be reached, but just in case
        return {
            "success": False,
            "error": "Max retries exceeded",
            "status_code": 0
        }
    
    def get_account_balances(self) -> Dict:
        """
        Get account balances.
        
        Returns:
            Account balances information
        """
        path = self.BALANCES_PATH.format(account_id=self.account_id)
        return self._make_request("GET", path)
    
    def get_account_positions(self) -> Dict:
        """
        Get account positions.
        
        Returns:
            Account positions information
        """
        path = self.POSITIONS_PATH.format(account_id=self.account_id)
        return self._make_request("GET", path)
    
    def get_account_orders(self, status: str = None) -> Dict:
        """
        Get account orders.
        
        Args:
            status: Filter orders by status (open, filled, canceled, expired, rejected, pending)
        
        Returns:
            Account orders information
        """
        path = self.ORDERS_PATH.format(account_id=self.account_id)
        params = {}
        if status:
            params["status"] = status
        return self._make_request("GET", path, params=params)
    
    def get_order(self, order_id: str) -> Dict:
        """
        Get specific order.
        
        Args:
            order_id: Order ID
        
        Returns:
            Order information
        """
        path = self.ORDER_PATH.format(account_id=self.account_id, order_id=order_id)
        return self._make_request("GET", path)
    
    def get_quotes(self, symbols: Union[str, List[str]], greeks: bool = False) -> Dict:
        """
        Get market quotes for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            greeks: Include Greek data for options (default: False)
        
        Returns:
            Market quotes information
        """
        params = {
            "symbols": ",".join(symbols) if isinstance(symbols, list) else symbols,
            "greeks": "true" if greeks else "false"
        }
        
        # Check cache first
        cache_key = f"quotes_{params['symbols']}_{params['greeks']}"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        response = self._make_request("GET", self.QUOTES_PATH, params=params)
        
        # Cache response for 1 minute
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 60  # 1 minute expiry
        
        return response
    
    def get_options_chain(self, symbol: str, expiration: str, greeks: bool = False) -> Dict:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration date (YYYY-MM-DD)
            greeks: Include Greek data (default: False)
        
        Returns:
            Options chain information
        """
        params = {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true" if greeks else "false"
        }
        
        # Check cache first
        cache_key = f"options_chain_{symbol}_{expiration}_{greeks}"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        response = self._make_request("GET", self.OPTIONS_CHAIN_PATH, params=params)
        
        # Cache response for 1 minute
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 60  # 1 minute expiry
        
        return response
    
    def get_options_expirations(self, symbol: str, includeAllRoots: bool = False, strikes: bool = False) -> Dict:
        """
        Get options expirations for a symbol.
        
        Args:
            symbol: Underlying symbol
            includeAllRoots: Include all options roots (default: False)
            strikes: Include strikes in response (default: False)
        
        Returns:
            Options expirations information
        """
        params = {
            "symbol": symbol,
            "includeAllRoots": "true" if includeAllRoots else "false",
            "strikes": "true" if strikes else "false"
        }
        
        # Check cache first
        cache_key = f"options_expirations_{symbol}_{includeAllRoots}_{strikes}"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        response = self._make_request("GET", self.OPTIONS_EXPIRATIONS_PATH, params=params)
        
        # Cache response for 5 minutes
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 300  # 5 minutes expiry
        
        return response
    
    def get_historical_quotes(self, symbol: str, interval: str = "daily", 
                              start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        Get historical quotes for a symbol.
        
        Args:
            symbol: Symbol
            interval: Quote interval (daily, weekly, monthly; default: daily)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Historical quotes information
        """
        params = {
            "symbol": symbol,
            "interval": interval
        }
        
        if start_date:
            params["start"] = start_date
        
        if end_date:
            params["end"] = end_date
        
        # Check cache first
        cache_key = f"history_{symbol}_{interval}_{start_date}_{end_date}"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        response = self._make_request("GET", self.HISTORY_PATH, params=params)
        
        # Cache response for 1 hour
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 3600  # 1 hour expiry
        
        return response
    
    def get_market_clock(self) -> Dict:
        """
        Get market clock information.
        
        Returns:
            Market clock information
        """
        # Check cache first
        cache_key = "market_clock"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        response = self._make_request("GET", self.CLOCK_PATH)
        
        # Cache response for 1 minute
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 60  # 1 minute expiry
        
        return response
    
    def get_market_calendar(self, month: Optional[int] = None, year: Optional[int] = None) -> Dict:
        """
        Get market calendar information.
        
        Args:
            month: Calendar month
            year: Calendar year
        
        Returns:
            Market calendar information
        """
        params = {}
        
        if month is not None:
            params["month"] = month
        
        if year is not None:
            params["year"] = year
        
        # Check cache first
        cache_key = f"market_calendar_{month}_{year}"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        response = self._make_request("GET", self.CALENDAR_PATH, params=params)
        
        # Cache response for 1 day
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 86400  # 1 day expiry
        
        return response
    
    def place_equity_order(self, 
                          symbol: str,
                          side: str,
                          quantity: int,
                          order_type: str = "market",
                          price: Optional[float] = None,
                          stop: Optional[float] = None,
                          duration: str = "day",
                          tag: Optional[str] = None) -> Dict:
        """
        Place an equity order.
        
        Args:
            symbol: Symbol
            side: Order side (buy, buy_to_cover, sell, sell_short)
            quantity: Order quantity
            order_type: Order type (market, limit, stop, stop_limit; default: market)
            price: Limit price (required for limit and stop_limit orders)
            stop: Stop price (required for stop and stop_limit orders)
            duration: Order duration (day, gtc, pre, post; default: day)
            tag: Order tag for reference
        
        Returns:
            Order placement result
        """
        if order_type not in self.ORDER_TYPES:
            raise ValueError(f"Invalid order type: {order_type}. Must be one of {self.ORDER_TYPES}")
        
        if duration not in self.DURATIONS:
            raise ValueError(f"Invalid duration: {duration}. Must be one of {self.DURATIONS}")
        
        if side not in ["buy", "buy_to_cover", "sell", "sell_short"]:
            raise ValueError(f"Invalid side: {side}. Must be one of ['buy', 'buy_to_cover', 'sell', 'sell_short']")
        
        if order_type in ["limit", "stop_limit"] and price is None:
            raise ValueError(f"Price is required for {order_type} orders")
        
        if order_type in ["stop", "stop_limit"] and stop is None:
            raise ValueError(f"Stop price is required for {order_type} orders")
        
        path = self.ORDERS_PATH.format(account_id=self.account_id)
        
        data = {
            "class": "equity",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": duration
        }
        
        if price is not None:
            data["price"] = price
        
        if stop is not None:
            data["stop"] = stop
        
        if tag is not None:
            data["tag"] = tag
        
        return self._make_request("POST", path, data=data)
    
    def place_option_order(self,
                          symbol: str,
                          side: str,
                          quantity: int,
                          order_type: str = "market",
                          price: Optional[float] = None,
                          stop: Optional[float] = None,
                          duration: str = "day",
                          tag: Optional[str] = None) -> Dict:
        """
        Place an option order.
        
        Args:
            symbol: Option symbol (e.g., AAPL220121C00150000)
            side: Order side (buy_to_open, buy_to_close, sell_to_open, sell_to_close)
            quantity: Order quantity
            order_type: Order type (market, limit, stop, stop_limit; default: market)
            price: Limit price (required for limit and stop_limit orders)
            stop: Stop price (required for stop and stop_limit orders)
            duration: Order duration (day, gtc; default: day)
            tag: Order tag for reference
        
        Returns:
            Order placement result
        """
        if order_type not in self.ORDER_TYPES:
            raise ValueError(f"Invalid order type: {order_type}. Must be one of {self.ORDER_TYPES}")
        
        if duration not in self.DURATIONS:
            raise ValueError(f"Invalid duration: {duration}. Must be one of {self.DURATIONS}")
        
        if side not in ["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]:
            raise ValueError(f"Invalid side: {side}. Must be one of ['buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close']")
        
        if order_type in ["limit", "stop_limit"] and price is None:
            raise ValueError(f"Price is required for {order_type} orders")
        
        if order_type in ["stop", "stop_limit"] and stop is None:
            raise ValueError(f"Stop price is required for {order_type} orders")
        
        path = self.ORDERS_PATH.format(account_id=self.account_id)
        
        data = {
            "class": "option",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": duration
        }
        
        if price is not None:
            data["price"] = price
        
        if stop is not None:
            data["stop"] = stop
        
        if tag is not None:
            data["tag"] = tag
        
        return self._make_request("POST", path, data=data)
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
        
        Returns:
            Order cancellation result
        """
        path = self.ORDER_PATH.format(account_id=self.account_id, order_id=order_id)
        return self._make_request("DELETE", path)
    
    def clear_cache(self) -> None:
        """Clear the market data cache."""
        self.market_data_cache.clear()
        self.cache_expiry.clear()
        logger.debug("Market data cache cleared")
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        clock_data = self.get_market_clock()
        
        if not clock_data.get("success"):
            logger.error(f"Failed to get market clock: {clock_data.get('error')}")
            return False
        
        # Check if market is open
        if "clock" in clock_data:
            return clock_data["clock"]["state"] == "open"
        
        return False
    
    def get_time_to_market_open(self) -> Optional[timedelta]:
        """
        Get time until market opens.
        
        Returns:
            Timedelta until market open, or None if error
        """
        clock_data = self.get_market_clock()
        
        if not clock_data.get("success"):
            logger.error(f"Failed to get market clock: {clock_data.get('error')}")
            return None
        
        # Parse market data
        if "clock" in clock_data:
            if clock_data["clock"]["state"] == "open":
                return timedelta(0)
            
            # Get next_open timestamp
            try:
                next_open_str = clock_data["clock"]["next_open"]
                next_open = datetime.fromisoformat(next_open_str.replace("Z", "+00:00"))
                now = datetime.now().astimezone()
                return next_open - now
            except Exception as e:
                logger.error(f"Error parsing market open time: {e}")
                return None
        
        return None
    
    def get_time_to_market_close(self) -> Optional[timedelta]:
        """
        Get time until market closes.
        
        Returns:
            Timedelta until market close, or None if error
        """
        clock_data = self.get_market_clock()
        
        if not clock_data.get("success"):
            logger.error(f"Failed to get market clock: {clock_data.get('error')}")
            return None
        
        # Parse market data
        if "clock" in clock_data:
            if clock_data["clock"]["state"] != "open":
                return timedelta(0)
            
            # Get next_close timestamp
            try:
                next_close_str = clock_data["clock"]["next_close"]
                next_close = datetime.fromisoformat(next_close_str.replace("Z", "+00:00"))
                now = datetime.now().astimezone()
                return next_close - now
            except Exception as e:
                logger.error(f"Error parsing market close time: {e}")
                return None
        
        return None
    
    def get_intraday_quotes(self, symbol: str, interval: str = "5min", start: Optional[str] = None, end: Optional[str] = None) -> Dict:
        """
        Get intraday quotes for a symbol.
        
        Args:
            symbol: Symbol
            interval: Quote interval (1min, 5min, 15min; default: 5min)
            start: Start time (YYYY-MM-DD HH:MM)
            end: End time (YYYY-MM-DD HH:MM)
        
        Returns:
            Intraday quotes information
        """
        if interval not in ["1min", "5min", "15min"]:
            raise ValueError(f"Invalid interval: {interval}. Must be one of ['1min', '5min', '15min']")
        
        params = {
            "symbol": symbol,
            "interval": interval
        }
        
        if start:
            params["start"] = start
        
        if end:
            params["end"] = end
        
        # Check cache first
        cache_key = f"timesales_{symbol}_{interval}_{start}_{end}"
        if cache_key in self.market_data_cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logger.debug(f"Using cached data for {cache_key}")
            return self.market_data_cache[cache_key]
        
        # Make request
        path = "/markets/timesales"
        response = self._make_request("GET", path, params=params)
        
        # Cache response for 5 minutes
        if response.get("success"):
            self.market_data_cache[cache_key] = response
            self.cache_expiry[cache_key] = time.time() + 300  # 5 minutes expiry
        
        return response 