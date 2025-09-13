import os
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Import tenacity for retry mechanisms
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tenacity not available. Retry functionality will be disabled. Install with: pip install tenacity")

# Import cachetools for better cache management
try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("cachetools not available. Using simple dict cache instead. Install with: pip install cachetools")

logger = logging.getLogger(__name__)

class TradierClient:
    """
    Client for interacting with the Tradier Brokerage API
    
    Handles authentication, account data, market data and order execution
    with robust error handling, retries, timeouts, and efficient caching
    """
    
    # API endpoints for sandbox and production
    SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"
    PRODUCTION_BASE_URL = "https://api.tradier.com/v1"
    
    # Default request timeout (seconds)
    DEFAULT_TIMEOUT = 15
    
    # Rate limiting parameters
    MAX_REQUESTS_PER_MINUTE = 60
    RATE_LIMIT_COOLDOWN = 60  # seconds to wait when rate limited
    
    def __init__(self, api_key: str, account_id: str, sandbox: bool = True):
        """
        Initialize the Tradier client
        
        Args:
            api_key: Tradier API access token
            account_id: Tradier account ID
            sandbox: Whether to use the sandbox environment (default: True)
        """
        self.api_key = api_key
        self.account_id = account_id
        self.sandbox = sandbox
        self.base_url = self.SANDBOX_BASE_URL if sandbox else self.PRODUCTION_BASE_URL
        
        # Standard headers for all requests
        # Note: Tradier order endpoints expect form-encoded payloads
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Cache for market data to reduce API calls
        if CACHETOOLS_AVAILABLE:
            # Use TTLCache for automatic expiration
            self.quote_cache = TTLCache(maxsize=1000, ttl=10)  # Cache up to 1000 symbols, 10s TTL
            logger.info("Using TTLCache for quote caching")
        else:
            # Fallback to manual cache management
            self.quote_cache = {}
            self.quote_cache_expiry = {}
            self.quote_cache_duration = 10  # seconds
            logger.info("Using simple dict cache for quotes")
        
        # Track request timestamps for rate limiting
        self.request_timestamps = []
        
        logger.info(f"Tradier client initialized for account {account_id} in {'sandbox' if sandbox else 'production'} mode")
    
    def _check_rate_limits(self):
        """
        Check and enforce rate limits.
        Removes timestamps older than 1 minute and sleeps if too many recent requests.
        """
        now = time.time()
        # Keep only timestamps from the last minute
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
        
        # If approaching limit, sleep to avoid hitting the rate limit
        if len(self.request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE - 5:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Approaching rate limit, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
    
    def _make_request_with_retry(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make a request to the Tradier API with retry logic
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            params: Query parameters (for GET requests)
            data: Form data (for POST requests)
            
        Returns:
            API response as a dictionary
        """
        # Implement basic rate limiting
        self._check_rate_limits()
        
        # Track this request
        self.request_timestamps.append(time.time())
        
        url = f"{self.base_url}{endpoint}"
        
        # Define retry parameters
        max_attempts = 3
        attempts = 0
        last_error = None
        
        while attempts < max_attempts:
            attempts += 1
            try:
                if method.upper() == "GET":
                    response = requests.get(
                        url, 
                        headers=self.headers, 
                        params=params, 
                        timeout=self.DEFAULT_TIMEOUT
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url, 
                        headers=self.headers, 
                        data=data, 
                        timeout=self.DEFAULT_TIMEOUT
                    )
                elif method.upper() == "PUT":
                    response = requests.put(
                        url, 
                        headers=self.headers, 
                        data=data, 
                        timeout=self.DEFAULT_TIMEOUT
                    )
                elif method.upper() == "DELETE":
                    response = requests.delete(
                        url, 
                        headers=self.headers, 
                        timeout=self.DEFAULT_TIMEOUT
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                # Handle rate limiting with special handling for 429 responses
                if response.status_code == 429:
                    logger.warning(f"Rate limited by Tradier API (attempt {attempts}/{max_attempts}). ")
                    if attempts < max_attempts:
                        wait_time = self.RATE_LIMIT_COOLDOWN * attempts  # Exponential backoff
                        logger.warning(f"Cooling down for {wait_time} seconds before retry.")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise TradierRateLimitError("Rate limited by Tradier API, max retries exceeded")
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                return response.json()
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # These errors are retryable
                last_error = e
                if attempts < max_attempts:
                    wait_time = 2 ** attempts  # Exponential backoff: 2, 4, 8...
                    logger.warning(f"Retryable error on attempt {attempts}/{max_attempts}: {str(e)}. ")
                    logger.warning(f"Retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Failed after {max_attempts} attempts: {str(e)}"
                    logger.error(error_msg)
                    raise TradierAPIError(error_msg) from e
            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP error making request to Tradier API: {str(e)}"
                logger.error(error_msg)
                
                # Try to parse error response if available
                try:
                    error_details = e.response.json()
                    logger.error(f"Tradier API error details: {json.dumps(error_details)}")
                except:
                    pass
                    
                # Only retry 5xx errors, not 4xx
                if e.response.status_code >= 500 and attempts < max_attempts:
                    wait_time = 2 ** attempts
                    logger.warning(f"Server error, retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    raise TradierAPIError(error_msg) from e
            except Exception as e:
                error_msg = f"Error making request to Tradier API: {str(e)}"
                logger.error(error_msg)
                raise TradierAPIError(error_msg) from e
        
        # If we reach here, we've exhausted all retries
        if last_error:
            error_msg = f"All retry attempts failed: {str(last_error)}"
            logger.error(error_msg)
            raise TradierAPIError(error_msg) from last_error
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make a request to the Tradier API with retries, timeouts and rate limiting
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            params: Query parameters (for GET requests)
            data: Form data (for POST requests)
            
        Returns:
            API response as a dictionary
        """
        # If tenacity is available, use it for retry logic
        if TENACITY_AVAILABLE:
            return self._make_request_with_tenacity(method, endpoint, params, data)
        else:
            # Fall back to simple retry implementation
            return self._make_request_with_retry(method, endpoint, params, data)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True
    )
    def _make_request_with_tenacity(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make a request to the Tradier API with tenacity retry library
        Only used if tenacity is available
        """
        # Implement basic rate limiting
        self._check_rate_limits()
        
        # Track this request
        self.request_timestamps.append(time.time())
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=self.DEFAULT_TIMEOUT
                )
            elif method.upper() == "POST":
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    data=data, 
                    timeout=self.DEFAULT_TIMEOUT
                )
            elif method.upper() == "PUT":
                response = requests.put(
                    url, 
                    headers=self.headers, 
                    data=data, 
                    timeout=self.DEFAULT_TIMEOUT
                )
            elif method.upper() == "DELETE":
                response = requests.delete(
                    url, 
                    headers=self.headers, 
                    timeout=self.DEFAULT_TIMEOUT
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Handle rate limiting with special handling for 429 responses
            if response.status_code == 429:
                logger.warning(f"Rate limited by Tradier API. Cooling down for {self.RATE_LIMIT_COOLDOWN} seconds.")
                time.sleep(self.RATE_LIMIT_COOLDOWN)
                # This will be retried by the @retry decorator
                raise TradierRateLimitError("Rate limited by Tradier API")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to Tradier API: {str(e)}"
            logger.error(error_msg)
            
            # Try to parse error response if available
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_details = e.response.json()
                    logger.error(f"Tradier API error details: {json.dumps(error_details)}")
            except:
                pass
                
            # Re-raise as custom exception
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                raise TradierRateLimitError(error_msg) from e
            else:
                raise TradierAPIError(error_msg) from e
    
    # --- Account Methods ---
    
    def get_account_balances(self) -> Dict:
        """
        Get account balances
        
        Returns:
            Dictionary with account balance information
        """
        endpoint = f"/accounts/{self.account_id}/balances"
        response = self._make_request("GET", endpoint)
        return response.get("balances", {})
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions
        
        Returns:
            List of positions
        """
        endpoint = f"/accounts/{self.account_id}/positions"
        response = self._make_request("GET", endpoint)
        positions_data = response.get("positions", {})
        
        # Handle the case when there are no positions
        if positions_data == "null" or not positions_data:
            return []
            
        positions = positions_data.get("position", [])
        # If there's only one position, the API returns it as a dict, not a list
        if isinstance(positions, dict):
            positions = [positions]
            
        return positions
    
    def get_orders(self, status: str = None) -> List[Dict]:
        """
        Get orders for the account
        
        Args:
            status: Filter by order status (open, filled, canceled, expired, rejected, pending)
            
        Returns:
            List of orders
        """
        endpoint = f"/accounts/{self.account_id}/orders"
        response = self._make_request("GET", endpoint)
        orders_data = response.get("orders", {})
        
        # Handle the case when there are no orders
        if orders_data == "null" or not orders_data:
            return []
            
        orders = orders_data.get("order", [])
        # If there's only one order, the API returns it as a dict, not a list
        if isinstance(orders, dict):
            orders = [orders]
            
        # Filter by status if specified
        if status:
            orders = [order for order in orders if order.get("status") == status]
            
        return orders
    
    def get_order(self, order_id: str) -> Dict:
        """
        Get a specific order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details
        """
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"
        response = self._make_request("GET", endpoint)
        return response.get("order", {})
    
    def get_account_history(self, limit: int = 25) -> List[Dict]:
        """
        Get account history
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of account activities
        """
        endpoint = f"/accounts/{self.account_id}/history"
        params = {"limit": limit}
        response = self._make_request("GET", endpoint, params=params)
        history_data = response.get("history", {})
        
        if history_data == "null" or not history_data:
            return []
            
        events = history_data.get("event", [])
        # If there's only one event, the API returns it as a dict, not a list
        if isinstance(events, dict):
            events = [events]
            
        return events
    
    # --- Market Data Methods ---
    
    def get_quotes(self, symbols: Union[str, List[str]], greeks: bool = False) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols with TTL caching
        
        Args:
            symbols: Single symbol or list of symbols
            greeks: Include Greeks for options (default: False)
            
        Returns:
            Dictionary of quotes keyed by symbol
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Return empty dict if no symbols
        if not symbols:
            return {}
        
        # Check which symbols are in cache and which need to be fetched
        now = datetime.now().timestamp()
        cached_quotes = {}
        symbols_to_fetch = []
        
        # Handle different cache implementations
        if CACHETOOLS_AVAILABLE:
            # Using TTLCache - expiration is automatic
            for symbol in symbols:
                if symbol in self.quote_cache:
                    cached_quotes[symbol] = self.quote_cache[symbol]
                else:
                    symbols_to_fetch.append(symbol)
        else:
            # Using manual cache expiration
            for symbol in symbols:
                if symbol in self.quote_cache and symbol in self.quote_cache_expiry:
                    # If cache valid, use it
                    if now - self.quote_cache_expiry[symbol] < self.quote_cache_duration:
                        cached_quotes[symbol] = self.quote_cache[symbol]
                    else:
                        symbols_to_fetch.append(symbol)
                else:
                    symbols_to_fetch.append(symbol)
        
        # If all symbols in cache, return cached quotes
        if not symbols_to_fetch:
            return cached_quotes
        
        # Fetch quotes for uncached symbols
        endpoint = "/markets/quotes"
        params = {
            "symbols": ",".join(symbols_to_fetch),
            "greeks": "true" if greeks else "false"
        }
        
        try:
            response = self._make_request("GET", endpoint, params=params)
            quotes_data = response.get("quotes", {})
            quotes = quotes_data.get("quote", [])
            
            # Handle single quote result
            if isinstance(quotes, dict):
                quotes = [quotes]
            
            # Process quotes and update cache
            for quote in quotes:
                symbol = quote.get("symbol")
                if symbol:
                    cached_quotes[symbol] = quote
                    
                    # Update cache based on implementation
                    if CACHETOOLS_AVAILABLE:
                        self.quote_cache[symbol] = quote
                    else:
                        self.quote_cache[symbol] = quote
                        self.quote_cache_expiry[symbol] = now
            
            return cached_quotes
            
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            return cached_quotes
        return cached_quotes
    
    def get_quote(self, symbol: str, greeks: bool = False) -> Dict:
        """
        Get quote for a single symbol
        
        Args:
            symbol: Symbol to get quote for
            greeks: Include Greeks for options (default: False)
            
        Returns:
            Quote data
        """
        quotes = self.get_quotes(symbol, greeks)
        return quotes.get(symbol, {})
    
    def get_option_chain(self, symbol: str, expiration: str = None, greeks: bool = True) -> List[Dict]:
        """
        Get option chain for a symbol
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date in YYYY-MM-DD format
            greeks: Include Greeks in the response
            
        Returns:
            List of options
        """
        endpoint = "/markets/options/chains"
        params = {
            "symbol": symbol,
            "greeks": "true" if greeks else "false"
        }
        
        if expiration:
            params["expiration"] = expiration
        
        response = self._make_request("GET", endpoint, params=params)
        options_data = response.get("options", {})
        
        if options_data == "null" or not options_data:
            return []
            
        options = options_data.get("option", [])
        # If there's only one option, the API returns it as a dict, not a list
        if isinstance(options, dict):
            options = [options]
            
        return options
    
    def get_option_expirations(self, symbol: str, include_all_roots: bool = False) -> List[str]:
        """
        Get available option expirations for a symbol
        
        Args:
            symbol: Underlying symbol
            include_all_roots: Include all options roots
            
        Returns:
            List of expiration dates (YYYY-MM-DD)
        """
        endpoint = "/markets/options/expirations"
        params = {
            "symbol": symbol,
            "includeAllRoots": "true" if include_all_roots else "false"
        }
        
        response = self._make_request("GET", endpoint, params=params)
        expirations_data = response.get("expirations", {})
        
        if expirations_data == "null" or not expirations_data:
            return []
            
        expirations = expirations_data.get("date", [])
        # If there's only one expiration, the API returns it as a string, not a list
        if isinstance(expirations, str):
            expirations = [expirations]
            
        return expirations
    
    def get_market_calendar(self, month: Optional[int] = None, year: Optional[int] = None) -> List[Dict]:
        """
        Get market calendar
        
        Args:
            month: Calendar month (1-12)
            year: Calendar year (YYYY)
            
        Returns:
            List of market calendar days
        """
        endpoint = "/markets/calendar"
        params = {}
        
        if month:
            params["month"] = month
        if year:
            params["year"] = year
        
        response = self._make_request("GET", endpoint, params=params)
        calendar_data = response.get("calendar", {})
        
        if calendar_data == "null" or not calendar_data:
            return []
            
        days = calendar_data.get("days", {}).get("day", [])
        # If there's only one day, the API returns it as a dict, not a list
        if isinstance(days, dict):
            days = [days]
            
        return days
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            True if market is open, False otherwise
        """
        endpoint = "/markets/clock"
        response = self._make_request("GET", endpoint)
        clock_data = response.get("clock", {})
        
        if clock_data == "null" or not clock_data:
            return False
            
        return clock_data.get("state") == "open"
    
    # --- Order Methods ---
    
    def place_equity_order(self, 
                           symbol: str, 
                           side: str, 
                           quantity: int, 
                           order_type: str = "market",
                           duration: str = "day",
                           price: float = None,
                           stop: float = None,
                           tag: Optional[str] = None) -> Dict:
        """
        Place an equity order
        
        Args:
            symbol: Symbol to trade
            side: Order side ('buy' or 'sell')
            quantity: Number of shares
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            duration: Time in force ('day', 'gtc', 'pre', 'post')
            price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop: Stop price (required for 'stop' and 'stop_limit' orders)
            
        Returns:
            Dictionary with order details
        """
        endpoint = f"/accounts/{self.account_id}/orders"
        
        # Validate required fields
        if order_type in ["limit", "stop_limit"] and price is None:
            raise ValueError("Price is required for limit orders")
        if order_type in ["stop", "stop_limit"] and stop is None:
            raise ValueError("Stop price is required for stop orders")
        
        # Prepare order data
        data = {
            "class": "equity",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": duration
        }
        
        # Add price and stop if applicable
        if price is not None:
            data["price"] = price
        if stop is not None:
            data["stop"] = stop
        
        # Optional idempotency tag
        if tag:
            data["tag"] = tag
        
        logger.info(f"Placing {side} order for {quantity} shares of {symbol} at {price if price else 'market price'}{f' tag={tag}' if tag else ''}")
        response = self._make_request("POST", endpoint, data=data)
        return response.get("order", {})
    
    def place_option_order(self,
                          option_symbol: str,
                          side: str,
                          quantity: int,
                          order_type: str = "market",
                          duration: str = "day",
                          price: float = None,
                          stop: float = None,
                          tag: Optional[str] = None) -> Dict:
        """
        Place an option order
        
        Args:
            option_symbol: OCC option symbol (e.g., 'AAPL220121C00150000')
            side: Order side ('buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close')
            quantity: Number of contracts
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            duration: Time in force ('day', 'gtc')
            price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop: Stop price (required for 'stop' and 'stop_limit' orders)
            
        Returns:
            Dictionary with order details
        """
        endpoint = f"/accounts/{self.account_id}/orders"
        
        # Validate required fields
        if order_type in ["limit", "stop_limit"] and price is None:
            raise ValueError("Price is required for limit orders")
        if order_type in ["stop", "stop_limit"] and stop is None:
            raise ValueError("Stop price is required for stop orders")
        
        # Prepare order data
        data = {
            "class": "option",
            "symbol": option_symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": duration
        }
        
        # Add price and stop if applicable
        if price is not None:
            data["price"] = price
        if stop is not None:
            data["stop"] = stop
        
        if tag:
            data["tag"] = tag
        
        logger.info(f"Placing {side} order for {quantity} contracts of {option_symbol} at {price if price else 'market price'}{f' tag={tag}' if tag else ''}")
        response = self._make_request("POST", endpoint, data=data)
        return response.get("order", {})
    
    def modify_order(self,
                    order_id: str,
                    order_type: str = None,
                    duration: str = None,
                    price: float = None,
                    stop: float = None) -> Dict:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            order_type: New order type
            duration: New time in force
            price: New limit price
            stop: New stop price
            
        Returns:
            Dictionary with modified order details
        """
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"
        
        # Prepare order data
        data = {}
        if order_type:
            data["type"] = order_type
        if duration:
            data["duration"] = duration
        if price is not None:
            data["price"] = price
        if stop is not None:
            data["stop"] = stop
        
        logger.info(f"Modifying order {order_id}")
        response = self._make_request("PUT", endpoint, data=data)
        return response.get("order", {})
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with cancellation details
        """
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"
        logger.info(f"Cancelling order {order_id}")
        response = self._make_request("DELETE", endpoint)
        return response.get("order", {})
    
    # --- Helper Methods ---
    
    def get_option_symbol(self, 
                         underlying: str, 
                         expiration: str, 
                         option_type: str, 
                         strike: float) -> str:
        """
        Generate an OCC option symbol
        
        Args:
            underlying: Underlying symbol (e.g., 'AAPL')
            expiration: Expiration date in YYYY-MM-DD format
            option_type: 'call' or 'put'
            strike: Strike price
            
        Returns:
            OCC option symbol (e.g., 'AAPL220121C00150000')
        """
        # Pad the underlying symbol to 6 characters
        padded_underlying = underlying.ljust(6)
        
        # Format the expiration date (YYMMDD)
        exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        exp_formatted = exp_date.strftime("%y%m%d")
        
        # Format the option type (C or P)
        option_type_code = "C" if option_type.lower() == "call" else "P"
        
        # Format the strike price (multiply by 1000 and pad to 8 digits)
        strike_formatted = f"{int(strike * 1000):08d}"
        
        # Combine to create the OCC symbol
        return f"{padded_underlying}{exp_formatted}{option_type_code}{strike_formatted}"
    
    def get_account_summary(self) -> Dict:
        """
        Get a comprehensive account summary including balances,
        positions, and open orders
        
        Returns:
            Dictionary with account summary
        """
        # Get account balances
        balances = self.get_account_balances()
        
        # Get positions
        positions = self.get_positions()
        
        # Get open orders
        open_orders = self.get_orders(status="open")
        
        # Calculate some additional metrics
        total_position_value = sum(float(position.get("market_value", 0)) for position in positions)
        cash = float(balances.get("cash", 0))
        equity = float(balances.get("equity", 0))
        
        return {
            "account_number": self.account_id,
            "type": balances.get("account_type", ""),
            "cash": cash,
            "equity": equity,
            "buying_power": float(balances.get("buying_power", 0)),
            "day_trades_remaining": balances.get("day_trading_buying_power", 0),
            "positions": {
                "count": len(positions),
                "total_value": total_position_value,
                "percentage_of_equity": (total_position_value / equity * 100) if equity > 0 else 0,
                "details": positions
            },
            "open_orders": {
                "count": len(open_orders),
                "details": open_orders
            },
            "market_hours": {
                "is_open": self.is_market_open()
            }
        }
    
    def get_historical_data(self, 
                          symbol: str, 
                          interval: str = "daily", 
                          start_date: str = None, 
                          end_date: str = None,
                          days_back: int = None) -> Dict:
        """
        Get historical market data for a symbol
        
        Args:
            symbol: Symbol to get data for
            interval: Data interval (daily, weekly, monthly)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            days_back: Alternative to start_date, get X days back from end_date or today
            
        Returns:
            Dictionary with historical price data
        """
        endpoint = "/markets/history"
        
        # Set dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if days_back and not start_date:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
        # Required parameters
        params = {
            "symbol": symbol,
            "interval": interval
        }
        
        # Add dates if provided
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        
        response = self._make_request("GET", endpoint, params=params)
        history_data = response.get("history", {})
        
        if history_data == "null" or not history_data:
            return {"day": []}
            
        return history_data


class TradierAPIError(Exception):
    """Exception raised when Tradier API returns an error"""
    pass 

class TradierRateLimitError(TradierAPIError):
    """Exception raised when Tradier API rate limits are exceeded"""
    pass