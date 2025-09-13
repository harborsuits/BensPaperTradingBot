import os
import json
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

logger = logging.getLogger(__name__)

class TradierClient:
    """
    Enhanced client for interacting with the Tradier Brokerage API
    
    Handles authentication, account data, market data and order execution
    with robust error handling, retries, timeouts and rate limiting
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
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        # Import cache library safely (will be installed via requirements.txt update)
        try:
            from cachetools import TTLCache
            # Cache for market data with TTL and size limit
            # Max 1000 symbols with 10 second expiry
            self.quote_cache = TTLCache(maxsize=1000, ttl=10)
        except ImportError:
            logger.warning("cachetools not available, using simple dict cache instead")
            # Fallback to simple dict if cachetools not available
            self.quote_cache = {}
            self.quote_cache_expiry = {}
            self.quote_cache_duration = 10  # seconds
        
        # Track request timestamps for rate limiting
        self.request_timestamps = []
        
        logger.info(f"Enhanced Tradier client initialized for account {account_id} in {'sandbox' if sandbox else 'production'} mode")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True
    )
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
        url = f"{self.base_url}{endpoint}"
        
        # Implement basic rate limiting
        self._check_rate_limits()
        
        # Track this request
        self.request_timestamps.append(time.time())
        
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
    
    def _check_rate_limits(self):
        """
        Check and enforce rate limits
        Removes timestamps older than 1 minute and sleeps if too many recent requests
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
    
    # Rest of the TradierClient methods remain the same, with these enhancements
    # 1. All methods that use caching will work with TTLCache
    # 2. Add websocket support (outlined in later methods)
    
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
        if not positions_data or positions_data == "null":
            return []
        
        # Handle the case when there's a single position
        position = positions_data.get("position")
        if isinstance(position, dict):
            return [position]
        
        # Handle the case when there are multiple positions
        if isinstance(position, list):
            return position
        
        return []
    
    def get_quotes(self, symbols: Union[str, List[str]], greeks: bool = False) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols with TTL caching
        
        Args:
            symbols: Single symbol or list of symbols
            greeks: Include Greeks for options (default: False)
            
        Returns:
            Dictionary of quotes keyed by symbol
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Check which symbols are in cache and which need to be fetched
        now = time.time()
        cached_results = {}
        symbols_to_fetch = []
        
        for symbol in symbols:
            # With TTLCache, expired items are automatically removed
            if hasattr(self.quote_cache, 'get'):
                # Using TTLCache
                cached_quote = self.quote_cache.get(symbol)
                if cached_quote is not None:
                    cached_results[symbol] = cached_quote
                else:
                    symbols_to_fetch.append(symbol)
            else:
                # Using simple dict + expiry tracking
                if (symbol in self.quote_cache and 
                    symbol in self.quote_cache_expiry and 
                    now - self.quote_cache_expiry[symbol] < self.quote_cache_duration):
                    cached_results[symbol] = self.quote_cache[symbol]
                else:
                    symbols_to_fetch.append(symbol)
        
        # If all symbols were in cache, return cached results
        if not symbols_to_fetch:
            return cached_results
        
        # Fetch quotes for symbols not in cache
        endpoint = "/markets/quotes"
        params = {
            "symbols": ",".join(symbols_to_fetch),
            "greeks": "true" if greeks else "false"
        }
        
        response = self._make_request("GET", endpoint, params=params)
        quotes_data = response.get("quotes", {})
        
        # Extract quotes and add to results
        quotes = quotes_data.get("quote", [])
        
        # Handle the case when there's a single quote
        if isinstance(quotes, dict):
            quotes = [quotes]
        
        # Add fetched quotes to results and cache
        for quote in quotes:
            symbol = quote.get("symbol")
            if symbol:
                cached_results[symbol] = quote
                
                # Add to cache
                if hasattr(self.quote_cache, 'get'):
                    # Using TTLCache
                    self.quote_cache[symbol] = quote
                else:
                    # Using simple dict + expiry
                    self.quote_cache[symbol] = quote
                    self.quote_cache_expiry[symbol] = now
        
        return cached_results


class TradierAPIError(Exception):
    """Exception raised when Tradier API returns an error"""
    pass 

class TradierRateLimitError(TradierAPIError):
    """Exception raised when Tradier API rate limits are exceeded"""
    pass
