#!/usr/bin/env python3
"""
Coinbase Advanced API Integration
================================

This script provides a reliable integration with Coinbase Advanced Trading API.
It handles:
1. Authentication without passphrase
2. Market data access
3. Account information
4. Order placement/management with optional read-only mode

Usage:
    python working_coinbase_integration.py
"""

import os
import time
import json
import base64
import hmac
import hashlib
from typing import Dict, Any, Optional, Tuple, List
import requests
from datetime import datetime

class CoinbaseAdvancedAPI:
    """
    Coinbase Advanced Trading API client - compatible with BenBot's account credentials
    """
    
    def __init__(
        self, 
        api_key: str = "acc.adb53c0e-35a0-4171-b237-a19fec741363",
        api_secret: str = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ==",
        base_url: str = "https://api.coinbase.com",
        read_only: bool = True
    ):
        """
        Initialize the Coinbase Advanced API client.
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            base_url: Base URL for Coinbase API
            read_only: If True, prevents any trading operations
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.read_only = read_only
        self.session = requests.Session()
        
    def _get_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        Generate the signature for Coinbase API authentication.
        
        Args:
            timestamp: Current timestamp
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body as JSON string (for POST requests)
            
        Returns:
            Base64-encoded signature
        """
        message = f"{timestamp}{method}{request_path}{body}"
        
        # Decode the base64 secret
        try:
            secret = base64.b64decode(self.api_secret)
        except Exception as e:
            raise ValueError(f"Invalid API secret: {str(e)}")
            
        # Create HMAC-SHA256 signature
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()
        
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Send a request to the Coinbase API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /api/v3/brokerage/accounts)
            params: Query parameters
            data: POST data as dictionary
            
        Returns:
            Tuple of (success, response_data)
        """
        # Build the request URL
        if not endpoint.startswith('/'):
            endpoint = f'/{endpoint}'
            
        url = f"{self.base_url}{endpoint}"
        
        # Convert data to JSON string if provided
        body = ""
        if data:
            body = json.dumps(data)
        
        # Prepare headers
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-SIGN": self._get_signature(timestamp, method, endpoint, body)
        }
        
        try:
            # Check if read-only mode is enabled for write operations
            if self.read_only and method != "GET":
                print(f"⚠️ READ-ONLY MODE: {method} request to {endpoint} blocked")
                return False, {"error": "Operation blocked due to read-only mode"}
            
            # Make the request
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                return False, {"error": f"Unsupported method: {method}"}
                
            # Handle the response
            if response.status_code == 200:
                return True, response.json()
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return False, {"error": error_msg, "status_code": response.status_code}
                
        except Exception as e:
            error_msg = f"Request error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, {"error": error_msg}
            
    # Public API methods (no authentication required)
    def get_products(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Get available trading products"""
        success, data = self._request("GET", "/api/v3/brokerage/products")
        if success:
            return True, data.get("products", [])
        return False, []
        
    def get_product_candles(self, product_id: str, start: str, end: str, granularity: str = "ONE_DAY") -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get historical candles for a product
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            start: Start time as ISO-8601 string
            end: End time as ISO-8601 string
            granularity: Candle interval (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, 
                         THIRTY_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY)
        """
        params = {
            "product_id": product_id,
            "start": start,
            "end": end,
            "granularity": granularity
        }
        success, data = self._request("GET", "/api/v3/brokerage/products/candles", params=params)
        if success:
            return True, data.get("candles", [])
        return False, []
    
    def get_product_ticker(self, product_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Get current ticker data for a product"""
        success, data = self._request("GET", f"/api/v3/brokerage/products/{product_id}/ticker")
        if success:
            return True, data
        return False, {}
    
    # Authenticated API methods
    def get_accounts(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Get user accounts"""
        success, data = self._request("GET", "/api/v3/brokerage/accounts")
        if success:
            return True, data.get("accounts", [])
        return False, []
    
    def get_account(self, account_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Get details for a specific account"""
        success, data = self._request("GET", f"/api/v3/brokerage/accounts/{account_id}")
        if success:
            return True, data.get("account", {})
        return False, {}
    
    def create_order(
        self, 
        product_id: str, 
        side: str,  # BUY or SELL
        order_type: str = "MARKET",  # MARKET, LIMIT, STOP, STOP_LIMIT
        client_order_id: str = None,
        base_size: str = None,  # Base currency amount
        quote_size: str = None,  # Quote currency amount
        limit_price: str = None,  # Required for LIMIT orders
        stop_price: str = None,  # Required for STOP orders
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Create a new order
        
        At least one of base_size or quote_size must be specified.
        """
        if self.read_only:
            print("⚠️ READ-ONLY MODE: Order placement blocked")
            return False, {"error": "Order placement blocked due to read-only mode"}
            
        # Validate inputs
        if not product_id or not side:
            return False, {"error": "product_id and side are required"}
            
        if not base_size and not quote_size:
            return False, {"error": "Either base_size or quote_size must be provided"}
            
        if order_type == "LIMIT" and not limit_price:
            return False, {"error": "limit_price is required for LIMIT orders"}
            
        if order_type in ["STOP", "STOP_LIMIT"] and not stop_price:
            return False, {"error": "stop_price is required for STOP orders"}
        
        # Build order data
        order_data = {
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                order_type.lower(): {}
            }
        }
        
        # Set size
        if base_size:
            order_data["order_configuration"][order_type.lower()]["base_size"] = base_size
        if quote_size:
            order_data["order_configuration"][order_type.lower()]["quote_size"] = quote_size
            
        # Set prices based on order type
        if order_type == "LIMIT":
            order_data["order_configuration"]["limit"]["limit_price"] = limit_price
            
        elif order_type == "STOP":
            order_data["order_configuration"]["stop"]["stop_price"] = stop_price
            
        elif order_type == "STOP_LIMIT":
            order_data["order_configuration"]["stop_limit"]["limit_price"] = limit_price
            order_data["order_configuration"]["stop_limit"]["stop_price"] = stop_price
            
        # Add client order ID if provided
        if client_order_id:
            order_data["client_order_id"] = client_order_id
            
        # Send the request
        return self._request("POST", "/api/v3/brokerage/orders", data=order_data)
    
    def get_orders(self, product_id: str = None, order_status: List[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get list of orders
        
        Args:
            product_id: Filter by product
            order_status: Filter by status (OPEN, FILLED, CANCELLED, EXPIRED, FAILED)
        """
        params = {}
        if product_id:
            params["product_id"] = product_id
        if order_status:
            params["order_status"] = order_status
            
        success, data = self._request("GET", "/api/v3/brokerage/orders/historical/batch", params=params)
        if success:
            return True, data.get("orders", [])
        return False, []
    
    def get_order(self, order_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Get details for a specific order"""
        success, data = self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
        if success:
            return True, data.get("order", {})
        return False, {}
    
    def cancel_order(self, order_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Cancel an open order"""
        if self.read_only:
            print("⚠️ READ-ONLY MODE: Order cancellation blocked")
            return False, {"error": "Order cancellation blocked due to read-only mode"}
            
        return self._request("DELETE", f"/api/v3/brokerage/orders/{order_id}")

# Function to test basic connectivity and authentication
def test_coinbase_connection(api_key, api_secret):
    """Test connection to Coinbase API"""
    print("\n--- TESTING COINBASE CONNECTION ---")
    
    # Initialize the client in read-only mode
    client = CoinbaseAdvancedAPI(api_key=api_key, api_secret=api_secret, read_only=True)
    
    # Test endpoints
    test_methods = [
        # Try public endpoints first
        ("Get Products", lambda: client.get_products()),
        ("Get BTC-USD Ticker", lambda: client.get_product_ticker("BTC-USD")),
        # Then try authenticated endpoints
        ("Get Accounts", lambda: client.get_accounts())
    ]
    
    results = []
    for name, method in test_methods:
        print(f"\nTesting: {name}")
        try:
            success, data = method()
            if success:
                print(f"✅ SUCCESS: {name}")
                results.append((name, "Success"))
            else:
                print(f"❌ FAILED: {name} - {data.get('error', 'Unknown error')}")
                results.append((name, "Failed"))
        except Exception as e:
            print(f"❌ ERROR: {name} - {str(e)}")
            results.append((name, f"Error: {str(e)}"))
    
    print("\n--- TEST RESULTS ---")
    for name, result in results:
        print(f"{name}: {result}")
    
    # Determine overall status
    if any(result.startswith("Success") for _, result in results):
        print("\n✅ CONNECTED: Some endpoints are accessible")
        return True
    else:
        print("\n❌ DISCONNECTED: All endpoints failed")
        return False

# Example usage
if __name__ == "__main__":
    print("Coinbase Advanced API Integration Test")
    print("=====================================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode: READ-ONLY (No trading operations will be performed)")
    
    # Test API with provided credentials
    api_key = "acc.adb53c0e-35a0-4171-b237-a19fec741363"
    api_secret = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
    
    connection_successful = test_coinbase_connection(api_key, api_secret)
    
    if connection_successful:
        # Initialize the full client
        coinbase = CoinbaseAdvancedAPI(api_key=api_key, api_secret=api_secret, read_only=True)
        
        # Demo retrieving BTC-USD price
        print("\n--- RETRIEVING MARKET DATA ---")
        success, ticker = coinbase.get_product_ticker("BTC-USD")
        
        if success:
            print(f"BTC-USD Price: ${ticker.get('price', 'N/A')}")
            print(f"24h Volume: {ticker.get('volume', 'N/A')} BTC")
        else:
            print("Failed to retrieve market data")
            
        # Demonstrate other capabilities in read-only mode
        print("\n--- ACCOUNT INFORMATION ---")
        success, accounts = coinbase.get_accounts()
        
        if success:
            print(f"Found {len(accounts)} accounts")
            for i, account in enumerate(accounts[:5]):  # Show first 5 accounts
                print(f"Account {i+1}: {account.get('name', 'Unknown')} - {account.get('available_balance', {}).get('value', '0')} {account.get('available_balance', {}).get('currency', '')}")
        else:
            print("Failed to retrieve account information")
            
        print("\n--- AVAILABLE PRODUCTS ---")
        success, products = coinbase.get_products()
        
        if success:
            print(f"Found {len(products)} trading products")
            for i, product in enumerate(products[:5]):  # Show first 5 products
                print(f"Product {i+1}: {product.get('product_id', 'Unknown')} - {product.get('display_name', 'Unknown')}")
        else:
            print("Failed to retrieve products")
            
        # Example of attempting a trade in read-only mode (will be blocked)
        print("\n--- ATTEMPTING TRADE (READ-ONLY MODE) ---")
        coinbase.create_order(
            product_id="BTC-USD",
            side="BUY",
            order_type="MARKET",
            quote_size="10.00"  # Buy $10 worth of BTC
        )
    
    print("\n--- INTEGRATION STATUS ---")
    if connection_successful:
        print("✅ Integration Ready: Use the CoinbaseAdvancedAPI class in your trading bot")
        print("   To enable real trading, initialize with read_only=False")
    else:
        print("❌ Integration Failed: Check your API credentials and permissions")
        print("   You may need to check IP whitelisting or API key permissions on Coinbase")
