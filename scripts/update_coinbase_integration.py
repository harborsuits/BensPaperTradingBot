#!/usr/bin/env python3
"""
Coinbase Cloud API Integration - Configured for IP-Whitelisted Access
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import requests

# Add crypto dependencies
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinbaseCloudAPI:
    """
    Coinbase Cloud API client using the working IP-whitelisted endpoint
    """
    
    def __init__(
        self, 
        api_key_name: str = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335",
        private_key: str = None,
        read_only: bool = True
    ):
        """
        Initialize Coinbase Cloud API client with API credentials
        
        Args:
            api_key_name: Coinbase Cloud API key name
            private_key: EC private key in PEM format 
            read_only: If True, prevents any trading operations
        """
        self.api_key_name = api_key_name
        
        # Load private key from string or file
        if private_key:
            self.private_key = private_key
        else:
            # Default private key if not provided
            self.private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
            
        self.read_only = read_only
        
        # Base URL for the Coinbase API (this is the one that's working with IP whitelist)
        self.base_url = "https://api.exchange.coinbase.com"
        
        # Authentication token cache
        self._auth_token = None
        self._token_expiry = 0
        
        # Test connection
        self.test_connection()
        
    def test_connection(self) -> bool:
        """Test connection to the Coinbase API"""
        try:
            # Try to get the list of products (this endpoint works)
            success, data = self.get_products()
            
            if success:
                logger.info(f"Successfully connected to Coinbase API - found {len(data)} products")
                return True
            else:
                logger.warning(f"Failed to connect to Coinbase API: {data.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False
            
    def get_auth_token(self) -> Optional[str]:
        """
        Get a JWT authentication token, generating a new one if expired
        
        Returns:
            JWT token string or None if generation failed
        """
        current_time = time.time()
        
        # If token exists and not expired (with 30s buffer), return it
        if self._auth_token and current_time < self._token_expiry - 30:
            return self._auth_token
            
        try:
            # Load the private key
            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None,
                backend=default_backend()
            )
            
            # Create JWT payload - expires in 60 seconds
            now = int(current_time)
            payload = {
                "sub": self.api_key_name,
                "iss": "coinbase-cloud",
                "nbf": now,
                "exp": now + 60,
                "aud": ["brokerage"]
            }
            
            # Generate JWT token
            token = jwt.encode(
                payload,
                private_key,
                algorithm="ES256"
            )
            
            # Cache token and expiry
            self._auth_token = token
            self._token_expiry = now + 60
            
            return token
            
        except Exception as e:
            logger.error(f"Error generating authentication token: {str(e)}")
            return None
            
    def make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        auth_required: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Make an API request to Coinbase
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/products')
            params: Query parameters
            data: Request body data
            auth_required: Whether authentication is required
            
        Returns:
            Tuple of (success, data)
        """
        # Ensure endpoint starts with a slash
        if not endpoint.startswith('/'):
            endpoint = f'/{endpoint}'
            
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if required
        if auth_required:
            token = self.get_auth_token()
            if not token:
                return False, {"error": "Failed to generate authentication token"}
                
            headers["Authorization"] = f"Bearer {token}"
        
        try:
            # Check if read-only mode prevents this operation
            if self.read_only and method != "GET":
                logger.warning(f"READ-ONLY MODE: Blocked {method} request to {endpoint}")
                return False, {"error": "Operation blocked due to read-only mode"}
                
            # Make the request
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                return False, {"error": f"Unsupported method: {method}"}
                
            # Handle response
            if response.status_code == 200:
                return True, response.json()
            else:
                error = f"API error ({response.status_code}): {response.text}"
                logger.error(error)
                return False, {"error": error, "status_code": response.status_code}
                
        except Exception as e:
            error = f"Request error: {str(e)}"
            logger.error(error)
            return False, {"error": error}
    
    # API Methods
    
    def get_products(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get available trading products
        
        Returns:
            Tuple of (success, products_list)
        """
        # This endpoint works without authentication
        return self.make_request("GET", "/products", auth_required=False)
        
    def get_product_ticker(self, product_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Get current ticker data for a product
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            Tuple of (success, ticker_data)
        """
        return self.make_request("GET", f"/products/{product_id}/ticker", auth_required=False)
        
    def get_product_trades(self, product_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get recent trades for a product
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            Tuple of (success, trades_list)
        """
        return self.make_request("GET", f"/products/{product_id}/trades", auth_required=False)
        
    def get_product_candles(
        self, 
        product_id: str,
        granularity: int = 3600,  # 1 hour in seconds
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Tuple[bool, List[List[float]]]:
        """
        Get historical candles for a product
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            granularity: Candle interval in seconds (60, 300, 900, 3600, 21600, 86400)
            start: Start time in ISO 8601 format
            end: End time in ISO 8601 format
            
        Returns:
            Tuple of (success, candles_data)
        """
        params = {"granularity": granularity}
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return self.make_request(
            "GET", 
            f"/products/{product_id}/candles", 
            params=params,
            auth_required=False
        )
        
    def get_accounts(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get user accounts (requires authentication and correct permissions)
        
        Returns:
            Tuple of (success, accounts_list)
        """
        return self.make_request("GET", "/accounts")
        
    def get_orders(self, status: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get orders with optional status filter
        
        Args:
            status: Order status filter (open, pending, active, done)
            
        Returns:
            Tuple of (success, orders_list)
        """
        params = {}
        if status:
            params["status"] = status
            
        return self.make_request("GET", "/orders", params=params)
        
    def place_market_order(
        self,
        product_id: str,
        side: str,  # buy or sell
        size: Optional[str] = None,  # Amount of base currency to buy or sell
        funds: Optional[str] = None  # Amount of quote currency to use
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Place a market order
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            size: Amount of base currency to buy or sell
            funds: Amount of quote currency to use
            
        Returns:
            Tuple of (success, order_data)
        """
        if self.read_only:
            logger.warning("READ-ONLY MODE: Market order placement blocked")
            return False, {"error": "Order placement blocked due to read-only mode"}
            
        # Either size or funds must be specified
        if not size and not funds:
            return False, {"error": "Either size or funds must be specified"}
            
        # Prepare order data
        data = {
            "product_id": product_id,
            "side": side,
            "type": "market"
        }
        
        if size:
            data["size"] = size
        if funds:
            data["funds"] = funds
            
        return self.make_request("POST", "/orders", data=data)
        
    def place_limit_order(
        self,
        product_id: str,
        side: str,  # buy or sell
        price: str,  # Price per unit of base currency
        size: str,  # Amount of base currency to buy or sell
        time_in_force: str = "GTC"  # GTC (Good Till Canceled), GTT, IOC, FOK
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Place a limit order
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            price: Price per unit of base currency
            size: Amount of base currency to buy or sell
            time_in_force: Time in force policy
            
        Returns:
            Tuple of (success, order_data)
        """
        if self.read_only:
            logger.warning("READ-ONLY MODE: Limit order placement blocked")
            return False, {"error": "Order placement blocked due to read-only mode"}
            
        # Prepare order data
        data = {
            "product_id": product_id,
            "side": side,
            "price": price,
            "size": size,
            "type": "limit",
            "time_in_force": time_in_force
        }
            
        return self.make_request("POST", "/orders", data=data)
        
    def cancel_order(self, order_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Tuple of (success, response_data)
        """
        if self.read_only:
            logger.warning("READ-ONLY MODE: Order cancellation blocked")
            return False, {"error": "Order cancellation blocked due to read-only mode"}
            
        return self.make_request("DELETE", f"/orders/{order_id}")

# Simple usage example
def test_market_data():
    """Test retrieving market data from Coinbase"""
    client = CoinbaseCloudAPI(read_only=True)
    
    print("\n--- COINBASE MARKET DATA TEST ---")
    
    # Get available trading pairs
    success, products = client.get_products()
    if success:
        print(f"Found {len(products)} trading products")
        
        # Display first 5 products
        for i, product in enumerate(products[:5]):
            print(f"  {i+1}. {product.get('display_name', 'Unknown')} ({product.get('id', 'Unknown')})")
            
        # Get BTC-USD data specifically
        btc_product = next((p for p in products if p.get('id') == 'BTC-USD'), None)
        if btc_product:
            print(f"\nBTC-USD Details:")
            print(f"  Base Currency: {btc_product.get('base_currency', 'Unknown')}")
            print(f"  Quote Currency: {btc_product.get('quote_currency', 'Unknown')}")
            print(f"  Minimum Size: {btc_product.get('base_min_size', 'Unknown')}")
            
            # Get current price
            success, ticker = client.get_product_ticker('BTC-USD')
            if success:
                print(f"\nBTC-USD Current Price: ${ticker.get('price', 'Unknown')}")
                print(f"24h Volume: {ticker.get('volume', 'Unknown')} BTC")
            else:
                print(f"\nError getting BTC-USD ticker: {ticker.get('error', 'Unknown error')}")
                
            # Get historical data
            print("\nGetting historical candles for BTC-USD...")
            end_time = datetime.utcnow().isoformat()
            start_time = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).isoformat()
            
            success, candles = client.get_product_candles('BTC-USD', granularity=3600, start=start_time, end=end_time)
            if success:
                print(f"Retrieved {len(candles)} hourly candles for today")
                if candles:
                    latest = candles[0]  # Format: [timestamp, low, high, open, close, volume]
                    print(f"Latest candle: Open=${latest[3]}, High=${latest[2]}, Low=${latest[1]}, Close=${latest[4]}, Volume={latest[5]}")
            else:
                print(f"Error getting BTC-USD candles: {candles.get('error', 'Unknown error')}")
    else:
        print(f"Error getting products: {products.get('error', 'Unknown error')}")

# Main script execution
if __name__ == "__main__":
    print("Coinbase Cloud API Integration - IP-Whitelisted")
    print("==============================================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test market data retrieval
    test_market_data()
    
    print("\nNext steps:")
    print("1. Replace existing CoinbaseCloudBroker with this implementation")
    print("2. Update trading_config.yaml if needed (IP whitelist is configured)")
    print("3. Test with additional strategies in read-only mode first")
    print("\nDone!")
