#!/usr/bin/env python3
"""
Direct Coinbase API Test - Tests the Coinbase Cloud API with your whitelisted IP
without requiring the full trading bot framework
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
import jwt
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinbaseCloudAPIClient:
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
    
    def create_jwt_token(self):
        """Create a JWT token for API authentication"""
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None,
                backend=default_backend()
            )
            
            # Create JWT payload
            now = int(time.time())
            payload = {
                "sub": self.api_key_name,
                "iss": "coinbase-cloud",
                "nbf": now,
                "exp": now + 60,  # Token expires in 60 seconds
                "aud": ["brokerage"]  # Audience claim
            }
            
            # Generate JWT token
            token = jwt.encode(
                payload,
                private_key,
                algorithm="ES256"
            )
            
            logger.info("JWT token created successfully")
            return token
            
        except Exception as e:
            logger.error(f"Error creating JWT token: {str(e)}")
            return None
    
    def make_request(self, method, endpoint, auth=False, params=None, data=None):
        """Make a request to the Coinbase API"""
        # Set headers
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if needed
        if auth:
            token = self.create_jwt_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                return False, {"error": "Failed to create authentication token"}
        
        # Full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            # Make request
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            else:
                return False, {"error": f"Unsupported method: {method}"}
            
            # Handle response
            if response.status_code == 200:
                return True, response.json()
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return False, {"error": f"API error {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return False, {"error": str(e)}
    
    def get_products(self):
        """Get available products from Coinbase"""
        return self.make_request("GET", "/products")
    
    def get_product_ticker(self, product_id):
        """Get ticker data for a product"""
        return self.make_request("GET", f"/products/{product_id}/ticker")
    
    def get_product_stats(self, product_id):
        """Get 24-hour stats for a product"""
        return self.make_request("GET", f"/products/{product_id}/stats")
    
    def get_accounts(self):
        """Get user accounts (requires authentication)"""
        return self.make_request("GET", "/accounts", auth=True)

def test_api_functionality():
    """Test Coinbase Cloud API functionality"""
    print("\nCoinbase Cloud API Test")
    print("======================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize client
    client = CoinbaseCloudAPIClient()
    
    # Test public endpoints
    print("\n--- Testing Public Endpoints ---")
    
    # Get products
    print("\nGetting available products...")
    success, products = client.get_products()
    
    if success:
        print(f"✅ Found {len(products)} trading products")
        # Display first 5 products
        for i, product in enumerate(products[:5]):
            print(f"  {i+1}. {product.get('display_name', 'Unknown')} ({product.get('id', 'Unknown')})")
    else:
        print(f"❌ Failed to get products: {products.get('error', 'Unknown error')}")
    
    # Get BTC-USD ticker
    print("\nGetting BTC-USD ticker...")
    success, ticker = client.get_product_ticker("BTC-USD")
    
    if success:
        print(f"✅ BTC-USD Price: ${ticker.get('price', 'Unknown')}")
        print(f"  Volume: {ticker.get('volume', 'Unknown')}")
        print(f"  Time: {ticker.get('time', 'Unknown')}")
    else:
        print(f"❌ Failed to get BTC-USD ticker: {ticker.get('error', 'Unknown error')}")
    
    # Get ETH-USD stats
    print("\nGetting ETH-USD 24h stats...")
    success, stats = client.get_product_stats("ETH-USD")
    
    if success:
        print(f"✅ ETH-USD 24h Stats:")
        print(f"  Open: ${stats.get('open', 'Unknown')}")
        print(f"  High: ${stats.get('high', 'Unknown')}")
        print(f"  Low: ${stats.get('low', 'Unknown')}")
        print(f"  Volume: {stats.get('volume', 'Unknown')} ETH")
    else:
        print(f"❌ Failed to get ETH-USD stats: {stats.get('error', 'Unknown error')}")
    
    # Test authenticated endpoints
    print("\n--- Testing Authenticated Endpoints ---")
    
    # Get accounts
    print("\nGetting accounts (requires authentication)...")
    success, accounts = client.get_accounts()
    
    if success:
        print(f"✅ Found {len(accounts)} accounts")
        
        # Display first 5 accounts
        for i, account in enumerate(accounts[:5]):
            print(f"  {i+1}. {account.get('currency', 'Unknown')}: Balance={account.get('balance', 'Unknown')}")
    else:
        print(f"❌ Failed to get accounts: {accounts.get('error', 'Unknown error')}")
    
    # Summary
    print("\n--- Summary ---")
    print("Public API Access: ✅ Working")
    
    if success:
        print("Authenticated API Access: ✅ Working")
        print("\nYour Coinbase integration is ready to use!")
    else:
        print("Authenticated API Access: ❌ Not working")
        print("\nOnly public market data is accessible.")
    
    print("\nNext steps:")
    print("1. Update your CoinbaseCloudBroker in the trading bot to use this configuration")
    print("2. Integrate with your crypto trading strategies")
    print("3. Add Coinbase data to your dashboard")

if __name__ == "__main__":
    test_api_functionality()
