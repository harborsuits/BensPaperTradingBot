#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Advanced API Test

This script tests the Coinbase Advanced Trading API, which has a different
authentication method than the Cloud API.
"""

import os
import logging
import json
import hmac
import hashlib
import time
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CoinbaseAdvancedTest")

class CoinbaseAdvancedClient:
    """Client for Coinbase Advanced Trading API"""
    
    # Base URL for Advanced API
    ADVANCED_API_URL = "https://api.coinbase.com"
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the client with API credentials
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        logger.info(f"Initialized CoinbaseAdvancedClient with API key: {api_key[:10]}...")
    
    def generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        Generate the signature for the API request
        
        Args:
            timestamp: Request timestamp as a string
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body as a string
            
        Returns:
            The generated signature
        """
        message = f"{timestamp}{method}{request_path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def make_request(self, method: str, endpoint: str, body: Dict = None) -> Tuple[bool, Dict]:
        """
        Make an authenticated request to the Coinbase Advanced API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            body: Request body as a dictionary
            
        Returns:
            Tuple of (success, response)
        """
        try:
            # Convert body to string if it exists
            body_str = json.dumps(body) if body else ""
            
            # Generate timestamp
            timestamp = str(int(time.time()))
            
            # Generate signature
            signature = self.generate_signature(timestamp, method, endpoint, body_str)
            
            # Set up headers
            headers = {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "Content-Type": "application/json"
            }
            
            # Build URL
            url = f"{self.ADVANCED_API_URL}{endpoint}"
            
            # Make request
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=body_str if body else None
            )
            
            # Handle response
            if response.status_code == 200:
                return True, response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return False, {
                    "status_code": response.status_code,
                    "error": response.text
                }
        
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return False, {"error": str(e)}
    
    def get_accounts(self) -> Tuple[bool, Dict]:
        """
        Get accounts information
        
        Returns:
            Tuple of (success, accounts)
        """
        return self.make_request("GET", "/api/v3/brokerage/accounts")
    
    def get_products(self) -> Tuple[bool, Dict]:
        """
        Get available products
        
        Returns:
            Tuple of (success, products)
        """
        return self.make_request("GET", "/api/v3/brokerage/products")
    
    def get_product_ticker(self, product_id: str) -> Tuple[bool, Dict]:
        """
        Get ticker information for a specific product
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
            
        Returns:
            Tuple of (success, ticker)
        """
        return self.make_request("GET", f"/api/v3/brokerage/products/{product_id}")

def run_advanced_test():
    """Run the Coinbase Advanced API test"""
    logger.info("Starting Coinbase Advanced API Test")
    
    # Load API credentials from environment variables or hardcode for testing
    # WARNING: Don't hardcode credentials in production code
    api_key = os.environ.get("COINBASE_API_KEY", "your_api_key_here")
    api_secret = os.environ.get("COINBASE_API_SECRET", "your_api_secret_here")
    
    # Create client
    client = CoinbaseAdvancedClient(api_key=api_key, api_secret=api_secret)
    
    # Test getting accounts
    logger.info("Testing accounts retrieval")
    success, accounts = client.get_accounts()
    if success:
        logger.info(f"Retrieved accounts: {len(accounts.get('accounts', []))} accounts")
    else:
        logger.error(f"Failed to get accounts: {accounts}")
    
    # Test getting products
    logger.info("Testing products retrieval")
    success, products_response = client.get_products()
    if success:
        products = products_response.get('products', [])
        logger.info(f"Retrieved products: {len(products)} products")
        # Display first 5 products
        for p in products[:5]:
            logger.info(f"Product: {p.get('product_id', 'unknown')}")
    else:
        logger.error(f"Failed to get products: {products_response}")
    
    # Test getting ticker for BTC-USD
    logger.info("Testing BTC-USD ticker")
    success, ticker = client.get_product_ticker("BTC-USD")
    if success:
        logger.info(f"BTC-USD ticker: {json.dumps(ticker, indent=2)}")
    else:
        logger.error(f"Failed to get ticker: {ticker}")
    
    logger.info("Coinbase Advanced API test completed")

if __name__ == "__main__":
    run_advanced_test()
