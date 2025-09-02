#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Coinbase Cloud API Test

This script tests the Coinbase Cloud API using the correct base URL.
"""

import os
import sys
import logging
import json
import time
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Import required libraries for JWT signing
import jwt
from cryptography.hazmat.primitives import serialization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FixedCoinbaseTest")

class FixedCoinbaseClient:
    """Fixed client for Coinbase Cloud API"""
    
    # Correct Base URL for Coinbase Cloud API
    CLOUD_API_URL = "https://api.cloud.coinbase.com"
    
    def __init__(self, api_key_name: str, private_key: str, sandbox: bool = False):
        """
        Initialize Coinbase Cloud API client
        
        Args:
            api_key_name: Coinbase Cloud API key name
            private_key: EC private key as a string
            sandbox: Whether to use sandbox environment
        """
        self.api_key_name = api_key_name
        self.private_key = private_key
        self.sandbox = sandbox
        
        # Log initialization
        logger.info(f"Initialized FixedCoinbaseClient with API key: {api_key_name}")
        logger.info(f"Using sandbox: {sandbox}")
    
    def create_jwt_token(self) -> Optional[str]:
        """
        Create a JWT token for authenticating with Coinbase Cloud API
        
        Returns:
            JWT token string or None if there was an error
        """
        try:
            # Parse the private key
            private_key_bytes = self.private_key.encode()
            private_key_obj = serialization.load_pem_private_key(
                private_key_bytes,
                password=None
            )
            
            # Define token payload
            now = int(time.time())
            payload = {
                "sub": self.api_key_name,
                "iss": "coinbase-cloud",
                "nbf": now - 60,  # Not before (1 minute ago)
                "exp": now + 600,  # Expiration (10 minutes from now)
                "aud": ["coinbase-cloud"]
            }
            
            # Create the JWT token
            token = jwt.encode(
                payload=payload,
                key=private_key_obj,
                algorithm="ES256"
            )
            
            return token
        
        except Exception as e:
            logger.error(f"Error creating JWT token: {str(e)}")
            return None
    
    def make_cloud_api_request(self, method: str, endpoint: str, body: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Make an authenticated request to the Coinbase Cloud API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/v1/brokerage/products")
            body: Request body (optional)
            
        Returns:
            Tuple of (success, response)
        """
        try:
            # Create JWT token
            token = self.create_jwt_token()
            if not token:
                return False, {"error": "Failed to create JWT token"}
            
            # Set up headers
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Build URL
            url = f"{self.CLOUD_API_URL}{endpoint}"
            
            # Log the request details
            logger.info(f"Making request to: {url}")
            logger.info(f"Method: {method}")
            logger.info(f"Headers: Authorization: Bearer [TOKEN], Content-Type: application/json")
            
            # Make request
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=body
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
    
    def get_available_products(self) -> Tuple[bool, List[Dict]]:
        """
        Get available trading products from Coinbase
        
        Returns:
            Tuple of (success, products list)
        """
        return self.make_cloud_api_request("GET", "/v1/brokerage/products")
    
    def get_ticker(self, product_id: str) -> Tuple[bool, Dict]:
        """
        Get ticker data for a specific product
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
            
        Returns:
            Tuple of (success, ticker data)
        """
        return self.make_cloud_api_request("GET", f"/v1/brokerage/products/{product_id}")
    
    def get_product_stats(self, product_id: str) -> Tuple[bool, Dict]:
        """
        Get 24h stats for a specific product
        
        Args:
            product_id: Product ID (e.g., "BTC-USD")
            
        Returns:
            Tuple of (success, stats data)
        """
        return self.make_cloud_api_request("GET", f"/v1/brokerage/products/{product_id}/stats")

def run_test():
    """Run the fixed Coinbase API test"""
    logger.info("Starting Fixed Coinbase Cloud API Test")
    
    # BenbotReal credentials
    api_key_name = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
    private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
    
    # Create client
    client = FixedCoinbaseClient(api_key_name=api_key_name, private_key=private_key, sandbox=False)
    
    # Test JWT token creation
    logger.info("Testing JWT token creation")
    jwt_token = client.create_jwt_token()
    logger.info(f"JWT token created: {jwt_token is not None}")
    if jwt_token:
        logger.info(f"Token: {jwt_token[:20]}...")
    
    # Test getting available products
    logger.info("Testing available products")
    success, products = client.get_available_products()
    if success:
        logger.info(f"Got {len(products)} available products")
        # Display first 5 products
        for p in products[:5]:
            logger.info(f"Product: {p.get('id', 'unknown')}")
    else:
        logger.error(f"Failed to get products: {products}")
    
    # Test getting ticker for BTC-USD
    logger.info("Testing ticker for BTC-USD")
    success, ticker = client.get_ticker("BTC-USD")
    if success:
        logger.info(f"BTC-USD ticker: {json.dumps(ticker, indent=2)}")
    else:
        logger.error(f"Failed to get ticker: {ticker}")
    
    # Test getting product stats for BTC-USD
    logger.info("Testing product stats for BTC-USD")
    success, stats = client.get_product_stats("BTC-USD")
    if success:
        logger.info(f"BTC-USD stats: {json.dumps(stats, indent=2)}")
    else:
        logger.error(f"Failed to get stats: {stats}")
    
    logger.info("Fixed Coinbase Cloud API test completed")

if __name__ == "__main__":
    run_test()
