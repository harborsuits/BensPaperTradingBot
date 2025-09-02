#!/usr/bin/env python3
"""
Coinbase API Test Script - Tests both authentication methods:
1. Coinbase Advanced Trading API (API Key + Secret)
2. Coinbase Cloud API (API Key + EC Private Key)
"""

import os
import json
import time
import base64
import hmac
import hashlib
from datetime import datetime
import requests
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# CONFIGURATION
# Uncomment the method you want to test
TEST_ADVANCED_API = True  # Coinbase Advanced Trading API
TEST_CLOUD_API = True     # Coinbase Cloud API

# CREDENTIALS - Advanced Trading API
ADVANCED_API_KEY = "acc.adb53c0e-35a0-4171-b237-a19fec741363"
ADVANCED_API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
ADVANCED_API_PASSPHRASE = ""  # Leave empty if not required

# CREDENTIALS - Cloud API
CLOUD_API_KEY_NAME = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
CLOUD_API_PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""

# API ENDPOINTS
# Advanced Trading API endpoints
ADV_BASE_URL = "https://api.coinbase.com"
ADV_ACCOUNTS_ENDPOINT = "/api/v3/brokerage/accounts"

# Cloud API endpoints
CLOUD_BASE_URLS = [
    "https://api.exchange.coinbase.com",
    "https://api.coinbase.com/api/v3/brokerage", 
    "https://api.cloud.coinbase.com/v1",
    "https://api.cloud.coinbase.com/api/v1"
]

def test_advanced_api():
    print("\n--- TESTING COINBASE ADVANCED TRADING API ---")
    
    timestamp = str(int(time.time()))
    request_path = ADV_ACCOUNTS_ENDPOINT
    method = "GET"
    body = ""
    
    # Create the message to sign
    message = f"{timestamp}{method}{request_path}{body}"
    
    try:
        # Convert the API secret from base64
        secret = base64.b64decode(ADVANCED_API_SECRET)
        
        # Create the HMAC signature using SHA-256
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        # Set the headers
        headers = {
            "CB-ACCESS-KEY": ADVANCED_API_KEY,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        if ADVANCED_API_PASSPHRASE:
            headers["CB-ACCESS-PASSPHRASE"] = ADVANCED_API_PASSPHRASE
        
        url = f"{ADV_BASE_URL}{request_path}"
        print(f"\nRequest URL: {url}")
        print(f"Request headers: {json.dumps(headers, indent=2)}")
        
        # Make the request
        response = requests.get(url, headers=headers)
        
        print(f"\nStatus code: {response.status_code}")
        print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ Coinbase Advanced Trading API authentication successful!")
            print("\nResponse (limited to 500 chars):")
            print(response.text[:500] + ("..." if len(response.text) > 500 else ""))
        else:
            print("\n❌ Coinbase Advanced Trading API authentication failed")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error testing Advanced Trading API: {str(e)}")

def test_cloud_api():
    print("\n--- TESTING COINBASE CLOUD API ---")
    
    try:
        # Load the private key
        private_key = serialization.load_pem_private_key(
            CLOUD_API_PRIVATE_KEY.encode(),
            password=None,
            backend=default_backend()
        )
        
        # Create JWT token
        now = int(time.time())
        payload = {
            "sub": CLOUD_API_KEY_NAME,
            "iss": "coinbase-cloud",
            "nbf": now,
            "exp": now + 60,  # Token expires in 60 seconds
            "aud": ["brokerage"]
        }
        
        # Generate JWT token
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256"
        )
        
        print(f"\nJWT Payload: {json.dumps(payload, indent=2)}")
        print(f"JWT Token: {token[:20]}...{token[-20:]}")
        
        # Try different endpoints
        for base_url in CLOUD_BASE_URLS:
            print(f"\nTrying base URL: {base_url}")
            
            # Endpoint paths to try
            endpoints = [
                "/accounts",  # Basic accounts endpoint
                "/portfolios",  # Portfolios endpoint
                "",  # Root endpoint
                "/public"  # Public API endpoint
            ]
            
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                print(f"\n  Trying endpoint: {url}")
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    print(f"  Status code: {response.status_code}")
                    
                    if response.status_code < 400:
                        print(f"\n✅ Successful response from {url}")
                        print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
                        print("Response (limited to 500 chars):")
                        print(response.text[:500] + ("..." if len(response.text) > 500 else ""))
                        return  # Exit on first successful response
                    elif response.status_code == 401:
                        print(f"  Authentication failed: {response.text}")
                    else:
                        print(f"  Error: {response.text}")
                        
                except Exception as e:
                    print(f"  Error: {str(e)}")
    
    except Exception as e:
        print(f"\n❌ Error testing Cloud API: {str(e)}")

def test_public_endpoints():
    """Test public endpoints that don't require authentication"""
    print("\n--- TESTING PUBLIC ENDPOINTS ---")
    
    endpoints = [
        ("https://api.exchange.coinbase.com/products", "Coinbase Exchange Products"),
        ("https://api.coinbase.com/v2/currencies", "Coinbase Currencies"),
        ("https://api.coinbase.com/v2/exchange-rates", "Coinbase Exchange Rates"),
        ("https://api.coinbase.com/v2/prices/BTC-USD/spot", "Coinbase BTC-USD Spot Price")
    ]
    
    for url, description in endpoints:
        print(f"\nTesting {description}: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Success!")
                print("Response (limited to 200 chars):")
                print(response.text[:200] + ("..." if len(response.text) > 200 else ""))
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\nCOINBASE API TEST SCRIPT")
    print("======================")
    print(f"Date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test public endpoints first
    test_public_endpoints()
    
    # Test authentication methods
    if TEST_ADVANCED_API:
        test_advanced_api()
        
    if TEST_CLOUD_API:
        test_cloud_api()
        
    print("\n--- TEST SUMMARY ---")
    print("1. Check if public endpoints work (basic connectivity)")
    print("2. Check if either authentication method worked")
    print("3. If both failed, verify:")
    print("   - API key permissions and restrictions")
    print("   - IP restrictions on the API key")
    print("   - Account status and verification level")
    print("   - API rate limits")
