#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Cloud API Verification Script

This script tests connectivity with the Coinbase Cloud API using
JWT authentication with the EC private key.
"""

import os
import sys
import time
import json
import datetime
import base64
import hashlib
import urllib.request
import urllib.error
from urllib.parse import urljoin

# API credentials
API_KEY_NAME = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""

# API endpoints
BASE_URL = "https://api.coinbase.com/api/v3/"

try:
    # Try to import cryptography for EC private key operations
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    import jwt  # PyJWT package
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("\n⚠️ Required packages not found. Installing...")
    print("This script requires 'cryptography' and 'pyjwt' packages.")
    print("Please install them with:")
    print("pip install cryptography pyjwt\n")
    sys.exit(1)

def create_jwt_token():
    """Create a JWT token signed with the EC private key"""
    try:
        # Load the private key
        private_key = load_pem_private_key(PRIVATE_KEY.encode(), password=None)
        
        # Create the JWT payload
        now = int(time.time())
        payload = {
            'sub': API_KEY_NAME,
            'iss': 'coinbase-cloud',
            'nbf': now,
            'exp': now + 60,  # Token expires in 60 seconds
            'aud': ['api.coinbase.com']
        }
        
        # Create the JWT token
        token = jwt.encode(
            payload, 
            private_key, 
            algorithm='ES256'
        )
        
        return token
    except Exception as e:
        print(f"❌ Error creating JWT token: {e}")
        return None

def make_request(method, endpoint, body=None):
    """Make a request to the Coinbase Cloud API"""
    url = urljoin(BASE_URL, endpoint)
    
    # Get JWT token
    token = create_jwt_token()
    if not token:
        return False, {"error": "Failed to create authentication token"}
    
    # Create headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    # Create request
    req = urllib.request.Request(url)
    
    # Add headers
    for key, value in headers.items():
        req.add_header(key, value)
    
    # Set method
    req.method = method
    
    # Add body if needed
    if body and method != 'GET':
        req.data = json.dumps(body).encode('utf-8')
    
    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode('utf-8')
            return True, json.loads(response_data)
    except urllib.error.HTTPError as e:
        error_message = e.read().decode('utf-8')
        return False, {"status_code": e.code, "error": error_message}
    except Exception as e:
        return False, {"error": str(e)}

def test_public_api():
    """Test public Coinbase endpoints (no auth required)"""
    print("\nTesting public Coinbase API...")
    try:
        req = urllib.request.Request("https://api.coinbase.com/v2/currencies")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            print(f"✅ Public API works - Found {len(data.get('data', []))} currencies")
            return True
    except Exception as e:
        print(f"❌ Public API failed: {e}")
        return False

def test_authenticated_api():
    """Test authenticated Coinbase Cloud API endpoints"""
    print("\nTesting authenticated Coinbase Cloud API...")
    
    # Test with a basic endpoint
    success, response = make_request('GET', 'brokerage/products')
    
    if success:
        print(f"✅ Authentication successful!")
        print(f"Found {len(response.get('products', []))} products")
        
        # Display some products
        if response.get('products'):
            print("\nSample products:")
            for product in response.get('products')[:5]:  # Show first 5
                print(f"  {product.get('product_id', 'Unknown')}")
        
        return True
    else:
        print(f"❌ Authentication failed: {response.get('status_code', 'Unknown')}")
        error_msg = response.get('error', '')
        if isinstance(error_msg, str):
            print(f"Error message: {error_msg[:200]}")
        else:
            print(f"Error: {json.dumps(error_msg)[:200]}")
        
        print("\nTroubleshooting steps:")
        print("1. Verify the API key name is correct")
        print("2. Check that the private key is valid and complete")
        print("3. Ensure the API key has the necessary permissions")
        print("4. Check if the Coinbase Cloud API is accessible from your location")
        
        return False

def main():
    """Run the Coinbase Cloud API test"""
    print("🔍 COINBASE CLOUD API TEST 🔍")
    print("-----------------------------")
    
    if not CRYPTO_AVAILABLE:
        print("❌ Required packages not available")
        return False
    
    # Test functionality
    print("Testing JWT token creation...")
    token = create_jwt_token()
    if token:
        print(f"✅ JWT token created successfully")
    else:
        print("❌ Failed to create JWT token")
        return False
    
    # Test public API
    public_success = test_public_api()
    
    # Test authenticated API
    auth_success = test_authenticated_api()
    
    # Print summary
    print("\n-----------------------------")
    if public_success and auth_success:
        print("✅ SUCCESS: Coinbase Cloud API is working correctly!")
        print("Your trading bot can now use the Coinbase broker.")
    else:
        if public_success:
            print("⚠️ PARTIAL SUCCESS: Public API works but Cloud API authentication failed.")
        else:
            print("❌ FAILURE: Could not connect to Coinbase API.")
    
    return public_success and auth_success

if __name__ == "__main__":
    main()
