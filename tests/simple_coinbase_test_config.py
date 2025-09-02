#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Coinbase API Configuration Test

Tests if the Coinbase API credentials stored in the configuration
can successfully connect to the Coinbase API without requiring
any external packages.
"""

import os
import sys
import base64
import hmac
import hashlib
import time
import json
import urllib.request
import urllib.error

# Coinbase API credentials - hardcoded for this test only
# These should match what's in the trading_config.yaml file
API_KEY = "adb53c0e-35a0-4171-b237-a19fec741363"
API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
API_PASSPHRASE = os.environ.get("COINBASE_PASSPHRASE", "")  # Get from environment if set
SANDBOX_MODE = False

# API base URL
BASE_URL = "https://api.exchange.coinbase.com" if not SANDBOX_MODE else "https://api-public.sandbox.exchange.coinbase.com"

def make_request(method, path, body=None):
    """Make a request to the Coinbase API"""
    url = BASE_URL + path
    
    # Create the signature
    timestamp = str(int(time.time()))
    message = timestamp + method + path + (json.dumps(body) if body else '')
    
    try:
        # Create signature
        secret = base64.b64decode(API_SECRET)
        signature = hmac.new(secret, message.encode('utf-8'), digestmod=hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        # Create headers
        headers = {
            'CB-ACCESS-KEY': API_KEY,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json',
        }
        
        # Add passphrase if available
        if API_PASSPHRASE:
            headers['CB-ACCESS-PASSPHRASE'] = API_PASSPHRASE
        
        # Create request
        req = urllib.request.Request(url, headers=headers)
        
        # Handle request body if provided
        if body and method != 'GET':
            req.data = json.dumps(body).encode('utf-8')
        
        # Set request method
        req.method = method
        
        # Send request
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode('utf-8')
            return True, json.loads(response_data)
            
    except urllib.error.HTTPError as e:
        error_message = e.read().decode('utf-8')
        return False, {"status_code": e.code, "error": error_message}
    except Exception as e:
        return False, {"error": str(e)}

def test_public_api():
    """Test public API endpoint (no authentication)"""
    print("\nTesting public API endpoint...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/products")
        with urllib.request.urlopen(req) as response:
            products = json.loads(response.read().decode('utf-8'))
            print(f"✅ Public API works - Found {len(products)} trading pairs")
            return True
    except Exception as e:
        print(f"❌ Public API failed: {str(e)}")
        return False

def test_authenticated_api():
    """Test authenticated API endpoint"""
    print("\nTesting authenticated API endpoint...")
    
    success, response = make_request('GET', '/accounts')
    
    if success:
        print(f"✅ Authentication successful! Found {len(response)} accounts")
        
        # Display non-zero balances
        non_zero = [a for a in response if float(a.get('balance', 0)) > 0]
        if non_zero:
            print("\nNon-zero balances:")
            for account in non_zero:
                print(f"  {account['currency']}: {account['balance']}")
        
        return True
    else:
        print(f"❌ Authentication failed: {response.get('status_code', 'Unknown error')}")
        print(f"Error message: {response.get('error', '')[:200]}")
        
        if response.get('status_code') == 401 and 'passphrase' in str(response.get('error', '')).lower():
            print("\nThe API key requires a passphrase.")
            print("Please set the COINBASE_PASSPHRASE environment variable:")
            print("export COINBASE_PASSPHRASE='your_passphrase_here'")
        
        return False

def main():
    """Run test for Coinbase API connection"""
    print("🔍 COINBASE API TEST 🔍")
    print("------------------------")
    
    print(f"Testing with configuration:")
    print(f"  API Key: {API_KEY[:8]}...")
    print(f"  API Secret: {API_SECRET[:8]}...")
    print(f"  Passphrase: {'Set' if API_PASSPHRASE else 'Not set'}")
    print(f"  Sandbox Mode: {'Enabled' if SANDBOX_MODE else 'Disabled'}")
    
    # Test public API
    public_success = test_public_api()
    
    # Test authenticated API if public API works
    if public_success:
        auth_success = test_authenticated_api()
    else:
        auth_success = False
    
    # Print summary
    print("\n------------------------")
    if public_success and auth_success:
        print("✅ SUCCESS: Coinbase API is configured correctly!")
        print("Your trading bot can now use the Coinbase broker.")
    else:
        print("❌ ERROR: Coinbase API connection failed.")
        if not public_success:
            print("  - Could not connect to Coinbase API. Check your internet connection.")
        elif not auth_success:
            print("  - Authentication failed. Check your API credentials.")
            print("  - If using Coinbase Advanced API, you may need to set a passphrase.")
    
    return public_success and auth_success

if __name__ == "__main__":
    main()
