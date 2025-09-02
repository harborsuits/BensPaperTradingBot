#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Advanced API Verification Script

This script tests both the Coinbase Advanced (Pro) API to verify
that the provided credentials work correctly.
"""

import hmac
import hashlib
import time
import base64
import requests
import json
from datetime import datetime
import sys

# API credentials
API_KEY = "adb53c0e-35a0-4171-b237-a19fec741363"
API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
API_PASSPHRASE = ""  # Optional, only needed for some Coinbase Advanced endpoints

# API endpoints
ADVANCED_API_URL = "https://api.exchange.coinbase.com"

def generate_advanced_signature(timestamp, method, request_path, body=''):
    """
    Generate signature for Coinbase Advanced API
    """
    message = timestamp + method + request_path + (body or '')
    
    # Convert secret to bytes and create signature
    secret = base64.b64decode(API_SECRET)
    signature = hmac.new(secret, message.encode('utf-8'), digestmod=hashlib.sha256)
    return base64.b64encode(signature.digest()).decode()

def advanced_api_request(method, endpoint, body=None):
    """
    Make a request to the Coinbase Advanced API
    """
    timestamp = str(int(time.time()))
    request_path = endpoint
    
    # Create request headers with auth
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': generate_advanced_signature(timestamp, method, request_path, json.dumps(body) if body else ''),
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': API_PASSPHRASE,
        'Content-Type': 'application/json'
    }
    
    url = ADVANCED_API_URL + endpoint
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=body)
        
        return response
    except Exception as e:
        print(f"Error making request: {e}")
        return None

def test_advanced_api():
    """
    Test Coinbase Advanced API connectivity
    """
    print("\n--- Testing Coinbase Advanced API ---")
    
    # Test public endpoint (doesn't require authentication)
    try:
        response = requests.get(f"{ADVANCED_API_URL}/products")
        if response.status_code == 200:
            products = response.json()
            print(f"✅ Public Advanced API works - Found {len(products)} trading pairs")
        else:
            print(f"❌ Public Advanced API failed: {response.status_code} {response.text[:100]}")
            return False
    except Exception as e:
        print(f"❌ Public Advanced API error: {e}")
        return False
    
    # Test authenticated endpoint - get accounts
    try:
        response = advanced_api_request('GET', '/accounts')
        
        if response and response.status_code == 200:
            accounts = response.json()
            print(f"✅ Authenticated Advanced API works - Found {len(accounts)} accounts")
            
            # Display non-zero balances
            non_zero = [a for a in accounts if float(a.get('balance', 0)) > 0]
            if non_zero:
                print("\nNon-zero balances:")
                for account in non_zero:
                    print(f"  {account['currency']}: {account['balance']}")
            
            return True
        else:
            error_msg = response.text if response else "No response"
            print(f"❌ Authenticated Advanced API failed: {response.status_code if response else 'N/A'} {error_msg[:100]}")
            return False
    except Exception as e:
        print(f"❌ Authenticated Advanced API error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 COINBASE API VERIFICATION 🔍")
    print("-------------------------------")
    
    advanced_result = test_advanced_api()
    
    print("\n-------------------------------")
    print(f"Coinbase Advanced API: {'✅ WORKING' if advanced_result else '❌ FAILED'}")
    
    if advanced_result:
        print("\n✅ SUCCESS: Your Coinbase API credentials are valid!")
        print("You can safely add these to your trading bot configuration.")
        sys.exit(0)
    else:
        print("\n❌ ERROR: API verification failed. Please check your credentials.")
        sys.exit(1)
