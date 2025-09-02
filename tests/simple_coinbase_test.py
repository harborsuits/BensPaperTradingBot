#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Coinbase API Test Script

This script tests the Coinbase API connection directly without 
requiring the full trading system dependencies.
"""

import os
import time
import hmac
import hashlib
import requests
import json
from datetime import datetime
import base64

# Coinbase API credentials
API_KEY = "adb53c0e-35a0-4171-b237-a19fec741363"
API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="

# Coinbase API endpoints
BASE_URL = "https://api.coinbase.com"
ADVANCED_URL = "https://api.exchange.coinbase.com"

def get_signature(api_secret, message):
    """Generate the signature for the API request"""
    hmac_key = base64.b64decode(api_secret)
    signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(signature.digest()).decode('utf-8')

def get_timestamp():
    """Get the current timestamp in ISO format"""
    return datetime.utcnow().isoformat() + 'Z'

def make_auth_request(url, method='GET', body=None):
    """Make an authenticated request to the Coinbase API"""
    timestamp = get_timestamp()
    path = url.replace(BASE_URL, '').replace(ADVANCED_URL, '')
    
    if body:
        body_json = json.dumps(body)
        message = timestamp + method + path + body_json
    else:
        message = timestamp + method + path
    
    signature = get_signature(API_SECRET, message)
    
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-VERSION': '2021-10-05',
        'Content-Type': 'application/json'
    }
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=body)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response.json()
    except Exception as e:
        print(f"Error making request: {e}")
        return None

def test_api_connection():
    """Test basic connection to the Coinbase API"""
    print("\n--- Testing Coinbase API Connection ---")
    try:
        # Test public endpoint
        response = requests.get(f"{BASE_URL}/v2/currencies")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Public API connection successful")
            print(f"Available currencies: {len(data['data'])}")
        else:
            print(f"❌ Public API connection failed: {response.status_code}")
            return False
        
        # Test authenticated endpoint - get accounts
        accounts = make_auth_request(f"{BASE_URL}/v2/accounts")
        if accounts and 'data' in accounts:
            print(f"✅ Authenticated API connection successful")
            print(f"Found {len(accounts['data'])} accounts")
            
            # Display account balances
            print("\n--- Account Balances ---")
            for account in accounts['data']:
                if float(account['balance']['amount']) > 0:
                    print(f"{account['balance']['currency']}: {account['balance']['amount']}")
        else:
            print(f"❌ Authenticated API connection failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ API connection test failed with error: {e}")
        return False

def test_market_data():
    """Test retrieving market data from Coinbase"""
    print("\n--- Testing Market Data Retrieval ---")
    try:
        # Get BTC-USD price
        response = requests.get(f"{BASE_URL}/v2/prices/BTC-USD/spot")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ BTC-USD Price: ${data['data']['amount']}")
        else:
            print(f"❌ Failed to get BTC price: {response.status_code}")
            
        # Get ETH-USD price
        response = requests.get(f"{BASE_URL}/v2/prices/ETH-USD/spot")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ETH-USD Price: ${data['data']['amount']}")
        else:
            print(f"❌ Failed to get ETH price: {response.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Market data test failed with error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🔒 COINBASE API READ-ONLY TEST 🔒")
    print("This script will only READ data, no trades will be placed")
    print("--------------------------------------------------")
    
    # Run tests
    connection_ok = test_api_connection()
    if connection_ok:
        market_data_ok = test_market_data()
    else:
        market_data_ok = False
    
    # Summary
    print("\n--------------------------------------------------")
    print("📊 TEST SUMMARY:")
    print(f"API Connection: {'✅ PASSED' if connection_ok else '❌ FAILED'}")
    print(f"Market Data: {'✅ PASSED' if market_data_ok else '❌ FAILED'}")
    print("--------------------------------------------------")
    
    if connection_ok and market_data_ok:
        print("🎉 All tests passed! Your Coinbase API is working correctly.")
        print("You're ready to integrate it with your trading system.")
    else:
        print("❌ Some tests failed. Please check your API credentials and try again.")

if __name__ == "__main__":
    main()
