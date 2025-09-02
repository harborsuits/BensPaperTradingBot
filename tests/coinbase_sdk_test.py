#!/usr/bin/env python3
"""
Coinbase API Test using the official SDK - tests both APIs

This script tests both authentication methods:
1. Coinbase Advanced Trading API
2. Coinbase Cloud API using SDK
"""

import os
import json
import time
import base64
import hmac
import hashlib
from datetime import datetime
import requests

print("Installing Coinbase SDK if needed...")
try:
    import coinbase
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "coinbase-python"])
    import coinbase

# CREDENTIALS - Advanced Trading API
ADV_API_KEY = "acc.adb53c0e-35a0-4171-b237-a19fec741363"
ADV_API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
ADV_API_PASSPHRASE = ""  # You may need to set this

# Advanced API Test
def test_advanced_api():
    print("\n--- TESTING COINBASE ADVANCED TRADING API ---")
    
    # Try multiple base URLs and endpoints
    base_urls = [
        "https://api.coinbase.com",
        "https://api.exchange.coinbase.com",
        "https://api.pro.coinbase.com"
    ]
    
    endpoints = [
        "/v2/accounts", 
        "/api/v3/brokerage/accounts",
        "/accounts"
    ]
    
    # Try both with and without passphrase
    for base_url in base_urls:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            print(f"\nTrying endpoint: {url}")
            
            # Create timestamp for the request
            timestamp = str(int(time.time()))
            method = "GET"
            request_path = endpoint
            body = ""
            
            # Create the message to sign
            message = f"{timestamp}{method}{request_path}{body}"
            
            try:
                # Convert the API secret from base64
                secret = base64.b64decode(ADV_API_SECRET)
                
                # Create the HMAC signature using SHA-256
                signature = hmac.new(secret, message.encode(), hashlib.sha256)
                signature_b64 = base64.b64encode(signature.digest()).decode()
                
                # First try without passphrase
                headers = {
                    "CB-ACCESS-KEY": ADV_API_KEY,
                    "CB-ACCESS-SIGN": signature_b64,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "Content-Type": "application/json"
                }
                
                response = requests.get(url, headers=headers)
                print(f"Status code (no passphrase): {response.status_code}")
                if response.status_code == 200:
                    print("✅ SUCCESS without passphrase!")
                    print(f"Response: {response.text[:200]}...")
                    return
                
                # Try with dummy passphrase (in case it needs one)
                if not ADV_API_PASSPHRASE:
                    for passphrase in ["", "coinbase", "default"]:
                        print(f"Trying with passphrase: '{passphrase}'")
                        headers["CB-ACCESS-PASSPHRASE"] = passphrase
                        
                        response = requests.get(url, headers=headers)
                        print(f"Status code (with passphrase '{passphrase}'): {response.status_code}")
                        
                        if response.status_code == 200:
                            print(f"✅ SUCCESS with passphrase: '{passphrase}'!")
                            print(f"Response: {response.text[:200]}...")
                            return
                else:
                    # Try with the provided passphrase
                    headers["CB-ACCESS-PASSPHRASE"] = ADV_API_PASSPHRASE
                    response = requests.get(url, headers=headers)
                    print(f"Status code (with provided passphrase): {response.status_code}")
                    if response.status_code == 200:
                        print("✅ SUCCESS with provided passphrase!")
                        print(f"Response: {response.text[:200]}...")
                        return
                        
            except Exception as e:
                print(f"Error: {str(e)}")

# SDK Test
def test_sdk():
    print("\n--- TESTING COINBASE SDK ---")
    
    try:
        # Try the SDK with Advanced API credentials
        print("\nTrying SDK with Advanced API credentials...")
        client = coinbase.Client(ADV_API_KEY, ADV_API_SECRET)
        
        try:
            user = client.get_current_user()
            print("✅ SUCCESS with SDK!")
            print(f"User ID: {user['data']['id']}")
            print(f"Name: {user['data'].get('name', 'Not provided')}")
            print(f"Email: {user['data'].get('email', 'Not provided')}")
            
            # Get accounts
            print("\nFetching accounts...")
            accounts = client.get_accounts()
            print(f"Found {len(accounts['data'])} accounts")
            
            # Display some account information
            for account in accounts['data'][:5]:  # Show first 5 accounts
                print(f"- {account['name']} ({account['balance']['currency']}): {account['balance']['amount']}")
                
            return True
            
        except Exception as e:
            print(f"SDK error with Advanced API: {str(e)}")
            
    except Exception as e:
        print(f"SDK initialization error: {str(e)}")
        
    return False

# Test public endpoints
def test_public_api():
    print("\n--- TESTING PUBLIC API ---")
    
    try:
        client = coinbase.Client()
        
        # Get exchange rates
        print("\nFetching exchange rates...")
        exchange_rates = client.get_exchange_rates(currency="USD")
        print(f"Exchange rates base: {exchange_rates['data']['currency']}")
        print(f"BTC rate: {exchange_rates['data']['rates'].get('BTC', 'Not available')}")
        print(f"ETH rate: {exchange_rates['data']['rates'].get('ETH', 'Not available')}")
        
        # Get buy/sell prices for BTC
        print("\nFetching BTC prices...")
        btc_buy_price = client.get_buy_price(currency_pair="BTC-USD")
        btc_sell_price = client.get_sell_price(currency_pair="BTC-USD")
        print(f"BTC Buy Price: {btc_buy_price['data']['amount']} {btc_buy_price['data']['currency']}")
        print(f"BTC Sell Price: {btc_sell_price['data']['amount']} {btc_sell_price['data']['currency']}")
        
        return True
        
    except Exception as e:
        print(f"Public API error: {str(e)}")
        return False

if __name__ == "__main__":
    print("\nCOINBASE API SDK TEST SCRIPT")
    print("==========================")
    print(f"Date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"SDK Version: {coinbase.__version__}")
    
    # Test public API first (should always work)
    public_success = test_public_api()
    if not public_success:
        print("\n⚠️ Warning: Even public API failed. Network or SDK issue.")
    
    # Test Advanced API
    test_advanced_api()
    
    # Test SDK with authentication
    sdk_success = test_sdk()
    
    print("\n--- SUMMARY ---")
    if public_success:
        print("✅ Public API access works")
    else:
        print("❌ Public API access failed - check your network or SDK")
        
    if sdk_success:
        print("✅ Authenticated SDK access works")
    else:
        print("❌ Authenticated SDK access failed - check your API credentials")
        
    print("\nRecommendations:")
    print("1. Verify API key permissions in your Coinbase account")
    print("2. Check for any IP restrictions on your API key")
    print("3. Contact Coinbase support if issues persist")
    print("4. For trading integration, use the successful authentication method")
