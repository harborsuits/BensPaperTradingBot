#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Advanced API Verification Script - With Alternative Methods

This script tests the Coinbase Advanced (Pro) API with multiple authentication methods
to determine which one works with your credentials.
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

# Try with an empty passphrase first
API_PASSPHRASE = ""  

# API endpoints
ADVANCED_API_URL = "https://api.exchange.coinbase.com"
API_VERSION = "2023-08-18"  # Coinbase API version header

def generate_signature(timestamp, method, request_path, body=''):
    """
    Generate signature for Coinbase Advanced API
    """
    message = timestamp + method + request_path + (body or '')
    
    # Convert secret to bytes and create signature
    try:
        # Method 1: Standard Base64 decoding
        secret = base64.b64decode(API_SECRET)
        signature = hmac.new(secret, message.encode('utf-8'), digestmod=hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()
    except Exception as e:
        print(f"Error generating signature: {e}")
        return None

def advanced_api_request(method, endpoint, body=None, pass_phrase=None, debug=False):
    """
    Make a request to the Coinbase Advanced API with configurable auth
    """
    timestamp = str(int(time.time()))
    request_path = endpoint
    
    json_body = json.dumps(body) if body else ''
    signature = generate_signature(timestamp, method, request_path, json_body)
    
    if not signature:
        return None
    
    # Create request headers with auth
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json',
    }
    
    # Add passphrase if provided
    if pass_phrase is not None:
        headers['CB-ACCESS-PASSPHRASE'] = pass_phrase
        
    # Add API version header
    headers['CB-VERSION'] = API_VERSION
    
    if debug:
        print(f"\nRequest Details:")
        print(f"URL: {ADVANCED_API_URL + endpoint}")
        print(f"Method: {method}")
        print(f"Headers: {headers}")
        print(f"Body: {json_body}")
    
    url = ADVANCED_API_URL + endpoint
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=body)
        
        if debug and response:
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Text: {response.text[:200]}...")
            
        return response
    except Exception as e:
        print(f"Error making request: {e}")
        return None

def test_with_passphrase(passphrase, debug=False):
    """Test API with a specific passphrase"""
    print(f"\nTesting with passphrase: '{passphrase if passphrase else 'empty'}'")
    
    response = advanced_api_request('GET', '/accounts', pass_phrase=passphrase, debug=debug)
    
    if response:
        if response.status_code == 200:
            accounts = response.json()
            print(f"✅ Authentication success! Found {len(accounts)} accounts.")
            
            # Display non-zero balances
            non_zero = [a for a in accounts if float(a.get('balance', 0)) > 0]
            if non_zero:
                print("Non-zero balances:")
                for account in non_zero:
                    print(f"  {account['currency']}: {account['balance']}")
            
            return True
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Error message: {response.text[:200]}")
    else:
        print("❌ Request failed to complete")
    
    return False

def ask_for_passphrase():
    """Ask the user to input a passphrase"""
    print("\nThe API key may require a passphrase.")
    print("If you have one, please enter it below (press Enter to skip):")
    return input("> ").strip()

def main():
    """Run verification with multiple options"""
    print("🔍 COINBASE ADVANCED API VERIFICATION 🔍")
    print("---------------------------------------")
    
    # Test public endpoint (doesn't require authentication)
    try:
        response = requests.get(f"{ADVANCED_API_URL}/products")
        if response.status_code == 200:
            products = response.json()
            print(f"✅ Public API works - Found {len(products)} trading pairs")
        else:
            print(f"❌ Public API failed: {response.status_code}")
            print(f"Error message: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Public API error: {e}")
        return False
    
    # First try with empty passphrase
    if test_with_passphrase(""):
        return True
        
    # Try with debug mode to get more info
    print("\nTrying with debug mode for more information:")
    test_with_passphrase("", debug=True)
    
    # Ask for passphrase input if previous attempt failed
    user_passphrase = ask_for_passphrase()
    if user_passphrase and test_with_passphrase(user_passphrase):
        return True
    
    print("\n❌ Authentication could not be completed with available credentials.")
    print("You might need to:")
    print("1. Check if the API key requires a passphrase")
    print("2. Verify the API key has proper permissions")
    print("3. Confirm the API secret is correctly formatted")
    print("4. Check if the API key is still active")
    
    return False

if __name__ == "__main__":
    if main():
        print("\n✅ SUCCESS: Coinbase API authentication works!")
        # Remove clear-text credentials from the file
        sys.exit(0)
    else:
        print("\n❌ ERROR: API verification failed.")
        sys.exit(1)
