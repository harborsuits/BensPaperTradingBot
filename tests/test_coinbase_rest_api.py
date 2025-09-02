#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase REST API Authentication Test

This script tests the Coinbase REST API v2 authentication which uses
a different approach than the Advanced API.
"""

import time
import hmac
import hashlib
import requests
import json
import base64
import os
from urllib.parse import urljoin

# API credentials - stored securely in variables
API_KEY = "adb53c0e-35a0-4171-b237-a19fec741363"
API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="

# REST API base URL
BASE_URL = "https://api.coinbase.com/v2/"

def generate_auth_headers(method, path, body=None):
    """Generate OAuth headers for Coinbase REST API"""
    timestamp = str(int(time.time()))
    message = timestamp + method + path + (json.dumps(body) if body else '')
    
    # Create signature
    try:
        key = base64.b64decode(API_SECRET)
        message = message.encode('utf-8')
        signature = hmac.new(key, message, hashlib.sha256).hexdigest()
        
        # Return headers
        return {
            'CB-ACCESS-KEY': API_KEY,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-VERSION': '2021-10-05',  # API version
            'Content-Type': 'application/json'
        }
    except Exception as e:
        print(f"Error generating signature: {e}")
        return None

def make_request(method, endpoint, body=None):
    """Make an authenticated request to the Coinbase REST API"""
    url = urljoin(BASE_URL, endpoint)
    headers = generate_auth_headers(method, '/' + endpoint, body)
    
    if not headers:
        return False, {"error": "Failed to generate authentication headers"}
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=body)
        else:
            return False, {"error": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "status_code": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return False, {"error": str(e)}

def test_public_api():
    """Test public endpoint (no authentication required)"""
    print("\nTesting public endpoint...")
    try:
        response = requests.get(urljoin(BASE_URL, "currencies"))
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Public API works - Found {len(data.get('data', []))} currencies")
            return True
        else:
            print(f"❌ Public API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Public API error: {e}")
        return False

def test_authenticated_api():
    """Test authenticated endpoints"""
    print("\nTesting authenticated endpoints...")
    
    # Test user account endpoint
    print("\nTesting user account endpoint...")
    success, response = make_request('GET', 'user')
    
    if success:
        print(f"✅ Authentication successful! User: {response.get('data', {}).get('name', 'Unknown')}")
        return True
    else:
        print(f"❌ Authentication failed: {response.get('status_code', 'Unknown')}")
        print(f"Error message: {response.get('error', '')[:200]}")
        
        # Get accounts as fallback
        print("\nTrying accounts endpoint as fallback...")
        success, accounts = make_request('GET', 'accounts')
        
        if success:
            print(f"✅ Accounts endpoint works - Found {len(accounts.get('data', []))} accounts")
            
            # Display accounts with non-zero balance
            non_zero = [a for a in accounts.get('data', []) if float(a.get('balance', {}).get('amount', 0)) > 0]
            if non_zero:
                print("\nAccounts with balance:")
                for account in non_zero:
                    print(f"  {account['balance']['currency']}: {account['balance']['amount']}")
            
            return True
        else:
            print(f"❌ Accounts endpoint failed: {accounts.get('status_code', 'Unknown')}")
            print(f"Error message: {accounts.get('error', '')[:200]}")
            
            # Provide troubleshooting guidance
            print("\nTroubleshooting steps:")
            print("1. Verify API key permissions in Coinbase settings")
            print("2. Check if API key is for Coinbase (not Coinbase Pro/Advanced)")
            print("3. Ensure API key is active and not expired")
            
            return False

def main():
    """Run Coinbase REST API tests"""
    print("🔍 COINBASE REST API TEST 🔍")
    print("----------------------------")
    
    # Test public endpoint
    public_success = test_public_api()
    
    # Test authenticated endpoints
    auth_success = test_authenticated_api()
    
    # Print summary
    print("\n----------------------------")
    if public_success and auth_success:
        print("✅ SUCCESS: Coinbase API is working correctly!")
        print("Your trading bot can now use the Coinbase broker.")
    else:
        if public_success:
            print("⚠️ PARTIAL SUCCESS: Public API works but authentication failed.")
        else:
            print("❌ FAILURE: Could not connect to Coinbase API.")
    
    return public_success and auth_success

if __name__ == "__main__":
    main()
