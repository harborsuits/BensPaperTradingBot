#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase API Configuration Test

This script tests if the Coinbase API credentials in trading_config.yaml
can successfully connect to the Coinbase API.
"""

import os
import sys
import yaml
import hmac
import hashlib
import time
import base64
import requests
import json
from datetime import datetime

# Set path to allow importing from trading_bot modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_coinbase_config():
    """Load Coinbase configuration from trading_config.yaml"""
    config_path = os.path.join('trading_bot', 'config', 'trading_config.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        coinbase_config = config.get('coinbase', {})
        
        # Handle environment variables in config
        if coinbase_config.get('passphrase', '').startswith('${') and coinbase_config.get('passphrase', '').endswith('}'):
            env_var = coinbase_config['passphrase'][2:-1]
            coinbase_config['passphrase'] = os.environ.get(env_var, '')
            
        return coinbase_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def generate_signature(api_secret, timestamp, method, request_path, body=''):
    """
    Generate signature for Coinbase Advanced API
    """
    message = timestamp + method + request_path + (body or '')
    
    try:
        # Convert secret to bytes and create signature
        secret = base64.b64decode(api_secret)
        signature = hmac.new(secret, message.encode('utf-8'), digestmod=hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()
    except Exception as e:
        print(f"Error generating signature: {e}")
        return None

def test_coinbase_connection(config):
    """Test connection to Coinbase API using the provided config"""
    api_key = config.get('api_key')
    api_secret = config.get('api_secret')
    passphrase = config.get('passphrase', '')
    sandbox = config.get('sandbox', False)
    
    if not api_key or not api_secret:
        print("❌ Missing required API credentials")
        return False
    
    # Set base URL based on environment
    if sandbox:
        base_url = "https://api-public.sandbox.exchange.coinbase.com"
    else:
        base_url = "https://api.exchange.coinbase.com"
    
    # First test unauthenticated endpoint
    print("\nTesting public API endpoint...")
    try:
        response = requests.get(f"{base_url}/products")
        if response.status_code == 200:
            products = response.json()
            print(f"✅ Public API works - Found {len(products)} trading pairs")
        else:
            print(f"❌ Public API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing public API: {e}")
        return False
    
    # Test authenticated endpoint
    print("\nTesting authenticated API endpoint...")
    
    # Create request headers
    timestamp = str(int(time.time()))
    method = 'GET'
    path = '/accounts'
    
    signature = generate_signature(api_secret, timestamp, method, path)
    if not signature:
        return False
    
    headers = {
        'CB-ACCESS-KEY': api_key,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json',
    }
    
    # Add passphrase if available
    if passphrase:
        headers['CB-ACCESS-PASSPHRASE'] = passphrase
    
    try:
        response = requests.get(f"{base_url}{path}", headers=headers)
        
        if response.status_code == 200:
            accounts = response.json()
            print(f"✅ Authentication successful! Found {len(accounts)} accounts")
            
            # Display non-zero balances
            non_zero = [a for a in accounts if float(a.get('balance', 0)) > 0]
            if non_zero:
                print("\nNon-zero balances:")
                for account in non_zero:
                    print(f"  {account['currency']}: {account['balance']}")
            
            return True
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Error message: {response.text[:200]}")
            
            if response.status_code == 401 and 'passphrase' in response.text.lower():
                print("\nThe API key requires a passphrase.")
                print("Please set the COINBASE_PASSPHRASE environment variable:")
                print("export COINBASE_PASSPHRASE='your_passphrase_here'")
            
            return False
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        return False

def main():
    """Run the Coinbase API configuration test"""
    print("🔍 COINBASE CONFIGURATION TEST 🔍")
    print("----------------------------------")
    
    # Load configuration
    config = load_coinbase_config()
    if not config:
        print("❌ Could not load Coinbase configuration from trading_config.yaml")
        return False
    
    print("Found Coinbase configuration in trading_config.yaml:")
    print(f"  API Key: {config.get('api_key', 'Not found')[:8]}...")
    print(f"  API Secret: {config.get('api_secret', 'Not found')[:8]}...")
    print(f"  Passphrase: {'Set' if config.get('passphrase') else 'Not set'}")
    print(f"  Sandbox Mode: {'Enabled' if config.get('sandbox') else 'Disabled'}")
    
    # Test connection
    success = test_coinbase_connection(config)
    
    print("\n----------------------------------")
    if success:
        print("✅ SUCCESS: Coinbase configuration is working correctly!")
        print("Your trading bot can now use the Coinbase broker.")
    else:
        print("❌ ERROR: Coinbase API connection failed.")
        print("Please check your configuration and try again.")
    
    return success

if __name__ == "__main__":
    main()
