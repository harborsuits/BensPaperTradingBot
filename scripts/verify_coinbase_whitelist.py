#!/usr/bin/env python3
"""
Coinbase API Whitelist Verification Tool

This script verifies that your Coinbase API keys are properly configured
with the correct IP whitelist settings.
"""

import os
import sys
import json
import time
import base64
import hmac
import hashlib
import requests
from datetime import datetime
from typing import Dict, Any, Tuple

# Import cryptography libraries if available
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    import jwt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: Cryptography libraries not available. JWT authentication will not work.")

# Configuration
API_KEYS = {
    "advanced": {
        "name": "BenBot Advanced Trading",
        "api_key": "acc.adb53c0e-35a0-4171-b237-a19fec741363",
        "api_secret": "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
    },
    "cloud": {
        "name": "BenbotReal Cloud",
        "api_key_name": "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335",
        "private_key": """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
    }
}

def get_public_ip() -> str:
    """Get the current public IP address"""
    try:
        response = requests.get("https://api.ipify.org")
        if response.status_code == 200:
            return response.text
        else:
            return "Unknown"
    except:
        return "Unknown"

def test_advanced_api() -> Tuple[bool, Dict[str, Any]]:
    """Test connection using Advanced Trading API credentials"""
    print("\n--- TESTING COINBASE ADVANCED TRADING API ---")
    
    # Get credentials
    api_key = API_KEYS["advanced"]["api_key"]
    api_secret = API_KEYS["advanced"]["api_secret"]
    
    # Set up the request
    timestamp = str(int(time.time()))
    method = "GET"
    endpoint = "/api/v3/brokerage/accounts"
    body = ""
    
    # Create the message to sign
    message = f"{timestamp}{method}{endpoint}{body}"
    
    try:
        # Create HMAC signature
        secret = base64.b64decode(api_secret)
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        # Set headers
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        # Make the request
        url = f"https://api.coinbase.com{endpoint}"
        print(f"Making request to: {url}")
        print(f"Using headers: {headers}")
        
        response = requests.get(url, headers=headers)
        
        # Display response
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Coinbase Advanced Trading API connection successful!")
            return True, response.json()
        else:
            print(f"❌ ERROR: {response.status_code} - {response.text}")
            
            if response.status_code == 401:
                print("\nPossible IP whitelist issues:")
                print(f"Your current public IP: {get_public_ip()}")
                print("Ensure this IP is added to Coinbase API key whitelist")
                
            return False, {"error": response.text}
            
    except Exception as e:
        print(f"❌ ERROR: Exception - {str(e)}")
        return False, {"error": str(e)}
        
def test_cloud_api() -> Tuple[bool, Dict[str, Any]]:
    """Test connection using Cloud API credentials with JWT"""
    if not CRYPTO_AVAILABLE:
        print("❌ Skipping Cloud API test - cryptography libraries not available")
        return False, {"error": "Cryptography libraries not available"}
        
    print("\n--- TESTING COINBASE CLOUD API ---")
    
    # Get credentials
    api_key_name = API_KEYS["cloud"]["api_key_name"]
    private_key = API_KEYS["cloud"]["private_key"]
    
    try:
        # Load the private key
        private_key_obj = serialization.load_pem_private_key(
            private_key.encode(),
            password=None,
            backend=default_backend()
        )
        
        # Create JWT token
        now = int(time.time())
        payload = {
            "sub": api_key_name,
            "iss": "coinbase-cloud",
            "nbf": now,
            "exp": now + 60,  # Token expires in 60 seconds
            "aud": ["brokerage"]  # Audience claim
        }
        
        token = jwt.encode(
            payload,
            private_key_obj,
            algorithm="ES256"
        )
        
        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Try different base URLs
        base_urls = [
            "https://api.coinbase.com/api/v3/brokerage",
            "https://api.exchange.coinbase.com",
            "https://api.cloud.coinbase.com/v1"
        ]
        
        for base_url in base_urls:
            endpoint = "/accounts"  # Common endpoint to test
            url = f"{base_url}{endpoint}"
            
            print(f"\nTrying URL: {url}")
            
            try:
                response = requests.get(url, headers=headers)
                print(f"Status code: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ SUCCESS: Coinbase Cloud API connection successful at {base_url}!")
                    return True, response.json()
                else:
                    print(f"❌ ERROR at {base_url}: {response.status_code} - {response.text}")
                    
                    if response.status_code == 401:
                        print("\nPossible IP whitelist issues:")
                        print(f"Your current public IP: {get_public_ip()}")
                        print("Ensure this IP is added to Coinbase API key whitelist")
                        
            except Exception as e:
                print(f"❌ ERROR with {base_url}: {str(e)}")
                
        return False, {"error": "Failed to connect to any Coinbase Cloud API endpoint"}
        
    except Exception as e:
        print(f"❌ ERROR: Exception - {str(e)}")
        return False, {"error": str(e)}

def check_ip_whitelist_status():
    """Check IP whitelist status for Coinbase API keys"""
    print("\n--- COINBASE API IP WHITELIST CHECK ---")
    print(f"Current public IP address: {get_public_ip()}")
    print("\nVerifying if this IP is whitelisted for your Coinbase API keys...")
    
    # Try public endpoints first (these should work regardless of IP whitelist)
    print("\nTesting public endpoints (should work without IP whitelist):")
    try:
        url = "https://api.coinbase.com/v2/currencies"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("✅ Public API accessible - internet connection confirmed")
        else:
            print(f"❌ Public API inaccessible - check internet connection: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cannot access public endpoints - check internet connection: {str(e)}")
    
    # Test advanced API
    advanced_success, _ = test_advanced_api()
    
    # Test cloud API
    cloud_success, _ = test_cloud_api()
    
    # Summary
    print("\n--- IP WHITELIST STATUS SUMMARY ---")
    if advanced_success:
        print("✅ Advanced API: IP is correctly whitelisted")
    else:
        print("❌ Advanced API: IP is NOT whitelisted correctly")
        
    if cloud_success:
        print("✅ Cloud API: IP is correctly whitelisted")
    else:
        print("❌ Cloud API: IP is NOT whitelisted correctly")
    
    print("\nWhitelisting instructions:")
    print(f"1. Log into your Coinbase account")
    print(f"2. Navigate to API settings")
    print(f"3. Add this IP address: {get_public_ip()} to the whitelist")
    print(f"4. Save changes and wait a few minutes for them to take effect")
    print(f"5. Run this script again to verify")
    
    if not (advanced_success or cloud_success):
        # Create an example .env file for Coinbase credentials
        env_content = f"""# Coinbase API Credentials
# Add to your .env file or environment variables

# Advanced Trading API (BenBot)
COINBASE_API_KEY=acc.adb53c0e-35a0-4171-b237-a19fec741363
COINBASE_API_SECRET=eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ==

# Cloud API (BenbotReal)
COINBASE_CLOUD_API_KEY_NAME=organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335
COINBASE_CLOUD_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----\\nMHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49\\nAwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX\\nWHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==\\n-----END EC PRIVATE KEY-----

# If deploying to a server, note its IP address here:
# SERVER_IP=your_server_ip_here
"""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coinbase.env")
        with open(env_path, "w") as f:
            f.write(env_content)
            
        print(f"\nCreated example environment file at: {env_path}")
        print("You can use these environment variables when deploying to a server")

if __name__ == "__main__":
    print("Coinbase API Whitelist Verification Tool")
    print("=======================================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_ip_whitelist_status()
    
    print("\nDone!")
