#!/usr/bin/env python3
"""
Quick test for BenbotReal Coinbase Cloud API with whitelisted IP
"""

import os
import time
import json
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt

# BenbotReal credentials
API_KEY_NAME = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""

def create_jwt_token():
    """Create JWT token for Coinbase Cloud API authentication"""
    try:
        # Load the private key
        private_key = serialization.load_pem_private_key(
            PRIVATE_KEY.encode(),
            password=None,
            backend=default_backend()
        )
        
        # Create JWT payload
        now = int(time.time())
        payload = {
            "sub": API_KEY_NAME,
            "iss": "coinbase-cloud",
            "nbf": now,
            "exp": now + 60,  # Token expires in 60 seconds
            "aud": ["brokerage"]  # Audience claim
        }
        
        # Generate JWT token
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256"
        )
        
        print(f"JWT token created successfully")
        return token
        
    except Exception as e:
        print(f"Error creating JWT token: {str(e)}")
        return None

def test_cloud_api():
    """Test BenbotReal Coinbase Cloud API connection"""
    token = create_jwt_token()
    if not token:
        return
    
    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Try different endpoints
    base_urls = [
        "https://api.coinbase.com/api/v3/brokerage",
        "https://api.coinbase.com/v3",
        "https://api.exchange.coinbase.com"
    ]
    
    endpoints = [
        "/accounts",
        "/products",
        "/portfolios",
        "",  # Base endpoint
        "/user"
    ]
    
    for base_url in base_urls:
        print(f"\nTrying base URL: {base_url}")
        
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            print(f"\n  Testing: {url}")
            
            try:
                response = requests.get(url, headers=headers)
                print(f"  Status: {response.status_code}")
                
                if response.status_code == 200:
                    print("  ✅ SUCCESS! Authentication working correctly")
                    print(f"  Response: {response.text[:200]}...")
                    return True
                elif response.status_code == 401:
                    print("  ❌ Authentication failed (401 Unauthorized)")
                else:
                    print(f"  ⚠️ Request failed with status: {response.status_code}")
                    print(f"  Response: {response.text[:100]}...")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
    
    return False

def test_advanced_api():
    """Test the original BenBot (Advanced API) credentials"""
    import base64
    import hmac
    import hashlib
    
    API_KEY = "acc.adb53c0e-35a0-4171-b237-a19fec741363"
    API_SECRET = "eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="
    
    print("\nTesting BenBot Advanced API credentials...")
    timestamp = str(int(time.time()))
    method = "GET"
    
    endpoints = [
        "/api/v3/brokerage/accounts",
        "/v2/accounts",
        "/v2/user"
    ]
    
    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")
        
        message = f"{timestamp}{method}{endpoint}"
        
        try:
            # Create signature
            secret = base64.b64decode(API_SECRET)
            signature = hmac.new(secret, message.encode(), hashlib.sha256)
            signature_b64 = base64.b64encode(signature.digest()).decode()
            
            # Set headers
            headers = {
                "CB-ACCESS-KEY": API_KEY,
                "CB-ACCESS-SIGN": signature_b64,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "Content-Type": "application/json"
            }
            
            url = f"https://api.coinbase.com{endpoint}"
            print(f"URL: {url}")
            
            response = requests.get(url, headers=headers)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS! Advanced API authentication working correctly")
                print(f"Response: {response.text[:200]}...")
                return True
            else:
                print(f"Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return False

if __name__ == "__main__":
    print("Testing Coinbase API with whitelisted IP")
    print("======================================")
    
    # Get current public IP
    try:
        ip_response = requests.get("https://api.ipify.org")
        if ip_response.status_code == 200:
            public_ip = ip_response.text
            print(f"Your current public IP: {public_ip}")
        else:
            print("Could not determine your public IP")
    except Exception as e:
        print(f"Error getting public IP: {str(e)}")
    
    # Test both API methods
    print("\n--- TESTING BENBOT REAL (CLOUD API) ---")
    cloud_success = test_cloud_api()
    
    print("\n--- TESTING BENBOT ORIGINAL (ADVANCED API) ---")
    advanced_success = test_advanced_api()
    
    # Summary
    print("\n=== AUTHENTICATION SUMMARY ===")
    if cloud_success:
        print("✅ BenbotReal (Cloud API): Working correctly")
    else:
        print("❌ BenbotReal (Cloud API): Not working yet")
        
    if advanced_success:
        print("✅ BenBot (Advanced API): Working correctly")
    else:
        print("❌ BenBot (Advanced API): Not working yet")
        
    if not (cloud_success or advanced_success):
        print("\nTroubleshooting steps:")
        print("1. Double-check that your IP whitelist includes your current public IP")
        print("2. Ensure the API keys have the necessary permissions")
        print("3. Wait a few minutes for the whitelist changes to take effect")
    else:
        print("\nCongratulations! At least one API method is working correctly.")
        print("You can now use these credentials with your trading system.")
