import os
import json
import time
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import requests

def debug_jwt_token():
    # Get credentials from environment variables
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    
    if not api_key or not api_secret:
        print("ERROR: API key or secret not found in environment variables")
        return None
        
    try:
        # Load private key
        private_key = serialization.load_pem_private_key(
            api_secret.encode(),
            password=None,
            backend=default_backend()
        )
        
        # Create JWT payload
        now = int(time.time())
        payload = {
            "sub": api_key,
            "iss": "coinbase-cloud",
            "nbf": now,
            "exp": now + 60,  # Token expires in 60 seconds
            "aud": ["brokerage"]
        }
        
        # Generate JWT token
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256"
        )
        
        # Print details for debugging
        print(f"API Key: {api_key[:5]}...{api_key[-5:]}")
        print(f"JWT Payload: {json.dumps(payload, indent=2)}")
        print(f"JWT Token: {token[:20]}...{token[-20:]}")
        
        return token
        
    except Exception as e:
        print(f"ERROR generating JWT token: {str(e)}")
        return None

def make_test_request(token):
    # Try multiple base URLs to see which one works
    base_urls = [
        "https://api.coinbase.com/api/v3/brokerage",
        "https://api.exchange.coinbase.com",
        "https://api.pro.coinbase.com",
        "https://api.cloud.coinbase.com/v1"
    ]
    
    endpoint = "/accounts"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    print("\nTrying multiple base URLs to find the correct one...")
    
    for base_url in base_urls:
        url = base_url + endpoint
        print(f"\nTrying URL: {url}")
        print(f"Headers: {json.dumps(headers, indent=2, default=str)}")
        
        try:
            response = requests.get(url, headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
            
            # Only show body if it's not too long
            body_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
            print(f"Response Body: {body_preview}")
            
            if response.status_code != 401:
                print("\n✅ This base URL appears to work!")
                
        except Exception as e:
            print(f"Request error: {str(e)}")

def try_passphrase():
    # Some Coinbase APIs require a passphrase
    passphrase = os.environ.get("COINBASE_PASSPHRASE")
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    
    if not passphrase:
        print("\nNo COINBASE_PASSPHRASE found in environment variables.")
        print("Some Coinbase APIs require a passphrase.")
        return
    
    print("\nAttempting authentication with passphrase...")
    
    # Try Coinbase Pro/Advanced authentication method
    base_url = "https://api.exchange.coinbase.com"
    endpoint = "/accounts"
    url = base_url + endpoint
    
    timestamp = str(int(time.time()))
    message = timestamp + "GET" + endpoint
    
    try:
        # Decode the base64 secret
        import base64
        import hmac
        import hashlib
        
        secret = base64.b64decode(api_secret)
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json"
        }
        
        print(f"URL: {url}")
        print(f"Headers: {json.dumps(headers, indent=2, default=str)}")
        
        response = requests.get(url, headers=headers)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        body_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
        print(f"Response Body: {body_preview}")
        
        if response.status_code != 401:
            print("\n✅ Authentication with passphrase appears to work!")
            
    except Exception as e:
        print(f"Error with passphrase authentication: {str(e)}")

if __name__ == "__main__":
    print("Coinbase Cloud API Debug Tool")
    print("============================")
    token = debug_jwt_token()
    
    if token:
        make_test_request(token)
        
    try_passphrase()
    
    print("\nDebugging Recommendations:")
    print("1. Check if you're using the right API (Cloud vs Advanced/Pro)")
    print("2. Verify the JWT token format and claims")
    print("3. Ensure all required headers are included")
    print("4. Check if IP restrictions are enabled on your API key")
    print("5. Verify your API key hasn't been revoked or disabled")
