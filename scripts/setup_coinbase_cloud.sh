#!/bin/bash
# Coinbase Cloud API Setup Script
# This script sets up a virtual environment with the required dependencies
# and runs a test to verify connectivity with the Coinbase Cloud API.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}  Coinbase Cloud API Setup & Test Script ${NC}"
echo -e "${YELLOW}=========================================${NC}"

# Set project root and environment path
PROJECT_ROOT="/Users/bendickinson/Desktop/Trading:BenBot"
ENV_PATH="/Users/bendickinson/Desktop/coinbase_venv"
cd "$PROJECT_ROOT" || exit 1

# Check if Python 3 is installed
echo -e "\n${YELLOW}Checking for Python 3...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3 and try again.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Found $PYTHON_VERSION${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "$ENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Reusing it.${NC}"
else
    python3 -m venv "$ENV_PATH"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install venv and try again.${NC}"
        echo -e "${YELLOW}On macOS, use: pip3 install virtualenv${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source "$ENV_PATH/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated successfully.${NC}"

# Install required packages
echo -e "\n${YELLOW}Installing required packages...${NC}"
pip install cryptography PyJWT
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install required packages.${NC}"
    exit 1
fi
echo -e "${GREEN}Packages installed successfully.${NC}"

# Create test script in the virtual environment
echo -e "\n${YELLOW}Creating test script...${NC}"
cat > test_coinbase_cloud_auth.py << 'EOL'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Cloud API Authentication Test (Full Version)

This script tests the Coinbase Cloud API connectivity with full JWT authentication.
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    import jwt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Error: Required packages not installed. Please run 'pip install cryptography PyJWT'")
    sys.exit(1)

# Load configuration from trading_config.yaml
def load_config():
    """Load configuration from trading_config.yaml."""
    try:
        import yaml
        config_path = os.path.join('trading_bot', 'config', 'trading_config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('coinbase', {})
    except ImportError:
        print("Warning: Could not import yaml. Using hardcoded credentials.")
        return {
            'api_key_name': "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335",
            'private_key': """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

# Create JWT token for authentication
def create_jwt_token(api_key_name, private_key_pem):
    """Create a JWT token signed with the EC private key."""
    try:
        # Load the private key
        private_key = load_pem_private_key(private_key_pem.encode(), password=None)
        
        # Create the JWT payload
        now = int(time.time())
        payload = {
            'sub': api_key_name,
            'iss': 'coinbase-cloud',
            'nbf': now,
            'exp': now + 60,  # Token expires in 60 seconds
            'aud': ['api.coinbase.com']
        }
        
        # Create the JWT token
        token = jwt.encode(
            payload, 
            private_key, 
            algorithm='ES256'
        )
        
        return token
    except Exception as e:
        print(f"Error creating JWT token: {e}")
        return None

# Test Coinbase Cloud API with authentication
def test_authenticated_api(api_key_name, private_key_pem):
    """Test authenticated Coinbase Cloud API endpoints."""
    token = create_jwt_token(api_key_name, private_key_pem)
    if not token:
        return False
    
    # Test endpoints
    endpoints = [
        # Format: (endpoint, method, description)
        ("https://api.coinbase.com/api/v3/brokerage/products", "GET", "Products"),
        ("https://api.coinbase.com/api/v3/brokerage/accounts", "GET", "Accounts")
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    success_count = 0
    
    for url, method, description in endpoints:
        print(f"\nTesting {description} endpoint ({url})...")
        
        try:
            import urllib.request
            import urllib.error
            
            req = urllib.request.Request(url, headers=headers)
            req.method = method
            
            try:
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    print(f"✅ SUCCESS: Authenticated access to {description}")
                    
                    # Display some data
                    if description == "Products":
                        products = data.get('products', [])
                        print(f"  Found {len(products)} products")
                        if products:
                            print("  Sample products:")
                            for product in products[:5]:
                                print(f"    - {product.get('product_id', 'Unknown')}")
                    elif description == "Accounts":
                        accounts = data.get('accounts', [])
                        print(f"  Found {len(accounts)} accounts")
                        
                    success_count += 1
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8')
                print(f"❌ ERROR: HTTP {e.code} - {error_body}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    return success_count > 0

def main():
    """Run the authentication test."""
    print("🔒 COINBASE CLOUD API AUTHENTICATION TEST 🔒")
    print("===========================================")
    
    # Load configuration
    config = load_config()
    api_key_name = config.get('api_key_name')
    private_key = config.get('private_key')
    
    if not api_key_name or not private_key:
        print("❌ ERROR: Missing API credentials in configuration.")
        return False
    
    print(f"API Key Name: {api_key_name[:20]}...")
    print(f"Private Key: {'Available' if private_key else 'Missing'}")
    
    # Create JWT token as a test
    print("\nTesting JWT token creation...")
    token = create_jwt_token(api_key_name, private_key)
    if token:
        print(f"✅ SUCCESS: JWT token created successfully")
        print(f"  Token: {token[:20]}...")
    else:
        print("❌ ERROR: Failed to create JWT token")
        return False
    
    # Test authenticated API
    print("\nTesting authenticated Coinbase Cloud API access...")
    auth_success = test_authenticated_api(api_key_name, private_key)
    
    # Print summary
    print("\n===========================================")
    if auth_success:
        print("✅ SUCCESS: Authentication and API access working!")
        print("Your Coinbase Cloud API integration is ready to use.")
    else:
        print("⚠️ WARNING: Authentication successful but API access failed.")
        print("This could be due to permissions or API restrictions.")
        print("Please check your API key permissions and try again.")
    
    return auth_success

if __name__ == "__main__":
    main()
EOL
echo -e "${GREEN}Test script created successfully.${NC}"

# Run the test script
echo -e "\n${YELLOW}Running Coinbase Cloud API test...${NC}"
python test_coinbase_cloud_auth.py
TEST_RESULT=$?

# Display next steps
echo -e "\n${YELLOW}=========================================${NC}"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}Setup completed successfully!${NC}"
else
    echo -e "${YELLOW}Setup completed with some issues.${NC}"
fi

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. To activate this environment in the future, run:"
echo -e "   ${GREEN}source /Users/bendickinson/Desktop/coinbase_venv/bin/activate${NC}"
echo -e "2. To use Coinbase as your default broker, run:"
echo -e "   ${GREEN}export DEFAULT_BROKER=coinbase${NC}"
echo -e "3. To deactivate the virtual environment when done:"
echo -e "   ${GREEN}deactivate${NC}\n"

echo -e "${YELLOW}Happy Trading!${NC}"
