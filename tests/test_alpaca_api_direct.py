#!/usr/bin/env python3
"""
Alpaca Paper Trading API Direct Test

This script tests the connection to Alpaca Paper Trading API
using direct HTTP requests instead of the alpaca-trade-api package.
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alpaca_direct_test")

# Constants
PAPER_API_URL = "https://paper-api.alpaca.markets/v2"

def load_alpaca_config():
    """Load Alpaca configuration from the config file"""
    config_path = Path(__file__).parent / "config" / "alpaca_config.json"
    logger.info(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return None

def test_alpaca_connection(api_key, api_secret):
    """Test connection to Alpaca API using direct HTTP requests"""
    # Set up auth headers
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
        'Accept': 'application/json'
    }
    
    try:
        logger.info(f"Connecting to Alpaca Paper API at {PAPER_API_URL}")
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        
        # Test account endpoint
        account_url = f"{PAPER_API_URL}/account"
        logger.info(f"Fetching account info from: {account_url}")
        
        response = requests.get(account_url, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        account = response.json()
        
        logger.info("✅ Connection successful!")
        logger.info(f"Account ID: {account.get('id')}")
        logger.info(f"Account Status: {account.get('status')}")
        logger.info(f"Buying Power: ${account.get('buying_power')}")
        logger.info(f"Cash: ${account.get('cash')}")
        logger.info(f"Portfolio Value: ${account.get('portfolio_value')}")
        
        # Get positions
        positions_url = f"{PAPER_API_URL}/positions"
        positions_response = requests.get(positions_url, headers=headers)
        positions_response.raise_for_status()
        
        positions = positions_response.json()
        logger.info(f"Current positions: {len(positions)}")
        for position in positions[:5]:  # Show up to 5 positions
            logger.info(f"  {position.get('symbol')}: {position.get('qty')} shares at ${position.get('avg_entry_price')}")
        
        # Check market clock
        clock_url = f"{PAPER_API_URL}/clock"
        clock_response = requests.get(clock_url, headers=headers)
        clock_response.raise_for_status()
        
        clock = clock_response.json()
        logger.info(f"Market is {'open' if clock.get('is_open') else 'closed'}")
        
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP Error: {e}")
        if e.response.status_code == 401:
            logger.error("Authentication failed. Please check your API credentials.")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to connect to Alpaca: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("=== Alpaca Paper Trading API Direct Test ===")
    
    # Load configuration
    config = load_alpaca_config()
    if not config:
        return 1
    
    # Extract Alpaca credentials
    alpaca_config = config.get('brokers', {}).get('alpaca', {})
    api_key = alpaca_config.get('api_key')
    api_secret = alpaca_config.get('api_secret')
    
    if not api_key or not api_secret:
        logger.error("Missing Alpaca API credentials in config")
        return 1
    
    # Test connection
    if test_alpaca_connection(api_key, api_secret):
        logger.info("✅ Alpaca connection test passed!")
    else:
        logger.error("❌ Alpaca connection test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
