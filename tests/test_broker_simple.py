#!/usr/bin/env python3
"""
Simple Broker Testing Script

This script demonstrates how to test broker connections
using the multi-broker configuration without requiring
additional dependencies.
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("broker_test")

# Constants
ALPACA_PAPER_API_URL = "https://paper-api.alpaca.markets/v2"
TRADIER_SANDBOX_API_URL = "https://sandbox.tradier.com/v1"

def load_broker_config(broker_name):
    """Load broker configuration from the config file"""
    config_path = Path(__file__).parent / "config" / f"{broker_name}_config.json"
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
        logger.info(f"Connecting to Alpaca Paper API at {ALPACA_PAPER_API_URL}")
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        
        # Test account endpoint
        account_url = f"{ALPACA_PAPER_API_URL}/account"
        logger.info(f"Fetching account info from: {account_url}")
        
        response = requests.get(account_url, headers=headers)
        response.raise_for_status()
        
        account = response.json()
        
        logger.info("✅ Connection successful!")
        logger.info(f"Account ID: {account.get('id')}")
        logger.info(f"Account Status: {account.get('status')}")
        logger.info(f"Buying Power: ${account.get('buying_power')}")
        logger.info(f"Cash: ${account.get('cash')}")
        logger.info(f"Portfolio Value: ${account.get('portfolio_value')}")
        
        # Get positions
        positions_url = f"{ALPACA_PAPER_API_URL}/positions"
        positions_response = requests.get(positions_url, headers=headers)
        positions_response.raise_for_status()
        
        positions = positions_response.json()
        logger.info(f"Current positions: {len(positions)}")
        for position in positions[:5]:  # Show up to 5 positions
            logger.info(f"  {position.get('symbol')}: {position.get('qty')} shares at ${position.get('avg_entry_price')}")
        
        # Check market clock
        clock_url = f"{ALPACA_PAPER_API_URL}/clock"
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

def test_tradier_connection(api_key, account_id):
    """Test connection to Tradier API using direct HTTP requests"""
    # Set up auth headers
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    try:
        logger.info(f"Connecting to Tradier Sandbox API at {TRADIER_SANDBOX_API_URL}")
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        
        # Test account endpoint
        account_url = f"{TRADIER_SANDBOX_API_URL}/user/profile"
        logger.info(f"Fetching user profile from: {account_url}")
        
        response = requests.get(account_url, headers=headers)
        response.raise_for_status()
        
        profile = response.json()
        
        logger.info("✅ Connection successful!")
        profile_data = profile.get('profile', {})
        logger.info(f"User ID: {profile_data.get('id')}")
        logger.info(f"Name: {profile_data.get('name')}")
        
        # Get account balances
        balances_url = f"{TRADIER_SANDBOX_API_URL}/accounts/{account_id}/balances"
        balances_response = requests.get(balances_url, headers=headers)
        balances_response.raise_for_status()
        
        balances = balances_response.json()
        balance_data = balances.get('balances', {})
        
        logger.info(f"Account: {account_id}")
        logger.info(f"Cash: ${balance_data.get('total_cash')}")
        logger.info(f"Buying Power: ${balance_data.get('cash_available')}")
        logger.info(f"Portfolio Value: ${balance_data.get('total_equity')}")
        
        # Get positions
        positions_url = f"{TRADIER_SANDBOX_API_URL}/accounts/{account_id}/positions"
        positions_response = requests.get(positions_url, headers=headers)
        positions_response.raise_for_status()
        
        positions_data = positions_response.json()
        positions = positions_data.get('positions', {}).get('position', [])
        
        if not positions:
            logger.info("No positions found")
        else:
            if not isinstance(positions, list):
                positions = [positions]  # Handle case where single position is not in a list
                
            logger.info(f"Current positions: {len(positions)}")
            for position in positions[:5]:  # Show up to 5 positions
                logger.info(f"  {position.get('symbol')}: {position.get('quantity')} shares at ${position.get('cost_basis')}")
        
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP Error: {e}")
        if e.response.status_code == 401:
            logger.error("Authentication failed. Please check your API credentials.")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to connect to Tradier: {str(e)}")
        return False

def test_broker(broker_name):
    """Test a specific broker"""
    logger.info(f"Testing {broker_name} broker...")
    
    if broker_name == "alpaca":
        # Load Alpaca config
        config = load_broker_config("alpaca")
        if not config:
            return False
        
        # Extract credentials
        alpaca_config = config.get('brokers', {}).get('alpaca', {})
        api_key = alpaca_config.get('api_key')
        api_secret = alpaca_config.get('api_secret')
        
        if not api_key or not api_secret:
            logger.error("Missing Alpaca API credentials in config")
            return False
        
        # Test connection
        return test_alpaca_connection(api_key, api_secret)
    
    elif broker_name == "tradier":
        # Load Tradier config
        config = load_broker_config("tradier")
        if not config:
            return False
        
        # Extract credentials
        tradier_config = config.get('brokers', {}).get('tradier', {})
        api_key = tradier_config.get('api_key')
        account_id = tradier_config.get('account_id')
        
        if not api_key or not account_id:
            logger.error("Missing Tradier API credentials in config")
            return False
        
        # Test connection
        return test_tradier_connection(api_key, account_id)
    
    elif broker_name == "all":
        # Test all configured brokers
        alpaca_success = test_broker("alpaca")
        tradier_success = test_broker("tradier")
        
        return alpaca_success or tradier_success
    
    else:
        logger.error(f"Unknown broker: {broker_name}")
        return False

def main():
    """Main function"""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test broker connections')
    parser.add_argument('--broker', type=str, choices=['alpaca', 'tradier', 'all'],
                        default='all', help='Broker to test')
    args = parser.parse_args()
    
    # Print welcome message
    logger.info("\n=== Broker Connection Test ===")
    
    # Test specified broker
    if test_broker(args.broker):
        logger.info(f"✅ {args.broker.title()} broker test passed!")
        return 0
    else:
        logger.error(f"❌ {args.broker.title()} broker test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
