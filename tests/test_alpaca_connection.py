#!/usr/bin/env python3
"""
Alpaca Paper Trading Connection Test

This script tests the connection to Alpaca Paper Trading API
using the credentials provided in the configuration.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alpaca_test")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Try to import alpaca-trade-api
try:
    import alpaca_trade_api
    from alpaca_trade_api.rest import REST
    logger.info(f"Alpaca SDK version: {alpaca_trade_api.__version__}")
except ImportError:
    logger.error("Alpaca SDK not installed. Please install with: pip install alpaca-trade-api")
    sys.exit(1)

def load_alpaca_config():
    """Load Alpaca configuration from the config file"""
    config_path = project_root / "config" / "alpaca_config.json"
    logger.info(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return None

def test_alpaca_connection(api_key, api_secret, paper_trading=True):
    """Test connection to Alpaca API"""
    try:
        # Determine the API URL based on paper_trading flag
        base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        
        logger.info(f"Connecting to Alpaca API at {base_url}")
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        
        # Create the Alpaca client
        api = REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url
        )
        
        # Test the connection
        logger.info("Testing connection...")
        account = api.get_account()
        
        logger.info("✅ Connection successful!")
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Buying Power: ${account.buying_power}")
        logger.info(f"Cash: ${account.cash}")
        logger.info(f"Portfolio Value: ${account.portfolio_value}")
        
        # Get positions
        positions = api.list_positions()
        logger.info(f"Current positions: {len(positions)}")
        for position in positions:
            logger.info(f"  {position.symbol}: {position.qty} shares at ${position.avg_entry_price}")
        
        # Check market clock
        clock = api.get_clock()
        logger.info(f"Market is {'open' if clock.is_open else 'closed'}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Alpaca: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("=== Alpaca Paper Trading Connection Test ===")
    
    # Load configuration
    config = load_alpaca_config()
    if not config:
        return 1
    
    # Extract Alpaca credentials
    alpaca_config = config.get('brokers', {}).get('alpaca', {})
    api_key = alpaca_config.get('api_key')
    api_secret = alpaca_config.get('api_secret')
    paper_trading = alpaca_config.get('paper_trading', True)
    
    if not api_key or not api_secret:
        logger.error("Missing Alpaca API credentials in config")
        return 1
    
    # Test connection
    if test_alpaca_connection(api_key, api_secret, paper_trading):
        logger.info("✅ Alpaca connection test passed!")
    else:
        logger.error("❌ Alpaca connection test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
