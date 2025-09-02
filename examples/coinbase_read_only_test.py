#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase API Read-Only Testing Script

This script demonstrates how to use the Coinbase API in read-only mode
to safely test integration without any risk to your live account.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.brokers.coinbase_broker_client import CoinbaseBrokerageClient
from trading_bot.brokers.broker_registry import get_broker_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReadOnlyCoinbaseClient:
    """
    A wrapper around the CoinbaseBrokerageClient that allows only read operations
    and blocks all trading/order operations.
    """
    
    def __init__(self, client):
        """
        Initialize with a real Coinbase client but intercept write operations.
        
        Args:
            client: The actual CoinbaseBrokerageClient instance
        """
        self.client = client
        self.blocked_calls = 0
        
    def __getattr__(self, name):
        """
        Intercept method calls to the wrapped client.
        
        Args:
            name: The method name being called
            
        Returns:
            The original method if it's a read operation, or a safe replacement
            if it's a write operation.
        """
        # Get the actual method from the client
        attr = getattr(self.client, name)
        
        # If it's not callable (e.g., a property), just return it
        if not callable(attr):
            return attr
            
        # List of methods that modify account state (place orders, etc.)
        write_methods = [
            'place_order', 'cancel_order', 'cancel_all_orders', 
            'create_market_order', 'create_limit_order', 
            'withdraw', 'transfer', 'deposit'
        ]
        
        # If it's a write method, return a safe replacement
        if name in write_methods:
            def safe_method(*args, **kwargs):
                self.blocked_calls += 1
                logger.warning(f"BLOCKED WRITE OPERATION: {name} with args={args}, kwargs={kwargs}")
                return {"status": "simulated", "message": f"Write operation '{name}' blocked in read-only mode"}
            
            return safe_method
            
        # For read methods, just pass through to the real client
        return attr


def test_coinbase_read_only():
    """Run a read-only test of the Coinbase API."""
    # Set up your Coinbase API credentials
    # Replace these with your actual Coinbase API credentials
    # For security, you should use environment variables instead
    coinbase_config = {
        'api_key': os.environ.get('COINBASE_API_KEY', 'your_api_key_here'),
        'api_secret': os.environ.get('COINBASE_API_SECRET', 'your_api_secret_here'),
        'passphrase': os.environ.get('COINBASE_PASSPHRASE', None),  # Optional for Advanced API
        'sandbox': False  # Using real API but in read-only mode
    }
    
    try:
        # Create the actual client
        real_client = CoinbaseBrokerageClient(**coinbase_config)
        
        # Wrap it with our read-only proxy
        safe_client = ReadOnlyCoinbaseClient(real_client)
        
        # Register with broker registry
        broker_registry = get_broker_registry()
        broker_registry.register_broker('coinbase_readonly', safe_client)
        
        logger.info("Coinbase read-only client registered successfully")
        
        # Test various read operations
        logger.info("Testing read operations...")
        
        # Get account information (should work)
        try:
            account_info = safe_client.get_account_info()
            logger.info(f"Account Information: {account_info}")
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
        
        # Get market data (should work)
        try:
            btc_quote = safe_client.get_quote("BTC-USD")
            logger.info(f"BTC-USD Quote: {btc_quote}")
        except Exception as e:
            logger.error(f"Error getting BTC quote: {str(e)}")
            
        # Get historical data (should work)
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            eth_bars = safe_client.get_bars(
                symbol="ETH-USD",
                timeframe="1h",
                start=start_date,
                end=end_date
            )
            logger.info(f"Retrieved {len(eth_bars)} bars for ETH-USD")
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
        
        # Test write operations (should be blocked)
        logger.info("Testing write operations (should be blocked)...")
        
        # Try to place an order (should be blocked)
        try:
            order_result = safe_client.place_order(
                symbol="BTC-USD",
                side="buy",
                quantity=0.001,
                order_type="market"
            )
            logger.info(f"Order result: {order_result}")
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            
        # Report blocked calls
        logger.info(f"Total blocked write operations: {safe_client.blocked_calls}")
        
    except Exception as e:
        logger.error(f"Error in read-only test: {str(e)}")
    finally:
        # Clean up
        broker_registry.disconnect_all()
        logger.info("Test completed")


if __name__ == "__main__":
    test_coinbase_read_only()
