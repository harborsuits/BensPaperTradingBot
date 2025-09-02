#!/usr/bin/env python3
"""
Test script for the updated CoinbaseCloudBroker implementation
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to ensure imports work
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the broker
from trading_bot.brokers.coinbase_cloud_broker import CoinbaseCloudBroker

def test_broker_initialization():
    """Test broker initialization with API credentials"""
    # API key name from BenbotReal
    api_key_name = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
    
    # Private key from BenbotReal
    private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
    
    # Initialize broker in read-only mode for safety
    broker = CoinbaseCloudBroker(api_key_name=api_key_name, private_key=private_key, sandbox=False)
    
    # Test connection
    connection_info = broker.check_connection()
    
    print(f"\nConnection Result: {'Success' if connection_info.get('connected', False) else 'Failed'}")
    print(f"Status: {connection_info.get('status', 'Unknown')}")
    
    return broker if connection_info.get('connected', False) else None

def test_market_data(broker):
    """Test market data retrieval functions"""
    print("\n--- TESTING MARKET DATA ---")
    
    # Test get_latest_price for BTC-USD
    print("\nGetting BTC-USD price...")
    btc_price = broker.get_latest_price("BTC-USD")
    print(f"BTC-USD Price: ${btc_price}")
    
    # Test get_quote for BTC-USD
    print("\nGetting BTC-USD quote...")
    success, quote = broker.get_quote("BTC-USD")
    print(f"Quote retrieval successful: {success}")
    if success:
        print(f"Bid: ${quote.get('bid', 'N/A')}")
        print(f"Ask: ${quote.get('ask', 'N/A')}")
        print(f"Spread: ${float(quote.get('ask', 0)) - float(quote.get('bid', 0)):.2f}")
    
    # Test get_bars for BTC-USD (last 24 hours)
    print("\nGetting BTC-USD historical bars (last 24 hours)...")
    end = datetime.now()
    start = end - timedelta(days=1)
    
    bars = broker.get_bars("BTC-USD", "1h", start, end)
    
    print(f"Retrieved {len(bars)} hourly bars")
    if bars:
        print("\nLatest 3 bars:")
        for i, bar in enumerate(bars[:3]):
            print(f"  {i+1}. {bar.get('timestamp')}: Open=${bar.get('open')}, Close=${bar.get('close')}, Volume={bar.get('volume')}")
    
    return True

def test_account_data(broker):
    """Test account data retrieval (requires authentication)"""
    print("\n--- TESTING ACCOUNT DATA ---")
    
    # Test get_account_balances
    print("\nGetting account balances...")
    balances = broker.get_account_balances()
    
    print(f"Found {len(balances)} currencies")
    if balances:
        print("\nTop balances:")
        for currency, balance in list(balances.items())[:5]:
            print(f"  {currency}: Available={balance.get('available', 0)}, Total={balance.get('total', 0)}")
    
    # Test get_positions
    print("\nGetting current positions...")
    positions = broker.get_positions()
    
    print(f"Found {len(positions)} open positions")
    if positions:
        print("\nOpen positions:")
        for pos in positions:
            print(f"  {pos.get('symbol')}: {pos.get('quantity')} @ ${pos.get('current_price')} = ${pos.get('value_usd')}")
    
    return True

def run_all_tests():
    """Run all broker tests"""
    print("Coinbase Cloud Broker Test")
    print("=========================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize broker
    broker = test_broker_initialization()
    
    if broker:
        # Test market data functions
        market_data_success = test_market_data(broker)
        
        # Test account data functions
        account_data_success = test_account_data(broker)
        
        # Print summary
        print("\n--- TEST SUMMARY ---")
        print(f"Broker Initialization: {'✅ Success' if broker else '❌ Failed'}")
        print(f"Market Data Retrieval: {'✅ Success' if market_data_success else '❌ Failed'}")
        print(f"Account Data Retrieval: {'✅ Success' if account_data_success else '❌ Failed'}")
        
        if market_data_success:
            print("\n✅ The broker is ready for read-only market data usage!")
        if account_data_success:
            print("✅ The broker is ready for account data usage!")
            
        print("\nNext Steps:")
        print("1. Integrate with your crypto trading strategies")
        print("2. Add Coinbase market data to your dashboard")
        print("3. Test with paper trading before enabling live trading")
    else:
        print("\n❌ Broker initialization failed. Please check API credentials and IP whitelist.")

if __name__ == "__main__":
    run_all_tests()
