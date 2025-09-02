#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public Coinbase API Test

This script tests the public Coinbase API endpoints that don't require authentication.
"""

import logging
import json
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PublicCoinbaseTest")

def test_public_coinbase_api():
    """Test the public Coinbase API endpoints"""
    logger.info("Starting Public Coinbase API Test")
    
    # Test 1: Get Bitcoin price
    logger.info("Testing Bitcoin price endpoint")
    try:
        response = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Bitcoin price: ${data['data']['amount']}")
        else:
            logger.error(f"Failed to get Bitcoin price: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error getting Bitcoin price: {str(e)}")
    
    # Test 2: Get Coinbase time
    logger.info("Testing Coinbase time endpoint")
    try:
        response = requests.get("https://api.coinbase.com/v2/time")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Coinbase time: {data['data']['iso']}")
        else:
            logger.error(f"Failed to get Coinbase time: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error getting Coinbase time: {str(e)}")
    
    # Test 3: Get supported currencies
    logger.info("Testing currencies endpoint")
    try:
        response = requests.get("https://api.coinbase.com/v2/currencies")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Supported currencies: {len(data['data'])} currencies")
            # Display first 5 currencies
            for currency in data['data'][:5]:
                logger.info(f"Currency: {currency['id']} - {currency['name']}")
        else:
            logger.error(f"Failed to get currencies: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error getting currencies: {str(e)}")
    
    # Test 4: Get exchange rates
    logger.info("Testing exchange rates endpoint")
    try:
        response = requests.get("https://api.coinbase.com/v2/exchange-rates?currency=USD")
        if response.status_code == 200:
            data = response.json()
            rates = data['data']['rates']
            logger.info(f"Exchange rates: {len(rates)} rates")
            # Display rates for common cryptocurrencies
            for crypto in ['BTC', 'ETH', 'SOL', 'DOGE', 'USDT']:
                if crypto in rates:
                    logger.info(f"1 USD = {rates[crypto]} {crypto}")
        else:
            logger.error(f"Failed to get exchange rates: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error getting exchange rates: {str(e)}")
    
    logger.info("Public Coinbase API test completed")

if __name__ == "__main__":
    test_public_coinbase_api()
