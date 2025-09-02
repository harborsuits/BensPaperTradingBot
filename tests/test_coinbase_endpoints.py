#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Coinbase API integration.
This standalone script verifies the functionality of the Coinbase API client
without requiring the full trading bot backend infrastructure.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CoinbaseTest")

# Add project root to path
sys.path.append("/Users/bendickinson/Desktop/Trading:BenBot")

# Import Coinbase broker
from trading_bot.brokers.coinbase_cloud_broker import CoinbaseCloudBroker

def test_coinbase_cloud_api():
    """Test the Coinbase Cloud API functionality"""
    logger.info("Starting Coinbase Cloud API test")
    
    # Create broker with the BenbotReal credentials
    api_key_name = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
    private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
    
    broker = CoinbaseCloudBroker(api_key_name=api_key_name, private_key=private_key, sandbox=False)
    
    # Test JWT token creation
    logger.info("Testing JWT token creation")
    jwt_token = broker.create_jwt_token()
    logger.info(f"JWT token created: {jwt_token is not None}")
    
    # Test getting available products
    logger.info("Testing available products")
    success, products = broker.get_available_products()
    if success:
        logger.info(f"Got {len(products)} available products")
        # Display first 5 products
        for p in products[:5]:
            logger.info(f"Product: {p.get('id')}")
    else:
        logger.error(f"Failed to get products: {products}")
    
    # Test getting ticker for BTC-USD
    logger.info("Testing ticker for BTC-USD")
    success, ticker = broker.get_ticker("BTC-USD")
    if success:
        logger.info(f"BTC-USD ticker: Price = ${ticker.get('price')}")
    else:
        logger.error(f"Failed to get ticker: {ticker}")
    
    # Test getting candles for BTC-USD
    logger.info("Testing candles for BTC-USD")
    end = datetime.now()
    start = end - timedelta(days=1)
    candles = broker.get_historical_candles("BTC-USD", 3600, start, end)
    
    if isinstance(candles, tuple):
        success, candles = candles
        if success:
            logger.info(f"Got {len(candles)} candles for BTC-USD")
        else:
            logger.error(f"Failed to get candles: {candles}")
    else:
        if candles and len(candles) > 0:
            logger.info(f"Got {len(candles)} candles for BTC-USD")
        else:
            logger.error("Failed to get candles or empty result")
    
    # Test getting order book for BTC-USD
    logger.info("Testing order book for BTC-USD")
    success, orderbook = broker.get_order_book("BTC-USD", level=2)
    if success:
        logger.info(f"BTC-USD order book: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
    else:
        logger.error(f"Failed to get order book: {orderbook}")
    
    logger.info("Coinbase Cloud API test completed")

if __name__ == "__main__":
    test_coinbase_cloud_api()
