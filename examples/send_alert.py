#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal client to send test alerts to the TradingView webhook.
"""

import json
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("AlertSender")

def send_test_alert(port=8080, asset_type="crypto"):
    """
    Send a test alert to the webhook.
    
    Args:
        port: Port number of the webhook server
        asset_type: Type of asset to send alert for
    """
    url = f"http://localhost:{port}/tradingview"
    headers = {"Content-Type": "application/json"}
    
    # Create test data based on asset type
    data = {
        "timestamp": datetime.now().isoformat()
    }
    
    if asset_type == "crypto":
        data.update({
            "symbol": "BTCUSDT",
            "asset_type": "crypto",
            "open": 40000,
            "high": 41000,
            "low": 39500,
            "close": 40500,
            "volume": 1500,
            "indicators": {
                "rsi": 65.5,
                "macd": 250.5,
                "macd_signal": 200.2,
                "macd_hist": 50.3,
                "bb_upper": 42000,
                "bb_middle": 40000,
                "bb_lower": 38000
            }
        })
    elif asset_type == "equity":
        data.update({
            "symbol": "AAPL",
            "asset_type": "stock",
            "open": 175.25,
            "high": 176.50,
            "low": 174.80,
            "close": 176.20,
            "volume": 45000000,
            "indicators": {
                "rsi": 58.2,
                "macd": 0.75,
                "macd_signal": 0.50,
                "macd_hist": 0.25,
                "bb_upper": 178.00,
                "bb_middle": 175.00,
                "bb_lower": 172.00
            }
        })
    elif asset_type == "forex":
        data.update({
            "symbol": "EURUSD",
            "asset_type": "forex",
            "open": 1.0950,
            "high": 1.0975,
            "low": 1.0940,
            "close": 1.0965,
            "volume": 125000,
            "indicators": {
                "rsi": 48.5,
                "macd": -0.0015,
                "macd_signal": -0.0010,
                "macd_hist": -0.0005,
                "atr": 0.0045
            }
        })
    
    try:
        logger.info(f"Sending test alert for {data['symbol']}")
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response text: {response.text}")
        return response
    except Exception as e:
        logger.error(f"Error sending test alert: {str(e)}")
        return None

if __name__ == "__main__":
    # Send test alerts for different asset types
    send_test_alert(asset_type="crypto")
    send_test_alert(asset_type="equity")
    send_test_alert(asset_type="forex") 