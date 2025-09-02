#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test for the TradingView webhook.
"""

import os
import sys
import logging
import time
import json
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("WebhookTest")

def run_webhook():
    """Run the TradingView webhook server."""
    # Add directory to python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import the webhook
    from trading_bot.integrations.tradingview_webhook import TradingViewWebhook
    
    # Create a webhook instance
    webhook = TradingViewWebhook(port=5000)
    
    # Start the webhook
    webhook.start()
    
    logger.info("Webhook server started on port 5000")
    
    # Wait a bit for the server to start
    time.sleep(1)
    
    return webhook

def send_test_alert(port=5000):
    """Send a test alert to the webhook."""
    url = f"http://localhost:{port}/tradingview"
    headers = {"Content-Type": "application/json"}
    
    # Simple test data
    data = {
        "symbol": "BTCUSDT",
        "asset_type": "crypto",
        "timestamp": datetime.now().isoformat(),
        "close": 40500,
        "indicators": {
            "rsi": 65.5,
            "macd": 250.5
        }
    }
    
    try:
        logger.info(f"Sending test alert to webhook: {json.dumps(data)}")
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response text: {response.text}")
        return response
    except Exception as e:
        logger.error(f"Error sending test alert: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the webhook
    webhook = run_webhook()
    
    # Send a test alert
    send_test_alert()
    
    logger.info("Webhook test complete. Press Ctrl+C to exit.")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...") 