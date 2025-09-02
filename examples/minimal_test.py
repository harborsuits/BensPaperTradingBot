#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test script for the webhook handler.

This script demonstrates how to use the WebhookHandler class to process incoming
webhook requests from TradingView or other sources.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the WebhookHandler from our package
try:
    from trading_bot.integrations.webhook_handler import WebhookHandler
except ImportError as e:
    logger.error(f"Error importing WebhookHandler: {e}")
    logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def alert_handler(data):
    """
    Process incoming alert data.
    
    Args:
        data: Dictionary containing alert data
    """
    logger.info(f"Alert received for symbol: {data.get('symbol')}")
    logger.info(f"Alert data: {json.dumps(data, indent=2)}")
    
    # You would typically implement your trading logic here
    # For example:
    # 1. Extract signal information
    # 2. Validate the signal
    # 3. Execute trades based on the signal
    
    # For this minimal example, we just log the data
    logger.info("Alert processed successfully")

def main():
    """Run the minimal webhook test."""
    try:
        # Create and configure the webhook handler
        webhook_handler = WebhookHandler(
            port=8080,
            path='/tradingview',
            # auth_token="your_secure_token_here",  # Uncomment to enable authentication
            rate_limit=60  # Max 60 requests per minute
        )
        
        # Register our alert handler
        webhook_handler.register_handler(alert_handler)
        
        # Start the webhook server
        logger.info("Starting webhook server...")
        webhook_handler.start()
        
        # Keep the main thread running
        logger.info(f"Webhook server is running on port {webhook_handler.port}")
        logger.info(f"Endpoint URL: http://localhost:{webhook_handler.port}/{webhook_handler.path}")
        logger.info("Press Ctrl+C to stop")
        
        # Wait for keyboard interrupt
        while True:
            try:
                input()  # This will block until Enter is pressed
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping server...")
                break
    
    except Exception as e:
        logger.error(f"Error in webhook server: {e}")
    finally:
        # Stop the server if it's running
        try:
            if 'webhook_handler' in locals() and webhook_handler.running:
                webhook_handler.stop()
                logger.info("Webhook server stopped")
        except Exception as e:
            logger.error(f"Error stopping webhook server: {e}")

if __name__ == "__main__":
    main() 