#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example script to test the Telegram notifier functionality.
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TelegramExample")

# Add the parent directory to the path so we can import the trading_bot package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TelegramNotifier class
from trading_bot.notification_manager.telegram_notifier import TelegramNotifier


def main():
    """Run the Telegram notification examples."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get Telegram credentials from environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.error("Telegram credentials not found in environment variables.")
        logger.error("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return
    
    logger.info(f"Initializing TelegramNotifier with chat ID: {chat_id}")
    
    # Initialize the Telegram notifier
    notifier = TelegramNotifier(
        bot_token=bot_token,
        default_chat_id=chat_id,
        log_dir="./logs",
        debug=True
    )
    
    # Send a simple message
    logger.info("Sending a simple test message...")
    notifier.send_message(
        text="🚀 <b>Hello from Trading Bot!</b>\n\nThis is a test message from the TelegramNotifier example script."
    )
    
    # Send a trade alert
    logger.info("Sending a trade alert...")
    notifier.send_trade_alert(
        trade_type="BUY",
        symbol="BTC/USD",
        price=45000.50,
        quantity=0.25,
        strategy="RSI_Divergence"
    )
    
    # Send an error notification
    logger.info("Sending an error notification...")
    notifier.send_error_notification(
        error_message="Failed to connect to exchange API",
        error_type="ConnectionError",
        importance="high"
    )
    
    # Send a daily summary
    logger.info("Sending a daily summary...")
    notifier.send_daily_summary(
        total_trades=15,
        profitable_trades=10,
        total_profit=450.75,
        win_rate=66.7,
        top_performers=[
            {"symbol": "ETH/USD", "pnl": 250.50},
            {"symbol": "BTC/USD", "pnl": 125.25},
            {"symbol": "SOL/USD", "pnl": 75.0}
        ]
    )
    
    logger.info("All messages sent successfully!")


if __name__ == "__main__":
    main() 