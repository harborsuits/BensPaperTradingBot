"""
Telegram Notifier Module

This module provides Telegram notification functionality for the trading bot.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime


class TelegramNotifier:
    """
    Telegram notification handler for the trading bot.
    
    This is a minimal implementation for the demo. In a real system, this
    would include more features like message formatting, button support, etc.
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_chat_id: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            default_chat_id: Default chat ID to send messages to
            debug: Whether to enable debug mode
        """
        self.logger = logging.getLogger("TelegramNotifier")
        
        # Set bot token
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            self.logger.warning("No Telegram bot token provided, notifications will be disabled")
        
        # Set default chat ID
        self.default_chat_id = default_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if not self.default_chat_id:
            self.logger.warning("No default chat ID provided, will need to specify for each message")
        
        # Debug mode
        self.debug = debug
        
        # API URL
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Message history
        self.message_history = []
        
        # Log initialization
        if self.bot_token and self.default_chat_id:
            self.logger.info("Telegram notifier initialized successfully")
        else:
            self.logger.warning("Telegram notifier initialized in disabled state")
    
    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False
    ) -> Dict[str, Any]:
        """
        Send a text message via Telegram.
        
        Args:
            text: Message text to send
            chat_id: Chat ID to send to (defaults to self.default_chat_id)
            parse_mode: Parse mode (HTML or Markdown)
            disable_notification: Whether to disable notification sounds
            
        Returns:
            API response dictionary
        """
        # Check if notifications are enabled
        if not self.bot_token:
            self.logger.warning("Telegram notifications disabled: No bot token")
            return {"ok": False, "error": "No bot token provided"}
        
        # Use default chat ID if not specified
        chat_id = chat_id or self.default_chat_id
        if not chat_id:
            self.logger.error("No chat ID provided or set as default")
            return {"ok": False, "error": "No chat ID provided"}
        
        # Prepare message data
        data = {
            "chat_id": chat_id,
            "text": text,
            "disable_notification": disable_notification
        }
        
        # Add parse mode if specified
        if parse_mode:
            data["parse_mode"] = parse_mode
        
        # Log message in debug mode
        if self.debug:
            self.logger.info(f"Sending Telegram message: {text[:100]}...")
        
        try:
            # Send request to Telegram API
            response = requests.post(
                f"{self.api_url}/sendMessage",
                data=data,
                timeout=10
            )
            
            # Parse response
            result = response.json()
            
            # Add to message history
            msg_record = {
                "timestamp": datetime.now().isoformat(),
                "chat_id": chat_id,
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "success": result.get("ok", False)
            }
            self.message_history.append(msg_record)
            
            # Trim history if needed
            if len(self.message_history) > 100:
                self.message_history = self.message_history[-100:]
            
            # Log result
            if result.get("ok", False):
                self.logger.info(f"Message sent successfully to chat {chat_id}")
            else:
                self.logger.error(f"Failed to send message: {result.get('description', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    def send_trade_notification(
        self,
        trade_type: str,
        symbol: str,
        price: float,
        quantity: float,
        strategy: str = "unknown",
        pnl: Optional[float] = None,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a trade notification.
        
        Args:
            trade_type: Type of trade (BUY, SELL, etc.)
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            strategy: Trading strategy name
            pnl: Profit/loss (if applicable)
            chat_id: Chat ID to send to
            
        Returns:
            API response dictionary
        """
        # Determine emoji based on trade type
        emoji = "🟢" if trade_type.upper() == "BUY" else "🔴" if trade_type.upper() == "SELL" else "🔄"
        
        # Build message
        message = f"{emoji} <b>{trade_type.upper()}: {symbol}</b>\n\n"
        message += f"Price: ${price:,.2f}\n"
        message += f"Quantity: {quantity}\n"
        message += f"Strategy: {strategy}\n"
        
        # Add PnL if provided
        if pnl is not None:
            pnl_emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
            message += f"PnL: {pnl_emoji} ${abs(pnl):,.2f}\n"
        
        # Add timestamp
        message += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        # Send message
        return self.send_message(
            text=message,
            chat_id=chat_id,
            parse_mode="HTML"
        )
    
    def send_error_notification(
        self,
        error_message: str,
        error_type: str = "General Error",
        module: str = "Unknown",
        importance: str = "medium",
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an error notification.
        
        Args:
            error_message: Error message
            error_type: Type of error
            module: Module where the error occurred
            importance: Error importance (low, medium, high)
            chat_id: Chat ID to send to
            
        Returns:
            API response dictionary
        """
        # Determine emoji based on importance
        emoji = "🔴" if importance == "high" else "🟠" if importance == "medium" else "🟡"
        
        # Build message
        message = f"{emoji} <b>ERROR: {error_type}</b>\n\n"
        message += f"Module: {module}\n"
        message += f"Importance: {importance.title()}\n"
        message += f"Message: {error_message}\n"
        
        # Add timestamp
        message += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        # Send message
        return self.send_message(
            text=message,
            chat_id=chat_id,
            parse_mode="HTML",
            disable_notification=importance == "low"
        )
    
    def get_message_history(
        self,
        limit: int = 10,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get message history.
        
        Args:
            limit: Maximum number of messages to return
            success_only: Whether to return only successful messages
            
        Returns:
            List of message records
        """
        # Filter if needed
        if success_only:
            filtered = [msg for msg in self.message_history if msg.get("success", False)]
        else:
            filtered = self.message_history
        
        # Return limited number of messages
        return filtered[-limit:]


# Testing function
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if we have the necessary env vars
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("Error: Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file")
        exit(1)
    
    # Create notifier
    notifier = TelegramNotifier(debug=True)
    
    # Send a test message
    print("Sending test message...")
    result = notifier.send_message("Hello from the trading bot! This is a test message.")
    
    # Check result
    if result.get("ok", False):
        print("Message sent successfully!")
    else:
        print(f"Failed to send message: {result}")
    
    # Send a test trade notification
    print("Sending test trade notification...")
    result = notifier.send_trade_notification(
        trade_type="BUY",
        symbol="BTC/USD",
        price=29876.45,
        quantity=0.15,
        strategy="momentum_breakout"
    )
    
    # Check result
    if result.get("ok", False):
        print("Trade notification sent successfully!")
    else:
        print(f"Failed to send trade notification: {result}")
    
    # Send a trade alert
    notifier.send_trade_alert(
        trade_type="BUY",
        symbol="BTC/USD",
        price=45000.50,
        quantity=0.25,
        strategy="RSI_Divergence"
    )
    
    # Send an error notification
    notifier.send_error_notification(
        error_message="Failed to connect to exchange API",
        error_type="ConnectionError",
        importance="high"
    )
    
    # Send a daily summary
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