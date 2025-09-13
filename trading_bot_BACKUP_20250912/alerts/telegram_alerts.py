"""
Telegram Alert Integration for BensBot

This module provides Telegram integration for sending risk alerts, strategy rotation notifications,
and important trading system updates to the user's Telegram account.
"""

import os
import logging
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

# Setup logging
logger = logging.getLogger("TelegramAlerts")

class TelegramAlertManager:
    """
    Manages sending alerts to Telegram using the Telegram Bot API.
    
    This class handles formatting, prioritization, and delivery of alerts
    related to risk management, strategy rotation, and trading events.
    """
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the Telegram Alert Manager.
        
        Args:
            bot_token: Telegram Bot API token
            chat_id: Telegram chat ID to send messages to
            config_path: Path to config file containing Telegram credentials
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "telegram_config.json")
        
        # Load config if not provided as parameters
        if bot_token is None or chat_id is None:
            self._load_config()
        
        # Keep track of sent alerts to avoid duplicates in a short period
        self.recent_alerts = {}
        self.MAX_RECENT_ALERTS = 100
        
        # API endpoint for sending messages
        self.telegram_api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
    
    def _load_config(self) -> None:
        """Load Telegram configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.bot_token = config.get('bot_token')
                    self.chat_id = config.get('chat_id')
                    logger.info("Loaded Telegram configuration from file")
            else:
                # Try environment variables
                self.bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
                self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
                if self.bot_token and self.chat_id:
                    logger.info("Loaded Telegram configuration from environment variables")
                else:
                    logger.warning("No Telegram configuration found")
        except Exception as e:
            logger.error(f"Error loading Telegram configuration: {e}")
    
    def save_config(self) -> bool:
        """Save current Telegram configuration to file."""
        try:
            config_dir = os.path.dirname(self.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            with open(self.config_path, 'w') as f:
                json.dump({
                    'bot_token': self.bot_token,
                    'chat_id': self.chat_id
                }, f, indent=4)
            
            logger.info("Saved Telegram configuration to file")
            return True
        except Exception as e:
            logger.error(f"Error saving Telegram configuration: {e}")
            return False
    
    def set_credentials(self, bot_token: str, chat_id: str) -> bool:
        """
        Set Telegram credentials and save to config.
        
        Args:
            bot_token: Telegram Bot API token
            chat_id: Telegram chat ID to send messages to
            
        Returns:
            bool: Success status
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        # Update API endpoint
        self.telegram_api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        # Save to config file
        return self.save_config()
    
    def is_configured(self) -> bool:
        """Check if Telegram alerts are configured."""
        return bool(self.bot_token and self.chat_id)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the Telegram connection by sending a test message.
        
        Returns:
            dict: Result with success status and message
        """
        if not self.is_configured():
            return {"success": False, "message": "Telegram is not configured"}
        
        try:
            message = "🤖 BensBot Telegram Alert System - Test Message"
            response = self.send_alert(
                message=message,
                severity="info",
                alert_type="test",
                title="Connection Test"
            )
            
            if response.get("success"):
                return {"success": True, "message": "Telegram connection test successful"}
            else:
                return {"success": False, "message": f"Telegram test failed: {response.get('message')}"}
        except Exception as e:
            return {"success": False, "message": f"Error during Telegram test: {e}"}
    
    def send_alert(self, message: str, severity: str = "info", alert_type: str = "general", 
                 title: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                 deduplication_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Send an alert message to Telegram.
        
        Args:
            message: The main alert message
            severity: Alert severity (info, low, medium, high, critical)
            alert_type: The type of alert (general, risk, strategy, trade, system)
            title: Optional title for the alert
            details: Optional details dictionary
            deduplication_key: Optional key to prevent duplicate alerts
            
        Returns:
            dict: Result with success status and message
        """
        if not self.is_configured():
            return {"success": False, "message": "Telegram is not configured"}
        
        # Check for recent duplicates
        if deduplication_key:
            current_time = time.time()
            if deduplication_key in self.recent_alerts:
                # Only deduplicate if sent within the last 5 minutes
                if current_time - self.recent_alerts[deduplication_key] < 300:  # 5 minutes
                    return {"success": False, "message": "Duplicate alert suppressed"}
            
            # Update recent alerts (with cleanup if needed)
            self.recent_alerts[deduplication_key] = current_time
            if len(self.recent_alerts) > self.MAX_RECENT_ALERTS:
                # Remove oldest item
                oldest_key = min(self.recent_alerts, key=self.recent_alerts.get)
                del self.recent_alerts[oldest_key]
        
        # Format the message with severity and alert type indicators
        severity_icons = {
            "info": "ℹ️",
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴",
            "critical": "⚠️"
        }
        
        alert_type_icons = {
            "general": "📊",
            "risk": "🛡️",
            "strategy": "🔄",
            "trade": "💹",
            "system": "⚙️",
            "test": "🧪"
        }
        
        # Get icons based on severity and type
        severity_icon = severity_icons.get(severity.lower(), "ℹ️")
        type_icon = alert_type_icons.get(alert_type.lower(), "📊")
        
        # Format title
        title_text = f"{severity_icon} {type_icon} {title}" if title else f"{severity_icon} {type_icon} BensBot Alert"
        
        # Construct the full message
        full_message = f"{title_text}\n\n{message}"
        
        # Add details if provided
        if details:
            details_text = "\n\n📋 Details:\n"
            for key, value in details.items():
                details_text += f"- {key}: {value}\n"
            full_message += details_text
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message += f"\n\n⏱️ {timestamp}"
        
        # Send the message
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': full_message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(self.telegram_api_url, json=payload)
            result = response.json()
            
            if result.get('ok'):
                logger.info(f"Sent Telegram alert: {alert_type} - {severity}")
                return {"success": True, "message": "Alert sent successfully"}
            else:
                logger.error(f"Failed to send Telegram alert: {result}")
                return {"success": False, "message": f"Failed to send alert: {result.get('description')}"}
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def send_risk_alert(self, risk_type: str, risk_level: str, message: str, 
                      details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a risk-specific alert.
        
        Args:
            risk_type: Type of risk (liquidity, correlation, drawdown, etc.)
            risk_level: Risk level (low, medium, high, critical)
            message: Alert message
            details: Optional details dictionary
            
        Returns:
            dict: Result with success status and message
        """
        title = f"Risk Alert: {risk_type.title()}"
        
        # Map risk levels to severity
        severity_map = {
            "low": "low",
            "medium": "medium", 
            "high": "high",
            "critical": "critical"
        }
        severity = severity_map.get(risk_level.lower(), "medium")
        
        # Create deduplication key based on risk type and message
        dedup_key = f"risk_{risk_type}_{hash(message)}"
        
        return self.send_alert(
            message=message,
            severity=severity,
            alert_type="risk",
            title=title,
            details=details,
            deduplication_key=dedup_key
        )
    
    def send_strategy_rotation_alert(self, trigger: str, old_strategies: List[str], 
                                   new_strategies: List[str], reason: str) -> Dict[str, Any]:
        """
        Send an alert for strategy rotation events.
        
        Args:
            trigger: What triggered the rotation
            old_strategies: List of previously active strategies
            new_strategies: List of newly active strategies
            reason: Reason for rotation
            
        Returns:
            dict: Result with success status and message
        """
        title = "Strategy Rotation Alert"
        
        # Format the message
        message = f"Strategy rotation triggered by: {trigger}\n\n"
        message += f"Reason: {reason}\n\n"
        
        # Show strategy changes
        message += "Strategy Changes:\n"
        message += "➖ Deactivated: " + (", ".join(old_strategies) if old_strategies else "None") + "\n"
        message += "➕ Activated: " + (", ".join(new_strategies) if new_strategies else "None")
        
        # Create details dictionary
        details = {
            "Trigger": trigger,
            "Previous Strategies": ", ".join(old_strategies) if old_strategies else "None",
            "New Strategies": ", ".join(new_strategies) if new_strategies else "None",
            "Reason": reason
        }
        
        # Create deduplication key based on trigger and new strategies
        dedup_key = f"rotation_{trigger}_{'-'.join(sorted(new_strategies))}"
        
        return self.send_alert(
            message=message,
            severity="medium",  # Strategy rotations are medium severity by default
            alert_type="strategy",
            title=title,
            details=details,
            deduplication_key=dedup_key
        )
    
    def send_trade_alert(self, action: str, symbol: str, quantity: float, price: float,
                       trade_type: str, reason: str) -> Dict[str, Any]:
        """
        Send an alert for significant trades.
        
        Args:
            action: Buy or Sell
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            trade_type: Type of trade (market, limit, etc.)
            reason: Reason for the trade
            
        Returns:
            dict: Result with success status and message
        """
        # Format action for better visibility
        if action.lower() == "buy":
            action_format = "BUY 📈"
            severity = "info"
        else:
            action_format = "SELL 📉"
            severity = "info"
        
        title = f"Trade Alert: {action_format} {symbol}"
        
        # Format the message
        message = f"{action_format} {quantity} {symbol} @ ${price:.2f} ({trade_type.upper()})\n\n"
        message += f"Reason: {reason}"
        
        # Create details dictionary
        details = {
            "Symbol": symbol,
            "Action": action,
            "Quantity": str(quantity),
            "Price": f"${price:.2f}",
            "Type": trade_type,
            "Reason": reason
        }
        
        # Create deduplication key based on action, symbol and timestamp to prevent exact duplicates
        # We allow multiple trades of same symbol with a timestamp-based deduplication
        current_timestamp = int(time.time())
        dedup_key = f"trade_{action}_{symbol}_{current_timestamp // 60}"  # Per minute deduplication
        
        return self.send_alert(
            message=message,
            severity=severity,
            alert_type="trade",
            title=title,
            details=details,
            deduplication_key=dedup_key
        )
    
    def send_system_alert(self, component: str, status: str, message: str,
                        severity: str = "info") -> Dict[str, Any]:
        """
        Send an alert about system status.
        
        Args:
            component: System component name
            status: Component status
            message: Alert message
            severity: Alert severity
            
        Returns:
            dict: Result with success status and message
        """
        title = f"System Alert: {component}"
        
        # Format status for better visibility
        status_icons = {
            "online": "✅",
            "starting": "🔄",
            "warning": "⚠️",
            "error": "❌",
            "offline": "⛔",
            "unknown": "❓"
        }
        
        status_icon = status_icons.get(status.lower(), "❓")
        
        # Format the message
        formatted_message = f"Component: {component}\nStatus: {status_icon} {status.upper()}\n\n{message}"
        
        # Create deduplication key based on component and status
        dedup_key = f"system_{component}_{status}"
        
        return self.send_alert(
            message=formatted_message,
            severity=severity,
            alert_type="system",
            title=title,
            deduplication_key=dedup_key
        )

# Singleton instance
_telegram_alert_manager = None

def get_telegram_alert_manager() -> TelegramAlertManager:
    """Get the global Telegram Alert Manager instance."""
    global _telegram_alert_manager
    if _telegram_alert_manager is None:
        _telegram_alert_manager = TelegramAlertManager()
    return _telegram_alert_manager

# Convenience functions for sending alerts
def send_risk_alert(risk_type: str, risk_level: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send a risk alert via Telegram."""
    return get_telegram_alert_manager().send_risk_alert(risk_type, risk_level, message, details)

def send_strategy_rotation_alert(trigger: str, old_strategies: List[str], new_strategies: List[str], reason: str) -> Dict[str, Any]:
    """Send a strategy rotation alert via Telegram."""
    return get_telegram_alert_manager().send_strategy_rotation_alert(trigger, old_strategies, new_strategies, reason)

def send_trade_alert(action: str, symbol: str, quantity: float, price: float, trade_type: str, reason: str) -> Dict[str, Any]:
    """Send a trade alert via Telegram."""
    return get_telegram_alert_manager().send_trade_alert(action, symbol, quantity, price, trade_type, reason)

def send_system_alert(component: str, status: str, message: str, severity: str = "info") -> Dict[str, Any]:
    """Send a system alert via Telegram."""
    return get_telegram_alert_manager().send_system_alert(component, status, message, severity)

if __name__ == "__main__":
    # For testing the Telegram integration
    logging.basicConfig(level=logging.INFO)
    
    # Create a manager
    manager = TelegramAlertManager()
    
    if not manager.is_configured():
        print("Telegram is not configured. Please set bot_token and chat_id.")
        bot_token = input("Enter bot token: ")
        chat_id = input("Enter chat ID: ")
        
        if bot_token and chat_id:
            manager.set_credentials(bot_token, chat_id)
            print("Credentials saved.")
        else:
            print("Invalid credentials.")
            exit(1)
    
    # Test connection
    test_result = manager.test_connection()
    print(f"Test result: {test_result['message']}")
    
    if test_result['success']:
        # Send a test risk alert
        risk_result = send_risk_alert(
            risk_type="drawdown",
            risk_level="high",
            message="Portfolio drawdown has exceeded 10% threshold",
            details={"Current Drawdown": "12.5%", "Threshold": "10%", "Duration": "5 days"}
        )
        print(f"Risk alert result: {risk_result['message']}")
        
        # Send a test strategy rotation alert
        rotation_result = send_strategy_rotation_alert(
            trigger="Market Regime Change",
            old_strategies=["MomentumStrategy", "TrendFollowingStrategy"],
            new_strategies=["MeanReversionStrategy", "LowVolatilityStrategy"],
            reason="Market has transitioned from trending to mean-reverting regime"
        )
        print(f"Strategy rotation alert result: {rotation_result['message']}")
