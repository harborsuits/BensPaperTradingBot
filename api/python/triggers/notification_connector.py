"""
Notification Connector for Market Intelligence

This module connects the market intelligence triggers to the notification system,
ensuring that important market events generate proper alerts across all configured
channels, with a focus on Telegram notifications.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union

# Import notification manager
from trading_bot.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger("NotificationConnector")

class NotificationConnector:
    """
    Connector class that links market intelligence events to the notification system.
    
    This class handles the formatting and routing of market intelligence alerts
    to the appropriate notification channels, with special handling for
    Telegram messages.
    """
    
    def __init__(self, telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
        """
        Initialize the notification connector.
        
        Args:
            telegram_token: Optional Telegram bot token (will use config if None)
            telegram_chat_id: Optional Telegram chat ID (will use config if None)
        """
        self.logger = logger
        
        # Create notification manager configuration
        notification_config = {
            "enabled": True,
            "min_level": "INFO",
            "slack": {
                "enabled": False  # Default to disabled
            },
            "desktop": {
                "enabled": True
            },
            "email": {
                "enabled": False  # Default to disabled
            }
        }
        
        # Add Telegram configuration
        if telegram_token and telegram_chat_id:
            notification_config["telegram"] = {
                "enabled": True,
                "token": telegram_token,
                "chat_id": telegram_chat_id
            }
        
        # Initialize the notification manager
        self.notification_manager = NotificationManager()
        
        # Update configuration with Telegram settings
        if telegram_token and telegram_chat_id:
            self.update_telegram_config(telegram_token, telegram_chat_id)
        
        self.logger.info("Notification connector initialized")
    
    def update_telegram_config(self, token: str, chat_id: str) -> None:
        """
        Update the Telegram configuration in the notification manager.
        
        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID
        """
        try:
            config = {
                "telegram": {
                    "enabled": True,
                    "token": token,
                    "chat_id": chat_id
                }
            }
            self.notification_manager.update_config(config)
            self.logger.info("Telegram configuration updated")
        except Exception as e:
            self.logger.error(f"Failed to update Telegram configuration: {str(e)}")
    
    def notify_market_update(self, update_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notification about a market update.
        
        Args:
            update_type: Type of update (market_data, symbol_data, full_update)
            details: Dictionary with update details
        
        Returns:
            Dictionary with notification results
        """
        # Create notification title and message
        title = f"Market Update: {update_type.replace('_', ' ').title()}"
        
        # Format message based on update type
        if update_type == "market_data":
            message = self._format_market_data_message(details)
        elif update_type == "symbol_data":
            message = self._format_symbol_data_message(details)
        elif update_type == "full_update":
            message = self._format_full_update_message(details)
        else:
            message = f"Market intelligence system updated {update_type}."
        
        # Add timestamp if available
        if "timestamp" in details:
            message += f"\nTimestamp: {details['timestamp']}"
        
        # Send notification
        return self.notification_manager.send_notification(
            title=title,
            message=message,
            level="INFO",
            metadata=details
        )
    
    def notify_market_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notification about a market event.
        
        Args:
            event_type: Type of event
            event_data: Dictionary with event data
        
        Returns:
            Dictionary with notification results
        """
        # Determine notification level based on event severity
        level = "INFO"
        if "severity" in event_data:
            severity = event_data["severity"].upper()
            if severity in ["HIGH", "CRITICAL"]:
                level = "WARNING"
            elif severity == "EXTREME":
                level = "CRITICAL"
        
        # Build title and message
        title = f"Market Event: {event_type.replace('_', ' ').title()}"
        message = self._format_event_message(event_type, event_data)
        
        # Send notification
        return self.notification_manager.send_notification(
            title=title,
            message=message,
            level=level,
            metadata=event_data
        )
    
    def notify_strategy_signal(self, strategy: str, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notification about a trading strategy signal.
        
        Args:
            strategy: Strategy name
            symbol: Symbol for the signal
            signal_data: Dictionary with signal data
        
        Returns:
            Dictionary with notification results
        """
        # Determine signal direction and notification level
        direction = signal_data.get("direction", "unknown")
        signal_type = signal_data.get("type", "unknown")
        
        level = "INFO"
        if signal_type in ["entry", "exit"]:
            level = "SUCCESS"
        elif signal_type == "warning":
            level = "WARNING"
        
        # Build title and message
        title = f"Strategy Signal: {strategy} - {symbol}"
        message = self._format_signal_message(strategy, symbol, signal_data)
        
        # Send notification
        return self.notification_manager.send_notification(
            title=title,
            message=message,
            level=level,
            metadata={
                "strategy": strategy,
                "symbol": symbol,
                **signal_data
            }
        )
    
    def notify_error(self, component: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send notification about an error in the market intelligence system.
        
        Args:
            component: Component where the error occurred
            error_message: Error message
            details: Optional details about the error
        
        Returns:
            Dictionary with notification results
        """
        # Build title and message
        title = f"Error in {component}"
        message = f"An error occurred in the {component} component: {error_message}"
        
        # Include details if provided
        metadata = details or {}
        
        # Send notification
        return self.notification_manager.send_notification(
            title=title,
            message=message,
            level="ERROR",
            metadata=metadata
        )
    
    def _format_market_data_message(self, details: Dict[str, Any]) -> str:
        """Format message for market data updates."""
        message = "Market data has been updated.\n\n"
        
        # Add market indices if available
        indices = details.get("indices", {})
        if indices:
            message += "Market Indices:\n"
            for index, value in indices.items():
                change = value.get("change_percent", 0)
                change_str = f"{change:+.2f}%" if isinstance(change, (int, float)) else change
                message += f"• {index}: {value.get('last', 'N/A')} ({change_str})\n"
        
        # Add market sentiment if available
        sentiment = details.get("sentiment")
        if sentiment:
            message += f"\nMarket Sentiment: {sentiment}"
        
        return message
    
    def _format_symbol_data_message(self, details: Dict[str, Any]) -> str:
        """Format message for symbol data updates."""
        symbols = details.get("symbols", [])
        if not symbols:
            return "Symbol data has been updated."
        
        # Basic message with symbols count
        message = f"Data updated for {len(symbols)} symbols.\n\n"
        
        # Add noteworthy symbols if available
        if "noteworthy" in details:
            message += "Noteworthy Symbols:\n"
            for symbol_info in details["noteworthy"][:5]:  # Limit to top 5
                symbol = symbol_info.get("symbol", "")
                reason = symbol_info.get("reason", "")
                message += f"• {symbol}: {reason}\n"
        
        return message
    
    def _format_full_update_message(self, details: Dict[str, Any]) -> str:
        """Format message for full system updates."""
        message = "Complete market intelligence update completed.\n\n"
        
        # Add market regime if available
        regime = details.get("market_regime")
        if regime:
            message += f"Current Market Regime: {regime}\n\n"
        
        # Add top strategies if available
        strategies = details.get("top_strategies", [])
        if strategies:
            message += "Top Performing Strategies:\n"
            for i, strategy in enumerate(strategies[:3], 1):  # Top 3 strategies
                score = strategy.get("score", 0)
                message += f"{i}. {strategy.get('name', 'Unknown')}: {score:.2f} score\n"
        
        return message
    
    def _format_event_message(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Format message for market events."""
        # Basic message with event description
        description = event_data.get("description", f"A {event_type} event has been detected.")
        message = f"{description}\n\n"
        
        # Add impact assessment if available
        impact = event_data.get("impact")
        if impact:
            message += f"Impact: {impact}\n"
        
        # Add recommendation if available
        recommendation = event_data.get("recommendation")
        if recommendation:
            message += f"\nRecommendation: {recommendation}"
        
        return message
    
    def _format_signal_message(self, strategy: str, symbol: str, signal_data: Dict[str, Any]) -> str:
        """Format message for strategy signals."""
        signal_type = signal_data.get("type", "unknown")
        direction = signal_data.get("direction", "unknown")
        
        # Basic message with signal info
        message = f"Strategy '{strategy}' generated a {signal_type} signal for {symbol}.\n\n"
        
        # Add price information if available
        price = signal_data.get("price")
        if price:
            message += f"Price: ${price:.2f}\n"
        
        # Add stop loss and target if available
        stop_loss = signal_data.get("stop_loss")
        if stop_loss:
            message += f"Stop Loss: ${stop_loss:.2f}\n"
        
        target = signal_data.get("target")
        if target:
            message += f"Target: ${target:.2f}\n"
        
        # Add confidence if available
        confidence = signal_data.get("confidence")
        if confidence:
            message += f"Confidence: {confidence:.2f}\n"
        
        # Add notes if available
        notes = signal_data.get("notes")
        if notes:
            message += f"\nNotes: {notes}"
        
        return message


# Singleton instance
_notification_connector = None

def get_notification_connector(telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
    """
    Get the singleton NotificationConnector instance.
    
    Args:
        telegram_token: Optional Telegram bot token (will use config if None)
        telegram_chat_id: Optional Telegram chat ID (will use config if None)
    
    Returns:
        NotificationConnector instance
    """
    global _notification_connector
    if _notification_connector is None:
        _notification_connector = NotificationConnector(telegram_token, telegram_chat_id)
    elif telegram_token and telegram_chat_id:
        # Update existing connector with new Telegram settings
        _notification_connector.update_telegram_config(telegram_token, telegram_chat_id)
    
    return _notification_connector
