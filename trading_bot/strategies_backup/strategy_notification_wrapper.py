"""
Strategy Notification Wrapper

This module provides a wrapper for trading strategies that adds notification capabilities.
It decorates strategy signals with notification functionality without modifying the original strategies.
"""

import logging
import datetime
from typing import Dict, Any, Optional, Callable, Union

from trading_bot.triggers.notification_connector import get_notification_connector

# Configure logging
logger = logging.getLogger("StrategyNotificationWrapper")

class StrategyNotificationWrapper:
    """
    Wrapper class that adds notification capabilities to any trading strategy.
    
    This wrapper intercepts the generate_signal method of strategies and
    sends notifications for significant signals via the notification system.
    """
    
    def __init__(self, strategy, telegram_token=None, telegram_chat_id=None):
        """
        Initialize the strategy notification wrapper.
        
        Args:
            strategy: The strategy object to wrap
            telegram_token: Optional Telegram bot token
            telegram_chat_id: Optional Telegram chat ID
        """
        self.strategy = strategy
        self.strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
        
        # Initialize notification connector
        self.notification_connector = get_notification_connector(telegram_token, telegram_chat_id)
        
        # Set up logging
        self.logger = logger
        
        # Remember original methods for delegation
        self._original_generate_signal = getattr(strategy, 'generate_signal', None)
        self._original_generate_signals = getattr(strategy, 'generate_signals', None)
        
        self.logger.info(f"Strategy notification wrapper initialized for {self.strategy_name}")
    
    def generate_signal(self, symbol, *args, **kwargs):
        """
        Generate a trading signal and send notification if significant.
        
        Args:
            symbol: The stock symbol
            *args: Arguments to pass to the original method
            **kwargs: Keyword arguments to pass to the original method
            
        Returns:
            Signal dictionary from the original strategy
        """
        # Call the original method if it exists
        if callable(self._original_generate_signal):
            signal = self._original_generate_signal(symbol, *args, **kwargs)
        elif callable(self._original_generate_signals):
            # Some strategies use generate_signals instead
            data = kwargs.get('data', {})
            if not data and symbol:
                # If data not provided but symbol is, create minimal data dict
                data = {'symbol': symbol}
            signal = self._original_generate_signals(data, *args, **kwargs)
        else:
            self.logger.warning(f"Strategy {self.strategy_name} has no signal generation method")
            return {'action': 'none', 'reason': 'No signal generation method available'}
        
        # If the signal is significant, send a notification
        if self._is_significant_signal(signal):
            try:
                notification_result = self.notification_connector.notify_strategy_signal(
                    self.strategy_name,
                    symbol,
                    {
                        "type": signal.get("action", "unknown"),
                        "direction": self._get_signal_direction(signal),
                        "price": signal.get("price"),
                        "stop_loss": signal.get("stop_loss"),
                        "target": signal.get("target"),
                        "confidence": signal.get("confidence"),
                        "notes": signal.get("reason") or signal.get("notes"),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                self.logger.info(f"Notification sent for {symbol} {signal.get('action', '')} signal")
            except Exception as e:
                self.logger.error(f"Error sending notification for {symbol} signal: {str(e)}")
        
        return signal
    
    def generate_signals(self, data, *args, **kwargs):
        """
        Generate signals for multiple symbols and send notifications if significant.
        
        Args:
            data: Data dictionary with market information
            *args: Arguments to pass to the original method
            **kwargs: Keyword arguments to pass to the original method
            
        Returns:
            Signals dictionary from the original strategy
        """
        # Call the original method if it exists
        if callable(self._original_generate_signals):
            signals = self._original_generate_signals(data, *args, **kwargs)
        elif callable(self._original_generate_signal):
            # If only generate_signal exists, try to adapt
            symbol = data.get('symbol')
            if symbol:
                signals = {symbol: self._original_generate_signal(symbol, *args, **kwargs)}
            else:
                self.logger.warning("Cannot generate signals: no symbol in data")
                return {}
        else:
            self.logger.warning(f"Strategy {self.strategy_name} has no signal generation methods")
            return {}
        
        # Send notifications for significant signals
        for symbol, signal in signals.items():
            if self._is_significant_signal(signal):
                try:
                    self.notification_connector.notify_strategy_signal(
                        self.strategy_name,
                        symbol,
                        {
                            "type": signal.get("action", "unknown"),
                            "direction": self._get_signal_direction(signal),
                            "price": signal.get("price"),
                            "stop_loss": signal.get("stop_loss"),
                            "target": signal.get("target"),
                            "confidence": signal.get("confidence"),
                            "notes": signal.get("reason") or signal.get("notes"),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                    self.logger.info(f"Notification sent for {symbol} {signal.get('action', '')} signal")
                except Exception as e:
                    self.logger.error(f"Error sending notification for {symbol} signal: {str(e)}")
        
        return signals
    
    def _is_significant_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Determine if a signal is significant enough to warrant a notification.
        
        Args:
            signal: The signal dictionary
            
        Returns:
            Boolean indicating if notification should be sent
        """
        # Get signal action
        action = signal.get('action', '').lower()
        
        # Always notify for entry and exit signals
        if action in ['buy', 'sell', 'long', 'short', 'exit']:
            return True
        
        # Notify for hold/warning signals only if confidence is high
        if action in ['hold', 'warning'] and signal.get('confidence', 0) > 0.75:
            return True
        
        # If the signal has a high urgency flag
        if signal.get('urgency', 'low').lower() in ['high', 'urgent', 'critical']:
            return True
        
        return False
    
    def _get_signal_direction(self, signal: Dict[str, Any]) -> str:
        """
        Determine the direction of a signal.
        
        Args:
            signal: The signal dictionary
            
        Returns:
            Direction string (buy, sell, hold)
        """
        action = signal.get('action', '').lower()
        
        if action in ['buy', 'long']:
            return 'buy'
        elif action in ['sell', 'short', 'exit']:
            return 'sell'
        else:
            return 'hold'
    
    def __getattr__(self, name):
        """
        Delegate method calls to the wrapped strategy.
        
        Args:
            name: Name of the attribute
            
        Returns:
            Attribute from the wrapped strategy
        """
        # Delegate all other method calls to the wrapped strategy
        return getattr(self.strategy, name)


def wrap_strategy_with_notifications(strategy, telegram_token=None, telegram_chat_id=None):
    """
    Wrap a strategy with notification capabilities.
    
    Args:
        strategy: Strategy object to wrap
        telegram_token: Optional Telegram bot token
        telegram_chat_id: Optional Telegram chat ID
        
    Returns:
        StrategyNotificationWrapper instance wrapping the strategy
    """
    return StrategyNotificationWrapper(strategy, telegram_token, telegram_chat_id)
