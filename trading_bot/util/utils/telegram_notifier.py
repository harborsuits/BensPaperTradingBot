#!/usr/bin/env python3
"""
Module for sending Telegram notifications about strategy allocation changes.
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Class for sending notifications via Telegram about strategy allocation changes."""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat ID to send messages to (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram notifications disabled: missing bot_token or chat_id")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram notifier initialized successfully")
    
    def send_allocation_update(self, 
                              old_allocations: Dict[str, float], 
                              new_allocations: Dict[str, float],
                              market_context: Optional[str] = None,
                              reasoning: Optional[str] = None) -> bool:
        """
        Send notification about strategy allocation changes.
        
        Args:
            old_allocations: Previous strategy allocations
            new_allocations: Updated strategy allocations
            market_context: Optional market context summary
            reasoning: Optional reasoning for allocation changes
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        if not self.enabled:
            logger.info("Telegram notifications disabled, skipping allocation update")
            return False
            
        # Construct message
        message = "🔄 *Strategy Allocation Update*\n\n"
        
        # Add market context if provided
        if market_context:
            message += f"*Market Context:*\n{market_context}\n\n"
        
        # Add allocation changes
        message += "*Allocation Changes:*\n"
        for strategy, new_alloc in new_allocations.items():
            old_alloc = old_allocations.get(strategy, 0.0)
            change = new_alloc - old_alloc
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "→"
            message += f"• {strategy}: {old_alloc:.1f}% {change_symbol} {new_alloc:.1f}% "
            if change != 0:
                message += f"({change:+.1f}%)"
            message += "\n"
        
        # Add reasoning if provided
        if reasoning:
            message += f"\n*Reasoning:*\n{reasoning}\n"
        
        return self.send_message(message)
    
    def send_message(self, message: str) -> bool:
        """
        Send a message via Telegram.
        
        Args:
            message: The message to send
            
        Returns:
            bool: Whether the message was sent successfully
        """
        if not self.enabled:
            return False
            
        try:
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(self.api_url, data=data)
            
            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram notification: {response.text}")
                return False
                
        except Exception as e:
            logger.exception(f"Error sending Telegram notification: {str(e)}")
            return False 