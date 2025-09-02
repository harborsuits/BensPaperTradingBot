#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SlackNotifier: A class to handle sending notifications to Slack channels for a trading system.

This module provides functionality for sending various types of notifications 
to Slack channels, including trade alerts, error messages, daily summaries,
and performance reports.
"""

import os
import json
import time
import logging
import datetime
from typing import Dict, List, Optional, Any, Union
import requests
from urllib.parse import urljoin

class SlackNotifier:
    """
    Send notifications to Slack about trading events, allocation changes, and system status.
    
    This class provides methods to post messages to Slack channels, including formatted
    messages for trade alerts, strategy rotations, and system status updates.
    """
    
    def __init__(self, webhook_url: Optional[str] = None, channel: str = "#trading-alerts", 
                 username: str = "TradingBot", icon_emoji: str = ":chart_with_upwards_trend:",
                 log_level: int = logging.INFO):
        """
        Initialize the Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL for posting messages. If None, will check SLACK_WEBHOOK_URL env var.
            channel: Channel to post messages to.
            username: Bot username that will appear in Slack.
            icon_emoji: Emoji icon for the bot.
            log_level: Logging level.
        """
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.is_enabled = True
        
        # Verify webhook URL is available
        if not self.webhook_url:
            self.logger.warning("Slack webhook URL not provided. Notifications will be logged but not sent.")
            self.is_enabled = False
    
    def send_message(self, text: str, blocks: Optional[List[Dict[str, Any]]] = None, 
                     attachments: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Send a message to Slack.
        
        Args:
            text: Message text (fallback for clients that don't support blocks).
            blocks: Message blocks for rich formatting (optional).
            attachments: Message attachments (optional).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.is_enabled:
            self.logger.info(f"[SLACK DISABLED] Would have sent: {text}")
            return True
            
        payload = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "text": text
        }
        
        if blocks:
            payload["blocks"] = blocks
            
        if attachments:
            payload["attachments"] = attachments
        
        try:
            response = requests.post(
                self.webhook_url, 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to send message to Slack. Status: {response.status_code}, Response: {response.text}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Exception when sending Slack message: {str(e)}")
            return False
    
    def send_trade_alert(self, symbol: str, action: str, price: float, quantity: int, 
                         strategy: str, confidence: Optional[float] = None, 
                         reasoning: Optional[str] = None) -> bool:
        """
        Send a trade alert to Slack with formatted details.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL").
            action: Trade action ("BUY" or "SELL").
            price: Execution price.
            quantity: Number of shares/contracts.
            strategy: Strategy name that generated the trade.
            confidence: AI confidence score (0-100) if available.
            reasoning: AI reasoning for the trade if available.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Determine color based on action
        color = "#36a64f" if action.upper() == "BUY" else "#ff2b2b"  # Green for buy, red for sell
        
        # Format confidence if provided
        confidence_text = f" with {confidence:.1f}% confidence" if confidence is not None else ""
        
        # Create message text
        message_text = f"*TRADE ALERT:* {action.upper()} {quantity} {symbol} @ ${price:.2f} ({strategy}{confidence_text})"
        
        # Create message blocks for rich formatting
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Trade Alert: {action.upper()} {symbol}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{symbol}"},
                    {"type": "mrkdwn", "text": f"*Action:*\n{action.upper()}"},
                    {"type": "mrkdwn", "text": f"*Price:*\n${price:.2f}"},
                    {"type": "mrkdwn", "text": f"*Quantity:*\n{quantity}"},
                    {"type": "mrkdwn", "text": f"*Strategy:*\n{strategy}"}
                ]
            }
        ]
        
        # Add confidence if available
        if confidence is not None:
            blocks[1]["fields"].append(
                {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.1f}%"}
            )
            
        # Add reasoning if available
        if reasoning:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*AI Reasoning:*\n{reasoning}"}
            })
            
        # Add divider
        blocks.append({"type": "divider"})
        
        # Create attachment with color
        attachments = [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Total Value: ${price * quantity:.2f}"
                            }
                        ]
                    }
                ]
            }
        ]
        
        return self.send_message(message_text, blocks, attachments)
    
    def send_strategy_rotation_alert(self, old_allocations: Dict[str, float], 
                                     new_allocations: Dict[str, float], 
                                     market_regime: str,
                                     reasoning: Optional[str] = None) -> bool:
        """
        Send an alert about strategy rotation with allocation changes.
        
        Args:
            old_allocations: Dict mapping strategy names to previous allocation percentages.
            new_allocations: Dict mapping strategy names to new allocation percentages.
            market_regime: Current market regime (e.g., "Bullish", "Bearish", "Volatile").
            reasoning: AI reasoning for the allocation changes if available.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Determine color based on market regime
        regime_colors = {
            "Bullish": "#36a64f",  # Green
            "Bearish": "#ff2b2b",  # Red
            "Volatile": "#ffb700",  # Yellow
            "Neutral": "#6d6d6d"    # Gray
        }
        color = regime_colors.get(market_regime, "#36a64f")
        
        # Create changes summary
        changes = []
        for strategy in set(old_allocations.keys()) | set(new_allocations.keys()):
            old_value = old_allocations.get(strategy, 0)
            new_value = new_allocations.get(strategy, 0)
            diff = new_value - old_value
            
            if diff > 0:
                changes.append(f"{strategy}: {old_value:.1f}% → {new_value:.1f}% (+{diff:.1f}%)")
            elif diff < 0:
                changes.append(f"{strategy}: {old_value:.1f}% → {new_value:.1f}% ({diff:.1f}%)")
            else:
                changes.append(f"{strategy}: {new_value:.1f}% (unchanged)")
        
        changes_text = "\n".join(changes)
        
        # Create message text
        message_text = f"*STRATEGY ROTATION:* Market regime is {market_regime}"
        
        # Create message blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Strategy Rotation: {market_regime} Market",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Allocation Changes:*\n{changes_text}"}
            }
        ]
        
        # Add reasoning if available
        if reasoning:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*AI Reasoning:*\n{reasoning}"}
            })
            
        # Add divider
        blocks.append({"type": "divider"})
        
        # Create attachment with color
        attachments = [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Rotation completed at {self._get_current_time()}"
                            }
                        ]
                    }
                ]
            }
        ]
        
        return self.send_message(message_text, blocks, attachments)
    
    def send_system_alert(self, level: str, title: str, message: str, 
                          details: Optional[str] = None) -> bool:
        """
        Send a system alert notification.
        
        Args:
            level: Alert level ("info", "warning", "error", "critical").
            title: Alert title.
            message: Alert message.
            details: Additional details or debug information.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Determine color and emoji based on level
        level_info = {
            "info": {"color": "#36a64f", "emoji": ":information_source:"},
            "warning": {"color": "#ffb700", "emoji": ":warning:"},
            "error": {"color": "#ff2b2b", "emoji": ":x:"},
            "critical": {"color": "#ff2b2b", "emoji": ":rotating_light:"}
        }
        
        level_data = level_info.get(level.lower(), level_info["info"])
        
        # Create message text
        message_text = f"{level_data['emoji']} *{title.upper()}:* {message}"
        
        # Create blocks
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message_text}
            }
        ]
        
        # Add details if available
        if details:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Details:*\n```{details}```"}
            })
        
        # Create attachment with color
        attachments = [
            {
                "color": level_data["color"],
                "blocks": [
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Alert time: {self._get_current_time()}"
                            }
                        ]
                    }
                ]
            }
        ]
        
        return self.send_message(message_text, blocks, attachments)
    
    def send_performance_update(self, metrics: Dict[str, Union[float, Dict]], 
                                timeframe: str = "daily") -> bool:
        """
        Send performance metrics update.
        
        Args:
            metrics: Performance metrics (win rate, profit factor, etc.)
            timeframe: Timeframe for the metrics ("daily", "weekly", "monthly")
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Get main metrics if available
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        
        # Determine color based on profit factor
        if profit_factor >= 2.0:
            color = "#36a64f"  # Green
        elif profit_factor >= 1.5:
            color = "#3aa3e3"  # Blue
        elif profit_factor >= 1.0:
            color = "#ffb700"  # Yellow
        else:
            color = "#ff2b2b"  # Red
            
        # Create message text
        message_text = f"*{timeframe.upper()} PERFORMANCE UPDATE:* Win Rate: {win_rate:.1f}%, Profit Factor: {profit_factor:.2f}"
        
        # Create blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{timeframe.title()} Performance Update",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Win Rate:*\n{win_rate:.1f}%"},
                    {"type": "mrkdwn", "text": f"*Profit Factor:*\n{profit_factor:.2f}"},
                    {"type": "mrkdwn", "text": f"*Sharpe Ratio:*\n{sharpe:.2f}"},
                    {"type": "mrkdwn", "text": f"*Max Drawdown:*\n{metrics.get('max_drawdown', 0):.2f}%"}
                ]
            }
        ]
        
        # Add strategy performance if available
        if "strategies" in metrics:
            strategy_text = "*Strategy Performance:*\n"
            for strategy, perf in metrics["strategies"].items():
                strategy_text += f"• {strategy}: Win Rate {perf.get('win_rate', 0):.1f}%, P&L {perf.get('pnl', 0):.2f}%\n"
                
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": strategy_text}
            })
            
        # Add divider
        blocks.append({"type": "divider"})
        
        # Create attachment with color
        attachments = [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Performance period: {metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}"
                            }
                        ]
                    }
                ]
            }
        ]
        
        return self.send_message(message_text, blocks, attachments)
    
    def _get_current_time(self) -> str:
        """Get current time formatted for messages."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def disable(self) -> None:
        """Disable Slack notifications (will only log messages)."""
        self.is_enabled = False
        self.logger.info("Slack notifications disabled")
        
    def enable(self) -> None:
        """Enable Slack notifications."""
        if self.webhook_url:
            self.is_enabled = True
            self.logger.info("Slack notifications enabled")
        else:
            self.logger.warning("Cannot enable Slack notifications without a webhook URL")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize notifier
    notifier = SlackNotifier()
    
    # Example trade alert
    notifier.send_trade_alert(
        symbol="AAPL",
        action="BUY",
        price=182.45,
        quantity=100,
        strategy="Momentum",
        confidence=87.5,
        reasoning="Strong technical breakout with increasing volume and sector momentum."
    )
    
    # Example strategy rotation
    old_alloc = {"Momentum": 30, "Mean Reversion": 25, "Breakout": 15, 
                "Volatility": 10, "Trend Following": 15, "Pairs Trading": 5}
    new_alloc = {"Momentum": 20, "Mean Reversion": 15, "Breakout": 25, 
                "Volatility": 25, "Trend Following": 10, "Pairs Trading": 5}
    
    notifier.send_strategy_rotation_alert(
        old_allocations=old_alloc,
        new_allocations=new_alloc,
        market_regime="Volatile",
        reasoning="Increased market volatility favors breakout and volatility-based strategies."
    )
    
    # Example system alert
    notifier.send_system_alert(
        level="warning",
        title="API Connection Issue",
        message="Temporary connection issue with broker API.",
        details="TimeoutError: Request timed out after 15 seconds. Retrying in 60 seconds."
    )
    
    # Example performance update
    metrics = {
        "win_rate": 68.5,
        "profit_factor": 1.92,
        "sharpe_ratio": 1.45,
        "max_drawdown": -4.2,
        "start_date": "2023-04-01",
        "end_date": "2023-04-30",
        "strategies": {
            "Momentum": {"win_rate": 72.1, "pnl": 3.4},
            "Mean Reversion": {"win_rate": 65.2, "pnl": 2.1},
            "Breakout": {"win_rate": 58.9, "pnl": 1.5}
        }
    }
    
    notifier.send_performance_update(metrics, "monthly") 