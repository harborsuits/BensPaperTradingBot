#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Evaluation Notifier

This module connects the GPT-4 Trade Scorer with our notification system,
allowing trade evaluations to be sent via Telegram.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from trading_bot.ai_scoring.trade_scorer import TradeScorer
from trading_bot.notification_manager.telegram_notifier import TelegramNotifier


class TradeEvaluationNotifier:
    """
    Handles notifications for trade evaluations from the GPT-4 Trade Scorer.
    
    This class connects the Trade Scorer with the Telegram notification system,
    sending detailed notifications when trades are evaluated.
    """
    
    def __init__(
        self,
        telegram_notifier: TelegramNotifier,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trade evaluation notifier.
        
        Args:
            telegram_notifier: Initialized TelegramNotifier for sending messages
            config: Configuration parameters
        """
        self.telegram = telegram_notifier
        self.logger = logging.getLogger("TradeEvaluationNotifier")
        
        # Default configuration
        self.config = {
            'notify_all_evaluations': True,
            'min_confidence_to_notify': 0.0,
            'include_reasoning': True,
            'include_market_context': False,
            'include_strategy_performance': False,
            'enable_charts': True
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
            
        self.logger.info("Trade Evaluation Notifier initialized")
    
    def send_evaluation_notification(
        self,
        evaluation: Dict[str, Any],
        trade_data: Dict[str, Any],
        context_data: Optional[Dict[str, Any]] = None,
        strategy_perf: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a notification with trade evaluation results.
        
        Args:
            evaluation: The evaluation results from the Trade Scorer
            trade_data: Details of the trade being evaluated
            context_data: Optional market context data
            strategy_perf: Optional strategy performance data
            chat_id: Optional specific Telegram chat ID to send to
            
        Returns:
            Response from the Telegram API
        """
        # Skip notification if below minimum confidence threshold
        confidence = evaluation.get('confidence_score', 0)
        if confidence < self.config['min_confidence_to_notify'] and not self.config['notify_all_evaluations']:
            self.logger.info(f"Skipping notification for evaluation with confidence {confidence}")
            return {"skipped": True, "reason": "Below confidence threshold"}
        
        # Extract key information
        symbol = trade_data.get('symbol', 'UNKNOWN')
        strategy = trade_data.get('strategy_name', 'UNKNOWN')
        direction = trade_data.get('direction', 'UNKNOWN').upper()
        entry = trade_data.get('entry', 0.0)
        target = trade_data.get('target', 0.0)
        stop = trade_data.get('stop', 0.0)
        
        # Calculate risk/reward ratio if possible
        risk_reward = "N/A"
        if stop and entry and target and stop != entry:
            if direction.lower() == 'long':
                risk = entry - stop
                reward = target - entry
            else:  # short
                risk = stop - entry
                reward = entry - target
                
            if risk > 0:
                risk_reward = f"{reward/risk:.2f}"
        
        # Determine recommendation emoji
        recommendation = evaluation.get('recommendation', 'Unknown')
        if 'proceed' in recommendation.lower():
            rec_emoji = "✅"
        elif 'reduce' in recommendation.lower():
            rec_emoji = "⚠️"
        else:  # skip
            rec_emoji = "❌"
            
        # Determine bias alignment emoji
        bias = evaluation.get('bias_alignment', 'None')
        if bias.lower() == 'strong':
            bias_emoji = "🔥"
        elif bias.lower() == 'moderate':
            bias_emoji = "👍"
        elif bias.lower() == 'weak':
            bias_emoji = "👌"
        else:  # None
            bias_emoji = "⚖️"
            
        # Build the message
        message = f"<b>🧠 GPT-4 TRADE EVALUATION</b>\n\n"
        
        # Trade details
        message += f"<b>Symbol:</b> {symbol}\n"
        message += f"<b>Direction:</b> {'🟢 LONG' if direction.lower() == 'long' else '🔴 SHORT'}\n"
        message += f"<b>Strategy:</b> {strategy}\n"
        message += f"<b>Entry:</b> ${entry:,.2f}\n"
        
        if stop:
            message += f"<b>Stop:</b> ${stop:,.2f}\n"
        
        if target:
            message += f"<b>Target:</b> ${target:,.2f}\n"
            
        if risk_reward != "N/A":
            message += f"<b>Risk/Reward:</b> {risk_reward}\n"
        
        # Evaluation results
        message += f"\n<b>GPT-4 Evaluation:</b>\n"
        message += f"<b>Confidence:</b> {confidence:.1f}/10.0\n"
        message += f"<b>Bias Alignment:</b> {bias_emoji} {bias}\n"
        message += f"<b>Recommendation:</b> {rec_emoji} {recommendation}\n"
        
        # Include reasoning if configured
        if self.config['include_reasoning'] and 'reasoning' in evaluation:
            message += f"\n<b>Reasoning:</b>\n<i>{evaluation['reasoning']}</i>\n"
            
        # Include market context if configured
        if self.config['include_market_context'] and context_data:
            message += "\n<b>Market Context:</b>\n"
            
            # Add key market context data
            if 'market_regime' in context_data:
                message += f"• Market: {context_data['market_regime']}\n"
                
            if 'volatility_index' in context_data:
                message += f"• VIX: {context_data['volatility_index']}\n"
                
            # Add recent news if available
            if 'recent_news' in context_data and context_data['recent_news']:
                news = context_data['recent_news'][0]
                if isinstance(news, dict) and 'headline' in news:
                    message += f"• News: {news['headline']}\n"
        
        # Include strategy performance if configured
        if self.config['include_strategy_performance'] and strategy_perf:
            message += "\n<b>Strategy Performance:</b>\n"
            
            if 'win_rate' in strategy_perf:
                message += f"• Win Rate: {strategy_perf['win_rate']*100:.1f}%\n"
                
            if 'profit_factor' in strategy_perf:
                message += f"• Profit Factor: {strategy_perf['profit_factor']:.2f}\n"
        
        # Add timestamp
        message += f"\n<i>Evaluated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        # Send the message
        return self.telegram.send_message(message, chat_id, parse_mode="HTML")
    
    def send_trade_execution_notification(
        self,
        trade_data: Dict[str, Any],
        evaluation: Dict[str, Any],
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a notification when a trade is executed after evaluation.
        
        Args:
            trade_data: Details of the trade being executed
            evaluation: The evaluation results that led to execution
            chat_id: Optional specific Telegram chat ID to send to
            
        Returns:
            Response from the Telegram API
        """
        symbol = trade_data.get('symbol', 'UNKNOWN')
        direction = trade_data.get('direction', 'UNKNOWN').upper()
        confidence = evaluation.get('confidence_score', 0)
        
        # Determine trade type based on direction
        trade_type = "BUY" if direction.lower() == 'long' else "SELL"
        
        # Add GPT confidence to the trade alert
        return self.telegram.send_trade_alert(
            trade_type=trade_type,
            symbol=symbol,
            price=trade_data.get('entry', 0.0),
            quantity=trade_data.get('quantity', 0.0),
            strategy=f"{trade_data.get('strategy_name', 'Unknown')} (GPT: {confidence:.1f}/10)",
            chat_id=chat_id
        )


# Example usage
if __name__ == "__main__":
    # Initialize the Telegram notifier
    telegram = TelegramNotifier(
        bot_token="7950183254:AAH8UYP4ah7tNPJwsvAINlONIlv7dhsiCy4",
        default_chat_id="5723076356"  # User's Telegram chat ID
    )
    
    # Initialize the evaluation notifier
    notifier = TradeEvaluationNotifier(telegram)
    
    # Example trade data
    trade_data = {
        "symbol": "BTC/USD",
        "strategy_name": "breakout_swing",
        "direction": "long",
        "entry": 65000.0,
        "stop": 63500.0,
        "target": 68000.0,
        "quantity": 0.15
    }
    
    # Example evaluation from GPT-4
    evaluation = {
        "confidence_score": 8.3,
        "bias_alignment": "Strong",
        "reasoning": "This trade aligns with the current bullish market sentiment and increased volume. Bitcoin is showing strong momentum after breaking a key resistance level and the strategy has a high win rate in similar market conditions.",
        "recommendation": "Proceed with trade",
        "should_execute": True
    }
    
    # Example market context
    context = {
        "market_regime": "bullish",
        "volatility_index": 15.7,
        "recent_news": [
            {
                "headline": "Institutional demand for Bitcoin ETFs exceeds expectations",
                "sentiment": "positive"
            }
        ]
    }
    
    # Send a notification
    notifier.send_evaluation_notification(
        evaluation=evaluation,
        trade_data=trade_data,
        context_data=context
    )
    
    # If the trade is executed, send an execution notification
    if evaluation.get("should_execute", False):
        notifier.send_trade_execution_notification(
            trade_data=trade_data,
            evaluation=evaluation
        ) 