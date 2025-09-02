#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script showing how to use the GPT-4 Trade Scorer with Telegram notifications.

This script demonstrates how to:
1. Set up the LLM client
2. Initialize the trade scorer
3. Connect it to the Telegram notifier
4. Score a sample trade
5. Send notifications about the evaluation
"""

import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradeScoringExample")

# Import required components
from trading_bot.utils.llm_client import create_llm_client, MockLLMClient
from trading_bot.ai_scoring.trade_scorer import TradeScorer
from trading_bot.notification_manager.telegram_notifier import TelegramNotifier
from trading_bot.ai_scoring.trade_evaluation_notifier import TradeEvaluationNotifier


def main():
    """Run the trade scoring with notifications example."""
    logger.info("Starting trade scoring with notifications example")
    
    # Step 1: Create an LLM client
    # For this example, we'll use the mock client to avoid API costs
    llm_client = MockLLMClient(
        response_delay=1.0,  # Simulate 1 second processing time
        default_response=json.dumps({
            "confidence_score": 7.8,
            "bias_alignment": "Moderate",
            "reasoning": "This trade shows a strong setup with the price breaking above a key resistance level with increased volume. The RSI is not overbought yet, and there's room for further upside. The trade aligns well with the current bullish market regime, and the sector has been outperforming. The strategy has a good track record in similar market conditions.",
            "recommendation": "Proceed with trade"
        })
    )
    
    # Step 2: Set up the Telegram notifier
    telegram = TelegramNotifier(
        bot_token="7950183254:AAH8UYP4ah7tNPJwsvAINlONIlv7dhsiCy4",
        default_chat_id="5723076356",  # User's Telegram chat ID
        debug=True
    )
    
    # Step 3: Create the evaluation notifier
    eval_notifier = TradeEvaluationNotifier(
        telegram_notifier=telegram,
        config={
            'include_reasoning': True,
            'include_market_context': True,
            'include_strategy_performance': True
        }
    )
    
    # Step 4: Initialize the trade scorer with the notifier
    trade_scorer = TradeScorer(
        llm_client=llm_client,
        config={
            'confidence_threshold': 6.0,
            'temperature': 0.4,
            'auto_notify': True  # Enable automatic notifications
        },
        notifier=eval_notifier
    )
    
    # Step 5: Prepare sample trade data
    trade_data = {
        "symbol": "AAPL",
        "strategy_name": "breakout_momentum",
        "direction": "long",
        "entry": 186.50,
        "stop": 182.00,
        "target": 195.00,
        "timeframe": "4h",
        "setup_type": "cup_and_handle",
        "quantity": 10
    }
    
    context_data = {
        "market_regime": "bullish",
        "sector_performance": {"technology": 2.3},
        "volatility_index": 15.7,
        "recent_news": [
            {
                "headline": "Apple announces new AI features for iPhone",
                "sentiment": "positive",
                "date": "2023-06-05"
            }
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    strategy_perf = {
        "win_rate": 0.72,
        "profit_factor": 3.2,
        "average_win": 2.5,
        "average_loss": -1.2,
        "regime_performance": {
            "bullish": {"win_rate": 0.80, "trades": 25},
            "bearish": {"win_rate": 0.45, "trades": 12},
            "neutral": {"win_rate": 0.65, "trades": 18}
        }
    }
    
    # Step 6: Score the trade (this will automatically send notifications)
    logger.info(f"Scoring trade for {trade_data['symbol']}")
    evaluation = trade_scorer.score_trade(
        trade_data=trade_data,
        context_data=context_data,
        strategy_perf=strategy_perf
    )
    
    # Step 7: Determine if trade should be executed
    should_execute = trade_scorer.should_execute_trade(evaluation)
    logger.info(f"Trade recommendation: {evaluation['recommendation']}")
    logger.info(f"Should execute: {should_execute}")
    
    # Step 8: Log the evaluation
    trade_scorer.log_evaluation(evaluation, should_execute)
    
    logger.info("Trade scoring example completed")
    
    return evaluation


if __name__ == "__main__":
    main() 