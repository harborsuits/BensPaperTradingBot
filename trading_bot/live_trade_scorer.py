#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trade Scoring System using GPT-4 evaluations.

This module integrates the GPT-4 trade evaluation system with the Telegram
notification system to provide real-time trade evaluations with notifications.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Import GPT-4 scoring components
from trading_bot.utils.llm_client import create_llm_client, MockLLMClient
from trading_bot.ai_scoring.trade_scorer import TradeScorer
from trading_bot.ai_scoring.trade_evaluation_notifier import TradeEvaluationNotifier
from trading_bot.notification_manager.telegram_notifier import TelegramNotifier


class LiveTradeScorer:
    """
    Live Trade Scoring System using GPT-4 evaluations.
    
    This class integrates the GPT-4 trade evaluation system with the Telegram
    notification system to provide real-time trade evaluations with notifications.
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        scorer_config: Optional[Dict[str, Any]] = None,
        telegram_config: Optional[Dict[str, Any]] = None,
        notifier_config: Optional[Dict[str, Any]] = None,
        use_mock: bool = False
    ):
        """
        Initialize the live trade scoring system.
        
        Args:
            llm_config: Configuration for the LLM client
            scorer_config: Configuration for the trade scorer
            telegram_config: Configuration for the Telegram notifier
            notifier_config: Configuration for the evaluation notifier
            use_mock: Whether to use a mock LLM client (for testing)
        """
        self.logger = logging.getLogger("LiveTradeScorer")
        
        # Load environment variables if not already loaded
        load_dotenv()
        
        # Get API keys and credentials from environment variables
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Check for required credentials
        if not telegram_token or not telegram_chat_id:
            self.logger.error("Telegram credentials not found in environment variables")
            raise ValueError("Telegram credentials not found in environment variables")
        
        if not openai_api_key and not use_mock:
            self.logger.warning("OpenAI API key not found in environment variables, using mock LLM client")
            use_mock = True
        
        # Initialize Telegram notifier
        telegram_config = telegram_config or {}
        self.telegram = TelegramNotifier(
            bot_token=telegram_config.get("bot_token", telegram_token),
            default_chat_id=telegram_config.get("default_chat_id", telegram_chat_id),
            debug=telegram_config.get("debug", True)
        )
        self.logger.info("Telegram notifier initialized")
        
        # Initialize LLM client
        if use_mock:
            self.logger.info("Using mock LLM client")
            self.llm_client = MockLLMClient(
                response_delay=0.5,
                default_response=json.dumps({
                    "confidence_score": 7.5,
                    "bias_alignment": "Moderate",
                    "reasoning": "This is a mock response for testing purposes.",
                    "recommendation": "Proceed with trade"
                })
            )
        else:
            llm_config = llm_config or {}
            self.llm_client = create_llm_client(
                provider=llm_config.get("provider", "openai"),
                api_key=llm_config.get("api_key", openai_api_key),
                model=llm_config.get("model", "gpt-4")
            )
        self.logger.info(f"LLM client initialized with {'mock' if use_mock else 'OpenAI API'}")
        
        # Initialize evaluation notifier
        notifier_config = notifier_config or {}
        self.evaluation_notifier = TradeEvaluationNotifier(
            telegram_notifier=self.telegram,
            config=notifier_config
        )
        self.logger.info("Trade evaluation notifier initialized")
        
        # Initialize trade scorer with notifier
        scorer_config = scorer_config or {}
        default_scorer_config = {
            'confidence_threshold': 6.0,
            'temperature': 0.4,
            'max_tokens': 800,
            'log_prompts': True,
            'log_responses': True,
            'auto_notify': True
        }
        default_scorer_config.update(scorer_config)
        
        self.trade_scorer = TradeScorer(
            llm_client=self.llm_client,
            config=default_scorer_config,
            notifier=self.evaluation_notifier
        )
        self.logger.info("Trade scorer initialized")
        
        # Send initialization notification
        self._send_initialization_notification()
    
    def _send_initialization_notification(self):
        """Send an initialization notification."""
        message = f"""🚀 <b>Live Trade Scoring System Initialized</b>

The GPT-4 trade evaluation system is now online and ready to evaluate trade setups.

<i>Initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        try:
            self.telegram.send_message(message)
            self.logger.info("Initialization notification sent")
        except Exception as e:
            self.logger.error(f"Failed to send initialization notification: {str(e)}")
    
    def evaluate_trade(
        self,
        trade_data: Dict[str, Any],
        context_data: Dict[str, Any],
        strategy_perf: Dict[str, Any],
        notify: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a trade setup using GPT-4.
        
        Args:
            trade_data: Trade setup data
            context_data: Market context data
            strategy_perf: Strategy performance data
            notify: Whether to send notifications
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating trade setup for {trade_data.get('symbol', 'unknown')}")
        
        # Score the trade
        try:
            evaluation = self.trade_scorer.score_trade(
                trade_data=trade_data,
                context_data=context_data,
                strategy_perf=strategy_perf,
                notify=notify
            )
            
            # Determine if trade should be executed
            should_execute = self.trade_scorer.should_execute_trade(evaluation)
            self.logger.info(
                f"Trade evaluation completed: Score={evaluation.get('confidence_score', 0):.1f}, "
                f"Execute={should_execute}"
            )
            
            # Log the evaluation
            self.trade_scorer.log_evaluation(evaluation, should_execute)
            
            return evaluation
        
        except Exception as e:
            self.logger.error(f"Error evaluating trade: {str(e)}")
            
            # Send error notification
            error_message = f"Error evaluating trade: {str(e)}"
            self.telegram.send_error_notification(
                error_message=error_message,
                error_type="Evaluation Error",
                importance="high"
            )
            
            # Return error evaluation
            return {
                'error': True,
                'error_message': str(e),
                'confidence_score': 0,
                'bias_alignment': 'None',
                'reasoning': f"Error: {str(e)}",
                'recommendation': 'Skip',
                'should_execute': False
            }
    
    def evaluate_multiple_trades(
        self,
        trades: List[Dict[str, Any]],
        context_data: Dict[str, Any],
        strategy_perf: Dict[str, Any],
        notify: bool = True,
        delay: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple trade setups sequentially.
        
        Args:
            trades: List of trade setups
            context_data: Market context data (can be shared or adapted for each trade)
            strategy_perf: Strategy performance data (can be shared or adapted for each trade)
            notify: Whether to send notifications
            delay: Delay between evaluations to avoid rate limits
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Evaluating {len(trades)} trade setups")
        
        results = []
        
        for i, trade_data in enumerate(trades):
            symbol = trade_data.get('symbol', 'unknown')
            self.logger.info(f"Evaluating trade {i+1}/{len(trades)}: {symbol}")
            
            # Add symbol-specific news if available
            symbol_context = self._add_symbol_context(context_data.copy(), symbol)
            
            # Get strategy-specific performance if available
            strategy = trade_data.get('strategy_name', 'unknown')
            strategy_specific_perf = self._get_strategy_specific_perf(strategy_perf, strategy)
            
            # Evaluate the trade
            evaluation = self.evaluate_trade(
                trade_data=trade_data,
                context_data=symbol_context,
                strategy_perf=strategy_specific_perf,
                notify=notify
            )
            
            results.append({
                'trade_data': trade_data,
                'evaluation': evaluation
            })
            
            # Add delay between evaluations (except for the last one)
            if i < len(trades) - 1 and delay > 0:
                time.sleep(delay)
        
        self.logger.info(f"Completed evaluation of {len(trades)} trade setups")
        
        # Send summary notification
        if notify and len(trades) > 1:
            self._send_batch_summary_notification(results)
        
        return results
    
    def _add_symbol_context(self, context_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Add symbol-specific context to market context data.
        
        This is a placeholder implementation. In a real system, this would
        fetch news, fundamental data, or other symbol-specific information.
        
        Args:
            context_data: Base market context data
            symbol: The trading symbol
            
        Returns:
            Enhanced context data
        """
        # This is just a simple example with hardcoded data
        # In a real system, this would fetch data from an API
        news_by_symbol = {
            "AAPL": [
                {
                    "headline": "Apple announces new AI features for iPhone",
                    "sentiment": "positive",
                    "relevance": 0.95,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "MSFT": [
                {
                    "headline": "Microsoft cloud growth exceeds expectations",
                    "sentiment": "positive",
                    "relevance": 0.9,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "TSLA": [
                {
                    "headline": "Tesla faces production challenges at German factory",
                    "sentiment": "negative",
                    "relevance": 0.85,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        if symbol in news_by_symbol:
            # Add the news to the context data
            if 'recent_news' not in context_data:
                context_data['recent_news'] = []
                
            context_data['recent_news'].extend(news_by_symbol[symbol])
        
        return context_data
    
    def _get_strategy_specific_perf(
        self,
        strategy_perf: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """
        Get strategy-specific performance metrics.
        
        Args:
            strategy_perf: Performance metrics for all strategies
            strategy: The specific strategy name
            
        Returns:
            Strategy-specific performance metrics
        """
        # Check if we have specific performance for this strategy
        if strategy in strategy_perf:
            return strategy_perf[strategy]
        
        # Otherwise return the general performance data
        return strategy_perf
    
    def _send_batch_summary_notification(self, results: List[Dict[str, Any]]):
        """
        Send a summary notification for a batch of trade evaluations.
        
        Args:
            results: List of evaluation results
        """
        if not results:
            return
        
        # Analyze results
        approved_count = sum(1 for r in results if r['evaluation'].get('should_execute', False))
        rejected_count = len(results) - approved_count
        
        # Calculate average confidence
        confidences = [r['evaluation'].get('confidence_score', 0) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Build message
        message = f"""📊 <b>Trade Evaluation Batch Summary</b>

<b>Evaluated:</b> {len(results)} trade setups
<b>Approved:</b> {approved_count}
<b>Rejected:</b> {rejected_count}
<b>Avg. Confidence:</b> {avg_confidence:.1f}/10.0

<b>Approved Trades:</b>
"""
        
        # Add details of approved trades
        approved_trades = [r for r in results if r['evaluation'].get('should_execute', False)]
        if approved_trades:
            for result in approved_trades:
                trade = result['trade_data']
                eval_data = result['evaluation']
                
                message += f"• {trade.get('symbol')}: {eval_data.get('confidence_score', 0):.1f}/10.0 ({trade.get('direction', 'unknown')})\n"
        else:
            message += "None\n"
        
        message += f"\n<i>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        try:
            self.telegram.send_message(message)
            self.logger.info("Batch summary notification sent")
        except Exception as e:
            self.logger.error(f"Failed to send batch summary notification: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the live trade scorer
    scorer = LiveTradeScorer(
        use_mock=True  # Use mock LLM client for testing
    )
    
    # Sample trade data
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
    
    # Sample market context
    context_data = {
        "market_regime": "bullish",
        "sector_performance": {"technology": 2.3},
        "volatility_index": 15.7,
        "recent_news": [
            {
                "headline": "Fed signals potential rate cut in next meeting",
                "sentiment": "positive",
                "relevance": 0.9,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    
    # Sample strategy performance
    strategy_perf = {
        "win_rate": 0.72,
        "profit_factor": 3.2,
        "avg_win": 2.5,
        "avg_loss": -1.2,
        "regime_performance": {
            "bullish": {"win_rate": 0.80, "trades": 25},
            "bearish": {"win_rate": 0.45, "trades": 12}
        }
    }
    
    # Evaluate a trade
    evaluation = scorer.evaluate_trade(
        trade_data=trade_data,
        context_data=context_data,
        strategy_perf=strategy_perf
    )
    
    # Print the results
    print(f"Confidence Score: {evaluation.get('confidence_score', 0):.1f}/10.0")
    print(f"Recommendation: {evaluation.get('recommendation', 'unknown')}")
    print(f"Should Execute: {evaluation.get('should_execute', False)}")
    
    # Example of evaluating multiple trades
    multiple_trades = [
        {
            "symbol": "AAPL",
            "strategy_name": "breakout_momentum",
            "direction": "long",
            "entry": 186.50,
            "stop": 182.00,
            "target": 195.00,
            "timeframe": "4h",
            "setup_type": "cup_and_handle",
            "quantity": 10
        },
        {
            "symbol": "MSFT",
            "strategy_name": "momentum",
            "direction": "long",
            "entry": 378.25,
            "stop": 370.50,
            "target": 395.00,
            "timeframe": "1d",
            "setup_type": "trend_continuation",
            "quantity": 5
        },
        {
            "symbol": "TSLA",
            "strategy_name": "mean_reversion",
            "direction": "long",
            "entry": 215.30,
            "stop": 210.00,
            "target": 230.00,
            "timeframe": "1h",
            "setup_type": "oversold_bounce",
            "quantity": 8
        }
    ]
    
    # Evaluate multiple trades
    results = scorer.evaluate_multiple_trades(
        trades=multiple_trades,
        context_data=context_data,
        strategy_perf=strategy_perf,
        delay=1.0  # 1 second delay between evaluations
    ) 