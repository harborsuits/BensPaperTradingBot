"""
Daily Trade Review Module

This script runs daily to evaluate recent trades, provide AI-driven feedback,
and adjust strategy weights based on performance in the current market regime.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/daily_review.log')
    ]
)
logger = logging.getLogger(__name__)

# Import necessary modules
try:
    from trading_bot.journal.trade_journal import TradeJournal
    from trading_bot.ai_analysis.llm_trade_evaluator import LLMTradeEvaluator
    from trading_bot.market.regime_classifier import MarketRegimeClassifier
    from trading_bot.data.market_data_provider import MarketDataProvider
    from trading_bot.ai_analysis.feedback_learning import FeedbackLearningModule
    from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
    from trading_bot.config.config_manager import ConfigManager
    from trading_bot.alerts.notification_manager import NotificationManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    sys.exit(1)

class DailyTradeReviewer:
    """
    Reviews daily trades, evaluates performance, and adjusts strategy weights.
    
    This class integrates the trade journal, LLM evaluator, and market regime classifier
    to provide comprehensive feedback on recent trades and optimize strategy allocation.
    """
    
    def __init__(
        self,
        config_path: str,
        journal_dir: str,
        output_path: Optional[str] = None,
        review_date: Optional[str] = None,
        lookback_days: int = 1,
        api_key: Optional[str] = None,
        notification_config: Optional[Dict] = None
    ):
        """
        Initialize the daily trade reviewer.
        
        Args:
            config_path: Path to configuration file
            journal_dir: Path to trade journal directory
            output_path: Path to save updated strategy weights
            review_date: Date to review in YYYY-MM-DD format (defaults to yesterday)
            lookback_days: Number of days to look back for trades
            api_key: API key for LLM service
            notification_config: Configuration for notifications
        """
        # Set up dates
        if review_date:
            self.review_date = datetime.strptime(review_date, "%Y-%m-%d").date()
        else:
            self.review_date = datetime.now().date() - timedelta(days=1)
            
        self.lookback_days = lookback_days
        self.start_date = self.review_date - timedelta(days=self.lookback_days - 1)
        
        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.load_config()
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
        
        # Initialize components
        self.journal_dir = journal_dir
        self.output_path = output_path or os.path.join(
            os.path.dirname(config_path),
            f"strategy_weights_{self.review_date.strftime('%Y%m%d')}.json"
        )
        
        # Set up API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or self.config.get("api_key")
        if not self.api_key:
            logger.warning("No API key provided - LLM evaluation may be limited")
        
        # Initialize trade journal
        self.trade_journal = TradeJournal(self.journal_dir)
        
        # Initialize LLM evaluator
        self.llm_evaluator = LLMTradeEvaluator(
            api_key=self.api_key,
            model=self.config.get("llm_config", {}).get("model", "gpt-4"),
            use_mock=not self.api_key
        )
        
        # Initialize market data provider
        self.market_data_provider = MarketDataProvider(
            api_key=self.config.get("market_data", {}).get("api_key"),
            sources=self.config.get("market_data", {}).get("sources", ["yahoo"])
        )
        
        # Initialize regime classifier
        self.regime_classifier = MarketRegimeClassifier(
            config_path=self.config.get("regime_classifier", {}).get("config_path"),
            data_provider=self.market_data_provider
        )
        
        # Initialize feedback learning module
        self.feedback_module = FeedbackLearningModule(
            config_path=self.config.get("feedback_learning", {}).get("config_path")
        )
        
        # Initialize notification manager if notifications are configured
        self.notification_manager = None
        if notification_config or self.config.get("notifications"):
            self.notification_manager = NotificationManager(
                notification_config or self.config.get("notifications", {})
            )
        
        # Initialize strategy prioritizer 
        strategy_config = self.config.get("strategies", {})
        self.strategy_prioritizer = StrategyPrioritizer(
            strategies=strategy_config.get("strategies", []),
            api_key=self.api_key,
            use_mock=not self.api_key
        )
        
        # Storage for review results
        self.review_results = {
            "date": self.review_date.strftime("%Y-%m-%d"),
            "trades_reviewed": 0,
            "successful_trades": 0,
            "unsuccessful_trades": 0,
            "neutral_trades": 0,
            "total_pnl": 0.0,
            "trading_decisions": [],
            "market_regime": "",
            "strategy_adjustments": {},
            "key_observations": [],
            "improvement_suggestions": []
        }
        
        logger.info(f"Daily Trade Reviewer initialized for {self.review_date}")
    
    def run_daily_review(self) -> Dict:
        """
        Run the daily trade review process.
        
        Steps:
        1. Fetch recent trades from the journal
        2. Evaluate each trade using the LLM
        3. Calculate strategy performance metrics
        4. Identify current market regime
        5. Update strategy allocations
        6. Save the results and updated weights
        
        Returns:
            Dictionary with review results
        """
        logger.info(f"Starting daily review for {self.review_date}")
        
        # Fetch market and news data for context
        market_data = self._fetch_market_data()
        
        # Get trades for the review period
        trades = self._get_trades_to_review()
        
        if not trades:
            logger.info(f"No trades found for review period {self.start_date} to {self.review_date}")
            return self.review_results
        
        self.review_results["trades_reviewed"] = len(trades)
        logger.info(f"Found {len(trades)} trades to review")
        
        # Evaluate each trade
        evaluation_results = self._evaluate_trades(trades, market_data)
        
        # Update review results with evaluations
        self.review_results.update(evaluation_results)
        
        # Determine current market regime
        market_regime = self._determine_market_regime()
        self.review_results["market_regime"] = market_regime
        
        # Update strategy scores
        strategy_updates = self._update_strategy_scores(evaluation_results)
        self.review_results["strategy_adjustments"] = strategy_updates
        
        # Generate feedback notes
        self._generate_feedback_notes()
        
        # Save review results
        self._save_review_results()
        
        # Notify about review results if configured
        if self.notification_manager:
            self._send_review_notification()
        
        logger.info(f"Daily review completed for {self.review_date}")
        return self.review_results
    
    def _get_trades_to_review(self) -> List[Dict]:
        """
        Get trades for the review period from the trade journal.
        
        Returns:
            List of trade dictionaries
        """
        try:
            # Convert dates to strings for journal query
            start_date_str = self.start_date.strftime("%Y-%m-%d")
            end_date_str = self.review_date.strftime("%Y-%m-%d")
            
            # Query trades from journal
            trades = self.trade_journal.get_trades(
                start_date=start_date_str,
                end_date=end_date_str,
                include_open=False  # Only review closed trades
            )
            
            # Sort trades by exit time
            trades = sorted(trades, key=lambda x: x.get("exit_time", x.get("entry_time", "")))
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades from journal: {str(e)}")
            return []
    
    def _fetch_market_data(self) -> Dict:
        """
        Fetch market data for the review period.
        
        Returns:
            Dictionary with market data
        """
        try:
            # Get primary index data
            indices = ["SPY", "QQQ", "IWM", "VIX"]
            
            # Convert dates to strings for data provider
            start_date_str = (self.start_date - timedelta(days=5)).strftime("%Y-%m-%d")  # Get a few extra days for context
            end_date_str = self.review_date.strftime("%Y-%m-%d")
            
            # Get historical data
            market_data = {
                "indices": self.market_data_provider.get_historical_data(
                    symbols=indices,
                    start_date=start_date_str,
                    end_date=end_date_str
                ),
                "date": self.review_date.strftime("%Y-%m-%d")
            }
            
            # Get news data if available
            try:
                news_data = self.market_data_provider.get_market_news(
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                market_data["news"] = news_data
            except Exception as e:
                logger.warning(f"Error fetching news data: {str(e)}")
                market_data["news"] = []
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return {"indices": {}, "news": [], "date": self.review_date.strftime("%Y-%m-%d")}
    
    def _evaluate_trades(self, trades: List[Dict], market_data: Dict) -> Dict:
        """
        Evaluate trades using the LLM evaluator.
        
        Args:
            trades: List of trade dictionaries
            market_data: Dictionary with market data for context
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_results = {
            "successful_trades": 0,
            "unsuccessful_trades": 0,
            "neutral_trades": 0,
            "total_pnl": 0.0,
            "trading_decisions": [],
            "trades_by_strategy": {},
            "trades_by_symbol": {}
        }
        
        # Group trades by strategy and symbol for later analysis
        trades_by_strategy = {}
        trades_by_symbol = {}
        
        # Process each trade
        for trade in trades:
            try:
                # Get strategy ID
                strategy_id = trade.get("strategy_id", "unknown")
                symbol = trade.get("symbol", "unknown")
                
                # Track P&L
                pnl = trade.get("realized_pnl", 0.0)
                evaluation_results["total_pnl"] += pnl
                
                # Add market context to trade
                trade_with_context = self._add_market_context_to_trade(trade, market_data)
                
                # Evaluate the trade
                evaluation = self.llm_evaluator.evaluate_trade(trade_with_context)
                
                # Add evaluation to trade
                trade["evaluation"] = evaluation
                
                # Track evaluation outcome
                evaluation_outcome = evaluation.get("outcome", "neutral")
                if evaluation_outcome == "successful":
                    evaluation_results["successful_trades"] += 1
                elif evaluation_outcome == "unsuccessful":
                    evaluation_results["unsuccessful_trades"] += 1
                else:
                    evaluation_results["neutral_trades"] += 1
                
                # Add to trading decisions
                evaluation_results["trading_decisions"].append({
                    "trade_id": trade.get("trade_id"),
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "entry_time": trade.get("entry_time"),
                    "exit_time": trade.get("exit_time"),
                    "pnl": pnl,
                    "outcome": evaluation_outcome,
                    "quality_score": evaluation.get("quality_score", 5),
                    "key_issues": evaluation.get("key_issues", []),
                    "strengths": evaluation.get("strengths", [])
                })
                
                # Group by strategy
                if strategy_id not in trades_by_strategy:
                    trades_by_strategy[strategy_id] = {
                        "count": 0,
                        "win_count": 0,
                        "loss_count": 0,
                        "total_pnl": 0.0,
                        "avg_quality": 0.0,
                        "quality_scores": []
                    }
                
                trades_by_strategy[strategy_id]["count"] += 1
                trades_by_strategy[strategy_id]["total_pnl"] += pnl
                
                if evaluation_outcome == "successful":
                    trades_by_strategy[strategy_id]["win_count"] += 1
                elif evaluation_outcome == "unsuccessful":
                    trades_by_strategy[strategy_id]["loss_count"] += 1
                
                quality_score = evaluation.get("quality_score", 5)
                trades_by_strategy[strategy_id]["quality_scores"].append(quality_score)
                
                # Group by symbol
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = {
                        "count": 0,
                        "win_count": 0,
                        "loss_count": 0,
                        "total_pnl": 0.0
                    }
                
                trades_by_symbol[symbol]["count"] += 1
                trades_by_symbol[symbol]["total_pnl"] += pnl
                
                if evaluation_outcome == "successful":
                    trades_by_symbol[symbol]["win_count"] += 1
                elif evaluation_outcome == "unsuccessful":
                    trades_by_symbol[symbol]["loss_count"] += 1
                
                logger.info(f"Evaluated trade {trade.get('trade_id')}: {evaluation_outcome} with quality {quality_score}")
                
            except Exception as e:
                logger.error(f"Error evaluating trade {trade.get('trade_id')}: {str(e)}")
        
        # Calculate averages for each strategy
        for strategy_id, data in trades_by_strategy.items():
            if data["count"] > 0:
                data["win_rate"] = data["win_count"] / data["count"] if data["count"] > 0 else 0
                if data["quality_scores"]:
                    data["avg_quality"] = sum(data["quality_scores"]) / len(data["quality_scores"])
        
        # Calculate win rates for symbols
        for symbol, data in trades_by_symbol.items():
            if data["count"] > 0:
                data["win_rate"] = data["win_count"] / data["count"] if data["count"] > 0 else 0
        
        evaluation_results["trades_by_strategy"] = trades_by_strategy
        evaluation_results["trades_by_symbol"] = trades_by_symbol
        
        return evaluation_results
    
    def _add_market_context_to_trade(self, trade: Dict, market_data: Dict) -> Dict:
        """
        Add market context to trade for better evaluation.
        
        Args:
            trade: Trade dictionary
            market_data: Market data dictionary
            
        Returns:
            Trade with added market context
        """
        # Create a copy to avoid modifying the original
        enhanced_trade = trade.copy()
        
        try:
            # Get entry and exit dates
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            
            if not entry_time or not exit_time:
                return enhanced_trade
            
            # Convert to datetime if needed
            if isinstance(entry_time, str):
                entry_date = datetime.strptime(entry_time.split()[0], "%Y-%m-%d").date()
            else:
                entry_date = entry_time.date()
                
            if isinstance(exit_time, str):
                exit_date = datetime.strptime(exit_time.split()[0], "%Y-%m-%d").date()
            else:
                exit_date = exit_time.date()
            
            # Get market data for entry and exit dates
            market_context = {
                "market_conditions": {}
            }
            
            # Extract SPY data for market direction
            spy_data = market_data.get("indices", {}).get("SPY", pd.DataFrame())
            
            if not spy_data.empty:
                # Filter data for trade period
                entry_date_str = entry_date.strftime("%Y-%m-%d")
                exit_date_str = exit_date.strftime("%Y-%m-%d")
                
                # Get entry day data
                entry_day_data = spy_data[spy_data['date'] == entry_date_str]
                if not entry_day_data.empty:
                    market_context["market_conditions"]["entry_day"] = {
                        "open": entry_day_data['open'].iloc[0],
                        "close": entry_day_data['close'].iloc[0],
                        "high": entry_day_data['high'].iloc[0],
                        "low": entry_day_data['low'].iloc[0],
                        "volume": entry_day_data['volume'].iloc[0],
                        "change": entry_day_data['close'].iloc[0] / entry_day_data['open'].iloc[0] - 1
                    }
                
                # Get exit day data
                exit_day_data = spy_data[spy_data['date'] == exit_date_str]
                if not exit_day_data.empty:
                    market_context["market_conditions"]["exit_day"] = {
                        "open": exit_day_data['open'].iloc[0],
                        "close": exit_day_data['close'].iloc[0],
                        "high": exit_day_data['high'].iloc[0],
                        "low": exit_day_data['low'].iloc[0],
                        "volume": exit_day_data['volume'].iloc[0],
                        "change": exit_day_data['close'].iloc[0] / exit_day_data['open'].iloc[0] - 1
                    }
                
                # Get VIX data for volatility context
                vix_data = market_data.get("indices", {}).get("VIX", pd.DataFrame())
                if not vix_data.empty:
                    # Get entry day VIX
                    entry_vix = vix_data[vix_data['date'] == entry_date_str]
                    if not entry_vix.empty:
                        market_context["market_conditions"]["entry_day"]["vix"] = entry_vix['close'].iloc[0]
                    
                    # Get exit day VIX
                    exit_vix = vix_data[vix_data['date'] == exit_date_str]
                    if not exit_vix.empty:
                        market_context["market_conditions"]["exit_day"]["vix"] = exit_vix['close'].iloc[0]
            
            # Add relevant news if available
            if "news" in market_data and market_data["news"]:
                trade_news = []
                for news_item in market_data["news"]:
                    news_date = datetime.strptime(news_item.get("date", "1970-01-01"), "%Y-%m-%d").date()
                    # Include news from entry to exit date (inclusive)
                    if entry_date <= news_date <= exit_date:
                        # If the news mentions the traded symbol, add it
                        symbol = trade.get("symbol", "").upper()
                        if symbol in news_item.get("title", "") or symbol in news_item.get("summary", ""):
                            trade_news.append(news_item)
                
                # Limit to top 3 most relevant news items
                if trade_news:
                    market_context["relevant_news"] = trade_news[:3]
            
            # Add market context to trade
            enhanced_trade["market_context"] = market_context
            
        except Exception as e:
            logger.error(f"Error adding market context to trade: {str(e)}")
        
        return enhanced_trade
    
    def _determine_market_regime(self) -> str:
        """
        Determine the current market regime.
        
        Returns:
            Market regime string
        """
        try:
            # Use the regime classifier to determine the current regime
            market_regime = self.regime_classifier.classify_regime()
            logger.info(f"Current market regime: {market_regime}")
            return market_regime
            
        except Exception as e:
            logger.error(f"Error determining market regime: {str(e)}")
            return "neutral"  # Default to neutral regime on error
    
    def _update_strategy_scores(self, evaluation_results: Dict) -> Dict:
        """
        Update strategy scores based on evaluation results.
        
        Args:
            evaluation_results: Dictionary with evaluation results
            
        Returns:
            Dictionary with strategy updates
        """
        try:
            # Get current strategy weights
            current_weights = self.strategy_prioritizer.get_current_weights()
            
            # Get strategy performance data
            strategy_performance = evaluation_results.get("trades_by_strategy", {})
            
            # Calculate updates
            strategy_updates = {}
            
            for strategy_id, performance in strategy_performance.items():
                if strategy_id not in current_weights:
                    logger.warning(f"Strategy {strategy_id} not found in current weights - skipping")
                    continue
                
                current_weight = current_weights.get(strategy_id, 0.0)
                
                # Only update if we have sufficient data
                if performance.get("count", 0) < 2:
                    logger.info(f"Not enough trades for strategy {strategy_id} to update weight")
                    strategy_updates[strategy_id] = {
                        "previous_weight": current_weight,
                        "new_weight": current_weight,
                        "change": 0.0,
                        "reason": "Insufficient data for adjustment"
                    }
                    continue
                
                # Calculate adjustments based on win rate and quality
                win_rate = performance.get("win_rate", 0.0)
                avg_quality = performance.get("avg_quality", 5.0)
                
                # Normalize quality score to -1 to 1 range (5 is neutral)
                quality_modifier = (avg_quality - 5) / 5
                
                # Calculate weight adjustment
                if win_rate > 0.6:
                    # Increase weight for strong performers
                    adjustment = 0.1 + quality_modifier * 0.05
                elif win_rate < 0.4:
                    # Decrease weight for poor performers
                    adjustment = -0.1 + quality_modifier * 0.05
                else:
                    # Small adjustment based on quality for neutral performers
                    adjustment = quality_modifier * 0.05
                
                # Apply current market regime modifier
                market_regime = self.review_results.get("market_regime", "neutral")
                
                # Check if this strategy performs well in current regime
                strategy_config = self.config.get("strategies", {}).get("strategies", [])
                strategy_info = next((s for s in strategy_config if s.get("id") == strategy_id), {})
                
                # Get preferred regimes for this strategy
                preferred_regimes = strategy_info.get("preferred_regimes", [])
                
                if market_regime in preferred_regimes:
                    # Boost strategies that match current regime
                    regime_modifier = 0.05
                    logger.info(f"Strategy {strategy_id} matches current regime {market_regime} - applying boost")
                else:
                    # Slightly reduce strategies that don't match current regime
                    regime_modifier = -0.02
                
                total_adjustment = adjustment + regime_modifier
                
                # Apply adjustment (ensure weight stays between 0 and 1)
                new_weight = max(0.0, min(1.0, current_weight + total_adjustment))
                
                # Record update
                strategy_updates[strategy_id] = {
                    "previous_weight": current_weight,
                    "new_weight": new_weight,
                    "change": new_weight - current_weight,
                    "win_rate": win_rate,
                    "avg_quality": avg_quality,
                    "performance_adjustment": adjustment,
                    "regime_adjustment": regime_modifier,
                    "reason": self._generate_adjustment_reason(
                        win_rate, avg_quality, market_regime, strategy_id in preferred_regimes
                    )
                }
                
                logger.info(f"Updated weight for strategy {strategy_id}: {current_weight:.2f} -> {new_weight:.2f}")
            
            # Check for missing strategies in the update
            for strategy_id in current_weights:
                if strategy_id not in strategy_updates:
                    # Keep weight unchanged
                    current_weight = current_weights.get(strategy_id, 0.0)
                    strategy_updates[strategy_id] = {
                        "previous_weight": current_weight,
                        "new_weight": current_weight,
                        "change": 0.0,
                        "reason": "No trades executed with this strategy during review period"
                    }
            
            # Normalize weights to ensure they sum to 1
            total_weight = sum(update["new_weight"] for update in strategy_updates.values())
            
            if total_weight > 0:
                for strategy_id in strategy_updates:
                    strategy_updates[strategy_id]["new_weight"] /= total_weight
                    strategy_updates[strategy_id]["normalized_change"] = (
                        strategy_updates[strategy_id]["new_weight"] - 
                        strategy_updates[strategy_id]["previous_weight"]
                    )
            
            # Update strategy prioritizer with new weights
            new_weights = {
                strategy_id: update["new_weight"] 
                for strategy_id, update in strategy_updates.items()
            }
            
            # Save the normalized weights to the strategy prioritizer
            self.strategy_prioritizer.update_weights(new_weights)
            
            # Save updated weights to file
            self._save_updated_weights(new_weights)
            
            return strategy_updates
            
        except Exception as e:
            logger.error(f"Error updating strategy scores: {str(e)}")
            return {}
    
    def _generate_adjustment_reason(
        self, 
        win_rate: float, 
        quality: float, 
        market_regime: str, 
        is_preferred_regime: bool
    ) -> str:
        """
        Generate a human-readable reason for strategy adjustment.
        
        Args:
            win_rate: Strategy win rate
            quality: Average trade quality score
            market_regime: Current market regime
            is_preferred_regime: Whether current regime is preferred for this strategy
            
        Returns:
            Reason string
        """
        reasons = []
        
        # Win rate component
        if win_rate > 0.7:
            reasons.append(f"Excellent win rate ({win_rate:.0%})")
        elif win_rate > 0.6:
            reasons.append(f"Strong win rate ({win_rate:.0%})")
        elif win_rate < 0.3:
            reasons.append(f"Poor win rate ({win_rate:.0%})")
        elif win_rate < 0.4:
            reasons.append(f"Below average win rate ({win_rate:.0%})")
        else:
            reasons.append(f"Average win rate ({win_rate:.0%})")
        
        # Quality component
        if quality > 8:
            reasons.append("excellent trade execution")
        elif quality > 7:
            reasons.append("good trade execution")
        elif quality < 3:
            reasons.append("poor trade execution")
        elif quality < 4:
            reasons.append("below average trade execution")
        
        # Regime component
        if is_preferred_regime:
            reasons.append(f"well-suited for current {market_regime} regime")
        else:
            reasons.append(f"not optimized for current {market_regime} regime")
        
        return " with ".join(reasons)
    
    def _generate_feedback_notes(self) -> None:
        """
        Generate overall feedback and improvement suggestions.
        """
        try:
            # Extract key metrics
            trades_reviewed = self.review_results.get("trades_reviewed", 0)
            successful = self.review_results.get("successful_trades", 0)
            unsuccessful = self.review_results.get("unsuccessful_trades", 0)
            total_pnl = self.review_results.get("total_pnl", 0.0)
            
            if trades_reviewed == 0:
                self.review_results["key_observations"] = ["No trades to review for this period"]
                self.review_results["improvement_suggestions"] = ["Consider adjusting entry criteria to find more opportunities"]
                return
            
            # Calculate win rate
            win_rate = successful / trades_reviewed if trades_reviewed > 0 else 0
            
            # Generate key observations
            observations = []
            
            # Overall performance
            if win_rate > 0.6 and total_pnl > 0:
                observations.append(f"Strong overall performance with {win_rate:.0%} win rate and {total_pnl:.2f} P&L")
            elif win_rate < 0.4 or total_pnl < 0:
                observations.append(f"Underperformance with {win_rate:.0%} win rate and {total_pnl:.2f} P&L")
            else:
                observations.append(f"Mixed results with {win_rate:.0%} win rate and {total_pnl:.2f} P&L")
            
            # Strategy insights
            strategy_performance = self.review_results.get("trades_by_strategy", {})
            top_strategies = []
            struggling_strategies = []
            
            for strategy_id, perf in strategy_performance.items():
                if perf.get("count", 0) >= 2:
                    if perf.get("win_rate", 0) > 0.6 and perf.get("total_pnl", 0) > 0:
                        top_strategies.append(strategy_id)
                    elif perf.get("win_rate", 0) < 0.4 or perf.get("total_pnl", 0) < 0:
                        struggling_strategies.append(strategy_id)
            
            if top_strategies:
                observations.append(f"Top performing strategies: {', '.join(top_strategies)}")
            if struggling_strategies:
                observations.append(f"Underperforming strategies: {', '.join(struggling_strategies)}")
            
            # Market regime context
            market_regime = self.review_results.get("market_regime", "neutral")
            observations.append(f"Current market regime is {market_regime}")
            
            # Generate improvement suggestions
            suggestions = []
            
            # Strategy adjustments
            if struggling_strategies:
                suggestions.append(f"Review parameters for {', '.join(struggling_strategies)}")
            
            # Add specific suggestions based on trade evaluations
            common_issues = self._identify_common_issues()
            for issue, count in common_issues.items():
                if count >= 2:  # Only suggest if issue appears multiple times
                    suggestions.append(f"Address recurring issue: {issue}")
            
            # Add market regime suggestions
            if market_regime != "neutral":
                suggestions.append(f"Favor strategies suited for {market_regime} conditions")
            
            # Feed suggestions to the feedback learning module
            learned_suggestions = self.feedback_module.get_suggestions(
                self.review_results,
                market_regime
            )
            
            if learned_suggestions:
                suggestions.extend(learned_suggestions)
            
            self.review_results["key_observations"] = observations
            self.review_results["improvement_suggestions"] = suggestions
            
        except Exception as e:
            logger.error(f"Error generating feedback notes: {str(e)}")
            self.review_results["key_observations"] = ["Error generating observations"]
            self.review_results["improvement_suggestions"] = ["Review system logs for errors"]
    
    def _identify_common_issues(self) -> Dict[str, int]:
        """
        Identify common issues across all evaluated trades.
        
        Returns:
            Dictionary mapping issues to frequency counts
        """
        issues = {}
        
        try:
            # Extract issues from trade evaluations
            trading_decisions = self.review_results.get("trading_decisions", [])
            
            for decision in trading_decisions:
                key_issues = decision.get("key_issues", [])
                
                for issue in key_issues:
                    # Normalize issue text
                    normalized_issue = issue.lower().strip()
                    if normalized_issue in issues:
                        issues[normalized_issue] += 1
                    else:
                        issues[normalized_issue] = 1
            
            # Sort by frequency
            return dict(sorted(issues.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error identifying common issues: {str(e)}")
            return {}
    
    def _save_review_results(self) -> None:
        """
        Save the review results to a JSON file.
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Create results file path
            results_path = os.path.join(
                os.path.dirname(self.output_path),
                f"review_results_{self.review_date.strftime('%Y%m%d')}.json"
            )
            
            # Save results
            with open(results_path, 'w') as f:
                json.dump(self.review_results, f, indent=2)
                
            logger.info(f"Saved review results to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving review results: {str(e)}")
    
    def _save_updated_weights(self, weights: Dict[str, float]) -> None:
        """
        Save updated strategy weights to a JSON file.
        
        Args:
            weights: Dictionary mapping strategy IDs to weights
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save weights
            with open(self.output_path, 'w') as f:
                json.dump({
                    "date": self.review_date.strftime("%Y-%m-%d"),
                    "market_regime": self.review_results.get("market_regime", "neutral"),
                    "weights": weights
                }, f, indent=2)
                
            logger.info(f"Saved updated strategy weights to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error saving updated weights: {str(e)}")
    
    def _send_review_notification(self) -> None:
        """
        Send notifications about review results.
        """
        if not self.notification_manager:
            return
        
        try:
            # Create notification message
            message = f"Daily Trade Review for {self.review_date}\n"
            message += f"Market Regime: {self.review_results.get('market_regime', 'neutral')}\n"
            message += f"Trades Reviewed: {self.review_results.get('trades_reviewed', 0)}\n"
            message += f"Win Rate: {self.review_results.get('successful_trades', 0) / self.review_results.get('trades_reviewed', 1):.0%}\n"
            message += f"P&L: {self.review_results.get('total_pnl', 0.0):.2f}\n\n"
            
            # Add strategy adjustments
            message += "Strategy Adjustments:\n"
            for strategy_id, update in self.review_results.get("strategy_adjustments", {}).items():
                if abs(update.get("change", 0)) > 0.01:  # Only show significant changes
                    message += f"- {strategy_id}: {update.get('previous_weight', 0):.2f} -> {update.get('new_weight', 0):.2f}\n"
            
            # Add improvement suggestions
            message += "\nImprovement Suggestions:\n"
            for suggestion in self.review_results.get("improvement_suggestions", [])[:3]:  # Limit to top 3
                message += f"- {suggestion}\n"
            
            # Send notification
            self.notification_manager.send_notification(
                title=f"Daily Trade Review - {self.review_date}",
                message=message,
                level="info"
            )
            
            logger.info("Sent review notification")
            
        except Exception as e:
            logger.error(f"Error sending review notification: {str(e)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Daily Trade Review")
    
    parser.add_argument(
        "--date", 
        type=str,
        help="Date to review in YYYY-MM-DD format (defaults to yesterday)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/trading_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--journal", 
        type=str, 
        default="data/journal",
        help="Path to trade journal directory"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Path to save updated strategy weights"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LLM service (overrides config)"
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=1,
        help="Number of days to look back for trades"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the daily trade review."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Create reviewer
        reviewer = DailyTradeReviewer(
            config_path=args.config,
            journal_dir=args.journal,
            output_path=args.output,
            review_date=args.date,
            lookback_days=args.lookback,
            api_key=args.api_key
        )
        
        # Run review
        results = reviewer.run_daily_review()
        
        # Print summary
        print(f"\nDaily Trade Review Summary for {results['date']}")
        print(f"Market Regime: {results['market_regime']}")
        print(f"Trades Reviewed: {results['trades_reviewed']}")
        print(f"Success Rate: {results['successful_trades'] / results['trades_reviewed']:.0%} ({results['successful_trades']}/{results['trades_reviewed']})")
        print(f"Total P&L: {results['total_pnl']:.2f}")
        
        print("\nKey Observations:")
        for obs in results["key_observations"]:
            print(f"- {obs}")
        
        print("\nImprovement Suggestions:")
        for sugg in results["improvement_suggestions"]:
            print(f"- {sugg}")
        
        print("\nStrategy Adjustments:")
        for strategy_id, update in results["strategy_adjustments"].items():
            print(f"- {strategy_id}: {update['previous_weight']:.2f} -> {update['new_weight']:.2f} ({update.get('change', 0):.2f})")
        
    except Exception as e:
        logger.error(f"Error in daily trade review: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 