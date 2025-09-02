#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News Trading Strategy Module

This module implements strategies for exploiting short-lived volatility and 
directional moves around scheduled news catalysts using sentiment analysis
and disciplined execution.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import re
import time
import math
from concurrent.futures import ThreadPoolExecutor

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Setup logging
logger = logging.getLogger(__name__)

class NewsTradingStrategy(StrategyOptimizable):
    """
    News Trading Strategy designed to exploit short-lived volatility and directional moves.
    
    This strategy reacts to scheduled and unscheduled news events by analyzing sentiment
    and executing trades with precise timing and strict risk management.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize News Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # 1. Strategy Philosophy parameters
            "strategy_name": "news_trading",
            "strategy_version": "1.0.0",
            
            # 2. Market Universe & Event Selection parameters
            "universe": ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "JPM", "EUR/USD", "BTC/USD"],
            "event_types": ["macro", "earnings", "regulatory", "merger", "unscheduled"],
            "priority_events": ["FOMC", "CPI", "NFP", "GDP", "earnings", "FDA_approval"],
            "update_calendar_interval_hours": 24,
            
            # 3. Data & News Sources parameters
            "news_sources": ["bloomberg", "reuters", "wsj", "marketwatch", "twitter_verified"],
            "latency_target_ms": 500,
            "max_acceptable_latency_ms": 1000,
            "use_fallback_feeds": True,
            
            # 4. Sentiment & Signal Generation parameters
            "sentiment_model_path": "models/news_sentiment_transformer.pkl",
            "min_sentiment_threshold": 0.65,
            "strong_sentiment_threshold": 0.85,
            "min_surprise_threshold": 0.10,  # 10% difference from consensus
            "max_model_age_days": 90,  # Model retraining frequency
            
            # 5. Position Sizing & Risk Control parameters
            "directional_risk_per_event_pct": 0.005,  # 0.5% of equity
            "volatility_risk_per_event_pct": 0.01,    # 1.0% of equity
            "max_concurrent_news_trades": 3,
            "max_exposure_pct": 0.10,  # 10% of equity
            "drawdown_stop_pct": 0.01,  # 1% of equity
            
            # 6. Entry Rules parameters
            "order_delay_ms": 200,  # Submit order 200ms after signal
            "directional_order_type": "limit",
            "volatility_order_type": "market",
            "limit_buffer_ticks": 1,
            "dynamic_slippage_adjustment": True,
            
            # 7. Exit Rules parameters
            "directional_profit_target_pct": 0.008,  # 0.8% move
            "volatility_profit_target_pct": 0.40,    # 40% premium decay capture
            "directional_stop_loss_pct": 0.004,      # 0.4% adverse move
            "volatility_stop_loss_pct": 0.50,        # 50% premium loss
            "time_exit_minutes": 15,                 # Force exit after 15 minutes
            
            # 8. Execution Guidelines parameters
            "use_dual_broker": True,
            "failover_timeout_ms": 300,
            "max_order_size": 100,  # Max contracts per order batch
            "order_chunk_size": 10,  # Size of order chunks for large orders
            "post_trade_check_delay_ms": 50,
            
            # 9. Backtesting & Performance Metrics parameters
            "backtest_years": 2,
            "min_hit_rate": 0.55,  # Minimum acceptable hit rate
            "max_acceptable_slippage_bps": 5,  # 5 basis points
            "max_latency_impact_bps": 3,  # 3 basis points of P&L decay
            
            # 10. Continuous Optimization parameters
            "sentiment_model_retrain_days": 90,
            "threshold_adjustment_frequency_days": 30,
            "latency_alert_threshold_ms": 700,
            "sentiment_accuracy_alert_threshold": 0.80  # 80% accuracy minimum
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize event calendar
        self.event_calendar = self._load_event_calendar()
        
        # Initialize sentiment model
        self.sentiment_model = self._load_sentiment_model()
        
        # Initialize performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "event_type_performance": {},
            "avg_latency_ms": 0,
            "avg_slippage_bps": 0
        }
        
        logger.info(f"Initialized News Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "min_sentiment_threshold": [0.55, 0.60, 0.65, 0.70, 0.75],
            "strong_sentiment_threshold": [0.80, 0.85, 0.90],
            "min_surprise_threshold": [0.05, 0.10, 0.15, 0.20],
            "directional_risk_per_event_pct": [0.003, 0.005, 0.007, 0.01],
            "volatility_risk_per_event_pct": [0.005, 0.01, 0.015],
            "order_delay_ms": [100, 200, 300, 400],
            "directional_profit_target_pct": [0.005, 0.008, 0.01, 0.015],
            "directional_stop_loss_pct": [0.003, 0.004, 0.005],
            "time_exit_minutes": [10, 15, 20, 30]
        }
    
    # === 1. Strategy Philosophy Implementation ===
    def _evaluate_market_conditions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Evaluate current market conditions to adjust strategy parameters.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price data
            
        Returns:
            Dictionary of market condition assessments
        """
        conditions = {}
        
        for symbol, df in data.items():
            if len(df) < 20:
                continue
                
            # Calculate recent volatility
            df['returns'] = df['close'].pct_change()
            recent_vol = df['returns'].tail(20).std() * np.sqrt(252)
            
            # Determine volatility regime
            if recent_vol > 0.30:
                vol_regime = "high_volatility"
            elif recent_vol < 0.15:
                vol_regime = "low_volatility"
            else:
                vol_regime = "normal_volatility"
                
            # Determine trend direction (simple SMA-based assessment)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            latest_close = df['close'].iloc[-1]
            latest_sma20 = df['sma_20'].iloc[-1]
            latest_sma50 = df['sma_50'].iloc[-1]
            
            if latest_close > latest_sma20 and latest_sma20 > latest_sma50:
                trend = "bullish"
            elif latest_close < latest_sma20 and latest_sma20 < latest_sma50:
                trend = "bearish"
            else:
                trend = "neutral"
                
            conditions[symbol] = {
                "volatility": vol_regime,
                "trend": trend,
                "recent_volatility": recent_vol,
            }
        
        return conditions
    
    # === 2. Market Universe & Event Selection ===
    def _load_event_calendar(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load and prepare the economic and corporate event calendar.
        
        Returns:
            Dictionary mapping event types to lists of event details
        """
        # TODO: Implement calendar fetching from external source
        # This would typically connect to a data provider API
        
        # For now, return a placeholder calendar with sample events
        today = datetime.now().date()
        calendar = {
            "macro": [
                {
                    "event_name": "FOMC Rate Decision",
                    "event_date": today + timedelta(days=5),
                    "event_time": "14:00:00",
                    "impact": "high",
                    "consensus": "5.25-5.50%",
                    "previous": "5.25-5.50%",
                    "related_symbols": ["SPY", "QQQ", "TLT", "UUP"]
                },
                {
                    "event_name": "US CPI",
                    "event_date": today + timedelta(days=3),
                    "event_time": "08:30:00",
                    "impact": "high",
                    "consensus": "3.1% y/y",
                    "previous": "3.2% y/y",
                    "related_symbols": ["SPY", "TLT", "GLD"]
                }
            ],
            "earnings": [
                {
                    "event_name": "AAPL Earnings",
                    "event_date": today + timedelta(days=10),
                    "event_time": "16:30:00",
                    "impact": "high",
                    "consensus": "1.58 EPS",
                    "previous": "1.52 EPS",
                    "related_symbols": ["AAPL", "QQQ", "XLK"]
                }
            ],
            "regulatory": [],
            "merger": [],
            "unscheduled": []
        }
        
        logger.info(f"Loaded event calendar with {sum(len(events) for events in calendar.values())} upcoming events")
        return calendar
    
    def update_event_calendar(self):
        """
        Update the event calendar with the latest scheduled events.
        
        This method should be called periodically to refresh the calendar.
        """
        # TODO: Implement calendar update logic
        # This would typically call an API to get the latest calendar data
        
        last_updated = getattr(self, "calendar_last_updated", None)
        now = datetime.now()
        
        if (last_updated is None or 
            (now - last_updated).total_seconds() > self.parameters["update_calendar_interval_hours"] * 3600):
            
            try:
                self.event_calendar = self._load_event_calendar()
                self.calendar_last_updated = now
                logger.info("Successfully updated event calendar")
            except Exception as e:
                logger.error(f"Failed to update event calendar: {e}")
    
    def filter_universe(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter universe to symbols with upcoming events or news sensitivity.
        
        Args:
            data: Dictionary of all available symbols with their data
            
        Returns:
            Filtered dictionary of symbols for news trading
        """
        filtered_data = {}
        universe = self.parameters.get("universe", [])
        
        # Update event calendar if needed
        self.update_event_calendar()
        
        # Get symbols with upcoming events
        symbols_with_events = set()
        for event_type, events in self.event_calendar.items():
            for event in events:
                event_date = event.get("event_date")
                if event_date and (event_date - datetime.now().date()).days <= 3:
                    symbols = event.get("related_symbols", [])
                    symbols_with_events.update(symbols)
        
        # Filter data to symbols in universe with adequate liquidity
        for symbol, df in data.items():
            if symbol not in universe:
                continue
                
            # Give priority to symbols with upcoming events
            if symbol in symbols_with_events:
                filtered_data[symbol] = df
                continue
                
            # Check liquidity for other symbols
            if 'volume' in df.columns:
                avg_volume = df['volume'].tail(20).mean()
                if avg_volume >= 1000000:  # At least 1M average daily volume
                    filtered_data[symbol] = df
        
        logger.info(f"Filtered universe contains {len(filtered_data)} symbols for news trading")
        return filtered_data

    # === 3. Data & News Sources ===
    def _load_sentiment_model(self):
        """
        Load the sentiment analysis model.
        
        Returns:
            Sentiment model object
        """
        # TODO: Implement sentiment model loading
        # This would typically load a pre-trained machine learning model
        
        model_path = self.parameters.get("sentiment_model_path")
        logger.info(f"Loading sentiment model from {model_path}")
        
        # Placeholder model - in production, you would load an actual ML model
        class PlaceholderSentimentModel:
            def predict(self, text, timestamp=None):
                """Predict sentiment for a news headline or article"""
                # Simple rule-based sentiment analysis for placeholder
                positive_words = ["increase", "rise", "grow", "beat", "exceed", "positive", "bullish", "up", "gain"]
                negative_words = ["decrease", "fall", "drop", "miss", "negative", "bearish", "down", "cut", "loss"]
                
                text = text.lower()
                pos_count = sum(word in text for word in positive_words)
                neg_count = sum(word in text for word in negative_words)
                
                # Generate sentiment score between -1 and 1
                if pos_count > neg_count:
                    score = min(0.9, 0.5 + 0.1 * (pos_count - neg_count))
                elif neg_count > pos_count:
                    score = max(-0.9, -0.5 - 0.1 * (neg_count - pos_count))
                else:
                    score = 0.0
                    
                # Add some randomness for realistic simulation
                score += np.random.normal(0, 0.1)
                score = max(-1.0, min(1.0, score))
                
                latency = np.random.randint(5, 50)  # Simulated processing time in ms
                
                return {
                    "sentiment": "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral",
                    "score": score,
                    "confidence": abs(score) * 0.8 + 0.2,
                    "processing_time_ms": latency
                }
        
        return PlaceholderSentimentModel()
    
    def _process_news_data(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess, deduplicate, and enrich news data.
        
        Args:
            news_items: List of raw news items
            
        Returns:
            List of processed news items
        """
        # Deduplication dictionary using headline as key
        unique_items = {}
        
        for item in news_items:
            # Create a key for deduplication
            headline = item.get("headline", "")
            source = item.get("source", "")
            key = f"{headline.lower().strip()}_{source}"
            
            # If item already exists, take the one with earliest timestamp
            if key in unique_items:
                existing_ts = unique_items[key].get("timestamp", datetime.max)
                current_ts = item.get("timestamp", datetime.max)
                
                if current_ts < existing_ts:
                    unique_items[key] = item
            else:
                unique_items[key] = item
        
        # Process each unique item
        processed_items = []
        for item in unique_items.values():
            # Map headline to tickers if not already present
            if "symbols" not in item or not item["symbols"]:
                item["symbols"] = self._map_headline_to_tickers(item.get("headline", ""))
            
            # Add timestamp if missing
            if "timestamp" not in item:
                item["timestamp"] = datetime.now()
                
            # Normalize source
            if "source" in item:
                item["source"] = item["source"].lower()
                
            # Add event type classification if missing
            if "event_type" not in item:
                item["event_type"] = self._classify_news_event_type(item)
                
            processed_items.append(item)
        
        # Sort by timestamp (most recent first)
        processed_items.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        
        return processed_items
    
    def _map_headline_to_tickers(self, headline: str) -> List[str]:
        """
        Map a news headline to relevant ticker symbols.
        
        Args:
            headline: News headline text
            
        Returns:
            List of ticker symbols mentioned or relevant to the headline
        """
        # TODO: Implement more sophisticated headline-to-ticker mapping
        # This would typically use a more complex entity extraction algorithm
        
        # Simple implementation: check for symbols in our universe
        universe = self.parameters.get("universe", [])
        mentioned_symbols = []
        
        for symbol in universe:
            # Direct symbol mention
            if re.search(r'\b' + re.escape(symbol) + r'\b', headline, re.IGNORECASE):
                mentioned_symbols.append(symbol)
                
        # If no direct mentions, check for common company names
        if not mentioned_symbols:
            company_to_ticker = {
                "apple": "AAPL",
                "microsoft": "MSFT",
                "amazon": "AMZN",
                "google": "GOOG",
                "alphabet": "GOOGL",
                "meta": "META",
                "facebook": "META",
                "nvidia": "NVDA",
                "federal reserve": "SPY",
                "fed": "SPY",
                "fomc": "SPY",
                "s&p": "SPY",
                "dow": "DIA",
                "nasdaq": "QQQ"
            }
            
            for company, ticker in company_to_ticker.items():
                if re.search(r'\b' + re.escape(company) + r'\b', headline, re.IGNORECASE):
                    if ticker in universe:
                        mentioned_symbols.append(ticker)
        
        # For macro news, add market indices
        macro_keywords = ["gdp", "inflation", "cpi", "unemployment", "jobs", "nfp", "fomc", "interest rate", "fed", "treasury"]
        if any(keyword in headline.lower() for keyword in macro_keywords):
            macro_symbols = [s for s in ["SPY", "QQQ", "DIA", "IWM"] if s in universe]
            for symbol in macro_symbols:
                if symbol not in mentioned_symbols:
                    mentioned_symbols.append(symbol)
        
        return mentioned_symbols
    
    def _classify_news_event_type(self, news_item: Dict[str, Any]) -> str:
        """
        Classify a news item into an event type.
        
        Args:
            news_item: News item data
            
        Returns:
            Event type classification
        """
        headline = news_item.get("headline", "").lower()
        content = news_item.get("content", "").lower()
        text = headline + " " + content
        
        # Check for earnings-related news
        earnings_keywords = ["earnings", "eps", "revenue", "profit", "quarterly", "q1", "q2", "q3", "q4"]
        if any(keyword in text for keyword in earnings_keywords):
            return "earnings"
        
        # Check for macro news
        macro_keywords = ["fed", "fomc", "interest rate", "inflation", "cpi", "gdp", "jobs", "nfp", "unemployment", "treasury"]
        if any(keyword in text for keyword in macro_keywords):
            return "macro"
        
        # Check for regulatory news
        regulatory_keywords = ["fda", "approval", "regulation", "regulatory", "sec", "lawsuit", "settlement", "fine"]
        if any(keyword in text for keyword in regulatory_keywords):
            return "regulatory"
        
        # Check for merger/acquisition news
        merger_keywords = ["merger", "acquisition", "takeover", "buyout", "acquiring", "acquire", "acquired"]
        if any(keyword in text for keyword in merger_keywords):
            return "merger"
        
        # Default to unscheduled
        return "unscheduled"
    
    # === 4. Sentiment & Signal Generation ===
    def analyze_sentiment(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the sentiment of a news item.
        
        Args:
            news_item: News item data
            
        Returns:
            Sentiment analysis results
        """
        headline = news_item.get("headline", "")
        content = news_item.get("content", "")
        timestamp = news_item.get("timestamp")
        
        # Combine headline and content for analysis
        # Give more weight to the headline by repeating it
        text = headline + " " + headline + " " + content
        
        # Start sentiment analysis timer
        start_time = time.time()
        
        # Get sentiment from model
        sentiment_result = self.sentiment_model.predict(text, timestamp)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Add processing time to result
        sentiment_result["processing_time_ms"] = processing_time_ms
        
        # Check if latency exceeds target
        latency_target = self.parameters.get("latency_target_ms", 500)
        if processing_time_ms > latency_target:
            logger.warning(f"Sentiment analysis latency ({processing_time_ms:.2f} ms) exceeds target ({latency_target} ms)")
        
        # Add normalized sentiment score (-1 to 1 scale)
        if "score" not in sentiment_result:
            if sentiment_result.get("sentiment") == "positive":
                sentiment_result["score"] = 0.7
            elif sentiment_result.get("sentiment") == "negative":
                sentiment_result["score"] = -0.7
            else:
                sentiment_result["score"] = 0.0
        
        return sentiment_result
    
    def check_surprise_factor(self, news_item: Dict[str, Any], event_data: Dict[str, Any]) -> float:
        """
        Calculate the surprise factor of news relative to expectations.
        
        Args:
            news_item: News item data
            event_data: Event data with consensus expectations
            
        Returns:
            Surprise factor as a percentage (-1.0 to 1.0)
        """
        # Not all news has a quantifiable surprise factor
        if not event_data or not news_item:
            return 0.0
            
        # Try to extract actual value from news
        headline = news_item.get("headline", "")
        content = news_item.get("content", "")
        text = headline + " " + content
        
        # Get consensus from event data
        consensus = event_data.get("consensus", "")
        previous = event_data.get("previous", "")
        
        # Placeholder for extraction logic
        # In a real implementation, this would use NLP to extract the actual value
        
        # For now, use a simple regex for common formats
        # Example: "Unemployment Rate at 3.7% vs 3.5% expected"
        actual = None
        
        def extract_percentage(text):
            matches = re.findall(r'(\d+\.\d+)%', text)
            return [float(m) for m in matches] if matches else None
            
        def extract_numeric(text):
            matches = re.findall(r'(\d+\.\d+)', text)
            return [float(m) for m in matches] if matches else None
        
        # Try to extract percentage values
        percentages = extract_percentage(text)
        if percentages and len(percentages) >= 1:
            actual = percentages[0]
            
            # Try to extract consensus as percentage
            consensus_values = extract_percentage(consensus)
            if consensus_values and len(consensus_values) >= 1:
                consensus_value = consensus_values[0]
                
                # Calculate surprise factor
                if consensus_value != 0:
                    surprise = (actual - consensus_value) / consensus_value
                else:
                    surprise = 0.0 if actual == 0 else 1.0 if actual > 0 else -1.0
                
                # Normalize to -1.0 to 1.0 range
                surprise = max(-1.0, min(1.0, surprise))
                return surprise
        
        # Default: no surprise factor detected
        return 0.0
    
    def generate_news_signal(self, news_item: Dict[str, Any], sentiment: Dict[str, Any], 
                           market_conditions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a trading signal based on news sentiment and market conditions.
        
        Args:
            news_item: News item data
            sentiment: Sentiment analysis results
            market_conditions: Market condition assessment for relevant symbols
            
        Returns:
            Signal details if a valid signal is generated, None otherwise
        """
        # Check if news item has associated symbols
        symbols = news_item.get("symbols", [])
        if not symbols:
            return None
            
        # Get sentiment score and check against threshold
        sentiment_score = sentiment.get("score", 0)
        min_threshold = self.parameters.get("min_sentiment_threshold", 0.65)
        strong_threshold = self.parameters.get("strong_sentiment_threshold", 0.85)
        
        # If sentiment isn't strong enough, no signal
        if abs(sentiment_score) < min_threshold:
            return None
            
        # Get the event type
        event_type = news_item.get("event_type", "unscheduled")
        
        # Get surprise factor if it's a scheduled event
        surprise_factor = 0.0
        if event_type in ["macro", "earnings"]:
            # Find matching event in calendar
            event_name = news_item.get("headline", "").lower()
            for event in self.event_calendar.get(event_type, []):
                if event.get("event_name", "").lower() in event_name:
                    surprise_factor = self.check_surprise_factor(news_item, event)
                    break
        
        # Determine signal type based on sentiment direction
        signal_type = None
        if sentiment_score > 0:
            signal_type = "BUY"
        elif sentiment_score < 0:
            signal_type = "SELL"
        else:
            return None
            
        # Use first symbol for signal creation
        # In a more advanced implementation, you might create signals for multiple symbols
        primary_symbol = symbols[0]
        
        # Determine confidence based on sentiment strength and surprise factor
        base_confidence = abs(sentiment_score)
        surprise_boost = abs(surprise_factor) * 0.3  # Up to 30% boost from surprise
        confidence = min(0.99, base_confidence + surprise_boost)
        
        # Determine trading approach based on sentiment strength and event type
        if event_type == "earnings" and abs(sentiment_score) < strong_threshold:
            # For earnings with moderate sentiment, consider volatility approach (straddle)
            approach = "volatility"
        else:
            # For strong sentiment or other events, use directional approach
            approach = "directional"
        
        # Create signal
        signal = {
            "symbol": primary_symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "approach": approach,
            "event_type": event_type,
            "sentiment_score": sentiment_score,
            "surprise_factor": surprise_factor,
            "news_timestamp": news_item.get("timestamp"),
            "generation_timestamp": datetime.now(),
            "latency_ms": sentiment.get("processing_time_ms", 0),
            "related_symbols": symbols,
            "headline": news_item.get("headline")
        }
        
        logger.info(f"Generated {signal_type} signal for {primary_symbol} with {confidence:.2f} confidence")
        return signal

    # === 5. Position Sizing & Risk Controls ===
    def _calculate_position_size(self, symbol: str, signal: Dict[str, Any], 
                               account_value: float, current_price: float) -> int:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Ticker symbol
            signal: Trading signal details
            account_value: Current account value
            current_price: Current price of the symbol
            
        Returns:
            Position size (number of shares/contracts)
        """
        # Determine risk per trade based on approach
        approach = signal.get("approach", "directional")
        
        if approach == "directional":
            risk_pct = self.parameters.get("directional_risk_per_event_pct", 0.005)
        else:  # volatility approach
            risk_pct = self.parameters.get("volatility_risk_per_event_pct", 0.01)
            
        # Adjust risk based on confidence
        confidence = signal.get("confidence", 0.5)
        confidence_factor = 0.5 + (confidence * 0.5)  # 0.5-1.0 range
        risk_pct = risk_pct * confidence_factor
        
        # Calculate dollar risk
        dollar_risk = account_value * risk_pct
        
        # Determine stop loss distance
        stop_loss_pct = self.parameters.get("directional_stop_loss_pct", 0.004)
        stop_loss_distance = current_price * stop_loss_pct
        
        # Calculate position size
        if stop_loss_distance > 0:
            # Risk per share = stop loss distance
            shares = math.floor(dollar_risk / stop_loss_distance)
        else:
            # Fallback if stop loss can't be calculated
            shares = math.floor(dollar_risk / (current_price * 0.01))  # Risk 1% of price
            
        # Check against max exposure
        max_exposure_pct = self.parameters.get("max_exposure_pct", 0.10)
        max_exposure_dollars = account_value * max_exposure_pct
        max_shares = math.floor(max_exposure_dollars / current_price)
        
        # Take the smaller value
        shares = min(shares, max_shares)
        
        # Ensure we have at least 1 share if trading
        if shares <= 0:
            shares = 0
        
        logger.info(f"Calculated position size for {symbol}: {shares} shares (${(shares * current_price):.2f})")
        return shares
    
    def _check_risk_limits(self, active_positions: Dict[str, Dict[str, Any]], 
                          new_signal: Dict[str, Any], account_value: float) -> bool:
        """
        Check if new trade would violate risk limits.
        
        Args:
            active_positions: Dictionary of active positions
            new_signal: New trading signal to be evaluated
            account_value: Current account value
            
        Returns:
            Boolean indicating if new trade is within risk limits
        """
        # Check max concurrent news trades limit
        max_concurrent = self.parameters.get("max_concurrent_news_trades", 3)
        if len(active_positions) >= max_concurrent:
            logger.warning(f"Max concurrent news trades limit reached ({max_concurrent})")
            return False
            
        # Check drawdown stop
        strategy_pnl = sum(pos.get("unrealized_pnl", 0) for pos in active_positions.values())
        drawdown_limit = account_value * self.parameters.get("drawdown_stop_pct", 0.01)
        
        if strategy_pnl <= -drawdown_limit:
            logger.warning(f"News trading on hold due to drawdown limit (${drawdown_limit:.2f})")
            return False
            
        # Check correlation with existing positions
        new_symbol = new_signal.get("symbol")
        
        # List of active position symbols
        active_symbols = [pos.get("symbol") for pos in active_positions.values()]
        
        # Check for duplicate symbol
        if new_symbol in active_symbols:
            logger.warning(f"Already have active position in {new_symbol}")
            return False
            
        # Check if any active symbols are in related_symbols of new signal
        related_symbols = new_signal.get("related_symbols", [])
        for symbol in active_symbols:
            if symbol in related_symbols:
                logger.warning(f"Correlated position already exists ({symbol} vs {new_symbol})")
                return False
                
        # Check exposure limit
        total_exposure = sum(pos.get("exposure", 0) for pos in active_positions.values())
        max_exposure = account_value * self.parameters.get("max_exposure_pct", 0.10)
        
        if total_exposure >= max_exposure:
            logger.warning(f"Max exposure limit reached (${max_exposure:.2f})")
            return False
            
        return True
    
    def _calculate_exit_prices(self, signal: Dict[str, Any], entry_price: float) -> Dict[str, float]:
        """
        Calculate exit prices for a signal.
        
        Args:
            signal: Trading signal details
            entry_price: Entry price of the position
            
        Returns:
            Dictionary with stop loss, take profit, and time exit details
        """
        approach = signal.get("approach", "directional")
        signal_type = signal.get("signal_type")
        
        if approach == "directional":
            # Set stop loss and take profit based on directional parameters
            stop_loss_pct = self.parameters.get("directional_stop_loss_pct", 0.004)
            profit_target_pct = self.parameters.get("directional_profit_target_pct", 0.008)
            
            if signal_type == "BUY":
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + profit_target_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - profit_target_pct)
                
        else:  # volatility approach
            # For volatility trades (e.g., straddles), percentage-based on premium
            stop_loss_pct = self.parameters.get("volatility_stop_loss_pct", 0.50)
            profit_target_pct = self.parameters.get("volatility_profit_target_pct", 0.40)
            
            # For simplicity, treat entry_price as premium cost for volatility trades
            stop_loss = entry_price * (1 + stop_loss_pct)  # Maximum acceptable loss
            take_profit = entry_price * (1 - profit_target_pct)  # Target value after decay
        
        # Time exit (in minutes)
        time_exit_minutes = self.parameters.get("time_exit_minutes", 15)
        time_exit = (datetime.now() + timedelta(minutes=time_exit_minutes)).timestamp()
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "time_exit": time_exit
        }
    
    # === 6. Entry Rules ===
    def _prepare_entry_order(self, signal: Dict[str, Any], current_price: float, 
                           position_size: int) -> Dict[str, Any]:
        """
        Prepare an order for entering a position based on a news signal.
        
        Args:
            signal: Trading signal details
            current_price: Current price of the symbol
            position_size: Position size in shares/contracts
            
        Returns:
            Order details
        """
        if position_size <= 0:
            return None
            
        symbol = signal.get("symbol")
        signal_type = signal.get("signal_type")
        approach = signal.get("approach", "directional")
        
        # Calculate order price based on approach and order type
        if approach == "directional":
            order_type = self.parameters.get("directional_order_type", "limit")
            
            if order_type == "limit":
                # Calculate limit price with buffer
                tick_size = 0.01  # Default tick size
                buffer_ticks = self.parameters.get("limit_buffer_ticks", 1)
                
                if signal_type == "BUY":
                    # Buy at current price plus buffer
                    limit_price = current_price + (tick_size * buffer_ticks)
                else:  # SELL
                    # Sell at current price minus buffer
                    limit_price = current_price - (tick_size * buffer_ticks)
                    
                # Adjust for dynamic slippage if enabled
                if self.parameters.get("dynamic_slippage_adjustment", True):
                    # In a real implementation, this would use historical spread data
                    # and current market conditions to adjust the buffer
                    # For now, use a simple volatility-based adjustment
                    volatility_factor = 1.0  # Default factor
                    limit_price = limit_price * volatility_factor
                    
            else:  # market order
                limit_price = None
                
        else:  # volatility approach (e.g., straddle)
            # For volatility approaches, typically use market orders
            order_type = self.parameters.get("volatility_order_type", "market")
            limit_price = None
            
        # Prepare order
        action = "BUY" if signal_type == "BUY" else "SELL"
        
        order = {
            "symbol": symbol,
            "action": action,
            "quantity": position_size,
            "order_type": order_type,
            "limit_price": limit_price,
            "approach": approach,
            "signal_timestamp": signal.get("generation_timestamp"),
            "order_timestamp": datetime.now(),
            "event_type": signal.get("event_type"),
            "headline": signal.get("headline")
        }
        
        # Calculate order delay based on latency target
        order_delay_ms = self.parameters.get("order_delay_ms", 200)
        
        # In a real implementation, we would delay the order submission
        # For now, just log the intended delay
        logger.info(f"Order will be submitted after {order_delay_ms} ms delay")
        
        return order
    
    def _execute_entry_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an entry order with appropriate timing and monitoring.
        
        This is a placeholder for actual order execution logic.
        
        Args:
            order: Order details
            
        Returns:
            Execution results
        """
        # TODO: Implement actual order execution logic
        # This would use the trading system's order execution API
        
        # For now, simulate a successful execution
        symbol = order.get("symbol")
        action = order.get("action")
        quantity = order.get("quantity")
        order_type = order.get("order_type")
        limit_price = order.get("limit_price")
        
        # Simulate execution price
        if order_type == "limit" and limit_price is not None:
            execution_price = limit_price
        else:  # market order
            # Simulate some slippage
            base_price = limit_price if limit_price is not None else 100.0  # Placeholder price
            slippage_pct = 0.001  # 0.1% slippage
            
            if action == "BUY":
                execution_price = base_price * (1 + slippage_pct)
            else:  # SELL
                execution_price = base_price * (1 - slippage_pct)
        
        # Calculate fill metrics
        commission = quantity * 0.005  # $0.005 per share
        
        execution_result = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "requested_price": limit_price,
            "execution_price": execution_price,
            "commission": commission,
            "status": "filled",
            "execution_timestamp": datetime.now(),
            "latency_ms": np.random.randint(10, 50),  # Simulated execution latency
            "order_id": f"order_{int(time.time())}"
        }
        
        logger.info(f"Executed {action} order for {quantity} shares of {symbol} at ${execution_price:.2f}")
        
        return execution_result

    # === 7. Exit Rules ===
    def _check_exit_conditions(self, position: Dict[str, Any], current_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for an existing position.
        
        Args:
            position: Current position details
            current_data: Current market data including prices
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if not position:
            return False, ""
            
        symbol = position.get("symbol")
        
        if not symbol or symbol not in current_data:
            return False, ""
            
        # Get current price
        current_price = current_data[symbol].iloc[-1]['close']
        
        # Get exit targets
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        time_exit = position.get("time_exit")
        entry_price = position.get("entry_price", current_price)
        position_type = position.get("position_type", "long")  # long or short
        approach = position.get("approach", "directional")
        
        # Check time exit first
        current_time = datetime.now().timestamp()
        if time_exit and current_time >= time_exit:
            return True, "Time exit condition met"
            
        # For directional trades, check price-based exits
        if approach == "directional":
            # Check stop loss
            if position_type == "long":
                if stop_loss and current_price <= stop_loss:
                    return True, f"Stop loss triggered: {current_price:.2f} <= {stop_loss:.2f}"
                    
                # Check take profit
                if take_profit and current_price >= take_profit:
                    return True, f"Take profit reached: {current_price:.2f} >= {take_profit:.2f}"
                    
            else:  # short position
                if stop_loss and current_price >= stop_loss:
                    return True, f"Stop loss triggered: {current_price:.2f} >= {stop_loss:.2f}"
                    
                # Check take profit
                if take_profit and current_price <= take_profit:
                    return True, f"Take profit reached: {current_price:.2f} <= {take_profit:.2f}"
                    
        else:  # volatility approach (e.g., straddle)
            # For volatility trades, check premium-based conditions
            current_premium = position.get("current_premium", 0)
            initial_premium = position.get("initial_premium", 0)
            
            if initial_premium > 0:
                # Check stop loss (premium increase)
                if stop_loss and current_premium >= stop_loss:
                    return True, f"Volatility position stop loss triggered"
                    
                # Check take profit (premium decrease)
                if take_profit and current_premium <= take_profit:
                    return True, f"Volatility position profit target reached"
        
        # Check for secondary news that might affect the position
        if position.get("event_type") and self._check_conflicting_news(position):
            return True, "Conflicting news detected"
            
        return False, ""
    
    def _check_conflicting_news(self, position: Dict[str, Any]) -> bool:
        """
        Check if there's conflicting recent news that should trigger an exit.
        
        Args:
            position: Current position details
            
        Returns:
            Boolean indicating if conflicting news is detected
        """
        # This is a placeholder for a more sophisticated implementation
        # In a real system, this would query recent news and check for relevance
        
        # For now, simulate random news (1% chance per check)
        return np.random.random() < 0.01
    
    def _prepare_exit_order(self, position: Dict[str, Any], exit_reason: str) -> Dict[str, Any]:
        """
        Prepare an order to exit an existing position.
        
        Args:
            position: Current position details
            exit_reason: Reason for the exit
            
        Returns:
            Exit order details
        """
        symbol = position.get("symbol")
        position_type = position.get("position_type", "long")
        quantity = position.get("quantity", 0)
        approach = position.get("approach", "directional")
        
        if quantity <= 0:
            return None
            
        # Determine action (opposite of position type)
        action = "SELL" if position_type == "long" else "BUY"
        
        # For news trading, typically use market orders for exits to ensure execution
        order_type = "market"
        limit_price = None
        
        # For time-sensitive exits like stop losses, definitely use market orders
        if "stop loss" in exit_reason.lower():
            order_type = "market"
            
        # For volatility trades, may need to exit individual legs
        legs = []
        if approach == "volatility" and "legs" in position:
            for leg in position["legs"]:
                leg_action = "BUY" if leg.get("action") == "SELL" else "SELL"
                legs.append({
                    "symbol": leg.get("symbol"),
                    "action": leg_action,
                    "quantity": leg.get("quantity", 0),
                    "order_type": "market"
                })
        
        # Prepare the exit order
        exit_order = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "approach": approach,
            "exit_reason": exit_reason,
            "order_timestamp": datetime.now(),
            "legs": legs
        }
        
        logger.info(f"Preparing to exit {symbol} position: {exit_reason}")
        
        return exit_order
    
    def _execute_exit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an exit order with high priority and monitoring.
        
        This is a placeholder for actual exit order execution logic.
        
        Args:
            order: Exit order details
            
        Returns:
            Execution results
        """
        # TODO: Implement actual exit order execution logic
        # This would use the trading system's order execution API
        
        # For now, simulate a successful execution
        symbol = order.get("symbol")
        action = order.get("action")
        quantity = order.get("quantity")
        order_type = order.get("order_type")
        
        # Simulate execution price
        execution_price = 100.0  # Placeholder
        
        # Calculate fill metrics
        commission = quantity * 0.005  # $0.005 per share
        
        execution_result = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "execution_price": execution_price,
            "commission": commission,
            "status": "filled",
            "execution_timestamp": datetime.now(),
            "latency_ms": np.random.randint(5, 20),  # Exits typically prioritized for lower latency
            "order_id": f"exit_{int(time.time())}"
        }
        
        logger.info(f"Executed exit {action} order for {quantity} shares of {symbol} at ${execution_price:.2f}")
        
    # === 9. Backtesting & Performance Metrics ===
    def calculate_performance_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key performance metrics for news trading strategy.
        
        Args:
            backtest_results: DataFrame with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        if backtest_results.empty:
            return {}
            
        metrics = {}
        
        # Basic metrics
        total_trades = len(backtest_results)
        metrics['total_trades'] = total_trades
        
        if total_trades > 0:
            winning_trades = len(backtest_results[backtest_results['profit'] > 0])
            metrics['win_rate'] = winning_trades / total_trades
            metrics['avg_profit_per_trade'] = backtest_results['profit'].mean()
            metrics['avg_win'] = backtest_results.loc[backtest_results['profit'] > 0, 'profit'].mean() if winning_trades > 0 else 0
            metrics['avg_loss'] = backtest_results.loc[backtest_results['profit'] <= 0, 'profit'].mean() if total_trades - winning_trades > 0 else 0
            
            # Profit factor
            gross_profit = backtest_results.loc[backtest_results['profit'] > 0, 'profit'].sum()
            gross_loss = abs(backtest_results.loc[backtest_results['profit'] <= 0, 'profit'].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expected payoff
            metrics['expected_payoff'] = (metrics['win_rate'] * metrics['avg_win']) + ((1 - metrics['win_rate']) * metrics['avg_loss'])
            
            # Sharpe ratio (if timestamps available)
            if 'timestamp' in backtest_results.columns:
                backtest_results = backtest_results.sort_values('timestamp')
                daily_returns = backtest_results.set_index('timestamp')['profit'].resample('D').sum()
                metrics['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Event type performance
        if 'event_type' in backtest_results.columns:
            event_types = backtest_results['event_type'].unique()
            event_metrics = {}
            
            for event_type in event_types:
                event_df = backtest_results[backtest_results['event_type'] == event_type]
                event_total = len(event_df)
                event_wins = len(event_df[event_df['profit'] > 0])
                
                event_metrics[event_type] = {
                    'count': event_total,
                    'win_rate': event_wins / event_total if event_total > 0 else 0,
                    'avg_profit': event_df['profit'].mean() if event_total > 0 else 0
                }
                
            metrics['event_type_performance'] = event_metrics
        
        # Execution metrics
        if 'slippage_bps' in backtest_results.columns:
            metrics['avg_slippage_bps'] = backtest_results['slippage_bps'].mean()
            
        if 'latency_ms' in backtest_results.columns:
            metrics['avg_latency_ms'] = backtest_results['latency_ms'].mean()
            
        # PnL decay analysis
        if all(col in backtest_results.columns for col in ['signal_to_fill_ms', 'profit']):
            # Group by latency buckets and analyze performance
            backtest_results['latency_bucket'] = pd.cut(
                backtest_results['signal_to_fill_ms'],
                bins=[0, 100, 250, 500, 1000, float('inf')],
                labels=['0-100ms', '100-250ms', '250-500ms', '500-1000ms', '1000ms+']
            )
            
            latency_metrics = {}
            for bucket, group in backtest_results.groupby('latency_bucket'):
                latency_metrics[str(bucket)] = {
                    'count': len(group),
                    'avg_profit': group['profit'].mean(),
                    'win_rate': len(group[group['profit'] > 0]) / len(group) if len(group) > 0 else 0
                }
                
            metrics['latency_performance'] = latency_metrics
        
        # Market regime analysis
        if 'market_regime' in backtest_results.columns:
            regime_metrics = {}
            for regime, group in backtest_results.groupby('market_regime'):
                regime_metrics[regime] = {
                    'count': len(group),
                    'win_rate': len(group[group['profit'] > 0]) / len(group) if len(group) > 0 else 0,
                    'avg_profit': group['profit'].mean()
                }
                
            metrics['regime_performance'] = regime_metrics
            
        return metrics
    
    def analyze_backtest_slippage(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze slippage patterns in backtest results.
        
        Args:
            backtest_results: DataFrame with backtest results
            
        Returns:
            Slippage analysis results
        """
        if backtest_results.empty or 'slippage_bps' not in backtest_results.columns:
            return {}
            
        analysis = {}
        
        # Basic slippage stats
        analysis['mean_slippage_bps'] = backtest_results['slippage_bps'].mean()
        analysis['median_slippage_bps'] = backtest_results['slippage_bps'].median()
        analysis['max_slippage_bps'] = backtest_results['slippage_bps'].max()
        
        # Slippage distribution
        distribution = {}
        buckets = [-float('inf'), 0, 1, 2, 5, 10, float('inf')]
        labels = ['Negative', '0-1 bps', '1-2 bps', '2-5 bps', '5-10 bps', '10+ bps']
        
        backtest_results['slippage_bucket'] = pd.cut(
            backtest_results['slippage_bps'],
            bins=buckets,
            labels=labels
        )
        
        for bucket, group in backtest_results.groupby('slippage_bucket'):
            distribution[str(bucket)] = {
                'count': len(group),
                'percentage': len(group) / len(backtest_results) * 100
            }
            
        analysis['distribution'] = distribution
        
        # Slippage by event type
        if 'event_type' in backtest_results.columns:
            event_slippage = {}
            for event_type, group in backtest_results.groupby('event_type'):
                event_slippage[event_type] = group['slippage_bps'].mean()
                
            analysis['event_type_slippage'] = event_slippage
            
        # Slippage by signal strength
        if 'sentiment_score' in backtest_results.columns:
            backtest_results['sentiment_magnitude'] = backtest_results['sentiment_score'].abs()
            backtest_results['sentiment_bucket'] = pd.cut(
                backtest_results['sentiment_magnitude'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['0-0.3', '0.3-0.6', '0.6-0.8', '0.8-1.0']
            )
            
            sentiment_slippage = {}
            for bucket, group in backtest_results.groupby('sentiment_bucket'):
                sentiment_slippage[str(bucket)] = group['slippage_bps'].mean()
                
            analysis['sentiment_slippage'] = sentiment_slippage
            
        return analysis
    
    def evaluate_latency_impact(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the impact of latency on trading performance.
        
        Args:
            backtest_results: DataFrame with backtest results
            
        Returns:
            Latency impact analysis results
        """
        if backtest_results.empty or 'signal_to_fill_ms' not in backtest_results.columns:
            return {}
            
        analysis = {}
        
        # Basic latency stats
        analysis['mean_latency_ms'] = backtest_results['signal_to_fill_ms'].mean()
        analysis['median_latency_ms'] = backtest_results['signal_to_fill_ms'].median()
        analysis['max_latency_ms'] = backtest_results['signal_to_fill_ms'].max()
        
        # PnL by latency buckets
        backtest_results['latency_bucket'] = pd.cut(
            backtest_results['signal_to_fill_ms'],
            bins=[0, 100, 250, 500, 1000, float('inf')],
            labels=['0-100ms', '100-250ms', '250-500ms', '500-1000ms', '1000ms+']
        )
        
        bucket_performance = {}
        for bucket, group in backtest_results.groupby('latency_bucket'):
            bucket_performance[str(bucket)] = {
                'count': len(group),
                'win_rate': len(group[group['profit'] > 0]) / len(group) if len(group) > 0 else 0,
                'avg_profit': group['profit'].mean(),
                'avg_slippage_bps': group['slippage_bps'].mean() if 'slippage_bps' in group.columns else None
            }
            
        analysis['bucket_performance'] = bucket_performance
        
        # Calculate PnL decay per millisecond
        if 'profit' in backtest_results.columns:
            # Group by latency ranges and calculate average profit
            latency_ranges = [0, 100, 200, 300, 400, 500, 750, 1000]
            profits_by_range = []
            
            for i in range(len(latency_ranges) - 1):
                lower = latency_ranges[i]
                upper = latency_ranges[i+1]
                
                range_df = backtest_results[
                    (backtest_results['signal_to_fill_ms'] >= lower) & 
                    (backtest_results['signal_to_fill_ms'] < upper)
                ]
                
                if not range_df.empty:
                    avg_profit = range_df['profit'].mean()
                    profits_by_range.append((lower, upper, avg_profit))
            
            if len(profits_by_range) >= 2:
                # Calculate average decay rate
                decay_rates = []
                for i in range(len(profits_by_range) - 1):
                    profit_diff = profits_by_range[i+1][2] - profits_by_range[i][2]
                    latency_diff = profits_by_range[i+1][0] - profits_by_range[i][0]  # Use lower bounds
                    
                    if latency_diff > 0:
                        decay_rate = profit_diff / latency_diff
                        decay_rates.append(decay_rate)
                
                if decay_rates:
                    avg_decay_rate = sum(decay_rates) / len(decay_rates)
                    analysis['avg_pnl_decay_per_ms'] = avg_decay_rate
                    
                    # Extrapolate to basis points
                    # Assuming a reference P&L and using the decay rate
                    reference_pnl = backtest_results['profit'].mean()
                    if reference_pnl != 0:
                        decay_bps_per_ms = (avg_decay_rate / reference_pnl) * 10000
                        analysis['decay_bps_per_ms'] = decay_bps_per_ms
        
        return analysis
    
    # === 10. Continuous Optimization ===
    def check_model_freshness(self) -> bool:
        """
        Check if the sentiment model needs retraining.
        
        Returns:
            Boolean indicating if model needs retraining
        """
        # TODO: Implement proper model timestamp tracking
        
        model_max_age_days = self.parameters.get("max_model_age_days", 90)
        model_age = getattr(self, "model_age_days", 0)
        
        # In a real implementation, you would track when the model was last trained
        needs_retraining = model_age >= model_max_age_days
        
        if needs_retraining:
            logger.warning(f"Sentiment model requires retraining (age: {model_age} days)")
            
        return needs_retraining
    
    def optimize_sentiment_thresholds(self, historical_performance: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize sentiment thresholds based on historical performance.
        
        Args:
            historical_performance: DataFrame with historical performance data
            
        Returns:
            Optimized thresholds
        """
        if historical_performance.empty or 'sentiment_score' not in historical_performance.columns:
            return {}
            
        # Group by sentiment score ranges and analyze profit
        historical_performance['sentiment_magnitude'] = historical_performance['sentiment_score'].abs()
        
        # Create sentiment buckets
        buckets = np.arange(0.0, 1.05, 0.05)
        
        threshold_performance = {}
        for threshold in buckets:
            # Trades with sentiment above threshold
            trades = historical_performance[historical_performance['sentiment_magnitude'] >= threshold]
            
            if len(trades) > 0:
                win_rate = len(trades[trades['profit'] > 0]) / len(trades)
                avg_profit = trades['profit'].mean()
                
                threshold_performance[threshold] = {
                    'count': len(trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'expected_value': win_rate * avg_profit
                }
        
        # Find threshold with best expected value
        best_threshold = max(threshold_performance.items(), 
                            key=lambda x: x[1]['expected_value'] if len(x[1]) > 0 else -float('inf'))[0]
        
        # Find strong sentiment threshold (higher win rate)
        strong_candidates = {t: p for t, p in threshold_performance.items() 
                            if t >= best_threshold + 0.1 and p['count'] >= 10}
        
        strong_threshold = max(strong_candidates.items(), 
                              key=lambda x: x[1]['win_rate'] if len(x[1]) > 0 else -float('inf'))[0] if strong_candidates else best_threshold + 0.2
        
        logger.info(f"Optimized sentiment thresholds: min={best_threshold:.2f}, strong={strong_threshold:.2f}")
        
        return {
            "min_sentiment_threshold": best_threshold,
            "strong_sentiment_threshold": strong_threshold
        }
    
    def update_risk_parameters(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Update risk parameters based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            Updated risk parameters
        """
        updated_params = {}
        
        # Adjust position sizing based on win rate
        win_rate = performance_metrics.get('win_rate', 0.5)
        
        if win_rate >= 0.65:  # Very strong performance
            # Increase risk slightly
            updated_params['directional_risk_per_event_pct'] = min(0.01, self.parameters.get('directional_risk_per_event_pct', 0.005) * 1.2)
            updated_params['volatility_risk_per_event_pct'] = min(0.015, self.parameters.get('volatility_risk_per_event_pct', 0.01) * 1.1)
        elif win_rate <= 0.45:  # Poor performance
            # Reduce risk
            updated_params['directional_risk_per_event_pct'] = max(0.003, self.parameters.get('directional_risk_per_event_pct', 0.005) * 0.8)
            updated_params['volatility_risk_per_event_pct'] = max(0.005, self.parameters.get('volatility_risk_per_event_pct', 0.01) * 0.9)
        
        # Adjust time exit based on timing analysis
        if 'latency_performance' in performance_metrics:
            latency_perf = performance_metrics['latency_performance']
            
            # If faster execution performs significantly better, reduce time exit
            fast_trades = latency_perf.get('0-100ms', {}).get('win_rate', 0.5)
            slow_trades = latency_perf.get('500-1000ms', {}).get('win_rate', 0.5)
            
            if fast_trades > slow_trades + 0.2:  # 20% better win rate for fast trades
                # Reduce time exit window - move faster
                updated_params['time_exit_minutes'] = max(5, self.parameters.get('time_exit_minutes', 15) - 2)
                
        # Adjust profit targets based on market regime
        if 'regime_performance' in performance_metrics:
            regime_perf = performance_metrics['regime_performance']
            
            # If performing better in high volatility, increase profit targets
            high_vol = regime_perf.get('high_volatility', {}).get('avg_profit', 0)
            low_vol = regime_perf.get('low_volatility', {}).get('avg_profit', 0)
            
            if high_vol > low_vol * 1.5:  # 50% better performance in high volatility
                updated_params['directional_profit_target_pct'] = min(0.015, self.parameters.get('directional_profit_target_pct', 0.008) * 1.2)
            elif high_vol < low_vol * 0.5:  # Much worse in high volatility
                updated_params['directional_profit_target_pct'] = max(0.005, self.parameters.get('directional_profit_target_pct', 0.008) * 0.8)
        
        logger.info(f"Updated risk parameters based on performance metrics: {updated_params}")
        return updated_params
    
    def monitor_latency_integrity(self, recent_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor system latency and integrity.
        
        Args:
            recent_trades: List of recent trade execution details
            
        Returns:
            Monitoring results
        """
        if not recent_trades:
            return {"status": "unknown", "latency_alert": False}
            
        # Calculate average latencies
        sentiments_ms = [t.get('sentiment_latency_ms', 0) for t in recent_trades if 'sentiment_latency_ms' in t]
        signal_to_order_ms = [t.get('signal_to_order_ms', 0) for t in recent_trades if 'signal_to_order_ms' in t]
        order_to_fill_ms = [t.get('order_to_fill_ms', 0) for t in recent_trades if 'order_to_fill_ms' in t]
        total_latency_ms = [t.get('total_latency_ms', 0) for t in recent_trades if 'total_latency_ms' in t]
        
        results = {
            "status": "ok",
            "latency_alert": False,
            "avg_sentiment_latency_ms": sum(sentiments_ms) / len(sentiments_ms) if sentiments_ms else None,
            "avg_signal_to_order_ms": sum(signal_to_order_ms) / len(signal_to_order_ms) if signal_to_order_ms else None,
            "avg_order_to_fill_ms": sum(order_to_fill_ms) / len(order_to_fill_ms) if order_to_fill_ms else None,
            "avg_total_latency_ms": sum(total_latency_ms) / len(total_latency_ms) if total_latency_ms else None,
        }
        
        # Check against latency alert threshold
        latency_alert_threshold = self.parameters.get("latency_alert_threshold_ms", 700)
        
        if results["avg_total_latency_ms"] and results["avg_total_latency_ms"] > latency_alert_threshold:
            results["status"] = "warning"
            results["latency_alert"] = True
            logger.warning(f"Latency alert: {results['avg_total_latency_ms']:.1f} ms > {latency_alert_threshold} ms threshold")
        
        return results
    
    # Main methods required by the framework
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate strategy indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Filter universe to relevant symbols
        filtered_data = self.filter_universe(data)
        
        # Evaluate market conditions
        market_conditions = self._evaluate_market_conditions(filtered_data)
        
        for symbol, df in filtered_data.items():
            try:
                # Calculate standard technical indicators that might affect news trading
                
                # Volatility indicators
                df_indicators = pd.DataFrame(index=df.index)
                
                # ATR for volatility assessment
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    high_low = df['high'] - df['low']
                    high_close_prev = abs(df['high'] - df['close'].shift(1))
                    low_close_prev = abs(df['low'] - df['close'].shift(1))
                    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                    df_indicators['atr_14'] = tr.rolling(window=14).mean()
                    
                    # Normalized ATR (ATR as percentage of price)
                    df_indicators['atr_pct'] = df_indicators['atr_14'] / df['close']
                
                # Volume indicators
                if 'volume' in df.columns:
                    # Volume moving averages
                    df_indicators['volume_ma_20'] = df['volume'].rolling(window=20).mean()
                    
                    # Relative volume
                    df_indicators['rel_volume'] = df['volume'] / df_indicators['volume_ma_20']
                
                # Price moving averages for trend assessment
                df_indicators['sma_20'] = df['close'].rolling(window=20).mean()
                df_indicators['sma_50'] = df['close'].rolling(window=50).mean()
                df_indicators['sma_ratio'] = df_indicators['sma_20'] / df_indicators['sma_50']
                
                # Trend strength indicator (similar to ADX but simplified)
                df_indicators['price_trend'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
                
                # Market condition assessment
                df_indicators['market_regime'] = 'neutral'
                
                if symbol in market_conditions:
                    regime = market_conditions[symbol].get('volatility', 'normal_volatility')
                    trend = market_conditions[symbol].get('trend', 'neutral')
                    
                    # Combine into overall regime assessment
                    if trend == 'bullish' and regime != 'high_volatility':
                        df_indicators['market_regime'] = 'bullish_trend'
                    elif trend == 'bearish' and regime != 'high_volatility':
                        df_indicators['market_regime'] = 'bearish_trend'
                    elif regime == 'high_volatility':
                        df_indicators['market_regime'] = 'high_volatility'
                    else:
                        df_indicators['market_regime'] = 'range_bound'
                
                # News sensitivity (placeholder - in real implementation this would be calculated from historical data)
                df_indicators['news_sensitivity'] = 0.5
                
                # Store in the indicators dictionary
                indicators[symbol] = {
                    "volatility": pd.DataFrame({
                        "atr_14": df_indicators['atr_14'] if 'atr_14' in df_indicators else None,
                        "atr_pct": df_indicators['atr_pct'] if 'atr_pct' in df_indicators else None
                    }),
                    "volume": pd.DataFrame({
                        "volume_ma_20": df_indicators['volume_ma_20'] if 'volume_ma_20' in df_indicators else None,
                        "rel_volume": df_indicators['rel_volume'] if 'rel_volume' in df_indicators else None
                    }),
                    "trend": pd.DataFrame({
                        "sma_20": df_indicators['sma_20'],
                        "sma_50": df_indicators['sma_50'],
                        "sma_ratio": df_indicators['sma_ratio'],
                        "price_trend": df_indicators['price_trend']
                    }),
                    "market_regime": pd.DataFrame({
                        "regime": df_indicators['market_regime']
                    }),
                    "news": pd.DataFrame({
                        "sensitivity": df_indicators['news_sensitivity']
                    })
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals based on news sentiment and market conditions.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Filter universe to relevant symbols
        filtered_data = self.filter_universe(data)
        
        # Calculate indicators
        indicators = self.calculate_indicators(filtered_data)
        
        # Simulate news data (in a real implementation, this would come from news feeds)
        # TODO: Implement actual news feed integration
        simulated_news = self._simulate_news_data(filtered_data)
        
        # Process news data
        processed_news = self._process_news_data(simulated_news)
        
        # Generate signals from news
        signals = {}
        
        for news_item in processed_news:
            try:
                # Analyze sentiment
                sentiment = self.analyze_sentiment(news_item)
                
                # Get related symbols
                symbols = news_item.get("symbols", [])
                
                # Skip news without associated symbols
                if not symbols:
                    continue
                
                # Get market conditions for the symbols
                symbol_conditions = {}
                for symbol in symbols:
                    if symbol in indicators and "market_regime" in indicators[symbol]:
                        regime_df = indicators[symbol]["market_regime"]
                        if not regime_df.empty:
                            symbol_conditions[symbol] = {
                                "regime": regime_df["regime"].iloc[-1]
                            }
                
                # Generate signal based on sentiment and market conditions
                news_signal = self.generate_news_signal(news_item, sentiment, symbol_conditions)
                
                if not news_signal:
                    continue
                
                # Extract the primary symbol for the signal
                symbol = news_signal.get("symbol")
                
                if not symbol or symbol not in filtered_data:
                    continue
                
                # Get latest data for the symbol
                latest_data = filtered_data[symbol].iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Determine signal type
                signal_type = SignalType.BUY if news_signal.get("signal_type") == "BUY" else SignalType.SELL
                
                # Calculate confidence
                confidence = news_signal.get("confidence", 0.5)
                
                # Calculate stop loss and take profit levels
                if signal_type == SignalType.BUY:
                    stop_loss = latest_price * (1 - self.parameters.get("directional_stop_loss_pct", 0.004))
                    take_profit = latest_price * (1 + self.parameters.get("directional_profit_target_pct", 0.008))
                else:  # SELL
                    stop_loss = latest_price * (1 + self.parameters.get("directional_stop_loss_pct", 0.004))
                    take_profit = latest_price * (1 - self.parameters.get("directional_profit_target_pct", 0.008))
                
                # Create signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "strategy_type": "news_trading",
                        "event_type": news_signal.get("event_type"),
                        "sentiment_score": news_signal.get("sentiment_score"),
                        "approach": news_signal.get("approach"),
                        "headline": news_signal.get("headline"),
                        "news_timestamp": news_signal.get("news_timestamp"),
                        "latency_ms": news_signal.get("latency_ms")
                    }
                )
                
                logger.info(f"Generated {signal_type.value} signal for {symbol} based on news: {news_signal.get('headline', '')[:50]}...")
                
            except Exception as e:
                logger.error(f"Error generating signal from news: {e}")
        
        return signals
    
    def _simulate_news_data(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Simulate news data for development and testing.
        
        Args:
            data: Dictionary of market data
            
        Returns:
            List of simulated news items
        """
        # This is only a placeholder for development/testing
        # In production, this would be replaced with actual news feed integration
        
        simulated_news = []
        
        # Only simulate news occasionally (1% chance per run)
        if np.random.random() > 0.01:
            return []
            
        # Get a random symbol from the data
        symbols = list(data.keys())
        if not symbols:
            return []
            
        symbol = np.random.choice(symbols)
        
        # Generate random event type
        event_types = ["earnings", "macro", "regulatory", "merger", "unscheduled"]
        event_weights = [0.3, 0.3, 0.2, 0.1, 0.1]  # Higher probability for earnings and macro
        event_type = np.random.choice(event_types, p=event_weights)
        
        # Generate bullish/bearish sentiment (slightly biased to bullish)
        sentiment_direction = np.random.choice(["positive", "negative"], p=[0.55, 0.45])
        
        # Create headline based on event type and sentiment
        headline = ""
        if event_type == "earnings":
            if sentiment_direction == "positive":
                headline = f"{symbol} Earnings Beat Expectations, Revenue Up 12%"
            else:
                headline = f"{symbol} Misses Earnings Estimates, Cuts Guidance"
        elif event_type == "macro":
            if sentiment_direction == "positive":
                headline = "Fed Signals Potential Rate Cut, Markets Rally"
            else:
                headline = "Inflation Data Higher Than Expected, Rate Concerns Rise"
        elif event_type == "regulatory":
            if sentiment_direction == "positive":
                headline = f"{symbol} Receives Regulatory Approval for New Product"
            else:
                headline = f"Regulatory Investigation Opened into {symbol}'s Business Practices"
        elif event_type == "merger":
            headline = f"{symbol} in Advanced Talks for Potential Acquisition"
        else:  # unscheduled
            if sentiment_direction == "positive":
                headline = f"{symbol} Announces Major New Contract Win"
            else:
                headline = f"Executive Departures at {symbol} Raise Concerns"
        
        # Create news item
        news_item = {
            "headline": headline,
            "content": headline + " " + "Additional details would appear here in a real news item.",
            "source": np.random.choice(["bloomberg", "reuters", "wsj", "cnbc"]),
            "timestamp": datetime.now(),
            "symbols": [symbol],
            "event_type": event_type
        }
        
        simulated_news.append(news_item)
        logger.info(f"Simulated news item: {headline}")
        
        return simulated_news