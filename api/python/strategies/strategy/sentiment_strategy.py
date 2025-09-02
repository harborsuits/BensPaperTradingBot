#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment Strategy - NLP-based sentiment analysis strategies for trading
that integrate with the StrategyRotator system.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json
import re
from collections import Counter, defaultdict

# Import base strategy class
from trading_bot.strategy.strategy_rotator import Strategy
from trading_bot.common.config_utils import setup_directories, load_config, save_state, load_state

# Setup logging
logger = logging.getLogger("SentimentStrategy")

class SentimentStrategy(Strategy):
    """Base class for sentiment analysis trading strategies"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a sentiment strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Setup paths
        self.paths = setup_directories(
            data_dir=config.get("data_dir"),
            component_name=f"sentiment_strategy_{name}"
        )
        
        # Sentiment data cache
        self.sentiment_data = {}
        self.last_update_dates = {}
        
        # Configure sentiment analysis
        self.news_weight = config.get("news_weight", 0.5)
        self.social_weight = config.get("social_weight", 0.3)
        self.filings_weight = config.get("filings_weight", 0.2)
        
        # Decay factors for historical sentiment
        self.daily_decay = config.get("daily_decay", 0.9)  # Sentiment decays by 10% per day
        
        # Update frequency in seconds
        self.update_frequency = config.get("update_frequency", 3600)  # Hourly by default
        
        # Sentiment threshold for signal generation
        self.positive_threshold = config.get("positive_threshold", 0.2)
        self.negative_threshold = config.get("negative_threshold", -0.2)
        
        # Load cached sentiment data if available
        self._load_cached_data()
    
    def _load_cached_data(self) -> None:
        """Load cached sentiment data from disk."""
        cache_path = os.path.join(self.paths["data_dir"], "sentiment_cache.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    self.sentiment_data = data.get("data", {})
                    self.last_update_dates = {
                        k: datetime.fromisoformat(v) if v else None
                        for k, v in data.get("last_updates", {}).items()
                    }
                logger.info(f"Loaded sentiment data cache for {len(self.sentiment_data)} symbols")
            except Exception as e:
                logger.error(f"Error loading sentiment data cache: {e}")
    
    def _save_cached_data(self) -> None:
        """Save sentiment data cache to disk."""
        cache_path = os.path.join(self.paths["data_dir"], "sentiment_cache.json")
        
        try:
            # Convert datetime objects to ISO format strings
            last_updates = {
                k: v.isoformat() if v else None
                for k, v in self.last_update_dates.items()
            }
            
            with open(cache_path, 'w') as f:
                json.dump({
                    "data": self.sentiment_data,
                    "last_updates": last_updates
                }, f)
            logger.info(f"Saved sentiment data cache for {len(self.sentiment_data)} symbols")
        except Exception as e:
            logger.error(f"Error saving sentiment data cache: {e}")
    
    def update_sentiment_data(self, symbols: List[str], force: bool = False) -> None:
        """
        Update sentiment data for specified symbols.
        
        Args:
            symbols: List of symbols to update
            force: Force update even if data is recent
        """
        now = datetime.now()
        updated_symbols = []
        
        for symbol in symbols:
            # Check if update is needed
            last_update = self.last_update_dates.get(symbol)
            
            if (not force and last_update and 
                (now - last_update).total_seconds() < self.update_frequency):
                logger.debug(f"Skipping update for {symbol}, data is recent")
                continue
            
            try:
                # Fetch data for symbol
                data = self._fetch_sentiment_data(symbol)
                
                if data:
                    self.sentiment_data[symbol] = data
                    self.last_update_dates[symbol] = now
                    updated_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error updating sentiment data for {symbol}: {e}")
        
        if updated_symbols:
            logger.info(f"Updated sentiment data for {len(updated_symbols)} symbols")
            self._save_cached_data()
    
    def _fetch_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch sentiment data for a symbol.
        
        Args:
            symbol: Symbol to fetch data for
            
        Returns:
            Dict with sentiment data
        """
        # In a real implementation, this would fetch data from various sources
        # such as news APIs, social media, SEC filings, etc.
        
        logger.warning(f"Using mock sentiment data for {symbol}")
        
        # Generate random sentiment data for demonstration
        now = datetime.now()
        
        # News sentiment (articles from last 7 days)
        news_items = []
        for i in range(10):  # 10 mock news articles
            days_ago = np.random.randint(0, 7)
            date = (now - timedelta(days=days_ago)).isoformat()
            news_items.append({
                "date": date,
                "headline": f"Mock headline {i+1} for {symbol}",
                "source": f"Source {i % 3 + 1}",
                "sentiment": np.random.normal(0, 0.3),  # Random sentiment score
                "relevance": np.random.uniform(0.5, 1.0)  # Relevance of article to the company
            })
        
        # Social media sentiment (posts from last 3 days)
        social_items = []
        for i in range(20):  # 20 mock social media posts
            hours_ago = np.random.randint(0, 72)
            date = (now - timedelta(hours=hours_ago)).isoformat()
            social_items.append({
                "date": date,
                "platform": ["Twitter", "Reddit", "StockTwits"][i % 3],
                "sentiment": np.random.normal(0, 0.5),  # More volatile than news
                "engagement": np.random.randint(1, 1000)  # Likes, shares, comments
            })
        
        # SEC filings (last 4 quarterly reports)
        filing_items = []
        for i in range(4):
            months_ago = i * 3
            date = (now - timedelta(days=months_ago * 30)).isoformat()
            filing_items.append({
                "date": date,
                "type": "10-Q" if i < 3 else "10-K",
                "sentiment": np.random.normal(0, 0.2),  # More conservative sentiment
                "risk_score": np.random.uniform(0, 1),  # Risk assessment from filing
                "key_topics": ["revenue", "growth", "expenses", "outlook"][i % 4]
            })
        
        return {
            "news": news_items,
            "social": social_items,
            "filings": filing_items,
            "aggregated": {
                "overall_sentiment": np.random.normal(0, 0.3),
                "sentiment_trend": np.random.normal(0, 0.1),
                "volume_trend": np.random.uniform(-0.2, 0.2)
            }
        }
    
    def calculate_news_sentiment(self, symbol: str, days: int = 7) -> Tuple[float, float]:
        """
        Calculate news sentiment for a symbol over the specified period.
        
        Args:
            symbol: Symbol to calculate sentiment for
            days: Number of days to analyze
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if symbol not in self.sentiment_data:
            logger.warning(f"No sentiment data for {symbol}")
            return 0.0, 0.0
        
        # Get news data
        news_items = self.sentiment_data[symbol].get("news", [])
        
        if not news_items:
            return 0.0, 0.0
        
        # Filter for recent news
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_items = [item for item in news_items if item["date"] >= cutoff_date]
        
        if not recent_items:
            return 0.0, 0.0
        
        # Calculate weighted sentiment
        total_relevance = 0
        weighted_sentiment = 0
        
        for item in recent_items:
            sentiment = item.get("sentiment", 0)
            relevance = item.get("relevance", 1.0)
            
            # Apply time decay (more recent news is more important)
            item_date = datetime.fromisoformat(item["date"])
            days_old = (datetime.now() - item_date).days
            time_factor = self.daily_decay ** days_old
            
            weighted_sentiment += sentiment * relevance * time_factor
            total_relevance += relevance * time_factor
        
        # Calculate average sentiment and confidence
        if total_relevance > 0:
            avg_sentiment = weighted_sentiment / total_relevance
            confidence = min(1.0, len(recent_items) / 10)  # More articles = higher confidence
            return avg_sentiment, confidence
        else:
            return 0.0, 0.0
    
    def calculate_social_sentiment(self, symbol: str, days: int = 3) -> Tuple[float, float]:
        """
        Calculate social media sentiment for a symbol over the specified period.
        
        Args:
            symbol: Symbol to calculate sentiment for
            days: Number of days to analyze
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if symbol not in self.sentiment_data:
            logger.warning(f"No sentiment data for {symbol}")
            return 0.0, 0.0
        
        # Get social data
        social_items = self.sentiment_data[symbol].get("social", [])
        
        if not social_items:
            return 0.0, 0.0
        
        # Filter for recent posts
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_items = [item for item in social_items if item["date"] >= cutoff_date]
        
        if not recent_items:
            return 0.0, 0.0
        
        # Calculate weighted sentiment
        total_engagement = 0
        weighted_sentiment = 0
        
        for item in recent_items:
            sentiment = item.get("sentiment", 0)
            engagement = item.get("engagement", 1)
            
            # Apply time decay
            item_date = datetime.fromisoformat(item["date"])
            hours_old = (datetime.now() - item_date).total_seconds() / 3600
            time_factor = self.daily_decay ** (hours_old / 24)
            
            weighted_sentiment += sentiment * engagement * time_factor
            total_engagement += engagement * time_factor
        
        # Calculate average sentiment and confidence
        if total_engagement > 0:
            avg_sentiment = weighted_sentiment / total_engagement
            confidence = min(1.0, total_engagement / 1000)  # More engagement = higher confidence
            return avg_sentiment, confidence
        else:
            return 0.0, 0.0
    
    def calculate_filing_sentiment(self, symbol: str, quarters: int = 4) -> Tuple[float, float]:
        """
        Calculate SEC filing sentiment for a symbol over the specified period.
        
        Args:
            symbol: Symbol to calculate sentiment for
            quarters: Number of quarters to analyze
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if symbol not in self.sentiment_data:
            logger.warning(f"No sentiment data for {symbol}")
            return 0.0, 0.0
        
        # Get filing data
        filing_items = self.sentiment_data[symbol].get("filings", [])
        
        if not filing_items:
            return 0.0, 0.0
        
        # Filter for recent filings (simple approach)
        recent_items = filing_items[:quarters] if len(filing_items) >= quarters else filing_items
        
        if not recent_items:
            return 0.0, 0.0
        
        # Calculate filing sentiment (more conservative)
        sentiments = [item.get("sentiment", 0) for item in recent_items]
        risk_scores = [item.get("risk_score", 0.5) for item in recent_items]
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        avg_risk = sum(risk_scores) / len(risk_scores)
        
        # Adjust sentiment based on risk (higher risk reduces sentiment)
        adjusted_sentiment = avg_sentiment * (1 - avg_risk * 0.5)
        
        # Filings generally have high confidence
        confidence = 0.8
        
        return adjusted_sentiment, confidence
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based on sentiment analysis.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update sentiment data if needed
        if symbol not in self.sentiment_data:
            self.update_sentiment_data([symbol])
        
        # Calculate component sentiment scores
        news_sentiment, news_confidence = self.calculate_news_sentiment(symbol)
        social_sentiment, social_confidence = self.calculate_social_sentiment(symbol)
        filing_sentiment, filing_confidence = self.calculate_filing_sentiment(symbol)
        
        # Generate weighted signal based on confidence
        if news_confidence == 0 and social_confidence == 0 and filing_confidence == 0:
            return 0.0  # No data
        
        # Combine confidences and weights
        news_weight = self.news_weight * news_confidence
        social_weight = self.social_weight * social_confidence
        filing_weight = self.filings_weight * filing_confidence
        
        # Normalize weights
        total_weight = news_weight + social_weight + filing_weight
        
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted sentiment
        signal = (
            news_sentiment * news_weight +
            social_sentiment * social_weight +
            filing_sentiment * filing_weight
        ) / total_weight
        
        # Apply thresholds to generate signal
        if signal > self.positive_threshold:
            # Strong positive sentiment
            final_signal = min(1.0, signal * 2)  # Scale up for stronger signals
        elif signal < self.negative_threshold:
            # Strong negative sentiment
            final_signal = max(-1.0, signal * 2)  # Scale up for stronger signals
        else:
            # Neutral sentiment
            final_signal = signal
        
        # Update last signal and time
        self.last_signal = final_signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated sentiment signal for {symbol}: {final_signal:.4f}")
        logger.debug(f"Components: News={news_sentiment:.2f} (conf={news_confidence:.2f}), "
                    f"Social={social_sentiment:.2f} (conf={social_confidence:.2f}), "
                    f"Filings={filing_sentiment:.2f} (conf={filing_confidence:.2f})")
        
        return final_signal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        base_dict = super().to_dict()
        # Add any additional fields specific to sentiment strategies
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentStrategy':
        """Create strategy from dictionary."""
        return super().from_dict(data)


class NewsSentimentStrategy(SentimentStrategy):
    """Strategy focusing primarily on news sentiment"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize News Sentiment strategy"""
        super().__init__(name, config)
        
        # Override weights to focus on news
        self.news_weight = config.get("news_weight", 0.8)
        self.social_weight = config.get("social_weight", 0.2)
        self.filings_weight = config.get("filings_weight", 0.0)
        
        # Shorter time frame for analysis
        self.news_days = config.get("news_days", 3)  # Look at more recent news
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based primarily on news sentiment.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update sentiment data if needed
        if symbol not in self.sentiment_data:
            self.update_sentiment_data([symbol])
        
        # Focus on recent news
        news_sentiment, news_confidence = self.calculate_news_sentiment(symbol, days=self.news_days)
        
        # Include some social media for validation
        social_sentiment, social_confidence = self.calculate_social_sentiment(symbol)
        
        # Generate weighted signal
        if news_confidence == 0 and social_confidence == 0:
            return 0.0
        
        # Normalize weights
        total_weight = (self.news_weight * news_confidence + 
                        self.social_weight * social_confidence)
        
        if total_weight == 0:
            return 0.0
        
        signal = (
            news_sentiment * self.news_weight * news_confidence +
            social_sentiment * self.social_weight * social_confidence
        ) / total_weight
        
        # Apply thresholds with stronger response
        if signal > self.positive_threshold:
            final_signal = min(1.0, signal * 2.5)
        elif signal < self.negative_threshold:
            final_signal = max(-1.0, signal * 2.5)
        else:
            final_signal = signal
        
        # Update last signal and time
        self.last_signal = final_signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated news-focused signal for {symbol}: {final_signal:.4f}")
        
        return final_signal


class SECFilingStrategy(SentimentStrategy):
    """Strategy focusing on SEC filing analysis and financial statements"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize SEC Filing strategy"""
        super().__init__(name, config)
        
        # Override weights to focus on filings
        self.news_weight = config.get("news_weight", 0.2)
        self.social_weight = config.get("social_weight", 0.0)
        self.filings_weight = config.get("filings_weight", 0.8)
        
        # Risk sensitivity - how much risk factors reduce sentiment
        self.risk_sensitivity = config.get("risk_sensitivity", 0.6)
    
    def calculate_filing_sentiment(self, symbol: str, quarters: int = 4) -> Tuple[float, float]:
        """
        Calculate SEC filing sentiment with enhanced risk analysis.
        
        Args:
            symbol: Symbol to calculate sentiment for
            quarters: Number of quarters to analyze
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        # Get base sentiment from parent method
        sentiment, confidence = super().calculate_filing_sentiment(symbol, quarters)
        
        if symbol not in self.sentiment_data:
            return sentiment, confidence
        
        # Get filing data
        filing_items = self.sentiment_data[symbol].get("filings", [])
        
        if not filing_items:
            return sentiment, confidence
        
        # Enhanced analysis: look for risk factor trends
        # In a real implementation, this would parse actual risk factors
        risk_scores = [item.get("risk_score", 0.5) for item in filing_items[:quarters]]
        
        if len(risk_scores) > 1:
            # Check if risks are increasing (negative signal)
            risk_trend = risk_scores[0] - risk_scores[-1]
            
            # Adjust sentiment based on risk trend
            sentiment -= risk_trend * self.risk_sensitivity
        
        return sentiment, confidence
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based primarily on SEC filings.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Update sentiment data if needed
        if symbol not in self.sentiment_data:
            self.update_sentiment_data([symbol])
        
        # SEC filings are the primary focus
        filing_sentiment, filing_confidence = self.calculate_filing_sentiment(symbol)
        
        # Include some news for context
        news_sentiment, news_confidence = self.calculate_news_sentiment(symbol)
        
        # Generate weighted signal
        if filing_confidence == 0 and news_confidence == 0:
            return 0.0
        
        # Normalize weights
        total_weight = (self.filings_weight * filing_confidence + 
                        self.news_weight * news_confidence)
        
        if total_weight == 0:
            return 0.0
        
        signal = (
            filing_sentiment * self.filings_weight * filing_confidence +
            news_sentiment * self.news_weight * news_confidence
        ) / total_weight
        
        # SEC-based signals are more conservative
        final_signal = signal * 0.8  # Dampen the signal strength
        
        # Update last signal and time
        self.last_signal = final_signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated SEC-focused signal for {symbol}: {final_signal:.4f}")
        
        return final_signal


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a sentiment strategy
    sentiment_strategy = NewsSentimentStrategy("NewsSentimentStrategy")
    
    # Create mock market data
    market_data = {
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 1000000
    }
    
    # Generate signal
    signal = sentiment_strategy.generate_signal(market_data)
    print(f"News Sentiment signal for AAPL: {signal:.4f}")
    
    # Try another strategy
    filing_strategy = SECFilingStrategy("SECFilingStrategy")
    signal = filing_strategy.generate_signal(market_data)
    print(f"SEC Filing signal for AAPL: {signal:.4f}") 