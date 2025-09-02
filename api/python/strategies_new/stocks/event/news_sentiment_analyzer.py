#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News Sentiment Analyzer

This module provides analysis capabilities for news sentiment data,
including sentiment aggregation, trend identification, and signal generation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import copy

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    Analyzer for processing and evaluating news and social media sentiment data.
    
    This class handles sentiment aggregation, scoring, trend identification,
    and signal generation based on sentiment patterns and changes.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the News Sentiment Analyzer.
        
        Args:
            parameters: Configuration parameters for sentiment analysis
        """
        # Default parameters
        self.parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.5, 'social_media': 0.3, 'analyst_reports': 0.2},
            
            # Sentiment thresholds
            'bullish_threshold': 0.6,      # Minimum score to consider bullish (0-1)
            'bearish_threshold': 0.4,      # Maximum score to consider bearish (0-1)
            'neutral_band': 0.1,           # Neutral zone around 0.5 (0.5 +/- this value)
            'significant_change': 0.15,    # Minimum change to be considered significant
            
            # Analysis parameters
            'lookback_period_days': 3,     # Days to look back for sentiment analysis
            'sentiment_smoothing': 7,      # Rolling window for sentiment smoothing
            'min_sentiment_volume': 5,     # Minimum news volume to consider reliable
            
            # Contrarian settings
            'contrarian_threshold': 0.8,   # Threshold for extreme sentiment (0-1)
            'contrarian_lookback_days': 5, # Days to analyze for contrarian setup
        }
        
        # Override with any provided parameters
        if parameters:
            self.parameters.update(parameters)
            
        # Initialize data structures
        self.sentiment_history = []  # Raw sentiment data from various sources
        self.sentiment_data = {      # Aggregated and processed sentiment metrics
            'current_score': 0.5,    # Current aggregate sentiment score (0-1)
            'previous_score': 0.5,   # Previous sentiment score for comparison
            'daily_change': 0.0,     # Day-over-day sentiment change
            'weekly_change': 0.0,    # Week-over-week sentiment change
            'volume': 0,             # Number of sentiment data points
            'trend': 'neutral',      # Current sentiment trend
            'news_buzz': 0,          # Relative news volume
            'extremes': {
                'bullish': 0.0,       # Most bullish recent sentiment
                'bearish': 1.0,       # Most bearish recent sentiment
            },
            'source_scores': {       # Sentiment by source
                'news': 0.5,
                'social_media': 0.5,
                'analyst_reports': 0.5
            },
            'divergence': 0.0,       # Divergence from price action
            'reversal_signal': None, # Potential sentiment reversal
            'last_updated': datetime.now()
        }
    
    def add_sentiment_item(self, source: str, sentiment: float, metadata: Dict[str, Any] = None) -> None:
        """
        Add a new sentiment data point to the history.
        
        Args:
            source: Source of the sentiment data (news, social_media, analyst_reports)
            sentiment: Sentiment score (0-1, where 0 is bearish, 1 is bullish)
            metadata: Additional metadata about the sentiment item
        """
        if sentiment < 0 or sentiment > 1:
            logger.warning(f"Invalid sentiment score: {sentiment}. Must be between 0 and 1.")
            sentiment = max(0, min(1, sentiment))  # Clamp to valid range
            
        item = {
            'timestamp': datetime.now(),
            'source': source,
            'sentiment': sentiment,
        }
        
        # Add any metadata
        if metadata:
            item.update(metadata)
            
        # Add to history
        self.sentiment_history.append(item)
        
        # Update aggregated sentiment data
        self.update_sentiment_data()
        
        logger.debug(f"Added {source} sentiment data: {sentiment:.2f}")
    
    def update_sentiment_data(self) -> Dict[str, Any]:
        """
        Aggregate and process sentiment data from all sources.
        
        Returns:
            Updated sentiment data dictionary
        """
        # Skip if no sentiment history yet
        if not self.sentiment_history:
            return self.sentiment_data
            
        # Store previous score for change calculation
        previous_score = self.sentiment_data['current_score']
        
        # Set cutoff dates for lookback periods
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        one_week_ago = now - timedelta(days=7)
        
        # Filter relevant entries within lookback period
        recent_entries = [entry for entry in self.sentiment_history 
                         if entry['timestamp'] >= now - timedelta(days=self.parameters['lookback_period_days'])]
        
        # Count total entries by source
        source_counts = {}
        for source in self.parameters['sentiment_sources']:
            source_counts[source] = len([e for e in recent_entries if e['source'] == source])
        
        # Calculate weighted average sentiment by source
        source_scores = {}
        for source in self.parameters['sentiment_sources']:
            source_entries = [e for e in recent_entries if e['source'] == source]
            if source_entries:
                source_scores[source] = sum(e['sentiment'] for e in source_entries) / len(source_entries)
            else:
                source_scores[source] = 0.5  # Neutral if no entries
        
        # Compute weighted aggregate sentiment
        weights = self.parameters['source_weights']
        total_weight = sum(weights[s] for s in source_scores.keys() if s in weights)
        
        if total_weight > 0:
            aggregate_score = sum(source_scores[s] * weights[s] / total_weight 
                               for s in source_scores.keys() if s in weights)
        else:
            aggregate_score = 0.5  # Neutral if no weights
        
        # Calculate daily and weekly changes
        day_entries = [e for e in self.sentiment_history if e['timestamp'] >= one_day_ago]
        week_entries = [e for e in self.sentiment_history if e['timestamp'] >= one_week_ago]
        
        day_ago_score = 0.5
        week_ago_score = 0.5
        
        if day_entries:
            day_ago_score = sum(e['sentiment'] for e in day_entries) / len(day_entries)
        
        if week_entries:
            week_ago_score = sum(e['sentiment'] for e in week_entries) / len(week_entries)
        
        daily_change = aggregate_score - day_ago_score
        weekly_change = aggregate_score - week_ago_score
        
        # Determine sentiment trend
        trend = 'neutral'
        if daily_change > self.parameters['significant_change']:
            trend = 'improving'
        elif daily_change < -self.parameters['significant_change']:
            trend = 'deteriorating'
        elif aggregate_score > self.parameters['bullish_threshold']:
            trend = 'bullish'
        elif aggregate_score < self.parameters['bearish_threshold']:
            trend = 'bearish'
        
        # Find sentiment extremes in lookback period
        if recent_entries:
            extremes = {
                'bullish': max(e['sentiment'] for e in recent_entries),
                'bearish': min(e['sentiment'] for e in recent_entries)
            }
        else:
            extremes = {'bullish': 0.5, 'bearish': 0.5}
        
        # Calculate news buzz (relative volume)
        avg_daily_volume = len(self.sentiment_history) / max(1, (now - self.sentiment_history[0]['timestamp']).days) \
            if self.sentiment_history else 0
        recent_volume = len(recent_entries)
        news_buzz = recent_volume / max(1, avg_daily_volume * self.parameters['lookback_period_days'])
        
        # Check for sentiment reversals
        reversal_signal = self._check_sentiment_reversal(aggregate_score, daily_change, weekly_change)
        
        # Update sentiment data
        self.sentiment_data = {
            'current_score': aggregate_score,
            'previous_score': previous_score,
            'daily_change': daily_change,
            'weekly_change': weekly_change,
            'volume': recent_volume,
            'trend': trend,
            'news_buzz': news_buzz,
            'extremes': extremes,
            'source_scores': source_scores,
            'divergence': 0.0,  # Will be updated when market data available
            'reversal_signal': reversal_signal,
            'last_updated': now
        }
        
        return self.sentiment_data
    
    def _check_sentiment_reversal(self, current_score: float, daily_change: float, weekly_change: float) -> Optional[str]:
        """
        Check for potential sentiment trend reversals.
        
        Args:
            current_score: Current sentiment score
            daily_change: Day-over-day sentiment change
            weekly_change: Week-over-week sentiment change
            
        Returns:
            Reversal signal type or None
        """
        # No reversal if change is not significant
        if abs(daily_change) < self.parameters['significant_change']:
            return None
            
        # Look for bullish reversals
        if current_score > 0.5 and weekly_change < 0 and daily_change > 0:
            # Score is now above neutral, but was declining, and just turned up
            return 'bullish_reversal'
            
        # Look for bearish reversals
        if current_score < 0.5 and weekly_change > 0 and daily_change < 0:
            # Score is now below neutral, but was improving, and just turned down
            return 'bearish_reversal'
            
        # Extreme sentiment reversals (contrarian)
        if current_score > self.parameters['contrarian_threshold'] and daily_change < 0:
            # Extremely bullish sentiment just started turning down
            return 'extreme_bullish_reversal'
            
        if current_score < (1 - self.parameters['contrarian_threshold']) and daily_change > 0:
            # Extremely bearish sentiment just started turning up
            return 'extreme_bearish_reversal'
            
        return None
    
    def calculate_sentiment_price_divergence(self, sentiment_score: float, price_data: pd.DataFrame) -> float:
        """
        Calculate divergence between sentiment and recent price action.
        
        Args:
            sentiment_score: Current sentiment score (0-1)
            price_data: DataFrame containing price history with at least 'close' column
            
        Returns:
            Divergence metric (-1 to 1, 0 = no divergence)
        """
        # Need price data to calculate divergence
        if price_data.empty or len(price_data) < 5:
            return 0.0
            
        # Calculate recent price change (normalized to -1 to 1 scale)
        close_prices = price_data['close']
        lookback = min(5, len(close_prices)-1)
        price_change = (close_prices.iloc[-1] - close_prices.iloc[-lookback-1]) / close_prices.iloc[-lookback-1]
            
        # Normalize to a similar scale as sentiment
        if abs(price_change) > 0.1:
            price_change = 0.1 * (price_change / abs(price_change))  # Cap at +/- 0.1
            
        # Normalize price change to 0-1 scale (0.5 = neutral)
        price_sentiment = 0.5 + (price_change * 5)  # Scale factor of 5 converts +/-0.1 to +/-0.5
        price_sentiment = max(0.0, min(1.0, price_sentiment))
        
        # Calculate divergence (positive = sentiment more bullish than price action)
        divergence = sentiment_score - price_sentiment
        
        # Update divergence in sentiment data
        self.sentiment_data['divergence'] = divergence
        
        return divergence
    
    def get_sentiment_signals(self, symbol: str, price_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on current sentiment data.
        
        Args:
            symbol: Symbol the signals apply to
            price_data: Optional price data to enhance signal generation
            
        Returns:
            List of sentiment-based signal dictionaries
        """
        signals = []
        
        # Skip if insufficient sentiment data
        if self.sentiment_data['volume'] < self.parameters['min_sentiment_volume']:
            return signals
            
        # Extract sentiment metrics
        sentiment = self.sentiment_data['current_score']
        trend = self.sentiment_data['trend']
        reversal = self.sentiment_data['reversal_signal']
        
        # Calculate divergence if price data provided
        divergence = 0.0
        if price_data is not None and not price_data.empty:
            divergence = self.calculate_sentiment_price_divergence(sentiment, price_data)
        
        # TREND FOLLOWING SIGNALS
        # Check for bullish signal
        if sentiment > self.parameters['bullish_threshold'] and trend in ['bullish', 'improving']:
            signals.append({
                'type': 'long',
                'confidence': sentiment - 0.5,  # 0-0.5 scale
                'source': 'sentiment_trend_following',
                'reason': f"Bullish sentiment ({sentiment:.2f}) with {trend} trend"
            })
            
        # Check for bearish signal
        elif sentiment < self.parameters['bearish_threshold'] and trend in ['bearish', 'deteriorating']:
            signals.append({
                'type': 'short',
                'confidence': 0.5 - sentiment,  # 0-0.5 scale
                'source': 'sentiment_trend_following',
                'reason': f"Bearish sentiment ({sentiment:.2f}) with {trend} trend"
            })
        
        # REVERSAL SIGNALS
        if reversal:
            if reversal == 'bullish_reversal':
                signals.append({
                    'type': 'long',
                    'confidence': 0.3,  # Medium confidence
                    'source': 'sentiment_reversal',
                    'reason': f"Bullish sentiment reversal detected"
                })
            elif reversal == 'bearish_reversal':
                signals.append({
                    'type': 'short',
                    'confidence': 0.3,  # Medium confidence
                    'source': 'sentiment_reversal',
                    'reason': f"Bearish sentiment reversal detected"
                })
            elif reversal == 'extreme_bullish_reversal':
                signals.append({
                    'type': 'short',
                    'confidence': 0.4,  # Higher confidence for contrarian
                    'source': 'sentiment_contrarian',
                    'reason': f"Contrarian signal on extreme bullish sentiment reversal"
                })
            elif reversal == 'extreme_bearish_reversal':
                signals.append({
                    'type': 'long',
                    'confidence': 0.4,  # Higher confidence for contrarian
                    'source': 'sentiment_contrarian',
                    'reason': f"Contrarian signal on extreme bearish sentiment reversal"
                })
                
        # DIVERGENCE SIGNALS
        if abs(divergence) > 0.3:  # Significant divergence
            if divergence > 0.3:
                signals.append({
                    'type': 'long',
                    'confidence': min(0.5, divergence * 0.8),  # Scale confidence
                    'source': 'sentiment_divergence',
                    'reason': f"Bullish sentiment divergence ({divergence:.2f})"
                })
            elif divergence < -0.3:
                signals.append({
                    'type': 'short',
                    'confidence': min(0.5, abs(divergence) * 0.8),  # Scale confidence
                    'source': 'sentiment_divergence',
                    'reason': f"Bearish sentiment divergence ({divergence:.2f})"
                })
        
        # Add symbol to all signals
        for signal in signals:
            signal['symbol'] = symbol
            signal['timestamp'] = datetime.now()
        
        return signals
    
    def get_sentiment_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of current sentiment analysis.
        
        Returns:
            Dictionary with sentiment data and analysis
        """
        # Deep copy to avoid modifying original
        report = copy.deepcopy(self.sentiment_data)
        
        # Add additional context
        report['total_sources'] = len(self.sentiment_history)
        report['by_source'] = {}
        
        # Count items by source
        for source in self.parameters['sentiment_sources']:
            source_items = [e for e in self.sentiment_history if e['source'] == source]
            report['by_source'][source] = {
                'count': len(source_items),
                'average': sum(e['sentiment'] for e in source_items) / len(source_items) if source_items else 0.5
            }
        
        # Add sentiment signals summary (without requiring price data)
        trend_signal = None
        if report['current_score'] > self.parameters['bullish_threshold']:
            trend_signal = 'bullish'
        elif report['current_score'] < self.parameters['bearish_threshold']:
            trend_signal = 'bearish'
            
        report['signal_summary'] = {
            'trend': trend_signal,
            'reversal': report['reversal_signal'],
            'strength': abs(report['current_score'] - 0.5) * 2  # 0-1 scale
        }
        
        return report
