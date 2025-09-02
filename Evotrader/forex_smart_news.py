#!/usr/bin/env python3
"""
Forex Smart News Analysis - Enhanced News Intelligence Module

This module provides advanced news awareness capabilities:
- News impact prediction
- News volatility profiling
- Post-news entry strategies
- News-based position sizing
"""

import os
import sys
import yaml
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import re
from pathlib import Path
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_smart_news')


class SmartNewsAnalyzer:
    """
    Enhanced news intelligence for Forex trading.
    Provides advanced analytics and predictions for economic news events.
    """
    
    def __init__(self, news_guard=None, config: Dict[str, Any] = None):
        """
        Initialize the smart news analyzer.
        
        Args:
            news_guard: News guard instance
            config: Configuration dictionary
        """
        self.news_guard = news_guard
        self.config = config or {}
        
        # Historical news impact database
        self.news_impact_history = {}
        
        # News event categories and importance levels
        self.news_categories = {
            'interest_rate': {'importance': 5, 'volatility': 5},
            'gdp': {'importance': 4, 'volatility': 4},
            'employment': {'importance': 4, 'volatility': 4},
            'inflation': {'importance': 4, 'volatility': 3},
            'retail_sales': {'importance': 3, 'volatility': 3},
            'manufacturing': {'importance': 3, 'volatility': 2},
            'trade_balance': {'importance': 2, 'volatility': 2},
            'sentiment': {'importance': 2, 'volatility': 2},
            'housing': {'importance': 2, 'volatility': 1},
            'other': {'importance': 1, 'volatility': 1}
        }
        
        # Currency sensitivity to news categories
        self.currency_sensitivities = {
            'USD': {
                'interest_rate': 5, 'employment': 5, 'inflation': 4, 
                'gdp': 4, 'retail_sales': 3, 'manufacturing': 3
            },
            'EUR': {
                'interest_rate': 5, 'gdp': 4, 'inflation': 4, 
                'manufacturing': 4, 'sentiment': 3, 'trade_balance': 3
            },
            'GBP': {
                'interest_rate': 5, 'inflation': 5, 'employment': 4,
                'gdp': 4, 'retail_sales': 3, 'trade_balance': 2
            },
            'JPY': {
                'interest_rate': 5, 'trade_balance': 4, 'gdp': 3,
                'manufacturing': 3, 'inflation': 3, 'employment': 2
            },
            'AUD': {
                'interest_rate': 5, 'employment': 4, 'gdp': 4,
                'trade_balance': 4, 'inflation': 3, 'retail_sales': 3
            },
            'CAD': {
                'interest_rate': 5, 'employment': 4, 'gdp': 4,
                'trade_balance': 4, 'inflation': 3, 'retail_sales': 3
            },
            'NZD': {
                'interest_rate': 5, 'gdp': 4, 'trade_balance': 4,
                'employment': 3, 'inflation': 3, 'dairy_prices': 3
            },
            'CHF': {
                'interest_rate': 5, 'gdp': 3, 'trade_balance': 3,
                'inflation': 3, 'manufacturing': 2, 'retail_sales': 2
            }
        }
        
        # Load news impact history if available
        self._load_news_impact_history()
        
        # Initialize simple ML model for impact prediction
        self.news_impact_model = self._initialize_impact_model()
        
        logger.info("Smart News Analyzer initialized")
    
    def _load_news_impact_history(self) -> None:
        """Load historical news impact data from file if available."""
        impact_file = self.config.get('impact_history_file', 'news_impact_history.json')
        
        if os.path.exists(impact_file):
            try:
                with open(impact_file, 'r') as f:
                    self.news_impact_history = json.load(f)
                logger.info(f"Loaded news impact history from {impact_file}")
            except Exception as e:
                logger.error(f"Error loading news impact history: {e}")
    
    def save_news_impact_history(self) -> None:
        """Save news impact history to file."""
        impact_file = self.config.get('impact_history_file', 'news_impact_history.json')
        
        try:
            with open(impact_file, 'w') as f:
                json.dump(self.news_impact_history, f, indent=2)
            logger.info(f"Saved news impact history to {impact_file}")
        except Exception as e:
            logger.error(f"Error saving news impact history: {e}")
    
    def _initialize_impact_model(self) -> Dict[str, Any]:
        """
        Initialize a simple model for news impact prediction.
        
        In a production environment, this would be a trained ML model.
        For now, we'll use a rule-based approach with historical data.
        
        Returns:
            Model configuration dictionary
        """
        return {
            'type': 'rule_based',
            'version': '0.1',
            'default_impact': {
                'pips': 15.0,
                'duration_minutes': 60,
                'direction': 0  # 0 = neutral, 1 = positive for currency, -1 = negative
            }
        }
    
    def predict_news_impact(self, news_event: Dict[str, Any], pair: str) -> Dict[str, Any]:
        """
        Predict impact of news event on given pair.
        
        Args:
            news_event: News event data
            pair: Currency pair
            
        Returns:
            Dictionary with impact predictions
        """
        # Extract features from news event
        features = self._extract_news_features(news_event, pair)
        
        # Get currencies involved in the pair
        base_currency = pair[:3]
        quote_currency = pair[3:]
        
        # Determine relevant currencies for this news
        relevant_currency = None
        if 'country' in news_event:
            country_currency_map = {
                'US': 'USD', 'Euro Zone': 'EUR', 'UK': 'GBP', 'Japan': 'JPY',
                'Australia': 'AUD', 'Canada': 'CAD', 'New Zealand': 'NZD',
                'Switzerland': 'CHF'
            }
            if news_event['country'] in country_currency_map:
                relevant_currency = country_currency_map[news_event['country']]
        
        # If can't determine currency from country, try to infer from event title
        if not relevant_currency and 'title' in news_event:
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'NZD', 'CHF']
            for currency in currencies:
                if currency in news_event['title']:
                    relevant_currency = currency
                    break
        
        # Calculate base impact
        base_impact = self._calculate_base_impact(news_event, relevant_currency)
        
        # Adjust for pair specifics
        adjusted_impact = self._adjust_impact_for_pair(base_impact, relevant_currency, pair)
        
        # Check for historical impact data
        historical_impact = self._get_historical_impact(news_event, pair)
        
        # Blend predictions
        final_prediction = self._blend_predictions(adjusted_impact, historical_impact, features)
        
        return final_prediction
    
    def _extract_news_features(self, news_event: Dict[str, Any], pair: str) -> Dict[str, Any]:
        """
        Extract relevant features from news event.
        
        Args:
            news_event: News event data
            pair: Currency pair
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Basic news properties
        features['title'] = news_event.get('title', '')
        features['importance'] = news_event.get('importance', 'medium')
        features['country'] = news_event.get('country', '')
        features['forecast'] = news_event.get('forecast', None)
        features['previous'] = news_event.get('previous', None)
        
        # Determine news category
        features['category'] = self._determine_news_category(news_event)
        
        # Check if forecast is different from previous
        if features['forecast'] is not None and features['previous'] is not None:
            try:
                forecast_value = float(features['forecast'])
                previous_value = float(features['previous'])
                features['forecast_change'] = forecast_value - previous_value
                features['forecast_change_pct'] = (forecast_value / previous_value - 1) * 100 if previous_value != 0 else 0
            except (ValueError, TypeError):
                features['forecast_change'] = None
                features['forecast_change_pct'] = None
        else:
            features['forecast_change'] = None
            features['forecast_change_pct'] = None
        
        # Affected currencies
        base_currency = pair[:3]
        quote_currency = pair[3:]
        features['base_currency'] = base_currency
        features['quote_currency'] = quote_currency
        
        # Determine directly affected currency
        affected_currency = None
        if features['country'] == 'US':
            affected_currency = 'USD'
        elif features['country'] == 'Euro Zone':
            affected_currency = 'EUR'
        elif features['country'] == 'UK':
            affected_currency = 'GBP'
        elif features['country'] == 'Japan':
            affected_currency = 'JPY'
        elif features['country'] == 'Australia':
            affected_currency = 'AUD'
        elif features['country'] == 'Canada':
            affected_currency = 'CAD'
        elif features['country'] == 'New Zealand':
            affected_currency = 'NZD'
        elif features['country'] == 'Switzerland':
            affected_currency = 'CHF'
        
        features['affected_currency'] = affected_currency
        
        # Is affected currency in pair?
        features['is_base_affected'] = affected_currency == base_currency
        features['is_quote_affected'] = affected_currency == quote_currency
        
        return features
    
    def _determine_news_category(self, news_event: Dict[str, Any]) -> str:
        """
        Determine the category of a news event.
        
        Args:
            news_event: News event data
            
        Returns:
            Category string
        """
        title = news_event.get('title', '').lower()
        
        # Check for keywords in title
        if any(kw in title for kw in ['rate', 'interest', 'fomc', 'fed', 'boe', 'ecb', 'rba', 'boc']):
            return 'interest_rate'
        elif any(kw in title for kw in ['gdp', 'growth']):
            return 'gdp'
        elif any(kw in title for kw in ['nfp', 'payroll', 'employment', 'unemployment', 'jobless']):
            return 'employment'
        elif any(kw in title for kw in ['cpi', 'inflation', 'price index', 'ppi']):
            return 'inflation'
        elif any(kw in title for kw in ['retail', 'sales', 'consumer spending']):
            return 'retail_sales'
        elif any(kw in title for kw in ['pmi', 'manufacturing', 'industrial', 'production']):
            return 'manufacturing'
        elif any(kw in title for kw in ['trade', 'export', 'import', 'balance']):
            return 'trade_balance'
        elif any(kw in title for kw in ['sentiment', 'confidence', 'survey']):
            return 'sentiment'
        elif any(kw in title for kw in ['home', 'housing', 'building', 'construction']):
            return 'housing'
        
        # Default category
        return 'other'
    
    def _calculate_base_impact(self, news_event: Dict[str, Any], currency: Optional[str]) -> Dict[str, Any]:
        """
        Calculate base impact of news event.
        
        Args:
            news_event: News event data
            currency: Affected currency
            
        Returns:
            Dictionary with base impact
        """
        # Default impact
        base_impact = {
            'pips': 15.0,
            'duration_minutes': 60,
            'direction': 0  # 0 = neutral, 1 = positive for currency, -1 = negative
        }
        
        # Determine category
        category = self._determine_news_category(news_event)
        
        # Get importance level from news event or default to medium
        importance = news_event.get('importance', 'medium')
        importance_factor = 1.0
        if importance == 'high':
            importance_factor = 2.0
        elif importance == 'low':
            importance_factor = 0.5
        
        # Adjust base impact by category
        if category in self.news_categories:
            cat_info = self.news_categories[category]
            base_impact['pips'] = cat_info['volatility'] * 5.0 * importance_factor
            base_impact['duration_minutes'] = cat_info['volatility'] * 30 * importance_factor
        
        # Adjust by currency sensitivity if available
        if currency and category in self.currency_sensitivities.get(currency, {}):
            sensitivity = self.currency_sensitivities[currency][category]
            base_impact['pips'] *= sensitivity / 3.0
        
        # Try to determine direction from forecast vs previous
        if 'forecast' in news_event and 'previous' in news_event:
            try:
                forecast = float(news_event['forecast'])
                previous = float(news_event['previous'])
                
                # Determine if higher is better (depends on category)
                higher_is_better = True
                if category in ['unemployment', 'inflation']:
                    higher_is_better = False
                
                if forecast > previous:
                    base_impact['direction'] = 1 if higher_is_better else -1
                elif forecast < previous:
                    base_impact['direction'] = -1 if higher_is_better else 1
            except (ValueError, TypeError):
                pass
        
        return base_impact
    
    def _adjust_impact_for_pair(self, base_impact: Dict[str, Any], 
                              affected_currency: Optional[str], 
                              pair: str) -> Dict[str, Any]:
        """
        Adjust impact prediction for specific currency pair.
        
        Args:
            base_impact: Base impact prediction
            affected_currency: Currency affected by the news
            pair: Currency pair
            
        Returns:
            Adjusted impact prediction
        """
        adjusted_impact = base_impact.copy()
        
        base_currency = pair[:3]
        quote_currency = pair[3:]
        
        # If affected currency is not in the pair, reduce impact
        if affected_currency and affected_currency not in pair:
            adjusted_impact['pips'] *= 0.3
            adjusted_impact['duration_minutes'] *= 0.5
            return adjusted_impact
        
        # Adjust direction based on which currency in the pair is affected
        if affected_currency == base_currency:
            # Direction stays the same - base currency strengthening means pair goes up
            pass
        elif affected_currency == quote_currency:
            # Reverse direction - quote currency strengthening means pair goes down
            adjusted_impact['direction'] *= -1
        
        # JPY pairs need special handling for pip value
        if 'JPY' in pair:
            adjusted_impact['pips'] *= 0.01  # Convert to JPY pips
        
        return adjusted_impact
    
    def _get_historical_impact(self, news_event: Dict[str, Any], pair: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve historical impact data for similar news events.
        
        Args:
            news_event: News event data
            pair: Currency pair
            
        Returns:
            Historical impact data or None if not available
        """
        # Create a key for this type of news event
        country = news_event.get('country', '')
        title = news_event.get('title', '')
        category = self._determine_news_category(news_event)
        
        event_key = f"{country}_{category}"
        pair_key = f"{event_key}_{pair}"
        
        # Try to get impact data for this specific event and pair
        if pair_key in self.news_impact_history:
            return self.news_impact_history[pair_key]
        
        # Try to get impact data for this event type across all pairs
        if event_key in self.news_impact_history:
            return self.news_impact_history[event_key]
        
        # No historical data available
        return None
    
    def _blend_predictions(self, model_prediction: Dict[str, Any], 
                         historical_prediction: Optional[Dict[str, Any]], 
                         features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Blend model and historical predictions.
        
        Args:
            model_prediction: Model-based prediction
            historical_prediction: Historical data-based prediction
            features: News event features
            
        Returns:
            Blended prediction
        """
        # Start with model prediction
        final_prediction = model_prediction.copy()
        
        # Blend with historical data if available
        if historical_prediction:
            # Determine weight for historical data (0.0-1.0)
            # More weight if we have a lot of historical data
            history_weight = historical_prediction.get('sample_size', 1) / 10.0
            history_weight = min(history_weight, 0.7)  # Cap at 0.7
            
            # Blend predictions
            final_prediction['pips'] = (
                (1 - history_weight) * model_prediction['pips'] +
                history_weight * historical_prediction.get('pips', model_prediction['pips'])
            )
            
            final_prediction['duration_minutes'] = (
                (1 - history_weight) * model_prediction['duration_minutes'] +
                history_weight * historical_prediction.get('duration_minutes', model_prediction['duration_minutes'])
            )
            
            # Direction is trickier to blend - use historical if we're confident
            if abs(historical_prediction.get('direction', 0)) > 0.7:
                final_prediction['direction'] = historical_prediction['direction']
        
        # Add confidence score
        if features.get('forecast_change') is not None:
            # Higher confidence if forecast differs significantly from previous
            forecast_diff_pct = abs(features.get('forecast_change_pct', 0))
            diff_confidence = min(forecast_diff_pct / 2.0, 0.5)  # Cap at 0.5
            
            # Base confidence on importance
            importance_confidence = 0.3
            if features.get('importance') == 'high':
                importance_confidence = 0.7
            elif features.get('importance') == 'medium':
                importance_confidence = 0.5
            
            # Blend confidence factors
            final_prediction['confidence'] = (diff_confidence + importance_confidence) / 2
        else:
            # Default confidence based on importance
            if features.get('importance') == 'high':
                final_prediction['confidence'] = 0.7
            elif features.get('importance') == 'medium':
                final_prediction['confidence'] = 0.5
            else:
                final_prediction['confidence'] = 0.3
        
        # Add additional information
        final_prediction['category'] = features.get('category', 'other')
        final_prediction['affected_currency'] = features.get('affected_currency', None)
        
        return final_prediction
    
    def update_news_impact_history(self, news_event: Dict[str, Any], 
                                 pair: str, 
                                 actual_impact: Dict[str, Any]) -> None:
        """
        Update historical impact data with actual observed impact.
        
        Args:
            news_event: News event data
            pair: Currency pair
            actual_impact: Observed impact data
        """
        # Create keys for this type of news event
        country = news_event.get('country', '')
        category = self._determine_news_category(news_event)
        
        event_key = f"{country}_{category}"
        pair_key = f"{event_key}_{pair}"
        
        # Update pair-specific history
        if pair_key not in self.news_impact_history:
            self.news_impact_history[pair_key] = {
                'pips': actual_impact['pips'],
                'duration_minutes': actual_impact['duration_minutes'],
                'direction': actual_impact['direction'],
                'sample_size': 1
            }
        else:
            history = self.news_impact_history[pair_key]
            sample_size = history['sample_size']
            new_sample_size = sample_size + 1
            
            # Exponential moving average update
            alpha = 0.3  # Weight for new observation
            history['pips'] = (1 - alpha) * history['pips'] + alpha * actual_impact['pips']
            history['duration_minutes'] = (1 - alpha) * history['duration_minutes'] + alpha * actual_impact['duration_minutes']
            history['direction'] = (1 - alpha) * history['direction'] + alpha * actual_impact['direction']
            history['sample_size'] = new_sample_size
            
            self.news_impact_history[pair_key] = history
        
        # Also update event-type history (across all pairs)
        if event_key not in self.news_impact_history:
            self.news_impact_history[event_key] = {
                'pips': actual_impact['pips'],
                'duration_minutes': actual_impact['duration_minutes'],
                'direction': 0,  # Direction is pair-specific, so use neutral for general history
                'sample_size': 1
            }
        else:
            history = self.news_impact_history[event_key]
            sample_size = history['sample_size']
            new_sample_size = sample_size + 1
            
            # Exponential moving average update
            alpha = 0.2  # Weight for new observation
            history['pips'] = (1 - alpha) * history['pips'] + alpha * actual_impact['pips']
            history['duration_minutes'] = (1 - alpha) * history['duration_minutes'] + alpha * actual_impact['duration_minutes']
            history['sample_size'] = new_sample_size
            
            self.news_impact_history[event_key] = history
    
    def calculate_news_position_size(self, normal_size: float, 
                                   news_events: List[Dict[str, Any]], 
                                   pair: str, 
                                   hours_until_nearest: float) -> float:
        """
        Calculate appropriate position size based on upcoming news events.
        
        Args:
            normal_size: Normal position size
            news_events: List of upcoming news events
            pair: Currency pair
            hours_until_nearest: Hours until nearest event
            
        Returns:
            Adjusted position size
        """
        # Default to normal size
        adjusted_size = normal_size
        
        # If no events, return normal size
        if not news_events:
            return normal_size
        
        # Prepare to analyze impact of each event
        relevant_events = []
        for event in news_events:
            impact = self.predict_news_impact(event, pair)
            hours_until = (event.get('timestamp', datetime.datetime.now()) - datetime.datetime.now()).total_seconds() / 3600
            
            # Only consider events in next 24 hours
            if hours_until <= 24:
                relevant_events.append({
                    'event': event,
                    'impact': impact,
                    'hours_until': hours_until
                })
        
        # If no relevant events, return normal size
        if not relevant_events:
            return normal_size
        
        # Sort by time
        relevant_events.sort(key=lambda x: x['hours_until'])
        
        # Get nearest event
        nearest_event = relevant_events[0]
        
        # Calculate reduction factor based on proximity and predicted impact
        if nearest_event['hours_until'] < 1:
            # Very close to news - reduce size dramatically
            time_factor = 0.2
        elif nearest_event['hours_until'] < 3:
            # Within a few hours - moderate reduction
            time_factor = 0.5
        elif nearest_event['hours_until'] < 6:
            # Several hours away - slight reduction
            time_factor = 0.8
        else:
            # Far away - no reduction
            time_factor = 1.0
        
        # Consider impact magnitude
        impact_factor = 1.0
        impact_pips = nearest_event['impact'].get('pips', 0)
        impact_confidence = nearest_event['impact'].get('confidence', 0.5)
        
        if impact_pips > 50 and impact_confidence > 0.6:
            # High impact, high confidence - reduce size more
            impact_factor = 0.3
        elif impact_pips > 30 and impact_confidence > 0.5:
            # Medium impact - moderate reduction
            impact_factor = 0.6
        elif impact_pips > 15:
            # Lower impact - slight reduction
            impact_factor = 0.8
        
        # Combined reduction factor
        reduction_factor = min(time_factor, impact_factor)
        
        # Apply reduction
        adjusted_size = normal_size * reduction_factor
        
        return adjusted_size
    
    def is_post_news_entry_opportunity(self, news_event: Dict[str, Any], 
                                      pair: str, 
                                      minutes_since_news: float,
                                      market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Determine if current conditions represent a post-news entry opportunity.
        
        Args:
            news_event: News event data
            pair: Currency pair
            minutes_since_news: Minutes elapsed since news release
            market_data: Recent market data (optional)
            
        Returns:
            Dictionary with opportunity assessment
        """
        # Predict impact of this news event
        impact = self.predict_news_impact(news_event, pair)
        
        # Calculate expected volatility decay
        expected_duration = impact.get('duration_minutes', 60)
        volatility_decay = math.exp(-minutes_since_news / expected_duration)
        current_volatility = impact.get('pips', 15) * volatility_decay
        
        # Determine if we've passed peak volatility but still have momentum
        is_opportunity = False
        entry_type = None
        confidence = 0.0
        
        if minutes_since_news > 5 and minutes_since_news < expected_duration * 0.5:
            # We're past initial volatility spike but still in active phase
            # This is often a good time to enter in the direction of the move
            is_opportunity = True
            entry_type = 'trend_continuation'
            
            # Direction confidence
            direction = impact.get('direction', 0)
            if abs(direction) > 0.7:
                confidence = 0.7
            elif abs(direction) > 0.3:
                confidence = 0.5
            else:
                confidence = 0.3
                
        elif minutes_since_news >= expected_duration * 0.5 and minutes_since_news < expected_duration:
            # We're in the later phase of the news reaction
            # This can be a good time for mean reversion if the move was exaggerated
            is_opportunity = True
            entry_type = 'mean_reversion'
            confidence = 0.6
        
        # Look at market data if available to confirm
        if market_data is not None and len(market_data) > 5:
            # Simple trend detection
            recent_close = market_data['close'].iloc[-1]
            pre_news_close = market_data['close'].iloc[0]
            actual_move = (recent_close - pre_news_close) * self._get_pip_multiplier(pair)
            
            # Determine actual direction
            actual_direction = 1 if actual_move > 0 else -1 if actual_move < 0 else 0
            
            # Compare with predicted direction
            predicted_direction = impact.get('direction', 0)
            
            # If directions match for trend continuation, increase confidence
            if entry_type == 'trend_continuation' and actual_direction * predicted_direction > 0:
                confidence += 0.1
            # If directions oppose for mean reversion, increase confidence
            elif entry_type == 'mean_reversion' and actual_direction * predicted_direction < 0:
                confidence += 0.1
        
        return {
            'is_opportunity': is_opportunity,
            'entry_type': entry_type,
            'direction': impact.get('direction', 0),
            'confidence': min(confidence, 0.9),
            'remaining_volatility': current_volatility,
            'minutes_since_news': minutes_since_news
        }
    
    def _get_pip_multiplier(self, pair: str) -> float:
        """
        Get pip multiplier for a currency pair.
        
        Args:
            pair: Currency pair
            
        Returns:
            Pip multiplier
        """
        if 'JPY' in pair:
            return 100  # JPY pairs have 2 decimal places
        else:
            return 10000  # Most pairs have 4 decimal places


# Test function
def test_smart_news():
    """Test the smart news analyzer functionality."""
    analyzer = SmartNewsAnalyzer()
    
    # Test impact prediction
    test_event = {
        'title': 'US Non-Farm Payrolls',
        'importance': 'high',
        'country': 'US',
        'forecast': '200K',
        'previous': '180K'
    }
    
    impact = analyzer.predict_news_impact(test_event, 'EURUSD')
    print(f"Predicted impact for EURUSD on US NFP: {impact}")
    
    # Test news-based position sizing
    adjusted_size = analyzer.calculate_news_position_size(0.1, [test_event], 'EURUSD', 2.5)
    print(f"Adjusted position size: {adjusted_size:.2f} lots (normal: 0.10 lots)")
    
    # Test post-news opportunity detection
    opportunity = analyzer.is_post_news_entry_opportunity(test_event, 'EURUSD', 15)
    print(f"Post-news opportunity: {opportunity}")
    
    return "Smart news tests completed"


if __name__ == "__main__":
    test_smart_news()
