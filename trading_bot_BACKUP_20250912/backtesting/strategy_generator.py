#!/usr/bin/env python3
"""
Strategy Generator for Autonomous ML Backtesting

This module creates trading strategies based on ML analysis of market data,
news sentiment, and historical patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

class StrategyTemplateLibrary:
    """Library of strategy templates that can be configured with parameters"""
    
    def __init__(self):
        self.templates = {
            'moving_average_crossover': {
                'name': 'Moving Average Crossover',
                'description': 'Strategy based on fast MA crossing above slow MA',
                'default_params': {
                    'fast_period': 20,
                    'slow_period': 50,
                    'signal_lookback': 3
                },
                'param_ranges': {
                    'fast_period': (5, 50),
                    'slow_period': (20, 200),
                    'signal_lookback': (1, 10)
                }
            },
            'rsi_reversal': {
                'name': 'RSI Reversal',
                'description': 'Mean reversion strategy using RSI indicator',
                'default_params': {
                    'rsi_period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                },
                'param_ranges': {
                    'rsi_period': (7, 21),
                    'oversold_threshold': (20, 40),
                    'overbought_threshold': (60, 80)
                }
            },
            'breakout_momentum': {
                'name': 'Breakout Momentum',
                'description': 'Volatility breakout strategy with momentum confirmation',
                'default_params': {
                    'breakout_period': 20,
                    'volume_factor': 1.5,
                    'momentum_period': 10
                },
                'param_ranges': {
                    'breakout_period': (10, 50),
                    'volume_factor': (1.2, 3.0),
                    'momentum_period': (5, 15)
                }
            },
            'news_sentiment_momentum': {
                'name': 'News Sentiment Momentum',
                'description': 'Strategy combining news sentiment with price momentum',
                'default_params': {
                    'sentiment_threshold': 0.2,
                    'momentum_period': 5,
                    'holding_period': 3
                },
                'param_ranges': {
                    'sentiment_threshold': (0.1, 0.5),
                    'momentum_period': (3, 15),
                    'holding_period': (1, 10)
                }
            },
            'dual_momentum': {
                'name': 'Dual Momentum',
                'description': 'Absolute and relative momentum strategy',
                'default_params': {
                    'lookback_period': 12,
                    'momentum_threshold': 0.0
                },
                'param_ranges': {
                    'lookback_period': (1, 24),
                    'momentum_threshold': (-0.05, 0.05)
                }
            }
        }
    
    def get_template(self, template_name):
        """
        Get a strategy template by name
        
        Args:
            template_name: Name of the template
            
        Returns:
            dict: Template definition or None if not found
        """
        return self.templates.get(template_name)
    
    def get_all_templates(self):
        """
        Get all available templates
        
        Returns:
            dict: All templates
        """
        return self.templates


class RiskManager:
    """Manages risk parameters for trading strategies"""
    
    def __init__(self, default_risk_level='moderate'):
        """
        Initialize risk manager
        
        Args:
            default_risk_level: Default risk level (conservative, moderate, aggressive)
        """
        self.risk_levels = {
            'conservative': {
                'max_position_size': 0.05,  # 5% of portfolio
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'max_trades_per_day': 2,
                'trailing_stop_enabled': True
            },
            'moderate': {
                'max_position_size': 0.10,  # 10% of portfolio
                'stop_loss_atr_multiple': 1.5,
                'take_profit_atr_multiple': 2.5,
                'max_trades_per_day': 4,
                'trailing_stop_enabled': True
            },
            'aggressive': {
                'max_position_size': 0.20,  # 20% of portfolio
                'stop_loss_atr_multiple': 1.0,
                'take_profit_atr_multiple': 2.0,
                'max_trades_per_day': 8,
                'trailing_stop_enabled': False
            }
        }
        self.default_risk_level = default_risk_level
    
    def calculate_risk_parameters(self, strategy_name, strategy_params, price_data, risk_level=None):
        """
        Calculate risk parameters for a strategy
        
        Args:
            strategy_name: Name of the strategy
            strategy_params: Strategy parameters
            price_data: Historical price data
            risk_level: Risk level or None to use default
            
        Returns:
            dict: Risk parameters
        """
        if risk_level is None:
            risk_level = self.default_risk_level
            
        # Ensure valid risk level
        if risk_level not in self.risk_levels:
            logger.warning(f"Invalid risk level: {risk_level}. Using {self.default_risk_level}.")
            risk_level = self.default_risk_level
            
        risk_profile = self.risk_levels[risk_level]
        
        # Calculate Average True Range for dynamic stop loss
        atr = self._calculate_atr(price_data, period=14)
        
        # Get last price
        last_price = price_data['close'].iloc[-1] if price_data is not None and len(price_data) > 0 else 100.0
        
        # Calculate stop loss and take profit levels
        stop_loss_amount = atr * risk_profile['stop_loss_atr_multiple']
        stop_loss_percentage = (stop_loss_amount / last_price) * 100
        
        take_profit_amount = atr * risk_profile['take_profit_atr_multiple']
        take_profit_percentage = (take_profit_amount / last_price) * 100
        
        # Adjust position size based on volatility
        volatility_adjustment = 1.0
        if atr > 0:
            volatility_adjustment = min(1.0, 0.02 / (atr / last_price))
            
        position_size = risk_profile['max_position_size'] * volatility_adjustment
        
        return {
            'risk_level': risk_level,
            'position_size': position_size,
            'stop_loss_percentage': stop_loss_percentage,
            'take_profit_percentage': take_profit_percentage,
            'max_trades_per_day': risk_profile['max_trades_per_day'],
            'trailing_stop_enabled': risk_profile['trailing_stop_enabled'],
            'atr_value': atr,
            'volatility_adjustment': volatility_adjustment
        }
        
    def _calculate_atr(self, price_data, period=14):
        """
        Calculate Average True Range
        
        Args:
            price_data: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            float: ATR value
        """
        if price_data is None or len(price_data) < period:
            return 1.0  # Default value
            
        try:
            # Calculate true range
            high = price_data['high']
            low = price_data['low']
            close = price_data['close']
            
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return 1.0  # Default value


class MLStrategyModel:
    """Mock ML model for strategy selection and optimization"""
    
    def __init__(self):
        """Initialize ML model for strategy selection"""
        self.strategy_performance = {
            'moving_average_crossover': 0.6,
            'rsi_reversal': 0.7,
            'breakout_momentum': 0.8,
            'news_sentiment_momentum': 0.75,
            'dual_momentum': 0.65
        }
        logger.info("MLStrategyModel initialized (mock implementation)")
        
    def predict_strategy_performance(self, features):
        """
        Predict performance scores for strategies
        
        Args:
            features: Features extracted from market data
            
        Returns:
            dict: Strategy performance scores
        """
        # In a real implementation, this would use an actual ML model
        # For now, we'll add some randomness to the base scores
        
        # Create a copy of base scores
        scores = self.strategy_performance.copy()
        
        # Add random adjustment based on sentiment
        sentiment = features.get('sentiment', 0)
        market_trend = features.get('market_trend', 0)
        
        # Adjust scores based on market conditions
        for strategy, score in scores.items():
            # Add some randomness
            random_factor = np.random.normal(0, 0.1)
            
            # Adjust for market trend
            if strategy == 'rsi_reversal':
                # Mean reversion works better in sideways markets
                trend_adjustment = -0.1 * abs(market_trend)
            elif strategy in ['breakout_momentum', 'news_sentiment_momentum']:
                # Momentum strategies work better in trending markets
                trend_adjustment = 0.2 * abs(market_trend)
            else:
                trend_adjustment = 0.1 * market_trend
                
            # Adjust for sentiment
            if strategy == 'news_sentiment_momentum':
                # News sentiment strategy works better with strong sentiment
                sentiment_adjustment = 0.3 * abs(sentiment)
            else:
                sentiment_adjustment = 0.1 * sentiment
                
            # Apply adjustments
            scores[strategy] = min(1.0, max(0.1, score + random_factor + trend_adjustment + sentiment_adjustment))
            
        return scores
        
    def optimize_parameters(self, template, integrated_data):
        """
        Optimize strategy parameters
        
        Args:
            template: Strategy template
            integrated_data: Integrated market and news data
            
        Returns:
            dict: Optimized parameters
        """
        if not template:
            return {}
            
        # Start with default parameters
        params = template['default_params'].copy()
        param_ranges = template['param_ranges']
        
        # In a real implementation, this would use ML optimization
        # For now, we'll randomly adjust parameters within allowed ranges
        for param, value in params.items():
            if param in param_ranges:
                min_val, max_val = param_ranges[param]
                
                # Adjust parameter randomly within range
                random_factor = np.random.random() * 0.4 + 0.8  # 0.8 to 1.2
                
                # Calculate new value within bounds
                new_value = value * random_factor
                new_value = max(min_val, min(max_val, new_value))
                
                # Round to integer if the original value was an integer
                if isinstance(value, int):
                    new_value = int(round(new_value))
                    
                params[param] = new_value
                
        return params


class StrategyGenerator:
    """Generates trading strategies based on ML analysis"""
    
    def __init__(self, ml_model=None, strategy_templates=None, risk_manager=None):
        """
        Initialize with ML model and strategy templates
        
        Parameters:
            ml_model: Trained ML model for strategy selection
            strategy_templates: Library of strategy templates
            risk_manager: Risk management component
        """
        self.ml_model = ml_model if ml_model else MLStrategyModel()
        self.strategy_templates = strategy_templates if strategy_templates else StrategyTemplateLibrary()
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        logger.info("StrategyGenerator initialized")
        
    def generate_strategies(self, integrated_data, num_strategies=5):
        """
        Generate strategies based on integrated data
        
        Parameters:
            integrated_data: Combined dataset from DataIntegrationLayer
            num_strategies: Number of strategies to generate
            
        Returns:
            list: Generated strategies with parameters
        """
        logger.info(f"Generating {num_strategies} strategies")
        
        # Extract features for ML model
        features = self._extract_features(integrated_data)
        
        # Use ML to select best strategy types for current conditions
        strategy_scores = self.ml_model.predict_strategy_performance(features)
        
        # Select top-performing strategy templates
        top_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[:num_strategies]
        
        # Generate complete strategies with parameters
        complete_strategies = []
        for strategy_name, score in top_strategies:
            template = self.strategy_templates.get_template(strategy_name)
            
            if not template:
                logger.warning(f"Template not found for strategy: {strategy_name}")
                continue
                
            # Optimize parameters for this strategy
            params = self.ml_model.optimize_parameters(template, integrated_data)
            
            # Apply risk management rules
            risk_params = self.risk_manager.calculate_risk_parameters(
                strategy_name, 
                params,
                integrated_data.get("price_data")
            )
            
            # Compile full strategy
            complete_strategy = {
                "id": f"ml_{strategy_name}_{str(uuid.uuid4())[:8]}",
                "name": f"{template['name']} v{self._get_version(strategy_name)}",
                "template": strategy_name,
                "params": params,
                "risk_params": risk_params,
                "confidence_score": score,
                "timestamp": datetime.now(),
                "reasoning": self._generate_reasoning(strategy_name, integrated_data, params)
            }
            
            complete_strategies.append(complete_strategy)
            
        return complete_strategies
        
    def _extract_features(self, integrated_data):
        """
        Extract features from integrated data for ML model
        
        Args:
            integrated_data: Combined dataset
            
        Returns:
            dict: Features for ML model
        """
        features = {}
        
        # Extract sentiment features
        sentiment_data = integrated_data.get('sentiment', {})
        features['sentiment'] = sentiment_data.get('overall_sentiment', 0)
        features['political_sentiment'] = sentiment_data.get('political_sentiment', 0)
        features['social_sentiment'] = sentiment_data.get('social_sentiment', 0)
        features['economic_sentiment'] = sentiment_data.get('economic_sentiment', 0)
        features['news_count'] = sentiment_data.get('news_count', 0)
        
        # Extract market trend features from price data
        price_data = integrated_data.get('price_data')
        if price_data is not None and len(price_data) > 0:
            # Calculate short-term trend (last 20 days)
            close_prices = price_data['close']
            if len(close_prices) >= 20:
                short_term_return = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1)
                features['market_trend'] = short_term_return
            else:
                features['market_trend'] = 0
                
            # Add volatility feature
            if len(close_prices) >= 20:
                returns = close_prices.pct_change().dropna()
                features['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            else:
                features['volatility'] = 0.2  # Default value
        else:
            features['market_trend'] = 0
            features['volatility'] = 0.2
            
        # Extract indicator features
        indicators = integrated_data.get('indicators', {})
        features['rsi'] = indicators.get('rsi', {}).get('value', 50)
        
        # Add more features as needed...
        
        return features
        
    def _generate_reasoning(self, strategy_name, data, params):
        """
        Generate detailed reasoning for why this strategy was selected
        
        Returns:
            dict: Multi-dimensional reasoning explanation
        """
        # Extract key insights from data
        news_insights = self._extract_news_insights(data.get('news', []), data.get('sentiment', {}))
        technical_insights = self._extract_technical_insights(data.get('indicators', {}))
        
        # Generate reasoning text
        political_factors = self._extract_political_factors(data.get('sentiment', {}))
        social_factors = self._extract_social_factors(data.get('sentiment', {}))
        economic_factors = self._extract_economic_factors(data.get('sentiment', {}), data.get('indicators', {}))
        
        # Create strategy-specific reasoning
        strategy_specific = self._get_strategy_specific_reasoning(strategy_name, params, data)
        
        return {
            "summary": f"Selected {strategy_name} based on {len(news_insights)} news factors and {len(technical_insights)} technical indicators",
            "political_factors": political_factors,
            "social_factors": social_factors,
            "economic_factors": economic_factors,
            "technical_indicators": technical_insights,
            "news_sentiment": news_insights,
            "parameter_justification": self._justify_parameters(params),
            "strategy_specific": strategy_specific
        }
        
    def _extract_news_insights(self, news, sentiment):
        """Extract key insights from news articles"""
        insights = []
        
        # Get top articles by sentiment impact
        analyzed_articles = sentiment.get('analyzed_articles', [])
        
        # Sort by absolute score to get most impactful articles
        sorted_articles = sorted(analyzed_articles, key=lambda x: abs(x.get('overall_score', 0)), reverse=True)
        
        # Take top articles
        for article in sorted_articles[:5]:
            score = article.get('overall_score', 0)
            insights.append({
                'headline': article.get('headline', 'Unknown'),
                'source': article.get('source', 'Unknown'),
                'sentiment': score,
                'impact': 'High' if abs(score) > 0.7 else 'Medium' if abs(score) > 0.3 else 'Low'
            })
            
        return insights
        
    def _extract_technical_insights(self, indicators):
        """Extract insights from technical indicators"""
        insights = []
        
        for indicator_name, indicator_data in indicators.items():
            # Skip if not a dictionary
            if not isinstance(indicator_data, dict):
                continue
                
            insights.append({
                'name': indicator_name.upper() if indicator_name.isupper() else indicator_name.replace('_', ' ').title(),
                'value': indicator_data.get('value', 0),
                'signal': indicator_data.get('signal', 'neutral'),
                'description': indicator_data.get('description', f"{indicator_name} analysis")
            })
            
        return insights
        
    def _extract_political_factors(self, sentiment):
        """Extract political factors from sentiment"""
        factors = []
        political_sentiment = sentiment.get('political_sentiment', 0)
        
        if abs(political_sentiment) < 0.1:
            factors.append("Neutral political environment with minimal impact on markets")
        elif political_sentiment > 0.5:
            factors.append("Highly favorable political environment for markets")
            factors.append("Supportive policy outlook with potential positive impacts")
        elif political_sentiment > 0.2:
            factors.append("Moderately positive political sentiment")
            factors.append("Some supportive policies detected in news coverage")
        elif political_sentiment < -0.5:
            factors.append("Challenging political environment with potential negative impact")
            factors.append("Regulatory or policy concerns detected in news coverage")
        elif political_sentiment < -0.2:
            factors.append("Slightly negative political sentiment")
            factors.append("Minor regulatory concerns detected in news coverage")
            
        return factors
        
    def _extract_social_factors(self, sentiment):
        """Extract social factors from sentiment"""
        factors = []
        social_sentiment = sentiment.get('social_sentiment', 0)
        
        if abs(social_sentiment) < 0.1:
            factors.append("Neutral social sentiment with minimal market impact")
        elif social_sentiment > 0.5:
            factors.append("Strongly positive social perception")
            factors.append("Potential brand value appreciation due to positive social factors")
        elif social_sentiment > 0.2:
            factors.append("Moderately positive social sentiment")
            factors.append("Some positive social responsibility factors detected")
        elif social_sentiment < -0.5:
            factors.append("Significant social controversy or negative perception")
            factors.append("Potential brand value impact from negative social factors")
        elif social_sentiment < -0.2:
            factors.append("Slightly negative social perception")
            factors.append("Minor social concerns detected in news coverage")
            
        return factors
        
    def _extract_economic_factors(self, sentiment, indicators):
        """Extract economic factors from sentiment and indicators"""
        factors = []
        economic_sentiment = sentiment.get('economic_sentiment', 0)
        
        # Economic sentiment factors
        if abs(economic_sentiment) < 0.1:
            factors.append("Neutral economic outlook in news coverage")
        elif economic_sentiment > 0.5:
            factors.append("Highly positive economic signals in news coverage")
            factors.append("Strong earnings or growth expectations")
        elif economic_sentiment > 0.2:
            factors.append("Moderately positive economic sentiment")
            factors.append("Favorable financial news detected")
        elif economic_sentiment < -0.5:
            factors.append("Strongly negative economic indicators in news")
            factors.append("Concerns about financial performance or sector challenges")
        elif economic_sentiment < -0.2:
            factors.append("Slight economic concerns detected in news coverage")
            
        # Add indicator-based economic factors
        if indicators:
            # Example: add factors based on moving averages
            ma_data = indicators.get('moving_averages', {})
            if ma_data:
                ma_signal = ma_data.get('signal')
                if ma_signal == 'bullish':
                    factors.append("Positive trend indicated by moving averages")
                elif ma_signal == 'bearish':
                    factors.append("Negative trend indicated by moving averages")
            
        return factors
        
    def _justify_parameters(self, params):
        """Generate justification for parameter selections"""
        justifications = []
        
        for param, value in params.items():
            param_name = param.replace('_', ' ').title()
            
            if 'period' in param:
                justifications.append(f"{param_name} of {value} optimized for current market conditions")
            elif 'threshold' in param:
                justifications.append(f"{param_name} set to {value} based on recent market volatility and trend analysis")
            else:
                justifications.append(f"{param_name} optimized to {value}")
                
        return justifications
        
    def _get_strategy_specific_reasoning(self, strategy_name, params, data):
        """Generate strategy-specific reasoning"""
        if strategy_name == 'moving_average_crossover':
            return f"Moving average crossover strategy selected due to trending market conditions. Fast period of {params.get('fast_period')} and slow period of {params.get('slow_period')} optimized for current market volatility."
            
        elif strategy_name == 'rsi_reversal':
            return f"RSI reversal strategy selected for potential mean reversion opportunities. RSI period of {params.get('rsi_period')} chosen to balance sensitivity with reliability."
            
        elif strategy_name == 'breakout_momentum':
            return f"Breakout momentum strategy selected due to building volatility. Breakout period of {params.get('breakout_period')} days with {params.get('volume_factor')}x volume confirmation."
            
        elif strategy_name == 'news_sentiment_momentum':
            sentiment = data.get('sentiment', {}).get('overall_sentiment', 0)
            sentiment_str = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
            return f"News sentiment momentum strategy selected due to {sentiment_str} news sentiment ({sentiment:.2f}). Strategy will use {params.get('momentum_period')} day momentum confirmation with sentiment threshold of {params.get('sentiment_threshold')}."
            
        elif strategy_name == 'dual_momentum':
            return f"Dual momentum strategy selected to capture both absolute and relative momentum. Using {params.get('lookback_period')} month lookback period."
            
        return f"Strategy parameters optimized for current market conditions based on technical and sentiment analysis."
        
    def _get_version(self, strategy_name):
        """Generate a version number for the strategy"""
        # In a real implementation, this would track versions in a database
        return datetime.now().strftime("%Y%m%d") 