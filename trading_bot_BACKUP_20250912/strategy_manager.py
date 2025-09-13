#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Manager Module

This module implements a comprehensive strategy management system that serves as the 
"brain" of the trading bot. It evaluates market conditions, selects appropriate 
strategies, allocates capital, and coordinates the overall trading approach.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import json
import importlib
import os
from collections import defaultdict

# Setup logging
logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """
    Classifies the current market regime based on various indicators and metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market regime classifier.
        
        Args:
            config: Configuration parameters for classification
        """
        self.config = config or {
            # Trend detection parameters
            "trend_ma_short": 20,
            "trend_ma_long": 50,
            "trend_strength_threshold": 0.05,
            
            # Volatility classification parameters
            "volatility_lookback": 20,
            "volatility_percentile_high": 75,
            "volatility_percentile_low": 25,
            
            # Correlation analysis
            "correlation_lookback": 30,
            
            # Liquidity metrics
            "liquidity_threshold_low": 0.5,  # Volume ratio to moving average
            
            # Seasonality factors
            "enable_seasonality": True
        }
    
    def classify_trend(self, data: pd.DataFrame) -> str:
        """
        Determine the trend regime (bullish, bearish, sideways).
        
        Args:
            data: Price data with OHLCV columns
            
        Returns:
            String indicating trend regime
        """
        if len(data) < self.config["trend_ma_long"]:
            return "undefined"
            
        # Calculate moving averages
        ma_short = data['close'].rolling(window=self.config["trend_ma_short"]).mean()
        ma_long = data['close'].rolling(window=self.config["trend_ma_long"]).mean()
        
        # Current values
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate trend strength
        ma_diff = (current_ma_short / current_ma_long) - 1
        price_vs_ma = (current_price / current_ma_long) - 1
        
        # Classify trend
        threshold = self.config["trend_strength_threshold"]
        
        if ma_diff > threshold and price_vs_ma > threshold:
            return "strong_bullish"
        elif ma_diff > 0 and price_vs_ma > 0:
            return "bullish"
        elif ma_diff < -threshold and price_vs_ma < -threshold:
            return "strong_bearish"
        elif ma_diff < 0 and price_vs_ma < 0:
            return "bearish"
        else:
            # Check if price is range-bound
            recent_data = data.iloc[-self.config["trend_ma_short"]:]
            high_low_range = (recent_data['high'].max() - recent_data['low'].min()) / current_price
            
            if high_low_range < 0.05:  # Tight range
                return "tight_range"
            else:
                return "sideways"
    
    def classify_volatility(self, data: pd.DataFrame) -> str:
        """
        Determine the volatility regime (high, normal, low).
        
        Args:
            data: Price data with OHLCV columns
            
        Returns:
            String indicating volatility regime
        """
        if len(data) < self.config["volatility_lookback"] * 2:
            return "undefined"
            
        # Calculate historical volatility
        returns = np.log(data['close'] / data['close'].shift(1))
        current_vol = returns.rolling(window=self.config["volatility_lookback"]).std() * np.sqrt(252)
        
        # Get historical percentiles
        vol_history = current_vol[~current_vol.isna()]
        high_percentile = np.percentile(vol_history, self.config["volatility_percentile_high"])
        low_percentile = np.percentile(vol_history, self.config["volatility_percentile_low"])
        
        # Current volatility
        current_vol_value = current_vol.iloc[-1]
        
        # Classify volatility
        if current_vol_value > high_percentile:
            return "high"
        elif current_vol_value < low_percentile:
            return "low"
        else:
            return "normal"
    
    def analyze_correlations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Analyze correlations between different assets.
        
        Args:
            market_data: Dictionary of dataframes with price data for various assets
            
        Returns:
            Dictionary with correlation metrics
        """
        # Extract closing prices into a single dataframe
        close_prices = pd.DataFrame()
        
        for symbol, data in market_data.items():
            if len(data) >= self.config["correlation_lookback"]:
                close_prices[symbol] = data['close']
        
        if len(close_prices.columns) < 2:
            return {"avg_correlation": 0}
            
        # Calculate correlation matrix
        correlation_matrix = close_prices.pct_change().iloc[-self.config["correlation_lookback"]:].corr()
        
        # Average correlation (excluding self-correlations)
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                correlations.append(correlation_matrix.iloc[i, j])
        
        avg_correlation = np.mean(correlations)
        
        return {
            "avg_correlation": avg_correlation,
            "market_correlation_regime": "high" if avg_correlation > 0.7 else "moderate" if avg_correlation > 0.3 else "low"
        }
    
    def assess_liquidity(self, data: pd.DataFrame) -> str:
        """
        Assess market liquidity based on volume and spread metrics.
        
        Args:
            data: Price data with OHLCV columns
            
        Returns:
            String indicating liquidity conditions
        """
        if 'volume' not in data.columns or len(data) < 20:
            return "undefined"
            
        # Volume relative to moving average
        avg_volume = data['volume'].rolling(window=20).mean()
        relative_volume = data['volume'].iloc[-1] / avg_volume.iloc[-1]
        
        if relative_volume < self.config["liquidity_threshold_low"]:
            return "low"
        elif relative_volume > 2.0:
            return "high"
        else:
            return "normal"
    
    def detect_seasonality(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, Any]:
        """
        Detect seasonality factors like time-of-day, day-of-week effects.
        
        Args:
            data: Historical price data
            current_time: Current datetime
            
        Returns:
            Dictionary with seasonality information
        """
        if not self.config["enable_seasonality"]:
            return {}
            
        # Extract day of week
        day_of_week = current_time.weekday()
        
        # Extract hour of day
        hour_of_day = current_time.hour
        
        # Determine market session
        if 9 <= hour_of_day < 11:
            market_session = "opening"
        elif 14 <= hour_of_day < 16:
            market_session = "closing"
        elif 11 <= hour_of_day < 14:
            market_session = "midday"
        else:
            market_session = "after_hours"
        
        return {
            "day_of_week": day_of_week,
            "hour_of_day": hour_of_day,
            "market_session": market_session
        }
    
    def classify_regime(self, market_data: Dict[str, pd.DataFrame], current_time: datetime = None) -> Dict[str, Any]:
        """
        Perform comprehensive market regime classification.
        
        Args:
            market_data: Dictionary of dataframes with price data for various assets
            current_time: Current datetime (default: now)
            
        Returns:
            Dictionary with regime classification
        """
        if current_time is None:
            current_time = datetime.now()
            
        # Select a reference index (e.g., SPY) for overall market assessment
        reference_symbol = next((s for s in ["SPY", "QQQ", "^GSPC"] if s in market_data), list(market_data.keys())[0])
        reference_data = market_data[reference_symbol]
        
        # Classify trend
        trend_regime = self.classify_trend(reference_data)
        
        # Classify volatility
        volatility_regime = self.classify_volatility(reference_data)
        
        # Analyze correlations
        correlation_info = self.analyze_correlations(market_data)
        
        # Assess liquidity
        liquidity_regime = self.assess_liquidity(reference_data)
        
        # Detect seasonality
        seasonality_info = self.detect_seasonality(reference_data, current_time)
        
        # Comprehensive regime
        if trend_regime in ["strong_bullish", "bullish"] and volatility_regime != "high":
            overall_regime = "bull_trend"
        elif trend_regime in ["strong_bearish", "bearish"] and volatility_regime != "high":
            overall_regime = "bear_trend"
        elif trend_regime in ["sideways", "tight_range"] and volatility_regime == "low":
            overall_regime = "range_bound"
        elif volatility_regime == "high":
            overall_regime = "high_volatility"
        else:
            overall_regime = "mixed"
        
        return {
            "overall_regime": overall_regime,
            "trend_regime": trend_regime,
            "volatility_regime": volatility_regime,
            "liquidity_regime": liquidity_regime,
            "correlation_info": correlation_info,
            "seasonality_info": seasonality_info,
            "timestamp": current_time
        }

class ExternalDataIntegrator:
    """
    Integrates external data sources like news and economic data into the strategy system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize external data integrator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {
            "enable_news_sentiment": True,
            "enable_economic_calendar": True,
            "enable_social_sentiment": True,
            "sentiment_lookback_days": 2,
            "high_impact_event_window_hours": 24
        }
        
        self.economic_calendar = pd.DataFrame()  # Will hold economic events
        self.news_sentiment = defaultdict(list)  # Symbol -> list of sentiment scores
        self.social_sentiment = defaultdict(list)  # Symbol -> list of social sentiment data
    
    def update_economic_calendar(self, calendar_data: pd.DataFrame):
        """
        Update economic calendar with upcoming events.
        
        Args:
            calendar_data: DataFrame with economic events
        """
        self.economic_calendar = calendar_data
    
    def update_news_sentiment(self, symbol: str, sentiment_data: Dict[str, Any]):
        """
        Update news sentiment for a specific symbol.
        
        Args:
            symbol: The ticker symbol
            sentiment_data: Dictionary with sentiment information
        """
        self.news_sentiment[symbol].append(sentiment_data)
        
        # Trim old data
        cutoff_time = datetime.now() - timedelta(days=self.config["sentiment_lookback_days"])
        self.news_sentiment[symbol] = [
            item for item in self.news_sentiment[symbol] 
            if item.get("timestamp", datetime.now()) > cutoff_time
        ]
    
    def update_social_sentiment(self, symbol: str, sentiment_data: Dict[str, Any]):
        """
        Update social media sentiment for a specific symbol.
        
        Args:
            symbol: The ticker symbol
            sentiment_data: Dictionary with social sentiment information
        """
        self.social_sentiment[symbol].append(sentiment_data)
        
        # Trim old data
        cutoff_time = datetime.now() - timedelta(days=self.config["sentiment_lookback_days"])
        self.social_sentiment[symbol] = [
            item for item in self.social_sentiment[symbol] 
            if item.get("timestamp", datetime.now()) > cutoff_time
        ]
    
    def get_upcoming_economic_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Get upcoming economic events within the specified window.
        
        Args:
            hours_ahead: Hours to look ahead
            
        Returns:
            List of upcoming economic events
        """
        if self.economic_calendar.empty:
            return []
            
        now = datetime.now()
        future_cutoff = now + timedelta(hours=hours_ahead)
        
        upcoming_events = self.economic_calendar[
            (self.economic_calendar['datetime'] > now) & 
            (self.economic_calendar['datetime'] <= future_cutoff)
        ]
        
        return upcoming_events.to_dict('records')
    
    def get_high_impact_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Get high-impact economic events within the specified window.
        
        Args:
            hours_ahead: Hours to look ahead
            
        Returns:
            List of high-impact events
        """
        upcoming_events = self.get_upcoming_economic_events(hours_ahead)
        
        # Filter for high-impact events
        high_impact_events = [
            event for event in upcoming_events
            if event.get('impact', '').lower() in ['high', 'h']
        ]
        
        return high_impact_events
    
    def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined sentiment data for a specific symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Dictionary with sentiment information
        """
        # News sentiment
        news_items = self.news_sentiment.get(symbol, [])
        avg_news_sentiment = np.mean([item.get('sentiment_score', 0) for item in news_items]) if news_items else 0
        
        # Social sentiment
        social_items = self.social_sentiment.get(symbol, [])
        avg_social_sentiment = np.mean([item.get('sentiment_score', 0) for item in social_items]) if social_items else 0
        
        # Combine with custom weights
        combined_sentiment = 0.7 * avg_news_sentiment + 0.3 * avg_social_sentiment
        
        # Classify sentiment
        if combined_sentiment > 0.3:
            sentiment_classification = "bullish"
        elif combined_sentiment < -0.3:
            sentiment_classification = "bearish"
        else:
            sentiment_classification = "neutral"
            
        return {
            "combined_sentiment_score": combined_sentiment,
            "news_sentiment_score": avg_news_sentiment,
            "social_sentiment_score": avg_social_sentiment,
            "sentiment_classification": sentiment_classification,
            "news_count": len(news_items),
            "social_mentions_count": len(social_items)
        }
    
    def should_adjust_risk(self, symbol: str = None) -> Tuple[bool, str, float]:
        """
        Determine if risk should be adjusted based on external factors.
        
        Args:
            symbol: Optional symbol to check for specific adjustments
            
        Returns:
            Tuple of (should_adjust, reason, adjustment_factor)
        """
        # Check for high-impact economic events
        high_impact_events = self.get_high_impact_events(
            self.config["high_impact_event_window_hours"]
        )
        
        if high_impact_events:
            return (True, f"High-impact economic event upcoming: {high_impact_events[0].get('event')}", 0.5)
        
        # Check sentiment if symbol is provided
        if symbol:
            sentiment = self.get_symbol_sentiment(symbol)
            
            if sentiment["sentiment_classification"] == "bearish" and sentiment["news_count"] > 3:
                return (True, "Negative news sentiment", 0.7)
            elif sentiment["sentiment_classification"] == "bullish" and sentiment["news_count"] > 3:
                return (True, "Positive news sentiment", 1.2)
        
        return (False, "", 1.0)

class StrategyLibrary:
    """
    Manages the library of available trading strategies with metadata.
    """
    
    def __init__(self, strategies_path: str = "trading_bot/strategies"):
        """
        Initialize strategy library.
        
        Args:
            strategies_path: Path to strategies directory
        """
        self.strategies_path = strategies_path
        self.strategies = {}  # name -> metadata
        self._load_strategies()
    
    def _load_strategies(self):
        """Load available strategies from the strategies directory."""
        # Define strategy categories and their directories
        strategy_categories = {
            "timeframe": ["swing_trading", "day_trading", "position_trading", "scalping"],
            "options_income": ["covered_call", "cash_secured_put", "married_put", "collar"],
            "options_spreads": ["butterfly_spread", "iron_condor", "bull_call_spread", "bear_put_spread"]
        }
        
        # Strategy performance characteristics based on market regimes
        # This would typically be learned from backtesting or updated dynamically
        strategy_characteristics = {
            "swing_trading": {
                "optimal_regimes": ["bull_trend", "bear_trend"],
                "avoid_regimes": ["high_volatility"],
                "time_horizon": "medium",  # days to weeks
                "capital_efficiency": 0.7,
                "risk_profile": "medium"
            },
            "day_trading": {
                "optimal_regimes": ["high_volatility", "mixed"],
                "avoid_regimes": [],
                "time_horizon": "short",  # hours to day
                "capital_efficiency": 0.9,
                "risk_profile": "high"
            },
            "position_trading": {
                "optimal_regimes": ["bull_trend", "bear_trend"],
                "avoid_regimes": ["range_bound"],
                "time_horizon": "long",  # weeks to months
                "capital_efficiency": 0.6,
                "risk_profile": "medium"
            },
            "butterfly_spread": {
                "optimal_regimes": ["range_bound"],
                "avoid_regimes": ["high_volatility", "bull_trend", "bear_trend"],
                "time_horizon": "medium",  # weeks
                "capital_efficiency": 0.8,
                "risk_profile": "low"
            },
            "iron_condor": {
                "optimal_regimes": ["range_bound"],
                "avoid_regimes": ["high_volatility"],
                "time_horizon": "medium",  # weeks
                "capital_efficiency": 0.8,
                "risk_profile": "low"
            }
        }
        
        # Find strategy files
        for category, strategy_list in strategy_categories.items():
            for strategy_name in strategy_list:
                # Build path to potential strategy file
                strategy_file = os.path.join(self.strategies_path, f"{strategy_name}.py")
                category_strategy_file = os.path.join(self.strategies_path, category, f"{strategy_name}.py")
                
                # Check if strategy exists in either location
                if os.path.exists(strategy_file) or os.path.exists(category_strategy_file):
                    # Create metadata entry
                    metadata = {
                        "name": strategy_name,
                        "category": category,
                        "status": "active",
                        "characteristics": strategy_characteristics.get(strategy_name, {
                            "optimal_regimes": [],
                            "avoid_regimes": [],
                            "time_horizon": "medium",
                            "capital_efficiency": 0.7,
                            "risk_profile": "medium"
                        }),
                        "path": strategy_file if os.path.exists(strategy_file) else category_strategy_file
                    }
                    
                    self.strategies[strategy_name] = metadata
                    logger.info(f"Found strategy: {strategy_name} in category {category}")
        
        logger.info(f"Loaded {len(self.strategies)} strategies from {self.strategies_path}")
    
    def get_strategy_class(self, strategy_name: str):
        """
        Dynamically import and return the strategy class.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            The strategy class
        """
        metadata = self.strategies.get(strategy_name)
        if not metadata:
            raise ValueError(f"Strategy {strategy_name} not found in library")
            
        # Determine module path
        if "category" in metadata and os.path.exists(metadata["path"]):
            if "category" in metadata["path"]:
                module_path = f"trading_bot.strategies.{metadata['category']}.{strategy_name}"
            else:
                module_path = f"trading_bot.strategies.{strategy_name}"
                
            try:
                module = importlib.import_module(module_path)
                
                # Find the strategy class in the module
                # We assume the class name follows CamelCase convention
                class_name = "".join(word.capitalize() for word in strategy_name.split("_")) + "Strategy"
                
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                else:
                    # Try to find any class that ends with 'Strategy'
                    for name in dir(module):
                        if name.endswith("Strategy") and name != "StrategyTemplate" and name != "StrategyOptimizable":
                            return getattr(module, name)
                            
                raise ValueError(f"Strategy class not found in module {module_path}")
                
            except ImportError as e:
                logger.error(f"Failed to import strategy {strategy_name}: {e}")
                raise
        else:
            raise ValueError(f"Strategy file not found for {strategy_name}")
    
    def get_strategy_metadata(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy metadata
        """
        return self.strategies.get(strategy_name, {})
    
    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available strategies.
        
        Returns:
            Dictionary of strategy name -> metadata
        """
        return self.strategies
    
    def get_suitable_strategies(self, market_regime: str) -> List[Dict[str, Any]]:
        """
        Get strategies suitable for the current market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            List of suitable strategy metadata
        """
        suitable_strategies = []
        
        for name, metadata in self.strategies.items():
            characteristics = metadata.get("characteristics", {})
            optimal_regimes = characteristics.get("optimal_regimes", [])
            avoid_regimes = characteristics.get("avoid_regimes", [])
            
            if market_regime in optimal_regimes and market_regime not in avoid_regimes:
                suitable_strategies.append(metadata)
        
        return suitable_strategies
    
    def update_strategy_performance(self, strategy_name: str, performance_metrics: Dict[str, Any]):
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_metrics: Dictionary of performance metrics
        """
        if strategy_name in self.strategies:
            if "performance" not in self.strategies[strategy_name]:
                self.strategies[strategy_name]["performance"] = {}
                
            # Update with new metrics
            self.strategies[strategy_name]["performance"].update(performance_metrics)
            
            # Calculate score based on performance
            win_rate = performance_metrics.get("win_rate", 0)
            profit_factor = performance_metrics.get("profit_factor", 1)
            sharpe = performance_metrics.get("sharpe_ratio", 0)
            
            # Simple scoring formula (customize as needed)
            score = (0.4 * win_rate) + (0.4 * profit_factor) + (0.2 * sharpe)
            self.strategies[strategy_name]["performance"]["score"] = score

class StrategyManager:
    """
    Central intelligence hub for managing and coordinating trading strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy manager.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {
            "max_strategies_active": 5,
            "default_capital_allocation": 0.2,  # 20% per strategy by default
            "risk_adjustment_enabled": True,
            "dynamic_optimization_enabled": True,
            "min_strategy_score": 0.3,  # Minimum score to consider a strategy
            "correlation_threshold": 0.7  # Maximum allowed correlation between strategies
        }
        
        # Initialize components
        self.regime_classifier = MarketRegimeClassifier()
        self.external_data = ExternalDataIntegrator()
        self.strategy_library = StrategyLibrary()
        
        # Strategy state tracking
        self.active_strategies = {}  # name -> strategy_instance
        self.strategy_allocations = {}  # name -> allocation_percentage
        self.regime_history = []  # List of regime classifications
        self.last_optimization = datetime.now()
    
    def update_market_data(self, market_data: Dict[str, pd.DataFrame], current_time: Optional[datetime] = None):
        """
        Update market data and perform regime classification.
        
        Args:
            market_data: Dictionary of dataframes with price data for various assets
            current_time: Current datetime (default: now)
        """
        if current_time is None:
            current_time = datetime.now()
            
        # Classify market regime
        regime = self.regime_classifier.classify_regime(market_data, current_time)
        self.regime_history.append(regime)
        
        # Trim history if needed
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
            
        logger.info(f"Market regime updated: {regime['overall_regime']}")
    
    def evaluate_strategies(self) -> Dict[str, float]:
        """
        Evaluate and score available strategies based on current conditions.
        
        Returns:
            Dictionary mapping strategy names to scores
        """
        if not self.regime_history:
            logger.warning("No regime history available for strategy evaluation")
            return {}
            
        # Get current regime
        current_regime = self.regime_history[-1]["overall_regime"]
        
        # Get suitable strategies for this regime
        suitable_strategies = self.strategy_library.get_suitable_strategies(current_regime)
        
        # Score strategies
        strategy_scores = {}
        
        for metadata in suitable_strategies:
            strategy_name = metadata["name"]
            
            # Base score on regime suitability
            characteristics = metadata.get("characteristics", {})
            optimal_regimes = characteristics.get("optimal_regimes", [])
            avoid_regimes = characteristics.get("avoid_regimes", [])
            
            # Initial score based on regime match
            if current_regime in optimal_regimes:
                regime_score = 1.0
            elif current_regime in avoid_regimes:
                regime_score = 0.0
            else:
                regime_score = 0.5
                
            # Factor in historical performance if available
            performance = metadata.get("performance", {})
            performance_score = performance.get("score", 0.5)
            
            # Combine scores (customize weights as needed)
            final_score = (0.4 * regime_score) + (0.6 * performance_score)
            
            strategy_scores[strategy_name] = final_score
        
        return strategy_scores
    
    def select_strategies(self, strategy_scores: Dict[str, float]) -> List[str]:
        """
        Select strategies to use based on scores and constraints.
        
        Args:
            strategy_scores: Dictionary mapping strategy names to scores
            
        Returns:
            List of selected strategy names
        """
        # Filter strategies with minimum score
        qualified_strategies = {
            name: score for name, score in strategy_scores.items()
            if score >= self.config["min_strategy_score"]
        }
        
        # Sort by score (descending)
        sorted_strategies = sorted(
            qualified_strategies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Limit to maximum number of active strategies
        selected_strategies = [name for name, _ in sorted_strategies[:self.config["max_strategies_active"]]]
        
        # TODO: Add correlation/diversification check to ensure strategies aren't too similar
        
        return selected_strategies
    
    def allocate_capital(self, selected_strategies: List[str], strategy_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate capital among selected strategies.
        
        Args:
            selected_strategies: List of selected strategy names
            strategy_scores: Dictionary mapping strategy names to scores
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        if not selected_strategies:
            return {}
            
        # Equal allocation as a baseline
        equal_allocation = 1.0 / len(selected_strategies)
        
        # Score-weighted allocation
        total_score = sum(strategy_scores[name] for name in selected_strategies)
        
        if total_score == 0:
            # Fall back to equal allocation
            allocations = {name: equal_allocation for name in selected_strategies}
        else:
            # Allocate proportionally to score
            allocations = {
                name: strategy_scores[name] / total_score
                for name in selected_strategies
            }
            
        # Adjust allocations based on strategy characteristics
        for name in selected_strategies:
            metadata = self.strategy_library.get_strategy_metadata(name)
            characteristics = metadata.get("characteristics", {})
            
            # Adjust for capital efficiency
            capital_efficiency = characteristics.get("capital_efficiency", 0.7)
            allocations[name] *= capital_efficiency
            
            # Adjust for risk profile
            risk_profile = characteristics.get("risk_profile", "medium")
            risk_multiplier = 1.0
            if risk_profile == "high":
                risk_multiplier = 0.8
            elif risk_profile == "low":
                risk_multiplier = 1.2
                
            allocations[name] *= risk_multiplier
                
        # Normalize to ensure sum is 1.0
        total_allocation = sum(allocations.values())
        normalized_allocations = {
            name: alloc / total_allocation
            for name, alloc in allocations.items()
        }
        
        return normalized_allocations
    
    def adjust_for_external_factors(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust allocations based on external factors like news and economic events.
        
        Args:
            allocations: Dictionary mapping strategy names to allocation percentages
            
        Returns:
            Adjusted allocations
        """
        if not self.config["risk_adjustment_enabled"]:
            return allocations
            
        # Check for global risk adjustments
        should_adjust, reason, adjustment_factor = self.external_data.should_adjust_risk()
        
        if should_adjust:
            logger.info(f"Adjusting risk due to external factors: {reason} (factor: {adjustment_factor})")
            
            # Apply global adjustment
            adjusted_allocations = {
                name: alloc * adjustment_factor
                for name, alloc in allocations.items()
            }
            
            # Reallocate to cash if reducing risk
            if adjustment_factor < 1.0:
                # Calculate how much was reduced
                original_total = sum(allocations.values())
                adjusted_total = sum(adjusted_allocations.values())
                cash_allocation = original_total - adjusted_total
                
                # Add cash allocation
                adjusted_allocations["cash"] = cash_allocation
                
            return adjusted_allocations
            
        # Check for symbol-specific adjustments
        # This would require mapping strategies to symbols they trade
        # Simplified implementation
        
        return allocations
    
    def instantiate_strategy(self, strategy_name: str) -> Any:
        """
        Instantiate a strategy class with appropriate parameters.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy instance
        """
        # Get strategy class
        strategy_class = self.strategy_library.get_strategy_class(strategy_name)
        
        # Get strategy metadata
        metadata = self.strategy_library.get_strategy_metadata(strategy_name)
        
        # Instantiate strategy (with appropriate parameters)
        # This would typically load optimized parameters for the current market regime
        strategy_instance = strategy_class(
            name=strategy_name,
            parameters=None,  # Should load optimized parameters
            metadata=metadata
        )
        
        return strategy_instance
    
    def update_active_strategies(self, allocations: Dict[str, float]):
        """
        Update the active strategies based on allocations.
        
        Args:
            allocations: Dictionary mapping strategy names to allocation percentages
        """
        # Track which strategies need to be activated or deactivated
        strategies_to_activate = set(allocations.keys()) - set(self.active_strategies.keys())
        strategies_to_deactivate = set(self.active_strategies.keys()) - set(allocations.keys())
        strategies_to_maintain = set(allocations.keys()) & set(self.active_strategies.keys())
        
        # Activate new strategies
        for strategy_name in strategies_to_activate:
            if strategy_name == "cash":
                continue  # Skip cash, it's just a placeholder
                
            try:
                strategy_instance = self.instantiate_strategy(strategy_name)
                self.active_strategies[strategy_name] = strategy_instance
                logger.info(f"Activated strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Failed to activate strategy {strategy_name}: {e}")
        
        # Deactivate strategies
        for strategy_name in strategies_to_deactivate:
            # Perform any cleanup needed
            del self.active_strategies[strategy_name]
            logger.info(f"Deactivated strategy: {strategy_name}")
        
        # Update allocations
        self.strategy_allocations = allocations
        
        # Log active strategies
        active_str = ", ".join(f"{name}: {alloc:.1%}" for name, alloc in self.strategy_allocations.items())
        logger.info(f"Active strategies: {active_str}")
    
    def should_optimize_strategies(self) -> bool:
        """
        Determine if strategies should be optimized.
        
        Returns:
            Boolean indicating if optimization should occur
        """
        if not self.config["dynamic_optimization_enabled"]:
            return False
            
        # Check if enough time has passed since last optimization
        time_since_last = datetime.now() - self.last_optimization
        if time_since_last.days < 7:  # Weekly optimization by default
            return False
            
        # Check if market regime has changed
        if len(self.regime_history) >= 2:
            current_regime = self.regime_history[-1]["overall_regime"]
            previous_regime = self.regime_history[-2]["overall_regime"]
            
            if current_regime != previous_regime:
                return True
                
        return False
    
    def optimize_strategies(self):
        """Optimize active strategies for current market conditions."""
        if not self.should_optimize_strategies():
            return
            
        logger.info("Optimizing strategies for current market conditions")
        
        # Get current regime
        current_regime = self.regime_history[-1]["overall_regime"]
        
        # Update last optimization timestamp
        self.last_optimization = datetime.now()
        
        # For each active strategy, trigger optimization
        for strategy_name, strategy_instance in self.active_strategies.items():
            if hasattr(strategy_instance, "optimize"):
                try:
                    logger.info(f"Optimizing strategy: {strategy_name}")
                    strategy_instance.optimize(market_regime=current_regime)
                except Exception as e:
                    logger.error(f"Failed to optimize strategy {strategy_name}: {e}")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signals from active strategies.
        
        Args:
            market_data: Dictionary of dataframes with price data for various assets
            
        Returns:
            Dictionary of signals from all active strategies
        """
        all_signals = {}
        
        for strategy_name, strategy_instance in self.active_strategies.items():
            try:
                # Generate signals from this strategy
                signals = strategy_instance.generate_signals(market_data)
                
                # Store signals with metadata
                for symbol, signal in signals.items():
                    # Add strategy information to signal
                    signal.metadata["strategy"] = strategy_name
                    signal.metadata["allocation"] = self.strategy_allocations.get(strategy_name, 0)
                    
                    # Store signal
                    if symbol not in all_signals:
                        all_signals[symbol] = []
                        
                    all_signals[symbol].append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_name}: {e}")
        
        return all_signals
    
    def adjust_position_sizes(self, signals: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Adjust position sizes based on strategy allocations and risk factors.
        
        Args:
            signals: Dictionary mapping symbols to lists of signals
            
        Returns:
            Adjusted signals
        """
        # Apply allocation percentages to position sizes
        for symbol, symbol_signals in signals.items():
            for signal in symbol_signals:
                strategy_name = signal.metadata.get("strategy")
                allocation = self.strategy_allocations.get(strategy_name, 0)
                
                # Adjust position size based on allocation
                if hasattr(signal, "position_size") and signal.position_size is not None:
                    signal.position_size *= allocation
                
                # Adjust for symbol-specific external factors
                should_adjust, reason, adjustment_factor = self.external_data.should_adjust_risk(symbol)
                
                if should_adjust and hasattr(signal, "position_size") and signal.position_size is not None:
                    signal.position_size *= adjustment_factor
                    signal.metadata["risk_adjustment"] = {
                        "reason": reason,
                        "factor": adjustment_factor
                    }
        
        return signals
    
    def run_strategy_cycle(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Any]]:
        """
        Run a complete strategy management cycle:
        1. Update market data and classify regime
        2. Evaluate and select strategies
        3. Allocate capital
        4. Generate and adjust signals
        
        Args:
            market_data: Dictionary of dataframes with price data for various assets
            
        Returns:
            Dictionary of adjusted signals
        """
        # Update market data and regime
        self.update_market_data(market_data)
        
        # Evaluate strategies
        strategy_scores = self.evaluate_strategies()
        
        # Select strategies
        selected_strategies = self.select_strategies(strategy_scores)
        
        # Allocate capital
        allocations = self.allocate_capital(selected_strategies, strategy_scores)
        
        # Adjust for external factors
        adjusted_allocations = self.adjust_for_external_factors(allocations)
        
        # Update active strategies
        self.update_active_strategies(adjusted_allocations)
        
        # Optimize strategies if needed
        self.optimize_strategies()
        
        # Generate signals
        signals = self.generate_signals(market_data)
        
        # Adjust position sizes
        adjusted_signals = self.adjust_position_sizes(signals)
        
        return adjusted_signals
    
    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report for all strategies.
        
        Returns:
            Dictionary with performance information
        """
        report = {
            "active_strategies": [],
            "overall_allocation": self.strategy_allocations,
            "current_regime": self.regime_history[-1] if self.regime_history else None
        }
        
        # Add active strategy details
        for strategy_name, strategy_instance in self.active_strategies.items():
            metadata = self.strategy_library.get_strategy_metadata(strategy_name)
            
            strategy_report = {
                "name": strategy_name,
                "allocation": self.strategy_allocations.get(strategy_name, 0),
                "category": metadata.get("category"),
                "characteristics": metadata.get("characteristics", {}),
                "performance": metadata.get("performance", {})
            }
            
            report["active_strategies"].append(strategy_report)
        
        return report

# Example usage:
if __name__ == "__main__":
    # Initialize the strategy manager
    manager = StrategyManager()
    
    # Simulate market data (in a real system, this would be live data)
    import yfinance as yf
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOG"]
    
    market_data = {}
    for symbol in symbols:
        try:
            data = yf.download(symbol, period="60d", interval="1d")
            market_data[symbol] = data
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
    
    # Run a strategy cycle
    signals = manager.run_strategy_cycle(market_data)
    
    # Print signals
    for symbol, symbol_signals in signals.items():
        for signal in symbol_signals:
            print(f"Signal for {symbol}: {signal.signal_type} from {signal.metadata.get('strategy')}")
            
    # Print performance report
    import json
    print(json.dumps(manager.get_strategy_performance_report(), indent=2)) 