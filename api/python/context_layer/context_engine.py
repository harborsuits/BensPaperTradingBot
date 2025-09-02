import logging
import json
import requests
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from context_layer.strategy_selector import StrategySelector
from strategy_integration import StrategyIntegration

logger = logging.getLogger(__name__)

class ContextEngine:
    """
    Contextual AI engine that analyzes global, economic, and market context
    to provide trading bias signals and strategy recommendations.
    """
    
    def __init__(self, config):
        """Initialize the context engine with configuration."""
        self.config = config
        self.contextual_ai_config = config.get("contextual_ai", {})
        self.enabled = self.contextual_ai_config.get("enabled", False)
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector(config)
        
        # Initialize strategy integration
        self.strategy_integration = StrategyIntegration(config, self)
        
        # Bias cache to avoid repeated calls for the same day
        self.bias_cache = {}
        self.cache_timestamp = None
        
        # Market indicators cache
        self.market_indicators = {}
        self.indicators_timestamp = None
        
        logger.info(f"Context Engine initialized (enabled={self.enabled})")
    
    def get_trading_bias(self):
        """
        Get the current trading bias based on multiple context sources.
        
        Returns:
            dict: Trading bias data including:
                - overall_bias: 'bullish', 'bearish', 'neutral', 'volatile', 'uncertain'
                - confidence: 0.0 to 1.0
                - sources: Dict of individual bias sources and their values
                - recommended_strategy: Strategy name that fits current context
                - avoid_sectors: List of sectors to avoid
                - focus_sectors: List of sectors to focus on
        """
        # If contextual AI is disabled, return a neutral bias with default strategy
        if not self.enabled:
            return {
                "overall_bias": "neutral",
                "confidence": 0.0,
                "sources": {},
                "recommended_strategy": self.config.get("default_strategy", "rsi_ema"),
                "avoid_sectors": [],
                "focus_sectors": []
            }
        
        # Check cache if we already calculated bias today
        today = datetime.now().strftime("%Y-%m-%d")
        if self.cache_timestamp and self.cache_timestamp.startswith(today) and today in self.bias_cache:
            logger.info("Using cached trading bias")
            return self.bias_cache[today]
        
        # Get bias from various sources
        bias_sources = {}
        
        # Update market indicators
        market_indicators = self.get_market_indicators()
        
        # Determine market condition
        market_condition = self.strategy_selector.get_market_condition(market_indicators)
        
        # GPT-based headline analysis
        if self.contextual_ai_config.get("gpt_bias_filter", False):
            gpt_bias = self._get_gpt_headline_bias()
            if gpt_bias:
                bias_sources["gpt_headlines"] = gpt_bias
        
        # FinBERT sentiment
        if self.contextual_ai_config.get("finbert_sentiment", False):
            finbert_bias = self._get_finbert_sentiment()
            if finbert_bias:
                bias_sources["finbert"] = finbert_bias
        
        # Macro economic indicators
        if self.contextual_ai_config.get("macro_bias", False):
            macro_bias = self._get_macro_bias()
            if macro_bias:
                bias_sources["macro"] = macro_bias
        
        # Historical pattern analysis
        if self.contextual_ai_config.get("pattern_scoring", False):
            pattern_bias = self._get_pattern_bias()
            if pattern_bias:
                bias_sources["pattern"] = pattern_bias
        
        # Calculate overall bias by combining all sources
        if bias_sources:
            overall_bias = self._calculate_overall_bias(bias_sources)
        else:
            # Default to neutral if no bias sources are available
            overall_bias = {
                "bias": "neutral",
                "confidence": 0.0
            }
        
        # Map market condition to bias string
        condition_to_bias = {
            "bullish": "bullish",
            "bearish": "bearish",
            "sideways": "neutral",
            "high_volatility": "volatile"
        }
        
        # Use market condition to inform bias if we don't have strong AI signals
        if overall_bias["confidence"] < 0.3 and market_condition in condition_to_bias:
            overall_bias["bias"] = condition_to_bias[market_condition]
            overall_bias["confidence"] = 0.5
        
        # Get recommended strategies for this market condition
        recommended_strategies = self.strategy_selector.get_recommended_strategies(market_condition)
        
        # Determine the primary recommended strategy
        if recommended_strategies:
            recommended_strategy = recommended_strategies[0]
        else:
            recommended_strategy = self.config.get("default_strategy", "rsi_ema")
        
        # Get strategy details from the strategy selector
        strategy_data = self.strategy_selector.get_strategy_details(recommended_strategy)
        
        # Get sector recommendations based on market condition
        avoid_sectors = []
        focus_sectors = []
        
        # Use market conditions from strategy_selector to get sector recommendations
        market_conditions = self.strategy_selector.strategies_config.get("market_conditions", {})
        if market_condition in market_conditions:
            avoid_sectors = market_conditions[market_condition].get("strategies_to_avoid", [])
            focus_sectors = market_conditions[market_condition].get("strategies_to_favor", [])
        
        # Compile full trading bias
        trading_bias = {
            "overall_bias": overall_bias["bias"],
            "confidence": overall_bias["confidence"],
            "market_condition": market_condition,
            "sources": bias_sources,
            "recommended_strategy": recommended_strategy,
            "alternative_strategies": recommended_strategies[1:] if len(recommended_strategies) > 1 else [],
            "avoid_sectors": avoid_sectors,
            "focus_sectors": focus_sectors,
            "market_indicators": {k: v for k, v in market_indicators.items() if isinstance(v, (int, float, bool, str))}
        }
        
        # Cache the result
        self.bias_cache[today] = trading_bias
        self.cache_timestamp = datetime.now().isoformat()
        
        logger.info(f"Generated trading bias: {overall_bias['bias']} (confidence: {overall_bias['confidence']:.2f})")
        logger.info(f"Market condition: {market_condition}, recommended strategy: {recommended_strategy}")
        return trading_bias
    
    def get_market_indicators(self):
        """
        Get current market indicators for strategy selection and bias determination.
        
        Returns:
            dict: Market indicators
        """
        # Check cache if we already calculated indicators recently (within last hour)
        now = datetime.now()
        if (self.indicators_timestamp and 
            (now - datetime.fromisoformat(self.indicators_timestamp)).total_seconds() < 3600):
            return self.market_indicators
        
        # Placeholder for market indicators
        # In a real implementation, this would call APIs to get real data
        indicators = {
            # Market internals
            "spy_above_20dma": True,  # Example: SPY is above its 20-day moving average
            "spy_above_50dma": True,  # Example: SPY is above its 50-day moving average
            "spy_below_20dma": False,
            "spy_below_50dma": False,
            "new_highs": 150,  # Example: Number of new 52-week highs
            "new_lows": 50,    # Example: Number of new 52-week lows
            "advance_decline_line_rising": True,
            
            # Volatility metrics
            "vix": 15.5,  # Example: VIX level
            "adr_percentage": 1.2,  # Example: Average daily range as percentage
            "put_call_ratio": 0.9,  # Example: Put/call ratio
            
            # Market conditions
            "market_trending_day": True,
            "premarket_volume": 800000,
            "clean_price_structure": True,
            "low_news_interference": True,
            "no_major_negative_news": True,
            "bullish_macro_bias": True,
            "no_earnings_announcement_pending": True,
            "no_major_catalysts_pending": True,
            "earnings_season_active": False,
            
            # Implied volatility metrics
            "iv_percentile": 35,
            "iv_rank": 45,
            "high_implied_volatility": False,
            
            # Sector metrics
            "sector_in_focus": "technology",
            "sector_rotation_to_target_sector": True,
            "bullish_sector_rotation": True
        }
        
        # Cache indicators
        self.market_indicators = indicators
        self.indicators_timestamp = now.isoformat()
        
        return indicators
    
    def _get_gpt_headline_bias(self):
        """
        Analyze recent headlines using GPT to determine market bias.
        This is a placeholder - in a real implementation, this would call an API.
        
        Returns:
            dict: GPT headline bias data
        """
        # Placeholder - in a real implementation, this would:
        # 1. Fetch recent financial headlines
        # 2. Send them to GPT API for analysis
        # 3. Get a structured response with market bias
        
        logger.info("GPT headline analysis not yet implemented - returning placeholder")
        return {
            "bias": "neutral",
            "confidence": 0.0,
            "headline_count": 0,
            "top_topics": []
        }
    
    def _get_finbert_sentiment(self):
        """
        Get market sentiment from financial news using FinBERT.
        This is a placeholder - in a real implementation, this would use the FinBERT model.
        
        Returns:
            dict: FinBERT sentiment data
        """
        # Placeholder - in a real implementation, this would:
        # 1. Fetch financial news articles
        # 2. Process them with FinBERT
        # 3. Calculate sentiment scores
        
        logger.info("FinBERT sentiment analysis not yet implemented - returning placeholder")
        return {
            "bias": "neutral",
            "confidence": 0.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 0.0
        }
    
    def _get_macro_bias(self):
        """
        Analyze macroeconomic indicators for market bias.
        This is a placeholder - in a real implementation, this would fetch and analyze macro data.
        
        Returns:
            dict: Macroeconomic bias data
        """
        # Placeholder - in a real implementation, this would:
        # 1. Fetch economic data (inflation, GDP, unemployment, etc.)
        # 2. Compare to historic ranges
        # 3. Determine bias based on trends and current levels
        
        logger.info("Macro analysis not yet implemented - returning placeholder")
        return {
            "bias": "neutral",
            "confidence": 0.0,
            "indicators": {
                "inflation": 0.0,
                "gdp_growth": 0.0,
                "unemployment": 0.0,
                "interest_rates": 0.0
            }
        }
    
    def _get_pattern_bias(self):
        """
        Analyze historical market patterns for similarity to current conditions.
        This is a placeholder - in a real implementation, this would use pattern recognition.
        
        Returns:
            dict: Pattern analysis bias data
        """
        # Placeholder - in a real implementation, this would:
        # 1. Compare current market behavior to historical patterns
        # 2. Identify similar regimes (e.g., 2008, 2020)
        # 3. Calculate probability of various outcomes
        
        logger.info("Pattern analysis not yet implemented - returning placeholder")
        return {
            "bias": "neutral",
            "confidence": 0.0,
            "similar_periods": [],
            "predicted_outcomes": {}
        }
    
    def _calculate_overall_bias(self, bias_sources):
        """
        Calculate overall bias by combining multiple sources.
        
        Args:
            bias_sources (dict): Dictionary of bias sources
        
        Returns:
            dict: Overall bias with confidence
        """
        # Map bias strings to numeric values
        bias_map = {
            "strongly_bearish": -1.0,
            "bearish": -0.5,
            "neutral": 0.0,
            "bullish": 0.5,
            "strongly_bullish": 1.0,
            "volatile": 0.0,  # Neutral but with high uncertainty
            "uncertain": 0.0  # Neutral but with high uncertainty
        }
        
        # Source weights - in a real implementation, these could be configurable
        source_weights = {
            "gpt_headlines": 0.2,
            "finbert": 0.2,
            "macro": 0.4,
            "pattern": 0.2
        }
        
        # Calculate weighted average bias
        total_weight = 0.0
        weighted_bias = 0.0
        confidence_sum = 0.0
        
        for source, data in bias_sources.items():
            if source in source_weights and "bias" in data and "confidence" in data:
                weight = source_weights[source]
                bias_value = bias_map.get(data["bias"], 0.0)
                confidence = data["confidence"]
                
                weighted_bias += bias_value * weight * confidence
                confidence_sum += confidence * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            normalized_bias = weighted_bias / total_weight
            normalized_confidence = confidence_sum / total_weight
        else:
            normalized_bias = 0.0
            normalized_confidence = 0.0
        
        # Convert numeric bias back to string
        if normalized_bias <= -0.75:
            bias_str = "strongly_bearish"
        elif normalized_bias <= -0.25:
            bias_str = "bearish"
        elif normalized_bias <= 0.25:
            bias_str = "neutral"
        elif normalized_bias <= 0.75:
            bias_str = "bullish"
        else:
            bias_str = "strongly_bullish"
        
        # Check for volatility/uncertainty
        has_volatility = any(source.get("bias") == "volatile" for source in bias_sources.values())
        has_uncertainty = any(source.get("bias") == "uncertain" for source in bias_sources.values())
        
        if has_volatility:
            bias_str = "volatile"
        elif has_uncertainty and normalized_confidence < 0.5:
            bias_str = "uncertain"
        
        return {
            "bias": bias_str,
            "confidence": normalized_confidence
        }
    
    def should_execute_trade(self, trade_data):
        """
        Determine if a trade should be executed based on context.
        
        Args:
            trade_data (dict): Trade data including symbol, action, etc.
        
        Returns:
            tuple: (should_execute, reason)
        """
        # If in manual mode, always execute trades
        if self.config.get("mode") == "manual" or not self.enabled:
            return True, "Manual mode - context checks disabled"
        
        # Get current trading bias and market indicators
        bias = self.get_trading_bias()
        market_indicators = self.get_market_indicators()
        
        # Extract key information
        market_condition = bias.get("market_condition", "sideways")
        strategy_name = trade_data.get("strategy", self.config.get("default_strategy", "rsi_ema"))
        
        # Check if the strategy is appropriate for current market conditions
        should_execute, reason = self.strategy_selector.should_execute_strategy(
            strategy_name, 
            market_condition, 
            market_indicators
        )
        
        if not should_execute:
            return False, reason
        
        # Get symbol sector (placeholder - would need a real data source)
        symbol_sector = self._get_symbol_sector(trade_data.get("ticker", ""))
        
        # Check if the sector is in the avoid list
        if symbol_sector in bias["avoid_sectors"]:
            return False, f"Sector {symbol_sector} is on the avoid list for current {bias['overall_bias']} bias"
        
        # Check if trade direction matches bias
        action = trade_data.get("action", "")
        
        if bias["overall_bias"] in ["strongly_bearish", "bearish"]:
            if action in ["buy", "buy_to_open"] and trade_data.get("option_type") != "put":
                return False, f"Bearish bias conflicts with {action} action"
        
        elif bias["overall_bias"] in ["strongly_bullish", "bullish"]:
            if action in ["sell", "sell_to_open"] and trade_data.get("option_type") != "call":
                return False, f"Bullish bias conflicts with {action} action"
        
        # If volatile/uncertain, check confidence
        if bias["overall_bias"] in ["volatile", "uncertain"] and bias["confidence"] > 0.7:
            if trade_data.get("strategy") != bias["recommended_strategy"]:
                return False, f"High volatility detected, recommended strategy is {bias['recommended_strategy']}"
        
        # Check if current strategy matches recommended strategy
        if trade_data.get("strategy") != bias["recommended_strategy"] and bias["confidence"] > 0.8:
            logger.warning(f"Trade using {trade_data.get('strategy')}, but recommended strategy is {bias['recommended_strategy']}")
        
        # Get risk parameters for the strategy
        risk_params = self.strategy_selector.get_risk_parameters(strategy_name, market_condition)
        
        # Log the risk parameters
        logger.info(f"Risk parameters for {strategy_name}: {json.dumps(risk_params)}")
        
        return True, "Trade passes contextual filters"
    
    def _get_symbol_sector(self, symbol):
        """
        Get the sector for a symbol.
        This is a placeholder - in a real implementation, this would look up actual sector data.
        
        Args:
            symbol (str): The stock symbol
        
        Returns:
            str: Sector name or None
        """
        # Placeholder sector mapping for demo purposes
        sector_map = {
            "AAPL": "technology",
            "MSFT": "technology",
            "GOOGL": "technology",
            "AMZN": "consumer_discretionary",
            "META": "technology",
            "TSLA": "consumer_discretionary",
            "JPM": "financials",
            "BAC": "financials",
            "JNJ": "healthcare",
            "PFE": "healthcare",
            "XOM": "energy",
            "CVX": "energy",
            "PG": "consumer_staples",
            "KO": "consumer_staples",
            "DIS": "communication_services",
            "NFLX": "communication_services"
        }
        
        return sector_map.get(symbol, "unknown")
    
    def get_integrated_strategy_recommendation(self, ticker=None, account_value=None):
        """
        Get integrated strategy recommendations based on current market conditions.
        
        Args:
            ticker (str, optional): Ticker symbol for targeted recommendations
            account_value (float, optional): Account value for position sizing
            
        Returns:
            dict: Integrated strategy recommendations
        """
        # Get market indicators 
        market_indicators = self.get_market_indicators()
        
        # Get market condition
        market_condition = self.strategy_selector.get_market_condition(market_indicators)
        
        # Get recommended strategies
        recommendation = self.strategy_integration.recommend_strategy_integration(ticker, account_value)
        
        # Add trading bias
        trading_bias = self.get_trading_bias()
        recommendation["trading_bias"] = trading_bias.get("overall_bias", "neutral")
        recommendation["bias_confidence"] = trading_bias.get("confidence", 0)
        
        return recommendation
        
    def evaluate_strategy_conversion(self, current_position):
        """
        Evaluate if a position should be converted between equity and options.
        
        Args:
            current_position (dict): Current position details
            
        Returns:
            dict: Strategy conversion recommendation
        """
        market_indicators = self.get_market_indicators()
        
        # Check if conversion is warranted
        should_convert, reason, conversion_type = self.strategy_integration.should_convert_strategy(
            current_position, market_indicators
        )
        
        if should_convert:
            # Generate conversion plan
            plan = self.strategy_integration.generate_conversion_plan(
                current_position, conversion_type, market_indicators
            )
            
            return {
                "should_convert": True,
                "reason": reason,
                "conversion_type": conversion_type,
                "conversion_plan": plan
            }
        
        return {
            "should_convert": False,
            "reason": reason,
            "conversion_type": None,
            "conversion_plan": None
        } 