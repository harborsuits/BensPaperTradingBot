"""
SymbolRanker - Intelligent symbol scoring and ranking system
This module provides a robust framework for scoring and ranking symbols
based on multiple factors including technicals, fundamentals, news sentiment,
and their suitability for specific trading strategies.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add parent directory to path to import from trading_bot.market_context
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trading_bot.market_context.market_context import get_market_context

class SymbolRanker:
    """
    Scores and ranks symbols based on multiple weighted factors.
    Integrates with MarketContext to use comprehensive market data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the symbol ranker with configuration.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        self.logger = logging.getLogger("SymbolRanker")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Default scoring weights
        self.weights = self._config.get("weights", {
            "technical": 0.40,
            "sentiment": 0.20,
            "fundamental": 0.25,
            "regime_alignment": 0.15
        })
        
        # Access the singleton market context
        self.market_context = get_market_context()
        
        self.logger.info("SymbolRanker initialized")
    
    def rank_symbols_for_strategy(self, strategy_id, symbols=None, limit=10):
        """
        Rank symbols based on their suitability for a specific strategy.
        
        Args:
            strategy_id: ID of the strategy to rank symbols for
            symbols: List of symbols to rank or None to use all symbols in the market context
            limit: Maximum number of symbols to return
        
        Returns:
            List of ranked symbols with scores and reasoning
        """
        self.logger.info(f"Ranking symbols for strategy '{strategy_id}'")
        
        # Get current market context
        context = self.market_context.get_market_context()
        
        # Get all symbols from context if not specified
        if symbols is None:
            symbols = list(context.get("symbols", {}).keys())
        
        # Get current market regime
        market_regime = context.get("market", {}).get("regime", "unknown")
        
        # Results will be stored here
        ranked_symbols = []
        
        for symbol in symbols:
            # Skip if symbol data is not in context
            if symbol not in context.get("symbols", {}):
                continue
            
            symbol_data = context["symbols"][symbol]
            
            # Calculate various scores with reasoning
            technical_score, technical_reasoning = self._calculate_technical_score(symbol_data, strategy_id)
            sentiment_score, sentiment_reasoning = self._calculate_sentiment_score(symbol, context)
            fundamental_score, fundamental_reasoning = self._calculate_fundamental_score(symbol_data)
            regime_score, regime_reasoning = self._calculate_regime_alignment(symbol_data, market_regime, strategy_id)
            
            # Calculate weighted total score
            total_score = (
                technical_score * self.weights["technical"] +
                sentiment_score * self.weights["sentiment"] +
                fundamental_score * self.weights["fundamental"] +
                regime_score * self.weights["regime_alignment"]
            )
            
            # Calculate confidence based on the dispersion of scores
            scores = [technical_score, sentiment_score, fundamental_score, regime_score]
            confidence = 1.0 - (max(scores) - min(scores)) / 2
            
            # Compile all reasoning into a single list
            # Start with the most significant factors based on weights and scores
            all_reasoning = []
            reasoning_components = [
                (technical_reasoning, self.weights["technical"], technical_score),
                (sentiment_reasoning, self.weights["sentiment"], sentiment_score),
                (regime_reasoning, self.weights["regime_alignment"], regime_score),
                (fundamental_reasoning, self.weights["fundamental"], fundamental_score)
            ]
            
            # Sort by contribution to final score (weight * score)
            reasoning_components.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            # Add reasoning in order of contribution
            for reasoning_list, _, _ in reasoning_components:
                all_reasoning.extend(reasoning_list)
            
            # Store the result with detailed reasoning
            ranked_symbols.append({
                "symbol": symbol,
                "strategy": strategy_id,
                "score": total_score,
                "confidence": confidence,
                "reasoning": all_reasoning,
                "details": {
                    "technical": {
                        "score": technical_score,
                        "reasoning": technical_reasoning
                    },
                    "sentiment": {
                        "score": sentiment_score,
                        "reasoning": sentiment_reasoning
                    },
                    "fundamental": {
                        "score": fundamental_score,
                        "reasoning": fundamental_reasoning
                    },
                    "regime_alignment": {
                        "score": regime_score,
                        "reasoning": regime_reasoning
                    }
                },
                "market_regime": market_regime
            })
        
        # Sort by score (descending)
        ranked_symbols = sorted(ranked_symbols, key=lambda x: x["score"], reverse=True)
        
        # Limit results
        if limit > 0:
            ranked_symbols = ranked_symbols[:limit]
        
        return ranked_symbols
    
    def rank_symbols_for_all_strategies(self, symbols=None, limit_per_strategy=5):
        """
        Rank symbols for all available strategies.
        
        Args:
            symbols: List of symbols to rank or None to use all symbols in market context
            limit_per_strategy: Maximum number of symbols to return per strategy
        
        Returns:
            Dictionary mapping strategy IDs to ranked symbol lists
        """
        # Get current market context
        context = self.market_context.get_market_context()
        
        # Get all available strategies
        strategies = [
            strategy["id"] for strategy in 
            context.get("strategies", {}).get("ranked", [])
        ]
        
        # Results will be stored here
        all_rankings = {}
        
        for strategy_id in strategies:
            ranked_symbols = self.rank_symbols_for_strategy(
                strategy_id, symbols, limit_per_strategy
            )
            all_rankings[strategy_id] = ranked_symbols
        
        return all_rankings
    
    def find_best_symbol_strategy_pairs(self, limit=10):
        """
        Find the best symbol-strategy pairs across all strategies.
        
        Args:
            limit: Maximum number of pairs to return
        
        Returns:
            List of symbol-strategy pairs sorted by score
        """
        # Get all rankings for all strategies
        all_rankings = self.rank_symbols_for_all_strategies(limit_per_strategy=0)
        
        # Flatten into a single list
        all_pairs = []
        for strategy_id, ranked_symbols in all_rankings.items():
            all_pairs.extend(ranked_symbols)
        
        # Sort by score (descending)
        all_pairs = sorted(all_pairs, key=lambda x: x["score"], reverse=True)
        
        # Limit results
        if limit > 0:
            all_pairs = all_pairs[:limit]
        
        return all_pairs
    
    def _calculate_technical_score(self, symbol_data, strategy_id):
        """
        Calculate technical score based on technicals and strategy.
        
        Args:
            symbol_data: Dictionary containing symbol data
            strategy_id: ID of the strategy
        
        Returns:
            Tuple of (score, reasoning_list)
        """
        if "technicals" not in symbol_data:
            return 0.5, ["Insufficient technical data"]  # Neutral score if no data
        
        technicals = symbol_data["technicals"]
        score = 0.5  # Start with neutral score
        reasoning = []  # List to store reasoning logs
        
        # Strategy-specific scoring logic
        if strategy_id == "momentum_etf":
            # For momentum strategy, prefer bullish trend, with rising prices
            if technicals.get("trend") == "bullish":
                score += 0.3
                reasoning.append("Bullish trend detected (+30%)")
            
            # Not overbought
            if not technicals.get("overbought", False):
                score += 0.2
                reasoning.append("Not overbought, room to run (+20%)")
            else:
                reasoning.append("Overbought conditions detected (0%)")
                
            # MACD positive
            if technicals.get("macd", 0) > 0:
                score += 0.1
                reasoning.append("Positive MACD crossover (+10%)")
            elif technicals.get("macd", 0) < 0:
                reasoning.append("Negative MACD, waiting for crossover (0%)")
                
        elif strategy_id == "value_dividend":
            # For value strategy, prefer oversold conditions
            if technicals.get("oversold", True):
                score += 0.3
                reasoning.append("Oversold conditions detected (+30%)")
            
            # Prefer stocks below SMA 200
            if "sma_200" in technicals and "price" in symbol_data:
                if symbol_data["price"]["current"] < technicals["sma_200"]:
                    score += 0.2
                    reasoning.append("Trading below 200-day SMA, potential value (+20%)")
                else:
                    reasoning.append("Trading above 200-day SMA (0%)")
                    
        elif strategy_id == "mean_reversion":
            # For mean reversion, prefer overbought or oversold conditions
            if technicals.get("overbought", False):
                score += 0.4
                reasoning.append("Overbought conditions ideal for mean reversion (+40%)")
            elif technicals.get("oversold", False):
                score += 0.4
                reasoning.append("Oversold conditions ideal for mean reversion (+40%)")
            else:
                reasoning.append("Neither overbought nor oversold (0%)")
                
            # Prefer stocks near Bollinger Band edges
            if "bollinger_bands" in technicals and "price" in symbol_data:
                current = symbol_data["price"]["current"]
                upper = technicals["bollinger_bands"]["upper"]
                lower = technicals["bollinger_bands"]["lower"]
                
                # Close to upper or lower band
                if current > 0.9 * upper:
                    score += 0.2
                    reasoning.append(f"Price near upper Bollinger Band, potential reversal (+20%)")
                elif current < 1.1 * lower:
                    score += 0.2
                    reasoning.append(f"Price near lower Bollinger Band, potential reversal (+20%)")
        
        # Ensure score is between 0 and 1
        final_score = max(0, min(1, score))
        
        return final_score, reasoning
    
    def _calculate_sentiment_score(self, symbol, context):
        """
        Calculate sentiment score based on news sentiment.
        
        Args:
            symbol: Stock symbol
            context: Market context dictionary
        
        Returns:
            Tuple of (score, reasoning_list)
        """
        # Default to neutral sentiment
        score = 0.5
        reasoning = ["No recent news, defaulting to neutral sentiment"]
        
        # Check if we have news sentiment for this symbol
        if symbol in context.get("news", {}).get("symbols", {}):
            news_items = context["news"]["symbols"][symbol]
            
            # Skip if no news
            if not news_items:
                return score, reasoning
            
            # Count positive and negative sentiment
            positive = 0
            negative = 0
            neutral = 0
            
            for item in news_items:
                sentiment = item.get("sentiment", "Neutral")
                if sentiment == "Positive" or sentiment == "Bullish":
                    positive += 1
                elif sentiment == "Negative" or sentiment == "Bearish":
                    negative += 1
                else:
                    neutral += 1
            
            total = positive + negative + neutral
            if total > 0:
                # Calculate weighted sentiment score
                score = (0.7 * positive + 0.5 * neutral + 0.3 * negative) / total
                
                # Create reasoning text
                if positive > negative:
                    sentiment_strength = "Strong" if score > 0.65 else "Mild"
                    reasoning = [f"{sentiment_strength} bullish news sentiment: {positive} positive, {negative} negative stories"]
                elif negative > positive:
                    sentiment_strength = "Strong" if score < 0.35 else "Mild"
                    reasoning = [f"{sentiment_strength} bearish news sentiment: {negative} negative, {positive} positive stories"]
                else:
                    reasoning = [f"Mixed news sentiment: {positive} positive, {negative} negative, {neutral} neutral stories"]
        
        return score, reasoning
    
    def _calculate_fundamental_score(self, symbol_data):
        """
        Calculate fundamental score (placeholder).
        In a real implementation, this would include PE ratios, 
        EPS growth, dividend yield, etc.
        
        Args:
            symbol_data: Dictionary containing symbol data
        
        Returns:
            Tuple of (score, reasoning_list)
        """
        # This is a placeholder - in a real implementation,
        # you would use actual fundamental data
        score = 0.65  # Slightly positive bias
        
        # In a real implementation, you would analyze actual fundamentals
        # and provide detailed reasoning based on those metrics
        reasoning = ["Fundamental analysis unavailable, using baseline score"]
        
        return score, reasoning
    
    def _calculate_regime_alignment(self, symbol_data, market_regime, strategy_id):
        """
        Calculate how well the symbol aligns with the current market regime
        for the given strategy.
        
        Args:
            symbol_data: Dictionary containing symbol data
            market_regime: Current market regime
            strategy_id: ID of the strategy
        
        Returns:
            Tuple of (score, reasoning_list)
        """
        # Default to medium alignment
        score = 0.5
        reasoning = []
        
        # Strategy-regime alignment rules
        if strategy_id == "momentum_etf":
            if market_regime in ["bullish_trend", "recovery"]:
                score = 0.9
                reasoning.append(f"Market regime '{market_regime}' is ideal for momentum strategies (+90%)")
            elif market_regime in ["stable"]:
                score = 0.7
                reasoning.append(f"Market regime '{market_regime}' supports momentum with some caution (+70%)")
            elif market_regime in ["unsettled"]:
                score = 0.5
                reasoning.append(f"Market regime '{market_regime}' requires caution for momentum plays (+50%)")
            else:
                score = 0.3
                reasoning.append(f"Market regime '{market_regime}' is not favorable for momentum strategies (+30%)")
                
        elif strategy_id == "value_dividend":
            if market_regime in ["correction", "deteriorating"]:
                score = 0.9
                reasoning.append(f"Market regime '{market_regime}' is ideal for value strategies (+90%)")
            elif market_regime in ["stable", "unsettled"]:
                score = 0.7
                reasoning.append(f"Market regime '{market_regime}' is supportive for value strategies (+70%)")
            else:
                score = 0.5
                reasoning.append(f"Market regime '{market_regime}' requires selective value approach (+50%)")
                
        elif strategy_id == "mean_reversion":
            if market_regime in ["high_volatility", "unsettled"]:
                score = 0.9
                reasoning.append(f"Market regime '{market_regime}' is ideal for mean reversion strategies (+90%)")
            elif market_regime in ["correction", "recovery"]:
                score = 0.7
                reasoning.append(f"Market regime '{market_regime}' supports mean reversion selectively (+70%)")
            else:
                score = 0.5
                reasoning.append(f"Market regime '{market_regime}' is neutral for mean reversion (+50%)")
        
        return score, reasoning


# Create a singleton instance
_symbol_ranker = None

def get_symbol_ranker(config=None):
    """
    Get the singleton SymbolRanker instance.
    
    Args:
        config: Optional configuration for the symbol ranker
    
    Returns:
        SymbolRanker instance
    """
    global _symbol_ranker
    if _symbol_ranker is None:
        _symbol_ranker = SymbolRanker(config)
    return _symbol_ranker
