"""
MarketContext - Unified Market Data Provider
This module provides a centralized JSON object that serves as the single source of truth
for all market data, technical indicators, news sentiment, and other market intelligence.
All downstream components (strategy selection, backtesting, ML/RL) consume this object.
"""

import json
import time
import datetime
import logging
import threading
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np

# Import your existing services
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from news_api import news_service

class MarketContext:
    """
    Unified market context object that serves as the central data provider for all
    trading decisions, strategy selection, and machine learning models.
    """
    
    def __init__(self, config=None):
        """
        Initialize the MarketContext with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self._config = config or {}
        self._data = {
            "meta": {
                "timestamp": time.time(),
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "version": "1.0.0"
            },
            "market": {
                "indicators": {},
                "regime": "unknown",
                "volatility": {},
                "sectors": {}
            },
            "symbols": {},
            "news": {
                "market": [],
                "symbols": {}
            },
            "strategies": {
                "ranked": [],
                "performance": {}
            },
            "symbol_strategy_pairs": []
        }
        
        # Set up logging
        self.logger = logging.getLogger("MarketContext")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Thread lock for updating the context
        self._lock = threading.RLock()
        
        self.logger.info("MarketContext initialized")
    
    def update_market_data(self):
        """Update all market-wide data"""
        self.logger.info("Updating market data")
        
        with self._lock:
            # Get market indicators from the news service
            try:
                # This calls your existing economic digest function
                market_data = news_service.get_economic_digest()
                
                if "market_indicators" in market_data:
                    self._data["market"]["indicators"] = market_data["market_indicators"]
                
                # Extract sector performance
                if "market_indicators" in market_data and "sector_performance" in market_data["market_indicators"]:
                    self._data["market"]["sectors"] = market_data["market_indicators"]["sector_performance"]
                
                # Set market regime based on indicators
                self._data["market"]["regime"] = self._determine_market_regime(market_data)
                
                # Set volatility metrics
                if "market_indicators" in market_data and "vix" in market_data["market_indicators"]:
                    vix = market_data["market_indicators"]["vix"]
                    self._data["market"]["volatility"] = {
                        "vix": vix,
                        "level": "high" if vix > 25 else "medium" if vix > 15 else "low"
                    }
                
                # Update market news
                if "high_impact" in market_data:
                    self._data["news"]["market"] = market_data["high_impact"]
                
                self._data["meta"]["timestamp"] = time.time()
                self._data["meta"]["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
                self._data["meta"]["time"] = datetime.datetime.now().strftime("%H:%M:%S")
            
            except Exception as e:
                self.logger.error(f"Error updating market data: {str(e)}")
    
    def update_symbol_data(self, symbols):
        """
        Update data for specific symbols.
        
        Args:
            symbols: List of stock symbols to update
        """
        self.logger.info(f"Updating symbol data for {symbols}")
        
        with self._lock:
            for symbol in symbols:
                try:
                    # Get price data
                    price_data = news_service.get_stock_price(symbol, force_refresh=True)
                    
                    # Get news data
                    news_data = news_service.get_news_by_symbol(symbol)
                    
                    # Calculate technical indicators
                    indicators = self._calculate_technical_indicators(symbol, price_data)
                    
                    # Store everything in the context
                    if symbol not in self._data["symbols"]:
                        self._data["symbols"][symbol] = {}
                    
                    self._data["symbols"][symbol]["price"] = price_data
                    self._data["symbols"][symbol]["technicals"] = indicators
                    
                    if symbol not in self._data["news"]["symbols"]:
                        self._data["news"]["symbols"][symbol] = []
                    
                    self._data["news"]["symbols"][symbol] = news_data
                
                except Exception as e:
                    self.logger.error(f"Error updating data for {symbol}: {str(e)}")
    
    def _determine_market_regime(self, market_data):
        """
        Determine the current market regime based on indicators.
        
        Args:
            market_data: Dictionary containing market indicators
        
        Returns:
            String representing the market regime
        """
        # Simple regime classification logic - can be enhanced with ML later
        if "market_indicators" not in market_data:
            return "unknown"
        
        indicators = market_data["market_indicators"]
        
        # Default to neutral
        regime = "neutral"
        
        # Check VIX (volatility)
        vix = indicators.get("vix", 15)
        
        # Check market direction
        market_direction = indicators.get("market_direction", "neutral")
        
        # High volatility regimes
        if vix > 30:
            if market_direction == "bearish":
                regime = "crisis"
            else:
                regime = "high_volatility"
        # Moderate volatility
        elif vix > 20:
            if market_direction == "bearish":
                regime = "correction"
            elif market_direction == "bullish":
                regime = "recovery"
            else:
                regime = "unsettled"
        # Low volatility
        else:
            if market_direction == "bearish":
                regime = "deteriorating"
            elif market_direction == "bullish":
                regime = "bullish_trend"
            else:
                regime = "stable"
                
        return regime
    
    def _calculate_technical_indicators(self, symbol, price_data):
        """
        Calculate technical indicators for a symbol based on price data.
        This is a placeholder - in a real implementation, you'd calculate actual indicators.
        
        Args:
            symbol: The stock symbol
            price_data: Dictionary containing price data
        
        Returns:
            Dictionary of technical indicators
        """
        # Simple placeholder indicators - this would be expanded in a real implementation
        current = price_data.get("current", 0)
        
        # This is just placeholder data - in a real implementation,
        # you would calculate these from historical data
        indicators = {
            "sma_50": current * (1 + (np.random.random() - 0.5) * 0.05),
            "sma_200": current * (1 + (np.random.random() - 0.5) * 0.1),
            "rsi_14": min(max(30 + np.random.random() * 40, 0), 100),
            "macd": (np.random.random() - 0.5) * 2,
            "bollinger_bands": {
                "upper": current * 1.05,
                "middle": current,
                "lower": current * 0.95
            }
        }
        
        # Add some derived signals
        indicators["trend"] = "bullish" if indicators["sma_50"] > indicators["sma_200"] else "bearish"
        indicators["overbought"] = indicators["rsi_14"] > 70
        indicators["oversold"] = indicators["rsi_14"] < 30
        
        return indicators
    
    def update_strategy_rankings(self):
        """
        Update strategy rankings based on the current market context.
        """
        self.logger.info("Updating strategy rankings")
        
        with self._lock:
            try:
                # This would call your strategy ranking system
                # For now, we'll add placeholder data
                self._data["strategies"]["ranked"] = [
                    {"id": "momentum_etf", "score": 0.85, "suitable_regimes": ["bullish_trend", "recovery"]},
                    {"id": "value_dividend", "score": 0.82, "suitable_regimes": ["stable", "deteriorating"]},
                    {"id": "sector_rotation", "score": 0.78, "suitable_regimes": ["recovery", "stable"]},
                    {"id": "volatility_etf", "score": 0.65, "suitable_regimes": ["high_volatility", "crisis"]},
                    {"id": "mean_reversion", "score": 0.62, "suitable_regimes": ["high_volatility", "unsettled"]}
                ]
                
                # Filter strategies that match the current regime
                current_regime = self._data["market"]["regime"]
                matching_strategies = [
                    strategy for strategy in self._data["strategies"]["ranked"]
                    if current_regime in strategy.get("suitable_regimes", [])
                ]
                
                # If no matching strategies, use the top 3 overall
                if not matching_strategies and self._data["strategies"]["ranked"]:
                    matching_strategies = self._data["strategies"]["ranked"][:3]
                
                # Store the filtered strategies
                self._data["strategies"]["matching"] = matching_strategies
                
            except Exception as e:
                self.logger.error(f"Error updating strategy rankings: {str(e)}")
    
    def match_symbols_to_strategies(self):
        """
        Match symbols to appropriate strategies based on their characteristics.
        """
        self.logger.info("Matching symbols to strategies")
        
        with self._lock:
            try:
                # Get top strategies
                strategies = self._data["strategies"].get("matching", [])
                if not strategies and "ranked" in self._data["strategies"]:
                    strategies = self._data["strategies"]["ranked"][:3]
                
                # Clear existing pairs
                self._data["symbol_strategy_pairs"] = []
                
                # For each symbol, find the best matching strategy
                for symbol, data in self._data["symbols"].items():
                    # Skip symbols with no technical data
                    if "technicals" not in data:
                        continue
                    
                    technicals = data["technicals"]
                    
                    # Simple matching logic - this would be much more sophisticated in a real implementation
                    best_strategy = None
                    best_score = 0
                    
                    for strategy in strategies:
                        strategy_id = strategy["id"]
                        
                        # Calculate match score based on simple rules
                        # This is just an example - real implementation would be more sophisticated
                        score = 0
                        
                        # Momentum strategy likes bullish trends and not overbought
                        if strategy_id == "momentum_etf":
                            if technicals.get("trend") == "bullish":
                                score += 0.5
                            if not technicals.get("overbought", False):
                                score += 0.3
                            
                        # Value strategy likes oversold conditions
                        elif strategy_id == "value_dividend":
                            if technicals.get("oversold", False):
                                score += 0.7
                            if technicals.get("trend") == "bearish":
                                score += 0.2
                        
                        # Mean reversion likes overbought or oversold conditions
                        elif strategy_id == "mean_reversion":
                            if technicals.get("overbought", False) or technicals.get("oversold", False):
                                score += 0.8
                        
                        # Adjust score by strategy's overall ranking
                        base_score = strategy.get("score", 0.5)
                        adjusted_score = score * base_score
                        
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_strategy = strategy_id
                    
                    if best_strategy and best_score > 0.3:  # Threshold for including a pair
                        pair = {
                            "symbol": symbol,
                            "strategy": best_strategy,
                            "score": best_score,
                            "technicals": technicals
                        }
                        self._data["symbol_strategy_pairs"].append(pair)
                
                # Sort pairs by score (descending)
                self._data["symbol_strategy_pairs"] = sorted(
                    self._data["symbol_strategy_pairs"],
                    key=lambda x: x.get("score", 0),
                    reverse=True
                )
                
            except Exception as e:
                self.logger.error(f"Error matching symbols to strategies: {str(e)}")
    
    def get_market_context(self):
        """
        Get the complete market context as a JSON-serializable dictionary.
        
        Returns:
            Dictionary containing the complete market context
        """
        with self._lock:
            return self._data.copy()
    
    def get_top_symbol_strategy_pairs(self, limit=5):
        """
        Get the top symbol-strategy pairs.
        
        Args:
            limit: Maximum number of pairs to return
        
        Returns:
            List of symbol-strategy pairs
        """
        with self._lock:
            pairs = self._data.get("symbol_strategy_pairs", [])
            return pairs[:limit]
    
    def save_to_file(self, filepath):
        """
        Save the current market context to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        with self._lock:
            with open(filepath, 'w') as f:
                json.dump(self._data, f, indent=2)
        
        self.logger.info(f"Saved market context to {filepath}")
    
    def load_from_file(self, filepath):
        """
        Load market context from a JSON file.
        
        Args:
            filepath: Path to load the JSON file from
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                self._data = data
            
            self.logger.info(f"Loaded market context from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading market context from {filepath}: {str(e)}")


# Create a singleton instance
_market_context = None

def get_market_context(config=None):
    """
    Get the singleton MarketContext instance.
    
    Args:
        config: Optional configuration for the market context
    
    Returns:
        MarketContext instance
    """
    global _market_context
    if _market_context is None:
        _market_context = MarketContext(config)
    return _market_context
