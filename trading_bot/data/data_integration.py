#!/usr/bin/env python3
"""
Data Integration Layer for BensBot

This module provides a unified interface for accessing market data, news, indicators,
and sentiment data across all components of the trading system.
"""

import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.data.data_manager import DataManager

# Set up logger
logger = logging.getLogger("data_integration")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class DataIntegrationLayer:
    """
    Central data integration layer that provides unified access to:
    - Market data (prices, volumes, etc.)
    - News and economic events
    - Technical indicators
    - Sentiment data
    - Fundamental data
    
    This class ensures all components of the system use the same high-quality
    data sources with proper caching and freshness controls.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton access method"""
        if cls._instance is None:
            cls._instance = DataIntegrationLayer()
        return cls._instance
    
    def __init__(self):
        self.initialized = False
        self.news_service = None
        self.data_manager = None
        self.using_real_api = False
        self.source_health = {}
        self.last_health_check = None
        
        # Initialize components
        self._initialize()
    
    def _initialize(self):
        """Initialize all data components and connections"""
        try:
            logger.info("Initializing data integration layer")
            
            # Try to connect to the news service
            try:
                from news_api import news_service
                self.news_service = news_service
                self.using_real_api = True
                logger.info("Successfully connected to news service")
            except Exception as e:
                logger.warning(f"Could not load news service: {e}")
                self.using_real_api = False
            
            # Check if DataManager exists in ServiceRegistry
            if ServiceRegistry.has_service("data_manager"):
                self.data_manager = ServiceRegistry.get("data_manager")
                logger.info("Using existing DataManager from ServiceRegistry")
            else:
                # Initialize data manager with default configuration
                try:
                    # Default configuration
                    data_config = {
                        "enable_cache": True,
                        "cache_expiry_minutes": 30,
                        "data_providers": [
                            {"name": "yahoo_finance", "type": "python_library"}
                        ],
                        "realtime_providers": [],
                        "data_storage": {
                            "base_dir": "data"
                        }
                    }
                    
                    # Create DataManager - it self-registers in ServiceRegistry
                    self.data_manager = DataManager(data_config)
                    logger.info("Created new DataManager and registered in ServiceRegistry")
                except Exception as e:
                    logger.error(f"Failed to initialize DataManager: {e}")
                    self.data_manager = None
            
            self.initialized = True
            self._check_health()
            
            logger.info("Data integration layer initialization complete")
            
        except Exception as e:
            logger.error(f"Error during data integration layer initialization: {e}")
            traceback.print_exc()
    
    def _check_health(self):
        """Check health of all data sources and update status"""
        self.source_health = {
            "news_service": {
                "available": self.news_service is not None,
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Connected" if self.news_service is not None else "Unavailable"
            },
            "data_manager": {
                "available": self.data_manager is not None,
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Connected" if self.data_manager is not None else "Unavailable"
            }
        }
        
        # Check market data providers
        if self.data_manager:
            for provider_name in self.data_manager.market_data_providers:
                self.source_health[f"provider.{provider_name}"] = {
                    "available": True,
                    "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Connected"
                }
        
        self.last_health_check = datetime.now()
        return self.source_health
    
    def get_market_data(self, symbol, timeframe="1d", force_refresh=False):
        """
        Get market price data for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe (1m, 5m, 1h, 1d, etc.)
            force_refresh: Whether to force fresh data
            
        Returns:
            Dict with price data
        """
        symbol = symbol.upper().strip()  # Standardize format
        
        try:
            # First try DataManager if available
            if self.data_manager:
                try:
                    logger.debug(f"Getting market data for {symbol} from DataManager")
                    result = self.data_manager.get_market_data(
                        symbols=symbol,
                        start_date=datetime.now() - timedelta(days=365),
                        end_date=datetime.now(),
                        use_cache=not force_refresh
                    )
                    if result and symbol in result:
                        df = result[symbol]
                        price_data = {
                            "price": float(df.iloc[-1]["close"]) if not df.empty else None,
                            "change": float(df.iloc[-1]["close"] - df.iloc[-2]["close"]) if len(df) > 1 else 0,
                            "source": "data_manager",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "raw_data": df
                        }
                        return price_data
                except Exception as e:
                    logger.warning(f"DataManager failed to get market data: {e}")
            
            # Fall back to news_service for price data if available
            if self.news_service:
                try:
                    logger.debug(f"Getting market data for {symbol} from news_service")
                    fresh_data = self.news_service.get_stock_price(symbol, force_refresh=force_refresh)
                    
                    price_data = {
                        "price": fresh_data.get("current") or fresh_data.get("price"),
                        "change": fresh_data.get("change"),
                        "source": fresh_data.get("source"),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "raw_data": fresh_data
                    }
                    return price_data
                except Exception as e:
                    logger.warning(f"News service failed to get market data: {e}")
            
            # Fall back to mock data if all else fails
            logger.warning(f"Using mock data for {symbol} - no data sources available")
            return {
                "price": 100.0 + (hash(symbol) % 900),  # Deterministic mock data
                "change": (hash(symbol) % 21 - 10) / 10,
                "source": "mock_data",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "raw_data": {}
            }
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {
                "price": 0.0,
                "change": 0.0,
                "source": "error",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
    
    def get_news(self, symbols=None, max_age_minutes=60):
        """
        Get news data for specified symbols.
        
        Args:
            symbols: Symbol or list of symbols (None for market news)
            max_age_minutes: Maximum age of news in minutes
            
        Returns:
            List of news items
        """
        try:
            # Standardize symbols format
            if symbols:
                if isinstance(symbols, str):
                    symbols = [symbols.upper().strip()]
                else:
                    symbols = [s.upper().strip() for s in symbols]
            
            # Try to get news from news_service
            if self.news_service and self.using_real_api:
                try:
                    if symbols:
                        # Get news for specific symbols
                        all_news = []
                        for symbol in symbols:
                            symbol_news = self.news_service.get_symbol_news(symbol)
                            all_news.extend(symbol_news)
                        return all_news
                    else:
                        # Get general market news
                        return self.news_service.get_economic_digest()
                except Exception as e:
                    logger.warning(f"Failed to get news from news_service: {e}")
            
            # Fall back to mock news data
            logger.warning("Using mock news data - news_service unavailable")
            return self._get_mock_news(symbols)
            
        except Exception as e:
            logger.error(f"Error getting news data: {e}")
            return []
    
    def _get_mock_news(self, symbols=None):
        """Generate mock news data for testing"""
        all_news = [
            {"symbol": "ALL", "title": "Markets rally on interest rate concerns", "date": "2025-04-23", "sentiment": "Neutral"},
            {"symbol": "ALL", "title": "Federal Reserve hints at policy change", "date": "2025-04-22", "sentiment": "Positive"},
            {"symbol": "AAPL", "title": "Apple announces new product line", "date": "2025-04-23", "sentiment": "Positive"},
            {"symbol": "MSFT", "title": "Microsoft cloud services revenue up 25%", "date": "2025-04-20", "sentiment": "Positive"},
            {"symbol": "GOOGL", "title": "Google faces new regulatory challenges", "date": "2025-04-21", "sentiment": "Negative"},
            {"symbol": "TSLA", "title": "Tesla production exceeds expectations", "date": "2025-04-22", "sentiment": "Positive"},
            {"symbol": "AMZN", "title": "Amazon expands same-day delivery", "date": "2025-04-19", "sentiment": "Positive"}
        ]
        
        # Filter news by symbols if provided
        if symbols:
            return [n for n in all_news if n["symbol"] in symbols or n["symbol"] == "ALL"]
        return all_news
    
    def get_indicators(self, symbol, indicator_types=None):
        """
        Get technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            indicator_types: List of indicator types to retrieve
            
        Returns:
            Dict of indicator values
        """
        try:
            symbol = symbol.upper().strip()
            
            # Define default indicator types if none specified
            if not indicator_types:
                indicator_types = ["ma", "rsi", "volume", "volatility", "trend", "momentum"]
            
            # Try to use the autonomous helpers function first
            try:
                from trading_bot.backtesting.autonomous_helpers import get_symbol_indicators
                indicators = get_symbol_indicators(symbol)
                return indicators
            except Exception as e:
                logger.warning(f"Failed to get indicators from helpers: {e}")
            
            # Fall back to mock indicators
            logger.warning(f"Using mock indicators for {symbol}")
            return self._get_mock_indicators(symbol)
            
        except Exception as e:
            logger.error(f"Error getting indicators for {symbol}: {e}")
            return {}
    
    def _get_mock_indicators(self, symbol):
        """Generate mock indicator data for testing"""
        # Make it deterministic but vary by symbol
        seed = sum(ord(c) for c in symbol)
        np.random.seed(seed)
        
        return {
            "ma_50": 100 + np.random.normal(0, 10),
            "ma_200": 105 + np.random.normal(0, 15),
            "rsi": min(max(np.random.normal(50, 15), 5), 95),
            "adx": min(max(np.random.normal(25, 10), 5), 60),
            "volatility": min(max(np.random.normal(15, 5), 5), 40),
            "trend_strength": min(max(np.random.normal(50, 20), 10), 90),
            "momentum": min(max(np.random.normal(0, 1), -3), 3),
            "mean_reversion": min(max(np.random.normal(0, 0.5), -1), 1)
        }
    
    def get_sentiment(self, symbol):
        """
        Get sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with sentiment data
        """
        try:
            symbol = symbol.upper().strip()
            
            # Try to use the autonomous helpers function
            try:
                from trading_bot.backtesting.autonomous_helpers import get_symbol_sentiment
                sentiment = get_symbol_sentiment(symbol)
                return sentiment
            except Exception as e:
                logger.warning(f"Failed to get sentiment from helpers: {e}")
            
            # Fall back to mock sentiment
            logger.warning(f"Using mock sentiment for {symbol}")
            return self._get_mock_sentiment(symbol)
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {}
    
    def _get_mock_sentiment(self, symbol):
        """Generate mock sentiment data for testing"""
        # Make it deterministic but vary by symbol
        seed = sum(ord(c) for c in symbol)
        np.random.seed(seed)
        
        return {
            "news_sentiment": min(max(np.random.normal(0, 0.5), -1), 1),
            "social_sentiment": min(max(np.random.normal(0, 0.6), -1), 1),
            "analyst_sentiment": min(max(np.random.normal(0.2, 0.4), -1), 1),
            "overall_sentiment": min(max(np.random.normal(0.1, 0.3), -1), 1),
            "sentiment_trend": ["Improving", "Stable", "Declining"][np.random.randint(0, 3)],
            "confidence": min(max(np.random.normal(70, 15), 30), 95)
        }
    
    def get_health_status(self):
        """
        Get health status of all data sources.
        
        Returns:
            Dict with health status of each data source
        """
        # Update health status if last check was more than 5 minutes ago
        if not self.last_health_check or \
           (datetime.now() - self.last_health_check).total_seconds() > 300:
            self._check_health()
        
        return self.source_health


# Initialize the singleton instance
data_layer = DataIntegrationLayer.get_instance()

# Helper functions for easy access

def get_market_data(symbol, timeframe="1d", force_refresh=False):
    """
    Get market data for a symbol, standardized across the application.
    
    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        force_refresh: Whether to force fresh data
        
    Returns:
        Dict with price data
    """
    return data_layer.get_market_data(symbol, timeframe, force_refresh)

def get_news(symbols=None, max_age_minutes=60):
    """
    Get news data, standardized across the application.
    
    Args:
        symbols: Symbol or list of symbols (None for market news)
        max_age_minutes: Maximum age of news in minutes
        
    Returns:
        List of news items
    """
    return data_layer.get_news(symbols, max_age_minutes)

def get_indicators(symbol, indicator_types=None):
    """
    Get technical indicators for a symbol, standardized across the application.
    
    Args:
        symbol: Stock symbol
        indicator_types: List of indicator types to retrieve
        
    Returns:
        Dict of indicator values
    """
    return data_layer.get_indicators(symbol, indicator_types)

def get_sentiment(symbol):
    """
    Get sentiment data for a symbol, standardized across the application.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dict with sentiment data
    """
    return data_layer.get_sentiment(symbol)

def get_health_status():
    """
    Get health status of all data sources.
    
    Returns:
        Dict with health status of each data source
    """
    return data_layer.get_health_status()
