# trading_bot/core/data_hub.py
"""
CentralDataHub: Unified data access layer for the trading system.
Fetches, caches, and serves news, sentiment, indicators, and price data to all modules.
"""

from trading_bot.backtesting.data_integration import DataIntegrationLayer
from trading_bot.data.sources.yahoo import YahooFinanceDataSource
from trading_bot.data.sources.alpha_vantage import AlphaVantageDataSource
# Add imports for any other data sources you want to support
import threading
import time
import logging

logger = logging.getLogger("CentralDataHub")

class CentralDataHub:
    def __init__(self, news_fetcher, indicator_calculator, cache_expiry_seconds=900):
        """
        Args:
            news_fetcher: News fetcher or adapter instance
            indicator_calculator: Indicator calculator instance
            cache_expiry_seconds: How long to cache results (default 15 min)
        """
        # Use Yahoo as default price data source, fallback to AlphaVantage
        self.price_data_sources = [YahooFinanceDataSource(), AlphaVantageDataSource()]
        self.data_layer = DataIntegrationLayer(news_fetcher, self.price_data_sources[0], indicator_calculator)
        self.cache_expiry = cache_expiry_seconds
        self._cache = {}
        self._cache_times = {}
        self._lock = threading.Lock()

    def _is_cache_valid(self, key):
        return key in self._cache and (time.time() - self._cache_times[key]) < self.cache_expiry

    def get_news(self, symbol=None):
        key = f"news:{symbol or 'market'}"
        with self._lock:
            if self._is_cache_valid(key):
                return self._cache[key]
        news = self.data_layer._get_news_data(symbol)
        with self._lock:
            self._cache[key] = news
            self._cache_times[key] = time.time()
        return news

    def get_sentiment(self, symbol=None):
        key = f"sentiment:{symbol or 'market'}"
        with self._lock:
            if self._is_cache_valid(key):
                return self._cache[key]
        news = self.get_news(symbol)
        sentiment = self.data_layer.sentiment_analyzer.analyze_dimensions(news)
        with self._lock:
            self._cache[key] = sentiment
            self._cache_times[key] = time.time()
        return sentiment

    def get_indicators(self, symbol, timeframe="1y"):
        key = f"indicators:{symbol}:{timeframe}"
        with self._lock:
            if self._is_cache_valid(key):
                return self._cache[key]
        price_data = self.get_price_data(symbol, timeframe)
        indicators = self.data_layer._calculate_indicators(price_data)
        with self._lock:
            self._cache[key] = indicators
            self._cache_times[key] = time.time()
        return indicators

    def get_price_data(self, symbol, timeframe="1y"):
        key = f"price:{symbol}:{timeframe}"
        with self._lock:
            if self._is_cache_valid(key):
                return self._cache[key]
        price_data = self.data_layer._get_market_data(symbol, timeframe)
        with self._lock:
            self._cache[key] = price_data
            self._cache_times[key] = time.time()
        return price_data

    def get_comprehensive_data(self, symbol, timeframe="1y", sectors=None):
        key = f"comprehensive:{symbol}:{timeframe}:{','.join(sectors) if sectors else 'all'}"
        with self._lock:
            if self._is_cache_valid(key):
                return self._cache[key]
        data = self.data_layer.get_comprehensive_data(symbol, timeframe, sectors)
        with self._lock:
            self._cache[key] = data
            self._cache_times[key] = time.time()
        return data
