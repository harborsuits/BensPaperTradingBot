"""
Market Analysis Integration Module

This module integrates various market analysis components with real market data sources.
It provides unified interfaces for retrieving market regime, sentiment analysis, and other 
market context information from actual data providers instead of simulated data.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from pydantic import BaseModel

from trading_bot.data_sources.market_data_adapter import MarketDataAdapter, OHLCV
from trading_bot.market_analysis.regime_detection import (
    RegimeDetector, MarketRegimeResult, create_regime_detector, RegimeMethod, MarketRegimeType
)
from trading_bot.market_analysis.sentiment_analysis import (
    SentimentAnalyzer, MarketSentimentResult, create_sentiment_analyzer, 
    NewsItem, SocialMediaPost
)

logger = logging.getLogger("market_analysis.analyzer")

class MarketAnalysisData(BaseModel):
    """Combined market analysis data"""
    regime: MarketRegimeResult
    sentiment: MarketSentimentResult
    timestamp: str
    ticker: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True


class MarketAnalyzer:
    """
    Integrates various market analysis components and connects them to real market data sources.
    """
    
    def __init__(
        self,
        market_data_adapters: Dict[str, MarketDataAdapter],
        config: Dict[str, Any] = None
    ):
        """
        Initialize the market analyzer with data adapters
        
        Args:
            market_data_adapters: Dictionary of market data adapters by name
            config: Configuration options
        """
        self.adapters = market_data_adapters
        self.config = config or {}
        self.logger = logging.getLogger("market_analyzer")
        
        # Initialize regime detector
        regime_method = self.config.get("regime_method", RegimeMethod.MULTI_FACTOR)
        regime_config = self.config.get("regime_config", {})
        self.regime_detector = create_regime_detector(regime_method, regime_config)
        
        # Initialize sentiment analyzer
        sentiment_method = self.config.get("sentiment_method", "rule_based")
        sentiment_config = self.config.get("sentiment_config", {})
        self.sentiment_analyzer = create_sentiment_analyzer(sentiment_method, sentiment_config)
        
        # Cache for analysis results to avoid excessive API calls
        self.analysis_cache = {}
        self.cache_expiry = self.config.get("cache_expiry_seconds", 300)  # 5 minutes default
    
    async def get_market_analysis(
        self, 
        symbol: str = "SPY",  # Default to SPY for broad market
        adapter_name: Optional[str] = None
    ) -> MarketAnalysisData:
        """
        Get comprehensive market analysis for a symbol
        
        Args:
            symbol: The ticker symbol to analyze
            adapter_name: Specific adapter to use, or None to use the first available
            
        Returns:
            Combined market analysis data
        """
        # Check cache first
        cache_key = f"{symbol}:{adapter_name or 'default'}"
        if cache_key in self.analysis_cache:
            cached_data, timestamp = self.analysis_cache[cache_key]
            # If cache is still fresh
            if (datetime.now() - timestamp).total_seconds() < self.cache_expiry:
                return cached_data
        
        # Get market data adapter
        adapter = self._get_adapter(adapter_name)
        if not adapter:
            self.logger.error(f"No market data adapter available for analysis: {adapter_name}")
            # Return default data in case of error
            return self._create_default_analysis(symbol)
        
        try:
            # Fetch OHLCV data for regime detection
            ohlcv_data = await adapter.get_price_history(
                symbol, 
                limit=60,  # Need enough data for reliable regime detection
                timeframe=adapter.get_default_timeframe()
            )
            
            # Fetch technical indicators
            indicators = await self._fetch_indicators(adapter, symbol)
            
            # Fetch news data for sentiment analysis
            news_items = await self._fetch_news_items(adapter, symbol)
            
            # Fetch additional market data for comprehensive analysis
            additional_data = await self._fetch_additional_data(adapter, symbol)
            
            # Perform regime detection
            regime_result = self.regime_detector.detect(
                ohlcv_data=ohlcv_data,
                indicators=indicators,
                additional_data=additional_data
            )
            
            # Perform sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_market_sentiment(
                news_items=news_items,
                social_posts=None,  # Placeholder for future social media integration
                additional_data=additional_data
            )
            
            # Combine results
            analysis_data = MarketAnalysisData(
                regime=regime_result,
                sentiment=sentiment_result,
                timestamp=datetime.now().isoformat(),
                ticker=symbol
            )
            
            # Update cache
            self.analysis_cache[cache_key] = (analysis_data, datetime.now())
            
            return analysis_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data: {str(e)}", exc_info=True)
            return self._create_default_analysis(symbol)
    
    async def _fetch_indicators(
        self,
        adapter: MarketDataAdapter,
        symbol: str
    ) -> Dict[str, Any]:
        """Fetch technical indicators for a symbol"""
        indicators = {}
        
        try:
            # Common technical indicators needed for regime detection
            indicator_names = ["rsi", "macd", "macd_signal", "macd_hist", "bbands", "atr"]
            
            # Fetch each indicator
            for name in indicator_names:
                indicator_data = await adapter.get_indicator(
                    symbol, 
                    name, 
                    timeframe=adapter.get_default_timeframe()
                )
                
                if indicator_data:
                    indicators[name] = indicator_data
            
        except Exception as e:
            self.logger.warning(f"Error fetching indicators for {symbol}: {str(e)}")
        
        return indicators
    
    async def _fetch_news_items(
        self,
        adapter: MarketDataAdapter,
        symbol: str
    ) -> List[NewsItem]:
        """Fetch news items for a symbol"""
        news_items = []
        
        try:
            # Fetch news from the adapter
            raw_news = await adapter.get_news(symbol, limit=20)
            
            # Convert to NewsItem objects
            if raw_news:
                for news in raw_news:
                    # Process each news item
                    news_item = NewsItem(
                        title=news.get("headline", ""),
                        source=news.get("source", ""),
                        url=news.get("url", ""),
                        published_at=news.get("datetime", datetime.now().isoformat()),
                        summary=news.get("summary", ""),
                        # Pre-analyze sentiment or use adapter-provided sentiment
                        sentiment=news.get("sentiment", 0.0),
                        relevance=news.get("relevance", 1.0),
                        topics=[],  # Will be filled by the sentiment analyzer
                        tickers=news.get("related", [symbol])
                    )
                    
                    # If no sentiment is provided, analyze it
                    if news_item.sentiment == 0.0 and news_item.summary:
                        analyzed_item = self.sentiment_analyzer.analyze_news_item(
                            news_item.title,
                            news_item.summary,
                            news_item.source
                        )
                        news_item.sentiment = analyzed_item.sentiment
                        news_item.topics = analyzed_item.topics
                    
                    news_items.append(news_item)
        
        except Exception as e:
            self.logger.warning(f"Error fetching news for {symbol}: {str(e)}")
        
        return news_items
    
    async def _fetch_additional_data(
        self,
        adapter: MarketDataAdapter,
        symbol: str
    ) -> Dict[str, Any]:
        """Fetch additional market data for comprehensive analysis"""
        additional_data = {}
        
        try:
            # Fetch market breadth data if available
            if hasattr(adapter, "get_market_breadth") and callable(getattr(adapter, "get_market_breadth")):
                breadth_data = await adapter.get_market_breadth()
                if breadth_data:
                    additional_data["market_breadth"] = breadth_data
            
            # Fetch sector data if available
            if hasattr(adapter, "get_sector_performance") and callable(getattr(adapter, "get_sector_performance")):
                sector_data = await adapter.get_sector_performance()
                if sector_data:
                    additional_data["sector_performance"] = sector_data
            
            # Fetch earnings data if available and relevant
            if symbol and hasattr(adapter, "get_earnings") and callable(getattr(adapter, "get_earnings")):
                earnings_data = await adapter.get_earnings(symbol)
                if earnings_data:
                    additional_data["earnings"] = earnings_data
            
        except Exception as e:
            self.logger.warning(f"Error fetching additional data: {str(e)}")
        
        return additional_data
    
    def _get_adapter(self, adapter_name: Optional[str] = None) -> Optional[MarketDataAdapter]:
        """Get the appropriate market data adapter"""
        if not self.adapters:
            return None
        
        if adapter_name and adapter_name in self.adapters:
            return self.adapters[adapter_name]
        
        # Use first available adapter if none specified
        return next(iter(self.adapters.values()))
    
    def _create_default_analysis(self, symbol: str) -> MarketAnalysisData:
        """Create default analysis in case of errors"""
        # Default regime: sideways with low confidence
        regime = MarketRegimeResult(
            primary_regime=MarketRegimeType.SIDEWAYS,
            confidence=0.5,
            features={"error": 1.0}
        )
        
        # Default sentiment: neutral
        sentiment = MarketSentimentResult(
            overall_sentiment=0.0,
            bullish_factors=[],
            bearish_factors=[]
        )
        
        return MarketAnalysisData(
            regime=regime,
            sentiment=sentiment,
            timestamp=datetime.now().isoformat(),
            ticker=symbol
        )

    async def get_multi_asset_analysis(
        self,
        symbols: List[str],
        adapter_name: Optional[str] = None
    ) -> Dict[str, MarketAnalysisData]:
        """
        Get market analysis for multiple assets in parallel
        
        Args:
            symbols: List of ticker symbols to analyze
            adapter_name: Specific adapter to use
            
        Returns:
            Dictionary of market analysis by symbol
        """
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_market_analysis(symbol, adapter_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analysis_by_symbol = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                self.logger.error(f"Error analyzing {symbol}: {str(result)}")
                analysis_by_symbol[symbol] = self._create_default_analysis(symbol)
            else:
                analysis_by_symbol[symbol] = result
        
        return analysis_by_symbol
    
    def get_cached_analysis(self, symbol: str, adapter_name: Optional[str] = None) -> Optional[MarketAnalysisData]:
        """
        Get cached analysis if available and not expired
        
        Args:
            symbol: Ticker symbol
            adapter_name: Adapter name
            
        Returns:
            Cached analysis or None if not available
        """
        cache_key = f"{symbol}:{adapter_name or 'default'}"
        if cache_key in self.analysis_cache:
            cached_data, timestamp = self.analysis_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_expiry:
                return cached_data
        return None
    
    def clear_cache(self):
        """Clear the analysis cache"""
        self.analysis_cache = {}


# Factory function
def create_market_analyzer(
    market_data_adapters: Dict[str, MarketDataAdapter],
    config: Dict[str, Any] = None
) -> MarketAnalyzer:
    """Create a market analyzer with the specified adapters and configuration"""
    return MarketAnalyzer(market_data_adapters, config)
