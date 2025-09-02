#!/usr/bin/env python
"""
Market Analysis Test Script

This script demonstrates how to use the market analysis framework
with either real API data or mock data for testing.
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import market analysis components
from trading_bot.data_sources.market_data_adapter import MarketDataConfig, TimeFrame
from trading_bot.data_sources.mock_adapter import MockMarketDataAdapter
from trading_bot.market_analysis.regime_detection import RegimeMethod
from trading_bot.market_analysis.market_analyzer import create_market_analyzer

# Try to import Finnhub adapter if available
try:
    from trading_bot.data_sources.finnhub_adapter import FinnhubAdapter
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

async def main():
    """Run the market analysis test"""
    logger = logging.getLogger("test_market_analysis")
    logger.info("Starting market analysis test")
    
    # Initialize adapters
    adapters = {}
    
    # Try to use Finnhub if API key is available
    finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
    if FINNHUB_AVAILABLE and finnhub_api_key:
        logger.info("Using Finnhub adapter with API key")
        finnhub_config = MarketDataConfig(
            name="finnhub",
            api_key=finnhub_api_key,
            base_url="https://finnhub.io/api/v1",
            source_type="REST",
            rate_limit=30
        )
        adapters["finnhub"] = FinnhubAdapter(finnhub_config)
    else:
        if not FINNHUB_AVAILABLE:
            logger.warning("Finnhub adapter not available")
        if not finnhub_api_key:
            logger.warning("Finnhub API key not found. Set FINNHUB_API_KEY environment variable for real data")
        
        # Use mock adapter for testing
        logger.info("Using mock adapter for testing")
        mock_config = MarketDataConfig(
            name="mock",
            api_key="mock_key",
            base_url="",
            source_type="MOCK",
            rate_limit=100
        )
        adapters["mock"] = MockMarketDataAdapter(mock_config)
    
    # Initialize market analyzer
    analyzer = create_market_analyzer(
        adapters,
        config={
            "regime_method": RegimeMethod.MULTI_FACTOR,
            "sentiment_method": "rule_based",
            "cache_expiry_seconds": 300,  # 5-minute cache for market analysis
        }
    )
    
    # Test symbols to analyze
    symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
    
    for symbol in symbols:
        logger.info(f"Analyzing {symbol}...")
        
        # Get market analysis
        analysis = await analyzer.get_market_analysis(symbol)
        
        # Print regime information
        logger.info(f"Market Regime for {symbol}: {analysis.regime.primary_regime} "
                   f"with {analysis.regime.confidence:.1%} confidence")
        
        if analysis.regime.secondary_regime:
            logger.info(f"Secondary Regime: {analysis.regime.secondary_regime} "
                       f"with {analysis.regime.secondary_confidence:.1%} confidence")
        
        # Print top regime features
        logger.info("Top Regime Features:")
        for name, value in analysis.regime.features.items():
            logger.info(f"  {name}: {value:.3f}")
        
        # Print sentiment information
        logger.info(f"Market Sentiment: {analysis.sentiment.overall_sentiment:.2f} "
                   f"(-1.0 to 1.0 scale, negative = bearish, positive = bullish)")
        
        if analysis.sentiment.bullish_factors:
            logger.info("Top Bullish Factors:")
            for factor in analysis.sentiment.bullish_factors[:3]:  # Top 3
                logger.info(f"  {factor.topic}: {factor.description}")
        
        if analysis.sentiment.bearish_factors:
            logger.info("Top Bearish Factors:")
            for factor in analysis.sentiment.bearish_factors[:3]:  # Top 3
                logger.info(f"  {factor.topic}: {factor.description}")
        
        logger.info("-" * 50)
    
    # Get some news data
    logger.info("Getting latest market news...")
    adapter = analyzer._get_adapter()
    news = await adapter.get_latest_news(limit=5)
    
    for item in news:
        logger.info(f"News: {item.title}")
        logger.info(f"  Source: {item.source}, Published: {item.published_at}")
        logger.info(f"  Sentiment: {item.sentiment_score:.2f} if hasattr(item, 'sentiment_score') else 'N/A'}")
        logger.info(f"  URL: {item.url}")
        logger.info("-" * 40)
    
    logger.info("Market analysis test completed")

def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for printing"""
    return json.dumps(data, indent=2, default=str)

if __name__ == "__main__":
    asyncio.run(main())
