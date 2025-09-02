"""
Market Analysis Package

This package provides market analysis functionality including regime detection,
sentiment analysis, and integration with real market data sources.
"""

from trading_bot.market_analysis.regime_detection import (
    MarketRegimeType, RegimeMethod, RegimeFeature, MarketRegimeResult,
    RegimeDetector, MultifactorRegimeDetector, create_regime_detector
)

from trading_bot.market_analysis.sentiment_analysis import (
    SentimentSource, SentimentTopic, SentimentInfluence, MarketSentimentResult,
    NewsItem, SocialMediaPost, SentimentAnalyzer, KeywordSentimentAnalyzer,
    RuleBasedSentimentAnalyzer, create_sentiment_analyzer
)

from trading_bot.market_analysis.market_analyzer import (
    MarketAnalysisData, MarketAnalyzer, create_market_analyzer
)
