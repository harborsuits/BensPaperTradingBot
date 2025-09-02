#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Strategy Factory

Factory class for creating and configuring event-driven trading strategies
with appropriate parameters and session settings.
"""

import logging
from typing import Dict, Any, Optional, List

from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksSession
from trading_bot.strategies_new.stocks.event.earnings_announcement_strategy import EarningsAnnouncementStrategy
from trading_bot.strategies_new.stocks.event.news_sentiment_strategy import NewsSentimentStrategy

# Configure logging
logger = logging.getLogger(__name__)

class EventStrategyFactory:
    """
    Factory for creating and configuring event-driven trading strategies.
    
    This factory creates properly configured event-driven strategy instances with:
    - Appropriate market parameters
    - Event-specific configuration
    - Historical performance analysis
    - Session settings
    """
    
    @staticmethod
    def create_news_sentiment_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        strategy_mode: str = 'adaptive',
        custom_parameters: Dict[str, Any] = None
    ) -> NewsSentimentStrategy:
        """
        Create a fully configured News Sentiment Strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            strategy_mode: 'trend_following', 'contrarian', or 'adaptive'
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured NewsSentimentStrategy instance
        """
        # Create session
        session = StocksSession(
            symbol=symbol,
            timeframe=timeframe,
            exchange='NYSE',  # Default, would be configured in real implementation
            lot_size=100      # Default lot size for stocks
        )
        
        # Create data pipeline
        data_pipeline = DataPipeline()
        
        # Base parameters
        base_parameters = {
            'strategy_mode': strategy_mode,
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.5, 'social_media': 0.3, 'analyst_reports': 0.2},
        }
        
        # Merge with custom parameters if provided
        parameters = base_parameters.copy()
        if custom_parameters:
            parameters.update(custom_parameters)
        
        # Create strategy
        strategy = NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters
        )
        
        logger.info(f"Created News Sentiment Strategy for {symbol} with mode={strategy_mode}")
        return strategy
    
    @staticmethod
    def create_trend_following_sentiment_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        custom_parameters: Dict[str, Any] = None
    ) -> NewsSentimentStrategy:
        """
        Create a sentiment strategy that follows the sentiment trend.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured trend-following NewsSentimentStrategy instance
        """
        base_params = {
            'strategy_mode': 'trend_following',
            'require_volume_confirmation': True,
            'min_sentiment_volume': 8  # Higher threshold for more certainty
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return EventStrategyFactory.create_news_sentiment_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='trend_following',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_contrarian_sentiment_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        custom_parameters: Dict[str, Any] = None
    ) -> NewsSentimentStrategy:
        """
        Create a sentiment strategy that trades against extreme sentiment.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured contrarian NewsSentimentStrategy instance
        """
        base_params = {
            'strategy_mode': 'contrarian',
            'enable_contrarian_mode': True,
            'contrarian_threshold': 0.75,
            'min_sentiment_volume': 10  # Higher threshold for more certainty
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return EventStrategyFactory.create_news_sentiment_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='contrarian',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_earnings_announcement_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        strategy_mode: str = 'adaptive',
        pre_earnings_entry: bool = True,
        post_earnings_entry: bool = True,
        custom_parameters: Dict[str, Any] = None
    ) -> EarningsAnnouncementStrategy:
        """
        Create a fully configured Earnings Announcement Strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            strategy_mode: 'bullish', 'bearish', or 'adaptive'
            pre_earnings_entry: Whether to enter before earnings
            post_earnings_entry: Whether to enter after earnings
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured EarningsAnnouncementStrategy instance
        """
        # Create session
        session = StocksSession(
            symbol=symbol,
            timeframe=timeframe,
            exchange='NYSE',  # Default, would be configured in real implementation
            lot_size=100      # Default lot size for stocks
        )
        
        # Create data pipeline
        data_pipeline = DataPipeline()
        
        # Base parameters
        base_parameters = {
            'strategy_mode': strategy_mode,
            'pre_earnings_entry': pre_earnings_entry,
            'post_earnings_entry': post_earnings_entry,
            'earnings_calendar': {}  # Would be populated with real data in production
        }
        
        # Merge with custom parameters if provided
        parameters = base_parameters.copy()
        if custom_parameters:
            parameters.update(custom_parameters)
        
        # Create strategy
        strategy = EarningsAnnouncementStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters
        )
        
        logger.info(f"Created Earnings Announcement Strategy for {symbol} with mode={strategy_mode}")
        return strategy
    
    @staticmethod
    def create_event_driven_portfolio(
        symbols: List[str],
        use_news_sentiment: bool = True,
        use_earnings: bool = True,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        custom_parameters: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a portfolio of event-driven strategies for multiple symbols.
        
        Args:
            symbols: List of stock symbols to trade
            use_news_sentiment: Whether to include news sentiment strategies
            use_earnings: Whether to include earnings strategies
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Dictionary of symbol -> strategies
        """
        portfolio = {}
        
        for symbol in symbols:
            portfolio[symbol] = {}
            
            if use_news_sentiment:
                portfolio[symbol]['news_sentiment'] = EventStrategyFactory.create_news_sentiment_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
                
            if use_earnings:
                portfolio[symbol]['earnings'] = EventStrategyFactory.create_earnings_announcement_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
            
        logger.info(f"Created event-driven strategy portfolio with {len(symbols)} symbols")
        return portfolio
