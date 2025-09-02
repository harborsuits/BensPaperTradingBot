#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News Sentiment Strategy Factory

This module provides the factory class for creating and configuring
NewsSentimentStrategy instances with various presets and customization options.

The factory supports creating strategies optimized for different sentiment sources,
market conditions, and trading styles.
"""

import logging
from typing import Dict, List, Optional, Any

from trading_bot.core.events.event_bus import EventBus
from trading_bot.core.constants import TimeFrame
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksSession
from trading_bot.strategies_new.stocks.event.news_sentiment_strategy import NewsSentimentStrategy
from trading_bot.strategies_new.factory.registry import register_strategy

logger = logging.getLogger(__name__)


class NewsSentimentFactory:
    """
    Factory class for creating and configuring NewsSentimentStrategy instances
    with various presets and customization options.
    """
    
    @classmethod
    def create_balanced_sentiment_strategy(cls,
                                         session: StocksSession,
                                         data_pipeline: DataPipeline,
                                         event_bus: Optional[EventBus] = None,
                                         **kwargs) -> NewsSentimentStrategy:
        """
        Create a balanced news sentiment strategy with equal weighting
        across sentiment sources and moderate risk parameters.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured NewsSentimentStrategy instance
        """
        parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.4, 'social_media': 0.3, 'analyst_reports': 0.3},
            
            # Sentiment thresholds
            'bullish_threshold': 0.6,       # Minimum score to consider bullish (0-1)
            'bearish_threshold': 0.4,       # Maximum score to consider bearish (0-1)
            'neutral_band': 0.1,            # Neutral zone around 0.5 (0.5 +/- this value)
            'significant_change': 0.15,     # Minimum change to be considered significant
            
            # Trading parameters
            'min_sentiment_volume': 5,      # Minimum news volume to consider reliable
            'lookback_period_days': 3,      # Days to look back for sentiment analysis
            'sentiment_smoothing': 7,       # Rolling window for sentiment smoothing
            'entry_delay_bars': 2,          # Bars to wait after sentiment signal before entry
            'max_hold_period_days': 5,      # Maximum days to hold a sentiment-based position
            
            # Risk management
            'position_size_pct': 0.03,      # Position size as percentage of portfolio
            'max_loss_pct': 0.02,           # Maximum loss percentage per trade
            'trailing_stop_enabled': True,  # Use trailing stops
            'trailing_stop_activation': 0.02, # Profit required to activate trailing stop (2%)
            'trailing_stop_distance': 0.015,  # Trailing stop distance (1.5%)
            
            # Trading mode
            'trade_mode': 'combined',       # Use both trend-following and contrarian signals
            'max_sentiment_trades_per_day': 3,  # Daily trade limit
            'require_volume_confirmation': True,
            'volume_confirmation_threshold': 1.5,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating balanced sentiment strategy for {session.symbol}")
        
        return NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters,
            event_bus=event_bus
        )
    
    @classmethod
    def create_news_focused_strategy(cls,
                                   session: StocksSession,
                                   data_pipeline: DataPipeline,
                                   event_bus: Optional[EventBus] = None,
                                   **kwargs) -> NewsSentimentStrategy:
        """
        Create a news-focused sentiment strategy that prioritizes
        traditional news sources over social media and analyst reports.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured NewsSentimentStrategy instance
        """
        parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.7, 'social_media': 0.1, 'analyst_reports': 0.2},
            
            # Sentiment thresholds
            'bullish_threshold': 0.65,      # More conservative threshold
            'bearish_threshold': 0.35,      # More conservative threshold
            'neutral_band': 0.15,           # Wider neutral zone
            'significant_change': 0.2,      # Higher threshold for significant change
            
            # Trading parameters
            'min_sentiment_volume': 3,      # Fewer news items required
            'lookback_period_days': 2,      # Shorter lookback
            'sentiment_smoothing': 5,       # Less smoothing
            'max_hold_period_days': 3,      # Shorter holding period
            
            # Risk management
            'position_size_pct': 0.025,     # Slightly smaller position size
            'max_loss_pct': 0.015,          # Tighter stop loss
            'trailing_stop_enabled': True,
            'trailing_stop_activation': 0.015,
            'trailing_stop_distance': 0.01,
            
            # Trading mode
            'trade_mode': 'trend_following', # Focus on trend following only
            'max_sentiment_trades_per_day': 2,
            'require_volume_confirmation': True,
            'volume_confirmation_threshold': 2.0, # Higher volume requirement
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating news-focused sentiment strategy for {session.symbol}")
        
        return NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters,
            event_bus=event_bus
        )
    
    @classmethod
    def create_social_media_strategy(cls,
                                   session: StocksSession,
                                   data_pipeline: DataPipeline,
                                   event_bus: Optional[EventBus] = None,
                                   **kwargs) -> NewsSentimentStrategy:
        """
        Create a social media focused sentiment strategy that prioritizes
        social signals like Reddit, Twitter, and StockTwits.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured NewsSentimentStrategy instance
        """
        parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.2, 'social_media': 0.7, 'analyst_reports': 0.1},
            
            # Sentiment thresholds
            'bullish_threshold': 0.55,      # Lower threshold for social signals
            'bearish_threshold': 0.45,      # Higher threshold for bearish
            'neutral_band': 0.05,           # Narrower neutral zone
            'significant_change': 0.1,      # Lower threshold for change
            
            # Trading parameters
            'min_sentiment_volume': 10,     # Higher volume requirement for social
            'lookback_period_days': 1,      # Very short lookback (social moves fast)
            'sentiment_smoothing': 3,       # Minimal smoothing
            'max_hold_period_days': 2,      # Short holding period
            
            # Risk management
            'position_size_pct': 0.02,      # Smaller positions due to higher volatility
            'max_loss_pct': 0.025,          # Wider stops for social volatility
            'trailing_stop_enabled': True,
            'trailing_stop_activation': 0.01, # Quicker trailing stop activation
            'trailing_stop_distance': 0.02,  # Wider trailing stop
            
            # Trading mode
            'trade_mode': 'combined',
            'enable_contrarian_mode': True,  # Social sentiment often exhibits extremes
            'max_sentiment_trades_per_day': 4,  # More active trading
            'require_volume_confirmation': True,
            'volume_confirmation_threshold': 2.5, # Strong volume confirmation
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating social media sentiment strategy for {session.symbol}")
        
        return NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters,
            event_bus=event_bus
        )
    
    @classmethod
    def create_analyst_focused_strategy(cls,
                                      session: StocksSession,
                                      data_pipeline: DataPipeline,
                                      event_bus: Optional[EventBus] = None,
                                      **kwargs) -> NewsSentimentStrategy:
        """
        Create an analyst-focused sentiment strategy that prioritizes
        analyst ratings, upgrades/downgrades, and price targets.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured NewsSentimentStrategy instance
        """
        parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.3, 'social_media': 0.1, 'analyst_reports': 0.6},
            
            # Sentiment thresholds
            'bullish_threshold': 0.6,
            'bearish_threshold': 0.4,
            'neutral_band': 0.1,
            'significant_change': 0.15,
            
            # Trading parameters
            'min_sentiment_volume': 2,      # Even a single analyst can be significant
            'lookback_period_days': 5,      # Longer lookback for analyst ratings
            'sentiment_smoothing': 10,      # More smoothing for stability
            'max_hold_period_days': 10,     # Longer holding period
            
            # Risk management
            'position_size_pct': 0.04,      # Larger position for analyst conviction
            'max_loss_pct': 0.03,           # Wider stop for longer horizon
            'trailing_stop_enabled': True,
            'trailing_stop_activation': 0.03,
            'trailing_stop_distance': 0.02,
            
            # Trading mode
            'trade_mode': 'trend_following',  # Analyst ratings typically trend
            'max_sentiment_trades_per_day': 2,
            'require_volume_confirmation': False,  # Analyst changes may not always cause volume
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating analyst-focused sentiment strategy for {session.symbol}")
        
        return NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters,
            event_bus=event_bus
        )
    
    @classmethod
    def create_contrarian_strategy(cls,
                                 session: StocksSession,
                                 data_pipeline: DataPipeline,
                                 event_bus: Optional[EventBus] = None,
                                 **kwargs) -> NewsSentimentStrategy:
        """
        Create a contrarian sentiment strategy that trades against extreme
        sentiment readings and significant reversals.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured NewsSentimentStrategy instance
        """
        parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.3, 'social_media': 0.5, 'analyst_reports': 0.2},
            
            # Sentiment thresholds
            'bullish_threshold': 0.7,       # Higher threshold for extreme sentiment
            'bearish_threshold': 0.3,       # Lower threshold for extreme sentiment
            'neutral_band': 0.2,            # Wider neutral zone to ignore
            'significant_change': 0.2,      # Higher threshold for change
            
            # Contrarian settings
            'enable_contrarian_mode': True,
            'contrarian_threshold': 0.75,   # Very high threshold for contrarian
            'contrarian_lookback_days': 7,  # Look for sustained extremes
            
            # Trading parameters
            'min_sentiment_volume': 8,      # Require more signals for contrarian
            'lookback_period_days': 5,      # Longer lookback
            'sentiment_smoothing': 3,       # Less smoothing to catch extremes
            'max_hold_period_days': 7,      # Moderate holding period
            
            # Risk management
            'position_size_pct': 0.02,      # Smaller positions for contrarian trades
            'max_loss_pct': 0.03,           # Wider stops for contrarian
            'trailing_stop_enabled': True,
            'trailing_stop_activation': 0.025,
            'trailing_stop_distance': 0.02,
            
            # Trading mode
            'trade_mode': 'contrarian',     # Only trade contrarian signals
            'max_sentiment_trades_per_day': 2,
            'require_volume_confirmation': True,
            'volume_confirmation_threshold': 2.0,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating contrarian sentiment strategy for {session.symbol}")
        
        return NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters,
            event_bus=event_bus
        )
    
    @classmethod
    def create_earnings_enhanced_strategy(cls,
                                        session: StocksSession,
                                        data_pipeline: DataPipeline,
                                        event_bus: Optional[EventBus] = None,
                                        **kwargs) -> NewsSentimentStrategy:
        """
        Create a sentiment strategy optimized for earnings announcements
        and corporate events.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured NewsSentimentStrategy instance
        """
        parameters = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.5, 'social_media': 0.2, 'analyst_reports': 0.3},
            
            # Sentiment thresholds
            'bullish_threshold': 0.6,
            'bearish_threshold': 0.4,
            'neutral_band': 0.1,
            'significant_change': 0.2,      # Higher threshold for earnings moves
            
            # Trading parameters
            'min_sentiment_volume': 6,      # Higher volume around earnings
            'lookback_period_days': 2,      # Shorter lookback for earnings
            'sentiment_smoothing': 3,       # Less smoothing for earnings moves
            'max_hold_period_days': 3,      # Shorter holding for earnings
            
            # Risk management
            'position_size_pct': 0.02,      # Smaller positions for earnings volatility
            'max_loss_pct': 0.04,           # Wider stops for earnings volatility
            'trailing_stop_enabled': True,
            'trailing_stop_activation': 0.03,
            'trailing_stop_distance': 0.025,
            
            # Trading mode
            'trade_mode': 'combined',
            'max_sentiment_trades_per_day': 3,
            'require_volume_confirmation': True,
            'volume_confirmation_threshold': 3.0,  # Higher volume for earnings
            
            # Special settings
            'react_to_earnings': True,
            'react_to_guidance': True,
            'prioritize_earnings_sentiment': True,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating earnings-enhanced sentiment strategy for {session.symbol}")
        
        return NewsSentimentStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters,
            event_bus=event_bus
        )
    
    @classmethod
    def create_from_config(cls,
                         session: StocksSession,
                         data_pipeline: DataPipeline,
                         config: Dict[str, Any],
                         event_bus: Optional[EventBus] = None) -> NewsSentimentStrategy:
        """
        Create a sentiment strategy from a configuration dictionary.
        
        Args:
            session: Stock session with symbol and timeframe
            data_pipeline: Data pipeline for market data
            config: Strategy configuration dictionary
            event_bus: Optional event bus for strategy events
            
        Returns:
            Configured NewsSentimentStrategy instance
        """
        strategy_type = config.pop('strategy_type', 'balanced')
        
        # Extract the common kwargs
        kwargs = config
        
        # Create the appropriate strategy type
        if strategy_type == 'news_focused':
            return cls.create_news_focused_strategy(session, data_pipeline, event_bus, **kwargs)
        elif strategy_type == 'social_media':
            return cls.create_social_media_strategy(session, data_pipeline, event_bus, **kwargs)
        elif strategy_type == 'analyst_focused':
            return cls.create_analyst_focused_strategy(session, data_pipeline, event_bus, **kwargs)
        elif strategy_type == 'contrarian':
            return cls.create_contrarian_strategy(session, data_pipeline, event_bus, **kwargs)
        elif strategy_type == 'earnings_enhanced':
            return cls.create_earnings_enhanced_strategy(session, data_pipeline, event_bus, **kwargs)
        else:
            # Default to balanced
            return cls.create_balanced_sentiment_strategy(session, data_pipeline, event_bus, **kwargs)
