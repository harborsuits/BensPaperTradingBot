#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Earnings Strategy Factory

Factory class for creating and configuring earnings-related trading strategies
with appropriate parameters and session settings.
"""

import logging
from typing import Dict, Any, Optional, List

from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksSession
from trading_bot.strategies_new.stocks.event.earnings_announcement_strategy import EarningsAnnouncementStrategy

# Configure logging
logger = logging.getLogger(__name__)

class EarningsStrategyFactory:
    """
    Factory for creating and configuring earnings-based trading strategies.
    
    This factory creates properly configured earnings strategy instances with:
    - Appropriate market parameters
    - Event-specific configuration
    - Historical performance analysis
    - Session settings
    """
    
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
    def create_pre_earnings_momentum_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        momentum_threshold: float = 5.0,
        custom_parameters: Dict[str, Any] = None
    ) -> EarningsAnnouncementStrategy:
        """
        Create a strategy focused on pre-earnings momentum trades only.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            momentum_threshold: Minimum price momentum required for entry
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured EarningsAnnouncementStrategy instance for pre-earnings momentum
        """
        base_params = {
            'strategy_mode': 'adaptive',
            'pre_earnings_entry': True,
            'post_earnings_entry': False,
            'hold_through_earnings': False,
            'momentum_threshold': momentum_threshold
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return EarningsStrategyFactory.create_earnings_announcement_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='adaptive',
            pre_earnings_entry=True,
            post_earnings_entry=False,
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_post_earnings_reaction_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        surprise_threshold: float = 10.0,
        custom_parameters: Dict[str, Any] = None
    ) -> EarningsAnnouncementStrategy:
        """
        Create a strategy focused on post-earnings reactions only.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            surprise_threshold: Minimum earnings surprise percentage required
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured EarningsAnnouncementStrategy instance for post-earnings reactions
        """
        base_params = {
            'strategy_mode': 'adaptive',
            'pre_earnings_entry': False,
            'post_earnings_entry': True,
            'hold_through_earnings': False,
            'surprise_threshold_percent': surprise_threshold,
            'post_earnings_size_percent': 100  # Full position size
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return EarningsStrategyFactory.create_earnings_announcement_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='adaptive',
            pre_earnings_entry=False,
            post_earnings_entry=True,
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_earnings_portfolio(
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        strategy_mode: str = 'adaptive',
        custom_parameters: Dict[str, Any] = None
    ) -> Dict[str, EarningsAnnouncementStrategy]:
        """
        Create a portfolio of earnings strategies for multiple symbols.
        
        Args:
            symbols: List of stock symbols to trade
            timeframe: Trading timeframe
            strategy_mode: 'bullish', 'bearish', or 'adaptive'
            custom_parameters: Any additional custom parameters
            
        Returns:
            Dictionary of symbol -> EarningsAnnouncementStrategy instances
        """
        strategies = {}
        
        for symbol in symbols:
            strategies[symbol] = EarningsStrategyFactory.create_earnings_announcement_strategy(
                symbol=symbol,
                timeframe=timeframe,
                strategy_mode=strategy_mode,
                custom_parameters=custom_parameters
            )
            
        logger.info(f"Created earnings strategy portfolio with {len(symbols)} symbols")
        return strategies
