#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap Strategy Factory

Factory class for creating and configuring Gap Trading Strategies
with appropriate parameters and session settings.
"""

import logging
from typing import Dict, Any, Optional

from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksSession
from trading_bot.strategies_new.stocks.gap.gap_trading_strategy import GapTradingStrategy
from trading_bot.strategies_new.stocks.gap.gap_trade_analyzer import GapTradeAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class GapStrategyFactory:
    """
    Factory for creating and configuring Gap Trading Strategies.
    
    This factory creates properly configured gap trading strategies with:
    - Appropriate market parameters
    - Strategy-specific configuration
    - Associated gap analyzers
    - Session settings
    """
    
    @staticmethod
    def create_gap_trading_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        strategy_mode: str = 'both',
        min_gap_percent: float = 1.5,
        custom_parameters: Dict[str, Any] = None
    ) -> GapTradingStrategy:
        """
        Create a fully configured Gap Trading Strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            strategy_mode: 'continuation', 'fade', or 'both'
            min_gap_percent: Minimum gap size to consider
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured GapTradingStrategy instance
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
        
        # Create gap analyzer
        analyzer = GapTradeAnalyzer(parameters={
            'min_gap_percent': min_gap_percent,
            'lookback_days': 60,
        })
        
        # Base parameters
        base_parameters = {
            'strategy_mode': strategy_mode,
            'min_gap_percent': min_gap_percent,
            'gap_analyzer': analyzer
        }
        
        # Merge with custom parameters if provided
        parameters = base_parameters.copy()
        if custom_parameters:
            parameters.update(custom_parameters)
        
        # Create strategy
        strategy = GapTradingStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters
        )
        
        logger.info(f"Created Gap Trading Strategy for {symbol} with mode={strategy_mode}")
        return strategy
    
    @staticmethod
    def create_gap_and_go_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        min_gap_percent: float = 1.5,
        custom_parameters: Dict[str, Any] = None
    ) -> GapTradingStrategy:
        """
        Create a Gap and Go (continuation) strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            min_gap_percent: Minimum gap size to consider
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured GapTradingStrategy instance for continuation
        """
        base_params = {
            'strategy_mode': 'continuation',
            'min_gap_percent': min_gap_percent,
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return GapStrategyFactory.create_gap_trading_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='continuation',
            min_gap_percent=min_gap_percent,
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_gap_fade_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        min_gap_percent: float = 1.5,
        custom_parameters: Dict[str, Any] = None
    ) -> GapTradingStrategy:
        """
        Create a Gap Fade (reversal) strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            min_gap_percent: Minimum gap size to consider
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured GapTradingStrategy instance for fading
        """
        base_params = {
            'strategy_mode': 'fade',
            'min_gap_percent': min_gap_percent,
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return GapStrategyFactory.create_gap_trading_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='fade',
            min_gap_percent=min_gap_percent,
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_adaptive_gap_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        min_gap_percent: float = 1.5,
        historical_data: Dict[str, Any] = None,
        custom_parameters: Dict[str, Any] = None
    ) -> GapTradingStrategy:
        """
        Create a gap strategy that adapts its approach based on historical performance.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            min_gap_percent: Minimum gap size to consider
            historical_data: Historical performance data
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured adaptive GapTradingStrategy instance
        """
        # Create analyzer
        analyzer = GapTradeAnalyzer(parameters={
            'min_gap_percent': min_gap_percent,
            'lookback_days': 60,
        })
        
        # If we have historical data, analyze it to determine best approach
        continuation_preference = 0.5  # Default balanced
        
        if historical_data:
            # Update analyzer with historical data
            for trade in historical_data.get('trades', []):
                analyzer.update_trade_performance(trade)
            
            # Get recommendations
            recommendations = analyzer.analyze_best_gap_trading_approach(symbol)
            strategy_mode = recommendations.get('recommended_approach', 'both')
            continuation_preference = recommendations.get('continuation_preference', 0.5)
        else:
            strategy_mode = 'both'
        
        # Base parameters
        base_params = {
            'strategy_mode': strategy_mode,
            'continuation_preference': continuation_preference,
            'min_gap_percent': min_gap_percent,
            'gap_analyzer': analyzer
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return GapStrategyFactory.create_gap_trading_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode=strategy_mode,
            min_gap_percent=min_gap_percent,
            custom_parameters=base_params
        )
