#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short Strategy Factory

Factory class for creating and configuring short selling trading strategies
with appropriate parameters and session settings.
"""

import logging
from typing import Dict, Any, Optional, List

from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksSession
from trading_bot.strategies_new.stocks.short.short_selling_strategy import ShortSellingStrategy

# Configure logging
logger = logging.getLogger(__name__)

class ShortStrategyFactory:
    """
    Factory for creating and configuring short selling strategies.
    
    This factory creates properly configured short selling strategy instances with:
    - Appropriate market parameters
    - Short selling specific configuration
    - Risk management tailored to short positions
    - Session settings
    """
    
    @staticmethod
    def create_short_selling_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        strategy_mode: str = 'technical',
        custom_parameters: Dict[str, Any] = None
    ) -> ShortSellingStrategy:
        """
        Create a fully configured Short Selling Strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            strategy_mode: 'technical', 'fundamental', or 'combined'
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured ShortSellingStrategy instance
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
            'rsi_overbought': 70,
            'min_down_days': 2,
            'respect_trend': True,
            'require_bearish_pattern': True
        }
        
        # Merge with custom parameters if provided
        parameters = base_parameters.copy()
        if custom_parameters:
            parameters.update(custom_parameters)
        
        # Create strategy
        strategy = ShortSellingStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters
        )
        
        logger.info(f"Created Short Selling Strategy for {symbol} with mode={strategy_mode}")
        return strategy
    
    @staticmethod
    def create_technical_short_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        custom_parameters: Dict[str, Any] = None
    ) -> ShortSellingStrategy:
        """
        Create a short selling strategy based on technical analysis.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured technical ShortSellingStrategy instance
        """
        base_params = {
            'strategy_mode': 'technical',
            'rsi_overbought': 70,
            'require_bearish_pattern': True,
            'respect_trend': True,
            'min_down_days': 2
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return ShortStrategyFactory.create_short_selling_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='technical',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_fundamental_short_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        custom_parameters: Dict[str, Any] = None
    ) -> ShortSellingStrategy:
        """
        Create a short selling strategy based on fundamental analysis.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured fundamental ShortSellingStrategy instance
        """
        base_params = {
            'strategy_mode': 'fundamental',
            'pe_ratio_threshold': 50,
            'debt_to_equity_threshold': 2.0,
            'negative_earnings_growth': True,
            'require_catalyst': True
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return ShortStrategyFactory.create_short_selling_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='fundamental',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_combined_short_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        custom_parameters: Dict[str, Any] = None
    ) -> ShortSellingStrategy:
        """
        Create a short selling strategy combining technical and fundamental analysis.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured combined ShortSellingStrategy instance
        """
        base_params = {
            'strategy_mode': 'combined',
            'rsi_overbought': 65,  # Slightly lower threshold when combined with fundamentals
            'pe_ratio_threshold': 40,
            'debt_to_equity_threshold': 1.8,
            'negative_earnings_growth': True,
            'respect_trend': True
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return ShortStrategyFactory.create_short_selling_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='combined',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_conservative_short_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        custom_parameters: Dict[str, Any] = None
    ) -> ShortSellingStrategy:
        """
        Create a more conservative short selling strategy with reduced risk parameters.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured conservative ShortSellingStrategy instance
        """
        base_params = {
            'strategy_mode': 'technical',
            'max_risk_per_trade_percent': 0.75,  # Lower risk per trade
            'position_size_factor': 0.7,         # Smaller positions
            'tight_stop_percent': 2.0,           # Tighter stops
            'max_loss_percent': 5.0,             # Lower max loss threshold
            'trailing_stop_activation': 3.0,      # Earlier trailing stop activation
            'trailing_stop_percent': 2.0,         # Tighter trailing stop
            'enable_short_squeeze_protection': True,
            'max_short_interest': 15,            # More conservative short interest threshold
            'min_float': 50000000                # Require larger float stocks
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return ShortStrategyFactory.create_short_selling_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='technical',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_aggressive_short_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        custom_parameters: Dict[str, Any] = None
    ) -> ShortSellingStrategy:
        """
        Create a more aggressive short selling strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured aggressive ShortSellingStrategy instance
        """
        base_params = {
            'strategy_mode': 'combined',
            'max_risk_per_trade_percent': 1.5,   # Higher risk per trade
            'position_size_factor': 1.0,         # Larger positions
            'tight_stop_percent': 5.0,           # Wider stops
            'profit_target_percent': 25.0,       # Higher profit target
            'rsi_overbought': 60,                # Lower threshold for entry
            'min_down_days': 1,                  # Only require 1 down day
            'bear_market_bias': 2.0              # Stronger bias in bear markets
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return ShortStrategyFactory.create_short_selling_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='combined',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_short_strategy_portfolio(
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.DAY_1,
        strategy_mode: str = 'technical',
        risk_level: str = 'balanced',
        custom_parameters: Dict[str, Any] = None
    ) -> Dict[str, ShortSellingStrategy]:
        """
        Create a portfolio of short selling strategies for multiple symbols.
        
        Args:
            symbols: List of stock symbols to trade
            timeframe: Trading timeframe
            strategy_mode: 'technical', 'fundamental', or 'combined'
            risk_level: 'conservative', 'balanced', or 'aggressive'
            custom_parameters: Any additional custom parameters
            
        Returns:
            Dictionary of symbol -> strategy
        """
        portfolio = {}
        
        # Create the appropriate strategy type for each symbol based on risk level
        for symbol in symbols:
            if risk_level == 'conservative':
                portfolio[symbol] = ShortStrategyFactory.create_conservative_short_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
            elif risk_level == 'aggressive':
                portfolio[symbol] = ShortStrategyFactory.create_aggressive_short_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
            else:  # balanced
                if strategy_mode == 'technical':
                    portfolio[symbol] = ShortStrategyFactory.create_technical_short_strategy(
                        symbol=symbol,
                        timeframe=timeframe,
                        custom_parameters=custom_parameters
                    )
                elif strategy_mode == 'fundamental':
                    portfolio[symbol] = ShortStrategyFactory.create_fundamental_short_strategy(
                        symbol=symbol,
                        timeframe=timeframe,
                        custom_parameters=custom_parameters
                    )
                else:  # combined
                    portfolio[symbol] = ShortStrategyFactory.create_combined_short_strategy(
                        symbol=symbol,
                        timeframe=timeframe,
                        custom_parameters=custom_parameters
                    )
        
        logger.info(f"Created short strategy portfolio with {len(symbols)} symbols using {strategy_mode} mode " +
                   f"and {risk_level} risk level")
        return portfolio
