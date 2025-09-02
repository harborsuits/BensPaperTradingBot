#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume Strategy Factory

Factory class for creating and configuring volume-based trading strategies
with appropriate parameters and session settings.
"""

import logging
from typing import Dict, Any, Optional, List

from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksSession
from trading_bot.strategies_new.stocks.volume.volume_surge_strategy import VolumeSurgeStrategy

# Configure logging
logger = logging.getLogger(__name__)

class VolumeStrategyFactory:
    """
    Factory for creating and configuring volume-based trading strategies.
    
    This factory creates properly configured volume strategy instances with:
    - Appropriate market parameters
    - Volume-specific configuration
    - Trade direction strategy
    - Session settings
    """
    
    @staticmethod
    def create_volume_surge_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        strategy_mode: str = 'adaptive',
        custom_parameters: Dict[str, Any] = None
    ) -> VolumeSurgeStrategy:
        """
        Create a fully configured Volume Surge Strategy.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            strategy_mode: 'breakout', 'reversal', or 'adaptive'
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured VolumeSurgeStrategy instance
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
            'volume_surge_threshold': 2.5,      # Multiple of average volume to consider a surge
            'volume_lookback_periods': 20,      # Periods to look back for volume average
            'volume_smoothing_periods': 5,      # Periods for smoothing volume
            'relative_volume_threshold': 2.0,   # Relative volume threshold for signals
        }
        
        # Merge with custom parameters if provided
        parameters = base_parameters.copy()
        if custom_parameters:
            parameters.update(custom_parameters)
        
        # Create strategy
        strategy = VolumeSurgeStrategy(
            session=session,
            data_pipeline=data_pipeline,
            parameters=parameters
        )
        
        logger.info(f"Created Volume Surge Strategy for {symbol} with mode={strategy_mode}")
        return strategy
    
    @staticmethod
    def create_breakout_volume_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        custom_parameters: Dict[str, Any] = None
    ) -> VolumeSurgeStrategy:
        """
        Create a volume strategy that trades breakouts on volume surges.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured breakout VolumeSurgeStrategy instance
        """
        base_params = {
            'strategy_mode': 'breakout',
            'require_price_confirmation': True,
            'min_price_move_percent': 1.5,      # Higher threshold for more meaningful breakouts
            'relative_volume_threshold': 2.5,   # Higher volume threshold for breakouts
            'trend_confirmation_bars': 2        # Require confirmation for trend before entry
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return VolumeStrategyFactory.create_volume_surge_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='breakout',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_reversal_volume_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        custom_parameters: Dict[str, Any] = None
    ) -> VolumeSurgeStrategy:
        """
        Create a volume strategy that trades reversals on volume climax patterns.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured reversal VolumeSurgeStrategy instance
        """
        base_params = {
            'strategy_mode': 'reversal',
            'require_price_confirmation': False,  # Price confirmation isn't required for reversals
            'volume_surge_threshold': 3.0,        # Higher for stronger reversal signals
            'volume_divergence_enabled': True,    # Look for price-volume divergences
            'trailing_stop_activation': 1.0,      # Quicker trailing stop activation
            'trailing_stop_distance': 0.8         # Tighter trailing stop for reversals
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return VolumeStrategyFactory.create_volume_surge_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='reversal',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_adaptive_volume_strategy(
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        custom_parameters: Dict[str, Any] = None
    ) -> VolumeSurgeStrategy:
        """
        Create a volume strategy that adapts to current market conditions.
        
        Args:
            symbol: Stock symbol to trade
            timeframe: Trading timeframe
            custom_parameters: Any additional custom parameters
            
        Returns:
            Configured adaptive VolumeSurgeStrategy instance
        """
        base_params = {
            'strategy_mode': 'adaptive',
            'calculate_volume_profile': True,     # Use volume profile for deeper analysis
            'volume_profile_bins': 20,            # More detailed volume profile
            'volume_divergence_enabled': True     # Enable volume divergence detection
        }
        
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return VolumeStrategyFactory.create_volume_surge_strategy(
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode='adaptive',
            custom_parameters=base_params
        )
    
    @staticmethod
    def create_volume_strategy_portfolio(
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        strategy_mode: str = 'adaptive',
        custom_parameters: Dict[str, Any] = None
    ) -> Dict[str, VolumeSurgeStrategy]:
        """
        Create a portfolio of volume-based strategies for multiple symbols.
        
        Args:
            symbols: List of stock symbols to trade
            timeframe: Trading timeframe
            strategy_mode: 'breakout', 'reversal', or 'adaptive'
            custom_parameters: Any additional custom parameters
            
        Returns:
            Dictionary of symbol -> strategy
        """
        portfolio = {}
        
        # Create volume surge strategies for each symbol
        for symbol in symbols:
            # Create strategy based on mode
            if strategy_mode == 'breakout':
                portfolio[symbol] = VolumeStrategyFactory.create_breakout_volume_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
            elif strategy_mode == 'reversal':
                portfolio[symbol] = VolumeStrategyFactory.create_reversal_volume_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
            else:  # adaptive or any other
                portfolio[symbol] = VolumeStrategyFactory.create_adaptive_volume_strategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    custom_parameters=custom_parameters
                )
        
        logger.info(f"Created volume strategy portfolio with {len(symbols)} symbols using {strategy_mode} mode")
        return portfolio
