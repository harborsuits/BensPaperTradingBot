#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Registry Module

This module provides a registry system for trading strategies, allowing dynamic
discovery and instantiation of strategies based on metadata.
"""

import logging
from typing import Dict, Type, List, Any, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Strategy types
class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    RANGE = "range"
    MOMENTUM = "momentum"
    CARRY = "carry"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"
    SWING = "swing"
    DAY_TRADING = "day_trading"
    POSITION = "position"
    ML_BASED = "ml_based"

# Asset classes
class AssetClass(Enum):
    FOREX = "forex"
    STOCKS = "stocks"
    OPTIONS = "options"
    CRYPTO = "crypto"
    MULTI_ASSET = "multi_asset"
    
# Market regimes
class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    ALL_WEATHER = "all_weather"

# Time frames
class TimeFrame(Enum):
    SCALPING = "scalping"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    MULTI_TIMEFRAME = "multi_timeframe"

class StrategyRegistry:
    """
    Registry for trading strategies.
    
    This class maintains a registry of available trading strategies, categorized by
    asset class, strategy type, and other metadata.
    """
    
    # Class variable to store all registered strategies
    _strategies: Dict[str, Type] = {}
    
    # Categorized registry
    _by_asset_class: Dict[AssetClass, Set[str]] = {asset_class: set() for asset_class in AssetClass}
    _by_strategy_type: Dict[StrategyType, Set[str]] = {strategy_type: set() for strategy_type in StrategyType}
    _by_market_regime: Dict[MarketRegime, Set[str]] = {regime: set() for regime in MarketRegime}
    _by_timeframe: Dict[TimeFrame, Set[str]] = {timeframe: set() for timeframe in TimeFrame}
    
    @classmethod
    def register(cls, strategy_class: Type, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a strategy class with the registry.
        
        Args:
            strategy_class: The strategy class to register
            metadata: Optional metadata to use instead of the class's metadata
        """
        name = strategy_class.__name__
        
        if name in cls._strategies:
            logger.warning(f"Strategy {name} already registered, overwriting")
        
        # Store the strategy class
        cls._strategies[name] = strategy_class
        
        # Extract metadata
        if metadata is None:
            metadata = getattr(strategy_class, 'METADATA', {})
        
        # Register in categorized registries
        if 'asset_class' in metadata:
            try:
                asset_class = AssetClass(metadata['asset_class'])
                cls._by_asset_class[asset_class].add(name)
            except (ValueError, KeyError):
                logger.warning(f"Invalid asset class in metadata for {name}")
        
        if 'strategy_type' in metadata:
            try:
                strategy_type = StrategyType(metadata['strategy_type'])
                cls._by_strategy_type[strategy_type].add(name)
            except (ValueError, KeyError):
                logger.warning(f"Invalid strategy type in metadata for {name}")
        
        if 'compatible_market_regimes' in metadata:
            for regime_str in metadata['compatible_market_regimes']:
                try:
                    regime = MarketRegime(regime_str)
                    cls._by_market_regime[regime].add(name)
                except (ValueError, KeyError):
                    logger.warning(f"Invalid market regime in metadata for {name}")
        
        if 'timeframe' in metadata:
            try:
                timeframe = TimeFrame(metadata['timeframe'])
                cls._by_timeframe[timeframe].add(name)
            except (ValueError, KeyError):
                logger.warning(f"Invalid timeframe in metadata for {name}")
        
        logger.info(f"Registered strategy: {name}")
    
    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type]:
        """Get a strategy class by name."""
        return cls._strategies.get(name)
    
    @classmethod
    def get_all_strategy_names(cls) -> List[str]:
        """Get all registered strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategies_by_asset_class(cls, asset_class: AssetClass) -> List[str]:
        """Get strategies for a specific asset class."""
        return list(cls._by_asset_class.get(asset_class, set()))
    
    @classmethod
    def get_strategies_by_type(cls, strategy_type: StrategyType) -> List[str]:
        """Get strategies of a specific type."""
        return list(cls._by_strategy_type.get(strategy_type, set()))
    
    @classmethod
    def get_strategies_by_market_regime(cls, regime: MarketRegime) -> List[str]:
        """Get strategies compatible with a specific market regime."""
        return list(cls._by_market_regime.get(regime, set()))
    
    @classmethod
    def get_strategies_by_timeframe(cls, timeframe: TimeFrame) -> List[str]:
        """Get strategies for a specific timeframe."""
        return list(cls._by_timeframe.get(timeframe, set()))
    
    @classmethod
    def get_strategy_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a specific strategy."""
        strategy_class = cls.get_strategy_class(name)
        if strategy_class:
            return getattr(strategy_class, 'METADATA', {})
        return {}
    
    @classmethod
    def find_strategies(cls, 
                        asset_class: Optional[AssetClass] = None,
                        strategy_type: Optional[StrategyType] = None,
                        market_regime: Optional[MarketRegime] = None,
                        timeframe: Optional[TimeFrame] = None) -> List[str]:
        """
        Find strategies matching the specified criteria.
        
        Args:
            asset_class: Optional asset class filter
            strategy_type: Optional strategy type filter
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            
        Returns:
            List of strategy names matching all specified criteria
        """
        candidates = set(cls._strategies.keys())
        
        if asset_class:
            candidates &= cls._by_asset_class.get(asset_class, set())
        
        if strategy_type:
            candidates &= cls._by_strategy_type.get(strategy_type, set())
        
        if market_regime:
            candidates &= cls._by_market_regime.get(market_regime, set())
        
        if timeframe:
            candidates &= cls._by_timeframe.get(timeframe, set())
        
        return list(candidates)


def register_strategy(metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to register a strategy class with the registry.
    
    Args:
        metadata: Optional metadata to use instead of the class's metadata
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        StrategyRegistry.register(cls, metadata)
        return cls
    return decorator
