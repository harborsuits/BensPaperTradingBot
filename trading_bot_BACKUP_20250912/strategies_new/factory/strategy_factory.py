#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Factory

This module provides a factory for creating strategy instances from the registry.
It handles parameter validation, dependency injection, and instance creation.
"""

import logging
import importlib
from typing import Dict, Any, List, Type, Optional, Union
import inspect

from trading_bot.core.session import Session
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import (
    get_registered_strategies,
    get_strategy_class,
    register_strategies_from_modules
)

# Configure logging
logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory for creating strategy instances.
    
    This factory uses the strategy registry to discover and instantiate strategies.
    It handles parameter validation, dependency injection, and caching.
    """
    
    def __init__(self, data_pipeline_provider: Optional[Any] = None):
        """
        Initialize the strategy factory.
        
        Args:
            data_pipeline_provider: Optional provider for data pipelines
        """
        self.data_pipeline_provider = data_pipeline_provider
        self._strategy_cache = {}
        self._discover_strategies()
    
    def _discover_strategies(self) -> None:
        """Discover and register strategies from known modules."""
        # Core strategy modules to scan for strategies
        modules = [
            'trading_bot.strategies_new.forex.trend',
            'trading_bot.strategies_new.forex.range',
            'trading_bot.strategies_new.forex.breakout',
            'trading_bot.strategies_new.forex.momentum',
            'trading_bot.strategies_new.forex.scalping',
            'trading_bot.strategies_new.forex.swing',
            'trading_bot.strategies_new.stocks.trend',
            'trading_bot.strategies_new.stocks.momentum',
            'trading_bot.strategies_new.crypto.trend',
            'trading_bot.strategies_new.crypto.momentum',
            'trading_bot.strategies_new.options.trend',
            # Newly implemented strategies
            'trading_bot.strategies_new.stocks.gap',
            'trading_bot.strategies_new.stocks.event',
            'trading_bot.strategies_new.stocks.volume',
            'trading_bot.strategies_new.stocks.short',
            'trading_bot.strategies_new.stocks.sector',
            # Crypto strategies
            'trading_bot.strategies_new.crypto.hodl',
            'trading_bot.strategies_new.crypto.portfolio_balancing',
            'trading_bot.strategies_new.crypto.support_resistance',
            # Options strategies
            'trading_bot.strategies_new.options.vertical_spreads',
            'trading_bot.strategies_new.options.complex_spreads',
            'trading_bot.strategies_new.options.time_spreads',
            'trading_bot.strategies_new.options.volatility_spreads',
            'trading_bot.strategies_new.options.advanced_spreads',
        ]
        
        # Register strategies from these modules
        register_strategies_from_modules(modules)
        
        # Log discovered strategies
        strategies = get_registered_strategies()
        logger.info(f"Discovered {len(strategies)} strategies")
        
        # Group strategies by market type for logging
        market_types = {}
        for name, info in strategies.items():
            market_type = info.get('market_type', 'unknown')
            if market_type not in market_types:
                market_types[market_type] = []
            market_types[market_type].append(name)
        
        # Log strategies by market type
        for market_type, strats in market_types.items():
            logger.info(f"  {market_type.capitalize()} strategies: {len(strats)}")
            for strat in strats:
                logger.debug(f"    - {strat}")
    
    def create_strategy(
        self, 
        strategy_name: str, 
        session: Session, 
        parameters: Dict[str, Any] = None,
        data_pipeline: Optional[DataPipeline] = None
    ) -> Any:
        """
        Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy to create
            session: Session object with symbol, timeframe, etc.
            parameters: Strategy parameters (optional)
            data_pipeline: Data pipeline instance (optional)
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found or required parameters missing
        """
        # Get strategy class
        strategy_class = get_strategy_class(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        
        # Get strategy metadata
        all_strategies = get_registered_strategies()
        strategy_info = all_strategies.get(strategy_name, {})
        
        # Check if strategy supports this timeframe
        timeframes = strategy_info.get('timeframes', [])
        if timeframes and session.timeframe not in timeframes:
            logger.warning(
                f"Strategy '{strategy_name}' may not be optimized for timeframe {session.timeframe}. "
                f"Supported timeframes: {', '.join(timeframes)}"
            )
        
        # Validate parameters
        if parameters:
            self._validate_parameters(parameters, strategy_info.get('parameters', {}))
        
        # Get data pipeline
        pipeline = data_pipeline
        if pipeline is None and self.data_pipeline_provider:
            pipeline = self.data_pipeline_provider.get_pipeline(session.symbol, session.timeframe)
        
        if pipeline is None:
            # Create a default pipeline if none provided
            from trading_bot.data.data_pipeline import create_data_pipeline
            pipeline = create_data_pipeline(
                config={'default_quality_threshold': 0.7, 'apply_all_cleaning': True},
                event_bus=None  # This should be provided in a real application
            )
            logger.warning(f"Created default data pipeline for strategy '{strategy_name}'")
        
        # Create the strategy instance
        try:
            strategy = strategy_class(session, pipeline, parameters)
            logger.info(f"Created strategy instance: {strategy_name}")
            return strategy
        except Exception as e:
            logger.error(f"Failed to create strategy '{strategy_name}': {str(e)}")
            raise
    
    def _validate_parameters(self, parameters: Dict[str, Any], parameter_specs: Dict[str, Dict[str, Any]]) -> None:
        """
        Validate strategy parameters against specifications.
        
        Args:
            parameters: Parameters to validate
            parameter_specs: Parameter specifications from registry
            
        Raises:
            ValueError: If parameters are invalid
        """
        for param_name, param_value in parameters.items():
            if param_name not in parameter_specs:
                logger.warning(f"Unknown parameter: {param_name}")
                continue
                
            spec = parameter_specs[param_name]
            param_type = spec.get('type')
            
            # Type checking
            if param_type == 'int' and not isinstance(param_value, int):
                raise ValueError(f"Parameter '{param_name}' must be an integer")
            elif param_type == 'float' and not isinstance(param_value, (int, float)):
                raise ValueError(f"Parameter '{param_name}' must be a number")
            elif param_type == 'bool' and not isinstance(param_value, bool):
                raise ValueError(f"Parameter '{param_name}' must be a boolean")
            elif param_type == 'str' and not isinstance(param_value, str):
                raise ValueError(f"Parameter '{param_name}' must be a string")
            
            # Range checking
            if 'min' in spec and param_value < spec['min']:
                logger.warning(
                    f"Parameter '{param_name}' value {param_value} is below minimum {spec['min']}"
                )
            if 'max' in spec and param_value > spec['max']:
                logger.warning(
                    f"Parameter '{param_name}' value {param_value} is above maximum {spec['max']}"
                )
            
            # Enum checking
            if 'enum' in spec and param_value not in spec['enum']:
                raise ValueError(
                    f"Parameter '{param_name}' must be one of: {', '.join(spec['enum'])}"
                )
    
    def get_strategy_by_market_condition(
        self,
        market_type: str,
        symbol: str,
        timeframe: str,
        market_regime: str,
        session_provider: Any
    ) -> Optional[Any]:
        """
        Find the best strategy for the current market conditions.
        
        This method evaluates all applicable strategies for the given market type,
        symbol, and timeframe, then selects the one with the highest compatibility
        score for the current market regime.
        
        Args:
            market_type: Type of market (forex, stocks, etc.)
            symbol: Trading symbol
            timeframe: Trading timeframe
            market_regime: Current market regime (trending, ranging, etc.)
            session_provider: Provider for session objects
            
        Returns:
            The best strategy instance for the current conditions, or None if no suitable strategy found
        """
        # Get all strategies for this market type
        all_strategies = get_registered_strategies()
        applicable_strategies = {
            name: info for name, info in all_strategies.items()
            if info['market_type'].lower() == market_type.lower() and
               (not info['timeframes'] or timeframe in info['timeframes'])
        }
        
        if not applicable_strategies:
            logger.warning(f"No applicable strategies found for {market_type}/{symbol}/{timeframe}")
            return None
        
        # Create a session for this symbol/timeframe
        session = session_provider.create_session(symbol, timeframe)
        
        # Evaluate each strategy's compatibility with the current regime
        compatibility_scores = {}
        for strategy_name, info in applicable_strategies.items():
            try:
                # Create a temporary strategy instance to evaluate compatibility
                strategy_class = info['class']
                temp_strategy = self.create_strategy(strategy_name, session)
                
                # Get compatibility score
                score = temp_strategy.regime_compatibility(market_regime)
                compatibility_scores[strategy_name] = score
                
                logger.debug(f"Strategy '{strategy_name}' compatibility with {market_regime}: {score:.2f}")
            except Exception as e:
                logger.error(f"Error evaluating strategy '{strategy_name}': {str(e)}")
        
        if not compatibility_scores:
            logger.warning(f"No compatible strategies found for {market_type}/{symbol}/{timeframe} in {market_regime} regime")
            return None
        
        # Find the strategy with the highest compatibility score
        best_strategy_name = max(compatibility_scores, key=compatibility_scores.get)
        best_score = compatibility_scores[best_strategy_name]
        
        logger.info(f"Selected strategy '{best_strategy_name}' for {symbol}/{timeframe} in {market_regime} regime (score: {best_score:.2f})")
        
        # Create and return the best strategy instance
        return self.create_strategy(best_strategy_name, session)
    
    def get_available_strategies(
        self, 
        market_type: Optional[str] = None,
        strategy_type: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get available strategies, optionally filtered.
        
        Args:
            market_type: Filter by market type (optional)
            strategy_type: Filter by strategy type (optional)
            timeframe: Filter by timeframe support (optional)
            
        Returns:
            Dictionary of available strategies with their metadata
        """
        strategies = get_registered_strategies()
        filtered_strategies = {}
        
        for name, info in strategies.items():
            # Apply filters
            if market_type and info.get('market_type', '').lower() != market_type.lower():
                continue
            if strategy_type and info.get('strategy_type', '').lower() != strategy_type.lower():
                continue
            if timeframe and timeframe not in info.get('timeframes', []):
                continue
                
            filtered_strategies[name] = info
        
        return filtered_strategies
