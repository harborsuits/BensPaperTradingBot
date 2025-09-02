#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Registry

This module provides a registry for trading strategies using a decorator pattern.
Strategies register themselves with metadata, allowing for discovery and instantiation.
"""

import logging
from typing import Dict, Any, List, Type, Callable, Optional
import inspect
import functools

# Configure logging
logger = logging.getLogger(__name__)

# Global registry of strategies
_STRATEGY_REGISTRY = {}

def register_strategy(
    name: str = None,
    market_type: str = None,
    description: str = None,
    timeframes: List[str] = None,
    parameters: Dict[str, Dict[str, Any]] = None
) -> Callable:
    """
    Decorator for registering a strategy class in the global registry.
    
    Args:
        name: Name of the strategy (defaults to class name)
        market_type: Type of market (forex, stocks, crypto, options)
        description: Description of the strategy
        timeframes: List of supported timeframes
        parameters: Dictionary of parameters with metadata
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Use class name if name not provided
        strategy_name = name or cls.__name__
        
        # Get the asset class from the module path if not provided
        if not market_type:
            module_path = cls.__module__.split('.')
            if len(module_path) >= 3 and 'strategies_new' in module_path:
                idx = module_path.index('strategies_new')
                if idx + 1 < len(module_path):
                    derived_market_type = module_path[idx + 1]
                else:
                    derived_market_type = "unknown"
            else:
                derived_market_type = "unknown"
        else:
            derived_market_type = market_type
        
        # Get strategy type from module path if possible
        strategy_type = "unknown"
        module_path = cls.__module__.split('.')
        if len(module_path) >= 4 and 'strategies_new' in module_path:
            idx = module_path.index('strategies_new')
            if idx + 2 < len(module_path):
                strategy_type = module_path[idx + 2]
        
        # Extract docstring if description not provided
        strategy_description = description or inspect.getdoc(cls) or "No description available"
        
        # Register the strategy
        _STRATEGY_REGISTRY[strategy_name] = {
            "class": cls,
            "market_type": derived_market_type,
            "strategy_type": strategy_type,
            "description": strategy_description,
            "timeframes": timeframes or [],
            "parameters": parameters or {},
            "module": cls.__module__
        }
        
        logger.info(f"Registered strategy: {strategy_name} ({derived_market_type}/{strategy_type})")
        
        return cls
    
    return decorator

def get_registered_strategies() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered strategies.
    
    Returns:
        Dictionary of registered strategies with their metadata
    """
    return _STRATEGY_REGISTRY

def get_strategy_class(strategy_name: str) -> Optional[Type]:
    """
    Get a strategy class by name.
    
    Args:
        strategy_name: Name of the strategy to retrieve
        
    Returns:
        Strategy class or None if not found
    """
    if strategy_name in _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY[strategy_name]["class"]
    return None

def get_strategies_by_market_type(market_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get strategies for a specific market type.
    
    Args:
        market_type: Market type to filter by
        
    Returns:
        Dictionary of strategies for the specified market type
    """
    return {
        name: info for name, info in _STRATEGY_REGISTRY.items() 
        if info["market_type"].lower() == market_type.lower()
    }

def get_strategies_by_strategy_type(strategy_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get strategies of a specific type.
    
    Args:
        strategy_type: Strategy type to filter by (trend, range, etc.)
        
    Returns:
        Dictionary of strategies of the specified type
    """
    return {
        name: info for name, info in _STRATEGY_REGISTRY.items() 
        if info["strategy_type"].lower() == strategy_type.lower()
    }

def get_strategies_by_timeframe(timeframe: str) -> Dict[str, Dict[str, Any]]:
    """
    Get strategies that support a specific timeframe.
    
    Args:
        timeframe: Timeframe to filter by
        
    Returns:
        Dictionary of strategies supporting the specified timeframe
    """
    return {
        name: info for name, info in _STRATEGY_REGISTRY.items() 
        if timeframe in info["timeframes"]
    }

def register_strategies_from_modules(modules: List[str]) -> None:
    """
    Import strategies from specified modules to trigger registration.
    
    Args:
        modules: List of module paths to import
    """
    for module_path in modules:
        try:
            __import__(module_path)
            logger.info(f"Imported strategies from {module_path}")
        except ImportError as e:
            logger.error(f"Failed to import strategies from {module_path}: {e}")

def clear_registry() -> None:
    """Clear the strategy registry (mainly for testing)."""
    _STRATEGY_REGISTRY.clear()
    logger.warning("Strategy registry cleared")

def get_strategy_info(strategy_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed info for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary of strategy metadata or None if not found
    """
    return _STRATEGY_REGISTRY.get(strategy_name)
