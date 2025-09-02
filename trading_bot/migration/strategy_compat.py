#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Compatibility Layer

This module provides compatibility functions for working with strategies
during the migration to the new organization structure.
"""

import logging
import warnings
import importlib
from typing import Dict, Any, Optional, Type, List, Union

logger = logging.getLogger(__name__)

def get_strategy(strategy_name: str) -> Optional[Type]:
    """
    Get a strategy class by name, trying both new and old organization.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy class or None if not found
    """
    # First try the new organization
    try:
        # Try forex strategies first (building on our successful ForexTrendFollowingStrategy)
        if strategy_name.startswith("Forex"):
            subtype = _determine_forex_subtype(strategy_name)
            module_path = f"trading_bot.strategies.forex.{subtype}.{_snake_case(strategy_name)}"
            module = importlib.import_module(module_path)
            return getattr(module, strategy_name)
        
        # Try other asset classes
        asset_classes = ["stocks", "crypto", "options", "multi_asset"]
        for asset_class in asset_classes:
            try:
                module_path = f"trading_bot.strategies.{asset_class}"
                module = importlib.import_module(module_path)
                if hasattr(module, strategy_name):
                    return getattr(module, strategy_name)
            except (ImportError, AttributeError):
                continue
    except (ImportError, AttributeError) as e:
        pass
    
    # Fall back to the old organization
    try:
        # Try direct import first
        module_path = f"trading_bot.strategies.{_snake_case(strategy_name)}"
        try:
            module = importlib.import_module(module_path)
            return getattr(module, strategy_name)
        except (ImportError, AttributeError):
            pass
        
        # Try various directories
        directories = ["forex", "stocks", "crypto", "options"]
        for directory in directories:
            try:
                module_path = f"trading_bot.strategies.{directory}.{_snake_case(strategy_name)}"
                module = importlib.import_module(module_path)
                return getattr(module, strategy_name)
            except (ImportError, AttributeError):
                continue
        
        # If still not found, try more variations
        snake_case_name = _snake_case(strategy_name)
        for suffix in ["_strategy", ""]:
            try:
                module_path = f"trading_bot.strategies.{snake_case_name}{suffix}"
                module = importlib.import_module(module_path)
                return getattr(module, strategy_name)
            except (ImportError, AttributeError):
                continue
                
    except Exception as e:
        logger.error(f"Error finding strategy {strategy_name}: {str(e)}")
    
    logger.warning(f"Strategy {strategy_name} not found in either organization")
    return None

def create_strategy(strategy_name: str, 
                   parameters: Optional[Dict[str, Any]] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a strategy instance, using the factory if available,
    otherwise falling back to direct instantiation.
    
    Args:
        strategy_name: Name of the strategy
        parameters: Strategy parameters
        metadata: Strategy metadata
        
    Returns:
        Strategy instance or None if not found
    """
    # First try using the new factory
    try:
        from trading_bot.strategies.factory.strategy_factory import StrategyFactory
        factory = StrategyFactory()
        return factory.create_strategy(strategy_name, parameters, metadata)
    except ImportError:
        # Factory not available, try direct instantiation
        strategy_class = get_strategy(strategy_name)
        if strategy_class:
            return strategy_class(
                name=strategy_name,
                parameters=parameters or {},
                metadata=metadata or {}
            )
        return None

def _determine_forex_subtype(strategy_name: str) -> str:
    """
    Determine the forex strategy subtype based on the name.
    
    Args:
        strategy_name: Strategy name
        
    Returns:
        Subtype directory
    """
    # Use our successful ForexTrendFollowingStrategy as a guide
    if "Trend" in strategy_name:
        return "trend"
    elif "Range" in strategy_name or "Oscillator" in strategy_name:
        return "range"
    elif "Breakout" in strategy_name:
        return "breakout"
    elif "Carry" in strategy_name:
        return "carry"
    elif "Momentum" in strategy_name:
        return "momentum"
    elif "Scalping" in strategy_name:
        return "scalping"
    elif "Swing" in strategy_name:
        return "swing"
    # Default
    return "trend"

def _snake_case(name: str) -> str:
    """Convert PascalCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def list_available_strategies() -> List[str]:
    """
    List all available strategies from both organizations.
    
    Returns:
        List of strategy names
    """
    strategies = set()
    
    # Try from new organization
    try:
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
        strategies.update(StrategyRegistry.get_all_strategy_names())
    except ImportError:
        pass
    
    # Add from old organization (simplified approach)
    # In a real implementation, this would scan the directory structure
    
    return sorted(list(strategies))
