#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processor Migration Utilities

This module provides utilities to help migrate from the old data processing structure
to the new architecture with clear separation between cleaning and quality assurance.
"""

import logging
import pandas as pd
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps

from trading_bot.data.data_pipeline import create_data_pipeline, DataPipeline
from trading_bot.core.event_system import EventBus

# Configure logging
logger = logging.getLogger("ProcessorMigration")

# Global event bus and pipeline instances for compatibility
_global_event_bus = None
_global_pipeline = None

def initialize_compatibility_layer(event_bus: Optional[EventBus] = None,
                                  config: Optional[Dict[str, Any]] = None) -> DataPipeline:
    """
    Initialize the compatibility layer for data processing.
    
    Args:
        event_bus: Optional event bus
        config: Optional configuration dictionary
        
    Returns:
        Configured DataPipeline
    """
    global _global_event_bus, _global_pipeline
    
    if event_bus:
        _global_event_bus = event_bus
    else:
        # Create a new event bus if none provided
        from trading_bot.core.event_system import EventBus
        _global_event_bus = EventBus()
        logger.info("Created new event bus for compatibility layer")
    
    # Create the global pipeline
    _global_pipeline = create_data_pipeline(
        config=config,
        event_bus=_global_event_bus
    )
    
    logger.info("Initialized data processor compatibility layer")
    return _global_pipeline

def get_pipeline() -> DataPipeline:
    """
    Get the global data pipeline, creating it if necessary.
    
    Returns:
        Global DataPipeline instance
    """
    global _global_pipeline
    
    if _global_pipeline is None:
        logger.warning("Data pipeline not initialized, creating with defaults")
        initialize_compatibility_layer()
    
    return _global_pipeline

def clean_data(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
              symbol: Optional[str] = None,
              source: Optional[str] = None) -> pd.DataFrame:
    """
    Compatibility function for data cleaning.
    
    This function provides backward compatibility for code that
    directly used the old DataCleaningProcessor.
    
    Args:
        data: Input DataFrame or dictionary of DataFrames
        symbol: Optional symbol identifier
        source: Optional source identifier
        
    Returns:
        Cleaned DataFrame
    """
    # Show deprecation warning
    warnings.warn(
        "clean_data is deprecated, use DataPipeline.process instead",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Get the global pipeline
    pipeline = get_pipeline()
    
    # Configure to skip quality checks
    original_skip_quality = pipeline.skip_quality
    pipeline.skip_quality = True
    
    try:
        # Process data through cleaning only
        result, _ = pipeline.process(data, symbol, source)
        return result
    finally:
        # Restore original settings
        pipeline.skip_quality = original_skip_quality

def validate_data_quality(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                         symbol: Optional[str] = None,
                         source: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compatibility function for data quality validation.
    
    This function provides backward compatibility for code that
    directly used the old quality validation functions.
    
    Args:
        data: Input DataFrame or dictionary of DataFrames
        symbol: Optional symbol identifier
        source: Optional source identifier
        
    Returns:
        Tuple of (validated_data, quality_metrics)
    """
    # Show deprecation warning
    warnings.warn(
        "validate_data_quality is deprecated, use DataPipeline.process instead",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Get the global pipeline
    pipeline = get_pipeline()
    
    # Configure to skip cleaning
    original_skip_cleaning = pipeline.skip_cleaning
    pipeline.skip_cleaning = True
    
    try:
        # Process data through quality only
        result, metadata = pipeline.process(data, symbol, source)
        return result, metadata.get('quality_stats', {})
    finally:
        # Restore original settings
        pipeline.skip_cleaning = original_skip_cleaning

def process_data(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                symbol: Optional[str] = None,
                source: Optional[str] = None) -> pd.DataFrame:
    """
    Unified function for data processing.
    
    This function provides a simplified interface to the new data pipeline.
    
    Args:
        data: Input DataFrame or dictionary of DataFrames
        symbol: Optional symbol identifier
        source: Optional source identifier
        
    Returns:
        Processed DataFrame
    """
    # Get the global pipeline
    pipeline = get_pipeline()
    
    # Process data through the complete pipeline
    result, _ = pipeline.process(data, symbol, source)
    return result

def migration_decorator(func):
    """
    Decorator to update strategy methods that use the old data processing.
    
    Example usage:
    
    @migration_decorator
    def your_strategy_method(self, data):
        # Your code here - data will already be processed through the new pipeline
        pass
    """
    @wraps(func)
    def wrapper(self, data, *args, **kwargs):
        # Extract symbol from kwargs or instance
        symbol = kwargs.get('symbol')
        if not symbol and hasattr(self, 'symbol'):
            symbol = self.symbol
        
        # Extract source from kwargs or instance
        source = kwargs.get('source')
        if not source and hasattr(self, 'data_source'):
            source = self.data_source
        
        # Process data if it's not already processed
        if isinstance(data, pd.DataFrame) and 'processed' not in data.attrs:
            pipeline = get_pipeline()
            processed_data, _ = pipeline.process(data, symbol, source)
            processed_data.attrs['processed'] = True
            
            # Call the original function with processed data
            return func(self, processed_data, *args, **kwargs)
        
        # Data already processed or not a DataFrame
        return func(self, data, *args, **kwargs)
    
    return wrapper

def update_strategy_references(strategy_class):
    """
    Update strategy class data processing references.
    
    This function adds the migration decorator to relevant methods
    and updates references to old data processors.
    
    Example usage:
    
    class YourStrategy(Strategy):
        # Your strategy code here
    
    update_strategy_references(YourStrategy)
    """
    # Methods that typically process data
    target_methods = [
        'generate_signals',
        'process_data',
        'calculate_indicators',
        'analyze_market_data'
    ]
    
    # Add decorator to relevant methods
    for method_name in target_methods:
        if hasattr(strategy_class, method_name):
            original_method = getattr(strategy_class, method_name)
            setattr(strategy_class, method_name, migration_decorator(original_method))
    
    logger.info(f"Updated data processing references in {strategy_class.__name__}")
    return strategy_class
