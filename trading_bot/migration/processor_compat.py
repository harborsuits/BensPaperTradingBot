#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processor Compatibility Layer

This module provides compatibility functions for data processing
during the migration to the new organization structure.
"""

import logging
import warnings
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def get_data_processor(processor_type: str = "cleaning", config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a data processor with support for both old and new implementations.
    
    Args:
        processor_type: Type of processor ("cleaning", "quality", or "combined")
        config: Processor configuration
        
    Returns:
        Data processor instance
    """
    if processor_type == "cleaning":
        return _get_cleaning_processor(config)
    elif processor_type == "quality":
        return _get_quality_processor(config)
    elif processor_type == "combined":
        return _get_combined_processor(config)
    else:
        logger.error(f"Unknown processor type: {processor_type}")
        return None

def _get_cleaning_processor(config: Optional[Dict[str, Any]] = None) -> Any:
    """Get a data cleaning processor."""
    # First try the new refactored processor
    try:
        from trading_bot.data.processors.data_cleaning_processor_refactored import DataCleaningProcessor
        return DataCleaningProcessor(config=config)
    except ImportError:
        # Fall back to old processor
        try:
            from trading_bot.data.processors.data_cleaning_processor import DataCleaningProcessor
            return DataCleaningProcessor(config=config)
        except ImportError:
            logger.error("No DataCleaningProcessor implementation found")
            return None

def _get_quality_processor(config: Optional[Dict[str, Any]] = None) -> Any:
    """Get a data quality processor."""
    # Try the new quality processor
    try:
        from trading_bot.data.quality.integration import DataQualityProcessor
        from trading_bot.core.event_system import EventBus
        
        # Create event bus if needed
        event_bus = None
        try:
            event_bus = EventBus.instance()
        except:
            pass
            
        return DataQualityProcessor(config=config, event_bus=event_bus)
    except ImportError:
        logger.warning("DataQualityProcessor not found, falling back to DataCleaningProcessor")
        # Fall back to cleaning processor
        return _get_cleaning_processor(config)

def _get_combined_processor(config: Optional[Dict[str, Any]] = None) -> Any:
    """Get a combined processor that handles both cleaning and quality validation."""
    return CombinedDataProcessor(config=config)

class CombinedDataProcessor:
    """
    Adapter that combines data cleaning and quality validation.
    
    This processor wraps both the cleaning and quality processors to provide
    a unified interface during migration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the combined processor.
        
        Args:
            config: Processor configuration
        """
        self.config = config or {}
        self.cleaning_processor = _get_cleaning_processor(config)
        self.quality_processor = _get_quality_processor(config)
        
        # Check if we have both processors
        if not self.cleaning_processor:
            raise ValueError("Failed to initialize cleaning processor")
            
        # If quality processor failed, just use cleaning
        if not self.quality_processor:
            self.quality_processor = self.cleaning_processor
    
    def process(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbol: Optional[str] = None, 
               source: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Process data through both cleaning and quality processors.
        
        Args:
            data: Input data
            symbol: Symbol identifier (for DataQualityProcessor)
            source: Source identifier (for DataQualityProcessor)
            
        Returns:
            Processed data
        """
        # First apply cleaning
        cleaned_data = self.cleaning_processor.process(data)
        
        # Then apply quality validation
        if hasattr(self.quality_processor, 'check_data_quality') and symbol and source:
            # Direct call to DataQualityManager
            if isinstance(cleaned_data, pd.DataFrame):
                validated_data, report = self.quality_processor.check_data_quality(
                    cleaned_data, symbol, source, auto_repair=True
                )
                return validated_data
            else:
                # Handle dict of DataFrames
                result = {}
                for key, df in cleaned_data.items():
                    sym = symbol if symbol else key
                    validated_df, report = self.quality_processor.check_data_quality(
                        df, sym, source, auto_repair=True
                    )
                    result[key] = validated_df
                return result
        else:
            # Using standard process interface
            return self.quality_processor.process(cleaned_data, symbol, source)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined processing statistics."""
        stats = {}
        
        # Get cleaning stats
        if hasattr(self.cleaning_processor, 'get_cleaning_stats'):
            stats.update(self.cleaning_processor.get_cleaning_stats())
        
        # Get quality stats if available
        if hasattr(self.quality_processor, 'get_quality_summary'):
            stats.update(self.quality_processor.get_quality_summary())
        
        return stats
