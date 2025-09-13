#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataCleaningProcessor - Process raw market data for normalization, standardization and cleaning.

This refactored version focuses specifically on data transformation and normalization,
leaving validation and quality assurance to the DataQualityProcessor.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import time

from trading_bot.data.processors.base_processor import DataProcessor

logger = logging.getLogger("DataCleaningProcessor")

class DataCleaningProcessor(DataProcessor):
    """
    Processor for cleaning and normalizing market data to ensure consistent input for trading strategies.
    
    This processor handles:
    - Data normalization and scaling
    - Timestamp standardization and alignment
    - Handling split/dividend adjustments
    - Smoothing and filtering operations
    - Basic data preparation and transformation
    
    Quality validation and assurance are handled by the DataQualityProcessor.
    """
    
    def __init__(self, name: str = "DataCleaningProcessor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaningProcessor.
        
        Args:
            name: Processor name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Filter settings
        self.price_filter_window = self.config.get('price_filter_window', 21)
        self.volume_filter_window = self.config.get('volume_filter_window', 21)
        
        # Cleaning stats
        self.stats = {
            'columns_normalized': 0,
            'timestamps_standardized': 0,
            'splits_adjusted': 0,
            'rows_processed': 0,
            'last_run': None
        }
        
        self.init_from_config()
    
    def init_from_config(self) -> None:
        """Initialize attributes from configuration."""
        # Enable/disable specific cleaning operations
        self.standardize_timestamps = self.config.get('standardize_timestamps', True)
        self.normalize_volume = self.config.get('normalize_volume', False)
        self.adjust_for_splits = self.config.get('adjust_for_splits', True)
        self.apply_smoothing = self.config.get('apply_smoothing', False)
        
        # Technical parameters
        self.timestamp_format = self.config.get('timestamp_format', '%Y-%m-%d %H:%M:%S')
        self.volume_normalization_window = self.config.get('volume_normalization_window', 20)
        self.smoothing_method = self.config.get('smoothing_method', 'ewm')
        self.smoothing_window = self.config.get('smoothing_window', 5)
        self.smoothing_columns = self.config.get('smoothing_columns', ['open', 'high', 'low', 'close'])
    
    def process(self, data: Union[List, pd.DataFrame]) -> pd.DataFrame:
        """
        Process the input data to clean and normalize.
        
        Args:
            data: Input data (list of market data objects or DataFrame)
            
        Returns:
            Cleaned DataFrame
        """
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            df = self.convert_to_dataframe(data)
        else:
            df = data.copy()
        
        # Reset statistics
        self._reset_stats()
        
        start_time = time.time()
        logger.info(f"Starting data cleaning process for {len(df)} rows")
        
        # Record original row count
        original_rows = len(df)
        
        # Perform cleaning and normalization operations
        if self.standardize_timestamps:
            df = self._standardize_timestamps(df)
        
        if self.adjust_for_splits:
            df = self._adjust_for_splits(df)
        
        if self.apply_smoothing:
            df = self._apply_smoothing(df)
        
        if self.normalize_volume:
            df = self._normalize_volume(df)
        
        # Update statistics
        self.stats['rows_processed'] = len(df)
        self.stats['last_run'] = datetime.now().isoformat()
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Data cleaning completed in {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Rows processed: {self.stats['rows_processed']} (From: {original_rows})")
        
        return df
    
    def _reset_stats(self) -> None:
        """Reset cleaning statistics."""
        self.stats = {key: 0 for key in self.stats}
        self.stats['last_run'] = None
    
    def _get_stats_summary(self) -> str:
        """Get summary of cleaning statistics."""
        return f"Processed {self.stats['rows_processed']} rows. " \
               f"Standardized {self.stats['timestamps_standardized']} timestamps. " \
               f"Normalized {self.stats['columns_normalized']} columns."
    
    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize timestamps to ensure consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized timestamps
        """
        result = df.copy()
        
        # Ensure index is datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            if 'timestamp' in result.columns:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
                    result['timestamp'] = pd.to_datetime(result['timestamp'])
                
                # Set as index
                result.set_index('timestamp', inplace=True)
                logger.info("Converted timestamp column to DatetimeIndex")
                self.stats['timestamps_standardized'] += len(result)
            else:
                logger.warning("No timestamp column found, cannot standardize timestamps")
                return result
        
        # Ensure timezone-aware
        if result.index.tz is None:
            result.index = result.index.tz_localize('UTC')
            logger.info("Localized timestamps to UTC")
        
        # Sort by timestamp
        if not result.index.is_monotonic_increasing:
            result = result.sort_index()
            logger.info("Sorted data by timestamp")
        
        return result
    
    def _adjust_for_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust price and volume data for stock splits.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with split adjustments
        """
        # This implementation depends on having split adjustment data available
        # For now, we'll leave this as a placeholder for future implementation
        
        # Check if split data is available
        if 'split_adjustment' in self.config:
            logger.info("Split adjustment data found in config, applying adjustments")
            # Apply split adjustments based on config
            # Implementation would go here
            self.stats['splits_adjusted'] += len(df)
        
        return df
    
    def _apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothing to price data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with smoothed price data
        """
        result = df.copy()
        
        # Columns to smooth
        columns_to_smooth = [col for col in self.smoothing_columns if col in result.columns]
        
        if not columns_to_smooth:
            logger.warning("No price columns found for smoothing")
            return result
        
        # Create smoothed columns
        smoothed_data = {}
        
        for col in columns_to_smooth:
            # Apply the selected smoothing method
            if self.smoothing_method == 'sma':
                # Simple moving average
                smoothed_data[f"{col}_smooth"] = result[col].rolling(
                    window=self.smoothing_window, 
                    min_periods=1
                ).mean()
            
            elif self.smoothing_method == 'ewm':
                # Exponential weighted moving average
                span = self.smoothing_window * 2 - 1  # Conversion from window to span
                smoothed_data[f"{col}_smooth"] = result[col].ewm(
                    span=span, 
                    min_periods=1
                ).mean()
            
            elif self.smoothing_method == 'gaussian':
                # Gaussian-weighted moving average
                if pd.Series(result[col]).isnull().sum() == 0:  # Only if no NaN values
                    window = min(self.smoothing_window, len(result) - 1)
                    weights = np.exp(-0.5 * (np.arange(-window, window+1) / (window/2))**2)
                    weights /= weights.sum()
                    smoothed_data[f"{col}_smooth"] = pd.Series(
                        np.convolve(result[col], weights, mode='same'),
                        index=result.index
                    )
                else:
                    smoothed_data[f"{col}_smooth"] = result[col]
        
        # Add smoothed columns to result
        for col, data in smoothed_data.items():
            result[col] = data
            self.stats['columns_normalized'] += 1
        
        logger.info(f"Applied {self.smoothing_method} smoothing to {len(columns_to_smooth)} columns")
        
        return result
    
    def _normalize_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize volume data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized volume
        """
        result = df.copy()
        
        # Check if volume column exists
        if 'volume' not in result.columns:
            logger.warning("No volume column found for normalization")
            return result
        
        # Calculate rolling median of volume
        rolling_median = result['volume'].rolling(
            window=self.volume_normalization_window, 
            min_periods=1
        ).median()
        
        # Add relative volume
        result['relative_volume'] = result['volume'] / rolling_median
        self.stats['columns_normalized'] += 1
        
        logger.info("Added normalized volume metrics")
        
        return result
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        Get cleaning statistics.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.stats.copy()
    
    def __str__(self) -> str:
        """String representation of the DataCleaningProcessor."""
        return f"DataCleaningProcessor (Smoothing: {self.smoothing_method}, Window: {self.smoothing_window})"
