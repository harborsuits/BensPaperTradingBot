#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataCleaningProcessor - Process raw market data to handle outliers, missing values, and ensure data quality.
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
    Processor for cleaning and validating market data to ensure high quality input for trading strategies.
    
    This processor handles:
    - Detecting and handling outliers in price and volume data
    - Filling missing values using appropriate techniques
    - Data normalization and scaling
    - Timestamp consistency and alignment
    - Handling split/dividend adjustments
    - Data quality checks and reporting
    """
    
    def __init__(self, name: str = "DataCleaningProcessor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaningProcessor.
        
        Args:
            name: Processor name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Method for outlier detection
        self.outlier_method = self.config.get('outlier_method', 'z_score')
        
        # Missing value handling method
        self.fill_method = self.config.get('fill_method', 'ffill')
        
        # Outlier thresholds
        self.z_score_threshold = self.config.get('z_score_threshold', 3.0)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        self.price_jump_threshold = self.config.get('price_jump_threshold', 10.0)  # % change
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', 10.0)  # x median
        
        # Filter settings
        self.price_filter_window = self.config.get('price_filter_window', 21)
        self.volume_filter_window = self.config.get('volume_filter_window', 21)
        
        # Cleaning stats
        self.stats = {
            'outliers_detected': 0,
            'missing_values_filled': 0,
            'gaps_interpolated': 0,
            'errors_flagged': 0,
            'last_run': None,
            'processed_rows': 0
        }
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Enable/disable specific cleaning operations
        self.detect_price_outliers = self.config.get('detect_price_outliers', True)
        self.detect_volume_outliers = self.config.get('detect_volume_outliers', True)
        self.fill_missing_values = self.config.get('fill_missing_values', True)
        self.interpolate_gaps = self.config.get('interpolate_gaps', True)
        self.add_quality_indicators = self.config.get('add_quality_indicators', True)
        self.enforce_ohlc_integrity = self.config.get('enforce_ohlc_integrity', True)
        self.drop_incomplete_bars = self.config.get('drop_incomplete_bars', False)
        
        # Additional options
        self.normalize_volume = self.config.get('normalize_volume', False)
        self.adjust_for_splits = self.config.get('adjust_for_splits', True)
        self.adjust_timestamps = self.config.get('adjust_timestamps', True)
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.apply_smoothing = self.config.get('apply_smoothing', False)
        self.max_gap_interpolation = self.config.get('max_gap_interpolation', 5)  # max consecutive NaNs to interpolate
    
    def process(self, data: Union[List, pd.DataFrame]) -> pd.DataFrame:
        """
        Process the input data to clean and validate.
        
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
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' missing from input data")
                return df
        
        # Reset stats for this run
        self._reset_stats()
        self.stats['processed_rows'] = len(df)
        
        # Basic pre-checks
        df = self._perform_basic_checks(df)
        
        # Process data cleaning steps in sequence
        
        # 1. Remove duplicates if enabled
        if self.remove_duplicates:
            df = self._remove_duplicate_rows(df)
        
        # 2. Adjust timestamps if enabled
        if self.adjust_timestamps:
            df = self._standardize_timestamps(df)
        
        # 3. Enforce OHLC integrity if enabled
        if self.enforce_ohlc_integrity:
            df = self._enforce_ohlc_integrity(df)
        
        # 4. Detect and handle price outliers if enabled
        if self.detect_price_outliers:
            df = self._handle_price_outliers(df)
        
        # 5. Detect and handle volume outliers if enabled
        if self.detect_volume_outliers:
            df = self._handle_volume_outliers(df)
        
        # 6. Fill missing values if enabled
        if self.fill_missing_values:
            df = self._handle_missing_values(df)
        
        # 7. Apply smoothing if enabled
        if self.apply_smoothing:
            df = self._apply_smoothing(df)
        
        # 8. Normalize volume if enabled
        if self.normalize_volume:
            df = self._normalize_volume(df)
        
        # 9. Add data quality indicators if enabled
        if self.add_quality_indicators:
            df = self._add_quality_indicators(df)
        
        # 10. Drop incomplete bars if enabled
        if self.drop_incomplete_bars:
            df = self._drop_incomplete_bars(df)
        
        # Update last run time
        self.stats['last_run'] = datetime.now()
        
        logger.info(f"Data cleaning completed: {self._get_stats_summary()}")
        
        return df
    
    def _reset_stats(self) -> None:
        """Reset cleaning statistics."""
        self.stats['outliers_detected'] = 0
        self.stats['missing_values_filled'] = 0
        self.stats['gaps_interpolated'] = 0
        self.stats['errors_flagged'] = 0
    
    def _get_stats_summary(self) -> str:
        """Get summary of cleaning statistics."""
        return (f"Outliers: {self.stats['outliers_detected']}, "
                f"Missing values: {self.stats['missing_values_filled']}, "
                f"Gaps interpolated: {self.stats['gaps_interpolated']}, "
                f"Errors flagged: {self.stats['errors_flagged']}")
    
    def _perform_basic_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic checks on the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Check for NaN values in key columns
        nan_counts = df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN values detected in OHLCV columns: {nan_counts.to_dict()}")
        
        # Check for non-positive prices
        non_positive_prices = ((df['open'] <= 0) | (df['high'] <= 0) | 
                              (df['low'] <= 0) | (df['close'] <= 0)).sum()
        if non_positive_prices > 0:
            logger.warning(f"Non-positive prices detected in {non_positive_prices} rows")
        
        # Check for negative volume
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            logger.warning(f"Negative volume detected in {negative_volume} rows")
            # Set negative volumes to 0
            df.loc[df['volume'] < 0, 'volume'] = 0
        
        # Check for high-low relationship
        high_low_errors = (df['high'] < df['low']).sum()
        if high_low_errors > 0:
            logger.warning(f"High < Low detected in {high_low_errors} rows")
            self.stats['errors_flagged'] += high_low_errors
        
        # Check for open-high-low-close relationship
        ohlc_errors = ((df['open'] > df['high']) | (df['open'] < df['low']) | 
                      (df['close'] > df['high']) | (df['close'] < df['low'])).sum()
        if ohlc_errors > 0:
            logger.warning(f"OHLC relationship errors detected in {ohlc_errors} rows")
            self.stats['errors_flagged'] += ohlc_errors
        
        return df
    
    def _remove_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        # Check if the DataFrame has an index
        if isinstance(df.index, pd.DatetimeIndex):
            # Check for duplicate indices
            dup_indices = df.index.duplicated()
            if dup_indices.any():
                n_dups = dup_indices.sum()
                logger.warning(f"Found {n_dups} duplicate timestamps")
                
                # Keep the first occurrence
                df = df[~dup_indices]
        else:
            # Check for complete duplicate rows
            n_before = len(df)
            df = df.drop_duplicates()
            n_dups = n_before - len(df)
            
            if n_dups > 0:
                logger.warning(f"Removed {n_dups} duplicate rows")
        
        return df
    
    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize timestamps to ensure consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized timestamps
        """
        # Check if the DataFrame has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert 'timestamp' column if exists
            if 'timestamp' in df.columns:
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Set as index
                    df = df.set_index('timestamp')
                except Exception as e:
                    logger.error(f"Failed to convert timestamp column: {str(e)}")
        
        # If we now have a DatetimeIndex, ensure it's properly sorted
        if isinstance(df.index, pd.DatetimeIndex):
            # Check for timezone information
            if df.index.tz is None:
                logger.info("Adding UTC timezone to timestamps")
                df.index = df.index.tz_localize('UTC')
            
            # Sort by timestamp
            if not df.index.is_monotonic_increasing:
                logger.warning("Timestamps are not in ascending order, sorting...")
                df = df.sort_index()
        
        return df
    
    def _enforce_ohlc_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce OHLC relationships in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected OHLC values
        """
        # Create a mask for rows with OHLC issues
        high_low_mask = df['high'] < df['low']
        open_bounds_mask = (df['open'] > df['high']) | (df['open'] < df['low'])
        close_bounds_mask = (df['close'] > df['high']) | (df['close'] < df['low'])
        
        # Count issues
        issues = (high_low_mask | open_bounds_mask | close_bounds_mask).sum()
        
        if issues > 0:
            logger.warning(f"Enforcing OHLC integrity for {issues} rows")
            
            # Create corrected DF
            corrected = df.copy()
            
            # Fix high-low relationship
            if high_low_mask.any():
                # Swap high and low where high < low
                high_values = corrected.loc[high_low_mask, 'high'].copy()
                corrected.loc[high_low_mask, 'high'] = corrected.loc[high_low_mask, 'low']
                corrected.loc[high_low_mask, 'low'] = high_values
            
            # Fix open outside high-low bounds
            if open_bounds_mask.any():
                # Reset open to high if open > high
                corrected.loc[corrected['open'] > corrected['high'], 'open'] = corrected.loc[corrected['open'] > corrected['high'], 'high']
                # Reset open to low if open < low
                corrected.loc[corrected['open'] < corrected['low'], 'open'] = corrected.loc[corrected['open'] < corrected['low'], 'low']
            
            # Fix close outside high-low bounds
            if close_bounds_mask.any():
                # Reset close to high if close > high
                corrected.loc[corrected['close'] > corrected['high'], 'close'] = corrected.loc[corrected['close'] > corrected['high'], 'high']
                # Reset close to low if close < low
                corrected.loc[corrected['close'] < corrected['low'], 'close'] = corrected.loc[corrected['close'] < corrected['low'], 'low']
            
            # Update stats
            self.stats['errors_flagged'] += issues
            
            return corrected
        
        return df
    
    def _handle_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle price outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        result = df.copy()
        
        # Add a column to flag outliers
        result['price_outlier'] = False
        
        # Use the specified method for outlier detection
        if self.outlier_method == 'z_score':
            # Calculate rolling median and standard deviation
            rolling_median = result['close'].rolling(window=self.price_filter_window, center=True, min_periods=3).median()
            rolling_std = result['close'].rolling(window=self.price_filter_window, center=True, min_periods=3).std()
            
            # Calculate z-scores
            z_scores = abs((result['close'] - rolling_median) / rolling_std)
            
            # Flag outliers
            outliers = z_scores > self.z_score_threshold
            result.loc[outliers, 'price_outlier'] = True
            
        elif self.outlier_method == 'iqr':
            # Calculate rolling quantiles
            q1 = result['close'].rolling(window=self.price_filter_window, center=True, min_periods=5).quantile(0.25)
            q3 = result['close'].rolling(window=self.price_filter_window, center=True, min_periods=5).quantile(0.75)
            
            # Calculate IQR and bounds
            iqr = q3 - q1
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            # Flag outliers
            outliers = (result['close'] < lower_bound) | (result['close'] > upper_bound)
            result.loc[outliers, 'price_outlier'] = True
            
        elif self.outlier_method == 'pct_change':
            # Calculate percentage change
            pct_change = result['close'].pct_change().abs() * 100
            
            # Flag outliers
            outliers = pct_change > self.price_jump_threshold
            result.loc[outliers, 'price_outlier'] = True
        
        # Count outliers
        outlier_count = result['price_outlier'].sum()
        self.stats['outliers_detected'] += outlier_count
        
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} price outliers ({outlier_count/len(df)*100:.2f}%)")
            
            # Replace outlier values with the median of surrounding values
            if self.config.get('replace_outliers', True):
                for col in ['open', 'high', 'low', 'close']:
                    # Calculate rolling median excluding the current value
                    rolling_median = result[col].rolling(
                        window=self.price_filter_window, 
                        center=True, 
                        min_periods=3
                    ).median()
                    
                    # Replace outliers with rolling median
                    result.loc[result['price_outlier'], col] = rolling_median[result['price_outlier']]
                
                logger.info(f"Replaced {outlier_count} outlier prices with rolling median values")
        
        return result
    
    def _handle_volume_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle volume outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volume outliers handled
        """
        result = df.copy()
        
        # Add a column to flag outliers
        result['volume_outlier'] = False
        
        # Calculate rolling median volume
        rolling_median = result['volume'].rolling(
            window=self.volume_filter_window, 
            center=True, 
            min_periods=3
        ).median()
        
        # Flag extreme volume spikes
        outliers = result['volume'] > (rolling_median * self.volume_spike_threshold)
        result.loc[outliers, 'volume_outlier'] = True
        
        # Count outliers
        outlier_count = result['volume_outlier'].sum()
        self.stats['outliers_detected'] += outlier_count
        
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} volume outliers ({outlier_count/len(df)*100:.2f}%)")
            
            # Replace outlier values if configured
            if self.config.get('replace_volume_outliers', True):
                # Replace with rolling median
                result.loc[result['volume_outlier'], 'volume'] = rolling_median[result['volume_outlier']]
                logger.info(f"Replaced {outlier_count} outlier volumes with rolling median values")
        
        return result
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        result = df.copy()
        
        # Check for missing values
        missing_counts = result[['open', 'high', 'low', 'close', 'volume']].isna().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.info(f"Handling {total_missing} missing values: {missing_counts.to_dict()}")
            
            # Handle missing values based on configured method
            if self.fill_method == 'ffill':
                # Forward fill values
                result[['open', 'high', 'low', 'close', 'volume']] = result[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')
                
            elif self.fill_method == 'bfill':
                # Backward fill values
                result[['open', 'high', 'low', 'close', 'volume']] = result[['open', 'high', 'low', 'close', 'volume']].fillna(method='bfill')
                
            elif self.fill_method == 'linear':
                # Interpolate linearly
                result[['open', 'high', 'low', 'close']] = result[['open', 'high', 'low', 'close']].interpolate(method='linear', limit=self.max_gap_interpolation)
                # For volume, use 0 or the mean
                if self.config.get('fill_volume_with_zero', True):
                    result['volume'] = result['volume'].fillna(0)
                else:
                    result['volume'] = result['volume'].interpolate(method='linear', limit=self.max_gap_interpolation)
                
            elif self.fill_method == 'nearest':
                # Use nearest value
                result[['open', 'high', 'low', 'close']] = result[['open', 'high', 'low', 'close']].interpolate(method='nearest', limit=self.max_gap_interpolation)
                result['volume'] = result['volume'].interpolate(method='nearest', limit=self.max_gap_interpolation)
                
            elif self.fill_method == 'custom':
                # Custom method: OHLC uses previous close, volume uses median
                # Fill all missing OHLC with the previous close
                last_close = None
                for idx in result.index[result[['open', 'high', 'low', 'close']].isna().any(axis=1)]:
                    if last_close is not None:
                        if pd.isna(result.loc[idx, 'open']):
                            result.loc[idx, 'open'] = last_close
                        if pd.isna(result.loc[idx, 'high']):
                            result.loc[idx, 'high'] = last_close
                        if pd.isna(result.loc[idx, 'low']):
                            result.loc[idx, 'low'] = last_close
                        if pd.isna(result.loc[idx, 'close']):
                            result.loc[idx, 'close'] = last_close
                    
                    # Update last_close if we have a valid close for this row
                    if not pd.isna(result.loc[idx, 'close']):
                        last_close = result.loc[idx, 'close']
                
                # Fill volume with median of surrounding values
                if result['volume'].isna().any():
                    median_volume = result['volume'].rolling(window=self.volume_filter_window, center=True, min_periods=3).median()
                    result['volume'] = result['volume'].fillna(median_volume)
            
            # Check remaining missing values
            remaining_missing = result[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"{remaining_missing} missing values remain after filling")
                
                # Last resort: fill any remaining values
                result = result.fillna(method='ffill').fillna(method='bfill')
            
            # Update stats
            self.stats['missing_values_filled'] += total_missing - remaining_missing
            self.stats['gaps_interpolated'] += total_missing - remaining_missing
        
        return result
    
    def _apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothing to price data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with smoothed price data
        """
        result = df.copy()
        
        smoothing_method = self.config.get('smoothing_method', 'ewm')
        
        if smoothing_method == 'ewm':
            # Use exponentially weighted moving average for smoothing
            span = self.config.get('smoothing_span', 5)
            
            # Apply to OHLC
            smoothed_close = result['close'].ewm(span=span).mean()
            
            # Adjust high/low to maintain relationship with close
            high_ratio = result['high'] / result['close']
            low_ratio = result['low'] / result['close']
            
            result['close_raw'] = result['close'].copy()
            result['close'] = smoothed_close
            result['high'] = smoothed_close * high_ratio
            result['low'] = smoothed_close * low_ratio
            
            logger.info(f"Applied EWM smoothing with span={span}")
            
        elif smoothing_method == 'sma':
            # Use simple moving average
            window = self.config.get('smoothing_window', 3)
            
            # Apply to close price
            result['close_raw'] = result['close'].copy()
            result['close'] = result['close'].rolling(window=window, center=True, min_periods=1).mean()
            
            logger.info(f"Applied SMA smoothing with window={window}")
        
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
        
        # Store original volume
        result['volume_raw'] = result['volume'].copy()
        
        # Calculate rolling median and standard deviation
        rolling_median = result['volume'].rolling(window=self.volume_filter_window, min_periods=3).median()
        rolling_std = result['volume'].rolling(window=self.volume_filter_window, min_periods=3).std()
        
        # Normalize volume to z-score
        result['volume_z'] = (result['volume'] - rolling_median) / rolling_std
        
        # Winsorize extreme values
        max_z = self.config.get('max_volume_z', 3.0)
        result.loc[result['volume_z'] > max_z, 'volume_z'] = max_z
        result.loc[result['volume_z'] < -max_z, 'volume_z'] = -max_z
        
        # Add relative volume
        result['relative_volume'] = result['volume'] / rolling_median
        
        logger.info("Added normalized volume metrics")
        
        return result
    
    def _add_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add data quality indicators to the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with quality indicators
        """
        result = df.copy()
        
        # Add data quality score (0-100)
        quality_score = 100.0
        
        # Penalize for outliers
        price_outlier_pct = result['price_outlier'].mean() if 'price_outlier' in result else 0
        volume_outlier_pct = result['volume_outlier'].mean() if 'volume_outlier' in result else 0
        
        quality_score -= price_outlier_pct * 50
        quality_score -= volume_outlier_pct * 25
        
        # Penalize for out-of-bounds OHLC
        ohlc_issues = ((result['open'] > result['high']) | (result['open'] < result['low']) | 
                       (result['close'] > result['high']) | (result['close'] < result['low'])).mean()
        quality_score -= ohlc_issues * 100
        
        # Penalize for high-low inversion
        hl_issues = (result['high'] < result['low']).mean()
        quality_score -= hl_issues * 100
        
        # Add result
        result['data_quality_score'] = max(0, quality_score)
        
        # Add gap indicator
        if isinstance(result.index, pd.DatetimeIndex):
            # Calculate time delta between rows
            result['time_delta'] = result.index.to_series().diff().dt.total_seconds()
            
            # Flag abnormal gaps
            expected_seconds = self.config.get('expected_seconds_between_bars', 60)
            tolerance = self.config.get('time_gap_tolerance', 1.5)
            
            # Mark rows with abnormal gaps
            result['time_gap'] = result['time_delta'] > (expected_seconds * tolerance)
            
            # Count gaps
            gap_count = result['time_gap'].sum()
            if gap_count > 0:
                logger.info(f"Detected {gap_count} time gaps in the data")
        
        return result
    
    def _drop_incomplete_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with incomplete or invalid data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with incomplete bars removed
        """
        result = df.copy()
        
        # Create a mask for rows to drop
        rows_to_drop = pd.Series(False, index=result.index)
        
        # Check for NaN values
        rows_to_drop = rows_to_drop | result[['open', 'high', 'low', 'close', 'volume']].isna().any(axis=1)
        
        # Check for zero prices
        rows_to_drop = rows_to_drop | (result[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        
        # Check for OHLC integrity issues
        rows_to_drop = rows_to_drop | (result['high'] < result['low'])
        rows_to_drop = rows_to_drop | (result['open'] > result['high']) | (result['open'] < result['low'])
        rows_to_drop = rows_to_drop | (result['close'] > result['high']) | (result['close'] < result['low'])
        
        # Count rows to drop
        drop_count = rows_to_drop.sum()
        
        if drop_count > 0:
            logger.warning(f"Dropping {drop_count} incomplete/invalid rows ({drop_count/len(df)*100:.2f}%)")
            
            # Drop the rows
            result = result[~rows_to_drop]
        
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
        return f"DataCleaningProcessor (Outlier method: {self.outlier_method}, Fill method: {self.fill_method})" 