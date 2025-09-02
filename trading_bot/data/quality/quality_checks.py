#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataQualityCheck - Data quality check implementations for the trading platform.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import time

logger = logging.getLogger("DataQualityCheck")

class DataQualityCheck:
    """
    Static collection of data quality check implementations.
    
    Each method in this class performs a specific quality check and returns:
    1. The potentially fixed/cleaned DataFrame
    2. A dictionary containing check results and statistics
    """
    
    @staticmethod
    def check_duplicate_data(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check for and handle duplicate data in the DataFrame.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Check for duplicates
        duplicate_rows = df.duplicated(keep='first')
        duplicate_count = duplicate_rows.sum()
        duplicate_pct = duplicate_count / len(df) if len(df) > 0 else 0
        
        # Prepare results dictionary
        results = {
            "duplicate_count": duplicate_count,
            "duplicate_pct": duplicate_pct,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Log the findings
        if duplicate_count > 0:
            message = f"Found {duplicate_count} duplicate rows ({duplicate_pct:.2%}) in {symbol} data from {source}"
            
            # Add issue only if it's a significant problem
            if duplicate_pct > 0.01:  # More than 1% duplicates
                results["issues"].append({
                    "type": "duplicate_data",
                    "count": duplicate_count,
                    "percentage": duplicate_pct,
                    "message": message
                })
                
                # Apply quality deduction
                results["quality_deduction"] = min(15.0, duplicate_pct * 1000)  # Up to 15 point deduction
                logger.warning(message)
            else:
                results["warnings"].append({
                    "type": "duplicate_data",
                    "count": duplicate_count,
                    "percentage": duplicate_pct,
                    "message": message
                })
                logger.info(message)
            
            # Remove duplicates and record the fix
            result_df = df.drop_duplicates(keep='first')
            
            results["fixes"].append({
                "type": "remove_duplicates",
                "count": duplicate_count,
                "message": f"Removed {duplicate_count} duplicate rows from {symbol} data"
            })
        else:
            # No duplicates found
            result_df = df
            
        return result_df, results
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check for and handle missing values in the DataFrame.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Check for missing values in important columns
        important_cols = ['open', 'high', 'low', 'close']
        optional_cols = ['volume', 'adj_close']
        
        # Only check columns that actually exist in the DataFrame
        important_cols = [col for col in important_cols if col in df.columns]
        optional_cols = [col for col in optional_cols if col in df.columns]
        
        # Count missing values
        missing_important = df[important_cols].isna().sum().sum()
        missing_optional = df[optional_cols].isna().sum().sum() if optional_cols else 0
        total_missing = missing_important + missing_optional
        
        # Calculate missing percentages
        important_cells = len(df) * len(important_cols)
        optional_cells = len(df) * len(optional_cols) if optional_cols else 0
        total_cells = important_cells + optional_cells
        
        missing_important_pct = missing_important / important_cells if important_cells > 0 else 0
        missing_optional_pct = missing_optional / optional_cells if optional_cells > 0 else 0
        missing_total_pct = total_missing / total_cells if total_cells > 0 else 0
        
        # Prepare results dictionary
        results = {
            "missing_count": total_missing,
            "missing_important": missing_important,
            "missing_optional": missing_optional,
            "missing_pct": missing_total_pct,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Handle missing values if any
        if total_missing > 0:
            # Create a copy of the DataFrame for modifications
            result_df = df.copy()
            
            # Different handling for important vs optional columns
            missing_details = []
            
            # Handle important columns (OHLC)
            if missing_important > 0:
                missing_message = f"Found {missing_important} missing values ({missing_important_pct:.2%}) in critical columns of {symbol} data"
                
                # Add as issue if significant
                if missing_important_pct > 0.005:  # More than 0.5% missing in important cols
                    results["issues"].append({
                        "type": "missing_values",
                        "count": missing_important,
                        "percentage": missing_important_pct,
                        "message": missing_message,
                        "affected_columns": [col for col in important_cols if df[col].isna().any()]
                    })
                    
                    # Apply quality deduction
                    results["quality_deduction"] = min(25.0, missing_important_pct * 1500)  # Up to 25 point deduction
                    logger.warning(missing_message)
                else:
                    results["warnings"].append({
                        "type": "missing_values",
                        "count": missing_important,
                        "percentage": missing_important_pct,
                        "message": missing_message,
                        "affected_columns": [col for col in important_cols if df[col].isna().any()]
                    })
                    logger.info(missing_message)
                
                # Track details for each column
                for col in important_cols:
                    if df[col].isna().any():
                        missing_count = df[col].isna().sum()
                        missing_details.append(f"{col}: {missing_count} ({missing_count/len(df):.2%})")
                
                # Fill missing values in important columns
                # First try forward fill
                result_df[important_cols] = result_df[important_cols].fillna(method='ffill')
                # Then try backward fill for any remaining NaNs (e.g., at the beginning)
                result_df[important_cols] = result_df[important_cols].fillna(method='bfill')
                
                # For any still missing values, use nearby OHLC relationships
                for i, row in result_df.iterrows():
                    for col in important_cols:
                        if pd.isna(row[col]):
                            # Try to infer from other OHLC values in the same row
                            if col == 'open' and not pd.isna(row.get('close')):
                                result_df.at[i, 'open'] = row['close']
                            elif col == 'close' and not pd.isna(row.get('open')):
                                result_df.at[i, 'close'] = row['open']
                            elif col == 'high' and not pd.isna(row.get('open')) and not pd.isna(row.get('close')):
                                result_df.at[i, 'high'] = max(row['open'], row['close'])
                            elif col == 'low' and not pd.isna(row.get('open')) and not pd.isna(row.get('close')):
                                result_df.at[i, 'low'] = min(row['open'], row['close'])
                
                # Record the fix
                results["fixes"].append({
                    "type": "fill_missing_values",
                    "count": missing_important,
                    "message": f"Filled {missing_important} missing values in critical columns of {symbol} data",
                    "details": missing_details
                })
            
            # Handle optional columns (volume, adj_close, etc.)
            if missing_optional > 0:
                missing_message = f"Found {missing_optional} missing values ({missing_optional_pct:.2%}) in non-critical columns of {symbol} data"
                
                # Only add as warning
                results["warnings"].append({
                    "type": "missing_values",
                    "count": missing_optional,
                    "percentage": missing_optional_pct,
                    "message": missing_message,
                    "affected_columns": [col for col in optional_cols if df[col].isna().any()]
                })
                
                # Track details for each column
                missing_details = []
                for col in optional_cols:
                    if df[col].isna().any():
                        missing_count = df[col].isna().sum()
                        missing_details.append(f"{col}: {missing_count} ({missing_count/len(df):.2%})")
                
                # Fill missing values in optional columns
                for col in optional_cols:
                    if result_df[col].isna().any():
                        if col == 'volume':
                            # For volume, use 0 as a reasonable default
                            result_df[col] = result_df[col].fillna(0)
                        else:
                            # For other optional columns, use forward/backward fill
                            result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
                
                # Record the fix
                results["fixes"].append({
                    "type": "fill_missing_values",
                    "count": missing_optional,
                    "message": f"Filled {missing_optional} missing values in non-critical columns of {symbol} data",
                    "details": missing_details
                })
                
            return result_df, results
        else:
            # No missing values
            return df, results
    
    @staticmethod
    def check_price_outliers(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check for and handle price outliers in the DataFrame.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Create a copy for modifications
        result_df = df.copy()
        
        # Prepare results dictionary
        results = {
            "outlier_count": 0,
            "price_jump_count": 0,
            "outlier_pct": 0.0,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Check for basic price integrity
        if 'close' not in df.columns:
            return df, results  # Nothing to check
            
        # Method 1: Z-score outlier detection for close prices
        # Calculate rolling mean and standard deviation
        window_size = 20  # Look back window for outlier detection
        if len(df) >= window_size:
            rolling_mean = df['close'].rolling(window=window_size, min_periods=1).mean()
            rolling_std = df['close'].rolling(window=window_size, min_periods=1).std()
            
            # Calculate z-scores
            z_scores = np.abs((df['close'] - rolling_mean) / rolling_std.replace(0, np.nan))
            z_scores = z_scores.fillna(0)  # Replace NaN with 0
            
            # Identify outliers based on z-score
            z_score_threshold = 4.0  # Number of standard deviations to consider an outlier
            outliers_z = z_scores > z_score_threshold
            
            # Method 2: Price jump detection
            # Calculate percentage changes
            pct_changes = df['close'].pct_change().abs() * 100
            pct_changes = pct_changes.fillna(0)  # Replace NaN with 0
            
            # Identify large price jumps
            jump_threshold = 10.0  # 10% price change
            price_jumps = pct_changes > jump_threshold
            
            # Combine outlier detection methods
            # An outlier needs to be flagged by both methods to reduce false positives
            outliers = outliers_z & price_jumps
            outlier_count = outliers.sum()
            outlier_pct = outlier_count / len(df) if len(df) > 0 else 0
            
            # Update results
            results["outlier_count"] = outlier_count
            results["price_jump_count"] = price_jumps.sum()
            results["outlier_pct"] = outlier_pct
            
            if outlier_count > 0:
                message = f"Found {outlier_count} price outliers ({outlier_pct:.2%}) in {symbol} data from {source}"
                
                # Add as issue if significant
                if outlier_pct > 0.005:  # More than 0.5% outliers
                    results["issues"].append({
                        "type": "price_outliers",
                        "count": outlier_count,
                        "percentage": outlier_pct,
                        "message": message
                    })
                    
                    # Apply quality deduction
                    results["quality_deduction"] = min(20.0, outlier_pct * 1000)  # Up to 20 point deduction
                    logger.warning(message)
                else:
                    results["warnings"].append({
                        "type": "price_outliers",
                        "count": outlier_count,
                        "percentage": outlier_pct,
                        "message": message
                    })
                    logger.info(message)
                
                # Add a column to mark outliers
                result_df['price_outlier'] = outliers
                
                # Handle outliers
                # For outliers, replace with linearly interpolated values
                if outlier_count > 0 and outlier_count < len(df) * 0.1:  # Don't fix if too many outliers
                    # Make a copy of the close price
                    original_close = result_df['close'].copy()
                    
                    # Replace outliers with NaN and interpolate
                    result_df.loc[outliers, 'close'] = np.nan
                    result_df['close'] = result_df['close'].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                    
                    # Record the fix
                    results["fixes"].append({
                        "type": "fix_price_outliers",
                        "count": outlier_count,
                        "message": f"Replaced {outlier_count} price outliers with interpolated values in {symbol} data"
                    })
        
        return result_df, results
    
    @staticmethod
    def check_ohlc_integrity(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check and fix OHLC data integrity issues.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Check if this is OHLC data
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}  # Not OHLC data
            
        # Create a copy for modifications
        result_df = df.copy()
        
        # Prepare results dictionary
        results = {
            "integrity_violations": 0,
            "high_low_violations": 0,
            "open_bounds_violations": 0,
            "close_bounds_violations": 0,
            "fixed_violations": 0,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Check high >= low
        high_low_violations = (df['high'] < df['low'])
        high_low_count = high_low_violations.sum()
        
        # Check open is between high and low
        open_violations = (df['open'] > df['high']) | (df['open'] < df['low'])
        open_violation_count = open_violations.sum()
        
        # Check close is between high and low
        close_violations = (df['close'] > df['high']) | (df['close'] < df['low'])
        close_violation_count = close_violations.sum()
        
        # Total violations
        total_violations = high_low_count + open_violation_count + close_violation_count
        violation_pct = total_violations / (len(df) * 3) if len(df) > 0 else 0  # 3 checks per row
        
        # Update results
        results["integrity_violations"] = total_violations
        results["high_low_violations"] = high_low_count
        results["open_bounds_violations"] = open_violation_count
        results["close_bounds_violations"] = close_violation_count
        
        if total_violations > 0:
            message = f"Found {total_violations} OHLC integrity violations ({violation_pct:.2%}) in {symbol} data from {source}"
            
            # This is always a significant issue
            results["issues"].append({
                "type": "ohlc_integrity",
                "count": total_violations,
                "percentage": violation_pct,
                "message": message,
                "details": {
                    "high_low_violations": high_low_count,
                    "open_bounds_violations": open_violation_count,
                    "close_bounds_violations": close_violation_count
                }
            })
            
            # Apply quality deduction - this is serious
            results["quality_deduction"] = min(30.0, violation_pct * 2000)  # Up to 30 point deduction
            logger.warning(message)
            
            # Fix high-low violations
            if high_low_count > 0:
                for idx in result_df[high_low_violations].index:
                    # Swap high and low
                    temp_high = result_df.at[idx, 'high']
                    result_df.at[idx, 'high'] = result_df.at[idx, 'low']
                    result_df.at[idx, 'low'] = temp_high
            
            # Fix open violations
            fixed_open = 0
            if open_violation_count > 0:
                for idx in result_df[open_violations].index:
                    if result_df.at[idx, 'open'] > result_df.at[idx, 'high']:
                        result_df.at[idx, 'open'] = result_df.at[idx, 'high']
                        fixed_open += 1
                    elif result_df.at[idx, 'open'] < result_df.at[idx, 'low']:
                        result_df.at[idx, 'open'] = result_df.at[idx, 'low']
                        fixed_open += 1
            
            # Fix close violations
            fixed_close = 0
            if close_violation_count > 0:
                for idx in result_df[close_violations].index:
                    if result_df.at[idx, 'close'] > result_df.at[idx, 'high']:
                        result_df.at[idx, 'close'] = result_df.at[idx, 'high']
                        fixed_close += 1
                    elif result_df.at[idx, 'close'] < result_df.at[idx, 'low']:
                        result_df.at[idx, 'close'] = result_df.at[idx, 'low']
                        fixed_close += 1
            
            # Record the fixes
            total_fixed = high_low_count + fixed_open + fixed_close
            results["fixed_violations"] = total_fixed
            
            results["fixes"].append({
                "type": "fix_ohlc_integrity",
                "count": total_fixed,
                "message": f"Fixed {total_fixed} OHLC integrity violations in {symbol} data",
                "details": {
                    "high_low_swapped": high_low_count,
                    "open_fixed": fixed_open,
                    "close_fixed": fixed_close
                }
            })
        
        return result_df, results
    
    @staticmethod
    def check_data_gaps(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check for and handle data gaps in time series data.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert if the first column might be a datetime
            if df.index.name and df.index.name in df.columns and pd.api.types.is_datetime64_any_dtype(df[df.index.name]):
                df = df.set_index(df.index.name)
            else:
                # Not a time series
                return df, {"issues": [], "warnings": [], "quality_deduction": 0}
        
        # Prepare results dictionary
        results = {
            "gap_count": 0,
            "total_missing_bars": 0,
            "max_gap_size": 0,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Sort by index to ensure chronological order
        df = df.sort_index()
        
        # Detect frequency from the data
        if len(df) >= 3:
            # Calculate time differences between consecutive rows
            time_diffs = df.index.to_series().diff().dropna()
            
            # Get the most common time difference as the expected frequency
            if not time_diffs.empty:
                # Get mode of time differences (most common interval)
                most_common_diff = time_diffs.mode()[0]
                
                # Check for gaps based on the expected frequency
                expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=most_common_diff)
                missing_dates = expected_index.difference(df.index)
                
                gap_count = 0
                missing_bars = len(missing_dates)
                max_gap = 0
                current_gap = 0
                
                # Count actual gaps (consecutive missing dates)
                if missing_bars > 0:
                    for i in range(1, len(missing_dates)):
                        diff = missing_dates[i] - missing_dates[i-1]
                        if diff == most_common_diff:
                            current_gap += 1
                        else:
                            gap_count += 1
                            max_gap = max(max_gap, current_gap + 1)
                            current_gap = 0
                    
                    # Handle the last gap
                    if current_gap > 0:
                        gap_count += 1
                        max_gap = max(max_gap, current_gap + 1)
                
                # Update results
                results["gap_count"] = gap_count
                results["total_missing_bars"] = missing_bars
                results["max_gap_size"] = max_gap
                
                gap_pct = missing_bars / len(expected_index) if len(expected_index) > 0 else 0
                
                if missing_bars > 0:
                    message = f"Found {missing_bars} missing bars in {gap_count} gaps ({gap_pct:.2%}) in {symbol} data from {source}"
                    
                    # Add as issue if significant
                    if gap_pct > 0.01 or max_gap > 5:  # More than 1% missing or gaps > 5 bars
                        results["issues"].append({
                            "type": "data_gaps",
                            "count": missing_bars,
                            "gaps": gap_count,
                            "max_gap": max_gap,
                            "percentage": gap_pct,
                            "message": message
                        })
                        
                        # Apply quality deduction based on severity
                        deduction = min(15.0, gap_pct * 400 + max_gap * 0.5)  # Up to 15 point deduction
                        results["quality_deduction"] = deduction
                        logger.warning(message)
                    else:
                        results["warnings"].append({
                            "type": "data_gaps",
                            "count": missing_bars,
                            "gaps": gap_count,
                            "max_gap": max_gap,
                            "percentage": gap_pct,
                            "message": message
                        })
                        logger.info(message)
                    
                    # Fill gaps if not too large
                    if missing_bars > 0 and missing_bars < len(df) * 0.2:  # Don't fill if too many gaps
                        # Create a copy with the complete index
                        result_df = df.reindex(expected_index)
                        
                        # Fill only gaps smaller than a threshold
                        max_gap_to_fill = 5  # Maximum consecutive bars to fill
                        
                        if max_gap <= max_gap_to_fill:
                            # Interpolate missing values
                            for col in df.columns:
                                # Different interpolation based on column type
                                if col in ['open', 'high', 'low', 'close']:
                                    result_df[col] = result_df[col].interpolate(method='linear')
                                elif col == 'volume':
                                    # For volume, use average of nearby values
                                    result_df[col] = result_df[col].interpolate(method='linear').round()
                                    # Ensure non-negative
                                    result_df[col] = result_df[col].clip(lower=0)
                                else:
                                    # General interpolation for other columns
                                    result_df[col] = result_df[col].interpolate(method='linear')
                            
                            # Record the fix
                            results["fixes"].append({
                                "type": "fill_data_gaps",
                                "count": missing_bars,
                                "message": f"Filled {missing_bars} missing bars in {gap_count} gaps in {symbol} data"
                            })
                            
                            return result_df, results
        
        # No gaps found or not fixed
        return df, results
    
    @staticmethod
    def check_timestamp_irregularities(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check for timestamp irregularities in time series data.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Check if index is datetime
        is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        timestamp_col = None
        
        # If not datetime index, check if there's a timestamp column
        if not is_datetime_index:
            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if datetime_cols:
                timestamp_col = datetime_cols[0]
            else:
                # No timestamp data found
                return df, {"issues": [], "warnings": [], "quality_deduction": 0}
        
        # Prepare results dictionary
        results = {
            "out_of_order_count": 0,
            "duplicate_timestamps": 0,
            "future_timestamps": 0,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Create a copy for modifications
        result_df = df.copy()
        
        # Get timestamp series
        if is_datetime_index:
            timestamps = df.index.to_series()
        else:
            timestamps = df[timestamp_col]
        
        # Check for out-of-order timestamps
        is_sorted = timestamps.is_monotonic_increasing
        if not is_sorted:
            out_of_order_count = sum(timestamps.diff() < pd.Timedelta(0))
            results["out_of_order_count"] = out_of_order_count
            
            message = f"Found {out_of_order_count} out-of-order timestamps in {symbol} data from {source}"
            results["issues"].append({
                "type": "timestamp_irregularity",
                "subtype": "out_of_order",
                "count": out_of_order_count,
                "message": message
            })
            
            results["quality_deduction"] += min(10.0, out_of_order_count / len(df) * 500)
            logger.warning(message)
            
            # Fix: Sort by timestamp
            if is_datetime_index:
                result_df = result_df.sort_index()
            else:
                result_df = result_df.sort_values(timestamp_col)
                
            results["fixes"].append({
                "type": "sort_timestamps",
                "count": out_of_order_count,
                "message": f"Sorted {out_of_order_count} out-of-order timestamps in {symbol} data"
            })
        
        # Check for duplicate timestamps
        if is_datetime_index:
            duplicate_count = df.index.duplicated().sum()
        else:
            duplicate_count = df[timestamp_col].duplicated().sum()
            
        results["duplicate_timestamps"] = duplicate_count
        
        if duplicate_count > 0:
            message = f"Found {duplicate_count} duplicate timestamps in {symbol} data from {source}"
            results["issues"].append({
                "type": "timestamp_irregularity",
                "subtype": "duplicate_timestamps",
                "count": duplicate_count,
                "message": message
            })
            
            results["quality_deduction"] += min(10.0, duplicate_count / len(df) * 500)
            logger.warning(message)
            
            # Fix: Keep the first occurrence of each timestamp
            if is_datetime_index:
                result_df = result_df[~result_df.index.duplicated(keep='first')]
            else:
                result_df = result_df[~result_df[timestamp_col].duplicated(keep='first')]
                
            results["fixes"].append({
                "type": "remove_duplicate_timestamps",
                "count": duplicate_count,
                "message": f"Removed {duplicate_count} duplicate timestamp entries in {symbol} data"
            })
        
        # Check for future timestamps
        now = pd.Timestamp.now()
        if is_datetime_index:
            future_timestamps = (df.index > now).sum()
        else:
            future_timestamps = (df[timestamp_col] > now).sum()
            
        results["future_timestamps"] = future_timestamps
        
        if future_timestamps > 0:
            message = f"Found {future_timestamps} future timestamps in {symbol} data from {source}"
            results["issues"].append({
                "type": "timestamp_irregularity",
                "subtype": "future_timestamps",
                "count": future_timestamps,
                "message": message
            })
            
            results["quality_deduction"] += min(15.0, future_timestamps / len(df) * 1000)
            logger.warning(message)
            
            # Fix: Remove future timestamps
            if is_datetime_index:
                result_df = result_df[result_df.index <= now]
            else:
                result_df = result_df[result_df[timestamp_col] <= now]
                
            results["fixes"].append({
                "type": "remove_future_timestamps",
                "count": future_timestamps,
                "message": f"Removed {future_timestamps} future timestamp entries in {symbol} data"
            })
        
        return result_df, results
    
    @staticmethod
    def check_stale_data(df: pd.DataFrame, symbol: str, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check for stale data in the DataFrame.
        
        Args:
            df: Market data DataFrame
            symbol: Symbol identifier
            source: Data source identifier
            
        Returns:
            (cleaned_df, check_results)
        """
        if df is None or df.empty:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Check if this has price data
        if 'close' not in df.columns:
            return df, {"issues": [], "warnings": [], "quality_deduction": 0}
            
        # Prepare results dictionary
        results = {
            "stale_sections": 0,
            "stale_bars": 0,
            "issues": [],
            "warnings": [],
            "fixes": [],
            "quality_deduction": 0.0
        }
        
        # Create a copy for modifications
        result_df = df.copy()
        
        # Check for repeated unchanged values
        price_changes = df['close'].diff() != 0
        runs = (~price_changes).astype(int).groupby(price_changes.cumsum()).cumsum()
        
        # Consider 5+ identical prices in a row as stale data
        stale_threshold = 5
        stale_sections = (runs >= stale_threshold).sum()
        stale_bars = runs[runs >= stale_threshold].sum()
        
        results["stale_sections"] = stale_sections
        results["stale_bars"] = stale_bars
        
        if stale_sections > 0:
            stale_pct = stale_bars / len(df) if len(df) > 0 else 0
            message = f"Found {stale_bars} stale data points in {stale_sections} sections ({stale_pct:.2%}) in {symbol} data from {source}"
            
            # Add as issue if significant
            if stale_pct > 0.05:  # More than 5% stale data
                results["issues"].append({
                    "type": "stale_data",
                    "count": stale_bars,
                    "sections": stale_sections,
                    "percentage": stale_pct,
                    "message": message
                })
                
                # Apply quality deduction
                results["quality_deduction"] = min(10.0, stale_pct * 100)  # Up to 10 point deduction
                logger.warning(message)
            else:
                results["warnings"].append({
                    "type": "stale_data",
                    "count": stale_bars,
                    "sections": stale_sections,
                    "percentage": stale_pct,
                    "message": message
                })
                logger.info(message)
            
            # Add a column to mark stale data
            result_df['stale_data'] = runs >= stale_threshold
            
            # No automatic fixing for stale data - needs manual review
            
        return result_df, results
