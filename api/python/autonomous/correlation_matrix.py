#!/usr/bin/env python3
"""
Correlation Matrix

This module implements the core data structure for tracking pairwise correlations
between trading strategies and providing correlation analysis capabilities.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class CorrelationMatrix:
    """
    Tracks pairwise correlations between multiple strategies.
    
    This class provides the core data structure and methods for:
    - Maintaining time series of strategy returns
    - Calculating rolling correlation matrices
    - Identifying highly correlated strategy pairs
    - Tracking correlation changes over time
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 min_periods: int = 10,
                 correlation_method: str = 'pearson'):
        """
        Initialize the correlation matrix.
        
        Args:
            window_size: Window size for rolling correlation (days)
            min_periods: Minimum periods required for calculation
            correlation_method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.correlation_method = correlation_method
        
        # Store returns data as pandas DataFrame
        # Index: dates, Columns: strategy_ids
        self.returns_data = pd.DataFrame()
        
        # Store latest correlation matrix
        self.latest_correlation = pd.DataFrame()
        
        # Historical correlation matrices
        # Each entry is a tuple of (timestamp, correlation_matrix)
        self.historical_correlations: List[Tuple[datetime, pd.DataFrame]] = []
        
        # Maximum historical matrices to store
        self.max_historical = 30
        
        # Significant correlation changes (strategy_pair, old_corr, new_corr, timestamp)
        self.significant_changes: List[Tuple[Tuple[str, str], float, float, datetime]] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("Correlation Matrix initialized with window size {window_size}")
    
    def add_return_data(self, 
                        strategy_id: str, 
                        date: datetime, 
                        return_value: float) -> None:
        """
        Add daily return data for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            date: Date for the return
            return_value: Return value (percentage)
        """
        with self._lock:
            # Convert date to pandas Timestamp
            date_idx = pd.Timestamp(date.date())
            
            # If this strategy is new, add a column
            if strategy_id not in self.returns_data.columns:
                self.returns_data[strategy_id] = np.nan
                logger.info(f"Added new strategy {strategy_id} to correlation tracking")
            
            # Add the return data
            self.returns_data.loc[date_idx, strategy_id] = return_value
            
            # Sort index to ensure chronological order
            self.returns_data = self.returns_data.sort_index()
    
    def add_batch_return_data(self, 
                             returns_dict: Dict[str, Dict[datetime, float]]) -> None:
        """
        Add batch return data for multiple strategies.
        
        Args:
            returns_dict: Dict of strategy_id -> (date -> return_value)
        """
        with self._lock:
            for strategy_id, returns in returns_dict.items():
                for date, return_value in returns.items():
                    self.add_return_data(strategy_id, date, return_value)
    
    def calculate_correlation(self, 
                             as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix as of a specific date.
        
        Args:
            as_of_date: Date to calculate correlation up to (default: latest available)
            
        Returns:
            Correlation matrix as pandas DataFrame
        """
        with self._lock:
            # Handle empty data case
            if self.returns_data.empty:
                return pd.DataFrame()
            
            # Filter data up to as_of_date if provided
            if as_of_date:
                date_idx = pd.Timestamp(as_of_date.date())
                data = self.returns_data[self.returns_data.index <= date_idx]
            else:
                data = self.returns_data
            
            # If we don't have enough data yet, return empty matrix
            if len(data) < self.min_periods:
                return pd.DataFrame()
            
            # Calculate rolling correlation using the last N periods
            if len(data) <= self.window_size:
                # Use all available data if less than window size
                correlation = data.corr(method=self.correlation_method)
            else:
                # Use last window_size periods
                correlation = data.tail(self.window_size).corr(method=self.correlation_method)
            
            # Store latest correlation
            timestamp = datetime.now()
            self.latest_correlation = correlation
            
            # Store in historical data
            self.historical_correlations.append((timestamp, correlation))
            
            # Trim historical data if needed
            if len(self.historical_correlations) > self.max_historical:
                self.historical_correlations = self.historical_correlations[-self.max_historical:]
            
            # Check for significant changes
            self._check_for_significant_changes(correlation, timestamp)
            
            return correlation
    
    def _check_for_significant_changes(self, 
                                      correlation: pd.DataFrame, 
                                      timestamp: datetime,
                                      threshold: float = 0.3) -> None:
        """
        Check for significant changes in correlation.
        
        Args:
            correlation: New correlation matrix
            timestamp: Timestamp for the calculation
            threshold: Change threshold to consider significant
        """
        # Skip if we don't have historical data
        if len(self.historical_correlations) <= 1:
            return
        
        # Get previous correlation matrix
        prev_timestamp, prev_correlation = self.historical_correlations[-2]
        
        # Skip if the matrices don't have the same strategies
        if not correlation.index.equals(prev_correlation.index):
            return
        
        # Check each pair of strategies
        for i in range(len(correlation.index)):
            for j in range(i+1, len(correlation.index)):
                strategy1 = correlation.index[i]
                strategy2 = correlation.index[j]
                
                prev_value = prev_correlation.iloc[i, j]
                curr_value = correlation.iloc[i, j]
                
                # Check if change exceeds threshold
                if abs(curr_value - prev_value) >= threshold:
                    self.significant_changes.append(
                        ((strategy1, strategy2), prev_value, curr_value, timestamp)
                    )
                    
                    logger.info(
                        f"Significant correlation change between {strategy1} and {strategy2}: "
                        f"{prev_value:.2f} -> {curr_value:.2f}"
                    )
    
    def get_highly_correlated_pairs(self, 
                                   threshold: float = 0.7,
                                   as_of_date: Optional[datetime] = None) -> List[Tuple[str, str, float]]:
        """
        Get highly correlated strategy pairs.
        
        Args:
            threshold: Correlation threshold
            as_of_date: Date to calculate correlation up to
            
        Returns:
            List of (strategy1, strategy2, correlation) tuples
        """
        # Calculate correlation if needed
        if as_of_date or self.latest_correlation.empty:
            correlation = self.calculate_correlation(as_of_date)
        else:
            correlation = self.latest_correlation
        
        # Handle empty matrix case
        if correlation.empty:
            return []
        
        highly_correlated = []
        
        # Check each pair of strategies
        for i in range(len(correlation.index)):
            for j in range(i+1, len(correlation.index)):
                strategy1 = correlation.index[i]
                strategy2 = correlation.index[j]
                corr_value = correlation.iloc[i, j]
                
                # Add if above threshold (positive) or below negative threshold
                if abs(corr_value) >= threshold:
                    highly_correlated.append((strategy1, strategy2, corr_value))
        
        # Sort by absolute correlation (highest first)
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return highly_correlated
    
    def get_correlation_for_pair(self, 
                                strategy1: str, 
                                strategy2: str,
                                as_of_date: Optional[datetime] = None) -> Optional[float]:
        """
        Get correlation between two specific strategies.
        
        Args:
            strategy1: First strategy ID
            strategy2: Second strategy ID
            as_of_date: Date to calculate correlation up to
            
        Returns:
            Correlation value or None if not available
        """
        # Calculate correlation if needed
        if as_of_date or self.latest_correlation.empty:
            correlation = self.calculate_correlation(as_of_date)
        else:
            correlation = self.latest_correlation
        
        # Handle empty matrix case
        if correlation.empty:
            return None
        
        # Check if both strategies are in the matrix
        if strategy1 not in correlation.index or strategy2 not in correlation.index:
            return None
        
        # Return correlation value
        return correlation.loc[strategy1, strategy2]
    
    def get_rolling_correlation(self, 
                               strategy1: str, 
                               strategy2: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.Series:
        """
        Get rolling correlation between two strategies over time.
        
        Args:
            strategy1: First strategy ID
            strategy2: Second strategy ID
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            pandas Series with rolling correlation
        """
        with self._lock:
            # Check if we have both strategies
            if strategy1 not in self.returns_data.columns or strategy2 not in self.returns_data.columns:
                return pd.Series()
            
            # Extract return data for the two strategies
            data = self.returns_data[[strategy1, strategy2]].copy()
            
            # Filter by date range if provided
            if start_date:
                start_idx = pd.Timestamp(start_date.date())
                data = data[data.index >= start_idx]
            
            if end_date:
                end_idx = pd.Timestamp(end_date.date())
                data = data[data.index <= end_idx]
            
            # Handle insufficient data
            if len(data) < self.min_periods:
                return pd.Series()
            
            # Calculate rolling correlation
            rolling_corr = data[strategy1].rolling(
                window=self.window_size,
                min_periods=self.min_periods
            ).corr(data[strategy2])
            
            return rolling_corr
    
    def get_correlation_distances(self) -> pd.DataFrame:
        """
        Convert correlation matrix to distance matrix for clustering.
        
        Distance = 1 - abs(correlation), so highly correlated pairs have low distance.
        
        Returns:
            Distance matrix as pandas DataFrame
        """
        # Use latest correlation matrix
        if self.latest_correlation.empty:
            self.calculate_correlation()
        
        if self.latest_correlation.empty:
            return pd.DataFrame()
        
        # Convert correlation to distance
        # 1 - |correlation| so that highly correlated pairs have low distance
        distance = 1 - self.latest_correlation.abs()
        
        return distance
    
    def get_least_correlated_to(self, 
                               strategy_id: str,
                               top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Get strategies least correlated to a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            top_n: Number of strategies to return
            
        Returns:
            List of (strategy_id, correlation) tuples
        """
        # Use latest correlation matrix
        if self.latest_correlation.empty:
            self.calculate_correlation()
        
        if self.latest_correlation.empty or strategy_id not in self.latest_correlation.index:
            return []
        
        # Get correlations with this strategy
        correlations = self.latest_correlation[strategy_id].drop(strategy_id)
        
        # Sort by absolute correlation (lowest first)
        sorted_corr = correlations.abs().sort_values()
        
        # Return top N least correlated
        result = []
        for other_id, corr_value in sorted_corr.items():
            if len(result) >= top_n:
                break
            
            # Use the actual correlation value, not absolute
            actual_corr = correlations[other_id]
            result.append((other_id, actual_corr))
        
        return result
    
    def get_correlation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the correlation matrix.
        
        Returns:
            Dict with statistics about correlations
        """
        # Use latest correlation matrix
        if self.latest_correlation.empty:
            self.calculate_correlation()
        
        if self.latest_correlation.empty:
            return {
                "tracked_strategies": 0,
                "data_points": 0
            }
        
        # Calculate statistics
        # Get upper triangle values (excluding diagonal)
        upper_triangle = []
        for i in range(len(self.latest_correlation.index)):
            for j in range(i+1, len(self.latest_correlation.index)):
                upper_triangle.append(self.latest_correlation.iloc[i, j])
        
        # Convert to numpy array for calculations
        corr_values = np.array(upper_triangle)
        
        stats = {
            "tracked_strategies": len(self.latest_correlation.index),
            "data_points": len(self.returns_data),
            "avg_correlation": float(np.mean(corr_values)),
            "median_correlation": float(np.median(corr_values)),
            "min_correlation": float(np.min(corr_values)),
            "max_correlation": float(np.max(corr_values)),
            "std_correlation": float(np.std(corr_values)),
            "pct_highly_positive": float(np.mean(corr_values > 0.7) * 100),
            "pct_highly_negative": float(np.mean(corr_values < -0.7) * 100),
            "last_updated": datetime.now().isoformat()
        }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the correlation matrix to dictionary for serialization.
        
        Returns:
            Dict representation
        """
        with self._lock:
            # Convert returns data to dict
            returns_dict = {}
            for strategy_id in self.returns_data.columns:
                returns_dict[strategy_id] = self.returns_data[strategy_id].dropna().to_dict()
            
            # Convert latest correlation to dict
            latest_corr_dict = {}
            if not self.latest_correlation.empty:
                for strategy1 in self.latest_correlation.index:
                    latest_corr_dict[strategy1] = {}
                    for strategy2 in self.latest_correlation.columns:
                        latest_corr_dict[strategy1][strategy2] = float(
                            self.latest_correlation.loc[strategy1, strategy2]
                        )
            
            # Format significant changes
            sig_changes = []
            for (s1, s2), old_val, new_val, ts in self.significant_changes:
                sig_changes.append({
                    "strategy1": s1,
                    "strategy2": s2,
                    "old_value": float(old_val),
                    "new_value": float(new_val),
                    "timestamp": ts.isoformat()
                })
            
            return {
                "window_size": self.window_size,
                "min_periods": self.min_periods,
                "correlation_method": self.correlation_method,
                "returns_data": returns_dict,
                "latest_correlation": latest_corr_dict,
                "significant_changes": sig_changes,
                "tracked_strategies": list(self.returns_data.columns),
                "last_updated": datetime.now().isoformat()
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrelationMatrix':
        """
        Create from dictionary representation.
        
        Args:
            data: Dict representation
            
        Returns:
            CorrelationMatrix instance
        """
        # Create instance with config params
        instance = cls(
            window_size=data.get("window_size", 30),
            min_periods=data.get("min_periods", 10),
            correlation_method=data.get("correlation_method", "pearson")
        )
        
        # Convert returns data back to DataFrame
        returns_dict = data.get("returns_data", {})
        for strategy_id, returns in returns_dict.items():
            for date_str, value in returns.items():
                date = datetime.fromisoformat(date_str)
                instance.add_return_data(strategy_id, date, value)
        
        # Recalculate correlation to initialize latest_correlation
        if instance.returns_data.shape[0] >= instance.min_periods:
            instance.calculate_correlation()
        
        # Restore significant changes
        for change in data.get("significant_changes", []):
            s1 = change["strategy1"]
            s2 = change["strategy2"]
            old_val = change["old_value"]
            new_val = change["new_value"]
            timestamp = datetime.fromisoformat(change["timestamp"])
            
            instance.significant_changes.append(
                ((s1, s2), old_val, new_val, timestamp)
            )
        
        return instance
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save correlation matrix to file.
        
        Args:
            filepath: Path to output file
            
        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert to dict and save
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Saved correlation matrix to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving correlation matrix: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['CorrelationMatrix']:
        """
        Load correlation matrix from file.
        
        Args:
            filepath: Path to input file
            
        Returns:
            CorrelationMatrix instance or None if error
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return None
            
            # Load from file
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create from dict
            instance = cls.from_dict(data)
            
            logger.info(f"Loaded correlation matrix from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading correlation matrix: {e}")
            return None
