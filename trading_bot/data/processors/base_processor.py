#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base DataProcessor - Abstract class for all data processors.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.common.market_types import MarketData

logger = logging.getLogger("DataProcessor")

class DataProcessor(ABC):
    """
    Abstract base class for all data processors.
    
    A data processor is responsible for cleaning, transforming, and preparing
    raw data for feature extraction and strategy use.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            name: Name of the processor
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        
        # Set up any additional attributes from config
        self.init_from_config()
        
        logger.info(f"Initialized {self.name} data processor")
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Default implementation does nothing
        # Override in subclasses to set up specific configuration
        pass
    
    @abstractmethod
    def process(self, data: Union[List[MarketData], pd.DataFrame]) -> pd.DataFrame:
        """
        Process the input data.
        
        Args:
            data: Input data (list of MarketData objects or DataFrame)
            
        Returns:
            Processed DataFrame
        """
        pass
    
    def convert_to_dataframe(self, market_data_list: List[MarketData]) -> pd.DataFrame:
        """
        Convert a list of MarketData objects to a pandas DataFrame.
        
        Args:
            market_data_list: List of MarketData objects
            
        Returns:
            DataFrame with market data
        """
        if not market_data_list:
            return pd.DataFrame()
            
        # Extract data from MarketData objects
        data_dict = {
            "symbol": [],
            "timestamp": [],
            "price": [],
            "volume": [],
            "open": [],
            "high": [],
            "low": []
        }
        
        for md in market_data_list:
            data_dict["symbol"].append(md.symbol)
            data_dict["timestamp"].append(md.timestamp)
            data_dict["price"].append(md.price)
            data_dict["volume"].append(md.volume)
            data_dict["open"].append(md.open_price)
            data_dict["high"].append(md.high_price)
            data_dict["low"].append(md.low_price)
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers in a DataFrame column using z-score.
        
        Args:
            df: Input DataFrame
            column: Column name to check for outliers
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return pd.Series(False, index=df.index)
            
        # Calculate z-score
        series = df[column]
        z_scores = (series - series.mean()) / series.std()
        
        # Identify outliers
        outliers = abs(z_scores) > z_threshold
        
        logger.debug(f"Detected {outliers.sum()} outliers in column '{column}'")
        return outliers
    
    def fill_missing_values(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Fill missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to fill (if None, fill all)
            method: Fill method ('ffill', 'bfill', 'interpolate', 'mean', 'median', 'zero')
            
        Returns:
            DataFrame with filled values
        """
        if df.empty:
            return df
            
        # Select columns to fill
        cols_to_fill = columns or df.columns
        
        # Check if columns exist
        missing_cols = [col for col in cols_to_fill if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns {missing_cols} not found in DataFrame")
            cols_to_fill = [col for col in cols_to_fill if col in df.columns]
        
        # Fill missing values based on method
        if method == "ffill":
            df[cols_to_fill] = df[cols_to_fill].fillna(method="ffill")
        elif method == "bfill":
            df[cols_to_fill] = df[cols_to_fill].fillna(method="bfill")
        elif method == "interpolate":
            df[cols_to_fill] = df[cols_to_fill].interpolate()
        elif method == "mean":
            for col in cols_to_fill:
                df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            for col in cols_to_fill:
                df[col] = df[col].fillna(df[col].median())
        elif method == "zero":
            df[cols_to_fill] = df[cols_to_fill].fillna(0)
        else:
            logger.warning(f"Unknown fill method: {method}")
            
        missing_count = df[cols_to_fill].isna().sum().sum()
        logger.debug(f"Filled missing values with {method}, {missing_count} missing values remain")
        
        return df
    
    def normalize_data(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = "z-score"
    ) -> pd.DataFrame:
        """
        Normalize values in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize (if None, normalize all numeric)
            method: Normalization method ('z-score', 'min-max', 'robust')
            
        Returns:
            DataFrame with normalized values
        """
        if df.empty:
            return df
            
        # Select columns to normalize
        if columns is None:
            # Select only numeric columns
            cols_to_normalize = df.select_dtypes(include=['number']).columns.tolist()
        else:
            # Check if columns exist and are numeric
            cols_to_normalize = []
            for col in columns:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in DataFrame")
                elif not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column '{col}' is not numeric, skipping normalization")
                else:
                    cols_to_normalize.append(col)
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Normalize based on method
        if method == "z-score":
            for col in cols_to_normalize:
                result_df[col] = (df[col] - df[col].mean()) / df[col].std()
                
        elif method == "min-max":
            for col in cols_to_normalize:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    result_df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    logger.warning(f"Column '{col}' has equal min and max values, skipping normalization")
                    
        elif method == "robust":
            for col in cols_to_normalize:
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    result_df[col] = (df[col] - median) / iqr
                else:
                    logger.warning(f"Column '{col}' has zero IQR, skipping normalization")
                    
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return df
        
        logger.debug(f"Normalized {len(cols_to_normalize)} columns using {method} method")
        return result_df
    
    def __str__(self) -> str:
        """String representation of the data processor."""
        return f"{self.name} DataProcessor" 