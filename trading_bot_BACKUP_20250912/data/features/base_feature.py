#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base FeatureExtractor - Abstract class for all feature extractors.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger("FeatureExtractor")

class FeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    A feature extractor is responsible for deriving relevant features from
    processed market data for use in trading strategies.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            name: Name of the feature extractor
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        
        # Set up any additional attributes from config
        self.init_from_config()
        
        logger.info(f"Initialized {self.name} feature extractor")
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Default implementation does nothing
        # Override in subclasses to set up specific configuration
        pass
    
    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the input DataFrame.
        
        Args:
            df: Input DataFrame with processed market data
            
        Returns:
            DataFrame with extracted features
        """
        pass
    
    def calculate_returns(
        self, 
        df: pd.DataFrame, 
        price_col: str = "price", 
        periods: List[int] = [1],
        log_returns: bool = False,
        dropna: bool = True
    ) -> pd.DataFrame:
        """
        Calculate returns over different periods.
        
        Args:
            df: Input DataFrame
            price_col: Column name for price data
            periods: List of periods for return calculation
            log_returns: Whether to calculate log returns
            dropna: Whether to drop NaN values
            
        Returns:
            DataFrame with return columns added
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df
            
        result_df = df.copy()
        
        for period in periods:
            if log_returns:
                result_df[f"log_return_{period}"] = np.log(df[price_col] / df[price_col].shift(period))
            else:
                result_df[f"return_{period}"] = df[price_col].pct_change(period)
        
        # Drop NaN values if requested
        if dropna:
            result_df.dropna(inplace=True)
            
        logger.debug(f"Calculated returns for {len(periods)} periods")
        return result_df
    
    def calculate_moving_averages(
        self, 
        df: pd.DataFrame, 
        price_col: str = "price", 
        windows: List[int] = [10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate simple moving averages.
        
        Args:
            df: Input DataFrame
            price_col: Column name for price data
            windows: List of windows for moving average calculation
            
        Returns:
            DataFrame with moving average columns added
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df
            
        result_df = df.copy()
        
        for window in windows:
            result_df[f"ma_{window}"] = df[price_col].rolling(window=window).mean()
            
        logger.debug(f"Calculated moving averages for {len(windows)} windows")
        return result_df
    
    def calculate_exponential_moving_averages(
        self, 
        df: pd.DataFrame, 
        price_col: str = "price", 
        windows: List[int] = [10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate exponential moving averages.
        
        Args:
            df: Input DataFrame
            price_col: Column name for price data
            windows: List of windows for EMA calculation
            
        Returns:
            DataFrame with EMA columns added
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df
            
        result_df = df.copy()
        
        for window in windows:
            result_df[f"ema_{window}"] = df[price_col].ewm(span=window, adjust=False).mean()
            
        logger.debug(f"Calculated exponential moving averages for {len(windows)} windows")
        return result_df
    
    def calculate_rsi(
        self, 
        df: pd.DataFrame, 
        price_col: str = "price", 
        window: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: Input DataFrame
            price_col: Column name for price data
            window: Window for RSI calculation
            
        Returns:
            DataFrame with RSI column added
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df
            
        result_df = df.copy()
        
        # Calculate price changes
        delta = df[price_col].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate RSI
        result_df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
        
        logger.debug(f"Calculated RSI with window {window}")
        return result_df
    
    def calculate_macd(
        self, 
        df: pd.DataFrame, 
        price_col: str = "price", 
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: Input DataFrame
            price_col: Column name for price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD columns added
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df
            
        result_df = df.copy()
        
        # Calculate fast and slow EMAs
        ema_fast = df[price_col].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        result_df["macd_line"] = ema_fast - ema_slow
        
        # Calculate signal line
        result_df["macd_signal"] = result_df["macd_line"].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        result_df["macd_histogram"] = result_df["macd_line"] - result_df["macd_signal"]
        
        logger.debug(f"Calculated MACD with periods {fast_period}/{slow_period}/{signal_period}")
        return result_df
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        price_col: str = "price", 
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: Input DataFrame
            price_col: Column name for price data
            window: Window for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            DataFrame with Bollinger Band columns added
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df
            
        result_df = df.copy()
        
        # Calculate middle band (SMA)
        result_df["bb_middle"] = df[price_col].rolling(window=window).mean()
        
        # Calculate standard deviation
        result_df["bb_std"] = df[price_col].rolling(window=window).std()
        
        # Calculate upper and lower bands
        result_df["bb_upper"] = result_df["bb_middle"] + (result_df["bb_std"] * num_std)
        result_df["bb_lower"] = result_df["bb_middle"] - (result_df["bb_std"] * num_std)
        
        # Calculate %B indicator
        result_df["bb_percent_b"] = (df[price_col] - result_df["bb_lower"]) / (result_df["bb_upper"] - result_df["bb_lower"])
        
        # Calculate bandwidth
        result_df["bb_bandwidth"] = (result_df["bb_upper"] - result_df["bb_lower"]) / result_df["bb_middle"]
        
        logger.debug(f"Calculated Bollinger Bands with window {window} and {num_std} standard deviations")
        return result_df
    
    def calculate_volatility(
        self, 
        df: pd.DataFrame, 
        return_col: Optional[str] = None,
        price_col: str = "price",
        windows: List[int] = [10, 20, 30]
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility based on returns or log returns.
        
        Args:
            df: Input DataFrame
            return_col: Column name for return data (if None, calculate from price_col)
            price_col: Column name for price data (used if return_col is None)
            windows: List of windows for volatility calculation
            
        Returns:
            DataFrame with volatility columns added
        """
        result_df = df.copy()
        
        # Use or calculate returns
        if return_col is not None and return_col in df.columns:
            returns = df[return_col]
        elif price_col in df.columns:
            returns = df[price_col].pct_change()
        else:
            logger.warning(f"Neither return column '{return_col}' nor price column '{price_col}' found in DataFrame")
            return df
        
        # Calculate rolling volatility for different windows
        for window in windows:
            result_df[f"volatility_{window}"] = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
        logger.debug(f"Calculated volatility for {len(windows)} windows")
        return result_df
    
    def __str__(self) -> str:
        """String representation of the feature extractor."""
        return f"{self.name} FeatureExtractor" 