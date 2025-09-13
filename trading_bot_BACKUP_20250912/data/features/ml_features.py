#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Feature Extractor - Generates features specifically for machine learning models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple

from trading_bot.data.features.base_feature import FeatureExtractor

logger = logging.getLogger(__name__)


class MLFeatureExtractor(FeatureExtractor):
    """
    Feature extractor specialized for machine learning models.
    
    Generates a comprehensive set of features from market data,
    including technical indicators, statistical features, and
    domain-specific transformations.
    """
    
    def __init__(
        self,
        name: str = "ml_features",
        price_col: str = "close",
        volume_col: str = "volume",
        ma_windows: List[int] = [5, 10, 20, 50, 200],
        volatility_windows: List[int] = [5, 10, 20],
        rsi_windows: List[int] = [7, 14, 21],
        include_returns: bool = True,
        return_periods: List[int] = [1, 3, 5, 10],
        include_bbands: bool = True,
        include_volume_features: bool = True,
        include_trend_features: bool = True,
        include_pattern_features: bool = False,  # More computationally expensive
        normalize_features: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ML feature extractor.
        
        Args:
            name: Name of the feature extractor
            price_col: Column name for price data
            volume_col: Column name for volume data
            ma_windows: Windows for moving average calculation
            volatility_windows: Windows for volatility calculation
            rsi_windows: Windows for RSI calculation
            include_returns: Whether to include return features
            return_periods: Periods for return calculation
            include_bbands: Whether to include Bollinger Bands
            include_volume_features: Whether to include volume-based features
            include_trend_features: Whether to include trend indicators
            include_pattern_features: Whether to include pattern recognition
            normalize_features: Whether to normalize features
            config: Additional configuration
        """
        super().__init__(name, config)
        
        self.price_col = price_col
        self.volume_col = volume_col
        self.ma_windows = ma_windows
        self.volatility_windows = volatility_windows
        self.rsi_windows = rsi_windows
        self.include_returns = include_returns
        self.return_periods = return_periods
        self.include_bbands = include_bbands
        self.include_volume_features = include_volume_features
        self.include_trend_features = include_trend_features
        self.include_pattern_features = include_pattern_features
        self.normalize_features = normalize_features
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ML-specific features from the input DataFrame.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            DataFrame with extracted features
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check required columns
        if self.price_col not in result_df.columns:
            logger.warning(f"Price column '{self.price_col}' not found in DataFrame")
            return df
            
        # Calculate price-based features
        if self.include_returns:
            result_df = self._add_return_features(result_df)
            
        # Add moving average features
        result_df = self._add_ma_features(result_df)
        
        # Add volatility features
        result_df = self._add_volatility_features(result_df)
        
        # Add RSI features
        result_df = self._add_rsi_features(result_df)
        
        # Add Bollinger Bands features
        if self.include_bbands:
            result_df = self._add_bbands_features(result_df)
            
        # Add volume features
        if self.include_volume_features and self.volume_col in df.columns:
            result_df = self._add_volume_features(result_df)
            
        # Add trend features
        if self.include_trend_features:
            result_df = self._add_trend_features(result_df)
            
        # Add pattern recognition features
        if self.include_pattern_features:
            result_df = self._add_pattern_features(result_df)
            
        # Normalize features if requested
        if self.normalize_features:
            result_df = self._normalize_features(result_df)
            
        # Drop rows with NaN values
        result_df = result_df.dropna()
        
        logger.debug(f"Extracted {len(result_df.columns) - len(df.columns)} new features")
        return result_df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        result_df = df.copy()
        
        # Simple returns
        for period in self.return_periods:
            result_df[f'return_{period}'] = df[self.price_col].pct_change(period)
            
        # Log returns
        for period in self.return_periods:
            result_df[f'log_return_{period}'] = np.log(df[self.price_col] / df[self.price_col].shift(period))
            
        # Cumulative returns
        for period in self.return_periods:
            result_df[f'cum_return_{period}'] = df[self.price_col].pct_change(period).rolling(period).sum()
            
        return result_df
    
    def _add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        result_df = df.copy()
        
        # Simple Moving Averages
        for window in self.ma_windows:
            result_df[f'sma_{window}'] = df[self.price_col].rolling(window=window).mean()
            
        # Exponential Moving Averages
        for window in self.ma_windows:
            result_df[f'ema_{window}'] = df[self.price_col].ewm(span=window, adjust=False).mean()
            
        # Moving Average Crossovers
        for i, fast_window in enumerate(self.ma_windows[:-1]):
            for slow_window in self.ma_windows[i+1:]:
                # SMA crossover
                fast_ma = result_df[f'sma_{fast_window}']
                slow_ma = result_df[f'sma_{slow_window}']
                result_df[f'sma_crossover_{fast_window}_{slow_window}'] = (fast_ma - slow_ma) / slow_ma
                
                # EMA crossover
                fast_ema = result_df[f'ema_{fast_window}']
                slow_ema = result_df[f'ema_{slow_window}']
                result_df[f'ema_crossover_{fast_window}_{slow_window}'] = (fast_ema - slow_ema) / slow_ema
                
        # Price to MA ratios
        for window in self.ma_windows:
            result_df[f'price_to_sma_{window}'] = df[self.price_col] / result_df[f'sma_{window}']
            result_df[f'price_to_ema_{window}'] = df[self.price_col] / result_df[f'ema_{window}']
            
        return result_df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        result_df = df.copy()
        
        # Historical volatility
        for window in self.volatility_windows:
            # Daily returns
            returns = df[self.price_col].pct_change()
            
            # Standard deviation of returns (volatility)
            result_df[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # Normalized volatility (coefficient of variation)
            result_df[f'normalized_volatility_{window}'] = (
                returns.rolling(window=window).std() / 
                returns.rolling(window=window).mean().abs()
            )
            
            # Parkinson volatility estimator (uses high-low range)
            if 'high' in df.columns and 'low' in df.columns:
                high_low_ratio = np.log(df['high'] / df['low'])
                result_df[f'parkinson_volatility_{window}'] = (
                    np.sqrt(high_low_ratio.rolling(window=window).var() / (4 * np.log(2)))
                )
                
        # EWMA volatility (gives more weight to recent observations)
        for window in self.volatility_windows:
            returns = df[self.price_col].pct_change()
            result_df[f'ewma_volatility_{window}'] = returns.ewm(span=window).std()
            
        return result_df
    
    def _add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and related features."""
        result_df = df.copy()
        
        # Calculate RSI for different windows
        for window in self.rsi_windows:
            delta = df[self.price_col].diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            rsi = 100 - (100 / (1 + rs))
            
            result_df[f'rsi_{window}'] = rsi
            
            # RSI crossovers (70/30 levels)
            result_df[f'rsi_{window}_overbought'] = (rsi > 70).astype(float)
            result_df[f'rsi_{window}_oversold'] = (rsi < 30).astype(float)
            
            # RSI momentum (change in RSI)
            result_df[f'rsi_{window}_momentum'] = rsi - rsi.shift(1)
            
        return result_df
    
    def _add_bbands_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        result_df = df.copy()
        
        # Calculate Bollinger Bands
        window = 20  # Standard BB window
        std_dev = 2.0  # Standard 2-sigma bands
        
        # Middle band (20-day SMA)
        middle_band = df[self.price_col].rolling(window=window).mean()
        
        # Calculate standard deviation
        std = df[self.price_col].rolling(window=window).std()
        
        # Upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Add bands to result
        result_df['bb_middle'] = middle_band
        result_df['bb_upper'] = upper_band
        result_df['bb_lower'] = lower_band
        
        # Calculate %B indicator (position within bands)
        result_df['bb_percent_b'] = (df[self.price_col] - lower_band) / (upper_band - lower_band)
        
        # Calculate bandwidth
        result_df['bb_bandwidth'] = (upper_band - lower_band) / middle_band
        
        # Add band touch/cross features
        result_df['bb_upper_touch'] = (df[self.price_col] >= upper_band).astype(float)
        result_df['bb_lower_touch'] = (df[self.price_col] <= lower_band).astype(float)
        
        return result_df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        result_df = df.copy()
        
        # Volume moving averages
        for window in self.ma_windows:
            result_df[f'volume_sma_{window}'] = df[self.volume_col].rolling(window=window).mean()
            
        # Volume relative to moving average
        for window in self.ma_windows:
            result_df[f'volume_ratio_{window}'] = df[self.volume_col] / result_df[f'volume_sma_{window}']
            
        # Volume momentum
        for period in [1, 3, 5]:
            result_df[f'volume_momentum_{period}'] = df[self.volume_col] / df[self.volume_col].shift(period)
            
        # On-balance volume (OBV)
        obv = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df[self.price_col].iloc[i] > df[self.price_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df[self.volume_col].iloc[i]
            elif df[self.price_col].iloc[i] < df[self.price_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df[self.volume_col].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        result_df['obv'] = obv
        
        # Normalized OBV (to prevent scale issues in ML models)
        result_df['obv_normalized'] = (obv - obv.rolling(window=20).min()) / (obv.rolling(window=20).max() - obv.rolling(window=20).min())
        
        # Price-volume trend
        result_df['price_volume_trend'] = (
            (df[self.price_col].pct_change() * df[self.volume_col]).cumsum()
        )
        
        return result_df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features."""
        result_df = df.copy()
        
        # MACD
        ema12 = df[self.price_col].ewm(span=12, adjust=False).mean()
        ema26 = df[self.price_col].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        result_df['macd_line'] = macd_line
        result_df['macd_signal'] = signal_line
        result_df['macd_histogram'] = macd_line - signal_line
        result_df['macd_crossover'] = ((macd_line > signal_line) & 
                                      (macd_line.shift(1) <= signal_line.shift(1))).astype(float)
        
        # ADX (Average Directional Index) - simplified calculation
        if all(col in df.columns for col in ['high', 'low']):
            # True Range
            tr1 = abs(df['high'] - df['low'])
            tr2 = abs(df['high'] - df[self.price_col].shift(1))
            tr3 = abs(df['low'] - df[self.price_col].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            plus_dm = df['high'] - df['high'].shift(1)
            minus_dm = df['low'].shift(1) - df['low']
            
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # Smoothed True Range and Directional Movement
            period = 14
            smoothed_tr = tr.rolling(window=period).sum()
            smoothed_plus_dm = plus_dm.rolling(window=period).sum()
            smoothed_minus_dm = minus_dm.rolling(window=period).sum()
            
            # Directional Indicators
            plus_di = 100 * smoothed_plus_dm / smoothed_tr
            minus_di = 100 * smoothed_minus_dm / smoothed_tr
            
            # Directional Index
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Average Directional Index
            adx = dx.rolling(window=period).mean()
            
            result_df['plus_di'] = plus_di
            result_df['minus_di'] = minus_di
            result_df['adx'] = adx
            
        # Trend detection based on multiple MA crossovers
        fast_ema = df[self.price_col].ewm(span=10, adjust=False).mean()
        medium_ema = df[self.price_col].ewm(span=50, adjust=False).mean()
        slow_ema = df[self.price_col].ewm(span=200, adjust=False).mean()
        
        # Trend strength indicators
        result_df['trend_fast_medium'] = (fast_ema - medium_ema) / medium_ema
        result_df['trend_fast_slow'] = (fast_ema - slow_ema) / slow_ema
        result_df['trend_medium_slow'] = (medium_ema - slow_ema) / slow_ema
        
        # Simple trend direction
        result_df['trend_direction'] = np.sign(result_df['trend_medium_slow'])
        
        # Trend consistency (all EMAs aligned)
        result_df['trend_consistency'] = (
            (np.sign(result_df['trend_fast_medium']) == np.sign(result_df['trend_fast_slow'])) & 
            (np.sign(result_df['trend_fast_slow']) == np.sign(result_df['trend_medium_slow']))
        ).astype(float)
        
        return result_df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        result_df = df.copy()
        
        # Check for necessary columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Cannot add pattern features - missing required OHLC columns")
            return result_df
            
        # Simple patterns
        
        # Doji (open and close are very close)
        doji_threshold = 0.001  # 0.1% difference
        result_df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < doji_threshold).astype(float)
        
        # Hammer (long lower shadow, small body at the top, little/no upper shadow)
        result_df['hammer'] = (
            # Body is in the upper part of the range
            (df['close'] > df['low'] + 0.7 * (df['high'] - df['low'])) &
            (df['open'] > df['low'] + 0.7 * (df['high'] - df['low'])) &
            # Small body
            (abs(df['close'] - df['open']) < 0.3 * (df['high'] - df['low'])) &
            # Almost no upper shadow
            ((df['high'] - df[['open', 'close']].max(axis=1)) < 0.1 * (df['high'] - df['low']))
        ).astype(float)
        
        # Shooting star (long upper shadow, small body at the bottom, little/no lower shadow)
        result_df['shooting_star'] = (
            # Body is in the lower part of the range
            (df['close'] < df['low'] + 0.3 * (df['high'] - df['low'])) &
            (df['open'] < df['low'] + 0.3 * (df['high'] - df['low'])) &
            # Small body
            (abs(df['close'] - df['open']) < 0.3 * (df['high'] - df['low'])) &
            # Almost no lower shadow
            ((df[['open', 'close']].min(axis=1) - df['low']) < 0.1 * (df['high'] - df['low']))
        ).astype(float)
        
        # Engulfing patterns
        # Bullish engulfing
        result_df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is red
            (df['close'] > df['open']) &  # Current candle is green
            (df['open'] <= df['close'].shift(1)) &  # Current open lower than previous close
            (df['close'] >= df['open'].shift(1))  # Current close higher than previous open
        ).astype(float)
        
        # Bearish engulfing
        result_df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is green
            (df['close'] < df['open']) &  # Current candle is red
            (df['open'] >= df['close'].shift(1)) &  # Current open higher than previous close
            (df['close'] <= df['open'].shift(1))  # Current close lower than previous open
        ).astype(float)
        
        # Support and resistance using pivot points
        # Look back 5 candles
        window = 5
        
        # Pivot highs (peaks)
        pivot_high = df['high'].rolling(window=window*2+1, center=True).apply(
            lambda x: 1 if x.iloc[window] == max(x) else 0, raw=True
        )
        
        # Pivot lows (troughs)
        pivot_low = df['low'].rolling(window=window*2+1, center=True).apply(
            lambda x: 1 if x.iloc[window] == min(x) else 0, raw=True
        )
        
        result_df['pivot_high'] = pivot_high
        result_df['pivot_low'] = pivot_low
        
        return result_df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features to improve ML model performance.
        Uses Z-score normalization (mean=0, std=1).
        """
        result_df = df.copy()
        
        # Identify columns to normalize (numerical except for original data)
        original_cols = set(['open', 'high', 'low', 'close', 'volume', 'timestamp'])
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        cols_to_normalize = [col for col in numeric_cols if col not in original_cols]
        
        # Apply Z-score normalization
        for col in cols_to_normalize:
            mean = result_df[col].mean()
            std = result_df[col].std()
            
            if std > 0:  # Avoid division by zero
                result_df[col] = (result_df[col] - mean) / std
                
        return result_df 