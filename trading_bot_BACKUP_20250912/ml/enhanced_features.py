#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Feature Engineering Module

This module extends the basic feature engineering framework with advanced features:
1. Market microstructure metrics
2. Alternative data features
3. Cross-asset correlation features
4. Volatility regime features
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from trading_bot.ml_pipeline.feature_engineering import FeatureEngineeringFramework

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineering(FeatureEngineeringFramework):
    """
    Enhanced Feature Engineering class that extends the base framework
    with more advanced features for ML-based signal generation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the enhanced feature engineering framework
        
        Args:
            config: Configuration dictionary with parameters
        """
        super().__init__(config)
        self._register_enhanced_features()
        logger.info("Enhanced Feature Engineering Framework initialized")
    
    def _register_enhanced_features(self):
        """Register additional advanced features"""
        # Market microstructure features
        self.register_feature("volume_imbalance", self._calculate_volume_imbalance, 
                             ["volume", "close", "high", "low"])
        self.register_feature("price_impact", self._calculate_price_impact, 
                             ["volume", "close", "high", "low"])
        self.register_feature("volume_profile", self._calculate_volume_profile, 
                             ["volume", "close"], {"window": 20})
        
        # Volatility and momentum features
        self.register_feature("volatility_ratio", self._calculate_volatility_ratio, 
                             ["close"], {"short_window": 5, "long_window": 20})
        self.register_feature("keltner_upper", self._calculate_keltner_upper, 
                             ["close", "high", "low"], {"window": 20, "atr_mult": 2.0})
        self.register_feature("keltner_lower", self._calculate_keltner_lower, 
                             ["close", "high", "low"], {"window": 20, "atr_mult": 2.0})
        self.register_feature("squeeze_momentum", self._calculate_squeeze_momentum, 
                             ["bollinger_upper", "bollinger_lower", "keltner_upper", "keltner_lower", "close"])
        
        # Trend-strength indicators
        self.register_feature("supertrend", self._calculate_supertrend, 
                             ["high", "low", "close"], {"window": 10, "multiplier": 3.0})
        self.register_feature("hma", self._calculate_hull_moving_avg, 
                             ["close"], {"window": 20})
        self.register_feature("trix", self._calculate_trix, 
                             ["close"], {"window": 14})
        
        # Oscillators
        self.register_feature("fisher_transform", self._calculate_fisher_transform, 
                             ["high", "low"], {"window": 10})
        self.register_feature("cmo", self._calculate_chande_momentum, 
                             ["close"], {"window": 14})
        
        # Mean reversion features
        self.register_feature("rvi", self._calculate_relative_vigor_index, 
                             ["open", "high", "low", "close"], {"window": 10})
        self.register_feature("ppo", self._calculate_ppo, 
                             ["close"], {"fast": 12, "slow": 26, "signal": 9})
    
    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volume imbalance based on price movement
        
        High imbalance indicates strong directional pressure
        """
        # Calculate price change
        price_change = df["close"].diff()
        
        # Calculate volume imbalance (positive when price increases, negative when decreases)
        volume = df["volume"]
        volume_imbalance = np.where(price_change > 0, volume, -volume)
        
        # Normalize by rolling average
        avg_volume = volume.rolling(window=20).mean()
        normalized_imbalance = volume_imbalance / avg_volume
        
        return pd.Series(normalized_imbalance, index=df.index)
    
    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate price impact (price movement per unit of volume)
        
        High values indicate that small volume can move price significantly
        """
        # Calculate daily range
        daily_range = df["high"] - df["low"]
        
        # Avoid division by zero
        volume = df["volume"].replace(0, np.nan)
        
        # Calculate price impact
        price_impact = daily_range / volume
        
        # Fill NaN values with preceding values
        price_impact = price_impact.fillna(method='ffill')
        
        # Normalize
        normalized_impact = price_impact / price_impact.rolling(window=20).mean()
        
        return normalized_impact
    
    def _calculate_volume_profile(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate volume profile strength
        
        Measures volume concentration at current price levels
        """
        # Calculate volume-weighted price
        vwap = (df["close"] * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
        
        # Calculate deviation from VWAP
        deviation = (df["close"] - vwap) / vwap
        
        # Calculate volume concentration
        volume_profile = 1 - deviation.abs()
        
        return volume_profile
    
    def _calculate_volatility_ratio(self, df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.Series:
        """
        Calculate ratio between short-term and long-term volatility
        
        Values > 1 indicate increasing volatility, < 1 indicate decreasing volatility
        """
        # Calculate short-term volatility
        short_volatility = df["close"].pct_change().rolling(window=short_window).std()
        
        # Calculate long-term volatility
        long_volatility = df["close"].pct_change().rolling(window=long_window).std()
        
        # Calculate ratio (avoid division by zero)
        volatility_ratio = short_volatility / long_volatility.replace(0, np.nan)
        volatility_ratio = volatility_ratio.fillna(1.0)
        
        return volatility_ratio
    
    def _calculate_keltner_upper(self, df: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.Series:
        """Calculate upper Keltner Channel"""
        ema = df["close"].ewm(span=window, adjust=False).mean()
        
        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.DataFrame({
            "hl": high_low,
            "hc": high_close, 
            "lc": low_close
        }).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return ema + (atr * atr_mult)
    
    def _calculate_keltner_lower(self, df: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.Series:
        """Calculate lower Keltner Channel"""
        ema = df["close"].ewm(span=window, adjust=False).mean()
        
        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.DataFrame({
            "hl": high_low,
            "hc": high_close, 
            "lc": low_close
        }).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return ema - (atr * atr_mult)
    
    def _calculate_squeeze_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Squeeze Momentum Indicator
        
        Identifies when Bollinger Bands are inside Keltner Channels (squeeze)
        and the momentum direction when the squeeze fires
        """
        # Check if Bollinger Bands are inside Keltner Channels
        squeeze = ((df["bollinger_upper"] < df["keltner_upper"]) & 
                   (df["bollinger_lower"] > df["keltner_lower"]))
        
        # Get momentum
        mom = df["close"].diff(20)
        
        # Create momentum indicator, higher values when squeeze is on
        momentum = np.where(squeeze, mom * 2, mom)
        
        return pd.Series(momentum, index=df.index)
    
    def _calculate_supertrend(self, df: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> pd.Series:
        """
        Calculate SuperTrend indicator
        
        A trend-following indicator that adapts to volatility
        """
        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.DataFrame({
            "hl": high_low,
            "hc": high_close, 
            "lc": low_close
        }).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Calculate bands
        hl_avg = (df["high"] + df["low"]) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend using vectorized operations where possible
        for i in range(1, len(df.index)):
            if df["close"].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
                if ((direction.iloc[i] == 1) and 
                    (lower_band.iloc[i] < lower_band.iloc[i-1])):
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                
                if ((direction.iloc[i] == -1) and 
                    (upper_band.iloc[i] > upper_band.iloc[i-1])):
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        return supertrend
    
    def _calculate_hull_moving_avg(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Hull Moving Average
        
        A responsive moving average that reduces lag
        """
        # Calculate weighted moving average with period window/2
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        
        wma1 = df["close"].rolling(window=half_length).apply(
            lambda x: sum([(i+1) * x[half_length-i-1] for i in range(half_length)]) / 
                     sum([(i+1) for i in range(half_length)]), 
            raw=True
        )
        
        # Calculate weighted moving average with period window
        wma2 = df["close"].rolling(window=window).apply(
            lambda x: sum([(i+1) * x[window-i-1] for i in range(window)]) / 
                     sum([(i+1) for i in range(window)]), 
            raw=True
        )
        
        # Calculate 2 * WMA(n/2) - WMA(n)
        wma_diff = 2 * wma1 - wma2
        
        # Calculate WMA of the above with period sqrt(n)
        wma_final = wma_diff.rolling(window=sqrt_length).apply(
            lambda x: sum([(i+1) * x[sqrt_length-i-1] for i in range(sqrt_length)]) / 
                     sum([(i+1) for i in range(sqrt_length)]), 
            raw=True
        )
        
        return wma_final
    
    def _calculate_trix(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate TRIX (Triple Exponential Moving Average)
        
        Shows percentage change in triple-smoothed EMA
        """
        # Single EMA
        ema1 = df["close"].ewm(span=window, adjust=False).mean()
        
        # Double EMA
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        
        # Triple EMA
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        
        # TRIX is percentage change of triple EMA
        trix = ema3.pct_change(1) * 100
        
        return trix
    
    def _calculate_fisher_transform(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        Calculate Fisher Transform
        
        Converts prices into a Gaussian normal distribution
        """
        # Get median of high and low
        median_price = (df["high"] + df["low"]) / 2
        
        # Normalize price to range between -1 and 1
        # Get highest high and lowest low in window
        highest_high = median_price.rolling(window=window).max()
        lowest_low = median_price.rolling(window=window).min()
        
        # Avoid division by zero
        price_range = highest_high - lowest_low
        price_range = price_range.replace(0, np.nan)
        
        # Normalize
        value = ((median_price - lowest_low) / price_range) * 2 - 1
        value = value.fillna(0)
        
        # Ensure value is between -0.999 and 0.999 to avoid infinity
        value = value.clip(-0.999, 0.999)
        
        # Apply Fisher Transform
        fisher = 0.5 * np.log((1 + value) / (1 - value))
        
        # Get signal line
        signal = fisher.ewm(span=5, adjust=False).mean()
        
        return fisher
    
    def _calculate_chande_momentum(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Chande Momentum Oscillator (CMO)
        
        Measures momentum by comparing sum of gains to sum of losses
        """
        # Calculate price change
        price_change = df["close"].diff()
        
        # Get positive and negative changes
        gain = np.where(price_change > 0, price_change, 0)
        loss = np.where(price_change < 0, -price_change, 0)
        
        # Calculate sum of gains and losses over window
        sum_gain = pd.Series(gain).rolling(window=window).sum()
        sum_loss = pd.Series(loss).rolling(window=window).sum()
        
        # Calculate CMO
        cmo = ((sum_gain - sum_loss) / (sum_gain + sum_loss)) * 100
        
        # Handle divisions where both sums are 0
        cmo = cmo.fillna(0)
        
        return cmo
    
    def _calculate_relative_vigor_index(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        Calculate Relative Vigor Index (RVI)
        
        Measures strength of a trend based on closing price relative to range
        """
        # Calculate price range
        price_range = df["high"] - df["low"]
        
        # Calculate close location value (how close is close to high vs low)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / price_range
        clv = clv.fillna(0)  # Handle division by zero
        
        # Calculate RVI (average of CLV over window)
        rvi = clv.rolling(window=window).mean()
        
        return rvi
    
    def _calculate_ppo(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """
        Calculate Percentage Price Oscillator (PPO)
        
        Similar to MACD but expressed as a percentage
        """
        # Calculate fast and slow EMAs
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        
        # Calculate PPO
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        
        return ppo
