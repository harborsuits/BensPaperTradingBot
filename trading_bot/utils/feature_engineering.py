#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering Module for Trading Strategies

This module provides comprehensive feature engineering capabilities for trading strategies,
including technical indicators, transformations, and preprocessing functions.
"""

import numpy as np
import pandas as pd
import ta
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats

try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class FeatureEngineering:
    """
    Feature engineering class for creating and managing features for ML models.
    
    This class provides:
    1. Technical indicator calculation
    2. Feature transformations and normalization
    3. Feature selection and dimensionality reduction
    4. Time-based features
    5. Price pattern features
    6. Cross-asset features
    7. Market regime detection
    8. Adaptive feature generation
    9. GPU-optimized processing (when available)
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the feature engineering module.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.feature_scaler = None
        self.reference_features = None
        self.selected_features = None
        self.pca_model = None
        self.feature_masks = {}
        self.market_regime = "unknown"
        
        # Save configuration snapshot for reproducibility
        self._save_config_snapshot()
        
    def _save_config_snapshot(self):
        """Save a snapshot of configuration for reproducibility."""
        self.config_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'params': {k: str(v) if isinstance(v, (np.ndarray, pd.DataFrame)) else v 
                      for k, v in self.params.items() if k != 'market_data'},
            'version': '1.0.0'
        }
        
        # Save to disk if specified
        if self.params.get('save_config_snapshot', False):
            output_dir = self.params.get('output_dir', './output')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'feature_config_{timestamp}.json')
            
            with open(filename, 'w') as f:
                json.dump(self.config_snapshot, f, indent=2, default=str)
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from raw price/volume data.
        
        Args:
            df: DataFrame with price/volume data
            
        Returns:
            DataFrame with calculated features
        """
        if df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Make sure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close']
        
        # Check lowercase and uppercase variants
        for col in required_columns:
            if col not in result.columns and col.upper() in result.columns:
                result[col] = result[col.upper()]
        
        missing_columns = [col for col in required_columns if col not in result.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Try to use GPU if available and enabled
        use_gpu = self.params.get('use_gpu', False) and GPU_AVAILABLE
        if use_gpu:
            result = self._to_gpu(result)
            
        # Detect market regime first to guide feature generation
        if self.params.get('detect_market_regime', False):
            self._detect_market_regime(result)
            
        # Add technical indicators based on parameter settings
        if self.params.get('include_technicals', True):
            self._add_technical_indicators(result)
        
        # Add price pattern features
        if self.params.get('include_price_patterns', False):
            self._add_price_patterns(result)
        
        # Add time-based features
        if self.params.get('include_time_features', False):
            self._add_time_features(result)
        
        # Add custom features
        if self.params.get('include_custom_features', False):
            self._add_custom_features(result)
        
        # Add regime-specific features
        if self.params.get('include_regime_features', False):
            self._add_regime_specific_features(result)
            
        # Add lag features if enabled
        if self.params.get('include_lags', False):
            result = self.add_lag_features(result)
        
        # Drop non-feature columns like OHLCV
        features_df = result.drop(['open', 'high', 'low', 'close', 'volume'], 
                                 errors='ignore', axis=1)
        
        # Handle missing data - forward fill first, then backfill remaining
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Apply feature selection if configured
        if self.params.get('feature_selection', 'none') != 'none':
            features_df = self._select_features(features_df)
        
        # Apply feature masking if enabled
        if self.params.get('apply_feature_masks', False):
            features_df = self._apply_feature_masks(features_df)
        
        # Apply dimensionality reduction if configured
        if self.params.get('add_pca_components', False):
            features_df = self._add_pca_components(features_df)
        
        # Apply normalization if configured
        if self.params.get('normalize_features', True):
            features_df = self._normalize_features(features_df)
            
        # Convert back from GPU if necessary
        if use_gpu:
            features_df = self._from_gpu(features_df)
        
        return features_df
    
    def _to_gpu(self, df: pd.DataFrame) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """Convert pandas DataFrame to cuDF DataFrame if GPU is available."""
        if GPU_AVAILABLE:
            try:
                return cudf.DataFrame.from_pandas(df)
            except Exception as e:
                import logging
                logging.warning(f"Failed to convert to GPU DataFrame: {str(e)}")
        return df
    
    def _from_gpu(self, df: Union[pd.DataFrame, 'cudf.DataFrame']) -> pd.DataFrame:
        """Convert cuDF DataFrame back to pandas DataFrame."""
        if GPU_AVAILABLE and not isinstance(df, pd.DataFrame):
            try:
                return df.to_pandas()
            except Exception as e:
                import logging
                logging.warning(f"Failed to convert from GPU DataFrame: {str(e)}")
        return df
        
    def _detect_market_regime(self, df: pd.DataFrame) -> None:
        """
        Detect the current market regime with enhanced specificity for regime-specialist models.
        
        Args:
            df: DataFrame with price data
            
        Sets the market_regime attribute and modifies df to add regime column.
        """
        if 'close' not in df.columns or len(df) < 20:
            self.market_regime = "unknown"
            return
            
        close_prices = df['close']
        
        # Calculate metrics for regime detection
        returns = close_prices.pct_change().dropna()
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Get ATR if available, otherwise calculate volatility from returns
        if 'atr' in df.columns and 'close' in df.columns:
            volatility = df['atr'].iloc[-20:].mean() / df['close'].iloc[-20:].mean()
        else:
            volatility = returns.tail(20).std()
            
        # Get ADX if available for trend strength, otherwise use autocorrelation
        if 'adx' in df.columns:
            trend_strength = df['adx'].iloc[-1]
        else:
            # Use autocorrelation as a simple trend strength indicator
            trend_strength = returns.tail(20).autocorr() * 100
            trend_strength = abs(trend_strength) if not pd.isna(trend_strength) else 0
            
        # Hurst exponent to detect trending (>0.5) vs mean-reverting (<0.5) markets
        # Use last 100 periods or as many as available
        if len(returns) >= 30:
            try:
                hurst_exp = self._calculate_hurst_exponent(log_returns.tail(100))
            except:
                hurst_exp = 0.5  # Default to random walk if calculation fails
        else:
            hurst_exp = 0.5
            
        # Recent volatility relative to historical
        if len(returns) >= 30:
            recent_vol = returns.tail(20).std()
            historical_vol = returns.std()
            volatility_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        else:
            volatility_ratio = 1.0
            
        # Direction of recent trend
        if len(returns) >= 20:
            trend_direction = np.sign(close_prices.tail(20).mean() - close_prices.tail(40).head(20).mean())
        else:
            trend_direction = 0
        
        # Volatility threshold from parameters or default
        high_vol_threshold = self.params.get('high_vol_threshold', 0.015)
        trend_threshold = self.params.get('trend_threshold', 25)
        mean_rev_threshold = self.params.get('mean_rev_threshold', 0.4)
        
        # Determine regime with more specificity for specialist models
        if volatility_ratio > 1.5 or volatility > high_vol_threshold:
            # High volatility regimes
            if trend_strength > trend_threshold:
                regime = "high_vol_trending"
                if trend_direction > 0:
                    regime = "high_vol_uptrend"
                else:
                    regime = "high_vol_downtrend"
            else:
                regime = "high_vol"
        elif hurst_exp < mean_rev_threshold:
            # Mean-reverting (range-bound) market
            regime = "range_bound"
        elif hurst_exp > 0.6:
            # Trending market
            if trend_direction > 0:
                regime = "uptrend"
            else:
                regime = "downtrend"
        else:
            # Random walk / no clear regime
            regime = "random_walk"
            
        # Store the detected regime
        self.market_regime = regime
        
        # Add to dataframe
        df['market_regime'] = regime
        
        # Add regime numeric mapping for ML models
        regime_map = {
            "high_vol": 0,
            "high_vol_trending": 1,
            "high_vol_uptrend": 2,
            "high_vol_downtrend": 3,
            "range_bound": 4,
            "uptrend": 5,
            "downtrend": 6,
            "random_walk": 7,
            "unknown": -1
        }
        df['market_regime_numeric'] = regime_map.get(regime, -1)
    
    def _calculate_hurst_exponent(self, time_series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent of a time series.
        
        Args:
            time_series: Input time series
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent value
        """
        lags = range(2, min(max_lag, len(time_series) // 4))
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        
        # Linear fit on log-log plot
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst exponent is the slope
        hurst = m[0]
        
        return hurst
            
    def _add_regime_specific_features(self, df: pd.DataFrame) -> None:
        """
        Add features specific to the detected market regime.
        
        Args:
            df: DataFrame with price data
            
        Modifies df in place.
        """
        if self.market_regime == "unknown" or 'close' not in df.columns:
            return
            
        if self.market_regime == "trending" or self.market_regime == "downtrend":
            # Add momentum-focused features for trending markets
            if 'adx' not in df.columns and 'high' in df.columns and 'low' in df.columns:
                df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
                
            # Add trend strength metrics
            for period in [10, 20, 50]:
                if f'ema_{period}' in df.columns:
                    # Normalized distance from EMA
                    df[f'trend_strength_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
                    
                    # Measure EMA slopes
                    df[f'ema_{period}_slope'] = df[f'ema_{period}'].pct_change(3) * 100
                    
        elif self.market_regime == "mean_reverting":
            # Add mean-reversion focused features
            if 'close' in df.columns:
                for period in [5, 10, 20]:
                    # Calculate deviation from moving average
                    ma_col = f'sma_{period}'
                    if ma_col in df.columns:
                        df[f'mean_rev_{period}'] = (df['close'] - df[ma_col]) / df[ma_col] * 100
                        # Oscillator indicator using z-score
                        df[f'zscore_{period}'] = (df['close'] - df[ma_col]) / df['close'].rolling(period).std()
                    
        elif self.market_regime == "high_volatility":
            # Add volatility-focused features
            if 'atr' in df.columns:
                # Normalized ATR
                df['atr_norm'] = df['atr'] / df['close'] * 100
                
                # ATR acceleration
                df['atr_change'] = df['atr'].pct_change(5) * 100
                
            # Add volatility-adjusted indicators
            for indicator in ['rsi_14', 'cci', 'willr']:
                if indicator in df.columns and 'atr' in df.columns:
                    df[f'{indicator}_vol_adj'] = df[indicator] / df['atr_norm']
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Modifies df in place.
        """
        # Make sure column names are standardized for the ta library
        price_cols = {}
        for std_col, possible_cols in {
            'open': ['open', 'Open', 'OPEN'],
            'high': ['high', 'High', 'HIGH'],
            'low': ['low', 'Low', 'LOW'],
            'close': ['close', 'Close', 'CLOSE'],
            'volume': ['volume', 'Volume', 'VOLUME']
        }.items():
            for col in possible_cols:
                if col in df.columns:
                    price_cols[std_col] = col
                    break
        
        # Create a copy with standardized column names if needed
        if set(price_cols.keys()) != set(price_cols.values()):
            temp_df = pd.DataFrame()
            for std_col, actual_col in price_cols.items():
                temp_df[std_col] = df[actual_col]
        else:
            temp_df = df
        
        ta_feature_sets = self.params.get('ta_feature_sets', ['momentum', 'trend', 'volatility'])
        lookback_periods = self.params.get('lookback_periods', [5, 10, 20, 50])
        
        # Add trend indicators
        if 'trend' in ta_feature_sets:
            # Simple Moving Averages
            for period in lookback_periods:
                df[f'sma_{period}'] = ta.trend.sma_indicator(temp_df['close'], window=period)
                
                # Calculate price relative to SMA as a ratio
                df[f'close_to_sma_{period}'] = temp_df['close'] / df[f'sma_{period}']
            
            # Exponential Moving Averages
            for period in lookback_periods:
                df[f'ema_{period}'] = ta.trend.ema_indicator(temp_df['close'], window=period)
                
                # Calculate price relative to EMA as a ratio
                df[f'close_to_ema_{period}'] = temp_df['close'] / df[f'ema_{period}']
            
            # Moving Average Convergence Divergence (MACD)
            macd = ta.trend.MACD(temp_df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # Average Directional Index (ADX) - Trend strength
            adx = ta.trend.ADXIndicator(temp_df['high'], temp_df['low'], temp_df['close'])
            df['adx'] = adx.adx()
            
            # Additional trend indicators
            df['cci'] = ta.trend.cci(temp_df['high'], temp_df['low'], temp_df['close'])
            df['dpo'] = ta.trend.dpo(temp_df['close'])
            
            # Ichimoku Cloud components
            ichimoku = ta.trend.IchimokuIndicator(temp_df['high'], temp_df['low'])
            df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
            df['kijun_sen'] = ichimoku.ichimoku_base_line()
            df['senkou_span_a'] = ichimoku.ichimoku_a()
            df['senkou_span_b'] = ichimoku.ichimoku_b()
            df['chikou_span'] = temp_df['close'].shift(-26)
        
        # Add momentum indicators
        if 'momentum' in ta_feature_sets:
            # Relative Strength Index (RSI)
            df['rsi_14'] = ta.momentum.rsi(temp_df['close'], window=14)
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(temp_df['high'], temp_df['low'], temp_df['close'])
            df['slowk'] = stoch.stoch()
            df['slowd'] = stoch.stoch_signal()
            
            # Williams %R
            df['willr'] = ta.momentum.williams_r(temp_df['high'], temp_df['low'], temp_df['close'])
            
            # Rate of Change (ROC)
            for period in lookback_periods:
                df[f'roc_{period}'] = ta.momentum.roc(temp_df['close'], window=period)
            
            # Money Flow Index (MFI) - requires volume
            if 'volume' in temp_df.columns:
                df['mfi'] = ta.volume.money_flow_index(
                    temp_df['high'], 
                    temp_df['low'], 
                    temp_df['close'], 
                    temp_df['volume'],
                    window=14
                )
        
        # Add volatility indicators
        if 'volatility' in ta_feature_sets:
            # Average True Range (ATR)
            atr = ta.volatility.AverageTrueRange(
                temp_df['high'], temp_df['low'], temp_df['close'], window=14
            )
            df['atr'] = atr.average_true_range()
            df['atr_percent'] = df['atr'] / temp_df['close'] * 100
            
            # Bollinger Bands
            for period in [20]:  # Standard Bollinger Band period
                bollinger = ta.volatility.BollingerBands(
                    temp_df['close'], window=period, window_dev=2
                )
                df[f'bb_upper_{period}'] = bollinger.bollinger_hband()
                df[f'bb_middle_{period}'] = bollinger.bollinger_mavg()
                df[f'bb_lower_{period}'] = bollinger.bollinger_lband()
                df[f'bb_width_{period}'] = bollinger.bollinger_wband()
                df[f'bb_position_{period}'] = bollinger.bollinger_pband()
            
            # Normalized historical volatility
            for period in lookback_periods:
                returns = pd.Series(temp_df['close']).pct_change(1)
                df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            # Keltner Channels
            for period in [20]:
                keltner = ta.volatility.KeltnerChannel(
                    temp_df['high'], temp_df['low'], temp_df['close'], window=period
                )
                df[f'keltner_middle_{period}'] = keltner.keltner_channel_mband()
                df[f'keltner_upper_{period}'] = keltner.keltner_channel_hband()
                df[f'keltner_lower_{period}'] = keltner.keltner_channel_lband()
        
        # Add volume indicators
        if 'volume' in ta_feature_sets and 'volume' in temp_df.columns:
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.on_balance_volume(temp_df['close'], temp_df['volume'])
            
            # Volume Oscillator
            df['short_vol_ma'] = pd.Series(temp_df['volume']).rolling(window=5).mean()
            df['long_vol_ma'] = pd.Series(temp_df['volume']).rolling(window=10).mean()
            df['vol_osc'] = ((df['short_vol_ma'] - df['long_vol_ma']) / df['long_vol_ma']) * 100
            
            # Chaikin Money Flow
            df['cmf'] = ta.volume.chaikin_money_flow(
                temp_df['high'], temp_df['low'], temp_df['close'], temp_df['volume']
            )
            
            # Ease of Movement
            df['eom'] = ta.volume.ease_of_movement(
                temp_df['high'], temp_df['low'], temp_df['volume']
            )
            df['eom_ma'] = pd.Series(df['eom']).rolling(window=14).mean()
            
            # Volume-price trend
            df['pvt'] = ta.volume.volume_price_trend(temp_df['close'], temp_df['volume'])
    
    def _add_price_patterns(self, df: pd.DataFrame) -> None:
        """
        Add price pattern recognition features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Modifies df in place.
        """
        # Use pandas to detect some basic candlestick patterns
        # We won't have all the patterns talib provides, but we can implement some manually
        
        # Ensure we have enough data
        if len(df) < 5:
            return
            
        # Extract price data for easier access
        open_prices = df['open']
        high_prices = df['high']
        low_prices = df['low']
        close_prices = df['close']
        
        # Doji pattern (open and close are almost the same)
        df['cdl_doji'] = ((abs(open_prices - close_prices) / (high_prices - low_prices + 0.001)) < 0.1).astype(int)
        
        # Hammer pattern (small body, long lower shadow, little/no upper shadow)
        body_size = abs(close_prices - open_prices)
        lower_shadow = pd.Series(np.minimum(open_prices, close_prices) - low_prices)
        upper_shadow = pd.Series(high_prices - np.maximum(open_prices, close_prices))
        
        df['cdl_hammer'] = (
            (body_size / (high_prices - low_prices + 0.001) < 0.3) & 
            (lower_shadow > 2 * body_size) & 
            (upper_shadow < 0.1 * lower_shadow)
        ).astype(int)
        
        # Shooting Star (small body, long upper shadow, little/no lower shadow)
        df['cdl_shooting_star'] = (
            (body_size / (high_prices - low_prices + 0.001) < 0.3) & 
            (upper_shadow > 2 * body_size) & 
            (lower_shadow < 0.1 * upper_shadow)
        ).astype(int)
        
        # Engulfing patterns
        bull_engulf = (
            (close_prices.shift(1) < open_prices.shift(1)) &  # Previous day was bearish
            (close_prices > open_prices) &  # Current day is bullish
            (open_prices < close_prices.shift(1)) &  # Open below previous close
            (close_prices > open_prices.shift(1))  # Close above previous open
        )
        
        bear_engulf = (
            (close_prices.shift(1) > open_prices.shift(1)) &  # Previous day was bullish
            (close_prices < open_prices) &  # Current day is bearish
            (open_prices > close_prices.shift(1)) &  # Open above previous close
            (close_prices < open_prices.shift(1))  # Close below previous open
        )
        
        df['cdl_engulfing'] = bull_engulf.astype(int) - bear_engulf.astype(int)
        
        # Harami patterns (opposite of engulfing - smaller candle contained within previous)
        bull_harami = (
            (close_prices.shift(1) < open_prices.shift(1)) &  # Previous day was bearish
            (close_prices > open_prices) &  # Current day is bullish
            (open_prices > close_prices.shift(1)) &  # Open above previous close
            (close_prices < open_prices.shift(1))  # Close below previous open
        )
        
        bear_harami = (
            (close_prices.shift(1) > open_prices.shift(1)) &  # Previous day was bullish
            (close_prices < open_prices) &  # Current day is bearish
            (open_prices < close_prices.shift(1)) &  # Open below previous close
            (close_prices > open_prices.shift(1))  # Close above previous open
        )
        
        df['cdl_harami'] = bull_harami.astype(int) - bear_harami.astype(int)
        
        # Morning/Evening Star patterns are complex - simplified version
        # Morning Star (bearish, small body, bullish)
        df['cdl_morning_star'] = (
            (close_prices.shift(2) < open_prices.shift(2)) &  # First day bearish
            (abs(close_prices.shift(1) - open_prices.shift(1)) < 0.3 * abs(close_prices.shift(2) - open_prices.shift(2))) &  # Second day small
            (close_prices > open_prices) &  # Third day bullish
            (close_prices > (open_prices.shift(2) + close_prices.shift(2)) / 2)  # Recovery
        ).astype(int)
        
        # Evening Star (bullish, small body, bearish)
        df['cdl_evening_star'] = (
            (close_prices.shift(2) > open_prices.shift(2)) &  # First day bullish
            (abs(close_prices.shift(1) - open_prices.shift(1)) < 0.3 * abs(close_prices.shift(2) - open_prices.shift(2))) &  # Second day small
            (close_prices < open_prices) &  # Third day bearish
            (close_prices < (open_prices.shift(2) + close_prices.shift(2)) / 2)  # Drop
        ).astype(int)
        
        # These are simplified versions of the patterns and may not exactly match talib's implementation
        # but provide similar functionality without the dependency
    
    def _add_time_features(self, df: pd.DataFrame) -> None:
        """
        Add time-based features for seasonality.
        
        Args:
            df: DataFrame with time index
            
        Modifies df in place.
        """
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return
            
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        
        # Is month end/start
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        
        # Is quarter end/start
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        
        # Month, quarter
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclic features using sin/cos transformations for periodic features
        df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        
        df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
    
    def _add_custom_features(self, df: pd.DataFrame) -> None:
        """
        Add custom features specific to trading strategies.
        
        Args:
            df: DataFrame with price data
            
        Modifies df in place.
        """
        if 'close' not in df.columns:
            return
        
        close = df['close']
        
        # Price rate of change combinations
        for fast_period, slow_period in [(1, 5), (5, 20), (10, 50)]:
            fast_roc = close.pct_change(fast_period)
            slow_roc = close.pct_change(slow_period)
            df[f'roc_diff_{fast_period}_{slow_period}'] = fast_roc - slow_roc
            
        # Return mean deviations
        for period in [5, 10, 20, 50]:
            returns = close.pct_change()
            df[f'ret_dev_{period}'] = returns - returns.rolling(period).mean()
            
        # Rolling z-score
        for period in [20, 50]:
            rolling_mean = close.rolling(period).mean()
            rolling_std = close.rolling(period).std()
            df[f'zscore_{period}'] = (close - rolling_mean) / rolling_std
            
        # Up/down day streak
        df['price_diff'] = close.diff()
        df['up_streak'] = df['price_diff'].gt(0).astype(int)
        df['down_streak'] = df['price_diff'].lt(0).astype(int)
        
        # Fill consecutive up days
        up_mask = (df['up_streak'] == 1)
        down_mask = (df['down_streak'] == 1)
        
        streak_counter = df['up_streak'].copy()
        for i in range(1, len(df)):
            if up_mask.iloc[i]:
                streak_counter.iloc[i] = streak_counter.iloc[i-1] + 1
        df['up_streak'] = streak_counter
        
        streak_counter = df['down_streak'].copy()
        for i in range(1, len(df)):
            if down_mask.iloc[i]:
                streak_counter.iloc[i] = streak_counter.iloc[i-1] + 1
        df['down_streak'] = streak_counter
        
        # Gap metrics
        if 'open' in df.columns:
            df['gap'] = df['open'] / df['close'].shift(1) - 1
            df['gap_abs'] = df['gap'].abs()
        
        # High-Low range relative to price
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
            
            # Rolling range features
            for period in [5, 10, 20]:
                df[f'hl_range_pct_avg_{period}'] = df['hl_range_pct'].rolling(period).mean()
                df[f'hl_range_pct_dev_{period}'] = df['hl_range_pct'] / df[f'hl_range_pct_avg_{period}'] - 1
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using StandardScaler or MinMaxScaler.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        # Skip if empty
        if df.empty:
            return df
        
        # Get scaling method
        scaling_method = self.params.get('scaling_method', 'standard')
        
        # Create scaler if not already created
        if self.feature_scaler is None:
            if scaling_method == 'minmax':
                self.feature_scaler = MinMaxScaler()
            else:  # default to standard scaling
                self.feature_scaler = StandardScaler()
                
            # Fit the scaler
            self.reference_features = df.copy()
            
            # Only fit on numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            self.feature_scaler.fit(df[numeric_cols])
        
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Only normalize numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Apply the scaler
        if numeric_cols:
            normalized_df[numeric_cols] = self.feature_scaler.transform(df[numeric_cols])
        
        return normalized_df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select most important features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with selected features
        """
        # Skip if empty
        if df.empty:
            return df
        
        # Get selection method and max features
        selection_method = self.params.get('feature_selection', 'importance')
        max_features = min(self.params.get('max_features', 50), len(df.columns))
        
        # Skip if we don't have enough features
        if len(df.columns) <= max_features:
            return df
        
        # Check if we already selected features
        if self.selected_features is not None:
            # Only keep the selected features
            common_cols = [col for col in self.selected_features if col in df.columns]
            return df[common_cols]
        
        # Only consider numerical columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) == 0:
            return df
        
        if selection_method == 'statistical':
            # We need a target variable for statistical selection
            # Since we don't have it, we'll use the most recent value as a proxy
            proxy_target = df[numeric_cols].iloc[-1].values
            
            # Use f_regression with a proxy target
            selector = SelectKBest(f_regression, k=max_features)
            try:
                selector.fit(df[numeric_cols].iloc[:-1], proxy_target)
                # Get feature indices sorted by importance
                indices = np.argsort(selector.scores_)[::-1][:max_features]
                selected_cols = [numeric_cols[i] for i in indices]
                # Include non-numeric columns
                selected_cols += [col for col in df.columns if col not in numeric_cols]
                self.selected_features = selected_cols
                return df[selected_cols]
            except Exception as e:
                # Fallback to using all features
                return df
        else:
            # Default to using all features for now
            # In a real implementation, you'd use feature importances from a trained model
            return df
        
    def _add_pca_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add principal components for dimensionality reduction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with added PCA components
        """
        # Skip if empty
        if df.empty:
            return df
        
        # Get PCA parameters
        num_components = self.params.get('pca_components', 5)
        
        # Check if we have enough numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < num_components:
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Fit PCA model if not already fit
        if self.pca_model is None:
            self.pca_model = PCA(n_components=num_components)
            try:
                self.pca_model.fit(df[numeric_cols])
            except Exception as e:
                # Return original data if PCA fails
                return df
        
        # Transform the data
        pca_result = self.pca_model.transform(df[numeric_cols])
        
        # Add components to result
        for i in range(num_components):
            result_df[f'pca_{i+1}'] = pca_result[:, i]
        
        return result_df
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from a trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importances = {}
        
        try:
            # Extract from sklearn-compatible models
            if hasattr(model, 'feature_importances_'):
                # Random Forest, Gradient Boosting, etc.
                scores = model.feature_importances_
                importances = dict(zip(feature_names, scores))
            elif hasattr(model, 'coef_'):
                # Linear models
                scores = np.abs(model.coef_)
                if scores.ndim > 1:
                    scores = scores.mean(axis=0)
                importances = dict(zip(feature_names, scores))
            elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
                # GridSearchCV with RF/GB
                scores = model.best_estimator_.feature_importances_
                importances = dict(zip(feature_names, scores))
            elif hasattr(model, 'named_steps'):
                # Pipeline
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'feature_importances_'):
                        scores = step.feature_importances_
                        importances = dict(zip(feature_names, scores))
                        break
                    elif hasattr(step, 'coef_'):
                        scores = np.abs(step.coef_)
                        if scores.ndim > 1:
                            scores = scores.mean(axis=0)
                        importances = dict(zip(feature_names, scores))
                        break
        except Exception as e:
            # Return empty dict on error
            pass
        
        return importances

    def add_lag_features(self, df: pd.DataFrame, features: List[str] = None, lags: List[int] = None) -> pd.DataFrame:
        """
        Add lagged versions of important features to provide temporal context for ML models.
        
        Args:
            df: DataFrame with features
            features: List of features to create lags for (if None, use all numeric features or top features)
            lags: List of lag periods to create (defaults to [1, 3, 5, 10])
            
        Returns:
            DataFrame with added lag features
        """
        if df.empty:
            return df
            
        result_df = df.copy()
        lags = lags or self.params.get('lags', [1, 3, 5, 10])
        
        # If no features specified, use top features or all numeric features
        if features is None:
            if hasattr(self, 'selected_features') and self.selected_features:
                # Use selected features if available
                features = [f for f in self.selected_features if f in df.columns]
            else:
                # Otherwise use all numeric columns
                features = df.select_dtypes(include=np.number).columns.tolist()
                
                # Limit to top 10 if too many features to avoid explosion
                if len(features) > 10:
                    features = features[:10]
        
        # Create lag features
        for feature in features:
            if feature in result_df.columns:
                for lag in lags:
                    result_df[f'{feature}_lag_{lag}'] = result_df[feature].shift(lag)
        
        return result_df
    
    def add_context_aware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that adapt based on market context.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with added context-aware features
        """
        if df.empty or 'close' not in df.columns:
            return df
            
        result_df = df.copy()
        
        # Detect context if it hasn't been detected yet
        if self.market_regime == "unknown" and self.params.get('detect_market_regime', False):
            self._detect_market_regime(df)
            result_df['market_regime'] = self.market_regime
            
        # Calculate rolling volatility in different windows
        for window in [5, 10, 21]:
            if 'close' in result_df.columns:
                # Rolling volatility
                returns = result_df['close'].pct_change()
                result_df[f'volatility_{window}'] = returns.rolling(window).std() * 100
                
                # Volatility ratio (current / historical)
                if len(returns) > window * 2:
                    current_vol = returns.rolling(window).std().iloc[-window:]
                    hist_vol = returns.rolling(window*5).std().iloc[-window:]
                    if not hist_vol.empty and not np.isclose(hist_vol.mean(), 0):
                        result_df[f'vol_ratio_{window}'] = current_vol.mean() / hist_vol.mean()
                        
        # Calculate trend strength metrics
        for window in [5, 10, 21]:
            if 'close' in result_df.columns:
                # Calculate moving averages if not already present
                ma_col = f'sma_{window}'
                if ma_col not in result_df.columns:
                    result_df[ma_col] = ta.trend.sma_indicator(result_df['close'], window=window)
                    
                # Calculate trend strength
                result_df[f'trend_strength_{window}'] = (result_df['close'] - result_df[ma_col]) / result_df[ma_col] * 100
                
                # Calculate slope of moving average
                result_df[f'ma_slope_{window}'] = result_df[ma_col].pct_change(3) * 100
                
        # Add market condition indicators
        if 'rsi_14' in result_df.columns:
            # Market condition based on RSI
            result_df['overbought'] = (result_df['rsi_14'] > 70).astype(int)
            result_df['oversold'] = (result_df['rsi_14'] < 30).astype(int)
            
            # Market regime strength 
            if 'atr' in result_df.columns and 'atr_percent' in result_df.columns:
                # Combined trend and volatility metric
                result_df['trend_vol_signal'] = (result_df['rsi_14'] - 50) * result_df['atr_percent']
        
        return result_df
        
    def add_cross_asset_features(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add features based on correlations with other assets.
        
        Args:
            df: DataFrame with features for the primary asset
            market_data: Dictionary of DataFrames for other assets
            
        Returns:
            DataFrame with added cross-asset features
        """
        if df.empty or not market_data or 'close' not in df.columns:
            return df
            
        result_df = df.copy()
        primary_returns = result_df['close'].pct_change().fillna(0)
        
        # Set of correlation windows
        windows = self.params.get('correlation_windows', [10, 20, 60])
        
        # Add cross-asset correlation and relative strength features
        for asset_name, asset_df in market_data.items():
            if 'close' not in asset_df.columns or asset_df.empty:
                continue
                
            # Ensure the indices are aligned
            asset_df = asset_df.reindex(result_df.index, method='ffill')
            
            if asset_df['close'].isna().all():
                continue
                
            # Calculate returns for the other asset
            asset_returns = asset_df['close'].pct_change().fillna(0)
            
            # Calculate return differentials (relative strength)
            result_df[f'rel_strength_{asset_name}'] = primary_returns - asset_returns
            
            # Calculate rolling correlations
            for window in windows:
                if len(asset_returns) >= window:
                    correlation = primary_returns.rolling(window).corr(asset_returns)
                    result_df[f'corr_{asset_name}_{window}'] = correlation
                    
                    # Beta (sensitivity to the other asset's moves)
                    cov = primary_returns.rolling(window).cov(asset_returns)
                    var = asset_returns.rolling(window).var()
                    beta = cov / var.replace(0, np.nan)
                    result_df[f'beta_{asset_name}_{window}'] = beta.fillna(0)
                    
            # Add lead-lag features (does the other asset lead the primary?)
            for lag in [1, 2, 3]:
                result_df[f'{asset_name}_lead_{lag}'] = asset_returns.shift(-lag)
                result_df[f'{asset_name}_lag_{lag}'] = asset_returns.shift(lag)
                
        return result_df

    def safe_generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features with error handling and safety checks.
        
        Args:
            df: DataFrame with price/volume data
            
        Returns:
            DataFrame with calculated features
        """
        try:
            # Run the standard feature generation
            features_df = self.generate_features(df)
            
            # Add lag features if configured
            if self.params.get('include_lags', False):
                features_df = self.add_lag_features(features_df)
                
            # Add context-aware features if configured
            if self.params.get('include_context_features', False):
                features_df = self.add_context_aware_features(features_df)
                
            # Add cross-asset features if configured and market data is provided
            if self.params.get('include_cross_asset', False) and 'market_data' in self.params:
                features_df = self.add_cross_asset_features(features_df, self.params['market_data'])
                
            # Apply feature weights if configured
            if self.params.get('apply_weights', False):
                features_df = self.apply_feature_weights(features_df)
                
            # Safety: Clip extreme values to prevent outliers from breaking models
            numeric_cols = features_df.select_dtypes(include=np.number).columns
            features_df[numeric_cols] = np.clip(features_df[numeric_cols], -10, 10)
            
            # Safety: Replace any remaining infinities with NaNs, then fill NaNs
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            # Log the error
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating features: {str(e)}")
            
            # Return empty DataFrame as a fallback
            return pd.DataFrame(index=df.index)
            
    def apply_feature_weights(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Apply weights to features based on their importance.
        
        Args:
            df: DataFrame with features
            weights: Dictionary mapping feature names to importance weights
            
        Returns:
            DataFrame with weighted features
        """
        if df.empty:
            return df
            
        result_df = df.copy()
        
        # Use provided weights or default to regime-based weights
        if weights is None:
            if self.feature_masks and self.market_regime in self.feature_masks:
                weights = self.feature_masks[self.market_regime]
            else:
                # No weights available
                return result_df
                
        # Apply weights to features
        for feature, weight in weights.items():
            if feature in result_df.columns:
                result_df[feature] = result_df[feature] * weight
                
        return result_df

    def _apply_feature_masks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply relevance-aware feature masking based on market regime.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with masked features
        """
        if self.market_regime == "unknown" or df.empty:
            return df
            
        result_df = df.copy()
        
        # Define feature masks based on market regime if not already created
        if not self.feature_masks:
            self._create_feature_masks()
            
        # Apply the appropriate mask for the current regime
        mask = self.feature_masks.get(self.market_regime)
        if mask:
            # Use mask to include/exclude/weight features based on regime
            included_features = [col for col in df.columns if col in mask]
            
            # Apply feature weights for included features
            for feature in included_features:
                if feature in result_df.columns:
                    weight = mask[feature]
                    result_df[feature] = result_df[feature] * weight
                    
        return result_df
    
    def _create_feature_masks(self) -> None:
        """Create feature masks for different market regimes."""
        # These masks contain feature names and their weights (importance) for each regime
        
        # Trending market mask - emphasize momentum and trend indicators
        trending_mask = {}
        for col in ['adx', 'macd', 'macd_hist', 'cci']:
            trending_mask[col] = 1.5  # Boost importance
            
        for period in [5, 10, 20, 50]:
            trending_mask[f'roc_{period}'] = 1.3
            trending_mask[f'ema_{period}_slope'] = 1.3
            
        # Mean-reverting market mask - emphasize oscillators and overbought/oversold
        mean_reverting_mask = {}
        for col in ['rsi_14', 'willr', 'cci', 'stoch_k', 'stoch_d']:
            mean_reverting_mask[col] = 1.5
            
        for period in [5, 10, 20]:
            mean_reverting_mask[f'zscore_{period}'] = 1.4
            mean_reverting_mask[f'bb_position_{period}'] = 1.3
            
        # High volatility mask - emphasize volatility and risk metrics
        high_volatility_mask = {}
        for col in ['atr', 'atr_norm', 'atr_percent']:
            high_volatility_mask[col] = 1.5
            
        for period in [5, 10, 20]:
            high_volatility_mask[f'volatility_{period}'] = 1.4
            high_volatility_mask[f'bb_width_{period}'] = 1.3
            
        # Store masks
        self.feature_masks = {
            'trending': trending_mask,
            'downtrend': trending_mask,  # Same as trending for now
            'mean_reverting': mean_reverting_mask,
            'high_volatility': high_volatility_mask,
            'random_walk': {}  # No specific mask for random walk
        }
        
    def add_return_labels(self, df: pd.DataFrame, future_windows: List[int] = None, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Generate supervised learning labels from future returns.
        
        Args:
            df: DataFrame with close prices
            future_windows: List of prediction horizons (in periods)
            thresholds: List of return thresholds for classification
            
        Returns:
            DataFrame with added return labels
        """
        if df.empty or 'close' not in df.columns:
            return df
            
        result_df = df.copy()
        
        # Get parameters
        future_windows = future_windows or self.params.get('target_horizons', [1, 5, 10, 20])
        thresholds = thresholds or self.params.get('target_thresholds', [0.0, 0.01, 0.02])
        
        # Calculate future returns for different horizons
        for window in future_windows:
            # Regression target: future return
            future_return = df['close'].shift(-window) / df['close'] - 1
            result_df[f'future_return_{window}'] = future_return
            
            # Add rolling labels for increased stability
            if self.params.get('rolling_labels', False):
                # Calculate rolling mean of returns for more stable targets
                result_df[f'future_return_{window}_smooth'] = future_return.rolling(3, center=True).mean()
                
                # Calculate rolling volatility-adjusted returns
                rolling_std = future_return.rolling(10).std()
                result_df[f'future_return_{window}_vol_adj'] = future_return / (rolling_std + 1e-5)
            
            # Classification targets with different thresholds
            for threshold in thresholds:
                if threshold > 0:  # Skip 0 threshold in the column name
                    col_name = f'label_{window}d_{int(threshold*100)}pct'
                else:
                    col_name = f'label_{window}d'
                    
                result_df[col_name] = 0  # neutral by default
                result_df.loc[future_return > threshold, col_name] = 1  # buy signal
                result_df.loc[future_return < -threshold, col_name] = -1  # sell signal
                
                # Add regime-conditioned labels if regime detection is enabled
                if self.params.get('detect_market_regime', False) and 'market_regime_numeric' in result_df.columns:
                    for regime_code in result_df['market_regime_numeric'].unique():
                        regime_mask = result_df['market_regime_numeric'] == regime_code
                        result_df[f'{col_name}_regime_{int(regime_code)}'] = 0
                        result_df.loc[regime_mask & (future_return > threshold), f'{col_name}_regime_{int(regime_code)}'] = 1
                        result_df.loc[regime_mask & (future_return < -threshold), f'{col_name}_regime_{int(regime_code)}'] = -1
        
        return result_df
        
    def to_ml_dataset(self, df: pd.DataFrame, target_col: str = 'future_return_5', 
                      meta_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Convert a feature DataFrame to a machine learning dataset format.
        
        Args:
            df: DataFrame with features and target
            target_col: Column name of the prediction target
            meta_cols: Columns to preserve as metadata
            
        Returns:
            Tuple of (features, target, metadata)
        """
        if df.empty or target_col not in df.columns:
            return pd.DataFrame(), pd.Series(), pd.DataFrame()
            
        # Determine metadata columns
        if meta_cols is None:
            meta_cols = []
            # Add index if it's a DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                meta_cols.append(df.index.name or 'date')
                
            # Add common metadata columns if they exist
            for col in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'ticker', 'market_regime']:
                if col in df.columns:
                    meta_cols.append(col)
        
        # Extract target
        y = df[target_col]
        
        # Create metadata DataFrame
        meta = pd.DataFrame(index=df.index)
        for col in meta_cols:
            if col in df.columns:
                meta[col] = df[col]
                
        # If using DatetimeIndex, ensure it's part of metadata
        if isinstance(df.index, pd.DatetimeIndex):
            meta['timestamp'] = df.index
            
        # Create feature set by dropping target and metadata columns
        X = df.drop(columns=[target_col] + [col for col in meta_cols if col in df.columns], errors='ignore')
        
        # Add reference to config snapshot in metadata
        if hasattr(self, 'config_snapshot') and self.config_snapshot:
            meta['config_snapshot_id'] = self.config_snapshot.get('timestamp')
        
        return X, y, meta
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the feature engineering configuration for logging.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'ta_sets': self.params.get('ta_feature_sets', []),
            'lookbacks': self.params.get('lookback_periods', []),
            'normalize': self.params.get('normalize_features', True),
            'feature_selection': self.params.get('feature_selection', 'none'),
            'max_features': self.params.get('max_features', 0),
            'pca': self.params.get('add_pca_components', False),
            'include_lags': self.params.get('include_lags', False),
            'lags': self.params.get('lags', []),
            'include_context': self.params.get('include_context_features', False),
            'market_context': self.params.get('market_context', 'normal'),
            'include_cross_asset': self.params.get('include_cross_asset', False),
            'correlation_windows': self.params.get('correlation_windows', []),
            'detect_market_regime': self.params.get('detect_market_regime', False),
            'current_regime': self.market_regime,
            'apply_feature_masks': self.params.get('apply_feature_masks', False),
            'use_gpu': self.params.get('use_gpu', False) and GPU_AVAILABLE,
            'rolling_labels': self.params.get('rolling_labels', False),
            'config_snapshot_id': self.config_snapshot['timestamp'] if hasattr(self, 'config_snapshot') else None,
            'timestamp': pd.Timestamp.now().isoformat()
        } 