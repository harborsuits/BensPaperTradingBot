#!/usr/bin/env python3
"""
Market Regime Detector Module

This module provides functionality to detect and classify different market regimes
for improved strategy optimization and testing.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

class RegimeType(str, Enum):
    """Types of market regimes that can be detected."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CUSTOM = "custom"

class MarketRegimeDetector:
    """
    Detects market regimes using various technical and statistical methods.
    Can classify markets into different regimes such as:
    - Trending up/down
    - Range-bound
    - High/low volatility
    - Bull/bear markets
    """
    
    REGIME_TYPES = {
        'trend': ['strong_uptrend', 'uptrend', 'sideways', 'downtrend', 'strong_downtrend'],
        'volatility': ['very_low', 'low', 'medium', 'high', 'very_high'],
        'market_cycle': ['bull', 'bear', 'recovery', 'distribution']
    }
    
    def __init__(self, lookback_period: int = 90, n_regimes: int = 4, 
                 feature_set: str = 'full', log_level: str = 'INFO'):
        """
        Initialize the MarketRegimeDetector.
        
        Args:
            lookback_period: Number of days to look back for regime detection
            n_regimes: Number of regimes to cluster data into
            feature_set: Which features to use ('basic', 'full', 'custom')
            log_level: Logging level
        """
        self.lookback_period = lookback_period
        self.n_regimes = n_regimes
        self.feature_set = feature_set
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.regime_history = []
        self.features_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        self.logger.info(f"Initialized MarketRegimeDetector with {n_regimes} regimes and "
                        f"{lookback_period} day lookback period")
    
    def calculate_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features used for regime detection.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        # Ensure we have enough data
        if len(price_data) < self.lookback_period:
            self.logger.warning(f"Price data too short: {len(price_data)} < {self.lookback_period}")
            return pd.DataFrame()
        
        # Make sure price_data is sorted by date
        price_data = price_data.sort_index()
        
        # Create a new DataFrame for features
        features = pd.DataFrame(index=price_data.index)
        
        # Calculate returns
        features['returns'] = price_data['close'].pct_change()
        features['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        
        # Trend features
        for window in [5, 10, 20, 50]:
            # Moving averages
            features[f'ma_{window}'] = price_data['close'].rolling(window=window).mean()
            # Normalized price (distance from moving average)
            features[f'ma_dist_{window}'] = (price_data['close'] - features[f'ma_{window}']) / features[f'ma_{window}']
            # Moving average slopes
            features[f'ma_slope_{window}'] = features[f'ma_{window}'].pct_change(5)
        
        # Volatility features
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = features['log_returns'].rolling(window=window).std() * np.sqrt(252)
            features[f'true_range_{window}'] = (
                (price_data['high'] - price_data['low']) / price_data['close']
            ).rolling(window=window).mean()
        
        # Momentum features
        for window in [5, 10, 20, 50]:
            features[f'momentum_{window}'] = price_data['close'].pct_change(window)
            
        # RSI
        delta = price_data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features (if volume is available)
        if 'volume' in price_data.columns:
            features['volume_change'] = price_data['volume'].pct_change()
            features['volume_ma10'] = price_data['volume'].rolling(window=10).mean()
            features['volume_ma10_dist'] = (price_data['volume'] - features['volume_ma10']) / features['volume_ma10']
            
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def detect_regime_kmeans(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regimes using K-means clustering.
        
        Args:
            features: DataFrame with calculated features
            
        Returns:
            Dictionary with regime information
        """
        # Select features based on feature_set
        if self.feature_set == 'basic':
            selected_features = [col for col in features.columns 
                                if ('ma_dist' in col or 'volatility' in col or 'momentum' in col)]
        elif self.feature_set == 'full':
            selected_features = features.columns.tolist()
        else:  # custom feature set can be implemented here
            selected_features = [col for col in features.columns 
                                if ('ma_dist' in col or 'volatility' in col or 'momentum' in col)]
        
        # Ensure we have the requested features
        available_features = [f for f in selected_features if f in features.columns]
        if not available_features:
            self.logger.error("No valid features available for regime detection")
            return {'regime': -1, 'confidence': 0.0, 'features': {}}
        
        feature_matrix = features[available_features].iloc[-self.lookback_period:].copy()
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Run K-means clustering
        if self.kmeans_model is None:
            self.kmeans_model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            self.kmeans_model.fit(scaled_features)
        
        # Predict the cluster for the current regime
        current_regime = self.kmeans_model.predict(scaled_features[-1:].reshape(1, -1))[0]
        
        # Calculate distance to cluster center for confidence measure
        cluster_centers = self.kmeans_model.cluster_centers_
        current_point = scaled_features[-1:]
        distances = [np.linalg.norm(current_point - center) for center in cluster_centers]
        nearest_distance = min(distances)
        max_distance = np.max([np.linalg.norm(c1 - c2) for i, c1 in enumerate(cluster_centers) 
                              for c2 in cluster_centers[i+1:]])
        confidence = 1 - (nearest_distance / max_distance if max_distance > 0 else 0)
        
        # Get the regime characteristics
        regime_data = self._analyze_regime_characteristics(features, current_regime)
        
        self.logger.info(f"Detected market regime: {current_regime} with confidence: {confidence:.2f}")
        
        # Create output dictionary
        result = {
            'regime': int(current_regime),
            'confidence': float(confidence),
            'features': {
                'trend': regime_data['trend'],
                'volatility': regime_data['volatility'],
                'market_cycle': regime_data['market_cycle']
            },
            'raw_features': {f: float(features[f].iloc[-1]) for f in available_features}
        }
        
        # Add to history
        self.regime_history.append({
            'date': features.index[-1],
            'regime': current_regime,
            'confidence': confidence
        })
        
        self.features_history.append({
            'date': features.index[-1],
            'features': {f: float(features[f].iloc[-1]) for f in available_features}
        })
        
        return result
    
    def _analyze_regime_characteristics(self, features: pd.DataFrame, regime: int) -> Dict[str, str]:
        """
        Analyze the characteristics of the current regime.
        
        Args:
            features: DataFrame with calculated features
            regime: Current regime number
            
        Returns:
            Dictionary with regime characteristics
        """
        last_row = features.iloc[-1]
        
        # Determine trend
        ma_slopes = [last_row[f] for f in features.columns if 'ma_slope' in f]
        avg_slope = np.mean(ma_slopes) if ma_slopes else 0
        
        if avg_slope > 0.01:
            trend = 'strong_uptrend'
        elif avg_slope > 0.001:
            trend = 'uptrend'
        elif avg_slope < -0.01:
            trend = 'strong_downtrend'
        elif avg_slope < -0.001:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Determine volatility
        volatility_features = [last_row[f] for f in features.columns if 'volatility' in f]
        avg_volatility = np.mean(volatility_features) if volatility_features else 0
        
        if avg_volatility > 0.4:  # Very high volatility (>40% annualized)
            volatility = 'very_high'
        elif avg_volatility > 0.25:  # High volatility (25-40% annualized)
            volatility = 'high'
        elif avg_volatility > 0.15:  # Medium volatility (15-25% annualized)
            volatility = 'medium'
        elif avg_volatility > 0.08:  # Low volatility (8-15% annualized)
            volatility = 'low'
        else:  # Very low volatility (<8% annualized)
            volatility = 'very_low'
        
        # Determine market cycle
        momentum_features = [last_row[f] for f in features.columns if 'momentum' in f]
        avg_momentum = np.mean(momentum_features) if momentum_features else 0
        rsi = last_row.get('rsi', 50)
        
        if avg_momentum > 0.05 and rsi > 60:
            market_cycle = 'bull'
        elif avg_momentum < -0.05 and rsi < 40:
            market_cycle = 'bear'
        elif avg_momentum > 0 and rsi > 50:
            market_cycle = 'recovery'
        else:
            market_cycle = 'distribution'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'market_cycle': market_cycle
        }
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method to detect the current market regime.
        
        Args:
            price_data: DataFrame with OHLCV price data
            
        Returns:
            Dictionary with regime information
        """
        features = self.calculate_features(price_data)
        if features.empty:
            self.logger.warning("Could not calculate features for regime detection")
            return {'regime': -1, 'confidence': 0.0, 'features': {}}
        
        return self.detect_regime_kmeans(features)
    
    def classify_historical_regimes(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all historical data into regimes.
        
        Args:
            price_data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with regime classifications for each date
        """
        features = self.calculate_features(price_data)
        if features.empty:
            return pd.DataFrame()
        
        # Initialize the model if not already done
        feature_matrix = features.copy()
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        if self.kmeans_model is None:
            self.kmeans_model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            self.kmeans_model.fit(scaled_features)
        
        # Predict regimes for all data points
        regimes = self.kmeans_model.predict(scaled_features)
        
        # Create a DataFrame with the results
        regime_df = pd.DataFrame({
            'date': features.index,
            'regime': regimes
        })
        regime_df.set_index('date', inplace=True)
        
        # Add characteristics for each regime
        for regime in range(self.n_regimes):
            regime_indices = regime_df[regime_df['regime'] == regime].index
            if len(regime_indices) > 0:
                # Take a representative sample from each regime
                sample_idx = regime_indices[len(regime_indices) // 2]
                regime_features = features.loc[sample_idx:sample_idx]
                chars = self._analyze_regime_characteristics(regime_features, regime)
                
                # Add to DataFrame
                for char_type, value in chars.items():
                    regime_df.loc[regime_indices, f'regime_{char_type}'] = value
        
        self.logger.info(f"Classified {len(regime_df)} data points into {self.n_regimes} regimes")
        return regime_df
    
    def plot_regimes(self, price_data: pd.DataFrame, regime_data: pd.DataFrame = None) -> None:
        """
        Plot price data with regime classifications.
        
        Args:
            price_data: DataFrame with OHLCV price data
            regime_data: DataFrame with regime classifications (optional)
        """
        if regime_data is None:
            regime_data = self.classify_historical_regimes(price_data)
            
        if regime_data.empty:
            self.logger.warning("No regime data available for plotting")
            return
        
        # Create figure with price and regime plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(price_data.index, price_data['close'], color='black', linewidth=1.5)
        ax1.set_title('Price with Market Regime Classification')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # Color regions by regime
        regime_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Find continuous regions of the same regime
        current_regime = regime_data['regime'].iloc[0]
        start_idx = 0
        
        for i in range(1, len(regime_data)):
            if regime_data['regime'].iloc[i] != current_regime or i == len(regime_data) - 1:
                # End of a regime region, shade it
                end_idx = i
                if end_idx == len(regime_data) - 1 and regime_data['regime'].iloc[i] == current_regime:
                    end_idx += 1
                
                start_date = regime_data.index[start_idx]
                end_date = regime_data.index[min(end_idx, len(regime_data) - 1)]
                
                color = regime_colors[int(current_regime) % len(regime_colors)]
                ax1.axvspan(start_date, end_date, alpha=0.3, color=color)
                
                # Reset for next region
                current_regime = regime_data['regime'].iloc[i]
                start_idx = i
        
        # Plot regimes on the second axis
        ax2.scatter(regime_data.index, regime_data['regime'], c=regime_data['regime'].map(
            lambda x: regime_colors[int(x) % len(regime_colors)]), s=30)
        ax2.set_ylabel('Regime')
        ax2.set_yticks(range(self.n_regimes))
        ax2.grid(True, alpha=0.3)
        
        # Add legend for regimes
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=regime_colors[i % len(regime_colors)],
                                 label=f'Regime {i}', markersize=10) 
                          for i in range(self.n_regimes)]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Format dates
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get the most recently detected regime.
        
        Returns:
            Dictionary with current regime information
        """
        if not self.regime_history:
            return {'regime': -1, 'confidence': 0.0, 'features': {}}
        
        return self.regime_history[-1]
    
    def save_model(self, filepath: str) -> None:
        """
        Save the regime detection model.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        model_data = {
            'kmeans_model': self.kmeans_model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'lookback_period': self.lookback_period,
            'feature_set': self.feature_set
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved regime detection model.
        
        Args:
            filepath: Path to the saved model
        """
        import joblib
        try:
            model_data = joblib.load(filepath)
            self.kmeans_model = model_data['kmeans_model']
            self.scaler = model_data['scaler']
            self.n_regimes = model_data['n_regimes']
            self.lookback_period = model_data['lookback_period']
            self.feature_set = model_data['feature_set']
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def load_price_data(
        self, 
        price_data: pd.DataFrame,
        date_column: str = 'date',
        ensure_datetime_index: bool = True
    ) -> None:
        """
        Load price data for regime detection.
        
        Args:
            price_data: DataFrame with OHLCV price data
            date_column: Column name containing dates
            ensure_datetime_index: Whether to convert index to datetime
        """
        self.price_data = price_data.copy()
        
        if ensure_datetime_index and not isinstance(self.price_data.index, pd.DatetimeIndex):
            if date_column in self.price_data.columns:
                self.price_data.set_index(date_column, inplace=True)
            self.price_data.index = pd.to_datetime(self.price_data.index)
            
        logger.info(f"Loaded price data with {len(self.price_data)} rows")
    
    def add_technical_indicators(self) -> None:
        """
        Add technical indicators useful for regime detection.
        """
        if self.price_data is None:
            logger.error("No price data loaded. Load price data first.")
            return
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Convert column names to lowercase if they exist with different case
        for col in self.price_data.columns:
            if col.lower() in required_columns and col not in required_columns:
                self.price_data[col.lower()] = self.price_data[col]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in self.price_data.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}. Some indicators may not be calculated.")
        
        # Moving averages
        if 'close' in self.price_data.columns:
            # Simple moving averages
            for period in [20, 50, 200]:
                self.price_data[f'sma_{period}'] = self.price_data['close'].rolling(period).mean()
            
            # Exponential moving averages
            for period in [12, 26]:
                self.price_data[f'ema_{period}'] = self.price_data['close'].ewm(span=period).mean()
            
            # MACD
            if 'ema_12' in self.price_data.columns and 'ema_26' in self.price_data.columns:
                self.price_data['macd'] = self.price_data['ema_12'] - self.price_data['ema_26']
                self.price_data['macd_signal'] = self.price_data['macd'].ewm(span=9).mean()
                self.price_data['macd_hist'] = self.price_data['macd'] - self.price_data['macd_signal']
            
            # Daily returns
            self.price_data['returns'] = self.price_data['close'].pct_change()
            
            # Rolling volatility
            self.price_data['volatility'] = self.price_data['returns'].rolling(self.window_size).std() * np.sqrt(252)
            
            # Calculate z-score of returns (for mean reversion detection)
            self.price_data['returns_zscore'] = (
                self.price_data['returns'] - 
                self.price_data['returns'].rolling(self.window_size).mean()
            ) / self.price_data['returns'].rolling(self.window_size).std()
            
            # Bollinger Bands
            self.price_data['bb_middle'] = self.price_data['close'].rolling(self.window_size).mean()
            bb_std = self.price_data['close'].rolling(self.window_size).std()
            self.price_data['bb_upper'] = self.price_data['bb_middle'] + 2 * bb_std
            self.price_data['bb_lower'] = self.price_data['bb_middle'] - 2 * bb_std
            self.price_data['bb_width'] = (self.price_data['bb_upper'] - self.price_data['bb_lower']) / self.price_data['bb_middle']
            
            # RSI
            delta = self.price_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            self.price_data['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR if we have high/low data
        if all(col in self.price_data.columns for col in ['high', 'low', 'close']):
            tr1 = self.price_data['high'] - self.price_data['low']
            tr2 = abs(self.price_data['high'] - self.price_data['close'].shift())
            tr3 = abs(self.price_data['low'] - self.price_data['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.price_data['atr'] = tr.rolling(14).mean()
        
        logger.info("Added technical indicators to price data")
    
    def define_custom_regime(
        self, 
        regime_name: str,
        condition_func: Callable[[pd.DataFrame], pd.Series]
    ) -> None:
        """
        Define a custom market regime with a condition function.
        
        Args:
            regime_name: Name of the custom regime
            condition_func: Function that takes price_data and returns boolean series
        """
        self.custom_regime_funcs[regime_name] = condition_func
        logger.info(f"Added custom regime definition: {regime_name}")
    
    def detect_volatility_regimes(
        self, 
        volatility_threshold: float = 0.15,
        percentile_threshold: float = 75
    ) -> pd.Series:
        """
        Detect volatile and low-volatility regimes.
        
        Args:
            volatility_threshold: Threshold for high volatility
            percentile_threshold: Percentile threshold for relative volatility
        
        Returns:
            Series with regime classifications
        """
        if self.price_data is None or 'volatility' not in self.price_data.columns:
            logger.error("Volatility data not available. Call add_technical_indicators() first.")
            return None
        
        # Calculate volatility percentiles
        vol_percentile = self.price_data['volatility'].rolling(252).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])
        )
        
        # Classify regimes
        regimes = pd.Series(index=self.price_data.index, dtype='object')
        
        # Absolute volatility threshold
        high_vol_mask = self.price_data['volatility'] > volatility_threshold
        
        # Relative volatility (percentile-based)
        high_vol_percentile_mask = vol_percentile > percentile_threshold
        low_vol_percentile_mask = vol_percentile < (100 - percentile_threshold)
        
        # Combine criteria (both absolute and relative)
        regimes[high_vol_mask & high_vol_percentile_mask] = RegimeType.VOLATILE
        regimes[~high_vol_mask & low_vol_percentile_mask] = RegimeType.LOW_VOLATILITY
        
        # Fill remaining as sideways
        regimes.fillna(RegimeType.SIDEWAYS, inplace=True)
        
        return regimes
    
    def detect_trend_regimes(
        self,
        sma_short: int = 50,
        sma_long: int = 200,
        trend_strength_threshold: float = 0.1
    ) -> pd.Series:
        """
        Detect trending, bullish, and bearish regimes.
        
        Args:
            sma_short: Period for short-term moving average
            sma_long: Period for long-term moving average
            trend_strength_threshold: Threshold for trend strength
            
        Returns:
            Series with regime classifications
        """
        if self.price_data is None or 'close' not in self.price_data.columns:
            logger.error("Price data not available or missing 'close' column.")
            return None
        
        # Calculate moving averages if not already available
        if f'sma_{sma_short}' not in self.price_data.columns:
            self.price_data[f'sma_{sma_short}'] = self.price_data['close'].rolling(sma_short).mean()
            
        if f'sma_{sma_long}' not in self.price_data.columns:
            self.price_data[f'sma_{sma_long}'] = self.price_data['close'].rolling(sma_long).mean()
        
        # Calculate trend direction and strength
        sma_ratio = self.price_data[f'sma_{sma_short}'] / self.price_data[f'sma_{sma_long}'] - 1
        
        # Smoothed price change rate over trend period
        smoothed_change_rate = (
            (self.price_data['close'] / self.price_data['close'].shift(sma_short)) - 1
        ).rolling(sma_short).mean()
        
        # Classify regimes
        regimes = pd.Series(index=self.price_data.index, dtype='object')
        
        # Bullish trend
        bullish_mask = (sma_ratio > trend_strength_threshold) & (smoothed_change_rate > 0)
        regimes[bullish_mask] = RegimeType.BULLISH
        
        # Bearish trend
        bearish_mask = (sma_ratio < -trend_strength_threshold) & (smoothed_change_rate < 0)
        regimes[bearish_mask] = RegimeType.BEARISH
        
        # Trending (either direction but strong)
        trending_mask = (abs(sma_ratio) > trend_strength_threshold) & ~(bullish_mask | bearish_mask)
        regimes[trending_mask] = RegimeType.TRENDING
        
        # Default to sideways
        regimes.fillna(RegimeType.SIDEWAYS, inplace=True)
        
        return regimes
    
    def detect_mean_reversion_regimes(
        self, 
        zscore_threshold: float = 2.0,
        autocorrelation_window: int = 20
    ) -> pd.Series:
        """
        Detect mean-reverting regimes.
        
        Args:
            zscore_threshold: Z-score threshold for identifying extremes
            autocorrelation_window: Window for calculating autocorrelation
            
        Returns:
            Series with regime classifications
        """
        if self.price_data is None or 'returns' not in self.price_data.columns:
            logger.error("Returns data not available. Call add_technical_indicators() first.")
            return None
        
        # Calculate z-score if not already available
        if 'returns_zscore' not in self.price_data.columns:
            self.price_data['returns_zscore'] = (
                self.price_data['returns'] - 
                self.price_data['returns'].rolling(self.window_size).mean()
            ) / self.price_data['returns'].rolling(self.window_size).std()
        
        # Calculate autocorrelation of returns
        self.price_data['returns_autocorr'] = self.price_data['returns'].rolling(
            autocorrelation_window
        ).apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else np.nan)
        
        # Classify regimes
        regimes = pd.Series(index=self.price_data.index, dtype='object')
        
        # Mean reverting when autocorrelation is negative and significant
        mean_reverting_mask = (
            (self.price_data['returns_autocorr'] < -0.2) & 
            (abs(self.price_data['returns_zscore']) > zscore_threshold)
        )
        regimes[mean_reverting_mask] = RegimeType.MEAN_REVERTING
        
        # Trending when autocorrelation is positive and significant
        trending_mask = (
            (self.price_data['returns_autocorr'] > 0.2) & 
            (abs(self.price_data['returns_zscore']) <= zscore_threshold)
        )
        regimes[trending_mask] = RegimeType.TRENDING
        
        # Default to sideways
        regimes.fillna(RegimeType.SIDEWAYS, inplace=True)
        
        return regimes
    
    def detect_regimes(
        self, 
        method: str = 'combined',
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect market regimes using specified method.
        
        Args:
            method: 'volatility', 'trend', 'mean_reversion', 'combined', or 'custom'
            **kwargs: Additional parameters for specific methods
            
        Returns:
            DataFrame with regime classifications
        """
        if self.price_data is None:
            logger.error("No price data loaded. Load price data first.")
            return None
        
        # Ensure technical indicators are available
        if 'volatility' not in self.price_data.columns:
            self.add_technical_indicators()
        
        # Initialize regimes DataFrame
        regimes_df = pd.DataFrame(index=self.price_data.index)
        
        # Detect regimes using specified method
        if method == 'volatility':
            regimes_df['regime'] = self.detect_volatility_regimes(**kwargs)
        elif method == 'trend':
            regimes_df['regime'] = self.detect_trend_regimes(**kwargs)
        elif method == 'mean_reversion':
            regimes_df['regime'] = self.detect_mean_reversion_regimes(**kwargs)
        elif method == 'combined':
            # Combine multiple regime detection methods
            volatility_regimes = self.detect_volatility_regimes(**kwargs)
            trend_regimes = self.detect_trend_regimes(**kwargs)
            
            # Prioritize regime classifications
            regimes_df['regime'] = trend_regimes
            
            # Override with volatility regimes when relevant
            volatility_mask = volatility_regimes.isin([RegimeType.VOLATILE])
            regimes_df.loc[volatility_mask, 'regime'] = volatility_regimes[volatility_mask]
        elif method == 'custom':
            # Use custom regime definitions
            if not self.custom_regime_funcs:
                logger.error("No custom regime definitions available. Define custom regimes first.")
                return None
            
            # Initialize with default regime
            regimes_df['regime'] = RegimeType.SIDEWAYS
            
            # Apply each custom regime function in priority order
            for regime_name, condition_func in self.custom_regime_funcs.items():
                mask = condition_func(self.price_data)
                regimes_df.loc[mask, 'regime'] = regime_name
        else:
            logger.error(f"Unknown regime detection method: {method}")
            return None
        
        # Store the regimes
        self.regimes = regimes_df
        
        # Find regime changes
        self.regime_changes = regimes_df['regime'].ne(regimes_df['regime'].shift()).cumsum()
        
        # Calculate regime statistics
        self._calculate_regime_statistics()
        
        logger.info(f"Detected {len(regimes_df['regime'].unique())} distinct regime types using {method} method")
        
        return regimes_df
    
    def _calculate_regime_statistics(self) -> None:
        """
        Calculate statistics for each regime period.
        """
        if self.regimes is None or self.regime_changes is None:
            logger.error("No regimes detected. Call detect_regimes() first.")
            return
        
        # Group by regime change and calculate statistics
        regime_stats = []
        
        for regime_id, group in self.price_data.join(self.regimes).groupby(self.regime_changes):
            if len(group) < 1:
                continue
                
            regime_type = group['regime'].iloc[0]
            start_date = group.index[0]
            end_date = group.index[-1]
            duration = len(group)
            
            # Performance statistics (if 'close' and 'returns' are available)
            return_stats = {}
            if 'close' in group.columns:
                total_return = (group['close'].iloc[-1] / group['close'].iloc[0]) - 1
                return_stats['total_return'] = total_return
            
            if 'returns' in group.columns:
                returns = group['returns'].dropna()
                if len(returns) > 0:
                    return_stats.update({
                        'mean_return': returns.mean(),
                        'std_return': returns.std(),
                        'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                        'max_return': returns.max(),
                        'min_return': returns.min(),
                        'positive_days': (returns > 0).sum() / len(returns)
                    })
            
            # Volatility statistics
            vol_stats = {}
            if 'volatility' in group.columns:
                volatility = group['volatility'].dropna()
                if len(volatility) > 0:
                    vol_stats.update({
                        'mean_volatility': volatility.mean(),
                        'max_volatility': volatility.max(),
                        'min_volatility': volatility.min()
                    })
            
            # Combine all statistics
            stats = {
                'regime': regime_type,
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration,
                **return_stats,
                **vol_stats
            }
            
            regime_stats.append(stats)
        
        # Create DataFrame from statistics
        self.regime_stats = pd.DataFrame(regime_stats)
        
        # Sort by start date
        if not self.regime_stats.empty:
            self.regime_stats.sort_values('start_date', inplace=True)
            
            logger.info(f"Calculated statistics for {len(self.regime_stats)} regime periods")
    
    def get_regime_at_date(self, date: Union[str, datetime]) -> str:
        """
        Get the regime classification for a specific date.
        
        Args:
            date: Date to look up
            
        Returns:
            Regime type as string
        """
        if self.regimes is None:
            logger.error("No regimes detected. Call detect_regimes() first.")
            return None
        
        try:
            date = pd.Timestamp(date)
            return self.regimes.loc[date, 'regime']
        except (KeyError, ValueError) as e:
            logger.error(f"Error getting regime for date {date}: {e}")
            return None
    
    def get_regime_periods(
        self, 
        regime_type: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Get periods for specified regime type(s).
        
        Args:
            regime_type: Specific regime type(s) to filter by
            
        Returns:
            DataFrame with regime periods
        """
        if self.regime_stats is None:
            logger.error("No regime statistics available. Call detect_regimes() first.")
            return None
        
        if regime_type is None:
            return self.regime_stats
        
        # Convert single regime type to list
        if isinstance(regime_type, str):
            regime_type = [regime_type]
        
        # Filter by regime type
        return self.regime_stats[self.regime_stats['regime'].isin(regime_type)]
    
    def calculate_regime_transition_matrix(self) -> pd.DataFrame:
        """
        Calculate the probability of transitioning between regimes.
        
        Returns:
            DataFrame with transition probabilities
        """
        if self.regimes is None:
            logger.error("No regimes detected. Call detect_regimes() first.")
            return None
        
        # Get regime changes
        regime_series = self.regimes['regime']
        
        # Calculate transitions
        transitions = pd.crosstab(
            regime_series.shift(),
            regime_series,
            normalize='index'
        )
        
        return transitions
    
    def plot_regime_performance(
        self, 
        save_to_file: bool = True,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot performance statistics by regime.
        
        Args:
            save_to_file: Whether to save plot to file
            filename: Custom filename for the plot
        """
        if self.regime_stats is None or self.regime_stats.empty:
            logger.error("No regime statistics available. Call detect_regimes() first.")
            return
        
        if 'total_return' not in self.regime_stats.columns:
            logger.error("No return data available for performance comparison.")
            return
        
        # Group by regime type and calculate average performance
        regime_performance = self.regime_stats.groupby('regime').agg({
            'total_return': 'mean',
            'duration': 'mean',
            'mean_volatility': 'mean' if 'mean_volatility' in self.regime_stats.columns else 'count'
        })
        
        # Create plot with multiple subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
        # Plot returns by regime
        regime_performance['total_return'].sort_values().plot(
            kind='bar', 
            ax=axes[0], 
            color='skyblue'
        )
        axes[0].set_title('Average Return by Regime')
        axes[0].set_ylabel('Return')
        
        # Plot duration by regime
        regime_performance['duration'].sort_values().plot(
            kind='bar', 
            ax=axes[1], 
            color='lightgreen'
        )
        axes[1].set_title('Average Duration by Regime')
        axes[1].set_ylabel('Days')
        
        # Plot volatility by regime if available
        if 'mean_volatility' in regime_performance.columns:
            regime_performance['mean_volatility'].sort_values().plot(
                kind='bar', 
                ax=axes[2], 
                color='salmon'
            )
            axes[2].set_title('Average Volatility by Regime')
            axes[2].set_ylabel('Volatility')
        
        plt.tight_layout()
        
        if save_to_file:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"regime_performance_{timestamp}.png"
            
            filepath = self.results_dir / filename
            plt.savefig(filepath)
            logger.info(f"Saved performance plot to {filepath}")
        
        plt.close()
    
    def save_regime_data(
        self, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save regime data to file.
        
        Args:
            filename: Custom filename for the data
            
        Returns:
            Path to the saved file
        """
        if self.regimes is None:
            logger.error("No regimes detected. Call detect_regimes() first.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_regimes_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Prepare data for saving
        save_data = {
            'regime_periods': self.regime_stats.to_dict(orient='records') if self.regime_stats is not None else [],
            'metadata': {
                'window_size': self.window_size,
                'price_start_date': str(self.price_data.index[0]) if self.price_data is not None else None,
                'price_end_date': str(self.price_data.index[-1]) if self.price_data is not None else None,
                'regime_count': self.regimes['regime'].value_counts().to_dict() if self.regimes is not None else {}
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved regime data to {filepath}")
        
        return str(filepath)
    
    def load_regime_data(
        self, 
        filepath: str
    ) -> bool:
        """
        Load regime data from file.
        
        Args:
            filepath: Path to the regime data file
            
        Returns:
            Whether data was loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load regime periods
            if 'regime_periods' in data and data['regime_periods']:
                self.regime_stats = pd.DataFrame(data['regime_periods'])
                
                # Convert string dates to datetime
                date_columns = ['start_date', 'end_date']
                for col in date_columns:
                    if col in self.regime_stats.columns:
                        self.regime_stats[col] = pd.to_datetime(self.regime_stats[col])
            
            # Load metadata
            if 'metadata' in data:
                metadata = data['metadata']
                if 'window_size' in metadata:
                    self.window_size = metadata['window_size']
            
            logger.info(f"Loaded regime data from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading regime data: {e}")
            return False
    
    def analyze_strategy_performance_by_regime(
        self, 
        strategy_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze strategy performance across different regimes.
        
        Args:
            strategy_returns: Series of strategy returns
            
        Returns:
            DataFrame with performance metrics by regime
        """
        if self.regimes is None:
            logger.error("No regimes detected. Call detect_regimes() first.")
            return None
        
        # Ensure strategy returns have the same index as regimes
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns.index = pd.to_datetime(strategy_returns.index)
        
        # Combine strategy returns with regime data
        combined = pd.DataFrame({
            'returns': strategy_returns,
            'regime': self.regimes['regime']
        }).dropna()
        
        # Calculate performance metrics by regime
        regime_performance = []
        
        for regime_type, group in combined.groupby('regime'):
            returns = group['returns']
            
            # Skip if not enough data
            if len(returns) < 5:
                continue
            
            # Calculate performance metrics
            total_return = (1 + returns).prod() - 1
            cagr = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = cagr / volatility if volatility > 0 else 0
            max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
            win_rate = (returns > 0).mean()
            
            regime_performance.append({
                'regime': regime_type,
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_days': len(returns)
            })
        
        return pd.DataFrame(regime_performance)
    
    def get_regime_duration(self, date: Optional[Union[str, datetime]] = None) -> int:
        """
        Get the duration of the current regime at a specific date.
        
        Args:
            date: Date to check (default: most recent)
            
        Returns:
            Duration in days
        """
        if self.regimes is None or self.regime_changes is None:
            logger.error("No regimes detected. Call detect_regimes() first.")
            return 0
        
        if date is None:
            # Use most recent date
            date = self.regimes.index[-1]
        else:
            date = pd.Timestamp(date)
        
        try:
            # Get regime change value at date
            regime_change_val = self.regime_changes.loc[date]
            
            # Find all dates with same regime change value
            same_regime_mask = self.regime_changes == regime_change_val
            same_regime_dates = self.regime_changes[same_regime_mask].index
            
            # Calculate duration
            if len(same_regime_dates) > 0:
                first_date = same_regime_dates[0]
                duration = (date - first_date).days + 1
                return max(1, duration)
            
            return 1
        except (KeyError, ValueError):
            logger.error(f"Error calculating regime duration for date {date}")
            return 0 