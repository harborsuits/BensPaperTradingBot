"""
Market Regime Detector

This module provides advanced market regime detection capabilities,
adapted from Freqtrade's FreqAI system to identify different market
conditions and adapt trading strategies accordingly.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Advanced market regime detection system
    
    Identifies market regimes (trending, ranging, volatile) to
    adapt strategy parameters and risk management dynamically.
    """
    
    def __init__(self, config=None):
        """
        Initialize the market regime detector
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        
        # Regime detection parameters
        self.n_regimes = self.config.get('n_regimes', 3)
        self.lookback_window = self.config.get('lookback_window', 30)
        self.features = self.config.get('regime_features', [
            'volatility', 'trend_strength', 'volume_profile', 'correlation'
        ])
        self.use_adaptive_params = self.config.get('use_adaptive_params', True)
        
        # Regime classification model (K-means clustering)
        self.regime_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
        # Regime characteristics
        self.regime_characteristics = {}
        
        # Store regime history
        self.regime_history = pd.DataFrame()
        
        # Current regime
        self.current_regime = None
        self.regime_confidence = 0.0
        
        logger.info("Market Regime Detector initialized")
    
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the current market regime
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            Dict with regime information
        """
        # Check for minimum data
        if len(data) < self.lookback_window + 10:
            logger.warning("Insufficient data for regime detection")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'characteristics': {},
                'description': 'Insufficient data'
            }
        
        # Extract most recent window
        window_data = data.iloc[-self.lookback_window:].copy()
        
        # Calculate regime features
        regime_features = self._calculate_regime_features(window_data)
        
        # Detect regime using clustering model
        regime_info = self._classify_regime(regime_features)
        
        # Store regime history
        self._update_regime_history(regime_info)
        
        # Update current regime
        self.current_regime = regime_info['regime']
        self.regime_confidence = regime_info['confidence']
        
        return regime_info
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for regime detection
        
        Args:
            data: Price data for feature calculation
            
        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=[0])
        
        # Volatility features
        if 'volatility' in self.features:
            # Historical volatility (annualized)
            features['hist_volatility'] = data['close'].pct_change().std() * np.sqrt(252)
            
            # ATR-based volatility
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr_volatility'] = tr.mean() / data['close'].mean()
            
            # Volatility compared to historical
            rolling_vol = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            features['relative_volatility'] = features['hist_volatility'] / rolling_vol.mean()
        
        # Trend strength features
        if 'trend_strength' in self.features:
            # ADX-based trend strength
            # +DM and -DM
            high_diff = data['high'].diff()
            low_diff = data['low'].diff().mul(-1)
            
            plus_dm = ((high_diff > 0) & (high_diff > low_diff)) * high_diff
            minus_dm = ((low_diff > 0) & (low_diff > high_diff)) * low_diff
            
            # True Range (simplified from previous calculation)
            tr_for_adx = tr
            
            # Smoothed values
            window = 14  # Standard ADX window
            smoothed_plus_dm = plus_dm.rolling(window=window).mean()
            smoothed_minus_dm = minus_dm.rolling(window=window).mean()
            smoothed_tr = tr_for_adx.rolling(window=window).mean()
            
            # Directional indicators
            plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
            minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
            
            # Directional index
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
            
            # ADX (smoothed DX)
            adx = dx.rolling(window=window).mean().fillna(0)
            features['adx'] = adx.iloc[-1]
            
            # Moving average trend strength
            sma_short = data['close'].rolling(window=10).mean()
            sma_long = data['close'].rolling(window=30).mean()
            features['ma_trend_strength'] = ((sma_short.iloc[-1] / sma_long.iloc[-1]) - 1) * 100
        
        # Volume profile
        if 'volume_profile' in self.features:
            if 'volume' in data.columns:
                # Volume change
                features['volume_change'] = data['volume'].pct_change().mean() * 100
                
                # Volume vs historical
                features['volume_vs_avg'] = data['volume'].iloc[-5:].mean() / data['volume'].mean()
                
                # Price-volume correlation
                features['price_volume_corr'] = data['close'].pct_change().corr(data['volume'].pct_change())
        
        # Correlation profile
        if 'correlation' in self.features:
            # Auto-correlation of returns (trending vs mean-reverting behavior)
            returns = data['close'].pct_change().dropna()
            if len(returns) > 5:
                features['return_autocorr'] = returns.autocorr(lag=1)
            else:
                features['return_autocorr'] = 0
        
        return features
    
    def _train_regime_model(self, historical_data: pd.DataFrame):
        """
        Train the regime classification model using historical data
        
        Args:
            historical_data: Historical price data
        """
        if len(historical_data) < 100:
            logger.warning("Insufficient data to train regime model")
            return
        
        # Create overlapping windows
        windows = []
        window_size = self.lookback_window
        step = max(1, window_size // 5)  # 80% overlap
        
        for i in range(0, len(historical_data) - window_size, step):
            window = historical_data.iloc[i:i+window_size]
            if len(window) == window_size:
                windows.append(window)
        
        if not windows:
            logger.warning("No valid windows for regime model training")
            return
        
        # Calculate features for each window
        feature_data = []
        for window in windows:
            features = self._calculate_regime_features(window)
            feature_data.append(features)
            
        feature_df = pd.concat(feature_data, ignore_index=True)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_df)
        
        # Apply PCA to reduce dimensionality (optional)
        try:
            pca_features = self.pca.fit_transform(scaled_features)
        except Exception as e:
            logger.warning(f"PCA failed: {e}, using scaled features instead")
            pca_features = scaled_features
        
        # Train K-means clustering model
        self.regime_model = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=42)
        self.regime_model.fit(pca_features)
        
        # Get cluster characteristics
        if hasattr(self.regime_model, 'labels_'):
            # Get cluster centers in original feature space (approximate)
            for i in range(self.n_regimes):
                # Get samples in this cluster
                cluster_samples = feature_df.iloc[self.regime_model.labels_ == i]
                
                if len(cluster_samples) > 0:
                    # Calculate cluster characteristics
                    self.regime_characteristics[i] = {
                        'size': len(cluster_samples) / len(feature_df),
                        'center': cluster_samples.mean().to_dict()
                    }
                    
                    # Identify regime type based on characteristics
                    regime_type = self._identify_regime_type(self.regime_characteristics[i]['center'])
                    self.regime_characteristics[i]['type'] = regime_type
                    self.regime_characteristics[i]['description'] = self._get_regime_description(regime_type)
        
        logger.info(f"Trained regime model with {self.n_regimes} regimes")
    
    def _classify_regime(self, regime_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify the current market regime
        
        Args:
            regime_features: Features for regime classification
            
        Returns:
            Dict with regime information
        """
        # If model not trained, use simple classification
        if self.regime_model is None:
            return self._simple_regime_classification(regime_features)
        
        # Scale features
        scaled_features = self.scaler.transform(regime_features)
        
        # Apply PCA transform
        try:
            pca_features = self.pca.transform(scaled_features)
        except Exception as e:
            logger.warning(f"PCA transform failed: {e}, using scaled features")
            pca_features = scaled_features
        
        # Classify using K-means
        cluster = self.regime_model.predict(pca_features)[0]
        
        # Calculate confidence (distance to cluster center, inversely proportional)
        distances = self.regime_model.transform(pca_features)[0]
        min_distance = distances[cluster]
        confidence = 1.0 / (1.0 + min_distance)
        
        # Get regime type and characteristics
        if cluster in self.regime_characteristics:
            regime_type = self.regime_characteristics[cluster]['type']
            description = self.regime_characteristics[cluster]['description']
            characteristics = self.regime_characteristics[cluster]['center']
        else:
            # Fallback to simple classification
            regime_info = self._simple_regime_classification(regime_features)
            regime_type = regime_info['regime']
            description = regime_info['description']
            characteristics = regime_features.iloc[0].to_dict()
        
        return {
            'regime': regime_type,
            'regime_id': int(cluster),
            'confidence': float(confidence),
            'characteristics': characteristics,
            'description': description
        }
    
    def _simple_regime_classification(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple regime classification when model not trained
        
        Args:
            features: Feature dataframe
            
        Returns:
            Dict with regime information
        """
        regime_type = "unknown"
        description = "Unknown market regime"
        confidence = 0.5
        
        # Extract key features if available
        volatility = features.get('hist_volatility', pd.Series([0])).iloc[0]
        trend_strength = features.get('adx', pd.Series([0])).iloc[0]
        ma_trend = features.get('ma_trend_strength', pd.Series([0])).iloc[0]
        
        # Determine regime based on simple rules
        if trend_strength > 25:
            # Strong trend
            if ma_trend > 0:
                regime_type = "strong_uptrend"
                description = "Strong uptrend with momentum"
                confidence = min(trend_strength / 100, 0.9)
            else:
                regime_type = "strong_downtrend"
                description = "Strong downtrend with momentum"
                confidence = min(trend_strength / 100, 0.9)
        elif volatility > 0.2:  # High volatility
            regime_type = "volatile"
            description = "Highly volatile market conditions"
            confidence = min(volatility / 0.4, 0.9)
        elif abs(ma_trend) < 0.5 and trend_strength < 15:
            regime_type = "ranging"
            description = "Range-bound, sideways market"
            confidence = max(0.5, 1 - (trend_strength / 15))
        elif ma_trend > 0:
            regime_type = "weak_uptrend"
            description = "Weak uptrend, low momentum"
            confidence = 0.5 + (trend_strength / 40)
        else:
            regime_type = "weak_downtrend"
            description = "Weak downtrend, low momentum"
            confidence = 0.5 + (trend_strength / 40)
        
        return {
            'regime': regime_type,
            'confidence': confidence,
            'characteristics': features.iloc[0].to_dict(),
            'description': description
        }
    
    def _identify_regime_type(self, characteristics: Dict[str, float]) -> str:
        """
        Identify regime type based on characteristics
        
        Args:
            characteristics: Dictionary of regime characteristics
            
        Returns:
            String identifying the regime type
        """
        # Extract key characteristics
        volatility = characteristics.get('hist_volatility', 0)
        trend_adx = characteristics.get('adx', 0)
        ma_trend = characteristics.get('ma_trend_strength', 0)
        
        # Identify regime type based on characteristics
        if trend_adx > 25:  # Strong trend
            if ma_trend > 0:
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif volatility > 0.2:  # High volatility
            return "volatile"
        elif abs(ma_trend) < 0.5 and trend_adx < 15:
            return "ranging"
        elif ma_trend > 0:
            return "weak_uptrend"
        else:
            return "weak_downtrend"
    
    def _get_regime_description(self, regime_type: str) -> str:
        """
        Get human-readable description of a regime type
        
        Args:
            regime_type: Type of market regime
            
        Returns:
            String description
        """
        descriptions = {
            "strong_uptrend": "Strong uptrend with momentum - trend-following strategies favored",
            "weak_uptrend": "Weak uptrend with low momentum - cautious trend-following",
            "strong_downtrend": "Strong downtrend with momentum - trend-following short positions favored",
            "weak_downtrend": "Weak downtrend with low momentum - cautious short positions",
            "volatile": "Highly volatile market - reduced position sizing recommended",
            "ranging": "Range-bound, sideways market - mean-reversion strategies favored"
        }
        
        return descriptions.get(regime_type, "Unknown market regime")
    
    def _update_regime_history(self, regime_info: Dict[str, Any]):
        """
        Update the regime history with current detection
        
        Args:
            regime_info: Current regime information
        """
        # Create entry for regime history
        entry = {
            'timestamp': pd.Timestamp.now(),
            'regime': regime_info['regime'],
            'regime_id': regime_info.get('regime_id', 0),
            'confidence': regime_info['confidence']
        }
        
        # Add to history
        self.regime_history = pd.concat([
            self.regime_history,
            pd.DataFrame([entry])
        ], ignore_index=True)
        
        # Limit history size
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history.iloc[-1000:]
    
    def get_strategy_adjustments(self, strategy_name: str, base_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get strategy parameter adjustments based on current regime
        
        Args:
            strategy_name: Name of the strategy
            base_params: Base strategy parameters
            
        Returns:
            Dictionary of adjusted parameters
        """
        if not self.use_adaptive_params or not self.current_regime:
            return base_params or {}
        
        # Start with base parameters
        adjusted_params = base_params.copy() if base_params else {}
        
        # Adapt parameters based on regime
        if self.current_regime == "strong_uptrend":
            # More aggressive in strong uptrends
            self._adjust_trend_params(adjusted_params, aggressive=True, direction=1)
        
        elif self.current_regime == "weak_uptrend":
            # Slightly more conservative in weak uptrends
            self._adjust_trend_params(adjusted_params, aggressive=False, direction=1)
        
        elif self.current_regime == "strong_downtrend":
            # More aggressive in strong downtrends (for short positions)
            self._adjust_trend_params(adjusted_params, aggressive=True, direction=-1)
        
        elif self.current_regime == "weak_downtrend":
            # Slightly more conservative in weak downtrends
            self._adjust_trend_params(adjusted_params, aggressive=False, direction=-1)
        
        elif self.current_regime == "volatile":
            # More conservative in volatile regimes
            self._adjust_volatile_params(adjusted_params)
        
        elif self.current_regime == "ranging":
            # Optimize for mean-reversion in ranging markets
            self._adjust_ranging_params(adjusted_params)
        
        # Log the adjustments
        logger.debug(f"Adjusted {strategy_name} parameters for {self.current_regime} regime")
        
        return adjusted_params
    
    def _adjust_trend_params(self, params: Dict[str, Any], aggressive: bool, direction: int):
        """Adjust parameters for trending regimes"""
        # These are generic adjustments; specific strategies might need different ones
        if 'stop_loss_pct' in params:
            # Wider stops in strong trends, tighter in weak trends
            factor = 1.2 if aggressive else 0.9
            params['stop_loss_pct'] = params['stop_loss_pct'] * factor
        
        if 'take_profit_pct' in params:
            # Higher targets in strong trends
            factor = 1.3 if aggressive else 1.1
            params['take_profit_pct'] = params['take_profit_pct'] * factor
        
        if 'position_size_pct' in params:
            # Larger positions in strong trends, smaller in weak trends
            factor = 1.2 if aggressive else 0.9
            params['position_size_pct'] = params['position_size_pct'] * factor
    
    def _adjust_volatile_params(self, params: Dict[str, Any]):
        """Adjust parameters for volatile regimes"""
        # Reduce position sizes
        if 'position_size_pct' in params:
            params['position_size_pct'] = params['position_size_pct'] * 0.7
        
        # Tighter stops
        if 'stop_loss_pct' in params:
            params['stop_loss_pct'] = params['stop_loss_pct'] * 0.8
        
        # Faster exits
        if 'exit_after_bars' in params:
            params['exit_after_bars'] = int(params['exit_after_bars'] * 0.7)
    
    def _adjust_ranging_params(self, params: Dict[str, Any]):
        """Adjust parameters for ranging regimes"""
        # Mean-reversion optimizations
        if 'oversold_threshold' in params:
            params['oversold_threshold'] = params['oversold_threshold'] * 1.1
        
        if 'overbought_threshold' in params:
            params['overbought_threshold'] = params['overbought_threshold'] * 0.9
        
        if 'bollinger_dev' in params:
            params['bollinger_dev'] = params['bollinger_dev'] * 0.8  # Tighter bands
    
    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get information about the current detected market regime
        
        Returns:
            Dict with current regime information
        """
        if not self.current_regime:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'description': 'No regime detected yet'
            }
        
        # Find the regime description
        description = self._get_regime_description(self.current_regime)
        
        return {
            'regime': self.current_regime,
            'confidence': self.regime_confidence,
            'description': description,
            'last_update': self.regime_history['timestamp'].iloc[-1] if len(self.regime_history) > 0 else None
        }
    
    def get_regime_history(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Get history of regime changes
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with regime history
        """
        if len(self.regime_history) == 0:
            return pd.DataFrame()
        
        # Filter by date
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        filtered_history = self.regime_history[self.regime_history['timestamp'] >= cutoff_date]
        
        return filtered_history
