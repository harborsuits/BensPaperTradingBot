"""
ML-Enhanced Market Regime Detector

Uses machine learning to identify market regimes with higher accuracy,
combining supervised models (RandomForest, XGBoost) with unsupervised
approaches for a more robust regime classification system.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import original regime detector as base
from trading_bot.ml_pipeline.market_regime_detector import MarketRegimeDetector

# Optional XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Setup logger
logger = logging.getLogger(__name__)

class MLRegimeDetector(MarketRegimeDetector):
    """
    ML-Enhanced Market Regime Detector
    
    Extends the base MarketRegimeDetector with machine learning capabilities
    to provide more accurate and reliable regime detection.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ML regime detector
        
        Args:
            config: Configuration dictionary
        """
        # Initialize base class
        super().__init__(config)
        
        # ML-specific configuration
        self.config = config or {}
        self.models_dir = self.config.get('models_dir', 'models')
        self.model_name = self.config.get('model_name', 'regime_classifier')
        
        # Feature engineering parameters
        self.feature_window = self.config.get('feature_window', 20)
        self.prediction_threshold = self.config.get('prediction_threshold', 0.6)
        
        # Additional ML features
        self.ml_features = self.config.get('ml_features', [
            'price_momentum', 'volume_momentum', 'volatility_trend',
            'regime_persistence', 'market_correlation', 'sector_rotation'
        ])
        
        # Model selection
        self.model_type = self.config.get('model_type', 'randomforest')
        
        # Initialize ML components
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Label encodings
        self.regime_encodings = {
            'bullish_trend': 0,
            'bearish_trend': 1,
            'ranging': 2,
            'volatile': 3,
            'consolidation': 4
        }
        self.regime_decodings = {v: k for k, v in self.regime_encodings.items()}
        
        # Load model if available
        self._load_model()
        
        logger.info("ML-Enhanced Regime Detector initialized")
    
    def _load_model(self) -> bool:
        """
        Load trained model if available
        
        Returns:
            Boolean indicating if model was loaded successfully
        """
        model_path = os.path.join(self.models_dir, f"{self.model_name}.pkl")
        scaler_path = os.path.join(self.models_dir, f"{self.model_name}_scaler.pkl")
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.classifier = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded regime detection model from {model_path}")
                return True
            else:
                logger.info("No pre-trained regime detection model found")
                return False
        except Exception as e:
            logger.error(f"Error loading regime detection model: {e}")
            return False
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, f"{self.model_name}.pkl")
            scaler_path = os.path.join(self.models_dir, f"{self.model_name}_scaler.pkl")
            
            joblib.dump(self.classifier, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved regime detection model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving regime detection model: {e}")
    
    def train_model(self, historical_data: pd.DataFrame, 
                    labeled_regimes: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train regime detection model on historical data
        
        Args:
            historical_data: Historical OHLCV data
            labeled_regimes: Optional DataFrame with labeled regimes
            
        Returns:
            Dict with training results and metrics
        """
        logger.info("Training ML regime detection model")
        
        # Create features from historical data
        features = self._extract_ml_features(historical_data)
        
        # Generate labels from unsupervised method or use provided labels
        if labeled_regimes is None:
            logger.info("No labeled regimes provided, using unsupervised detection")
            labels = self._generate_regime_labels(historical_data)
        else:
            logger.info("Using provided regime labels")
            labels = labeled_regimes['regime'].map(self.regime_encodings).values
        
        # Ensure we have matching feature and label lengths
        if len(features) != len(labels):
            min_len = min(len(features), len(labels))
            features = features[-min_len:]
            labels = labels[-min_len:]
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the appropriate model
        if self.model_type == 'randomforest':
            logger.info("Training RandomForest classifier")
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
            self.classifier.fit(X_train_scaled, y_train)
        
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            logger.info("Training XGBoost classifier")
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.classifier.fit(X_train_scaled, y_train)
        
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM classifier")
            self.classifier = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.classifier.fit(X_train_scaled, y_train)
        
        else:
            logger.warning(f"Model type {self.model_type} not available, using RandomForest")
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
            self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Save model
        self._save_model()
        
        # Log results
        logger.info(f"Regime detection model trained with accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'model_type': self.model_type,
            'feature_count': features.shape[1],
            'training_samples': len(X_train)
        }
    
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regime using ML model
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            Dict with regime information
        """
        # Fall back to unsupervised method if model not available
        if self.classifier is None:
            logger.info("ML model not available, using unsupervised method")
            return super().detect_regime(data)
        
        # Check for minimum data
        if len(data) < self.feature_window + 10:
            logger.warning("Insufficient data for ML regime detection")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'characteristics': {},
                'description': 'Insufficient data for ML analysis'
            }
        
        # Extract ML features
        features = self._extract_ml_features(data)
        
        # Get only the most recent feature set
        recent_features = features.iloc[-1:].values
        
        # Scale features
        scaled_features = self.scaler.transform(recent_features)
        
        # Predict regime
        regime_pred = self.classifier.predict(scaled_features)[0]
        regime_probs = self.classifier.predict_proba(scaled_features)[0]
        
        # Get confidence as max probability
        confidence = max(regime_probs)
        
        # Map regime to name
        regime_name = self.regime_decodings.get(regime_pred, 'unknown')
        
        # Also run traditional method for comparison
        traditional_regime = super()._simple_regime_classification(data)
        
        # Use ensemble approach if confidence is low
        if confidence < self.prediction_threshold:
            logger.info(f"Low confidence ({confidence:.2f}), using ensemble approach")
            # Use traditional method's regime if ML confidence is low
            regime_name = traditional_regime['regime']
            confidence = (confidence + traditional_regime['confidence']) / 2
        
        # Create regime info
        regime_info = {
            'regime': regime_name,
            'confidence': float(confidence),
            'ml_confidence': float(confidence),
            'traditional_regime': traditional_regime['regime'],
            'traditional_confidence': traditional_regime['confidence'],
            'description': self._get_regime_description(regime_name),
            'timestamp': pd.Timestamp.now()
        }
        
        # Update regime history
        self._update_regime_history(regime_info)
        
        # Update current regime
        self.current_regime = regime_name
        self.regime_confidence = confidence
        
        return regime_info
    
    def _extract_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ML features from price data
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with ML features
        """
        # Create feature dataframe
        features = pd.DataFrame()
        
        # Make sure we have enough data
        if len(data) < self.feature_window:
            logger.warning(f"Not enough data for feature extraction. Need at least {self.feature_window} periods.")
            return pd.DataFrame()
        
        # Price-based features
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', pd.Series(np.ones(len(close)), index=close.index))
        
        # Basic features
        
        # Relative price changes at different timeframes
        for window in [1, 3, 5, 10, 20]:
            if len(close) > window:
                features[f'return_{window}d'] = close.pct_change(window).shift(1)
        
        # Volatility at different timeframes
        for window in [5, 10, 20]:
            if len(close) > window:
                features[f'volatility_{window}d'] = close.pct_change().rolling(window).std().shift(1)
        
        # Momentum indicators
        features['rsi_14'] = self._calculate_rsi(close, 14)
        features['roc_10'] = (close / close.shift(10) - 1).shift(1)
        
        # Trend strength
        features['adx_14'] = self._calculate_adx(high, low, close, 14)
        
        # Volatility indicators
        features['bollinger_width'] = self._calculate_bollinger_width(close, 20, 2)
        features['atr_14'] = self._calculate_atr(high, low, close, 14) / close
        
        # Volume profile
        if 'volume' in data.columns:
            features['volume_trend_10'] = (volume / volume.rolling(10).mean() - 1).shift(1)
            features['vol_price_trend'] = (volume * close.pct_change().abs()).shift(1)
        
        # Support/resistance proximity
        features['support_distance'] = (close / low.rolling(20).min() - 1).shift(1)
        features['resistance_distance'] = (high.rolling(20).max() / close - 1).shift(1)
        
        # Price patterns
        features['higher_highs'] = (high > high.shift(1)).rolling(5).sum().shift(1)
        features['lower_lows'] = (low < low.shift(1)).rolling(5).sum().shift(1)
        
        # Detect round numbers (psychological levels)
        close_rounded = np.round(close / 10) * 10
        features['near_round_number'] = (np.abs(close - close_rounded) / close < 0.01).astype(int).shift(1)
        
        # Advanced features
        
        # Fractal dimension (complexity of price movement)
        features['fractal_dimension'] = self._calculate_fractal_dimension(close, 20)
        
        # Trend reversal probability
        features['reversal_probability'] = self._calculate_reversal_prob(close, high, low)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = -delta.clip(upper=0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1)  # Shift to avoid lookahead bias
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Calculate Plus/Minus Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.shift(1)  # Shift to avoid lookahead bias
    
    def _calculate_bollinger_width(self, prices: pd.Series, 
                                  window: int, num_std: float) -> pd.Series:
        """Calculate Bollinger Band width"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        bandwidth = (upper_band - lower_band) / rolling_mean
        return bandwidth.shift(1)  # Shift to avoid lookahead bias
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.shift(1)  # Shift to avoid lookahead bias
    
    def _calculate_fractal_dimension(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate fractal dimension (measure of price complexity)"""
        # Use sliding window to calculate fractal dimension
        result = []
        
        for i in range(window, len(prices)):
            window_prices = prices.iloc[i-window:i]
            # Normalize prices to 0-1 range
            norm_prices = (window_prices - window_prices.min()) / (window_prices.max() - window_prices.min())
            # Calculate approximate fractal dimension using box-counting method
            changes = np.abs(np.diff(norm_prices))
            non_zero_changes = changes[changes > 0]
            if len(non_zero_changes) > 1:
                # Use log-log relationship to estimate fractal dimension
                fd = 1 + (np.log(len(non_zero_changes)) / np.log(1/non_zero_changes.mean()))
                result.append(min(fd, 2.0))  # Cap at 2.0 (maximum dimension in 2D space)
            else:
                result.append(1.0)  # Default to 1.0 (straight line)
        
        # Add NaN values for initial periods
        result = [np.nan] * window + result
        return pd.Series(result, index=prices.index).shift(1)  # Shift to avoid lookahead bias
    
    def _calculate_reversal_prob(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate probability of trend reversal based on candlestick patterns"""
        # Price movement
        body = abs(close - close.shift())
        # Upper and lower shadows
        upper_shadow = high - np.maximum(close, close.shift())
        lower_shadow = np.minimum(close, close.shift()) - low
        
        # Doji pattern (small body relative to shadows)
        doji = body / (high - low) < 0.1
        
        # Hammer/hanging man pattern
        small_upper = upper_shadow < body * 0.5
        large_lower = lower_shadow > body * 2
        hammer = small_upper & large_lower
        
        # Shooting star pattern
        small_lower = lower_shadow < body * 0.5
        large_upper = upper_shadow > body * 2
        shooting_star = small_lower & large_upper
        
        # Engulfing pattern
        bullish_engulf = (close > close.shift()) & (close > close.shift(2)) & (close.shift() < close.shift(2))
        bearish_engulf = (close < close.shift()) & (close < close.shift(2)) & (close.shift() > close.shift(2))
        
        # Calculate reversal probability
        reversal_prob = 0.1 * doji + 0.2 * hammer + 0.2 * shooting_star + 0.25 * bullish_engulf + 0.25 * bearish_engulf
        
        # Combine with overbought/oversold conditions from RSI
        rsi = self._calculate_rsi(close, 14)
        reversal_prob = reversal_prob + 0.5 * (rsi > 70) + 0.5 * (rsi < 30)
        
        # Normalize to 0-1 range
        reversal_prob = reversal_prob.clip(0, 1)
        
        return reversal_prob.shift(1)  # Shift to avoid lookahead bias
    
    def _generate_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate regime labels using unsupervised methods
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            Array of regime labels (integers)
        """
        # Calculate regime features
        features = self._calculate_regime_features(data)
        
        # Detect regimes using base class method
        regimes = []
        
        # Use a sliding window to detect regimes
        window_size = min(self.lookback_window, len(data) // 10)
        
        for i in range(window_size, len(data), window_size // 2):
            window_data = data.iloc[max(0, i-window_size):i]
            regime_info = super()._simple_regime_classification(window_data)
            
            # Map regime to integer encoding
            regime_code = self.regime_encodings.get(regime_info['regime'], 0)
            
            # Repeat the regime label for each day in the window
            regimes.extend([regime_code] * (min(window_size // 2, len(data) - i + window_size // 2)))
        
        # Ensure we have the right number of labels
        if len(regimes) < len(data):
            # Fill the beginning with the first regime
            regimes = [regimes[0]] * (len(data) - len(regimes)) + regimes
        elif len(regimes) > len(data):
            # Trim excess labels
            regimes = regimes[:len(data)]
        
        return np.array(regimes)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from ML model
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.classifier is None:
            logger.warning("No trained classifier available")
            return pd.DataFrame()
        
        try:
            # Get feature names from recent feature extraction
            dummy_data = pd.DataFrame(index=range(100))
            dummy_data['close'] = np.random.random(100) * 100
            dummy_data['high'] = dummy_data['close'] * (1 + np.random.random(100) * 0.02)
            dummy_data['low'] = dummy_data['close'] * (1 - np.random.random(100) * 0.02)
            dummy_data['volume'] = np.random.random(100) * 1000
            
            features = self._extract_ml_features(dummy_data)
            feature_names = features.columns.tolist()
            
            # Get importance from model
            if hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
            else:
                logger.warning("Model doesn't provide feature importances")
                return pd.DataFrame()
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def get_regime_forecast(self, data: pd.DataFrame, forecast_days: int = 5) -> Dict[str, Any]:
        """
        Generate a forecast of future regime probabilities
        
        Args:
            data: Historical OHLCV data
            forecast_days: Number of days to forecast
            
        Returns:
            Dict with forecast information
        """
        if self.classifier is None:
            logger.warning("No trained classifier available for forecasting")
            return {
                'forecast': None,
                'current_regime': self.current_regime,
                'error': 'No trained classifier available'
            }
        
        try:
            # Detect current regime
            current_regime = self.detect_regime(data)
            
            # Calculate regime transition probabilities from history
            if len(self.regime_history) > 10:
                transitions = {}
                
                prev_regime = None
                for _, row in self.regime_history.iterrows():
                    if prev_regime is not None:
                        if prev_regime not in transitions:
                            transitions[prev_regime] = {}
                        
                        current = row['regime']
                        if current not in transitions[prev_regime]:
                            transitions[prev_regime][current] = 0
                        
                        transitions[prev_regime][current] += 1
                    
                    prev_regime = row['regime']
                
                # Convert counts to probabilities
                transition_probs = {}
                for from_regime, to_regimes in transitions.items():
                    total = sum(to_regimes.values())
                    transition_probs[from_regime] = {
                        to_regime: count / total 
                        for to_regime, count in to_regimes.items()
                    }
            else:
                # If not enough history, use default persistence probabilities
                transition_probs = {
                    regime: {regime: 0.7} for regime in self.regime_decodings.values()
                }
                
                # Add smaller probabilities for transitions
                for from_regime in self.regime_decodings.values():
                    for to_regime in self.regime_decodings.values():
                        if from_regime != to_regime:
                            transition_probs[from_regime][to_regime] = 0.3 / (len(self.regime_decodings) - 1)
            
            # Generate forecast using Markov chain
            forecast = []
            current = current_regime['regime']
            
            for day in range(forecast_days):
                if current in transition_probs:
                    # Sort regimes by probability (descending)
                    next_regimes = sorted(
                        transition_probs[current].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Get top 3 most likely next regimes
                    forecast.append({
                        'day': day + 1,
                        'date': (pd.Timestamp.now() + pd.Timedelta(days=day+1)).strftime('%Y-%m-%d'),
                        'regimes': [
                            {
                                'regime': regime,
                                'probability': prob,
                                'description': self._get_regime_description(regime)
                            }
                            for regime, prob in next_regimes[:3]
                        ]
                    })
                    
                    # Use most likely regime for next iteration
                    current = next_regimes[0][0]
                else:
                    # If current regime has no transitions, use current regime
                    forecast.append({
                        'day': day + 1,
                        'date': (pd.Timestamp.now() + pd.Timedelta(days=day+1)).strftime('%Y-%m-%d'),
                        'regimes': [
                            {
                                'regime': current,
                                'probability': 0.9,
                                'description': self._get_regime_description(current)
                            }
                        ]
                    })
            
            return {
                'current_regime': current_regime,
                'forecast': forecast,
                'transition_matrix': transition_probs
            }
        
        except Exception as e:
            logger.error(f"Error generating regime forecast: {e}")
            return {
                'forecast': None,
                'current_regime': self.current_regime,
                'error': str(e)
            }
