"""
Price Prediction Model

This module implements a machine learning model for short-term price movement prediction
using technical indicators, price action patterns, and market microstructure features.
"""

import numpy as np
import pandas as pd
import logging
import joblib
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator

logger = logging.getLogger(__name__)

class PricePredictionModel:
    """
    Machine learning model for predicting short-term price movements.
    
    Uses a combination of technical indicators, price patterns, and market 
    microstructure features to predict future price movement with a confidence score.
    """
    
    def __init__(self, 
                 prediction_horizon=5, 
                 confidence_threshold=0.65,
                 model_type='gradient_boosting',
                 feature_groups=None,
                 model_path=None):
        """
        Initialize the price prediction model.
        
        Args:
            prediction_horizon (int): Number of periods to predict into the future
            confidence_threshold (float): Threshold for high-confidence predictions
            model_type (str): Type of model to use ('linear', 'random_forest', 'gradient_boosting')
            feature_groups (list): List of feature groups to use ('price', 'volume', 'technical', 'pattern')
            model_path (str): Path to save/load model
        """
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.feature_groups = feature_groups if feature_groups else ['price', 'volume', 'technical']
        self.model_path = model_path
        
        # Initialize models dictionary (for different symbols/timeframes)
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.model_metadata = {}
        
        # Model configuration
        self.model_config = {
            'linear': {
                'model': LinearRegression(),
                'params': {}
            },
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            }
        }
        
        logger.info(f"Price prediction model initialized with horizon={prediction_horizon}, model={model_type}")
    
    def create_features(self, data):
        """
        Create features from OHLCV data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        if data.empty:
            logger.error("Empty dataframe provided for feature creation")
            return pd.DataFrame()
            
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in df.columns for col in required_columns):
            # Try to standardize column names
            for col in df.columns:
                if col.lower() in ['open', 'o']:
                    df['open'] = df[col]
                elif col.lower() in ['high', 'h']:
                    df['high'] = df[col]
                elif col.lower() in ['low', 'l']:
                    df['low'] = df[col]
                elif col.lower() in ['close', 'c']:
                    df['close'] = df[col]
                elif col.lower() in ['volume', 'vol', 'v']:
                    df['volume'] = df[col]
        
        # Check again after standardization
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        if 'price' in self.feature_groups:
            # Basic price features
            features['close'] = df['close']
            features['open'] = df['open']
            features['high'] = df['high']
            features['low'] = df['low']
            
            # Price differences and ratios
            features['price_change'] = df['close'].pct_change()
            features['high_low_diff'] = (df['high'] - df['low']) / df['close']
            features['close_open_diff'] = (df['close'] - df['open']) / df['open']
            
            # Moving averages and differences
            for window in [5, 10, 20, 50]:
                features[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                features[f'ma_diff_{window}'] = features[f'ma_{window}'] / df['close'] - 1
            
            # Returns over different periods
            for window in [1, 3, 5, 10]:
                features[f'return_{window}d'] = df['close'].pct_change(periods=window)
            
            # Range-based features
            features['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Gap analysis
            features['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
            features['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
            features['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Volume-based features
        if 'volume' in self.feature_groups:
            # Basic volume
            features['volume'] = df['volume']
            features['volume_change'] = df['volume'].pct_change()
            
            # Volume moving averages and ratios
            for window in [5, 10, 20]:
                features[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                features[f'volume_ratio_{window}'] = df['volume'] / features[f'volume_ma_{window}']
            
            # Price-volume relationships
            features['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
            
            # OBV
            obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            features['obv'] = obv.on_balance_volume()
            features['obv_change'] = features['obv'].pct_change()
            
            # VPT
            vpt = VolumePriceTrendIndicator(close=df['close'], volume=df['volume'])
            features['vpt'] = vpt.volume_price_trend()
            
            # Volume profile
            features['volume_high_ratio'] = df['volume'] * (df['close'] >= df['open']).astype(int)
            features['volume_low_ratio'] = df['volume'] * (df['close'] < df['open']).astype(int)
            
            # Abnormal volume detection
            features['abnormal_volume'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # Technical indicators
        if 'technical' in self.feature_groups:
            # Trend indicators
            macd = MACD(close=df['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            
            for period in [14, 21]:
                rsi = RSIIndicator(close=df['close'], window=period)
                features[f'rsi_{period}'] = rsi.rsi()
            
            # Volatility indicators
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
            features['atr'] = atr.average_true_range()
            
            bb = BollingerBands(close=df['close'])
            features['bb_high'] = bb.bollinger_hband()
            features['bb_low'] = bb.bollinger_lband()
            features['bb_width'] = (features['bb_high'] - features['bb_low']) / df['close']
            features['bb_position'] = (df['close'] - features['bb_low']) / (features['bb_high'] - features['bb_low'])
            
            # Momentum indicators
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # Support/resistance proximity
            for window in [20, 50]:
                features[f'resistance_{window}'] = df['high'].rolling(window=window).max()
                features[f'support_{window}'] = df['low'].rolling(window=window).min()
                features[f'res_proximity_{window}'] = (features[f'resistance_{window}'] - df['close']) / df['close']
                features[f'sup_proximity_{window}'] = (df['close'] - features[f'support_{window}']) / df['close']
        
        # Pattern recognition
        if 'pattern' in self.feature_groups:
            # Candle pattern features
            features['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)
            features['hammer'] = (
                ((df['high'] - df['low']) > 3 * (df['open'] - df['close'])) & 
                ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6) &
                ((df['open'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6)
            ).astype(int)
            
            # Engulfing patterns
            features['bullish_engulfing'] = (
                (df['close'].shift(1) < df['open'].shift(1)) &  # Prior candle is bearish
                (df['open'] < df['close'].shift(1)) &           # Open below prior close
                (df['close'] > df['open'].shift(1))             # Close above prior open
            ).astype(int)
            
            features['bearish_engulfing'] = (
                (df['close'].shift(1) > df['open'].shift(1)) &  # Prior candle is bullish
                (df['open'] > df['close'].shift(1)) &           # Open above prior close
                (df['close'] < df['open'].shift(1))             # Close below prior open
            ).astype(int)
            
            # Pinbar detection
            features['pinbar_up'] = (
                (df['high'] - df['close']) / (df['high'] - df['low'] + 0.001) < 0.25 &
                (df['close'] > df['open'])
            ).astype(int)
            
            features['pinbar_down'] = (
                (df['close'] - df['low']) / (df['high'] - df['low'] + 0.001) < 0.25 &
                (df['close'] < df['open'])
            ).astype(int)
        
        # Drop na values that might have been introduced by rolling windows
        features = features.dropna()
        
        # Log the shape of features DataFrame
        logger.info(f"Created {features.shape[1]} features from {data.shape[0]} data points")
        
        return features
    
    def prepare_training_data(self, data):
        """
        Prepare training data with features and target variables.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            tuple: X (features), y (target), feature_names
        """
        # Create features
        features_df = self.create_features(data)
        if features_df.empty:
            return None, None, None
            
        # Create target variable (future price change)
        target = features_df['close'].shift(-self.prediction_horizon) / features_df['close'] - 1
        
        # Remove the target from features
        features_df = features_df.drop(['close'], axis=1)
        
        # Align features and target
        features_df = features_df[:-self.prediction_horizon]
        target = target[:-self.prediction_horizon]
        
        # Remove any rows with NaN values
        valid_indices = ~(features_df.isna().any(axis=1) | target.isna())
        features_df = features_df[valid_indices]
        target = target[valid_indices]
        
        # Get feature names
        feature_names = features_df.columns.tolist()
        
        # Convert to numpy arrays
        X = features_df.values
        y = target.values
        
        return X, y, feature_names
    
    def train(self, data, model_key='default'):
        """
        Train the price prediction model.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            model_key (str): Key to identify this model (e.g., symbol_timeframe)
            
        Returns:
            dict: Training metrics
        """
        logger.info(f"Training price prediction model for {model_key}")
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_data(data)
        if X is None or y is None:
            logger.error("Failed to prepare training data")
            return {"error": "Failed to prepare training data"}
        
        # Split into train/test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Standard scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store the scaler
        self.scalers[model_key] = scaler
        
        # Create and train the model
        model_config = self.model_config.get(self.model_type, self.model_config['gradient_boosting'])
        model = model_config['model']
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Store the model
        self.models[model_key] = model
        
        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            self.feature_importances[model_key] = feature_importance
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        direction_correct = np.mean((y_test > 0) == (y_pred > 0))
        
        # Store metadata
        self.model_metadata[model_key] = {
            'trained_at': datetime.now().isoformat(),
            'data_points': len(X),
            'features': len(feature_names),
            'feature_names': feature_names,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_correct
            },
            'prediction_horizon': self.prediction_horizon,
            'model_type': self.model_type,
            'feature_groups': self.feature_groups
        }
        
        logger.info(f"Model trained for {model_key} with direction accuracy: {direction_correct:.4f}, MSE: {mse:.6f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_correct,
            'model_key': model_key
        }
    
    def predict(self, data, model_key='default'):
        """
        Predict future price movement for the given data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            model_key (str): Key to identify which model to use
            
        Returns:
            dict: Prediction results with direction, confidence and target prices
        """
        # Check if model exists
        if model_key not in self.models:
            if 'default' in self.models:
                logger.warning(f"Model for {model_key} not found, using default model")
                model_key = 'default'
            else:
                logger.error(f"No model found for {model_key} and no default model available")
                return {"error": "Model not found"}
        
        # Get the model and scaler
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Create features
        features_df = self.create_features(data)
        if features_df.empty:
            return {"error": "Failed to create features"}
        
        # Get the latest data point for prediction
        latest_features = features_df.iloc[-1:].copy()
        
        # Remove close price from features
        if 'close' in latest_features.columns:
            current_price = latest_features['close'].values[0]
            latest_features = latest_features.drop(['close'], axis=1)
        else:
            current_price = data['close'].iloc[-1]
        
        # Scale features
        X = scaler.transform(latest_features.values)
        
        # Make prediction
        predicted_change = model.predict(X)[0]
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_change)
        
        # Calculate confidence
        confidence = self._calculate_confidence(model, X, predicted_change, model_key)
        
        # Determine direction
        direction = "up" if predicted_change > 0 else "down"
        
        # Calculate price range based on confidence
        prediction_error = 1.0 - confidence
        price_range = {
            'lower': predicted_price * (1 - prediction_error),
            'upper': predicted_price * (1 + prediction_error)
        }
        
        # Format results
        result = {
            'direction': direction,
            'confidence': confidence,
            'predicted_change': predicted_change,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_range': price_range,
            'timestamp': datetime.now().isoformat(),
            'horizon': self.prediction_horizon,
            'high_confidence': confidence >= self.confidence_threshold
        }
        
        return result
    
    def _calculate_confidence(self, model, X, prediction, model_key):
        """
        Calculate confidence score for the prediction.
        
        Args:
            model: Trained model
            X: Scaled feature values
            prediction: The predicted value
            model_key: Key for the model metadata
            
        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence on model type
        if self.model_type == 'random_forest' and hasattr(model, 'estimators_'):
            # For random forest, use the standard deviation of tree predictions
            tree_predictions = np.array([tree.predict(X)[0] for tree in model.estimators_])
            std = np.std(tree_predictions)
            
            # Normalize std to a confidence score (higher std = lower confidence)
            # This approach is simplistic but effective
            confidence = 1.0 / (1.0 + 5.0 * std)
            
        elif self.model_type == 'gradient_boosting' and hasattr(model, 'estimators_'):
            # For gradient boosting, use prediction intervals
            # This is an approximation
            residuals = model.train_score_
            confidence = 1.0 / (1.0 + abs(np.std(residuals) / (abs(prediction) + 0.001)))
            
        else:
            # For other models, use a fixed confidence based on model accuracy
            if model_key in self.model_metadata:
                accuracy = self.model_metadata[model_key]['metrics'].get('direction_accuracy', 0.6)
                confidence = min(0.95, max(0.5, accuracy))
            else:
                confidence = 0.6  # Default moderate confidence
        
        # Cap confidence
        confidence = min(0.95, max(0.5, confidence))
        
        # Adjust confidence based on prediction size
        # Extreme predictions should have lower confidence
        if abs(prediction) > 0.05:  # 5% price change is significant
            confidence = confidence * (0.05 / abs(prediction))**0.5
            confidence = min(0.9, max(0.4, confidence))
        
        return float(confidence)
    
    def get_feature_importance(self, model_key='default'):
        """
        Get feature importance for the trained model.
        
        Args:
            model_key (str): Key to identify which model to use
            
        Returns:
            dict: Feature importance values
        """
        if model_key not in self.feature_importances:
            if 'default' in self.feature_importances:
                model_key = 'default'
            else:
                return {"error": "No feature importance available"}
        
        # Sort by importance
        importance = self.feature_importances[model_key]
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self):
        """
        Save the trained models, scalers, and metadata to disk.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.model_path:
            logger.warning("No model path specified, model not saved")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save models and scalers
            for key in self.models:
                model_path = f"{self.model_path}_{key}_model.joblib"
                scaler_path = f"{self.model_path}_{key}_scaler.joblib"
                joblib.dump(self.models[key], model_path)
                joblib.dump(self.scalers[key], scaler_path)
            
            # Save metadata and feature importances
            metadata_path = f"{self.model_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'metadata': self.model_metadata,
                    'feature_importances': self.feature_importances,
                    'model_keys': list(self.models.keys()),
                    'config': {
                        'prediction_horizon': self.prediction_horizon,
                        'confidence_threshold': self.confidence_threshold,
                        'model_type': self.model_type,
                        'feature_groups': self.feature_groups
                    }
                }, f)
                
            logger.info(f"Saved {len(self.models)} models to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """
        Load trained models, scalers, and metadata from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not self.model_path:
            logger.warning("No model path specified, model not loaded")
            return False
        
        try:
            # Load metadata to get model keys
            metadata_path = f"{self.model_path}_metadata.json"
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found at {metadata_path}")
                return False
                
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                
            # Update metadata
            self.model_metadata = data.get('metadata', {})
            self.feature_importances = data.get('feature_importances', {})
            
            # Update configuration
            config = data.get('config', {})
            self.prediction_horizon = config.get('prediction_horizon', self.prediction_horizon)
            self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            self.model_type = config.get('model_type', self.model_type)
            self.feature_groups = config.get('feature_groups', self.feature_groups)
            
            # Load models and scalers
            for key in data.get('model_keys', []):
                model_path = f"{self.model_path}_{key}_model.joblib"
                scaler_path = f"{self.model_path}_{key}_scaler.joblib"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[key] = joblib.load(model_path)
                    self.scalers[key] = joblib.load(scaler_path)
                else:
                    logger.warning(f"Model or scaler files not found for {key}")
            
            logger.info(f"Loaded {len(self.models)} models from {self.model_path}")
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False 