#!/usr/bin/env python3
"""
Machine learning module for detecting market microstructure anomalies and unusual patterns.
Provides anomaly detection capabilities for identifying potentially abnormal market behavior.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import json
import os
from datetime import datetime, timedelta
import ta
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, Dropout
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class MarketAnomalyDetector:
    """
    Machine learning-based market anomaly detector.
    
    Detects unusual patterns in market microstructure data such as:
    - Order flow imbalances
    - Unusual price/volume movements
    - Liquidity anomalies
    - Spoofing and layering patterns
    - Flash crash precursors
    """
    
    def __init__(self,
                 symbol: str,
                 lookback_window: int = 20,
                 alert_threshold: float = 0.9,
                 model_dir: str = "models/anomaly_detection",
                 use_autoencoder: bool = True,
                 contamination: float = 0.01):
        """
        Initialize the market anomaly detector.
        
        Args:
            symbol: The trading symbol to monitor
            lookback_window: Number of periods to include in sequential analysis
            alert_threshold: Threshold for anomaly score to trigger alerts (0-1)
            model_dir: Directory to save/load trained models
            use_autoencoder: Whether to use LSTM autoencoder for sequential patterns
            contamination: Expected proportion of anomalies in the data (for IsolationForest)
        """
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.alert_threshold = alert_threshold
        self.model_dir = model_dir
        self.use_autoencoder = use_autoencoder
        self.contamination = contamination
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.isolation_forest = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        
        # Store detection history
        self.anomaly_history = []
        self.latest_scores = {}
        
        # Model files
        self.model_file_if = os.path.join(model_dir, f"{symbol}_isolation_forest.pkl")
        self.model_file_ae = os.path.join(model_dir, f"{symbol}_autoencoder.h5")
        self.scaler_file = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        self.feature_importance_file = os.path.join(model_dir, f"{symbol}_feature_importance.json")
        
        logger.info(f"Initialized MarketAnomalyDetector for {symbol}")
    
    def create_features(self, data, orderbook_data=None, trade_data=None, tick_data=None):
        """
        Create features for anomaly detection from market data.
        
        Args:
            data: Market OHLCV data
            orderbook_data: Order book data (optional)
            trade_data: Individual trade data (optional)
            tick_data: Tick-by-tick data (optional)
            
        Returns:
            DataFrame: Features for anomaly detection
        """
        # Create a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Basic OHLCV features
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # Price action features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volatility features
            df['returns_volatility'] = df['returns'].rolling(window=self.window_size).std()
            df['high_low_volatility'] = df['high_low_range'].rolling(window=self.window_size).std()
            
            # Price acceleration
            df['price_acceleration'] = df['returns'].diff()
            
            # Z-scores of returns
            df['returns_zscore'] = (df['returns'] - df['returns'].rolling(window=self.window_size).mean()) / \
                                  df['returns'].rolling(window=self.window_size).std()
            
            # Extremes in price movement
            df['extreme_up'] = (df['high'] / df['open'] - 1) * 100
            df['extreme_down'] = (df['low'] / df['open'] - 1) * 100
            
            # Gap indicators
            df['gap_up'] = (df['open'] / df['close'].shift(1) - 1) * 100
            df['gap_down'] = (df['close'].shift(1) / df['open'] - 1) * 100
            
            # Price reversal indicators
            df['price_reversal'] = ((df['close'] - df['open']) * 
                                   (df['open'] - df['close'].shift(1))) < 0
            
            # Consecutive moves
            for i in range(1, min(5, self.look_back)):
                df[f'consecutive_move_{i}'] = np.sign(df['returns']) == np.sign(df['returns'].shift(i))
        
        # Volume features
        if 'volume' in df.columns:
            # Basic volume indicators
            df['volume_change'] = df['volume'].pct_change()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=self.window_size).mean()) / \
                                 df['volume'].rolling(window=self.window_size).std()
            
            # Volume spikes
            df['volume_spike'] = df['volume'] / df['volume'].rolling(window=self.window_size).mean()
            
            # Volume and price relationship
            df['volume_price_correlation'] = df['returns'].rolling(window=self.window_size).corr(df['volume_change'])
            
            # Relative volume
            df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Volume momentum
            df['volume_momentum'] = df['volume'] - df['volume'].shift(5)
            df['volume_momentum_zscore'] = (df['volume_momentum'] - df['volume_momentum'].rolling(window=self.window_size).mean()) / \
                                          df['volume_momentum'].rolling(window=self.window_size).std()
            
            # Unusual volume without price movement
            df['volume_without_price_move'] = df['volume_spike'] * (1 - np.abs(df['returns_zscore']))
            
            # Price to volume ratio
            df['price_to_volume'] = np.abs(df['returns']) / df['volume']
            df['price_to_volume_zscore'] = (df['price_to_volume'] - df['price_to_volume'].rolling(window=self.window_size).mean()) / \
                                          df['price_to_volume'].rolling(window=self.window_size).std()
        
        # Order book features (if available)
        if orderbook_data is not None:
            # Merge orderbook data with main dataframe
            if isinstance(orderbook_data, pd.DataFrame) and len(orderbook_data) > 0:
                # Extract relevant order book features
                try:
                    # Order book imbalance
                    df['bid_ask_spread'] = orderbook_data['ask_price_0'] - orderbook_data['bid_price_0']
                    df['relative_spread'] = df['bid_ask_spread'] / orderbook_data['ask_price_0']
                    
                    # Calculate order book imbalance
                    bid_volume = sum([orderbook_data[f'bid_size_{i}'] for i in range(5) if f'bid_size_{i}' in orderbook_data])
                    ask_volume = sum([orderbook_data[f'ask_size_{i}'] for i in range(5) if f'ask_size_{i}' in orderbook_data])
                    
                    df['book_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                    df['book_pressure'] = df['book_imbalance'].rolling(window=5).mean()
                    
                    # Depth depletion
                    df['bid_depth_depletion'] = orderbook_data['bid_size_0'] / bid_volume
                    df['ask_depth_depletion'] = orderbook_data['ask_size_0'] / ask_volume
                    
                    # Top of book volatility
                    df['top_of_book_volatility'] = orderbook_data['bid_price_0'].diff().rolling(window=10).std()
                    
                    # Large orders at extremes
                    if len(orderbook_data) > 10:
                        df['large_bid_presence'] = orderbook_data['bid_size_0'] > orderbook_data['bid_size_0'].rolling(window=20).mean() * 3
                        df['large_ask_presence'] = orderbook_data['ask_size_0'] > orderbook_data['ask_size_0'].rolling(window=20).mean() * 3
                except Exception as e:
                    logger.warning(f"Error processing orderbook data: {str(e)}")
        
        # Trade-by-trade features (if available)
        if trade_data is not None and isinstance(trade_data, pd.DataFrame) and len(trade_data) > 0:
            try:
                # Extract trade sizes
                trade_sizes = trade_data['size'].values
                trade_prices = trade_data['price'].values
                trade_sides = trade_data['side'].values if 'side' in trade_data.columns else None
                
                # Trade size statistics
                df['avg_trade_size'] = np.mean(trade_sizes)
                df['max_trade_size'] = np.max(trade_sizes)
                df['trade_size_volatility'] = np.std(trade_sizes)
                
                # Unusual large trades
                df['large_trade_presence'] = np.max(trade_sizes) > np.mean(trade_sizes) * 5
                
                # Trade clustering
                if len(trade_prices) > 1:
                    price_changes = np.diff(trade_prices)
                    df['price_change_clustering'] = np.std(price_changes)
                
                # Buy/sell imbalance if side information is available
                if trade_sides is not None:
                    buy_volume = sum(trade_sizes[trade_sides == 'buy'])
                    sell_volume = sum(trade_sizes[trade_sides == 'sell'])
                    df['trade_imbalance'] = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
            except Exception as e:
                logger.warning(f"Error processing trade data: {str(e)}")
        
        # Tick data features (if available)
        if tick_data is not None and isinstance(tick_data, pd.DataFrame) and len(tick_data) > 0:
            try:
                # Tick frequency
                df['tick_frequency'] = len(tick_data)
                
                # Quote frequency
                if 'type' in tick_data.columns:
                    quote_count = sum(tick_data['type'] == 'quote')
                    trade_count = sum(tick_data['type'] == 'trade')
                    df['quote_to_trade_ratio'] = quote_count / trade_count if trade_count > 0 else 0
                
                # Quote volatility
                if 'bid' in tick_data.columns and 'ask' in tick_data.columns:
                    df['tick_bid_volatility'] = tick_data['bid'].std()
                    df['tick_ask_volatility'] = tick_data['ask'].std()
                    df['tick_midprice_volatility'] = ((tick_data['ask'] + tick_data['bid']) / 2).std()
            except Exception as e:
                logger.warning(f"Error processing tick data: {str(e)}")
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            
            # Proximity to market open/close
            market_open_hour = 9
            market_close_hour = 16
            df['mins_since_open'] = (df['hour'] - market_open_hour) * 60 + df['minute']
            df['mins_to_close'] = (market_close_hour - df['hour']) * 60 - df['minute']
            
            # Session indicators
            df['is_market_open'] = (df['hour'] >= market_open_hour) & (df['hour'] < market_close_hour)
            df['is_near_close'] = df['mins_to_close'] <= 15
            df['is_near_open'] = df['mins_since_open'] <= 15
        
        # Lag features for temporal patterns
        for feature in ['returns', 'volume', 'high_low_range']:
            if feature in df.columns:
                for lag in range(1, min(5, self.look_back)):
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Drop rows with missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Select numerical features only
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'hour', 'minute', 'day_of_week']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # Update feature names in metadata
        self.metadata['feature_names'] = feature_cols
        
        logger.info(f"Created {len(feature_cols)} features for anomaly detection")
        
        return df[feature_cols]
    
    def train(self, data, orderbook_data=None, trade_data=None, tick_data=None):
        """
        Train the anomaly detection model.
        
        Args:
            data: Market OHLCV data
            orderbook_data: Order book data (optional)
            trade_data: Individual trade data (optional)
            tick_data: Tick-by-tick data (optional)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Create features
        features_df = self.create_features(data, orderbook_data, trade_data, tick_data)
        
        if len(features_df) < 10:
            logger.warning("Insufficient data for training anomaly detection model")
            return False
        
        # Scale features
        X = self.scaler.fit_transform(features_df)
        
        # Create and train anomaly detection model
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42
            )
        elif self.model_type == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
        elif self.model_type == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif self.model_type == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return False
        
        # Fit the model
        try:
            self.model.fit(X)
            
            # For models that don't predict directly
            if self.model_type == 'dbscan':
                # In DBSCAN, -1 represents outliers
                y_pred = self.model.labels_
                # Convert to binary where -1 (outliers) become 1, and all others become -1
                # This follows the convention of other anomaly detection models
                y_pred = np.where(y_pred == -1, 1, -1)
            else:
                # For IsolationForest, OneClassSVM, and LOF
                # These predict -1 for anomalies and 1 for normal observations
                # We don't need to invert since we're using default convention
                y_pred = self.model.predict(X)
            
            # Calculate anomaly scores
            anomaly_scores = self._calculate_anomaly_scores(X)
            
            # Set anomaly threshold
            self.anomaly_threshold = np.percentile(anomaly_scores, (1 - self.contamination) * 100)
            
            # Update metadata
            self.metadata['training_date'] = datetime.now().isoformat()
            self.metadata['num_samples'] = len(X)
            self.metadata['anomaly_threshold'] = self.anomaly_threshold
            self.metadata['contamination'] = self.contamination
            
            # Count anomalies in training data
            anomaly_mask = anomaly_scores > self.anomaly_threshold
            self.metadata['anomaly_counts']['training'] = int(np.sum(anomaly_mask))
            
            logger.info(f"Trained {self.model_type} model with {len(X)} samples")
            logger.info(f"Detected {np.sum(anomaly_mask)} anomalies in training data")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {str(e)}")
            return False
    
    def _calculate_anomaly_scores(self, X):
        """
        Calculate anomaly scores for given data.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            ndarray: Anomaly scores
        """
        if self.model_type == 'isolation_forest':
            # Lower scores (more negative) = more anomalous
            # Convert to positive anomaly score where higher = more anomalous
            return -self.model.decision_function(X)
        
        elif self.model_type == 'lof':
            # Higher scores = more anomalous
            if hasattr(self.model, 'negative_outlier_factor_'):
                # If already fit, use negative_outlier_factor_
                return -self.model.negative_outlier_factor_
            else:
                # If in novelty mode, use decision function
                return -self.model.decision_function(X)
        
        elif self.model_type == 'one_class_svm':
            # Lower scores (more negative) = more anomalous
            # Convert to positive anomaly score where higher = more anomalous
            return -self.model.decision_function(X)
        
        elif self.model_type == 'dbscan':
            # For DBSCAN, use distance to nearest core point as anomaly score
            if not hasattr(self.model, 'components_'):
                # If no core samples, return array of ones (all anomalous)
                return np.ones(X.shape[0])
            
            # Calculate distance to nearest core point
            distances = np.min(
                np.sqrt(np.sum((X[:, np.newaxis, :] - self.model.components_[np.newaxis, :, :]) ** 2, axis=2)),
                axis=1
            )
            return distances
        
        else:
            # Default case - return zeros
            logger.warning(f"Unknown model type for anomaly scoring: {self.model_type}")
            return np.zeros(X.shape[0])
    
    def detect_anomalies(self, data, orderbook_data=None, trade_data=None, tick_data=None):
        """
        Detect anomalies in market data.
        
        Args:
            data: Market OHLCV data
            orderbook_data: Order book data (optional)
            trade_data: Individual trade data (optional)
            tick_data: Tick-by-tick data (optional)
            
        Returns:
            dict: Anomaly detection results with scores and details
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot detect anomalies.")
            return {
                'anomalies_detected': False,
                'error': 'Model not trained'
            }
        
        try:
            # Create features
            features_df = self.create_features(data, orderbook_data, trade_data, tick_data)
            
            if len(features_df) == 0:
                return {
                    'anomalies_detected': False,
                    'error': 'No valid features created'
                }
            
            # Scale features
            X = self.scaler.transform(features_df)
            
            # Get anomaly scores
            anomaly_scores = self._calculate_anomaly_scores(X)
            
            # Apply sensitivity factor to threshold
            effective_threshold = self.anomaly_threshold * self.sensitivity
            
            # Detect anomalies
            anomalies = anomaly_scores > effective_threshold
            
            # Get indices of anomalies
            anomaly_indices = np.where(anomalies)[0]
            
            # If the data has a DatetimeIndex, use it for timestamps
            timestamps = None
            if isinstance(data.index, pd.DatetimeIndex):
                timestamps = data.index.tolist()
                
            # Create result
            result = {
                'anomalies_detected': len(anomaly_indices) > 0,
                'num_anomalies': len(anomaly_indices),
                'total_samples': len(X),
                'anomaly_percentage': len(anomaly_indices) / len(X) * 100,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist() if len(anomaly_scores) < 1000 else "Scores too large to include",
                'threshold': effective_threshold,
                'model_type': self.model_type,
                'detection_time': datetime.now().isoformat()
            }
            
            # Add timestamps if available
            if timestamps and len(anomaly_indices) > 0:
                try:
                    # Convert timestamps to string format for JSON serialization
                    result['anomaly_timestamps'] = [timestamps[i].isoformat() for i in anomaly_indices]
                except Exception as e:
                    logger.warning(f"Could not add timestamps to anomaly results: {str(e)}")
            
            # Analyze feature contributions for anomalies
            if len(anomaly_indices) > 0:
                # Get feature values for anomalies
                anomaly_features = features_df.iloc[anomaly_indices]
                
                # Calculate Z-scores for each feature in anomalies
                feature_zscores = {}
                for col in features_df.columns:
                    mean = features_df[col].mean()
                    std = features_df[col].std()
                    if std > 0:
                        feature_zscores[col] = [(val - mean) / std for val in anomaly_features[col]]
                    else:
                        feature_zscores[col] = [0] * len(anomaly_features)
                
                # Identify top contributing features for each anomaly
                anomaly_explanations = []
                for i, idx in enumerate(anomaly_indices):
                    # Get top 3 features with highest absolute Z-scores
                    feature_scores = {col: abs(feature_zscores[col][i]) for col in features_df.columns}
                    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    anomaly_explanations.append({
                        'index': int(idx),
                        'score': float(anomaly_scores[idx]),
                        'contributing_features': [
                            {
                                'feature': feature,
                                'z_score': round(feature_zscores[feature][i], 2),
                                'value': float(anomaly_features.iloc[i][feature])
                            } for feature, _ in top_features
                        ]
                    })
                
                result['anomaly_explanations'] = anomaly_explanations
            
            # Add anomalies to history
            self._add_to_anomaly_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                'anomalies_detected': False,
                'error': str(e)
            }
    
    def _add_to_anomaly_history(self, anomaly_result):
        """
        Add anomaly detection result to history.
        
        Args:
            anomaly_result: Anomaly detection result
        """
        # Create simplified result for history
        history_entry = {
            'detection_time': anomaly_result['detection_time'],
            'num_anomalies': anomaly_result['num_anomalies'],
            'total_samples': anomaly_result['total_samples'],
            'anomaly_percentage': anomaly_result['anomaly_percentage']
        }
        
        # Add timestamps if available
        if 'anomaly_timestamps' in anomaly_result:
            history_entry['timestamps'] = anomaly_result['anomaly_timestamps']
        
        # Add to history
        self.anomaly_history.append(history_entry)
        
        # Keep history limited to 100 entries
        if len(self.anomaly_history) > 100:
            self.anomaly_history = self.anomaly_history[-100:]
    
    def get_feature_importance(self):
        """
        Get feature importance scores if available.
        
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            feature_names = self.metadata['feature_names']
            
            # For isolation forest, feature importance is available
            if self.model_type == 'isolation_forest' and hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                # Sort features by importance
                sorted_idx = np.argsort(importances)[::-1]
                feature_importance = {
                    feature_names[i]: float(importances[i]) 
                    for i in sorted_idx if i < len(feature_names)
                }
                
                return {
                    'feature_importance': feature_importance,
                    'model_type': self.model_type
                }
            else:
                # For other models, we can't easily get feature importance
                return {
                    'error': f'Feature importance not available for {self.model_type}'
                }
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {'error': str(e)}
    
    def adjust_sensitivity(self, new_sensitivity):
        """
        Adjust sensitivity for anomaly detection.
        
        Args:
            new_sensitivity: New sensitivity value (higher = more anomalies detected)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure sensitivity is positive
            if new_sensitivity <= 0:
                logger.warning("Sensitivity must be positive, setting to 0.1")
                new_sensitivity = 0.1
            
            # Update sensitivity
            self.sensitivity = new_sensitivity
            self.metadata['sensitivity'] = new_sensitivity
            
            logger.info(f"Adjusted anomaly detection sensitivity to {new_sensitivity}")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting sensitivity: {str(e)}")
            return False
    
    def save_model(self):
        """
        Save trained model, scaler, and metadata to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.model_path:
            logger.warning("No model path specified, model not saved")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            model_path = f"{self.model_path}_model.joblib"
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = f"{self.model_path}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata_path = f"{self.model_path}_metadata.json"
            metadata_copy = self.metadata.copy()
            
            # Ensure all values are JSON serializable
            for key, value in metadata_copy.items():
                if isinstance(value, np.ndarray):
                    metadata_copy[key] = value.tolist()
            
            # Add anomaly history
            metadata_copy['anomaly_history'] = self.anomaly_history
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_copy, f, indent=2)
            
            logger.info(f"Saved anomaly detection model to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """
        Load trained model, scaler, and metadata from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.model_path:
            logger.warning("No model path specified, model not loaded")
            return False
        
        try:
            # Load model
            model_path = f"{self.model_path}_model.joblib"
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
                
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = f"{self.model_path}_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = f"{self.model_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update metadata
                self.metadata = metadata
                
                # Update anomaly threshold
                if 'anomaly_threshold' in metadata:
                    self.anomaly_threshold = metadata['anomaly_threshold']
                
                # Update sensitivity
                if 'sensitivity' in metadata:
                    self.sensitivity = metadata['sensitivity']
                
                # Load anomaly history
                if 'anomaly_history' in metadata:
                    self.anomaly_history = metadata['anomaly_history']
            
            logger.info(f"Loaded anomaly detection model from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_anomaly_history_summary(self):
        """
        Get a summary of anomaly detection history.
        
        Returns:
            dict: Summary statistics of anomaly history
        """
        if not self.anomaly_history:
            return {
                'count': 0,
                'message': 'No anomaly detection history available'
            }
        
        try:
            # Count total anomalies
            total_anomalies = sum(entry['num_anomalies'] for entry in self.anomaly_history)
            
            # Calculate average anomaly percentage
            avg_percentage = sum(entry['anomaly_percentage'] for entry in self.anomaly_history) / len(self.anomaly_history)
            
            # Group by day if timestamps are available
            daily_counts = {}
            for entry in self.anomaly_history:
                if 'detection_time' in entry:
                    try:
                        day = entry['detection_time'].split('T')[0]
                        daily_counts[day] = daily_counts.get(day, 0) + entry['num_anomalies']
                    except:
                        pass
            
            # Get highest anomaly day
            highest_day = max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None
            
            return {
                'count': len(self.anomaly_history),
                'total_anomalies': total_anomalies,
                'average_percentage': avg_percentage,
                'daily_counts': daily_counts,
                'highest_anomaly_day': highest_day
            }
            
        except Exception as e:
            logger.error(f"Error generating anomaly history summary: {str(e)}")
            return {
                'error': str(e),
                'count': len(self.anomaly_history)
            }

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for anomaly detection from raw market data.
        
        Args:
            data: DataFrame with OHLCV and order book data
            
        Returns:
            DataFrame with engineered features
        """
        if data.empty:
            logger.warning("Empty data provided for feature engineering")
            return pd.DataFrame()
            
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Basic price and volume features
        features = pd.DataFrame()
        
        # Return if we don't have enough data
        if len(df) < 2:
            logger.warning("Not enough data points for feature engineering")
            return features
            
        # Price changes and ratios
        features['return'] = df['close'].pct_change()
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility measures
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['range_ma_ratio'] = features['price_range'] / features['price_range'].rolling(5).mean()
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        
        # Add bid-ask spread features if available
        if 'ask' in df.columns and 'bid' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['relative_spread'] = features['spread'] / df['close']
            features['spread_change'] = features['spread'].pct_change()
        
        # Add order book imbalance if available
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            features['book_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
            features['book_pressure'] = features['book_imbalance'].diff()
        
        # Rolling measures
        for window in [5, 10, 20]:
            features[f'return_z_{window}'] = (features['return'] - 
                                              features['return'].rolling(window).mean()) / features['return'].rolling(window).std()
            features[f'volume_z_{window}'] = (features['volume_change'] - 
                                              features['volume_change'].rolling(window).mean()) / features['volume_change'].rolling(window).std()
        
        # Handle NaNs (from rolling windows)
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        return features
    
    def train(self, historical_data: pd.DataFrame, save_model: bool = True) -> Dict[str, Any]:
        """
        Train anomaly detection models on historical data.
        
        Args:
            historical_data: DataFrame with historical market data
            save_model: Whether to save the trained model to disk
            
        Returns:
            Dictionary with training metrics
        """
        # Engineer features
        features = self._engineer_features(historical_data)
        
        if features.empty:
            logger.error("No valid features for training")
            return {"error": "No valid features for training"}
        
        # Train isolation forest
        logger.info(f"Training isolation forest for {self.symbol} with {len(features)} samples")
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit and transform the data
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # Train isolation forest
        self.isolation_forest.fit(scaled_features)
        
        # Train autoencoder if enabled
        if self.use_autoencoder:
            logger.info(f"Training LSTM autoencoder for {self.symbol}")
            self.autoencoder = self._create_and_train_autoencoder(features)
        
        # Save models if requested
        if save_model:
            self._save_models()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "samples": len(features),
            "isolation_forest_trained": self.isolation_forest is not None,
            "autoencoder_trained": self.autoencoder is not None if self.use_autoencoder else False
        }
        
        return results
    
    def _create_and_train_autoencoder(self, features: pd.DataFrame) -> Model:
        """
        Create and train an LSTM autoencoder for sequential anomaly detection.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Trained autoencoder model
        """
        # Prepare sequences for LSTM
        sequences = self._create_sequences(features)
        
        if len(sequences) == 0:
            logger.warning("Not enough data to create sequences for autoencoder")
            return None
        
        # Define model architecture
        n_features = sequences.shape[2]
        
        # Create simple LSTM autoencoder
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.lookback_window, n_features), return_sequences=False),
            RepeatVector(self.lookback_window),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model with early stopping
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Split data into train/validation
        train_size = int(len(sequences) * 0.8)
        x_train, x_val = sequences[:train_size], sequences[train_size:]
        
        # Train
        history = model.fit(
            x_train, x_train,
            epochs=50,
            batch_size=32,
            validation_data=(x_val, x_val),
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info(f"Autoencoder trained for {len(history.history['loss'])} epochs")
        
        return model
    
    def _create_sequences(self, features: pd.DataFrame) -> np.ndarray:
        """
        Create sequences for LSTM autoencoder training.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Numpy array of sequences with shape (n_sequences, lookback_window, n_features)
        """
        data = features.values
        
        if len(data) <= self.lookback_window:
            return np.array([])
            
        sequences = []
        for i in range(len(data) - self.lookback_window + 1):
            sequences.append(data[i:i+self.lookback_window])
            
        return np.array(sequences)
    
    def _save_models(self):
        """Save trained models to disk."""
        # Create model directory with symbol name
        model_path = os.path.join(self.model_dir, self.symbol)
        os.makedirs(model_path, exist_ok=True)
        
        # Save isolation forest
        if self.isolation_forest is not None:
            with open(os.path.join(model_path, 'isolation_forest.pkl'), 'wb') as f:
                pickle.dump(self.isolation_forest, f)
            
            # Save scaler
            with open(os.path.join(model_path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save autoencoder
        if self.autoencoder is not None:
            self.autoencoder.save(os.path.join(model_path, 'autoencoder'))
        
        # Save metadata
        metadata = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "lookback_window": self.lookback_window,
            "alert_threshold": self.alert_threshold,
            "contamination": self.contamination,
            "use_autoencoder": self.use_autoencoder
        }
        
        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Models saved to {model_path}")
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            Boolean indicating if models were successfully loaded
        """
        model_path = os.path.join(self.model_dir, self.symbol)
        
        if not os.path.exists(model_path):
            logger.warning(f"No saved models found for {self.symbol}")
            return False
        
        # Load isolation forest
        forest_path = os.path.join(model_path, 'isolation_forest.pkl')
        if os.path.exists(forest_path):
            with open(forest_path, 'rb') as f:
                self.isolation_forest = pickle.load(f)
                
            # Load scaler
            scaler_path = os.path.join(model_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
        
        # Load autoencoder
        autoencoder_path = os.path.join(model_path, 'autoencoder')
        if os.path.exists(autoencoder_path):
            self.autoencoder = load_model(autoencoder_path)
            self.use_autoencoder = True
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.lookback_window = metadata.get('lookback_window', self.lookback_window)
                self.alert_threshold = metadata.get('alert_threshold', self.alert_threshold)
                self.contamination = metadata.get('contamination', self.contamination)
        
        logger.info(f"Models loaded for {self.symbol}")
        return True
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in market data.
        
        Args:
            data: DataFrame with recent market data
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Check if models are loaded
        if self.isolation_forest is None:
            if not self.load_models():
                logger.error("No trained models available for anomaly detection")
                return {"error": "No trained models available"}
        
        # Engineer features
        features = self._engineer_features(data)
        
        if features.empty:
            return {"error": "No valid features for anomaly detection"}
        
        # Calculate anomaly scores using isolation forest
        scaled_features = self.scaler.transform(features)
        isolation_scores = self.isolation_forest.decision_function(scaled_features)
        # Convert to anomaly scores (0-1 where 1 is most anomalous)
        isolation_scores = 1 - (isolation_scores + 0.5)  # Scale from [-0.5, 0.5] to [0, 1]
        
        # Calculate autoencoder scores if available
        autoencoder_scores = None
        if self.autoencoder is not None and len(features) >= self.lookback_window:
            autoencoder_scores = self._get_autoencoder_scores(features)
        
        # Combine scores
        combined_scores = isolation_scores
        if autoencoder_scores is not None:
            # Only use autoencoder scores for the latest data points where they're available
            n_overlap = min(len(isolation_scores), len(autoencoder_scores))
            combined_scores[-n_overlap:] = np.maximum(
                isolation_scores[-n_overlap:], 
                autoencoder_scores[-n_overlap:]
            )
        
        # Find anomalies above threshold
        anomalies = combined_scores > self.alert_threshold
        anomaly_indices = np.where(anomalies)[0]
        
        # Create results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "num_anomalies": int(np.sum(anomalies)),
            "anomaly_indices": anomaly_indices.tolist(),
            "max_anomaly_score": float(np.max(combined_scores)) if len(combined_scores) > 0 else 0,
            "latest_score": float(combined_scores[-1]) if len(combined_scores) > 0 else 0,
        }
        
        # Store latest scores for tracking
        self.latest_scores = {
            "isolation_forest": float(isolation_scores[-1]) if len(isolation_scores) > 0 else 0,
            "autoencoder": float(autoencoder_scores[-1]) if autoencoder_scores is not None and len(autoencoder_scores) > 0 else 0,
            "combined": float(combined_scores[-1]) if len(combined_scores) > 0 else 0
        }
        
        # Add to anomaly history
        if results["num_anomalies"] > 0:
            self.anomaly_history.append({
                "timestamp": results["timestamp"],
                "num_anomalies": results["num_anomalies"],
                "max_score": results["max_anomaly_score"]
            })
        
        return results
    
    def _get_autoencoder_scores(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores using the autoencoder.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Numpy array of anomaly scores
        """
        # Create sequences
        sequences = self._create_sequences(features)
        
        if len(sequences) == 0:
            return np.array([])
        
        # Predict and calculate reconstruction error
        reconstructions = self.autoencoder.predict(sequences, verbose=0)
        
        # MSE for each sequence and each feature
        mse = np.mean(np.square(sequences - reconstructions), axis=2)
        
        # Average MSE across features
        reconstruction_errors = np.mean(mse, axis=1)
        
        # Normalize to [0, 1]
        min_error = np.min(reconstruction_errors)
        max_error = np.max(reconstruction_errors)
        
        if max_error > min_error:
            normalized_errors = (reconstruction_errors - min_error) / (max_error - min_error)
        else:
            normalized_errors = np.zeros_like(reconstruction_errors)
        
        return normalized_errors
    
    def get_anomaly_features(self, data: pd.DataFrame, anomaly_indices: List[int]) -> pd.DataFrame:
        """
        Get the most important features contributing to anomalies.
        
        Args:
            data: Original data DataFrame
            anomaly_indices: Indices of detected anomalies
            
        Returns:
            DataFrame with feature importance for anomalies
        """
        if not anomaly_indices:
            return pd.DataFrame()
        
        # Engineer features
        features = self._engineer_features(data)
        
        if features.empty:
            return pd.DataFrame()
        
        # Get anomaly data
        anomaly_features = features.iloc[anomaly_indices]
        
        # Calculate z-scores relative to historical data
        feature_means = features.mean()
        feature_stds = features.std()
        
        z_scores = (anomaly_features - feature_means) / feature_stds
        
        # Return features sorted by absolute z-score
        return z_scores.abs().mean().sort_values(ascending=False)
    
    def get_alert_message(self, data: pd.DataFrame, results: Dict[str, Any]) -> str:
        """
        Generate alert message for detected anomalies.
        
        Args:
            data: Original data DataFrame
            results: Results from detect_anomalies()
            
        Returns:
            Alert message string
        """
        if "error" in results:
            return f"Error: {results['error']}"
        
        if results["num_anomalies"] == 0:
            return f"No anomalies detected for {self.symbol}"
        
        # Get last anomaly index
        last_anomaly = results["anomaly_indices"][-1]
        
        # Get important features
        feature_importance = self.get_anomaly_features(data, [last_anomaly])
        
        # Create message
        message = [
            f"ANOMALY ALERT: {self.symbol}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Anomaly Score: {results['latest_score']:.4f}",
            f"Total Anomalies: {results['num_anomalies']}"
        ]
        
        # Add top features
        if not feature_importance.empty:
            message.append("\nTop Contributing Factors:")
            for feature, score in feature_importance.head(3).items():
                message.append(f"- {feature}: {score:.2f}")
        
        # Add price action if available
        if "close" in data.columns and last_anomaly < len(data):
            current_price = data.iloc[last_anomaly]["close"]
            if last_anomaly > 0:
                prev_price = data.iloc[last_anomaly-1]["close"]
                pct_change = (current_price - prev_price) / prev_price * 100
                message.append(f"\nPrice: {current_price:.2f} ({pct_change:+.2f}%)")
            else:
                message.append(f"\nPrice: {current_price:.2f}")
        
        return "\n".join(message)
    
    def get_historical_anomalies(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical anomalies detected within the specified number of days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of historical anomaly events
        """
        if not self.anomaly_history:
            return []
        
        # Calculate cutoff time
        cutoff = datetime.now() - timedelta(days=days)
        
        # Filter anomalies
        recent_anomalies = []
        for anomaly in self.anomaly_history:
            anomaly_time = datetime.fromisoformat(anomaly["timestamp"])
            if anomaly_time >= cutoff:
                recent_anomalies.append(anomaly)
        
        return recent_anomalies

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for this anomaly detector.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "symbol": self.symbol,
            "total_anomalies_tracked": len(self.anomaly_history),
            "recent_anomalies": len(self.get_historical_anomalies(7)),
            "latest_scores": self.latest_scores,
            "alert_threshold": self.alert_threshold,
        }
    
    def plot_anomaly_history(self, days: int = 30) -> plt.Figure:
        """
        Plot anomaly history for visualization.
        
        Args:
            days: Number of days to plot
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter recent anomalies
        cutoff = datetime.now() - timedelta(days=days)
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_history
            if isinstance(anomaly.get("timestamp"), datetime) and anomaly["timestamp"] >= cutoff
        ]
        
        if not recent_anomalies:
            ax.text(0.5, 0.5, f"No anomalies in the last {days} days", 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"Anomaly History for {self.symbol}")
            return fig
        
        # Extract data
        timestamps = [anomaly["timestamp"] for anomaly in recent_anomalies]
        scores = [anomaly["score"] for anomaly in recent_anomalies]
        
        # Plot scores
        ax.plot(timestamps, scores, 'o-', color='blue', alpha=0.7)
        ax.axhline(y=self.alert_threshold, color='red', linestyle='--', label=f'Alert Threshold ({self.alert_threshold})')
        
        # Format the chart
        ax.set_title(f"Anomaly History for {self.symbol} (Last {days} Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Anomaly Score")
        ax.grid(True)
        ax.legend()
        
        fig.tight_layout()
        return fig 