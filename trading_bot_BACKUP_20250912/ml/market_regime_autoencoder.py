#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Autoencoder

This module implements a lightweight autoencoder for market regime detection
and anomaly detection in financial time series.
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
from typing import Dict, List, Any, Optional, Union, Tuple

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from trading_bot.ml.enhanced_features import EnhancedFeatureEngineering

logger = logging.getLogger(__name__)

class MarketRegimeAutoencoder:
    """Autoencoder for market regime detection and anomaly detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market regime autoencoder
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.feature_engine = EnhancedFeatureEngineering(config)
        
        # Set default configuration if not provided
        self._set_default_config()
        
        # Initialize model objects
        self.model = None
        self.encoder = None
        self.scaler = None
        self.loss_threshold = None  # For anomaly detection
        self.regime_centroids = None  # For regime clustering
        
        logger.info("Market Regime Autoencoder initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Autoencoder architecture
        self.config.setdefault("encoding_dim", 3)  # Dimension of the encoded space
        self.config.setdefault("hidden_layers", [32, 16, 8])  # Hidden layer sizes
        self.config.setdefault("dropout_rate", 0.2)
        self.config.setdefault("activation", "elu")
        self.config.setdefault("learning_rate", 0.001)
        
        # Training parameters
        self.config.setdefault("batch_size", 32)
        self.config.setdefault("epochs", 100)
        self.config.setdefault("validation_split", 0.2)
        self.config.setdefault("patience", 10)  # Early stopping patience
        
        # Anomaly detection
        self.config.setdefault("anomaly_threshold", 95)  # Percentile for reconstruction error
        
        # Regime detection
        self.config.setdefault("n_regimes", 4)
        self.config.setdefault("regime_window", 20)  # Rolling window for regime stability
        
        # Model persistence
        self.config.setdefault("model_dir", "models/autoencoder")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for autoencoder training or inference
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            Feature DataFrame suitable for autoencoder
        """
        logger.info("Preparing data for autoencoder")
        
        # Generate features
        df = self.feature_engine.generate_features(data.copy())
        
        # Drop non-feature columns and NaN values
        cols_to_drop = ["open", "high", "low", "close", "volume"]
        feature_cols = [col for col in df.columns if col not in cols_to_drop]
        
        X = df[feature_cols].copy()
        X = X.dropna()
        
        logger.info(f"Prepared data shape: {X.shape}")
        return X
    
    def build_model(self):
        """Build and compile the autoencoder model"""
        # Get model parameters
        encoding_dim = self.config["encoding_dim"]
        hidden_layers = self.config["hidden_layers"]
        dropout_rate = self.config["dropout_rate"]
        activation = self.config["activation"]
        learning_rate = self.config["learning_rate"]
        
        # Build encoder
        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        x = BatchNormalization()(inputs)
        
        # Encoder path
        for i, units in enumerate(hidden_layers):
            x = Dense(units, activation=activation, name=f'encoder_dense_{i}')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Encoded representation
        encoded = Dense(encoding_dim, activation='linear', name='encoded')(x)
        
        # Decoder path
        x = encoded
        for i, units in enumerate(reversed(hidden_layers)):
            x = Dense(units, activation=activation, name=f'decoder_dense_{i}')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.input_dim, activation='linear', name='decoder_output')(x)
        
        # Create full model
        autoencoder = Model(inputs=inputs, outputs=outputs, name='autoencoder')
        
        # Create encoder model
        encoder_model = Model(inputs=inputs, outputs=encoded, name='encoder')
        
        # Compile model
        autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Store models
        self.model = autoencoder
        self.encoder = encoder_model
        
        logger.info(f"Built autoencoder model with encoding dimension {encoding_dim}")
        logger.info(f"Input dimension: {self.input_dim}")
        logger.info(f"Model parameters: {autoencoder.count_params():,}")
        
        return autoencoder
    
    def train(self, data: pd.DataFrame, save_model: bool = True) -> Dict[str, Any]:
        """
        Train the autoencoder model
        
        Args:
            data: OHLCV data with datetime index
            save_model: Whether to save trained models to disk
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training autoencoder model")
        
        # Prepare data
        X = self.prepare_data(data)
        
        # Create and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Set input dimension
        self.input_dim = X_scaled.shape[1]
        
        # Build model
        self.build_model()
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["patience"],
                restore_best_weights=True
            )
        ]
        
        # Add model checkpoint if saving
        if save_model:
            os.makedirs(self.config["model_dir"], exist_ok=True)
            model_path = os.path.join(self.config["model_dir"], "autoencoder_model.h5")
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        history = self.model.fit(
            X_scaled, X_scaled,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            validation_split=self.config["validation_split"],
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Set anomaly threshold at specified percentile
        percentile = self.config["anomaly_threshold"]
        self.loss_threshold = np.percentile(reconstruction_errors, percentile)
        
        logger.info(f"Trained autoencoder model with final loss: {history.history['loss'][-1]:.6f}")
        logger.info(f"Anomaly threshold set at {percentile}th percentile: {self.loss_threshold:.6f}")
        
        # Generate regime clusters from encoded data
        self._generate_regime_clusters(X_scaled)
        
        # Save model and metadata if requested
        if save_model:
            self._save_model()
        
        # Return training metrics
        metrics = {
            "final_loss": float(history.history['loss'][-1]),
            "val_loss": float(history.history['val_loss'][-1]),
            "epochs": len(history.history['loss']),
            "anomaly_threshold": float(self.loss_threshold),
            "n_regimes": self.config["n_regimes"]
        }
        
        return metrics
    
    def _generate_regime_clusters(self, X_scaled: np.ndarray):
        """
        Generate regime clusters from encoded data
        
        Args:
            X_scaled: Scaled input data
        """
        from sklearn.cluster import KMeans
        
        # Generate encoded representations
        encoded_data = self.encoder.predict(X_scaled)
        
        # Cluster encoded data
        kmeans = KMeans(
            n_clusters=self.config["n_regimes"],
            random_state=42,
            n_init=10
        )
        
        clusters = kmeans.fit(encoded_data)
        centroids = clusters.cluster_centers_
        
        # Store centroids
        self.regime_centroids = centroids
        
        logger.info(f"Generated {self.config['n_regimes']} market regime clusters")
    
    def _save_model(self):
        """Save model, scaler, and metadata to disk"""
        # Create directory if it doesn't exist
        os.makedirs(self.config["model_dir"], exist_ok=True)
        
        # Save full model and encoder
        self.model.save(os.path.join(self.config["model_dir"], "autoencoder_model.h5"))
        self.encoder.save(os.path.join(self.config["model_dir"], "encoder_model.h5"))
        
        # Save scaler
        scaler_path = os.path.join(self.config["model_dir"], "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save loss threshold
        threshold_path = os.path.join(self.config["model_dir"], "loss_threshold.npy")
        np.save(threshold_path, self.loss_threshold)
        
        # Save regime centroids
        centroids_path = os.path.join(self.config["model_dir"], "regime_centroids.npy")
        np.save(centroids_path, self.regime_centroids)
        
        # Save config
        config_path = os.path.join(self.config["model_dir"], "config.joblib")
        joblib.dump(self.config, config_path)
        
        logger.info(f"Saved autoencoder model and metadata to {self.config['model_dir']}")
    
    def load_model(self) -> bool:
        """
        Load model, scaler, and metadata from disk
        
        Returns:
            Success flag
        """
        try:
            # Load models
            model_path = os.path.join(self.config["model_dir"], "autoencoder_model.h5")
            encoder_path = os.path.join(self.config["model_dir"], "encoder_model.h5")
            
            self.model = load_model(model_path)
            self.encoder = load_model(encoder_path)
            
            # Get input dimension from model
            self.input_dim = self.model.input_shape[1]
            
            # Load scaler
            scaler_path = os.path.join(self.config["model_dir"], "scaler.joblib")
            self.scaler = joblib.load(scaler_path)
            
            # Load loss threshold
            threshold_path = os.path.join(self.config["model_dir"], "loss_threshold.npy")
            self.loss_threshold = np.load(threshold_path)
            
            # Load regime centroids
            centroids_path = os.path.join(self.config["model_dir"], "regime_centroids.npy")
            self.regime_centroids = np.load(centroids_path)
            
            logger.info(f"Loaded autoencoder model and metadata from {self.config['model_dir']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading autoencoder model: {e}")
            return False
    
    def encode(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate encoded representations of data
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            Encoded representations
        """
        # Check if model is loaded
        if self.model is None:
            if not self.load_model():
                logger.error("No model available for encoding")
                return np.array([])
        
        # Prepare data
        X = self.prepare_data(data)
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Generate encoded representations
        encoded = self.encoder.predict(X_scaled)
        
        return encoded
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in data using reconstruction error
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        # Check if model is loaded
        if self.model is None:
            if not self.load_model():
                logger.error("No model available for anomaly detection")
                return pd.DataFrame()
        
        # Prepare data
        X = self.prepare_data(data)
        original_index = X.index
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Generate reconstructions
        reconstructions = self.model.predict(X_scaled)
        
        # Calculate reconstruction errors (MSE)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Create anomaly flags
        anomaly_flags = reconstruction_errors > self.loss_threshold
        
        # Create result DataFrame
        result = pd.DataFrame({
            'date': original_index,
            'reconstruction_error': reconstruction_errors,
            'anomaly_threshold': self.loss_threshold,
            'is_anomaly': anomaly_flags
        })
        
        logger.info(f"Detected {anomaly_flags.sum()} anomalies in {len(result)} samples")
        return result
    
    def detect_market_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes in data using clustering of encoded representations
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            DataFrame with regime labels and probabilities
        """
        from scipy.spatial.distance import cdist
        
        # Check if model and centroids are loaded
        if self.model is None or self.regime_centroids is None:
            if not self.load_model():
                logger.error("No model or centroids available for regime detection")
                return pd.DataFrame()
        
        # Get encoded representations
        encoded = self.encode(data)
        original_index = data.index[-(len(encoded)):]
        
        if len(encoded) == 0:
            logger.error("Failed to generate encoded representations")
            return pd.DataFrame()
        
        # Calculate distance to each centroid
        distances = cdist(encoded, self.regime_centroids)
        
        # Convert distances to probabilities using softmax
        def softmax(x):
            # Negative distances (closer = higher probability)
            exp_x = np.exp(-x)
            return exp_x / exp_x.sum(axis=1, keepdims=True)
        
        regime_probs = softmax(distances)
        
        # Assign regime labels
        regime_labels = np.argmax(regime_probs, axis=1)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'date': original_index,
            'regime': regime_labels
        })
        
        # Add probability columns for each regime
        for i in range(self.config["n_regimes"]):
            result[f'regime_{i}_prob'] = regime_probs[:, i]
        
        # Add rolling regime for stability
        if len(result) >= self.config["regime_window"]:
            # Most common regime in the window
            rolling_regime = result['regime'].rolling(
                window=self.config["regime_window"], 
                min_periods=1
            ).apply(lambda x: pd.Series(x).mode()[0])
            
            result['rolling_regime'] = rolling_regime
        else:
            result['rolling_regime'] = result['regime']
        
        logger.info(f"Detected market regimes for {len(result)} samples")
        return result

# Utility function to create a market regime autoencoder with default configuration
def create_market_regime_autoencoder(config: Dict[str, Any] = None) -> MarketRegimeAutoencoder:
    """
    Create a market regime autoencoder with default or custom configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized MarketRegimeAutoencoder instance
    """
    return MarketRegimeAutoencoder(config)
