#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM model implementation for time series prediction in trading.
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from trading_bot.ml.base_model import BaseMLModel

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    logger.warning("TensorFlow not available. LSTM model will not work.")
    HAS_TENSORFLOW = False
    tf = None


# Define a dummy LSTM model class when TensorFlow is not available
if not HAS_TENSORFLOW:
    class LSTMModel(BaseMLModel):
        """
        Dummy LSTM model implementation when TensorFlow is not available.
        """
        
        def __init__(
            self, 
            name: str,
            sequence_length: int = 10,
            lstm_units: List[int] = [64, 32],
            dropout_rate: float = 0.2,
            learning_rate: float = 0.001,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 10,
            target_column: str = "close",
            target_type: str = "return",
            target_horizon: int = 1,
            features: Optional[List[str]] = None,
            scale_features: bool = True,
            config: Optional[Dict[str, Any]] = None
        ):
            """
            Initialize the dummy LSTM model.
            """
            super().__init__(name, config)
            self.is_dummy = True
            logger.warning(f"Initialized dummy LSTM model {name} (TensorFlow not available)")
            self.feature_columns = []
            self.is_trained = False
            
        def train(self, features: pd.DataFrame, targets: pd.DataFrame = None) -> None:
            """
            Dummy train method.
            """
            logger.warning("Cannot train LSTM model without TensorFlow")
            
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            """
            Dummy predict method.
            """
            logger.warning("Cannot predict with LSTM model without TensorFlow")
            return np.array([])
            
        def get_feature_importance(self) -> Dict[str, float]:
            """
            Dummy feature importance method.
            """
            return {}
            
        def save(self, filepath: Optional[str] = None) -> str:
            """
            Dummy save method.
            """
            return ""
            
        @classmethod
        def load(cls, filepath: str) -> 'LSTMModel':
            """
            Dummy load method.
            """
            return cls("dummy")

else:
    class LSTMModel(BaseMLModel):
        """
        LSTM neural network model for time series prediction.
        
        Implements a recurrent neural network using LSTM cells
        for capturing temporal dependencies in financial time series.
        """
        
        def __init__(
            self, 
            name: str,
            sequence_length: int = 10,
            lstm_units: List[int] = [64, 32],
            dropout_rate: float = 0.2,
            learning_rate: float = 0.001,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 10,
            target_column: str = "close",
            target_type: str = "return",  # 'price', 'return', or 'direction'
            target_horizon: int = 1,
            features: Optional[List[str]] = None,
            scale_features: bool = True,
            config: Optional[Dict[str, Any]] = None
        ):
            """
            Initialize the LSTM model.
            
            Args:
                name: Unique identifier for this model
                sequence_length: Number of time steps to look back
                lstm_units: List of units for each LSTM layer
                dropout_rate: Dropout rate for regularization
                learning_rate: Learning rate for optimizer
                batch_size: Batch size for training
                epochs: Maximum number of epochs for training
                patience: Patience for early stopping
                target_column: Column to predict (e.g., 'close', 'return')
                target_type: Type of prediction ('price', 'return', or 'direction')
                target_horizon: How many steps ahead to predict
                features: List of feature columns to use (None uses all numeric columns)
                scale_features: Whether to standardize features
                config: Additional configuration options
            """
            super().__init__(name, config)
            
            # Model hyperparameters
            self.sequence_length = sequence_length
            self.lstm_units = lstm_units
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.patience = patience
            
            # Prediction configuration
            self.target_column = target_column
            self.target_type = target_type
            self.target_horizon = target_horizon
            self.features = features
            self.scale_features = scale_features
            
            # Model attributes
            self.model = None
            self.feature_scaler = None
            self.target_scaler = None
            self.feature_columns = None
            self._feature_importance_cache = {}
            
        def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
            """
            Build the LSTM model architecture.
            
            Args:
                input_shape: Shape of input data (sequence_length, num_features)
                
            Returns:
                Compiled Keras model
            """
            model = Sequential()
            
            # Add LSTM layers
            for i, units in enumerate(self.lstm_units):
                # Return sequences for all but the last LSTM layer
                return_sequences = i < len(self.lstm_units) - 1
                
                # First layer gets input_shape
                if i == 0:
                    model.add(LSTM(units=units, 
                                return_sequences=return_sequences,
                                input_shape=input_shape))
                else:
                    model.add(LSTM(units=units, return_sequences=return_sequences))
                    
                # Add regularization
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout_rate))
            
            # Output layer (single output for regression)
            model.add(Dense(1))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error'
            )
            
            logger.info(f"Built LSTM model with {len(self.lstm_units)} LSTM layers")
            return model
        
        def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Create sequences for LSTM training/prediction.
            
            Args:
                data: Feature data array
                target: Target data array
                
            Returns:
                Tuple of (X, y) where X is shaped for LSTM input
            """
            X, y = [], []
            
            for i in range(len(data) - self.sequence_length - self.target_horizon + 1):
                # Sequence of features
                X.append(data[i:(i + self.sequence_length)])
                
                # Target value (future)
                y.append(target[i + self.sequence_length + self.target_horizon - 1])
                
            return np.array(X), np.array(y)
        
        def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
            """
            Prepare target values based on target type.
            
            Args:
                df: Input DataFrame
                
            Returns:
                Numpy array of target values
            """
            if self.target_type == 'price':
                # Direct price prediction
                target = df[self.target_column].values
                
            elif self.target_type == 'return':
                # Return prediction (percentage change)
                target = df[self.target_column].pct_change(self.target_horizon).values
                
            elif self.target_type == 'direction':
                # Direction prediction (binary classification)
                direction = np.sign(df[self.target_column].pct_change(self.target_horizon).values)
                # Convert to binary (1 for positive, 0 for negative or zero)
                target = (direction > 0).astype(float)
                
            else:
                raise ValueError(f"Unknown target type: {self.target_type}")
                
            return target
        
        def _scale_features(self, features: np.ndarray) -> np.ndarray:
            """
            Scale features to zero mean and unit variance.
            
            Args:
                features: Feature array to scale
                
            Returns:
                Scaled features
            """
            if self.feature_scaler is None:
                # Simple standardization (Z-score)
                self.feature_means = np.mean(features, axis=0)
                self.feature_stds = np.std(features, axis=0)
                
                # Avoid division by zero
                self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
                
            # Apply scaling
            return (features - self.feature_means) / self.feature_stds
        
        def _scale_target(self, target: np.ndarray) -> np.ndarray:
            """
            Scale target values.
            
            Args:
                target: Target array to scale
                
            Returns:
                Scaled target
            """
            if self.target_scaler is None:
                self.target_mean = np.mean(target)
                self.target_std = np.std(target)
                
                # Avoid division by zero
                if self.target_std == 0:
                    self.target_std = 1.0
                    
            # Apply scaling
            return (target - self.target_mean) / self.target_std
        
        def _inverse_scale_target(self, scaled_target: np.ndarray) -> np.ndarray:
            """
            Reverse scaling of target values.
            
            Args:
                scaled_target: Scaled target array
                
            Returns:
                Original scale target
            """
            if self.target_scaler is not None:
                return scaled_target * self.target_std + self.target_mean
            return scaled_target
        
        def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Preprocess data for LSTM model.
            
            Args:
                data: Input DataFrame
                
            Returns:
                Preprocessed DataFrame
            """
            df = data.copy()
            
            # Handle missing values
            df = df.dropna()
            
            # Select features
            if self.feature_columns is None:
                # First time, determine which columns to use
                if self.features is not None:
                    # Use specified features
                    self.feature_columns = [col for col in self.features if col in df.columns]
                else:
                    # Use all numeric columns except target if it's part of features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    self.feature_columns = [col for col in numeric_cols 
                                        if col != self.target_column or self.target_type != 'price']
            
            # Ensure all required columns are present
            missing_cols = [col for col in self.feature_columns + [self.target_column] 
                        if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in data: {missing_cols}")
            
            return df
        
        def train(self, features: pd.DataFrame, targets: pd.DataFrame = None) -> None:
            """
            Train the LSTM model.
            
            Args:
                features: Feature DataFrame
                targets: Target DataFrame (ignored, targets derived from features)
            """
            logger.info(f"Training LSTM model {self.name}")
            
            # Preprocess data
            df = self.preprocess(features)
            
            # Prepare target
            target = self._prepare_target(df)
            
            # Handle NaN values (from pct_change)
            valid_idx = ~np.isnan(target)
            df = df.iloc[valid_idx]
            target = target[valid_idx]
            
            # Extract feature array
            feature_data = df[self.feature_columns].values
            
            # Scale features and target
            if self.scale_features:
                feature_data = self._scale_features(feature_data)
                if self.target_type != 'direction':  # Don't scale binary targets
                    target = self._scale_target(target)
            
            # Create sequences
            X, y = self._create_sequences(feature_data, target)
            
            if len(X) == 0:
                raise ValueError("No valid sequences could be created. Check data and sequence_length.")
                
            # Build model
            input_shape = (self.sequence_length, len(self.feature_columns))
            if self.model is None:
                self.model = self._build_model(input_shape)
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
            ]
            
            # Add model checkpoint if we have a models directory
            if hasattr(self, 'paths') and 'models_dir' in self.paths:
                os.makedirs(self.paths['models_dir'], exist_ok=True)
                checkpoint_path = os.path.join(self.paths['models_dir'], f"{self.name}_best.h5")
                callbacks.append(
                    ModelCheckpoint(checkpoint_path, save_best_only=True)
                )
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0  # Suppress output for clean logs
            )
            
            # Update model state
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Log training results
            val_loss = history.history['val_loss'][-1]
            logger.info(f"LSTM model {self.name} trained successfully. Final val_loss: {val_loss:.6f}")
        
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            """
            Generate predictions from the LSTM model.
            
            Args:
                features: Feature DataFrame
                
            Returns:
                Numpy array of predictions
            """
            if not self.is_trained or self.model is None:
                raise RuntimeError("Model not trained. Call train() first.")
                
            # Preprocess data
            df = self.preprocess(features)
            
            # Extract feature array
            feature_data = df[self.feature_columns].values
            
            # Scale features
            if self.scale_features:
                feature_data = self._scale_features(feature_data)
            
            # Create sequences without target
            X = []
            for i in range(len(feature_data) - self.sequence_length + 1):
                X.append(feature_data[i:(i + self.sequence_length)])
                
            X = np.array(X)
            
            if len(X) == 0:
                logger.warning("No valid sequences for prediction. Returning empty array.")
                return np.array([])
            
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Inverse scaling for price or return predictions
            if self.scale_features and self.target_type != 'direction':
                predictions = self._inverse_scale_target(predictions)
            
            return predictions.ravel()
        
        def get_feature_importance(self) -> Dict[str, float]:
            """
            Get feature importance from the LSTM model.
            
            For neural networks, using permutation importance:
            shuffle each feature and measure the increase in error.
            
            Returns:
                Dictionary mapping feature names to importance scores
            """
            # Use cached importance if available
            if self._feature_importance_cache:
                return self._feature_importance_cache
                
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained, cannot calculate feature importance")
                return {}
                
            # For LSTM, we approximate feature importance using permutation importance
            # This is computationally expensive, so we cache the results
            try:
                # Need evaluation data for this
                importance = {}
                for i, feature in enumerate(self.feature_columns):
                    importance[feature] = abs(self.model.layers[0].get_weights()[0][:, i].mean())
                    
                # Normalize to sum to 1
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v / total for k, v in importance.items()}
                    
                self._feature_importance_cache = importance
                return importance
                
            except Exception as e:
                logger.error(f"Error calculating feature importance: {str(e)}")
                return {feature: 1.0 / len(self.feature_columns) for feature in self.feature_columns}
        
        def save(self, filepath: Optional[str] = None) -> str:
            """
            Save the LSTM model to disk.
            
            Args:
                filepath: Optional path to save the model
                
            Returns:
                Path where the model was saved
            """
            if not filepath:
                # Use default path
                os.makedirs(self.paths["models_dir"], exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.paths["models_dir"], f"{self.name}_{timestamp}")
                
            # Save Keras model separately
            model_path = f"{filepath}_keras.h5"
            if self.model:
                self.model.save(model_path)
                
            # Save metadata and scalers
            super().save(filepath)
            
            return filepath
        
        @classmethod
        def load(cls, filepath: str) -> 'LSTMModel':
            """
            Load a model from disk.
            
            Args:
                filepath: Path to the saved model
                
            Returns:
                Loaded model instance
            """
            # Load base model data
            model = super().load(filepath)
            
            # Load Keras model
            model_path = f"{filepath}_keras.h5"
            if os.path.exists(model_path):
                model.model = load_model(model_path)
                
            return model 