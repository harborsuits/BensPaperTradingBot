#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BacktestDataManager - Handles data preparation for ML-based backtesting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle

logger = logging.getLogger(__name__)

class BacktestDataManager:
    """
    Handles data preparation, feature engineering, and dataset management
    for machine learning models in backtesting.
    """
    
    def __init__(
        self,
        results_dir: str = "results/ml_models",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        scaler_type: str = "standard"
    ):
        """
        Initialize the data manager.
        
        Args:
            results_dir: Directory to save/load model and feature data
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random state for train/test splitting
            scaler_type: Type of scaler to use (standard or minmax)
        """
        self.results_dir = results_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler_type = scaler_type
        
        # Initialize scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Initialized BacktestDataManager with results_dir={results_dir}")
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        sequence_length: Optional[int] = None,
        scale_features: bool = True,
        scale_target: bool = False,
        categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare features and target data from raw data.
        
        Args:
            data: Raw data DataFrame
            feature_columns: List of column names to use as features
            target_column: Column name to use as target
            sequence_length: Length of sequences for LSTM (None for non-sequential models)
            scale_features: Whether to scale features
            scale_target: Whether to scale target
            categorical_columns: List of categorical columns to one-hot encode
            
        Returns:
            Dictionary containing prepared data
        """
        logger.info(f"Preparing features from {len(data)} data points")
        
        # Handle missing values
        data = data.copy()
        data = self._handle_missing_values(data, feature_columns + [target_column])
        
        # Handle categorical columns if provided
        if categorical_columns:
            data = self._encode_categorical_features(data, categorical_columns)
            # Update feature columns with new one-hot columns
            for col in categorical_columns:
                if col in feature_columns:
                    feature_columns.remove(col)
                    # Add new one-hot columns
                    for cat_col in [c for c in data.columns if c.startswith(f"{col}_")]:
                        feature_columns.append(cat_col)
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values.reshape(-1, 1)
        
        # Scale features
        if scale_features:
            X, self.feature_scaler = self._scale_data(X, self.feature_scaler, is_target=False)
        
        # Scale target
        if scale_target:
            y, self.target_scaler = self._scale_data(y, self.target_scaler, is_target=True)
        
        # Split data
        result = self._split_data(X, y, sequence_length)
        
        # Add metadata to result
        result.update({
            "feature_columns": feature_columns,
            "target_column": target_column,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "scale_features": scale_features,
            "scale_target": scale_target,
            "sequence_length": sequence_length
        })
        
        logger.info(f"Prepared features: X_train shape={result['X_train'].shape}, y_train shape={result['y_train'].shape}")
        return result
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with possibly missing values
            columns: Columns to check for missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Check for missing values
        missing_count = data[columns].isnull().sum().sum()
        
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values. Filling with appropriate methods.")
            
            # For each column, apply an appropriate fill method
            for col in columns:
                # Skip if no missing values
                if data[col].isnull().sum() == 0:
                    continue
                
                # Use appropriate fill method based on column type
                if pd.api.types.is_numeric_dtype(data[col]):
                    # For numeric columns, use forward fill then backward fill
                    data[col] = data[col].ffill().bfill()
                    
                    # If still has missing values, fill with mean
                    if data[col].isnull().sum() > 0:
                        data[col] = data[col].fillna(data[col].mean())
                else:
                    # For non-numeric columns, use most frequent value
                    most_frequent = data[col].mode()[0]
                    data[col] = data[col].fillna(most_frequent)
        
        return data
    
    def _encode_categorical_features(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            data: DataFrame with categorical columns
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Ensure all categorical columns exist
        for col in categorical_columns:
            if col not in data.columns:
                logger.warning(f"Categorical column {col} not found in data")
                continue
            
            # One-hot encode
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
            data = pd.concat([data, dummies], axis=1)
        
        return data
    
    def _scale_data(
        self,
        data: np.ndarray,
        scaler: Any = None,
        is_target: bool = False
    ) -> Tuple[np.ndarray, Any]:
        """
        Scale data using the specified scaler.
        
        Args:
            data: Data to scale
            scaler: Existing scaler to use, or None to create a new one
            is_target: Whether this is target data (for logging)
            
        Returns:
            Tuple of (scaled data, scaler)
        """
        data_type = "target" if is_target else "feature"
        
        # Create a new scaler if none exists
        if scaler is None:
            if self.scaler_type == "standard":
                scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaler type: {self.scaler_type}. Using StandardScaler.")
                scaler = StandardScaler()
            
            # Fit the scaler
            scaler.fit(data)
            logger.info(f"Created new {self.scaler_type} scaler for {data_type} data")
        
        # Scale the data
        scaled_data = scaler.transform(data)
        
        return scaled_data, scaler
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Feature data
            y: Target data
            sequence_length: Length of sequences for LSTM (None for non-sequential)
            
        Returns:
            Dictionary with split datasets
        """
        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        
        # Then split train+val into train and val
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=self.random_state, shuffle=True
        )
        
        # Prepare sequences if needed
        if sequence_length is not None:
            X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, sequence_length)
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, sequence_length)
            X_test_seq, y_test_seq = self._prepare_sequences(X_test, y_test, sequence_length)
            
            return {
                "X_train": X_train_seq,
                "y_train": y_train_seq,
                "X_val": X_val_seq,
                "y_val": y_val_seq,
                "X_test": X_test_seq,
                "y_test": y_test_seq,
                "X_train_raw": X_train,
                "y_train_raw": y_train,
                "X_val_raw": X_val,
                "y_val_raw": y_val,
                "X_test_raw": X_test,
                "y_test_raw": y_test
            }
        else:
            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test
            }
    
    def _prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM models.
        
        Args:
            X: Feature data
            y: Target data
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X sequences, y sequences)
        """
        # Number of samples
        n_samples = X.shape[0] - sequence_length + 1
        
        # Feature dimension
        n_features = X.shape[1]
        
        # Initialize sequence arrays
        X_seq = np.zeros((n_samples, sequence_length, n_features))
        y_seq = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+sequence_length]
            y_seq[i] = y[i+sequence_length-1]
        
        return X_seq, y_seq
    
    def inverse_transform_target(
        self,
        y_scaled: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform scaled target data.
        
        Args:
            y_scaled: Scaled target data
            
        Returns:
            Original-scale target data
        """
        if self.target_scaler is None:
            logger.warning("No target scaler found. Returning data as is.")
            return y_scaled
        
        # Ensure data is in correct shape
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(y_scaled)
    
    def save_data_config(
        self,
        model_name: str,
        data_config: Dict[str, Any]
    ) -> bool:
        """
        Save data configuration for a model.
        
        Args:
            model_name: Name of the model
            data_config: Data configuration dictionary
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Path for data config file
            config_path = os.path.join(model_dir, "data_config.pkl")
            
            # Extract non-serializable objects
            config_to_save = data_config.copy()
            
            # Save scalers separately
            if "feature_scaler" in config_to_save:
                feature_scaler = config_to_save.pop("feature_scaler")
                if feature_scaler is not None:
                    pickle.dump(feature_scaler, open(os.path.join(model_dir, "feature_scaler.pkl"), "wb"))
            
            if "target_scaler" in config_to_save:
                target_scaler = config_to_save.pop("target_scaler")
                if target_scaler is not None:
                    pickle.dump(target_scaler, open(os.path.join(model_dir, "target_scaler.pkl"), "wb"))
            
            # Remove data arrays from config
            for key in list(config_to_save.keys()):
                if isinstance(config_to_save[key], np.ndarray):
                    config_to_save.pop(key)
            
            # Save config
            pickle.dump(config_to_save, open(config_path, "wb"))
            
            logger.info(f"Saved data configuration for model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data configuration: {str(e)}")
            return False
    
    def load_data_config(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Load data configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Data configuration dictionary
        """
        try:
            # Model directory
            model_dir = os.path.join(self.results_dir, model_name)
            
            # Path for data config file
            config_path = os.path.join(model_dir, "data_config.pkl")
            
            if not os.path.exists(config_path):
                logger.error(f"Data configuration file not found: {config_path}")
                return {}
            
            # Load config
            config = pickle.load(open(config_path, "rb"))
            
            # Load scalers
            feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
            if os.path.exists(feature_scaler_path):
                config["feature_scaler"] = pickle.load(open(feature_scaler_path, "rb"))
                self.feature_scaler = config["feature_scaler"]
            
            target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
            if os.path.exists(target_scaler_path):
                config["target_scaler"] = pickle.load(open(target_scaler_path, "rb"))
                self.target_scaler = config["target_scaler"]
            
            logger.info(f"Loaded data configuration for model {model_name}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading data configuration: {str(e)}")
            return {}
    
    def create_features_from_ohlcv(
        self,
        ohlcv_data: pd.DataFrame,
        include_ta: bool = True,
        include_lagged: bool = True,
        lag_periods: List[int] = None,
        target_column: str = "close",
        target_shift: int = 1,
        target_type: str = "returns"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features from OHLCV data.
        
        Args:
            ohlcv_data: OHLCV data DataFrame
            include_ta: Whether to include technical indicators
            include_lagged: Whether to include lagged features
            lag_periods: List of periods to lag (default [1, 2, 3, 5, 10])
            target_column: Column to use for target
            target_shift: Periods to shift for target (1 = next period)
            target_type: Type of target (returns, log_returns, direction, binary)
            
        Returns:
            Tuple of (DataFrame with features, List of feature column names)
        """
        data = ohlcv_data.copy()
        
        # Set default lag periods if not provided
        if lag_periods is None:
            lag_periods = [1, 2, 3, 5, 10]
        
        # Prepare target
        if target_type == "returns":
            data[f"{target_column}_target"] = data[target_column].pct_change(target_shift)
        elif target_type == "log_returns":
            data[f"{target_column}_target"] = np.log(data[target_column] / data[target_column].shift(target_shift))
        elif target_type == "direction":
            data[f"{target_column}_target"] = np.sign(data[target_column].diff(target_shift))
        elif target_type == "binary":
            data[f"{target_column}_target"] = (data[target_column].diff(target_shift) > 0).astype(int)
        else:
            logger.warning(f"Unknown target type: {target_type}. Using returns.")
            data[f"{target_column}_target"] = data[target_column].pct_change(target_shift)
        
        # Add basic price and volume features
        feature_columns = []
        
        # Add raw features
        raw_features = ["open", "high", "low", "close", "volume"]
        for col in raw_features:
            if col in data.columns:
                feature_columns.append(col)
        
        # Add price differences
        data["high_low"] = data["high"] - data["low"]
        data["close_open"] = data["close"] - data["open"]
        feature_columns.extend(["high_low", "close_open"])
        
        # Add returns
        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        feature_columns.extend(["returns", "log_returns"])
        
        # Add lagged features
        if include_lagged:
            for col in feature_columns.copy():
                for lag in lag_periods:
                    lag_col = f"{col}_lag_{lag}"
                    data[lag_col] = data[col].shift(lag)
                    feature_columns.append(lag_col)
        
        # Add technical indicators
        if include_ta:
            try:
                import pandas_ta as ta
                
                # Add moving averages
                data.ta.sma(length=20, append=True)
                data.ta.ema(length=20, append=True)
                feature_columns.extend(["SMA_20", "EMA_20"])
                
                # Add RSI
                data.ta.rsi(length=14, append=True)
                feature_columns.append("RSI_14")
                
                # Add MACD
                data.ta.macd(fast=12, slow=26, signal=9, append=True)
                feature_columns.extend(["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"])
                
                # Add Bollinger Bands
                data.ta.bbands(length=20, append=True)
                feature_columns.extend(["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"])
                
                # Add ATR
                data.ta.atr(length=14, append=True)
                feature_columns.append("ATR_14")
                
                # Add Stochastic
                data.ta.stoch(length=14, append=True)
                feature_columns.extend(["STOCHk_14_3_3", "STOCHd_14_3_3"])
                
            except ImportError:
                logger.warning("pandas_ta not installed. Technical indicators not added.")
                logger.warning("Install with 'pip install pandas_ta'")
                
            except Exception as e:
                logger.error(f"Error adding technical indicators: {str(e)}")
        
        # Drop NaN values
        data = data.dropna()
        
        logger.info(f"Created {len(feature_columns)} features from OHLCV data")
        return data, feature_columns
    
    def get_prediction_data(
        self,
        current_data: pd.DataFrame,
        feature_columns: List[str],
        sequence_length: Optional[int] = None,
        scale_features: bool = True
    ) -> np.ndarray:
        """
        Prepare data for prediction.
        
        Args:
            current_data: Current market data
            feature_columns: Feature columns to use
            sequence_length: Length of sequences for LSTM (None for non-sequential)
            scale_features: Whether to scale features
            
        Returns:
            Prepared data for prediction
        """
        # Extract features
        X = current_data[feature_columns].values
        
        # Scale features if needed
        if scale_features and self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        
        # Prepare sequence if needed
        if sequence_length is not None:
            if len(X) < sequence_length:
                logger.warning(f"Not enough data for sequence. Need {sequence_length}, got {len(X)}")
                return None
            
            X = X[-sequence_length:].reshape(1, sequence_length, X.shape[1])
        else:
            # Get the latest data point
            X = X[-1:].reshape(1, -1)
        
        return X 