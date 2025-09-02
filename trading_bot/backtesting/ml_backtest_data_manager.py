#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Backtest Data Manager - Manages data for ML-based backtesting.
"""

import logging
import os
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class BacktestDataManager:
    """
    Manages data operations for ML-based backtesting.
    
    This class handles:
    - Loading and preprocessing historical data
    - Feature generation and selection
    - Data splitting for training/validation/testing
    - Saving and loading learning iterations
    - Managing experiment tracking for ML backtests
    """
    
    def __init__(
        self, 
        data_dir: str = "data/backtest_data",
        results_dir: str = "data/learning_results",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ML backtest data manager.
        
        Args:
            data_dir: Directory for input data storage
            results_dir: Directory for learning results storage
            config: Configuration options
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.config = config or {}
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track currently loaded data and features
        self.current_data = {}
        self.feature_sets = {}
        self.learning_iterations = []
        
        # Initialize preprocessors
        self.scalers = {}
        
        logger.info(f"Initialized ML Backtest Data Manager with data dir: {data_dir}")
    
    def load_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """
        Load historical market data for a specific symbol.
        
        Args:
            symbol: Market symbol to load data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (1d, 1h, etc.)
            
        Returns:
            DataFrame with market data
        """
        # Construct file path based on symbol and frequency
        file_name = f"{symbol}_{frequency}.csv"
        file_path = self.data_dir / file_name
        
        try:
            # Load data from CSV
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(data)} rows for {symbol} ({frequency})")
            
            # Filter by date range if provided
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
                
            # Store in current data dictionary
            key = f"{symbol}_{frequency}"
            self.current_data[key] = data
            
            return data
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return pd.DataFrame()
    
    def generate_features(
        self,
        data: pd.DataFrame,
        feature_set: str = "default",
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate features from raw market data.
        
        Args:
            data: Input market data
            feature_set: Name of feature set to generate
            params: Parameters for feature generation
            
        Returns:
            DataFrame with generated features
        """
        params = params or {}
        features = data.copy()
        
        if feature_set == "default":
            # Basic price-based features
            if "close" in features.columns:
                # Price-based features
                features["returns"] = features["close"].pct_change()
                features["log_returns"] = np.log(features["close"] / features["close"].shift(1))
                
                # Moving averages
                for window in [5, 10, 20, 50, 200]:
                    features[f"ma_{window}"] = features["close"].rolling(window=window).mean()
                    features[f"ma_ratio_{window}"] = features["close"] / features[f"ma_{window}"]
                
                # Volatility estimates
                for window in [10, 20, 50]:
                    features[f"volatility_{window}"] = features["returns"].rolling(window=window).std()
                
                # RSI 
                delta = features["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                features["rsi_14"] = 100 - (100 / (1 + rs))
                
                # MACD
                features["ema_12"] = features["close"].ewm(span=12, adjust=False).mean()
                features["ema_26"] = features["close"].ewm(span=26, adjust=False).mean()
                features["macd"] = features["ema_12"] - features["ema_26"]
                features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
                features["macd_hist"] = features["macd"] - features["macd_signal"]
            
            # Volume-based features if available
            if "volume" in features.columns:
                features["volume_ma_10"] = features["volume"].rolling(window=10).mean()
                features["volume_ratio"] = features["volume"] / features["volume_ma_10"]
        
        elif feature_set == "advanced":
            # Include all default features first
            features = self.generate_features(data, feature_set="default")
            
            # Add advanced features
            if "high" in features.columns and "low" in features.columns:
                # Bollinger Bands
                window = params.get("bb_window", 20)
                std_dev = params.get("bb_std", 2)
                
                features[f"bb_ma_{window}"] = features["close"].rolling(window=window).mean()
                features[f"bb_std_{window}"] = features["close"].rolling(window=window).std()
                features[f"bb_upper_{window}"] = features[f"bb_ma_{window}"] + (features[f"bb_std_{window}"] * std_dev)
                features[f"bb_lower_{window}"] = features[f"bb_ma_{window}"] - (features[f"bb_std_{window}"] * std_dev)
                features[f"bb_width_{window}"] = (features[f"bb_upper_{window}"] - features[f"bb_lower_{window}"]) / features[f"bb_ma_{window}"]
                features[f"bb_pct_{window}"] = (features["close"] - features[f"bb_lower_{window}"]) / (features[f"bb_upper_{window}"] - features[f"bb_lower_{window}"])
                
                # ATR (Average True Range)
                tr1 = abs(features["high"] - features["low"])
                tr2 = abs(features["high"] - features["close"].shift(1))
                tr3 = abs(features["low"] - features["close"].shift(1))
                
                features["tr"] = pd.DataFrame([tr1, tr2, tr3]).max()
                features["atr_14"] = features["tr"].rolling(window=14).mean()
                
                # Stochastic Oscillator
                window = params.get("stoch_window", 14)
                k_period = params.get("stoch_k", 3)
                d_period = params.get("stoch_d", 3)
                
                # Calculate %K
                low_min = features["low"].rolling(window=window).min()
                high_max = features["high"].rolling(window=window).max()
                
                features["stoch_k"] = 100 * ((features["close"] - low_min) / (high_max - low_min))
                features["stoch_d"] = features["stoch_k"].rolling(window=d_period).mean()
                
                # Ichimoku Cloud
                tenkan_window = params.get("ichimoku_tenkan", 9)
                kijun_window = params.get("ichimoku_kijun", 26)
                senkou_span_b_window = params.get("ichimoku_senkou_b", 52)
                
                # Tenkan-sen (Conversion Line)
                features["ichimoku_tenkan"] = (features["high"].rolling(window=tenkan_window).max() + 
                                              features["low"].rolling(window=tenkan_window).min()) / 2
                
                # Kijun-sen (Base Line)
                features["ichimoku_kijun"] = (features["high"].rolling(window=kijun_window).max() + 
                                              features["low"].rolling(window=kijun_window).min()) / 2
                
                # Senkou Span A (Leading Span A)
                features["ichimoku_senkou_a"] = ((features["ichimoku_tenkan"] + features["ichimoku_kijun"]) / 2).shift(kijun_window)
                
                # Senkou Span B (Leading Span B)
                features["ichimoku_senkou_b"] = ((features["high"].rolling(window=senkou_span_b_window).max() + 
                                                features["low"].rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_window)
        
        elif feature_set == "ml_optimized":
            # Include advanced features
            features = self.generate_features(data, feature_set="advanced", params=params)
            
            # Add features specifically useful for ML models
            if "returns" in features.columns:
                # Lagged returns
                for lag in range(1, 6):
                    features[f"returns_lag_{lag}"] = features["returns"].shift(lag)
                
                # Lagged volatility
                for lag in range(1, 3):
                    features[f"volatility_10_lag_{lag}"] = features["volatility_10"].shift(lag)
                
                # Target variable engineering
                features["target_next_return"] = features["returns"].shift(-1)
                features["target_next_direction"] = np.sign(features["target_next_return"])
                
                # Rolling statistics of returns
                features["returns_skew_20"] = features["returns"].rolling(window=20).skew()
                features["returns_kurt_20"] = features["returns"].rolling(window=20).kurt()
        
        # Save the feature set
        self.feature_sets[feature_set] = features
        
        # Drop rows with NaN values that result from calculations
        features = features.dropna()
        
        return features
    
    def split_data(
        self,
        features: pd.DataFrame,
        target_col: str = "target_next_return",
        test_size: float = 0.2,
        validation_size: float = 0.2,
        shuffle: bool = False,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            features: Feature DataFrame
            target_col: Target column name
            test_size: Fraction of data for testing
            validation_size: Fraction of training data for validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if target_col not in features.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")
        
        # Extract target variable
        y = features[target_col]
        X = features.drop(columns=[target_col])
        
        if "target_next_direction" in X.columns:
            X = X.drop(columns=["target_next_direction"])
        
        # First split into train+val and test
        if shuffle:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            # Time series split (no shuffle)
            test_idx = int(len(X) * (1 - test_size))
            X_train_val, X_test = X.iloc[:test_idx], X.iloc[test_idx:]
            y_train_val, y_test = y.iloc[:test_idx], y.iloc[test_idx:]
        
        # Then split train+val into train and val
        if shuffle:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=validation_size / (1 - test_size),
                random_state=random_state
            )
        else:
            # Time series split (no shuffle)
            val_idx = int(len(X_train_val) * (1 - validation_size / (1 - test_size)))
            X_train, X_val = X_train_val.iloc[:val_idx], X_train_val.iloc[val_idx:]
            y_train, y_val = y_train_val.iloc[:val_idx], y_train_val.iloc[val_idx:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        scaler_type: str = "standard",
        feature_selection: bool = False,
        n_features: int = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Preprocess data by scaling and optionally performing feature selection.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            scaler_type: Type of scaler ('standard' or 'minmax')
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Tuple of (X_train_processed, X_val_processed, X_test_processed)
        """
        # Initialize scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit scaler on training data and transform
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Save the scaler
        self.scalers[scaler_type] = scaler
        
        # Transform validation data if provided
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        # Feature selection if enabled
        if feature_selection and n_features is not None:
            # Basic feature selection based on correlation with target
            # In a real implementation, you'd want more sophisticated methods
            # like mutual information, feature importance from a model, etc.
            feature_correlations = X_train.abs().mean().sort_values(ascending=False)
            selected_features = feature_correlations.head(n_features).index.tolist()
            
            X_train_scaled = X_train_scaled[selected_features]
            
            if X_val_scaled is not None:
                X_val_scaled = X_val_scaled[selected_features]
                
            if X_test_scaled is not None:
                X_test_scaled = X_test_scaled[selected_features]
                
            logger.info(f"Selected {n_features} features based on mean absolute value")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_learning_iteration(
        self,
        iteration_id: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        model_params: Dict[str, Any] = None,
        model = None
    ) -> bool:
        """
        Save a learning iteration including configuration, results, and model.
        
        Args:
            iteration_id: Unique identifier for this iteration
            config: Configuration used for this iteration
            results: Performance results
            model_params: Model parameters
            model: Trained model object (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create iteration directory
            iteration_dir = self.results_dir / iteration_id
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(iteration_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Save results
            with open(iteration_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Save model parameters if provided
            if model_params:
                with open(iteration_dir / "model_params.json", "w") as f:
                    json.dump(model_params, f, indent=2)
            
            # Save model if provided
            if model is not None:
                try:
                    with open(iteration_dir / "model.pkl", "wb") as f:
                        pickle.dump(model, f)
                except Exception as e:
                    logger.warning(f"Could not save model: {str(e)}")
            
            # Add to learning iterations
            iteration_info = {
                "id": iteration_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "config": config,
                "results": results
            }
            self.learning_iterations.append(iteration_info)
            
            # Save learning iterations index
            self._save_learning_index()
            
            logger.info(f"Saved learning iteration: {iteration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving learning iteration: {str(e)}")
            return False
    
    def load_learning_iteration(
        self,
        iteration_id: str
    ) -> Dict[str, Any]:
        """
        Load a learning iteration with its configuration, results, and model.
        
        Args:
            iteration_id: Unique identifier for the iteration
            
        Returns:
            Dictionary with iteration data
        """
        iteration_dir = self.results_dir / iteration_id
        
        if not iteration_dir.exists():
            logger.error(f"Learning iteration not found: {iteration_id}")
            return {}
        
        try:
            # Load configuration
            with open(iteration_dir / "config.json", "r") as f:
                config = json.load(f)
            
            # Load results
            with open(iteration_dir / "results.json", "r") as f:
                results = json.load(f)
            
            # Load model parameters if available
            model_params = {}
            if (iteration_dir / "model_params.json").exists():
                with open(iteration_dir / "model_params.json", "r") as f:
                    model_params = json.load(f)
            
            # Load model if available
            model = None
            if (iteration_dir / "model.pkl").exists():
                try:
                    with open(iteration_dir / "model.pkl", "rb") as f:
                        model = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Could not load model: {str(e)}")
            
            return {
                "id": iteration_id,
                "config": config,
                "results": results,
                "model_params": model_params,
                "model": model
            }
            
        except Exception as e:
            logger.error(f"Error loading learning iteration: {str(e)}")
            return {}
    
    def get_learning_iterations(
        self,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get a list of learning iterations.
        
        Args:
            sort_by: Field to sort by ('timestamp', 'metric_name', etc.)
            sort_order: Sort order ('asc' or 'desc')
            limit: Maximum number of iterations to return
            
        Returns:
            List of iteration dictionaries
        """
        # Load learning iterations index
        self._load_learning_index()
        
        # Make a copy to avoid modifying the original
        iterations = self.learning_iterations.copy()
        
        # Sort iterations
        if sort_by == "timestamp":
            iterations.sort(key=lambda x: x.get("timestamp", ""), reverse=(sort_order == "desc"))
        elif sort_by.startswith("results."):
            # Extract metric name from sort_by (e.g., "results.sharpe_ratio")
            metric = sort_by.split(".", 1)[1]
            iterations.sort(
                key=lambda x: x.get("results", {}).get(metric, 0), 
                reverse=(sort_order == "desc")
            )
        
        # Limit results
        return iterations[:limit]
    
    def _save_learning_index(self) -> None:
        """Save the learning iterations index file."""
        index_path = self.results_dir / "learning_index.json"
        try:
            with open(index_path, "w") as f:
                json.dump(self.learning_iterations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning index: {str(e)}")
    
    def _load_learning_index(self) -> None:
        """Load the learning iterations index file."""
        index_path = self.results_dir / "learning_index.json"
        
        if not index_path.exists():
            self.learning_iterations = []
            return
        
        try:
            with open(index_path, "r") as f:
                self.learning_iterations = json.load(f)
        except Exception as e:
            logger.error(f"Error loading learning index: {str(e)}")
            self.learning_iterations = []
    
    def delete_learning_iteration(self, iteration_id: str) -> bool:
        """
        Delete a learning iteration and its files.
        
        Args:
            iteration_id: ID of the iteration to delete
            
        Returns:
            Boolean indicating success
        """
        iteration_dir = self.results_dir / iteration_id
        
        if not iteration_dir.exists():
            logger.warning(f"Learning iteration not found for deletion: {iteration_id}")
            return False
        
        try:
            # Delete files
            for file_path in iteration_dir.glob("*"):
                os.remove(file_path)
            
            # Delete directory
            os.rmdir(iteration_dir)
            
            # Remove from index
            self._load_learning_index()
            self.learning_iterations = [
                it for it in self.learning_iterations 
                if it.get("id") != iteration_id
            ]
            self._save_learning_index()
            
            logger.info(f"Deleted learning iteration: {iteration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting learning iteration: {str(e)}")
            return False
            
    def compare_iterations(
        self,
        iteration_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple learning iterations.
        
        Args:
            iteration_ids: List of iteration IDs to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "iterations": [],
            "metrics": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        for iteration_id in iteration_ids:
            iteration = self.load_learning_iteration(iteration_id)
            
            if not iteration:
                continue
            
            # Add to iterations list
            comparison["iterations"].append({
                "id": iteration_id,
                "config": iteration.get("config", {}),
                "results": iteration.get("results", {})
            })
            
            # Track metrics for comparison
            for metric, value in iteration.get("results", {}).items():
                if metric not in comparison["metrics"]:
                    comparison["metrics"][metric] = []
                
                comparison["metrics"][metric].append({
                    "id": iteration_id,
                    "value": value
                })
        
        # Sort metrics by best value
        for metric, values in comparison["metrics"].items():
            # Assume higher is better for metrics
            values.sort(key=lambda x: x["value"], reverse=True) 