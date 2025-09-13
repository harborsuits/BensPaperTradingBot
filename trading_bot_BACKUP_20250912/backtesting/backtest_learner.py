#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BacktestLearner - Manages ML model training and prediction for backtesting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import json
import pickle
import matplotlib.pyplot as plt
import time
from datetime import datetime

from trading_bot.ml.model_factory import MLModelFactory
from trading_bot.backtesting.backtest_data_manager import BacktestDataManager
from trading_bot.strategies.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)

class BacktestLearner:
    """
    Manages the machine learning model training, evaluation, and prediction
    for backtesting purposes.
    """
    
    def __init__(
        self,
        model_factory: Optional[MLModelFactory] = None,
        data_manager: Optional[BacktestDataManager] = None,
        results_dir: str = "results/ml_models"
    ):
        """
        Initialize the backtesting learner.
        
        Args:
            model_factory: MLModelFactory instance
            data_manager: BacktestDataManager instance
            results_dir: Directory to save/load models and results
        """
        self.model_factory = model_factory if model_factory else MLModelFactory(results_dir=results_dir)
        self.data_manager = data_manager if data_manager else BacktestDataManager(results_dir=results_dir)
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Dictionary to store trained models
        self.models = {}
        
        # Dictionary to store model performance metrics
        self.metrics = {}
        
        logger.info(f"Initialized BacktestLearner with results_dir={results_dir}")
    
    def train_model(
        self,
        data: pd.DataFrame,
        model_type: str,
        model_name: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        sequence_length: Optional[int] = None,
        train_test_split: Optional[Dict[str, float]] = None,
        scale_features: bool = True,
        scale_target: bool = False,
        create_features: bool = False,
        include_ta: bool = True,
        target_type: str = "returns",
        target_shift: int = 1,
        save_model: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            data: Input data (OHLCV or feature data)
            model_type: Type of model (lstm, ensemble, xgboost)
            model_name: Name of the model for saving/loading
            feature_columns: List of feature column names
            target_column: Target column name
            hyperparameters: Model hyperparameters
            sequence_length: Length of sequences for LSTM
            train_test_split: Dictionary with test_size and val_size
            scale_features: Whether to scale features
            scale_target: Whether to scale target
            create_features: Whether to create features from OHLCV data
            include_ta: Whether to include technical indicators when creating features
            target_type: Type of target (returns, log_returns, direction, binary)
            target_shift: Periods to shift for target
            save_model: Whether to save the model
            **kwargs: Additional arguments for model creation
            
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
        
        # Create features if needed
        if create_features:
            logger.info("Creating features from OHLCV data")
            # Set default target column to close if not provided
            target_column = target_column or "close"
            # Create features
            feature_data, feature_columns = self.data_manager.create_features_from_ohlcv(
                data,
                include_ta=include_ta,
                target_column=target_column,
                target_type=target_type,
                target_shift=target_shift
            )
            # Use the generated target column
            target_column = f"{target_column}_target"
        else:
            feature_data = data
            # Use provided feature and target columns or defaults
            feature_columns = feature_columns or [c for c in data.columns if c != target_column]
            target_column = target_column or data.columns[-1]
        
        logger.info(f"Training {model_type} model '{model_name}' with {len(feature_columns)} features")
        
        # Set train/test split parameters
        if train_test_split:
            self.data_manager.test_size = train_test_split.get("test_size", 0.2)
            self.data_manager.val_size = train_test_split.get("val_size", 0.1)
        
        # Set default hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = {}
        
        # Prepare data
        prepared_data = self.data_manager.prepare_features(
            feature_data,
            feature_columns,
            target_column,
            sequence_length=sequence_length if model_type == "lstm" else None,
            scale_features=scale_features,
            scale_target=scale_target
        )
        
        # Create model
        model = None
        if model_type == "lstm":
            # Set input shape based on sequence length and number of features
            input_shape = (sequence_length, len(feature_columns))
            model = self.model_factory.create_lstm_model(
                input_shape=input_shape,
                **hyperparameters
            )
        elif model_type == "ensemble":
            model = self.model_factory.create_ensemble_model(
                **hyperparameters
            )
        elif model_type == "xgboost":
            model = self.model_factory.create_xgboost_model(
                **hyperparameters
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        history = None
        if model_type == "lstm":
            # Compile model
            model.compile(
                optimizer=hyperparameters.get("optimizer", "adam"),
                loss=hyperparameters.get("loss", "mean_squared_error"),
                metrics=hyperparameters.get("metrics", ["mae"])
            )
            
            # Train model
            history = model.fit(
                prepared_data["X_train"], prepared_data["y_train"],
                validation_data=(prepared_data["X_val"], prepared_data["y_val"]),
                epochs=hyperparameters.get("epochs", 100),
                batch_size=hyperparameters.get("batch_size", 32),
                verbose=hyperparameters.get("verbose", 1),
                callbacks=hyperparameters.get("callbacks", [])
            )
            
            history = history.history
        else:
            # Train ensemble or xgboost model
            model.fit(
                prepared_data["X_train"], prepared_data["y_train"].ravel()
            )
        
        # Evaluate model
        metrics = self._evaluate_model(
            model, model_type, prepared_data, scale_target, 
            prepared_data.get("target_scaler", None)
        )
        
        # Save model
        if save_model:
            self.model_factory.save_model(model, model_name, model_type)
            self.data_manager.save_data_config(model_name, prepared_data)
            # Save metrics
            self._save_metrics(model_name, metrics)
        
        # Store model and metrics
        self.models[model_name] = model
        self.metrics[model_name] = metrics
        
        # Store training info
        training_info = {
            "model_type": model_type,
            "model_name": model_name,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "sequence_length": sequence_length,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "history": history,
            "training_time": time.time() - start_time,
            "training_date": datetime.now().isoformat(),
            "data_shape": {
                "train": (prepared_data["X_train"].shape, prepared_data["y_train"].shape),
                "val": (prepared_data["X_val"].shape, prepared_data["y_val"].shape),
                "test": (prepared_data["X_test"].shape, prepared_data["y_test"].shape)
            }
        }
        
        # Save training info
        self._save_training_info(model_name, training_info)
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Training metrics: {metrics['train']}")
        logger.info(f"Validation metrics: {metrics['val']}")
        logger.info(f"Test metrics: {metrics['test']}")
        
        return training_info
    
    def _evaluate_model(
        self,
        model: Any,
        model_type: str,
        data: Dict[str, Any],
        scale_target: bool,
        target_scaler: Any
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on training, validation, and test data.
        
        Args:
            model: Trained model
            model_type: Type of model
            data: Data dictionary
            scale_target: Whether target was scaled
            target_scaler: Scaler for target
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "train": {},
            "val": {},
            "test": {}
        }
        
        # Evaluate on each dataset
        for dataset in ["train", "val", "test"]:
            X = data[f"X_{dataset}"]
            y_true = data[f"y_{dataset}"]
            
            # Make predictions
            if model_type == "lstm":
                y_pred = model.predict(X)
            else:
                y_pred = model.predict(X).reshape(-1, 1)
            
            # Inverse transform if target was scaled
            if scale_target and target_scaler is not None:
                y_true = target_scaler.inverse_transform(y_true)
                y_pred = target_scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            metrics[dataset] = self._calculate_metrics(y_true, y_pred)
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            r2_score, accuracy_score, precision_score,
            recall_score, f1_score
        )
        
        # Ensure correct shape
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        # Basic regression metrics
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        
        # Add classification metrics if appropriate
        # Check if data is binary (0/1)
        unique_values = set(np.unique(y_true).tolist())
        if len(unique_values) == 2 and (0 in unique_values or 1 in unique_values):
            # Convert predictions to binary
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate classification metrics
            metrics.update({
                "accuracy": accuracy_score(y_true, y_pred_binary),
                "precision": precision_score(y_true, y_pred_binary, zero_division=0),
                "recall": recall_score(y_true, y_pred_binary, zero_division=0),
                "f1": f1_score(y_true, y_pred_binary, zero_division=0)
            })
        
        return metrics
    
    def load_model(
        self,
        model_name: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model and its data configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (model, data_config)
        """
        try:
            # Load model
            model = self.model_factory.load_model(model_name)
            
            # Load data configuration
            data_config = self.data_manager.load_data_config(model_name)
            
            # Store model
            self.models[model_name] = model
            
            # Load metrics
            metrics_path = os.path.join(self.results_dir, model_name, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.metrics[model_name] = json.load(f)
            
            logger.info(f"Loaded model {model_name}")
            return model, data_config
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None, {}
    
    def predict(
        self,
        model_name: str,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        sequence_length: Optional[int] = None,
        scale_features: bool = True,
        scale_target: bool = False
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model
            data: Input data
            feature_columns: Feature columns to use
            sequence_length: Length of sequences for LSTM
            scale_features: Whether to scale features
            scale_target: Whether to scale target (for inverse transform)
            
        Returns:
            Numpy array with predictions
        """
        # Load model if not already loaded
        if model_name not in self.models:
            model, data_config = self.load_model(model_name)
            if model is None:
                logger.error(f"Failed to load model {model_name}")
                return None
            
            # Get feature columns and sequence length from data config if not provided
            feature_columns = feature_columns or data_config.get("feature_columns", None)
            sequence_length = sequence_length or data_config.get("sequence_length", None)
        else:
            model = self.models[model_name]
        
        # Check if feature columns are available
        if feature_columns is None:
            logger.error(f"Feature columns not provided for model {model_name}")
            return None
        
        # Prepare data for prediction
        X = self.data_manager.get_prediction_data(
            data,
            feature_columns,
            sequence_length=sequence_length,
            scale_features=scale_features
        )
        
        if X is None:
            logger.error("Failed to prepare prediction data")
            return None
        
        # Make prediction
        try:
            y_pred = model.predict(X)
            
            # Inverse transform if needed
            if scale_target and self.data_manager.target_scaler is not None:
                y_pred = self.data_manager.inverse_transform_target(y_pred)
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def create_ml_strategy(
        self,
        model_name: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> MLStrategy:
        """
        Create an ML strategy from a trained model.
        
        Args:
            model_name: Name of the model
            strategy_params: Additional strategy parameters
            
        Returns:
            MLStrategy instance
        """
        # Default strategy parameters
        if strategy_params is None:
            strategy_params = {}
        
        # Load model if not already loaded
        if model_name not in self.models:
            model, data_config = self.load_model(model_name)
            if model is None:
                logger.error(f"Failed to load model {model_name}")
                return None
        else:
            model = self.models[model_name]
            data_config = self.data_manager.load_data_config(model_name)
        
        # Create strategy
        strategy = self.model_factory.create_ml_strategy(
            model=model,
            model_name=model_name,
            feature_columns=data_config.get("feature_columns", []),
            sequence_length=data_config.get("sequence_length", None),
            scale_features=data_config.get("scale_features", True),
            scale_target=data_config.get("scale_target", False),
            feature_scaler=data_config.get("feature_scaler", None),
            target_scaler=data_config.get("target_scaler", None),
            **strategy_params
        )
        
        return strategy
    
    def plot_training_history(
        self,
        model_name: str,
        figsize: Tuple[int, int] = (12, 6),
        plot_validation: bool = True,
        save_fig: bool = False
    ) -> plt.Figure:
        """
        Plot the training history for an LSTM model.
        
        Args:
            model_name: Name of the model
            figsize: Figure size
            plot_validation: Whether to plot validation metrics
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Load training info if not available
        training_info_path = os.path.join(self.results_dir, model_name, "training_info.json")
        if not os.path.exists(training_info_path):
            logger.error(f"Training info not found for model {model_name}")
            return None
        
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        
        history = training_info.get("history", None)
        if history is None:
            logger.error(f"No training history found for model {model_name}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(history["loss"], label="Training Loss")
        if plot_validation and "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Validation Loss")
        axes[0].set_title(f"{model_name} - Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        
        # Plot metrics
        for metric in history.keys():
            if metric != "loss" and metric != "val_loss" and not metric.startswith("val_"):
                axes[1].plot(history[metric], label=f"Training {metric}")
                if plot_validation and f"val_{metric}" in history:
                    axes[1].plot(history[f"val_{metric}"], label=f"Validation {metric}")
        
        axes[1].set_title(f"{model_name} - Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric Value")
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            fig_path = os.path.join(self.results_dir, model_name, "training_history.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved training history figure to {fig_path}")
        
        return fig
    
    def plot_predictions(
        self,
        model_name: str,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        sequence_length: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6),
        num_points: int = 100,
        save_fig: bool = False
    ) -> plt.Figure:
        """
        Plot model predictions against actual values.
        
        Args:
            model_name: Name of the model
            data: Input data
            feature_columns: Feature columns to use
            target_column: Target column name
            sequence_length: Length of sequences for LSTM
            figsize: Figure size
            num_points: Number of points to plot
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Load model if not already loaded
        if model_name not in self.models:
            model, data_config = self.load_model(model_name)
            if model is None:
                logger.error(f"Failed to load model {model_name}")
                return None
            
            # Get feature columns, target column, and sequence length from data config if not provided
            feature_columns = feature_columns or data_config.get("feature_columns", None)
            target_column = target_column or data_config.get("target_column", None)
            sequence_length = sequence_length or data_config.get("sequence_length", None)
        else:
            model = self.models[model_name]
            data_config = self.data_manager.load_data_config(model_name)
        
        # Check if feature columns and target column are available
        if feature_columns is None or target_column is None:
            logger.error(f"Feature columns or target column not provided for model {model_name}")
            return None
        
        # Prepare data
        if sequence_length is not None:
            # For LSTM models
            # Need to create sequences
            X = np.zeros((len(data) - sequence_length, sequence_length, len(feature_columns)))
            y = np.zeros(len(data) - sequence_length)
            
            for i in range(len(data) - sequence_length):
                X[i] = data[i:i+sequence_length][feature_columns].values
                y[i] = data.iloc[i+sequence_length][target_column]
            
            # Scale features if needed
            if data_config.get("scale_features", True) and data_config.get("feature_scaler", None) is not None:
                X_reshaped = X.reshape(-1, X.shape[2])
                X_reshaped = data_config["feature_scaler"].transform(X_reshaped)
                X = X_reshaped.reshape(X.shape)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Inverse transform if needed
            if data_config.get("scale_target", False) and data_config.get("target_scaler", None) is not None:
                y = data_config["target_scaler"].inverse_transform(y.reshape(-1, 1)).ravel()
                y_pred = data_config["target_scaler"].inverse_transform(y_pred).ravel()
            else:
                y_pred = y_pred.ravel()
        else:
            # For non-sequential models
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Scale features if needed
            if data_config.get("scale_features", True) and data_config.get("feature_scaler", None) is not None:
                X = data_config["feature_scaler"].transform(X)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Inverse transform if needed
            if data_config.get("scale_target", False) and data_config.get("target_scaler", None) is not None:
                y = data_config["target_scaler"].inverse_transform(y.reshape(-1, 1)).ravel()
                y_pred = data_config["target_scaler"].inverse_transform(y_pred.reshape(-1, 1)).ravel()
            else:
                y_pred = y_pred.ravel()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Limit number of points to plot
        if len(y) > num_points:
            start_idx = len(y) - num_points
            y = y[start_idx:]
            y_pred = y_pred[start_idx:]
            plot_dates = data.index[start_idx+sequence_length:] if sequence_length is not None else data.index[start_idx:]
        else:
            plot_dates = data.index[sequence_length:] if sequence_length is not None else data.index
        
        # Plot actual values
        ax.plot(plot_dates, y, label="Actual", color="blue", alpha=0.7)
        
        # Plot predicted values
        ax.plot(plot_dates, y_pred, label="Predicted", color="red", alpha=0.7)
        
        # Add labels and legend
        ax.set_title(f"{model_name} - Predictions vs Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_column)
        ax.legend()
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            fig_path = os.path.join(self.results_dir, model_name, "predictions.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved predictions figure to {fig_path}")
        
        return fig
    
    def _save_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save model metrics to disk.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary with metrics
        """
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save metrics to JSON file
            metrics_path = os.path.join(model_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Saved metrics for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def _save_training_info(
        self,
        model_name: str,
        training_info: Dict[str, Any]
    ) -> None:
        """
        Save training information to disk.
        
        Args:
            model_name: Name of the model
            training_info: Dictionary with training information
        """
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Clean non-serializable objects
            training_info_serializable = {}
            for key, value in training_info.items():
                # Skip non-serializable objects
                if key in ["model", "feature_scaler", "target_scaler"]:
                    continue
                
                # Convert numpy arrays to lists
                if isinstance(value, np.ndarray):
                    training_info_serializable[key] = value.tolist()
                # Convert tuples to lists
                elif isinstance(value, tuple):
                    training_info_serializable[key] = list(value)
                else:
                    training_info_serializable[key] = value
            
            # Save training info to JSON file
            training_info_path = os.path.join(model_dir, "training_info.json")
            with open(training_info_path, "w") as f:
                json.dump(training_info_serializable, f, indent=4)
            
            logger.info(f"Saved training info for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving training info: {str(e)}")
    
    def get_model_info(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Get information about a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        try:
            # Check if model directory exists
            model_dir = os.path.join(self.results_dir, model_name)
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return {}
            
            # Load training info
            training_info_path = os.path.join(model_dir, "training_info.json")
            if not os.path.exists(training_info_path):
                logger.error(f"Training info not found: {training_info_path}")
                return {}
            
            with open(training_info_path, "r") as f:
                training_info = json.load(f)
            
            # Load metrics
            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                training_info["metrics"] = metrics
            
            return training_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def list_models(self) -> List[str]:
        """
        List all available trained models.
        
        Returns:
            List of model names
        """
        try:
            # List subdirectories in results directory
            model_dirs = [d for d in os.listdir(self.results_dir) 
                         if os.path.isdir(os.path.join(self.results_dir, d))]
            
            # Filter to only include directories with model files
            models = []
            for model_dir in model_dirs:
                model_path = os.path.join(self.results_dir, model_dir)
                # Check for model files
                if any(f.endswith(".h5") or f.endswith(".joblib") or f.endswith(".pkl") 
                       for f in os.listdir(model_path)):
                    models.append(model_dir)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def compare_models(
        self,
        model_names: List[str],
        metric: str = "rmse",
        dataset: str = "test"
    ) -> Dict[str, float]:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            model_names: List of model names to compare
            metric: Metric to use for comparison
            dataset: Dataset to use for comparison (train, val, test)
            
        Returns:
            Dictionary with model names and metric values
        """
        if dataset not in ["train", "val", "test"]:
            logger.error(f"Invalid dataset: {dataset}. Must be one of train, val, test.")
            return {}
        
        comparison = {}
        for model_name in model_names:
            # Get model metrics
            model_info = self.get_model_info(model_name)
            
            if not model_info or "metrics" not in model_info:
                logger.warning(f"No metrics found for model {model_name}")
                continue
            
            # Get metric value
            try:
                metric_value = model_info["metrics"][dataset][metric]
                comparison[model_name] = metric_value
            except KeyError:
                logger.warning(f"Metric {metric} not found for model {model_name} on {dataset} dataset")
        
        # Sort by metric value (assuming lower is better, e.g., RMSE, MSE, MAE)
        return dict(sorted(comparison.items(), key=lambda x: x[1]))
    
    def hyperparameter_search(
        self,
        data: pd.DataFrame,
        model_type: str,
        model_name_prefix: str,
        param_grid: Dict[str, List[Any]],
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        sequence_length: Optional[int] = None,
        scale_features: bool = True,
        scale_target: bool = False,
        create_features: bool = False,
        include_ta: bool = True,
        target_type: str = "returns",
        target_shift: int = 1,
        metric: str = "rmse",
        dataset: str = "val",
        n_iter: Optional[int] = None,
        random_search: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            data: Input data
            model_type: Type of model
            model_name_prefix: Prefix for model names
            param_grid: Parameter grid to search
            feature_columns: List of feature column names
            target_column: Target column name
            sequence_length: Length of sequences for LSTM
            scale_features: Whether to scale features
            scale_target: Whether to scale target
            create_features: Whether to create features from OHLCV data
            include_ta: Whether to include technical indicators when creating features
            target_type: Type of target (returns, log_returns, direction, binary)
            target_shift: Periods to shift for target
            metric: Metric to use for evaluation
            dataset: Dataset to use for evaluation (train, val, test)
            n_iter: Number of iterations for random search
            random_search: Whether to use random search instead of grid search
            **kwargs: Additional arguments for model creation
            
        Returns:
            Dictionary with search results
        """
        from sklearn.model_selection import ParameterGrid
        import random
        
        if dataset not in ["train", "val", "test"]:
            logger.error(f"Invalid dataset: {dataset}. Must be one of train, val, test.")
            return {}
        
        # Generate parameter combinations
        param_grid_list = list(ParameterGrid(param_grid))
        
        # If using random search, take a random subset of parameter combinations
        if random_search and n_iter is not None and n_iter < len(param_grid_list):
            param_grid_list = random.sample(param_grid_list, n_iter)
        
        logger.info(f"Performing hyperparameter search with {len(param_grid_list)} combinations")
        
        # Train model for each parameter combination
        results = []
        best_metric_value = float("inf") if metric in ["mse", "rmse", "mae"] else float("-inf")
        best_model_name = None
        
        for i, params in enumerate(param_grid_list):
            model_name = f"{model_name_prefix}_{i}"
            
            try:
                # Train model with current parameters
                training_info = self.train_model(
                    data=data,
                    model_type=model_type,
                    model_name=model_name,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    hyperparameters=params,
                    sequence_length=sequence_length,
                    scale_features=scale_features,
                    scale_target=scale_target,
                    create_features=create_features,
                    include_ta=include_ta,
                    target_type=target_type,
                    target_shift=target_shift,
                    save_model=True,
                    **kwargs
                )
                
                # Get metric value
                metric_value = training_info["metrics"][dataset][metric]
                
                # Check if this is the best model
                is_better = False
                if metric in ["mse", "rmse", "mae"]:
                    is_better = metric_value < best_metric_value
                else:
                    is_better = metric_value > best_metric_value
                
                if is_better:
                    best_metric_value = metric_value
                    best_model_name = model_name
                
                # Store result
                results.append({
                    "model_name": model_name,
                    "params": params,
                    f"{metric}_{dataset}": metric_value,
                    "training_time": training_info["training_time"]
                })
                
                logger.info(f"Combination {i+1}/{len(param_grid_list)}: {metric}={metric_value:.4f}")
                
            except Exception as e:
                logger.error(f"Error training model with parameters {params}: {str(e)}")
        
        # Sort results by metric value
        if metric in ["mse", "rmse", "mae"]:
            results = sorted(results, key=lambda x: x[f"{metric}_{dataset}"])
        else:
            results = sorted(results, key=lambda x: x[f"{metric}_{dataset}"], reverse=True)
        
        # Return search results
        search_results = {
            "best_model_name": best_model_name,
            "best_params": results[0]["params"] if results else None,
            f"best_{metric}_{dataset}": best_metric_value,
            "results": results
        }
        
        # Save search results
        search_results_path = os.path.join(self.results_dir, f"{model_name_prefix}_search_results.json")
        try:
            with open(search_results_path, "w") as f:
                json.dump(search_results, f, indent=4)
            logger.info(f"Saved search results to {search_results_path}")
        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
        
        return search_results 