#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-Based Signal Model

This module implements feature-based signal models using XGBoost classifiers
to predict market direction and generate trading signals.
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

from trading_bot.ml.enhanced_features import EnhancedFeatureEngineering

logger = logging.getLogger(__name__)

class SignalModel:
    """ML-based signal model for market direction prediction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.feature_engine = EnhancedFeatureEngineering(config)
        
        # Set default configuration if not provided
        self._set_default_config()
        
        # Initialize model objects
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.metrics = {}
        
        logger.info("Signal Model initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Time horizons for prediction (in days)
        self.config.setdefault("prediction_horizons", [1, 3, 5])
        
        # Target thresholds for binary classification (% return)
        self.config.setdefault("target_thresholds", {
            1: 0.001,  # 0.1% for 1-day horizon
            3: 0.003,  # 0.3% for 3-day horizon
            5: 0.005   # 0.5% for 5-day horizon
        })
        
        # XGBoost parameters
        self.config.setdefault("xgb_params", {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "tree_method": "hist"
        })
        
        # Confidence thresholds for signal generation
        self.config.setdefault("confidence_thresholds", {
            "strong_buy": 0.75,
            "buy": 0.65,
            "neutral": 0.45,
            "sell": 0.35,
            "strong_sell": 0.25
        })
        
        # Model training parameters
        self.config.setdefault("test_size", 0.2)
        self.config.setdefault("random_state", 42)
        self.config.setdefault("cv_folds", 5)
        self.config.setdefault("use_calibration", True)
        
        # Feature importance settings
        self.config.setdefault("top_features", 15)
        
        # Model persistence
        self.config.setdefault("model_dir", "models/signals")
    
    def prepare_data(self, data: pd.DataFrame, horizon: int, train: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training or prediction
        
        Args:
            data: OHLCV data with datetime index
            horizon: Prediction horizon in days
            train: Whether this is for training (includes target generation)
            
        Returns:
            X: Feature DataFrame
            y: Target Series (if train=True, otherwise None)
        """
        logger.info(f"Preparing data for horizon: {horizon} days")
        
        # Generate features
        df = self.feature_engine.generate_features(data.copy())
        
        # Generate target if in training mode
        y = None
        if train:
            # Calculate forward returns
            threshold = self.config["target_thresholds"][horizon]
            forward_returns = df["close"].pct_change(horizon).shift(-horizon)
            
            # Create binary target: 1 if return >= threshold, 0 otherwise
            y = (forward_returns >= threshold).astype(int)
            
            # Drop rows with NaN targets
            valid_idx = ~y.isnull()
            df = df.loc[valid_idx]
            y = y.loc[valid_idx]
        
        # Drop non-feature columns
        cols_to_drop = ["open", "high", "low", "close", "volume"]
        feature_cols = [col for col in df.columns if col not in cols_to_drop]
        
        # Select features and drop rows with NaN values
        X = df[feature_cols].copy()
        X = X.dropna()
        
        # Align y with X if in training mode
        if train:
            y = y.loc[X.index]
        
        logger.info(f"Prepared data shape: {X.shape}")
        return X, y
    
    def train(self, data: pd.DataFrame, save_model: bool = True) -> Dict[str, Any]:
        """
        Train signal models for all prediction horizons
        
        Args:
            data: OHLCV data with datetime index
            save_model: Whether to save trained models to disk
            
        Returns:
            Dictionary of training metrics
        """
        training_metrics = {}
        
        for horizon in self.config["prediction_horizons"]:
            logger.info(f"Training model for horizon: {horizon} days")
            
            # Prepare data
            X, y = self.prepare_data(data, horizon, train=True)
            
            # Split data by time (no shuffling)
            train_size = int((1 - self.config["test_size"]) * len(X))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
            
            # Create base model
            base_model = xgb.XGBClassifier(**self.config["xgb_params"])
            
            # Apply probability calibration if enabled
            if self.config["use_calibration"]:
                model = CalibratedClassifierCV(
                    base_model,
                    cv=self.config["cv_folds"],
                    method="isotonic"
                )
            else:
                model = base_model
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                "auc": roc_auc_score(y_test, y_pred_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "accuracy": (y_pred == y_test).mean()
            }
            
            # Get feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = model.estimator.feature_importances_
                
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save model and metadata
            self.models[horizon] = model
            self.scalers[horizon] = scaler
            self.feature_importances[horizon] = feature_importance
            self.metrics[horizon] = metrics
            
            # Save models to disk if requested
            if save_model:
                self._save_model(horizon)
            
            training_metrics[horizon] = metrics
            logger.info(f"Model for {horizon}-day horizon - AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return training_metrics
    
    def _save_model(self, horizon: int):
        """
        Save model, scaler, and metadata to disk
        
        Args:
            horizon: Prediction horizon
        """
        # Create directory if it doesn't exist
        os.makedirs(self.config["model_dir"], exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.config["model_dir"], f"signal_model_{horizon}d.joblib")
        joblib.dump(self.models[horizon], model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.config["model_dir"], f"signal_scaler_{horizon}d.joblib")
        joblib.dump(self.scalers[horizon], scaler_path)
        
        # Save feature importance
        importance_path = os.path.join(self.config["model_dir"], f"feature_importance_{horizon}d.csv")
        self.feature_importances[horizon].to_csv(importance_path, index=False)
        
        # Save metrics
        metrics_path = os.path.join(self.config["model_dir"], f"metrics_{horizon}d.json")
        pd.Series(self.metrics[horizon]).to_json(metrics_path)
        
        logger.info(f"Saved model and metadata for {horizon}-day horizon to {self.config['model_dir']}")
    
    def load_model(self, horizon: int) -> bool:
        """
        Load model, scaler, and metadata from disk
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Success flag
        """
        try:
            # Load model
            model_path = os.path.join(self.config["model_dir"], f"signal_model_{horizon}d.joblib")
            self.models[horizon] = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.config["model_dir"], f"signal_scaler_{horizon}d.joblib")
            self.scalers[horizon] = joblib.load(scaler_path)
            
            # Load feature importance
            importance_path = os.path.join(self.config["model_dir"], f"feature_importance_{horizon}d.csv")
            self.feature_importances[horizon] = pd.read_csv(importance_path)
            
            # Load metrics
            metrics_path = os.path.join(self.config["model_dir"], f"metrics_{horizon}d.json")
            self.metrics[horizon] = pd.read_json(metrics_path, typ="series").to_dict()
            
            logger.info(f"Loaded model and metadata for {horizon}-day horizon")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model for {horizon}-day horizon: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Generate predictions for all horizons
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            Dictionary of prediction DataFrames keyed by horizon
        """
        predictions = {}
        
        for horizon in self.config["prediction_horizons"]:
            # Check if model exists
            if horizon not in self.models:
                # Try to load model
                if not self.load_model(horizon):
                    logger.warning(f"No model available for {horizon}-day horizon")
                    continue
            
            # Prepare data
            X, _ = self.prepare_data(data, horizon, train=False)
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scalers[horizon].transform(X),
                index=X.index,
                columns=X.columns
            )
            
            # Generate predictions
            probabilities = self.models[horizon].predict_proba(X_scaled)[:, 1]
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame({
                'date': X.index,
                'probability': probabilities,
                'signal': self._probability_to_signal(probabilities),
                'horizon': horizon
            })
            
            predictions[horizon] = pred_df
            logger.info(f"Generated predictions for {horizon}-day horizon with {len(pred_df)} samples")
        
        return predictions
    
    def _probability_to_signal(self, probabilities: np.ndarray) -> pd.Series:
        """
        Convert prediction probabilities to signal categories
        
        Args:
            probabilities: Array of prediction probabilities
            
        Returns:
            Series of signal categories
        """
        thresholds = self.config["confidence_thresholds"]
        
        # Initialize with neutral signal
        signals = np.full(len(probabilities), "neutral")
        
        # Apply thresholds
        signals = np.where(probabilities >= thresholds["strong_buy"], "strong_buy", signals)
        signals = np.where((probabilities >= thresholds["buy"]) & 
                           (probabilities < thresholds["strong_buy"]), "buy", signals)
        signals = np.where((probabilities <= thresholds["strong_sell"]), "strong_sell", signals)
        signals = np.where((probabilities <= thresholds["sell"]) & 
                           (probabilities > thresholds["strong_sell"]), "sell", signals)
        
        return pd.Series(signals)
    
    def get_top_features(self, horizon: int, top_n: int = None) -> pd.DataFrame:
        """
        Get top features for a specific horizon
        
        Args:
            horizon: Prediction horizon
            top_n: Number of top features to return (default: from config)
            
        Returns:
            DataFrame of top features and their importance scores
        """
        if horizon not in self.feature_importances:
            if not self.load_model(horizon):
                logger.error(f"No feature importance data available for {horizon}-day horizon")
                return pd.DataFrame()
        
        if top_n is None:
            top_n = self.config["top_features"]
        
        return self.feature_importances[horizon].head(top_n)
    
    def generate_ensemble_signal(self, data: pd.DataFrame, weights: Dict[int, float] = None) -> pd.DataFrame:
        """
        Generate ensemble signal by combining predictions from all horizons
        
        Args:
            data: OHLCV data with datetime index
            weights: Optional dictionary of weights for each horizon (default: equal weights)
            
        Returns:
            DataFrame with ensemble signal
        """
        # Get predictions for each horizon
        predictions = self.predict(data)
        
        if not predictions:
            logger.error("No predictions available for ensemble")
            return pd.DataFrame()
        
        # Set default weights if not provided
        if weights is None:
            horizons = list(predictions.keys())
            weights = {h: 1.0 / len(horizons) for h in horizons}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {h: w / total_weight for h, w in weights.items()}
        
        # Get the most recent date for each horizon
        ensemble_signals = []
        
        for horizon, pred_df in predictions.items():
            if pred_df.empty:
                continue
                
            # Get most recent prediction
            latest_pred = pred_df.iloc[-1].copy()
            latest_pred["weight"] = weights.get(horizon, 0)
            ensemble_signals.append(latest_pred)
        
        if not ensemble_signals:
            logger.error("No valid signals for ensemble")
            return pd.DataFrame()
        
        # Convert to DataFrame
        ensemble_df = pd.DataFrame(ensemble_signals)
        
        # Calculate weighted probability
        weighted_prob = (ensemble_df["probability"] * ensemble_df["weight"]).sum()
        
        # Convert to signal
        signal = self._probability_to_signal(np.array([weighted_prob]))[0]
        
        # Create result DataFrame
        result = pd.DataFrame({
            "date": ensemble_df["date"].iloc[0],
            "weighted_probability": weighted_prob,
            "ensemble_signal": signal,
            "component_signals": [ensemble_df["signal"].tolist()],
            "component_horizons": [ensemble_df["horizon"].tolist()],
            "component_weights": [ensemble_df["weight"].tolist()]
        }, index=[0])
        
        return result


# Utility function to create a signal model with default configuration
def create_signal_model(config: Dict[str, Any] = None) -> SignalModel:
    """
    Create a signal model with default or custom configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized SignalModel instance
    """
    return SignalModel(config)
