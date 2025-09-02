#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Machine Learning Strategy implementation.

This module implements a robust machine learning strategy with model training,
validation, feature engineering, and deployment capabilities for trading.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import joblib
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from trading_bot.strategies.strategy_template import StrategyOptimizable
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame
from trading_bot.market.market_data import MarketData
from trading_bot.utils.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

class MLStrategy(StrategyOptimizable):
    """
    Advanced strategy based on machine learning predictions.
    
    This strategy:
    1. Provides comprehensive feature engineering
    2. Supports multiple ML algorithms with hyperparameter optimization
    3. Handles proper time-series cross validation
    4. Manages model lifecycle (training, validation, deployment)
    5. Includes model monitoring and diagnostics
    6. Offers ensemble methods and model stacking
    """
    
    # Default ML strategy parameters
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'ml_strategy',
        'strategy_version': '2.0.0',
        
        # Universe selection
        'min_price': 10.0,                     # Minimum price to consider
        'min_volume': 100000,                  # Minimum average volume
        
        # Feature engineering
        'lookback_periods': [5, 10, 20, 50],   # Lookback periods for features
        'include_technicals': True,            # Include technical indicators
        'include_fundamentals': False,         # Include fundamental data
        'include_sentiment': False,            # Include sentiment data
        'ta_feature_sets': ['momentum', 'trend', 'volatility', 'volume'],  # Technical feature categories
        
        # Feature engineering parameters
        'normalize_features': True,            # Whether to normalize features
        'add_pca_components': False,           # Add principal components
        'feature_selection': 'importance',     # Feature selection method ('none', 'importance', 'statistical')
        'max_features': 50,                    # Maximum number of features to use
        
        # Training parameters
        'model_type': 'ensemble',              # 'random_forest', 'gradient_boosting', 'logistic', 'neural_network', 'ensemble'
        'target_type': 'classification',       # 'classification' or 'regression'
        'target_horizon': 5,                   # Prediction horizon in days
        'target_threshold': 0.01,              # Minimum price change to consider for classification target
        'train_size': 0.7,                     # Percentage of data to use for training
        'cross_validation_folds': 5,           # Number of folds for cross-validation
        'train_frequency': 'monthly',          # How often to retrain the model
        
        # Signal generation
        'prediction_threshold': 0.6,           # Threshold for signal generation (for classification)
        'confidence_level': 0.1,               # Confidence level for signal filtering
        'signal_smoothing': True,              # Whether to apply signal smoothing
        'smoothing_window': 3,                 # Window size for signal smoothing
        
        # Position sizing
        'position_sizing': 'risk_adjusted',    # 'fixed', 'volatility_adjusted', 'risk_adjusted', 'prediction_scaled'
        'max_position_size': 0.05,             # Maximum position size as a fraction of portfolio value
        'min_position_size': 0.01,             # Minimum position size as a fraction of portfolio value
        
        # Risk management
        'stop_loss_atr_multiple': 2.0,         # Stop loss as multiple of ATR
        'take_profit_atr_multiple': 3.0,       # Take profit as multiple of ATR
        'trailing_stop': True,                 # Use trailing stop
        'max_drawdown_exit': 0.05,             # Maximum drawdown before exiting position
        
        # Model monitoring
        'enable_monitoring': True,             # Enable model monitoring
        'monitoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'drift_detection': True,               # Detect data drift
        
        # Deployment
        'model_save_path': 'models/',          # Path to save models
        'fallback_to_baseline': True,          # Fallback to baseline model if primary fails
        'validation_before_deployment': True,  # Validate model before deployment
    }
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        models: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the enhanced machine learning strategy.
        
        Args:
            name: Name of the strategy
            parameters: Strategy parameters (will override DEFAULT_PARAMS)
            metadata: Additional metadata
            models: Pre-trained models to use
        """
        # Start with default parameters
        ml_params = self.DEFAULT_PARAMS.copy()
        
        # Update with provided parameters
        if parameters:
            ml_params.update(parameters)
        
        # Initialize the parent class
        super().__init__(name=name, parameters=ml_params, metadata=metadata)
        
        # Initialize model pipeline
        self.models = models or {}  # Dictionary of trained models
        self.feature_engineering = FeatureEngineering(self.params)
        self.latest_features = {}  # Cache for latest feature data
        self.prediction_history = {}  # Track prediction history for evaluation
        self.model_metrics = {}  # Track model performance metrics
        
        # Create model save directory if it doesn't exist
        if self.params['model_save_path']:
            os.makedirs(self.params['model_save_path'], exist_ok=True)
        
        logger.info(f"Initialized ML Strategy: {name}")
    
    def prepare_training_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Prepare features and target variables for model training.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price/indicator data
            
        Returns:
            Tuple of (features_dict, targets_dict) with one entry per symbol
        """
        features_dict = {}
        targets_dict = {}
        
        for symbol, df in data.items():
            try:
                # Generate features
                features = self.feature_engineering.generate_features(df)
                
                # Generate target variables based on future returns
                if 'close' in df.columns:
                    horizon = self.params['target_horizon']
                    
                    # Calculate future returns
                    future_returns = df['close'].pct_change(horizon).shift(-horizon)
                    
                    if self.params['target_type'] == 'classification':
                        # Create binary target based on threshold
                        threshold = self.params['target_threshold']
                        target = pd.Series(0, index=df.index)
                        target[future_returns > threshold] = 1  # Buy signal
                        target[future_returns < -threshold] = -1  # Sell signal
                    else:  # regression
                        # Use future returns directly as target
                        target = future_returns
                    
                    # Remove NaN targets (usually at the end due to the future window)
                    valid_idx = ~target.isna()
                    if valid_idx.sum() == 0:
                        logger.warning(f"No valid target data for {symbol}")
                        continue
                    
                    features = features.loc[valid_idx]
                    target = target.loc[valid_idx]
                    
                    # Store prepared data
                    features_dict[symbol] = features
                    targets_dict[symbol] = target
                else:
                    logger.warning(f"No close price data for {symbol}")
            except Exception as e:
                logger.error(f"Error preparing training data for {symbol}: {str(e)}")
        
        return features_dict, targets_dict
    
    def train_model(self, features: pd.DataFrame, target: pd.Series, model_name: str = 'primary') -> Dict[str, Any]:
        """
        Train a machine learning model with proper cross-validation.
        
        Args:
            features: Feature DataFrame
            target: Target variable Series
            model_name: Name of the model to train
            
        Returns:
            Dictionary with trained model and training metrics
        """
        try:
            # Define preprocessing steps
            preprocessing_steps = []
            if self.params['normalize_features']:
                preprocessing_steps.append(('scaler', StandardScaler()))
            
            # Define model
            model_type = self.params['model_type']
            
            # Base estimator selection
            if model_type == 'random_forest':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [5, 10, 20, None],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1],
                    'model__max_depth': [3, 4, 5]
                }
            elif model_type == 'logistic':
                model = LogisticRegression(random_state=42)
                param_grid = {
                    'model__C': [0.1, 1.0, 10.0],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                }
            elif model_type == 'ensemble':
                # Create a voting ensemble of multiple models
                from sklearn.ensemble import VotingClassifier
                
                estimators = [
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
                    ('lr', LogisticRegression(C=1.0, random_state=42))
                ]
                model = VotingClassifier(estimators=estimators, voting='soft')
                param_grid = {}  # No hyperparameter tuning for the ensemble
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Create pipeline
            preprocessing_steps.append(('model', model))
            pipeline = Pipeline(preprocessing_steps)
            
            # Set up time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.params['cross_validation_folds'])
            
            # Perform hyperparameter tuning with time series CV if param_grid is not empty
            if param_grid:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(features, target)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_results = grid_search.cv_results_
            else:
                # Just fit the model without hyperparameter tuning
                pipeline.fit(features, target)
                best_model = pipeline
                best_params = {}
                cv_results = {}
            
            # Calculate feature importances for feature selection
            feature_importances = {}
            if hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
                model_step = best_model.named_steps['model']
                if hasattr(model_step, 'feature_importances_'):
                    importances = model_step.feature_importances_
                    feature_importances = dict(zip(features.columns, importances))
            
            # Calculate performance metrics on the training set
            predictions = best_model.predict(features)
            
            metrics = {}
            if self.params['target_type'] == 'classification':
                metrics['accuracy'] = accuracy_score(target, predictions)
                metrics['precision'] = precision_score(target, predictions, average='weighted')
                metrics['recall'] = recall_score(target, predictions, average='weighted')
                metrics['f1'] = f1_score(target, predictions, average='weighted')
                
                # Calculate ROC AUC if binary classification
                if len(np.unique(target)) == 2:
                    # For binary classification
                    prob_predictions = best_model.predict_proba(features)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(target, prob_predictions)
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics['mse'] = mean_squared_error(target, predictions)
                metrics['mae'] = mean_absolute_error(target, predictions)
                metrics['r2'] = r2_score(target, predictions)
            
            # Save the model
            if self.params['model_save_path']:
                model_path = os.path.join(
                    self.params['model_save_path'],
                    f"{self.name}_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                )
                joblib.dump(best_model, model_path)
                
                # Save feature importance
                if feature_importances:
                    feature_path = os.path.join(
                        self.params['model_save_path'],
                        f"{self.name}_{model_name}_features_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(feature_path, 'w') as f:
                        json.dump(feature_importances, f, indent=2)
            
            # Store model in memory
            self.models[model_name] = best_model
            self.model_metrics[model_name] = metrics
            
            # Log training results
            logger.info(f"Trained model {model_name} for {self.name} with metrics: {metrics}")
            
            return {
                'model': best_model,
                'metrics': metrics,
                'best_params': best_params,
                'feature_importances': feature_importances,
                'cv_results': cv_results
            }
            
        except Exception as e:
            logger.error(f"Error training model for {self.name}: {str(e)}")
            return None
    
    def validate_model(self, model: Any, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Validate a trained model on out-of-sample data.
        
        Args:
            model: Trained model
            features: Feature DataFrame
            target: Target variable Series
            
        Returns:
            Dictionary of validation metrics
        """
        try:
            # Make predictions
            predictions = model.predict(features)
            
            # Calculate performance metrics
            metrics = {}
            if self.params['target_type'] == 'classification':
                metrics['accuracy'] = accuracy_score(target, predictions)
                metrics['precision'] = precision_score(target, predictions, average='weighted')
                metrics['recall'] = recall_score(target, predictions, average='weighted')
                metrics['f1'] = f1_score(target, predictions, average='weighted')
                
                # Calculate ROC AUC if binary classification
                if len(np.unique(target)) == 2:
                    # For binary classification
                    prob_predictions = model.predict_proba(features)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(target, prob_predictions)
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics['mse'] = mean_squared_error(target, predictions)
                metrics['mae'] = mean_absolute_error(target, predictions)
                metrics['r2'] = r2_score(target, predictions)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            return {}
    
    def load_model(self, model_path: str, model_name: str = 'primary') -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            model_name: Name to assign to the loaded model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file {model_path} not found")
                return False
                
            model = joblib.load(model_path)
            self.models[model_name] = model
            logger.info(f"Loaded model {model_name} from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, features: pd.DataFrame, model_name: str = 'primary') -> np.ndarray:
        """
        Generate predictions using a trained model.
        
        Args:
            features: Feature DataFrame
            model_name: Name of the model to use
            
        Returns:
            Array of predictions
        """
        try:
            # Check if model exists
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return np.array([])
                
            model = self.models[model_name]
            
            # Make predictions
            predictions = model.predict(features)
            
            # If classification model, also get probabilities
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)
            
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {'predictions': np.array([]), 'probabilities': None}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals based on ML predictions.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price/indicator data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        for symbol, df in data.items():
            try:
                # Generate features
                features = self.feature_engineering.generate_features(df)
                
                # Store latest features for future reference
                self.latest_features[symbol] = features
                
                # Make sure we have a trained model
                if not self.models or 'primary' not in self.models:
                    logger.warning(f"No trained model available for {symbol}")
                    continue
                
                # Get latest feature values (most recent row)
                if features.empty:
                    logger.warning(f"No features available for {symbol}")
                    continue
                    
                latest_features = features.iloc[-1:].copy()
                
                # Make predictions
                prediction_result = self.predict(latest_features, 'primary')
                predictions = prediction_result['predictions']
                probabilities = prediction_result['probabilities']
                
                if len(predictions) == 0:
                    logger.warning(f"No predictions generated for {symbol}")
                    continue
                
                prediction = predictions[0]
                
                # Determine signal type based on prediction
                signal_type = SignalType.NEUTRAL
                if self.params['target_type'] == 'classification':
                    if prediction > 0:
                        signal_type = SignalType.BUY
                    elif prediction < 0:
                        signal_type = SignalType.SELL
                else:  # regression
                    threshold = self.params['prediction_threshold']
                    if prediction > threshold:
                        signal_type = SignalType.BUY
                    elif prediction < -threshold:
                        signal_type = SignalType.SELL
                
                # Calculate confidence based on probabilities
                confidence = 0.5
                if probabilities is not None and len(probabilities) > 0:
                    # Get probability of predicted class
                    if prediction > 0:  # Buy signal
                        confidence = probabilities[0][1]  # Probability of positive class
                    elif prediction < 0:  # Sell signal
                        confidence = probabilities[0][0]  # Probability of negative class
                
                # Only generate a signal if confidence exceeds threshold
                if confidence < self.params['confidence_level'] and signal_type != SignalType.NEUTRAL:
                    signal_type = SignalType.NEUTRAL
                
                # Get latest price data
                latest_price = df['close'].iloc[-1] if 'close' in df.columns else 0.0
                latest_timestamp = df.index[-1] if len(df.index) > 0 else datetime.datetime.now()
                
                # Calculate stop loss and take profit levels
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'].iloc[-1] - df['low'].iloc[-1])
                
                stop_loss = None
                take_profit = None
                
                if signal_type == SignalType.BUY:
                    stop_loss = latest_price - (atr * self.params['stop_loss_atr_multiple'])
                    take_profit = latest_price + (atr * self.params['take_profit_atr_multiple'])
                elif signal_type == SignalType.SELL:
                    stop_loss = latest_price + (atr * self.params['stop_loss_atr_multiple'])
                    take_profit = latest_price - (atr * self.params['take_profit_atr_multiple'])
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timeframe=TimeFrame.DAY_1,  # Assuming daily timeframe
                    metadata={
                        'prediction': float(prediction),
                        'model_name': 'primary',
                        'feature_count': len(latest_features.columns),
                        'strategy_type': 'ml_strategy'
                    }
                )
                
                signals[symbol] = signal
                
                # Track prediction for evaluation
                self.prediction_history.setdefault(symbol, []).append({
                    'timestamp': latest_timestamp,
                    'prediction': float(prediction),
                    'confidence': confidence,
                    'signal_type': signal_type.value,
                })
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        return signals
    
    def calculate_position_size(self, signal: Signal, account_value: float) -> float:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            signal: Signal object
            account_value: Current account value
            
        Returns:
            Position size in dollar amount
        """
        if signal.signal_type == SignalType.NEUTRAL:
            return 0.0
        
        sizing_method = self.params['position_sizing']
        max_position = self.params['max_position_size'] * account_value
        min_position = self.params['min_position_size'] * account_value
        
        if sizing_method == 'fixed':
            position_size = max_position
        
        elif sizing_method == 'volatility_adjusted':
            # Adjust position size based on volatility
            if 'atr' in signal.metadata:
                atr = signal.metadata['atr']
                atr_ratio = 0.02 / (atr / signal.price)  # Standardize to 2% ATR
                position_size = max_position * min(1.0, max(0.2, atr_ratio))
            else:
                position_size = max_position * 0.5  # Default if ATR is not available
        
        elif sizing_method == 'risk_adjusted':
            # Position size based on risk per trade
            risk_percent = 0.01  # Risk 1% of account per trade
            
            if signal.stop_loss is not None:
                # Risk amount divided by stop loss distance
                risk_amount = account_value * risk_percent
                stop_distance = abs(signal.price - signal.stop_loss)
                
                if stop_distance > 0:
                    raw_size = risk_amount / stop_distance
                    position_size = min(max_position, raw_size)
                else:
                    position_size = min_position
            else:
                position_size = min_position
        
        elif sizing_method == 'prediction_scaled':
            # Scale position size by prediction confidence
            position_size = min_position + (max_position - min_position) * signal.confidence
        
        else:
            # Default to fixed size
            position_size = max_position
        
        # Ensure position size is within limits
        position_size = min(max_position, max(min_position, position_size))
        
        return position_size
    
    def evaluate_predictions(self, window_days: int = 30) -> Dict[str, Dict[str, float]]:
        """
        Evaluate recent predictions against actual outcomes.
        
        Args:
            window_days: Number of days to evaluate
            
        Returns:
            Dictionary of evaluation metrics by symbol
        """
        metrics = {}
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=window_days)
        
        for symbol, predictions in self.prediction_history.items():
            # Filter recent predictions
            recent_predictions = [p for p in predictions if p['timestamp'] > cutoff_date]
            
            if not recent_predictions:
                continue
            
            # Placeholder for actual calculation once outcomes are available
            # This would require market data with future prices relative to prediction time
            
            # For now, just compute statistics about the predictions
            buy_signals = len([p for p in recent_predictions if p['signal_type'] == 'BUY'])
            sell_signals = len([p for p in recent_predictions if p['signal_type'] == 'SELL'])
            neutral_signals = len([p for p in recent_predictions if p['signal_type'] == 'NEUTRAL'])
            
            avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
            
            metrics[symbol] = {
                'total_predictions': len(recent_predictions),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'neutral_signals': neutral_signals,
                'avg_confidence': avg_confidence
            }
        
        return metrics
    
    def detect_data_drift(self, new_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect data drift between new data and reference data.
        
        Args:
            new_data: New feature data
            reference_data: Reference feature data
            
        Returns:
            Dictionary of drift metrics
        """
        if not self.params['drift_detection']:
            return {}
        
        try:
            from scipy.stats import ks_2samp
            
            drift_metrics = {}
            
            # Check common columns
            common_cols = set(new_data.columns) & set(reference_data.columns)
            
            for col in common_cols:
                # Skip non-numeric columns
                if not np.issubdtype(new_data[col].dtype, np.number):
                    continue
                
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = ks_2samp(new_data[col].dropna(), reference_data[col].dropna())
                
                drift_metrics[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }
            
            # Calculate overall drift score
            drift_detected = any(m['drift_detected'] for m in drift_metrics.values())
            avg_ks = np.mean([m['ks_statistic'] for m in drift_metrics.values()])
            
            return {
                'feature_metrics': drift_metrics,
                'drift_detected': drift_detected,
                'average_ks': avg_ks
            }
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return {'error': str(e)}
    
    def get_optimization_params(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            # Feature engineering parameters
            'lookback_periods': [[5, 10, 20], [5, 10, 20, 50], [10, 20, 50, 100]],
            'normalize_features': [True, False],
            'max_features': [20, 50, 100],
            
            # Model training parameters
            'model_type': ['random_forest', 'gradient_boosting', 'logistic', 'ensemble'],
            'target_horizon': [1, 3, 5, 10],
            'target_threshold': [0.005, 0.01, 0.02],
            
            # Signal generation parameters
            'prediction_threshold': [0.5, 0.6, 0.7],
            'confidence_level': [0.05, 0.1, 0.2],
            
            # Risk management parameters
            'stop_loss_atr_multiple': [1.5, 2.0, 2.5],
            'take_profit_atr_multiple': [2.0, 3.0, 4.0],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary for serialization.
        
        Returns:
            Strategy as dictionary
        """
        return {
            "type": "ml_strategy",
            "name": self.name,
            "parameters": self.params,
            "metadata": self.metadata,
            "model_metrics": self.model_metrics,
            "version": "2.0.0"
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MLStrategy":
        """
        Create strategy from dictionary.
        
        Args:
            config: Strategy configuration
            
        Returns:
            MLStrategy instance
        """
        name = config.get("name", "ml_strategy")
        parameters = config.get("parameters", {})
        metadata = config.get("metadata", {})
        
        # Create strategy instance
        strategy = cls(name=name, parameters=parameters, metadata=metadata)
        
        # Load model if path is provided
        model_path = parameters.get("model_path")
        if model_path and os.path.exists(model_path):
            strategy.load_model(model_path)
        
        return strategy 