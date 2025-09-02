#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Trainer Module for Trading Strategies

This module provides model training capabilities for trading strategies,
including interpretable models, feature importance analysis, and 
cross-validation methods designed for time series data.
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from copy import deepcopy

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb

# Import SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ModelTrainer:
    """
    Model trainer class for creating and training interpretable ML models.
    
    This class provides:
    1. Training of interpretable models (linear, tree-based)
    2. Time series cross-validation
    3. Feature importance analysis
    4. Regime-specific model training
    5. Model persistence and loading
    6. Performance evaluation
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model trainer module.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.shap_values = {}
        self.regime_models = {}
        
        # Save configuration snapshot for reproducibility
        self._save_config_snapshot()
    
    def _save_config_snapshot(self):
        """Save a snapshot of configuration for reproducibility."""
        self.config_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'params': {k: str(v) if isinstance(v, (np.ndarray, pd.DataFrame)) else v 
                      for k, v in self.params.items()},
            'version': '1.0.0'
        }
        
        # Save to disk if specified
        if self.params.get('save_config_snapshot', False):
            output_dir = self.params.get('output_dir', './output')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'model_config_{timestamp}.json')
            
            with open(filename, 'w') as f:
                json.dump(self.config_snapshot, f, indent=2, default=str)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'classification', 
                    model_name: str = 'default', regime: str = None) -> Any:
        """
        Train a model based on specified parameters.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: 'classification' or 'regression'
            model_name: Name to identify the model
            regime: Optional market regime identifier
            
        Returns:
            Trained model
        """
        if X.empty or y.empty:
            raise ValueError("Empty feature set or target provided")
        
        # Select model based on type and parameters
        model_algo = self.params.get('model_algorithm', 'random_forest')
        
        if model_type == 'classification':
            model = self._create_classification_model(model_algo)
        else:  # regression
            model = self._create_regression_model(model_algo)
        
        # Train the model
        model.fit(X, y)
        
        # Store the model
        model_key = f"{model_name}_{regime}" if regime else model_name
        self.models[model_key] = model
        
        # Extract and store feature importance
        self._extract_feature_importance(model, X.columns, model_key)
        
        # Calculate SHAP values if enabled
        if self.params.get('calculate_shap', True) and SHAP_AVAILABLE:
            self._calculate_shap_values(model, X, model_key)
        
        return model
    
    def _create_classification_model(self, model_algo: str) -> Any:
        """Create classification model based on algorithm name"""
        if model_algo == 'logistic_regression':
            return LogisticRegression(
                C=self.params.get('logistic_C', 1.0),
                max_iter=self.params.get('max_iter', 1000),
                class_weight=self.params.get('class_weight', 'balanced'),
                random_state=self.params.get('random_seed', 42)
            )
        elif model_algo == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', None),
                min_samples_split=self.params.get('min_samples_split', 2),
                class_weight=self.params.get('class_weight', 'balanced'),
                random_state=self.params.get('random_seed', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
        elif model_algo == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.1),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                random_state=self.params.get('random_seed', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
        elif model_algo == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.1),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                random_state=self.params.get('random_seed', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
        else:
            raise ValueError(f"Unsupported classification algorithm: {model_algo}")
    
    def _create_regression_model(self, model_algo: str) -> Any:
        """Create regression model based on algorithm name"""
        if model_algo == 'linear_regression':
            return LinearRegression(
                n_jobs=self.params.get('n_jobs', -1)
            )
        elif model_algo == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', None),
                min_samples_split=self.params.get('min_samples_split', 2),
                random_state=self.params.get('random_seed', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
        elif model_algo == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.1),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                random_state=self.params.get('random_seed', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
        elif model_algo == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.1),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                random_state=self.params.get('random_seed', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
        else:
            raise ValueError(f"Unsupported regression algorithm: {model_algo}")
    
    def _extract_feature_importance(self, model: Any, feature_names: List[str], model_key: str) -> None:
        """Extract and store feature importance from the model"""
        importance_dict = {}
        
        try:
            # Different models expose feature importance in different ways
            if hasattr(model, 'feature_importances_'):
                # Random Forest, Gradient Boosting, XGBoost
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models: take absolute values for importance ranking
                importance_values = np.abs(model.coef_)
                if importance_values.ndim > 1:
                    importance_values = importance_values.mean(axis=0)
            else:
                return
            
            # Create a dictionary of feature importances
            importance_dict = dict(zip(feature_names, importance_values))
            
            # Sort by importance (descending)
            importance_dict = {k: v for k, v in sorted(
                importance_dict.items(), key=lambda item: item[1], reverse=True
            )}
            
            # Store feature importance
            self.feature_importance[model_key] = importance_dict
            
        except Exception as e:
            print(f"Error extracting feature importance: {str(e)}")
    
    def _calculate_shap_values(self, model: Any, X: pd.DataFrame, model_key: str) -> None:
        """Calculate SHAP values for model interpretability"""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Different model types need different explainers
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor, 
                                 xgb.XGBClassifier, xgb.XGBRegressor)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, (LogisticRegression, LinearRegression)):
                explainer = shap.LinearExplainer(model, X)
            else:
                explainer = shap.KernelExplainer(model.predict, X.iloc[:100])  # Sample for kernel
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Store SHAP values
            self.shap_values[model_key] = {
                'values': shap_values,
                'explainer': explainer,
                'feature_names': X.columns.tolist()
            }
            
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
    
    def time_series_cv(self, X: pd.DataFrame, y: pd.Series, 
                       model_type: str = 'classification', 
                       model_name: str = 'default') -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: 'classification' or 'regression'
            model_name: Name to identify the model
            
        Returns:
            Dictionary with CV results
        """
        n_splits = self.params.get('cv_splits', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Results storage
        cv_scores = {
            'train_scores': [],
            'test_scores': [],
            'feature_importance': [],
            'fold_indices': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create and train model
            if model_type == 'classification':
                model = self._create_classification_model(self.params.get('model_algorithm', 'random_forest'))
            else:
                model = self._create_regression_model(self.params.get('model_algorithm', 'random_forest'))
            
            model.fit(X_train, y_train)
            
            # Evaluate
            if model_type == 'classification':
                train_score = accuracy_score(y_train, model.predict(X_train))
                test_score = accuracy_score(y_test, model.predict(X_test))
                
                # Detailed metrics for test set
                y_pred = model.predict(X_test)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                # Add confusion matrix
                if len(np.unique(y)) <= 10:  # Only for a reasonable number of classes
                    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            else:
                train_score = r2_score(y_train, model.predict(X_train))
                test_score = r2_score(y_test, model.predict(X_test))
                
                # Detailed metrics for test set
                y_pred = model.predict(X_test)
                metrics = {
                    'r2': test_score,
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred)
                }
            
            cv_scores['train_scores'].append(train_score)
            cv_scores['test_scores'].append(test_score)
            cv_scores[f'fold_{fold}_metrics'] = metrics
            cv_scores['fold_indices'].append((train_idx.tolist(), test_idx.tolist()))
            
            # Extract feature importance for this fold
            if hasattr(model, 'feature_importances_'):
                fold_importance = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                fold_importance = dict(zip(X.columns, coef))
            else:
                fold_importance = {}
                
            cv_scores['feature_importance'].append(fold_importance)
        
        # Store CV results
        self.performance_metrics[model_name] = {
            'mean_train_score': np.mean(cv_scores['train_scores']),
            'mean_test_score': np.mean(cv_scores['test_scores']),
            'std_test_score': np.std(cv_scores['test_scores']),
            'cv_results': cv_scores
        }
        
        return self.performance_metrics[model_name]
    
    def train_regime_specific_models(self, X: pd.DataFrame, y: pd.Series, 
                                regime_column: str, model_type: str = 'classification',
                                base_model_name: str = 'regime') -> Dict[str, Any]:
        """
        Train specialized models for each market regime in the data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            regime_column: Column name in X that identifies the regime
            model_type: 'classification' or 'regression'
            base_model_name: Base name for the regime models
            
        Returns:
            Dictionary with regime models
        """
        if regime_column not in X.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in features")
        
        # Get unique regimes in the dataset
        regimes = X[regime_column].unique()
        print(f"Found {len(regimes)} market regimes: {regimes}")
        
        # Minimum samples required for training a regime-specific model
        min_samples = self.params.get('min_regime_samples', 30)
        
        # Store models for each regime
        regime_models = {}
        
        # Train a model for each regime
        for regime in regimes:
            # Filter data for this regime
            regime_mask = X[regime_column] == regime
            X_regime = X[regime_mask].drop(columns=[regime_column])
            y_regime = y[regime_mask]
            
            # Check if we have enough samples
            if len(y_regime) < min_samples:
                print(f"Skipping regime '{regime}' with only {len(y_regime)} samples (min required: {min_samples})")
                continue
                
            # Generate a model name for this regime
            model_name = f"{base_model_name}_{regime}"
            
            print(f"Training model for regime '{regime}' with {len(y_regime)} samples")
            
            # Get algorithm params specific to this regime (if any)
            regime_params = self.params.get(f'regime_{regime}_params', {})
            merged_params = self.params.copy()
            merged_params.update(regime_params)
            
            # Use class-specific instance with regime-specific params
            temp_trainer = deepcopy(self)
            temp_trainer.params = merged_params
            
            # Train model for this regime
            if model_type == 'classification':
                model = temp_trainer.train_classification_model(X_regime, y_regime, model_name)
            else:
                model = temp_trainer.train_regression_model(X_regime, y_regime, model_name)
                
            # Store model in this instance
            self.models[model_name] = model
            self.model_metrics[model_name] = temp_trainer.model_metrics.get(model_name, {})
            self.feature_importance[model_name] = temp_trainer.feature_importance.get(model_name, {})
            
            # Add to result
            regime_models[regime] = model
            
            print(f"Finished training model for regime '{regime}'")
            
        return regime_models
    
    def predict(self, X: pd.DataFrame, model_name: str = 'default', 
                regime: str = None, regime_column: str = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            model_name: Name of the model to use
            regime: Specific regime to use for prediction
            regime_column: Column in X that contains regime information
            
        Returns:
            Numpy array with predictions
        """
        X_copy = X.copy()
        
        # Handle regime-specific prediction
        if regime_column is not None and regime_column in X_copy.columns:
            # Predict using regime-specific models where available
            predictions = np.zeros(len(X_copy))
            
            for idx, row_regime in enumerate(X_copy[regime_column]):
                if row_regime in self.regime_models:
                    # Use regime-specific model
                    row_features = X_copy.iloc[idx:idx+1].drop(columns=[regime_column])
                    predictions[idx] = self.regime_models[row_regime].predict(row_features)[0]
                else:
                    # Fallback to default model
                    if model_name in self.models:
                        row_features = X_copy.iloc[idx:idx+1].drop(columns=[regime_column])
                        predictions[idx] = self.models[model_name].predict(row_features)[0]
            
            return predictions
        
        # If regime is specified, use that specific model
        if regime is not None:
            model_key = f"{model_name}_{regime}"
            if model_key in self.models:
                return self.models[model_key].predict(X_copy)
        
        # Otherwise use the default model
        if model_name in self.models:
            return self.models[model_name].predict(X_copy)
        
        raise ValueError(f"Model '{model_name}' not found")
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = 'default', 
                      regime: str = None, regime_column: str = None) -> np.ndarray:
        """
        Make probability predictions (for classification models).
        
        Args:
            X: Feature DataFrame
            model_name: Name of the model to use
            regime: Specific regime to use for prediction
            regime_column: Column in X that contains regime information
            
        Returns:
            Numpy array with class probabilities
        """
        X_copy = X.copy()
        
        # Similar approach to predict() but with predict_proba()
        if regime_column is not None and regime_column in X_copy.columns:
            # Initialize with zeros - shape will be (n_samples, n_classes)
            # We need to determine number of classes
            # Let's get it from the first available model
            sample_model = next(iter(self.models.values()))
            n_classes = sample_model.predict_proba(X_copy.iloc[:1]).shape[1]
            predictions = np.zeros((len(X_copy), n_classes))
            
            for idx, row_regime in enumerate(X_copy[regime_column]):
                if row_regime in self.regime_models:
                    row_features = X_copy.iloc[idx:idx+1].drop(columns=[regime_column])
                    predictions[idx] = self.regime_models[row_regime].predict_proba(row_features)[0]
                else:
                    if model_name in self.models:
                        row_features = X_copy.iloc[idx:idx+1].drop(columns=[regime_column])
                        predictions[idx] = self.models[model_name].predict_proba(row_features)[0]
            
            return predictions
        
        if regime is not None:
            model_key = f"{model_name}_{regime}"
            if model_key in self.models:
                return self.models[model_key].predict_proba(X_copy)
        
        if model_name in self.models:
            return self.models[model_name].predict_proba(X_copy)
        
        raise ValueError(f"Model '{model_name}' not found")
    
    def get_top_features(self, model_name: str = 'default', regime: str = None, 
                         top_n: int = 10) -> Dict[str, float]:
        """
        Get the top N most important features for a model.
        
        Args:
            model_name: Name of the model
            regime: Optional regime specification
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features and their importance
        """
        model_key = f"{model_name}_{regime}" if regime else model_name
        
        if model_key not in self.feature_importance:
            raise ValueError(f"No feature importance available for model '{model_key}'")
        
        # Get all features, sorted by importance
        all_features = self.feature_importance[model_key]
        
        # Return top N
        return dict(list(all_features.items())[:top_n])
    
    def get_feature_explanation(self, X: pd.DataFrame, model_name: str = 'default', 
                               regime: str = None) -> List[Dict[str, Any]]:
        """
        Get explanations for predictions based on feature contributions.
        
        Args:
            X: Feature DataFrame to explain
            model_name: Name of the model
            regime: Optional regime specification
            
        Returns:
            List of dictionaries with feature explanations for each sample
        """
        model_key = f"{model_name}_{regime}" if regime else model_name
        
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found")
        
        model = self.models[model_key]
        explanations = []
        
        # If SHAP values available, use them
        if SHAP_AVAILABLE and model_key in self.shap_values:
            shap_data = self.shap_values[model_key]
            explainer = shap_data['explainer']
            feature_names = shap_data['feature_names']
            
            # Calculate SHAP values for this data
            shap_values = explainer.shap_values(X)
            
            # Process each sample
            for i in range(len(X)):
                # Get model prediction
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba(X.iloc[i:i+1])[0]
                    confidence = np.max(prediction)
                    predicted_class = np.argmax(prediction)
                else:
                    prediction = model.predict(X.iloc[i:i+1])[0]
                    confidence = None
                    predicted_class = None
                
                # Get SHAP values for this prediction
                if isinstance(shap_values, list):  # For multi-class
                    sample_shap = shap_values[predicted_class][i] if predicted_class is not None else shap_values[0][i]
                else:
                    sample_shap = shap_values[i]
                
                # Sort features by absolute SHAP value
                feature_contributions = dict(zip(feature_names, sample_shap))
                sorted_contributions = {k: v for k, v in sorted(
                    feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True
                )}
                
                # Create explanation
                explanation = {
                    'prediction': predicted_class if predicted_class is not None else prediction,
                    'confidence': confidence,
                    'top_features': dict(list(sorted_contributions.items())[:5])
                }
                explanations.append(explanation)
        else:
            # Fallback to simpler feature importance
            if model_key in self.feature_importance:
                feature_importance = self.feature_importance[model_key]
                
                # Process each sample
                for i in range(len(X)):
                    # Get model prediction
                    if hasattr(model, 'predict_proba'):
                        prediction = model.predict_proba(X.iloc[i:i+1])[0]
                        confidence = np.max(prediction)
                        predicted_class = np.argmax(prediction)
                    else:
                        prediction = model.predict(X.iloc[i:i+1])[0]
                        confidence = None
                        predicted_class = None
                    
                    # Create explanation based on global feature importance
                    explanation = {
                        'prediction': predicted_class if predicted_class is not None else prediction,
                        'confidence': confidence,
                        'top_features': dict(list(feature_importance.items())[:5])
                    }
                    explanations.append(explanation)
            else:
                # No feature importance available
                for i in range(len(X)):
                    if hasattr(model, 'predict_proba'):
                        prediction = model.predict_proba(X.iloc[i:i+1])[0]
                        confidence = np.max(prediction)
                        predicted_class = np.argmax(prediction)
                    else:
                        prediction = model.predict(X.iloc[i:i+1])[0]
                        confidence = None
                        predicted_class = None
                        
                    explanation = {
                        'prediction': predicted_class if predicted_class is not None else prediction,
                        'confidence': confidence,
                        'top_features': {}
                    }
                    explanations.append(explanation)
        
        return explanations
    
    def save_model(self, model_name: str = 'default', regime: str = None, 
                  filepath: str = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            regime: Optional regime specification
            filepath: Optional filepath to save to
            
        Returns:
            Path where model was saved
        """
        model_key = f"{model_name}_{regime}" if regime else model_name
        
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found")
        
        if filepath is None:
            output_dir = self.params.get('model_dir', './models')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(output_dir, f'{model_key}_{timestamp}.pkl')
        
        # Create model package with model and metadata
        model_package = {
            'model': self.models[model_key],
            'config_snapshot': self.config_snapshot,
            'feature_importance': self.feature_importance.get(model_key, {}),
            'performance_metrics': self.performance_metrics.get(model_name, {}),
            'creation_date': datetime.now().isoformat()
        }
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        return filepath
    
    def load_model(self, filepath: str, model_name: str = 'loaded', 
                  regime: str = None) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            model_name: Name to assign to the loaded model
            regime: Optional regime specification
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        
        # Load from disk
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        # Extract components
        model = model_package['model']
        feature_importance = model_package.get('feature_importance', {})
        performance_metrics = model_package.get('performance_metrics', {})
        
        # Store in class
        model_key = f"{model_name}_{regime}" if regime else model_name
        self.models[model_key] = model
        
        if feature_importance:
            self.feature_importance[model_key] = feature_importance
        
        if performance_metrics and model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = performance_metrics
        
        return model
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'default', 
                      regime: str = None) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Name of the model
            regime: Optional regime specification
            
        Returns:
            Dictionary with evaluation metrics
        """
        model_key = f"{model_name}_{regime}" if regime else model_name
        
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found")
        
        model = self.models[model_key]
        y_pred = model.predict(X)
        
        # Different metrics based on problem type
        if hasattr(model, 'predict_proba'):
            # Classification metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            # Add confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
            
            # Add classification report
            report = classification_report(y, y_pred, output_dict=True)
            metrics['classification_report'] = report
            
            # Add probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                # Store prediction probabilities for ROC curve
                metrics['prediction_proba'] = y_proba.tolist()
        else:
            # Regression metrics
            metrics = {
                'r2': r2_score(y, y_pred),
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'predictions': y_pred.tolist(),
                'actuals': y.tolist()
            }
        
        # Store metrics
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = {}
        
        eval_key = f"eval_{regime}" if regime else "eval"
        self.performance_metrics[model_name][eval_key] = metrics
        
        return metrics
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series, 
                      model_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple trained models on the same test data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_names: List of model names to compare
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                metrics = self.evaluate_model(X, y, model_name)
                results[model_name] = metrics
            else:
                print(f"Model '{model_name}' not found, skipping")
        
        # Determine best model based on metrics
        if results:
            if 'accuracy' in next(iter(results.values())):
                # Classification - use accuracy
                best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
                metric = 'accuracy'
            else:
                # Regression - use R²
                best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
                metric = 'r2'
            
            results['best_model'] = {
                'name': best_model,
                'metric': metric,
                'value': results[best_model][metric]
            }
        
        return results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model trainer configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'model_algorithm': self.params.get('model_algorithm', 'random_forest'),
            'n_estimators': self.params.get('n_estimators', 100),
            'max_depth': self.params.get('max_depth', None),
            'cv_splits': self.params.get('cv_splits', 5),
            'trained_models': list(self.models.keys()),
            'regime_models': list(self.regime_models.keys()) if self.regime_models else [],
            'has_feature_importance': list(self.feature_importance.keys()),
            'has_performance_metrics': list(self.performance_metrics.keys()),
            'config_snapshot_id': self.config_snapshot['timestamp'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def build_regime_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                         regime_column: str, model_type: str = 'classification',
                         base_name: str = 'regime', include_meta: bool = True) -> Dict[str, Any]:
        """
        Build an ensemble of regime-specialist models plus a meta-model combiner.
        
        This function:
        1. Trains a specialized model for each market regime
        2. Optionally trains a meta-model that combines specialist predictions
        
        Args:
            X: Feature DataFrame
            y: Target Series
            regime_column: Column name in X that identifies the regime
            model_type: 'classification' or 'regression'
            base_name: Base name for the regime models
            include_meta: Whether to train a meta-model combiner
            
        Returns:
            Dictionary with ensemble components
        """
        if regime_column not in X.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in features")
        
        # Step 1: Train regime-specific models
        regime_models = self.train_regime_specific_models(
            X=X, 
            y=y, 
            regime_column=regime_column,
            model_type=model_type,
            base_model_name=base_name
        )
        
        ensemble_components = {
            'regime_models': regime_models,
            'model_type': model_type,
            'regime_column': regime_column,
            'base_name': base_name
        }
        
        # Step 2: Train meta-model if requested
        if include_meta and regime_models:
            print(f"Training meta-model for ensemble...")
            
            # Generate regime-specialist predictions for training meta-model
            meta_X = self._generate_ensemble_features(X, regime_column, base_name, model_type)
            
            # Create and train meta-model
            meta_model_name = f"{base_name}_meta"
            meta_model = self.train_meta_model(meta_X, y, meta_model_name, model_type)
            
            ensemble_components['meta_model'] = meta_model
            ensemble_components['meta_model_name'] = meta_model_name
            
        return ensemble_components
    
    def train_meta_model(self, X: pd.DataFrame, y: pd.Series, model_name: str, 
                       model_type: str = 'classification') -> Any:
        """
        Train a meta-model that combines predictions from regime-specialist models.
        
        Args:
            X: Feature DataFrame with model outputs
            y: Target Series
            model_name: Name for the meta-model
            model_type: 'classification' or 'regression'
            
        Returns:
            Trained meta-model
        """
        meta_algo = self.params.get('meta_algorithm', 'logistic_regression')
        
        if model_type == 'classification':
            if meta_algo == 'logistic_regression':
                model = LogisticRegression(
                    C=self.params.get('meta_logistic_C', 1.0),
                    max_iter=self.params.get('max_iter', 1000),
                    class_weight=self.params.get('meta_class_weight', 'balanced'),
                    random_state=self.params.get('random_seed', 42)
                )
            elif meta_algo == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=self.params.get('meta_n_estimators', 50),
                    max_depth=self.params.get('meta_max_depth', 3),
                    random_state=self.params.get('random_seed', 42),
                    n_jobs=self.params.get('n_jobs', -1)
                )
            elif meta_algo == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=self.params.get('meta_n_estimators', 50),
                    max_depth=self.params.get('meta_max_depth', 3),
                    learning_rate=self.params.get('meta_learning_rate', 0.05),
                    random_state=self.params.get('random_seed', 42),
                    n_jobs=self.params.get('n_jobs', -1)
                )
            else:
                raise ValueError(f"Unsupported meta algorithm: {meta_algo}")
        else:
            if meta_algo == 'linear_regression':
                model = LinearRegression(
                    n_jobs=self.params.get('n_jobs', -1)
                )
            elif meta_algo == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=self.params.get('meta_n_estimators', 50),
                    max_depth=self.params.get('meta_max_depth', 3),
                    random_state=self.params.get('random_seed', 42),
                    n_jobs=self.params.get('n_jobs', -1)
                )
            elif meta_algo == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=self.params.get('meta_n_estimators', 50),
                    max_depth=self.params.get('meta_max_depth', 3),
                    learning_rate=self.params.get('meta_learning_rate', 0.05),
                    random_state=self.params.get('random_seed', 42),
                    n_jobs=self.params.get('n_jobs', -1)
                )
            else:
                raise ValueError(f"Unsupported meta algorithm: {meta_algo}")
        
        # Train the meta-model
        model.fit(X, y)
        
        # Store the model
        self.models[model_name] = model
        
        # Extract and store feature importance if available
        self._extract_feature_importance(model, X.columns, model_name)
        
        return model
    
    def _generate_ensemble_features(self, X: pd.DataFrame, 
                                   regime_column: str, 
                                   base_name: str = 'regime',
                                   model_type: str = 'classification') -> pd.DataFrame:
        """
        Generate features for meta-model by collecting predictions from all regime models.
        
        Args:
            X: Input features with regime column
            regime_column: Column name in X that identifies the regime
            base_name: Base name for regime models
            model_type: 'classification' or 'regression'
            
        Returns:
            DataFrame with predictions from all regime models as features
        """
        if regime_column not in X.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in features")
            
        # Get all regimes for which we have models
        all_regimes = [key.replace(f"{base_name}_", "") for key in self.models.keys() 
                      if key.startswith(f"{base_name}_")]
        
        # Create a DataFrame to hold ensemble features
        meta_features = pd.DataFrame(index=X.index)
        
        # For each regime model, get predictions for all data
        for regime in all_regimes:
            model_name = f"{base_name}_{regime}"
            
            # Skip if model doesn't exist
            if model_name not in self.models:
                continue
                
            # Get features without regime column
            X_features = X.drop(columns=[regime_column])
                
            # Get predictions
            if model_type == 'classification':
                # Add predicted class
                preds = self.predict_classification(X_features, model_name)
                meta_features[f"{model_name}_pred"] = preds
                
                # Try to add probabilities if available
                try:
                    probs = self.predict_proba(X_features, model_name)
                    # Add probability for each class
                    for i, prob in enumerate(probs.T):
                        meta_features[f"{model_name}_prob_{i}"] = prob
                except:
                    # If probabilities not available, just use predictions
                    pass
            else:
                # For regression, just add predicted values
                preds = self.predict_regression(X_features, model_name)
                meta_features[f"{model_name}_pred"] = preds
                
        # Add the regime as a feature
        meta_features['regime'] = X[regime_column]
        
        return meta_features
    
    def ensemble_predict(self, X: pd.DataFrame, 
                        regime_column: str, 
                        base_name: str = 'regime',
                        meta_model_name: str = 'meta_model',
                        model_type: str = 'classification') -> np.ndarray:
        """
        Make predictions using the ensemble of regime-specific models.
        
        Args:
            X: Input features with regime column
            regime_column: Column name in X that identifies the regime
            base_name: Base name for the regime models
            meta_model_name: Name of the meta-model
            model_type: 'classification' or 'regression'
            
        Returns:
            Array of predictions
        """
        if regime_column not in X.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in features")
            
        # If meta-model exists, use it for predictions
        if meta_model_name in self.models:
            # Generate meta-features
            meta_features = self._generate_ensemble_features(
                X, 
                regime_column=regime_column,
                base_name=base_name,
                model_type=model_type
            )
            
            # Predict using meta-model
            if model_type == 'classification':
                return self.predict_classification(meta_features, meta_model_name)
            else:
                return self.predict_regression(meta_features, meta_model_name)
        
        # Fallback: Use regime-specific models directly
        predictions = np.zeros(len(X))
        
        # For each row, use the appropriate regime-specific model
        for regime in X[regime_column].unique():
            # Get mask for this regime
            regime_mask = X[regime_column] == regime
            
            # Skip if no samples for this regime
            if not any(regime_mask):
                continue
                
            # Get model name
            model_name = f"{base_name}_{regime}"
            
            # Skip if model doesn't exist
            if model_name not in self.models:
                print(f"Warning: No model found for regime {regime}")
                continue
                
            # Get features without regime column
            X_regime = X.loc[regime_mask].drop(columns=[regime_column])
            
            # Predict using the appropriate regime-specific model
            if model_type == 'classification':
                regime_preds = self.predict_classification(X_regime, model_name)
            else:
                regime_preds = self.predict_regression(X_regime, model_name)
                
            # Store predictions
            predictions[regime_mask] = regime_preds
            
        return predictions
    
    def ensemble_predict_proba(self, X: pd.DataFrame, 
                              regime_column: str, 
                              base_name: str = 'regime',
                              meta_model_name: str = 'meta_model') -> np.ndarray:
        """
        Get probability predictions using the ensemble of regime-specific models.
        
        Args:
            X: Input features with regime column
            regime_column: Column name in X that identifies the regime
            base_name: Base name for the regime models
            meta_model_name: Name of the meta-model
            
        Returns:
            Array of class probabilities
        """
        if regime_column not in X.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in features")
            
        # Use meta-model if available
        if meta_model_name in self.models:
            # Generate meta-features
            meta_features = self._generate_ensemble_features(
                X, 
                regime_column=regime_column,
                base_name=base_name,
                model_type='classification'
            )
            
            # Get probability predictions from meta-model
            return self.predict_proba(meta_features, meta_model_name)
            
        # Fallback: Use regime-specific models directly
        # Get number of classes from the first available model
        model_names = [name for name in self.models.keys() if name.startswith(f"{base_name}_")]
        if not model_names:
            raise ValueError(f"No models found with base name '{base_name}'")
            
        # Try to get the number of classes from the first model
        first_preds = self.predict_proba(X.iloc[[0]].drop(columns=[regime_column]), model_names[0])
        n_classes = first_preds.shape[1]
        
        # Initialize probabilities with zeros
        probabilities = np.zeros((len(X), n_classes))
        
        # For each regime, predict using the corresponding model
        for regime in X[regime_column].unique():
            # Get mask for this regime
            regime_mask = X[regime_column] == regime
            
            # Skip if no samples for this regime
            if not any(regime_mask):
                continue
                
            # Get model name
            model_name = f"{base_name}_{regime}"
            
            # Skip if model doesn't exist
            if model_name not in self.models:
                print(f"Warning: No model found for regime {regime}")
                continue
                
            # Get features without regime column
            X_regime = X.loc[regime_mask].drop(columns=[regime_column])
            
            # Get probability predictions
            try:
                regime_probs = self.predict_proba(X_regime, model_name)
                # Store predictions
                probabilities[regime_mask] = regime_probs
            except:
                # If probabilities not available, set to -1 (indicating error)
                pass
                
        return probabilities
    
    def get_ensemble_explanation(self, X: pd.DataFrame, 
                               base_name: str = 'regime') -> Dict[str, Any]:
        """
        Get a detailed explanation of an ensemble prediction.
        
        Args:
            X: Feature DataFrame
            base_name: Base name used for the ensemble models
            
        Returns:
            Dictionary with ensemble explanation
        """
        # Get predictions and regime contributions
        prediction, regime_contribs = self.ensemble_predict(
            X, base_name, include_regime_probs=True
        )
        
        # Get meta-model explanation
        meta_model_name = f"{base_name}_meta"
        meta_model = self.models[meta_model_name]
        
        # Get current regime if available
        current_regime = regime_contribs.get('current_regime', 'unknown')
        
        # Get explanation text
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
            
        if hasattr(meta_model, 'predict_proba'):
            # Classification case
            decision = "BUY" if prediction > 0 else "SELL" if prediction < 0 else "HOLD"
            confidence = regime_contribs.get('meta_confidence', None)
            
            explanation_text = f"Ensemble prediction: {decision}"
            if confidence:
                explanation_text += f" with {confidence:.1%} confidence"
                
            explanation_text += f". Current market regime: {current_regime}."
        else:
            # Regression case
            direction = "upward" if prediction > 0 else "downward" if prediction < 0 else "flat"
            explanation_text = f"Ensemble predicts {direction} movement of {prediction:.2f}%. "
            explanation_text += f"Current market regime: {current_regime}."
            
        # Sort regime contributions by importance
        regime_importance = {k: v for k, v in regime_contribs.items() 
                           if k not in ['meta_confidence', 'current_regime']}
        
        sorted_regimes = sorted(regime_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Add regime contributions to explanation
        explanation_text += " Regime model contributions: "
        regime_explanations = []
        
        for regime, contribution in sorted_regimes[:3]:  # Top 3 contributing regimes
            direction = "positive" if contribution > 0 else "negative"
            regime_explanations.append(
                f"{regime} model: {direction} ({abs(contribution):.2f})"
            )
            
        explanation_text += ", ".join(regime_explanations)
        
        # Create explanation result
        explanation = {
            'prediction': prediction,
            'confidence': regime_contribs.get('meta_confidence'),
            'current_regime': current_regime,
            'regime_contributions': regime_importance,
            'explanation_text': explanation_text
        }
        
        return explanation
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, 
                          y_test: pd.Series,
                          regime_column: str,
                          base_name: str = 'regime',
                          meta_model_name: str = 'meta_model',
                          model_type: str = 'classification') -> Dict:
        """
        Evaluate the performance of a regime-based ensemble model.
        
        Args:
            X_test: Test feature DataFrame
            y_test: Test target Series
            regime_column: Column in X_test identifying the regime
            base_name: Base name for regime models
            meta_model_name: Name of the meta-model
            model_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with evaluation results
        """
        if regime_column not in X_test.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in features")
            
        # Get unique regimes
        unique_regimes = X_test[regime_column].unique()
        
        # Initialize results dictionary
        results = {
            'overall': {},
            'regimes': {},
            'models': {}
        }
        
        # Get ensemble predictions
        ensemble_preds = self.ensemble_predict(
            X_test, 
            regime_column=regime_column,
            base_name=base_name,
            meta_model_name=meta_model_name,
            model_type=model_type
        )
        
        # Calculate overall performance metrics
        if model_type == 'classification':
            # Classification metrics
            results['overall']['accuracy'] = accuracy_score(y_test, ensemble_preds)
            results['overall']['f1'] = f1_score(y_test, ensemble_preds, average='weighted')
            results['overall']['precision'] = precision_score(y_test, ensemble_preds, average='weighted')
            results['overall']['recall'] = recall_score(y_test, ensemble_preds, average='weighted')
            
            # Get classification report
            try:
                results['overall']['report'] = classification_report(y_test, ensemble_preds, output_dict=True)
            except:
                pass
                
            # Get confusion matrix
            try:
                results['overall']['confusion_matrix'] = confusion_matrix(y_test, ensemble_preds).tolist()
            except:
                pass
                
            # Try to get ROC AUC if binary classification
            try:
                # Get probabilities for positive class
                probs = self.ensemble_predict_proba(
                    X_test, 
                    regime_column=regime_column,
                    base_name=base_name,
                    meta_model_name=meta_model_name
                )[:, 1]
                results['overall']['roc_auc'] = roc_auc_score(y_test, probs)
            except:
                pass
        else:
            # Regression metrics
            results['overall']['mse'] = mean_squared_error(y_test, ensemble_preds)
            results['overall']['mae'] = mean_absolute_error(y_test, ensemble_preds)
            results['overall']['r2'] = r2_score(y_test, ensemble_preds)
            
        # Evaluate each regime-specific model
        for regime in unique_regimes:
            # Get regime-specific data
            regime_mask = X_test[regime_column] == regime
            X_regime = X_test.loc[regime_mask].drop(columns=[regime_column])
            y_regime = y_test.loc[regime_mask]
            
            # Skip if too few samples
            if len(y_regime) < 5:
                continue
                
            # Get model name for this regime
            model_name = f"{base_name}_{regime}"
            
            # Skip if model doesn't exist
            if model_name not in self.models:
                continue
                
            # Initialize regime results
            results['regimes'][regime] = {
                'sample_count': len(y_regime),
                'metrics': {}
            }
            
            # Get predictions from regime-specific model
            if model_type == 'classification':
                # Get predictions
                regime_preds = self.predict_classification(X_regime, model_name)
                
                # Calculate metrics
                results['regimes'][regime]['metrics']['accuracy'] = accuracy_score(y_regime, regime_preds)
                results['regimes'][regime]['metrics']['f1'] = f1_score(y_regime, regime_preds, average='weighted')
                results['regimes'][regime]['metrics']['precision'] = precision_score(y_regime, regime_preds, average='weighted')
                results['regimes'][regime]['metrics']['recall'] = recall_score(y_regime, regime_preds, average='weighted')
                
                # Try to get ROC AUC if binary classification
                try:
                    # Get probabilities
                    probs = self.predict_proba(X_regime, model_name)[:, 1]
                    results['regimes'][regime]['metrics']['roc_auc'] = roc_auc_score(y_regime, probs)
                except:
                    pass
            else:
                # Get predictions
                regime_preds = self.predict_regression(X_regime, model_name)
                
                # Calculate metrics
                results['regimes'][regime]['metrics']['mse'] = mean_squared_error(y_regime, regime_preds)
                results['regimes'][regime]['metrics']['mae'] = mean_absolute_error(y_regime, regime_preds)
                results['regimes'][regime]['metrics']['r2'] = r2_score(y_regime, regime_preds)
                
        # Add meta-model evaluation if available
        if meta_model_name in self.models:
            # Generate meta-features
            meta_features = self._generate_ensemble_features(
                X_test, 
                regime_column=regime_column,
                base_name=base_name,
                model_type=model_type
            )
            
            # Skip the regime column for prediction
            meta_features_pred = meta_features.drop(columns=['regime'])
            
            # Get predictions from meta-model
            if model_type == 'classification':
                meta_preds = self.predict_classification(meta_features_pred, meta_model_name)
                
                # Calculate metrics
                results['models'][meta_model_name] = {
                    'accuracy': accuracy_score(y_test, meta_preds),
                    'f1': f1_score(y_test, meta_preds, average='weighted'),
                    'precision': precision_score(y_test, meta_preds, average='weighted'),
                    'recall': recall_score(y_test, meta_preds, average='weighted')
                }
                
                # Try to get ROC AUC if binary classification
                try:
                    probs = self.predict_proba(meta_features_pred, meta_model_name)[:, 1]
                    results['models'][meta_model_name]['roc_auc'] = roc_auc_score(y_test, probs)
                except:
                    pass
                    
                # Compare with simple averaging (for classification)
                # Get columns with probabilities
                prob_cols = [col for col in meta_features.columns if 'prob_1' in col]
                
                if prob_cols:
                    # Average the probabilities
                    avg_probs = meta_features[prob_cols].mean(axis=1).values
                    # Convert to class labels using 0.5 threshold
                    avg_preds = (avg_probs > 0.5).astype(int)
                    
                    # Calculate metrics
                    results['models']['simple_average'] = {
                        'accuracy': accuracy_score(y_test, avg_preds),
                        'f1': f1_score(y_test, avg_preds, average='weighted'),
                        'precision': precision_score(y_test, avg_preds, average='weighted'),
                        'recall': recall_score(y_test, avg_preds, average='weighted')
                    }
                    
                    # Try to get ROC AUC
                    try:
                        results['models']['simple_average']['roc_auc'] = roc_auc_score(y_test, avg_probs)
                    except:
                        pass
            else:
                # Regression metrics
                meta_preds = self.predict_regression(meta_features_pred, meta_model_name)
                
                results['models'][meta_model_name] = {
                    'mse': mean_squared_error(y_test, meta_preds),
                    'mae': mean_absolute_error(y_test, meta_preds),
                    'r2': r2_score(y_test, meta_preds)
                }
                
        # Store evaluation results
        self.ensemble_evaluation = results
        
        return results 