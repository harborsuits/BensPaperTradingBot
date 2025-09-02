#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified BacktestLearner - Manages machine learning model training, evaluation, and optimization
for backtesting trading strategies.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Import scikit-learn components
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, RFE,
    SelectFromModel
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
)

# Import data manager
from trading_bot.data.data_manager import DataManager

logger = logging.getLogger(__name__)

class BacktestLearner:
    """
    Unified machine learning model manager for backtesting trading strategies.
    
    This class handles:
    - Training, evaluating, and optimizing machine learning models
    - Feature selection and importance analysis
    - Model persistence and management
    - Integration with backtesting workflows
    """
    
    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        models_dir: str = "models"
    ):
        """
        Initialize the BacktestLearner.
        
        Args:
            data_manager: DataManager instance for data operations
            models_dir: Directory for model storage
        """
        self.data_manager = data_manager if data_manager else DataManager(models_dir=models_dir)
        self.models_dir = Path(models_dir)
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store trained models
        self.models = {}
        
        # Dictionary to store model performance metrics
        self.metrics = {}
        
        # Dictionary of available model types
        self.available_models = {
            # Classification models
            'random_forest_clf': RandomForestClassifier,
            'gradient_boosting_clf': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svc': SVC,
            'mlp_clf': MLPClassifier,
            
            # Regression models
            'random_forest_reg': RandomForestRegressor,
            'gradient_boosting_reg': GradientBoostingRegressor,
            'linear_regression': LinearRegression,
            'svr': SVR,
            'mlp_reg': MLPRegressor,
        }
        
        logger.info(f"Initialized BacktestLearner with models_dir={models_dir}")
    
    def _get_model_class(self, model_type: str, is_classification: bool) -> Any:
        """
        Get model class based on type and task.
        
        Args:
            model_type: Type of model
            is_classification: Whether classification or regression
        
        Returns:
            Model class
        """
        if model_type not in self.available_models:
            raise ValueError(f"Model type not supported: {model_type}")
        
        # Check if model matches problem type (classification vs regression)
        model_name = model_type.lower()
        if is_classification and ('reg' in model_name):
            logger.warning(f"Requested regression model for classification task: {model_type}")
        elif not is_classification and ('clf' in model_name or 'logistic' in model_name):
            logger.warning(f"Requested classification model for regression task: {model_type}")
        
        return self.available_models[model_type]
    
    def train_model(
        self,
        model_type: str,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        model_params: Dict[str, Any] = {},
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        save_model: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a machine learning model.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            model_params: Parameters for the model
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (optional)
            model_name: Name for saving the model (optional)
            save_model: Whether to save the model
        
        Returns:
            Tuple of (trained model, training info)
        """
        # Convert pandas objects to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Determine if classification or regression
        unique_values = np.unique(y_train)
        is_classification = len(unique_values) < 10 or np.issubdtype(y_train.dtype, np.integer)
        
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Create and train model
        logger.info(f"Training {model_type} model with {len(X_train)} samples")
        start_time = datetime.now()
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"
        
        # Calculate training metrics
        train_metrics = self.evaluate_model(model, X_train, y_train, is_classification)
        
        # Calculate validation metrics if validation data provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate_model(model, X_val, y_val, is_classification)
        
        # Create training info
        training_info = {
            "model_name": model_name,
            "model_type": model_type,
            "is_classification": is_classification,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "feature_names": feature_names,
            "model_params": model_params,
            "training_time": training_time,
            "timestamp": datetime.now().isoformat(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        
        # Save model if requested
        if save_model:
            self.save_model(model, training_info, model_name)
        
        # Store model and metrics
        self.models[model_name] = model
        self.metrics[model_name] = {
            "train": train_metrics,
            "val": val_metrics
        }
        
        return model, training_info
    
    def evaluate_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        is_classification: bool = True,
        prediction_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            is_classification: Whether classification or regression
            prediction_threshold: Threshold for binary classification
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert pandas objects to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if is_classification:
            return self._evaluate_classification(model, X, y, prediction_threshold)
        else:
            return self._evaluate_regression(model, X, y)
    
    def _evaluate_classification(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        prediction_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model.
        
        Args:
            model: Trained classification model
            X: Feature data
            y: Target data
            prediction_threshold: Threshold for binary classification
        
        Returns:
            Dictionary with classification metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        
        # Get probability predictions if available
        try:
            y_prob = model.predict_proba(X)[:, 1]
            has_proba = True
            
            # Apply threshold for binary classification
            if len(np.unique(y)) == 2:
                y_pred_threshold = (y_prob >= prediction_threshold).astype(int)
            else:
                y_pred_threshold = y_pred
                
        except (AttributeError, IndexError):
            y_prob = None
            has_proba = False
            y_pred_threshold = y_pred
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred_threshold),
            "precision": precision_score(y, y_pred_threshold, average='weighted', zero_division=0),
            "recall": recall_score(y, y_pred_threshold, average='weighted', zero_division=0),
            "f1": f1_score(y, y_pred_threshold, average='weighted', zero_division=0),
        }
        
        # Add AUC if probabilities available and binary classification
        if has_proba and len(np.unique(y)) == 2:
            metrics["auc"] = roc_auc_score(y, y_prob)
        
        # Add confusion matrix
        cm = confusion_matrix(y, y_pred_threshold)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Add classification report (as dict)
        report = classification_report(y, y_pred_threshold, output_dict=True)
        metrics["classification_report"] = report
        
        return metrics
    
    def _evaluate_regression(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a regression model.
        
        Args:
            model: Trained regression model
            X: Feature data
            y: Target data
        
        Returns:
            Dictionary with regression metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }
        
        return metrics
    
    def generate_predictions(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        is_classification: bool = True,
        prediction_threshold: float = 0.5,
        get_probabilities: bool = False
    ) -> np.ndarray:
        """
        Generate predictions from a trained model.
        
        Args:
            model: Trained model
            X: Feature data
            is_classification: Whether classification or regression
            prediction_threshold: Threshold for binary classification
            get_probabilities: Whether to return probabilities (classification only)
        
        Returns:
            Numpy array with predictions
        """
        # Convert pandas objects to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if is_classification:
            if get_probabilities:
                try:
                    # Get probability predictions
                    y_prob = model.predict_proba(X)
                    
                    # For binary classification, return probability of positive class
                    if y_prob.shape[1] == 2:
                        return y_prob[:, 1]
                    else:
                        return y_prob
                except (AttributeError, IndexError):
                    # Fall back to regular predictions if probabilities not available
                    logger.warning("Probabilities not available for this model, returning class predictions")
                    return model.predict(X)
            else:
                # Get class predictions
                try:
                    # Try to get probabilities and apply threshold
                    y_prob = model.predict_proba(X)[:, 1]
                    return (y_prob >= prediction_threshold).astype(int)
                except (AttributeError, IndexError):
                    # Fall back to regular predictions
                    return model.predict(X)
        else:
            # Regression prediction
            return model.predict(X)
    
    def cross_validate(
        self,
        model_type: str,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_params: Dict[str, Any] = {},
        n_splits: int = 5,
        is_classification: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_type: Type of model to train
            X: Feature data
            y: Target data
            model_params: Parameters for the model
            n_splits: Number of cross-validation splits
            is_classification: Whether classification or regression
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary with cross-validation results
        """
        # Convert pandas objects to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = None
            
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Create CV splitter
        if is_classification:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Initialize metrics storage
        all_metrics = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train_cv, y_train_cv)
            
            # Evaluate model
            if is_classification:
                metrics = self._evaluate_classification(model, X_val_cv, y_val_cv)
            else:
                metrics = self._evaluate_regression(model, X_val_cv, y_val_cv)
            
            metrics["fold"] = fold
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        if is_classification:
            metrics_to_agg = ["accuracy", "precision", "recall", "f1"]
            if "auc" in all_metrics[0]:
                metrics_to_agg.append("auc")
        else:
            metrics_to_agg = ["mse", "rmse", "mae", "r2"]
        
        for metric in metrics_to_agg:
            values = [m[metric] for m in all_metrics if metric in m]
            aggregated[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values
            }
        
        return {
            "model_type": model_type,
            "model_params": model_params,
            "n_splits": n_splits,
            "metrics_by_fold": all_metrics,
            "aggregated_metrics": aggregated,
            "feature_names": feature_names
        }
    
    def optimize_hyperparameters(
        self,
        model_type: str,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, List[Any]],
        search_method: str = 'grid',
        cv: int = 3,
        n_iter: int = 10,
        scoring: Optional[str] = None,
        is_classification: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Grid of parameters to search
            search_method: Method for searching
                - 'grid': Grid search
                - 'random': Random search
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            scoring: Scoring metric for optimization
            is_classification: Whether classification or regression
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary with optimization results
        """
        # Convert pandas objects to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            feature_names = None
            
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
            
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Set default scoring if not provided
        if scoring is None:
            scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        
        # Create base model
        base_model = model_class()
        
        # Create search object
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
                random_state=random_state
            )
        elif search_method == 'random':
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported search method: {search_method}")
        
        # Perform search
        logger.info(f"Optimizing hyperparameters for {model_type} using {search_method} search")
        start_time = datetime.now()
        
        search.fit(X_train, y_train)
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        # Train model with best parameters
        best_model = model_class(**search.best_params_)
        best_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        if is_classification:
            metrics = self._evaluate_classification(best_model, X_val, y_val)
        else:
            metrics = self._evaluate_regression(best_model, X_val, y_val)
        
        # Generate model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_optimized_{timestamp}"
        
        # Create training info
        training_info = {
            "model_name": model_name,
            "model_type": model_type,
            "is_classification": is_classification,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "feature_names": feature_names,
            "model_params": search.best_params_,
            "training_time": optimization_time,
            "timestamp": datetime.now().isoformat(),
            "train_metrics": self._evaluate_classification(best_model, X_train, y_train) if is_classification else self._evaluate_regression(best_model, X_train, y_train),
            "val_metrics": metrics,
            "optimization": {
                "search_method": search_method,
                "param_grid": param_grid,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "cv_results": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in search.cv_results_.items()},
                "optimization_time": optimization_time
            }
        }
        
        # Store model and metrics
        self.models[model_name] = best_model
        self.metrics[model_name] = {
            "train": training_info["train_metrics"],
            "val": metrics
        }
        
        # Save model
        self.save_model(best_model, training_info, model_name)
        
        return {
            "model": best_model,
            "model_name": model_name,
            "best_params": search.best_params_,
            "metrics": metrics,
            "training_info": training_info
        }
    
    def feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        top_n: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get feature importance for a model.
        
        Args:
            model: Trained model
            feature_names: Names of features
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importances
        """
        try:
            # Try to get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                raise AttributeError("Model does not have feature_importances_ or coef_ attributes")
            
            # Create feature importance dictionary
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Create dictionary of feature importances
            importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(importances))}
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Limit to top_n if specified
            if top_n is not None and top_n < len(importance_dict):
                importance_dict = dict(list(importance_dict.items())[:top_n])
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def feature_selection(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        feature_names: List[str],
        method: str = 'importance',
        n_features: int = 10,
        model_type: Optional[str] = None,
        model_params: Dict[str, Any] = {},
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Perform feature selection.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            method: Method for feature selection
                - 'importance': Feature importance
                - 'univariate': Univariate selection
                - 'rfe': Recursive feature elimination
                - 'model': Model-based selection
            n_features: Number of features to select
            model_type: Type of model (for 'importance', 'rfe', 'model')
            model_params: Parameters for the model
            is_classification: Whether classification or regression
        
        Returns:
            Dictionary with feature selection results
        """
        # Convert pandas objects to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None or len(feature_names) == 0:
                feature_names = X_train.columns.tolist()
            X_train = X_train.values
            
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        if method == 'importance':
            return self._feature_selection_importance(
                X_train, y_train, feature_names, n_features,
                model_type, model_params, is_classification
            )
        elif method == 'univariate':
            return self._feature_selection_univariate(
                X_train, y_train, feature_names, n_features, is_classification
            )
        elif method == 'rfe':
            return self._feature_selection_rfe(
                X_train, y_train, feature_names, n_features,
                model_type, model_params, is_classification
            )
        elif method == 'model':
            return self._feature_selection_model(
                X_train, y_train, feature_names, n_features,
                model_type, model_params, is_classification
            )
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
    
    def _feature_selection_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features: int,
        model_type: Optional[str],
        model_params: Dict[str, Any],
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Feature selection using feature importance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            n_features: Number of features to select
            model_type: Type of model
            model_params: Parameters for the model
            is_classification: Whether classification or regression
        
        Returns:
            Dictionary with feature selection results
        """
        # Default to Random Forest if model type not specified
        if model_type is None:
            model_type = 'random_forest_clf' if is_classification else 'random_forest_reg'
        
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Get feature importances
        try:
            importances = model.feature_importances_
        except AttributeError:
            raise ValueError(f"Model {model_type} does not have feature_importances_ attribute")
        
        # Rank features by importance
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:n_features]
        
        # Create feature ranking
        ranking = []
        for i, idx in enumerate(indices):
            ranking.append({
                "rank": i + 1,
                "feature": feature_names[idx],
                "importance": float(importances[idx])
            })
        
        # Get selected features
        selected_indices = top_indices
        selected_features = [feature_names[i] for i in selected_indices]
        
        return {
            "method": "importance",
            "model_type": model_type,
            "n_features": n_features,
            "feature_ranking": ranking,
            "selected_indices": selected_indices.tolist(),
            "selected_features": selected_features
        }
    
    def _feature_selection_univariate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features: int,
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Feature selection using univariate selection.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            n_features: Number of features to select
            is_classification: Whether classification or regression
        
        Returns:
            Dictionary with feature selection results
        """
        # Choose scoring function
        score_func = f_classif if is_classification else f_regression
        
        # Create selector
        selector = SelectKBest(score_func=score_func, k=n_features)
        
        # Fit selector
        selector.fit(X_train, y_train)
        
        # Get scores
        scores = selector.scores_
        
        # Rank features by score
        indices = np.argsort(scores)[::-1]
        top_indices = indices[:n_features]
        
        # Create feature ranking
        ranking = []
        for i, idx in enumerate(indices):
            ranking.append({
                "rank": i + 1,
                "feature": feature_names[idx],
                "score": float(scores[idx])
            })
        
        # Get selected features
        selected_indices = top_indices
        selected_features = [feature_names[i] for i in selected_indices]
        
        return {
            "method": "univariate",
            "score_func": "f_classif" if is_classification else "f_regression",
            "n_features": n_features,
            "feature_ranking": ranking,
            "selected_indices": selected_indices.tolist(),
            "selected_features": selected_features
        }
    
    def _feature_selection_rfe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features: int,
        model_type: Optional[str],
        model_params: Dict[str, Any],
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Feature selection using recursive feature elimination.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            n_features: Number of features to select
            model_type: Type of model
            model_params: Parameters for the model
            is_classification: Whether classification or regression
        
        Returns:
            Dictionary with feature selection results
        """
        # Default to Random Forest if model type not specified
        if model_type is None:
            model_type = 'random_forest_clf' if is_classification else 'random_forest_reg'
        
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Create model
        model = model_class(**model_params)
        
        # Create RFE
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
        
        # Fit RFE
        rfe.fit(X_train, y_train)
        
        # Get rankings
        rankings = rfe.ranking_
        
        # Create feature ranking
        ranking = []
        for i, rank in enumerate(rankings):
            ranking.append({
                "feature": feature_names[i],
                "rank": int(rank)
            })
        
        # Sort by rank
        ranking.sort(key=lambda x: x["rank"])
        
        # Get selected features
        selected_indices = np.where(rfe.support_)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        return {
            "method": "rfe",
            "model_type": model_type,
            "n_features": n_features,
            "feature_ranking": ranking,
            "selected_indices": selected_indices.tolist(),
            "selected_features": selected_features
        }
    
    def _feature_selection_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features: int,
        model_type: Optional[str],
        model_params: Dict[str, Any],
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Feature selection using model-based selection.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            n_features: Number of features to select
            model_type: Type of model
            model_params: Parameters for the model
            is_classification: Whether classification or regression
        
        Returns:
            Dictionary with feature selection results
        """
        # Default to Random Forest if model type not specified
        if model_type is None:
            model_type = 'random_forest_clf' if is_classification else 'random_forest_reg'
        
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Create model
        model = model_class(**model_params)
        
        # Create selector
        selector = SelectFromModel(
            estimator=model,
            max_features=n_features,
            threshold=-np.inf  # Force max_features
        )
        
        # Fit selector
        selector.fit(X_train, y_train)
        
        # Get selected features
        selected_indices = np.where(selector.get_support())[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Train model on original data to get feature importances
        model.fit(X_train, y_train)
        
        # Get feature importances
        try:
            importances = model.feature_importances_
        except AttributeError:
            importances = np.zeros(len(feature_names))
        
        # Create feature ranking
        ranking = []
        for i, importance in enumerate(importances):
            ranking.append({
                "feature": feature_names[i],
                "importance": float(importance),
                "selected": i in selected_indices
            })
        
        # Sort by importance
        ranking.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "method": "model",
            "model_type": model_type,
            "n_features": n_features,
            "feature_ranking": ranking,
            "selected_indices": selected_indices.tolist(),
            "selected_features": selected_features
        }
    
    def save_model(
        self,
        model: Any,
        model_info: Dict[str, Any],
        model_name: str
    ) -> str:
        """
        Save a trained model.
        
        Args:
            model: Trained model
            model_info: Information about the model
            model_name: Name for the saved model
        
        Returns:
            Path to the saved model
        """
        return self.data_manager.save_model(model_name, model, model_info)
    
    def load_model(
        self,
        model_name: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Tuple of (loaded model, model info)
        """
        model, model_info = self.data_manager.load_model(model_name)
        
        if model is not None:
            self.models[model_name] = model
            
            # Load metrics if available
            if "train_metrics" in model_info:
                self.metrics[model_name] = {
                    "train": model_info["train_metrics"],
                    "val": model_info.get("val_metrics")
                }
        
        return model, model_info
    
    def list_models(self) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List of model names
        """
        return self.data_manager.get_models()
    
    def delete_model(
        self,
        model_name: str
    ) -> bool:
        """
        Delete a saved model.
        
        Args:
            model_name: Name of the model to delete
        
        Returns:
            True if successful
        """
        # Remove from in-memory cache
        if model_name in self.models:
            del self.models[model_name]
        
        if model_name in self.metrics:
            del self.metrics[model_name]
        
        return self.data_manager.delete_model(model_name)
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        title: str = "Feature Importance",
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            feature_names: Names of features
            title: Title for the plot
            top_n: Number of top features to plot
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get feature importance
        importance_dict = self.feature_importance(model, feature_names)
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Get top_n features
        if top_n is not None and top_n < len(importance_dict):
            importance_dict = dict(list(importance_dict.items())[:top_n])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_model_report(
        self,
        model: Any,
        model_info: Dict[str, Any],
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        feature_selection_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive model report.
        
        Args:
            model: Trained model
            model_info: Information about the model
            train_metrics: Metrics on training data
            val_metrics: Metrics on validation data
            test_metrics: Metrics on test data (if available)
            feature_selection_result: Result from feature selection (if available)
        
        Returns:
            Dictionary with model report
        """
        report = {
            "model_name": model_info.get("model_name"),
            "model_type": model_info.get("model_type"),
            "is_classification": model_info.get("is_classification", True),
            "n_samples": model_info.get("n_samples"),
            "n_features": model_info.get("n_features"),
            "feature_names": model_info.get("feature_names"),
            "model_params": model_info.get("model_params", {}),
            "training_time": model_info.get("training_time"),
            "timestamp": model_info.get("timestamp"),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        
        if test_metrics:
            report["test_metrics"] = test_metrics
        
        if feature_selection_result:
            report["feature_selection"] = {
                "method": feature_selection_result.get("method"),
                "selected_features": feature_selection_result.get("selected_features"),
                "n_features": feature_selection_result.get("n_features")
            }
        
        # Generate feature importance if possible
        try:
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                feature_names = model_info.get("feature_names", [])
                if feature_names:
                    report["feature_importance"] = self.feature_importance(model, feature_names)
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {str(e)}")
        
        # Save report
        model_name = model_info.get("model_name", "unknown")
        report_path = os.path.join(self.models_dir, f"{model_name}_report.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model report: {str(e)}")
        
        return report 