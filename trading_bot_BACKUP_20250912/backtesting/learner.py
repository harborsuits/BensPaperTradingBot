#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine learning model manager for backtesting.
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
import joblib

logger = logging.getLogger(__name__)

class BacktestLearner:
    """
    Machine learning model manager for backtesting.
    
    This class handles training, evaluating, and managing machine learning models
    for backtesting trading strategies.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the BacktestLearner.
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Dictionary of available models
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
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_params: Dict[str, Any] = {},
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a machine learning model.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            model_params: Parameters for the model
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (trained model, training info)
        """
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
        
        training_info = {
            "model_type": model_type,
            "is_classification": is_classification,
            "n_samples": len(X_train),
            "model_params": model_params,
            "training_time": training_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return model, training_info
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            is_classification: Whether classification or regression
        
        Returns:
            Dictionary with evaluation metrics
        """
        if is_classification:
            return self._evaluate_classification(model, X, y)
        else:
            return self._evaluate_regression(model, X, y)
    
    def _evaluate_classification(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model.
        
        Args:
            model: Trained classification model
            X: Feature data
            y: Target data
        
        Returns:
            Dictionary with classification metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        
        # Get probability predictions if available
        try:
            y_prob = model.predict_proba(X)[:, 1]
            has_proba = True
        except (AttributeError, IndexError):
            y_prob = None
            has_proba = False
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # Add AUC if probabilities available and binary classification
        if has_proba and len(np.unique(y)) == 2:
            metrics["auc"] = roc_auc_score(y, y_prob)
        
        # Add confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Add classification report (as dict)
        report = classification_report(y, y_pred, output_dict=True)
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
    
    def cross_validate(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Dict[str, Any] = {},
        n_splits: int = 5,
        is_classification: bool = True
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
        
        Returns:
            Dictionary with cross-validation results
        """
        # Get model class
        model_class = self._get_model_class(model_type, is_classification)
        
        # Create CV splitter
        if is_classification:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
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
            "aggregated_metrics": aggregated
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
        # Create save path
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        info_path = os.path.join(self.models_dir, f"{model_name}_info.json")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save model info
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_path
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Tuple of (loaded model, model info)
        """
        # Create load paths
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        info_path = os.path.join(self.models_dir, f"{model_name}_info.json")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Model info file not found: {info_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load model info
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        return model, model_info
    
    def get_model_list(self) -> List[str]:
        """
        Get list of available saved models.
        
        Returns:
            List of model names
        """
        models = []
        
        for file in os.listdir(self.models_dir):
            if file.endswith(".pkl") and not file.endswith("_info.pkl"):
                model_name = file.replace(".pkl", "")
                models.append(model_name)
        
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model.
        
        Args:
            model_name: Name of the model to delete
        
        Returns:
            True if successful
        """
        # Create paths
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        info_path = os.path.join(self.models_dir, f"{model_name}_info.json")
        
        # Delete files if they exist
        if os.path.exists(model_path):
            os.remove(model_path)
        
        if os.path.exists(info_path):
            os.remove(info_path)
        
        return True
    
    def optimize_hyperparameters(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, List[Any]],
        search_method: str = 'grid',
        cv: int = 3,
        n_iter: int = 10,
        scoring: Optional[str] = None,
        is_classification: bool = True
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
        
        Returns:
            Dictionary with optimization results
        """
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
                verbose=1
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
                random_state=42
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
        
        # Create results
        results = {
            "model_type": model_type,
            "search_method": search_method,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "validation_metrics": metrics,
            "cv_results": search.cv_results_,
            "optimization_time": optimization_time
        }
        
        return results
    
    def feature_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
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
    
    def plot_feature_importance(
        self,
        feature_selection_result: Dict[str, Any],
        title: str = "Feature Importance",
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Plot feature importance from feature selection.
        
        Args:
            feature_selection_result: Result from feature selection
            title: Title for the plot
            top_n: Number of top features to plot
        
        Returns:
            Dictionary with plot data
        """
        # Get feature ranking
        method = feature_selection_result["method"]
        
        if method == "importance" or method == "univariate":
            feature_ranking = feature_selection_result["feature_ranking"]
            
            # Get top features
            top_features = feature_ranking[:top_n]
            
            # Get feature names and values
            features = [f["feature"] for f in top_features]
            values = [f.get("importance", f.get("score", 0)) for f in top_features]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Importance/Score')
            ax.set_title(title)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.models_dir, "feature_importance.png"))
            plt.close()
            
            return {
                "plot_type": "feature_importance",
                "features": features,
                "values": values,
                "plot_path": os.path.join(self.models_dir, "feature_importance.png")
            }
        
        else:
            # For RFE and model selection, plot selected vs not selected
            selected_features = feature_selection_result["selected_features"]
            all_features = [f["feature"] for f in feature_selection_result.get("feature_ranking", [])]
            
            # Mark selected
            is_selected = [1 if f in selected_features else 0 for f in all_features]
            
            # Truncate to top_n
            if len(all_features) > top_n:
                all_features = all_features[:top_n]
                is_selected = is_selected[:top_n]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(all_features))
            ax.barh(y_pos, is_selected, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(all_features)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Selected (1) / Not Selected (0)')
            ax.set_title(title)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.models_dir, "feature_selection.png"))
            plt.close()
            
            return {
                "plot_type": "feature_selection",
                "features": all_features,
                "selected": is_selected,
                "plot_path": os.path.join(self.models_dir, "feature_selection.png")
            }
    
    def generate_model_report(
        self,
        model: Any,
        model_info: Dict[str, Any],
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any] = None,
        feature_selection_result: Dict[str, Any] = None
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
            "model_name": model_info.get("model_type"),
            "model_params": model_info.get("model_params", {}),
            "is_classification": model_info.get("is_classification", True),
            "n_samples": model_info.get("n_samples"),
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
        
        # Save report
        report_path = os.path.join(self.models_dir, f"{model_info.get('model_type')}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report 