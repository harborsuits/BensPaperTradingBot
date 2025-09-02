"""
Machine Learning Market Regime Classifier

This module implements machine learning models for classifying market regimes
based on technical and statistical features. It supports training, validation,
prediction, and model persistence.
"""

import os
import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime

# Machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from trading_bot.analytics.market_regime.detector import MarketRegimeType

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegimeMLClassifier:
    """
    Machine learning classifier for market regimes.
    
    This class implements training, prediction, and evaluation of ML models
    for market regime classification. It supports multiple algorithms and
    feature selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML classifier.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - model_type: Algorithm to use (rf, gb, svm, knn, mlp)
                - feature_columns: List of feature columns to use
                - model_dir: Directory to save/load models
                - scaler: Whether to scale features
                - test_size: Fraction of data to use for testing
                - random_state: Random seed for reproducibility
        """
        self.config = config or {}
        
        # Set default configuration
        self.model_type = self.config.get('model_type', 'rf')
        self.feature_columns = self.config.get('feature_columns', [
            'trend_strength', 'volatility', 'momentum', 'trading_range',
            'rsi', 'macd', 'bollinger_width', 'atr_percent', 'volume_change'
        ])
        self.model_dir = self.config.get('model_dir', 'data/market_regime/models')
        self.random_state = self.config.get('random_state', 42)
        self.test_size = self.config.get('test_size', 0.2)
        self.use_scaler = self.config.get('scaler', True)
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler() if self.use_scaler else None
        self.feature_importances = {}
        
        # Class labels mapping
        self.regime_mapping = {
            regime.value: i for i, regime in enumerate(MarketRegimeType)
        }
        self.reverse_mapping = {
            i: regime for regime, i in self.regime_mapping.items()
        }
        
        # Performance metrics
        self.metrics = {}
        
        logger.debug(f"Initialized ML classifier with model_type={self.model_type}")
    
    def _create_model(self) -> BaseEstimator:
        """
        Create a new model instance based on the configured model_type.
        
        Returns:
            A scikit-learn estimator
        """
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=self.random_state
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=300,
                random_state=self.random_state
            )
        else:
            logger.warning(f"Unknown model type {self.model_type}, defaulting to RandomForest")
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
    
    def train(self, data: pd.DataFrame, target_column: str = 'regime') -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            data: DataFrame with features and target column
            target_column: Name of the target column (default: 'regime')
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training model with {len(data)} samples")
        
        # Validate data
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Ensure all feature columns are present
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}")
            # Filter to available features
            self.feature_columns = [col for col in self.feature_columns if col in data.columns]
            
        if not self.feature_columns:
            raise ValueError("No valid feature columns found in data")
        
        # Map regime strings to integer labels
        data = data.copy()
        data['target'] = data[target_column].apply(lambda x: self.regime_mapping.get(x, 0))
        
        # Extract features and target
        X = data[self.feature_columns].values
        y = data['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Create and train model
        logger.info(f"Creating {self.model_type} model")
        model = self._create_model()
        
        if self.use_scaler:
            # Fit scaler on training data only
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        logger.info("Fitting model")
        model.fit(X_train, y_train)
        self.model = model
        
        # Save model
        self._save_model()
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_size': len(X_test),
            'train_size': len(X_train),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate feature importances if available
        self._calculate_feature_importances()
        
        # Save metrics
        self._save_metrics()
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        return self.metrics
    
    def _calculate_feature_importances(self) -> Dict[str, float]:
        """
        Calculate feature importances if the model supports it.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            return {}
        
        # Different models expose feature importances differently
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            logger.debug(f"Model {self.model_type} does not support feature importances")
            return {}
        
        # Map importances to feature names
        self.feature_importances = {
            feature: float(importance)
            for feature, importance in zip(self.feature_columns, importances)
        }
        
        # Sort by importance
        self.feature_importances = dict(
            sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        )
        
        return self.feature_importances
    
    def predict(self, features: Dict[str, float]) -> Tuple[MarketRegimeType, float]:
        """
        Predict the market regime for the given features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (MarketRegimeType, confidence)
        """
        if self.model is None:
            logger.warning("Model not trained, loading from disk")
            if not self._load_model():
                raise ValueError("No trained model available")
        
        # Extract features in the correct order
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        
        # Scale if needed
        if self.use_scaler and self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get prediction and probabilities
        regime_idx = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        # Map back to regime type
        regime_value = self.reverse_mapping.get(regime_idx, MarketRegimeType.UNKNOWN.value)
        regime = MarketRegimeType(regime_value)
        
        return regime, confidence
    
    def hyperparam_search(self, data: pd.DataFrame, target_column: str = 'regime') -> Dict[str, Any]:
        """
        Perform hyperparameter search for the current model type.
        
        Args:
            data: DataFrame with features and target column
            target_column: Name of the target column
            
        Returns:
            Best parameters and CV results
        """
        logger.info(f"Performing hyperparameter search for {self.model_type}")
        
        # Prepare data
        data = data.copy()
        data['target'] = data[target_column].apply(lambda x: self.regime_mapping.get(x, 0))
        
        X = data[self.feature_columns].values
        y = data['target'].values
        
        # Define parameter grid based on model type
        if self.model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'gb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1]
            }
            model = SVC(probability=True, random_state=self.random_state)
        elif self.model_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # Manhattan or Euclidean
            }
            model = KNeighborsClassifier()
        elif self.model_type == 'mlp':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
            model = MLPClassifier(random_state=self.random_state, max_iter=300)
        else:
            logger.warning(f"Unknown model type {self.model_type}, using RandomForest defaults")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
            model = RandomForestClassifier(random_state=self.random_state)
        
        # Create pipeline with scaling
        if self.use_scaler:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            # Prefix params with 'model__'
            param_grid = {f'model__{k}': v for k, v in param_grid.items()}
        else:
            pipeline = model
        
        # Create grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Get best model
        if self.use_scaler:
            self.model = grid_search.best_estimator_.named_steps['model']
            self.scaler = grid_search.best_estimator_.named_steps['scaler']
        else:
            self.model = grid_search.best_estimator_
        
        # Save model
        self._save_model()
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        # Save metrics and best params
        self.metrics['hyperparameter_search'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }
        self._save_metrics()
        
        logger.info(f"Hyperparameter search complete. Best score: {grid_search.best_score_:.4f}")
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def _save_model(self) -> str:
        """
        Save the model to disk.
        
        Returns:
            Path to the saved model file
        """
        if self.model is None:
            logger.warning("No model to save")
            return ""
        
        # Create model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_type}_model_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            if self.use_scaler and self.scaler is not None:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'regime_mapping': self.regime_mapping,
                    'config': self.config
                }, f)
            else:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'regime_mapping': self.regime_mapping,
                    'config': self.config
                }, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Create symlink to latest model
        latest_path = os.path.join(self.model_dir, f"{self.model_type}_model_latest.pkl")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_filename, latest_path)
        
        # Save feature importances
        if self.feature_importances:
            importances_path = os.path.join(self.model_dir, f"{self.model_type}_feature_importances.json")
            with open(importances_path, 'w') as f:
                json.dump(self.feature_importances, f, indent=2)
        
        return model_path
    
    def _save_metrics(self) -> str:
        """
        Save metrics to disk.
        
        Returns:
            Path to the saved metrics file
        """
        if not self.metrics:
            logger.warning("No metrics to save")
            return ""
        
        # Create metrics filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_filename = f"{self.model_type}_metrics_{timestamp}.json"
        metrics_path = os.path.join(self.model_dir, metrics_filename)
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Create symlink to latest metrics
        latest_path = os.path.join(self.model_dir, f"{self.model_type}_metrics_latest.json")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(metrics_filename, latest_path)
        
        return metrics_path
    
    def _load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the model file, or None to load the latest
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        if model_path is None:
            # Try to load latest model
            model_path = os.path.join(self.model_dir, f"{self.model_type}_model_latest.pkl")
            if not os.path.exists(model_path) or os.path.islink(model_path) and not os.path.exists(os.readlink(model_path)):
                # Try to find any model of this type
                models = [f for f in os.listdir(self.model_dir) if f.startswith(f"{self.model_type}_model_") and f.endswith(".pkl")]
                if not models:
                    logger.warning(f"No {self.model_type} model found in {self.model_dir}")
                    return False
                
                # Sort by timestamp (newest first)
                models.sort(reverse=True)
                model_path = os.path.join(self.model_dir, models[0])
        
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Load model
            self.model = saved_data['model']
            
            # Load scaler if available
            if 'scaler' in saved_data and self.use_scaler:
                self.scaler = saved_data['scaler']
            
            # Load feature columns
            if 'feature_columns' in saved_data:
                self.feature_columns = saved_data['feature_columns']
            
            # Load regime mapping
            if 'regime_mapping' in saved_data:
                self.regime_mapping = saved_data['regime_mapping']
                self.reverse_mapping = {
                    i: regime for regime, i in self.regime_mapping.items()
                }
            
            # Load config
            if 'config' in saved_data:
                # Update with saved config, but keep current model_dir
                model_dir = self.config.get('model_dir')
                self.config.update(saved_data['config'])
                if model_dir:
                    self.config['model_dir'] = model_dir
            
            logger.info(f"Model loaded from {model_path}")
            
            # Try to load metrics
            metrics_path = os.path.join(self.model_dir, f"{self.model_type}_metrics_latest.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        self.metrics = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading metrics: {str(e)}")
            
            # Try to load feature importances
            self._calculate_feature_importances()
            
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def evaluate(self, data: pd.DataFrame, target_column: str = 'regime') -> Dict[str, Any]:
        """
        Evaluate the model on new data.
        
        Args:
            data: DataFrame with features and target column
            target_column: Name of the target column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.warning("Model not trained, loading from disk")
            if not self._load_model():
                raise ValueError("No trained model available")
        
        logger.info(f"Evaluating model on {len(data)} samples")
        
        # Prepare data
        data = data.copy()
        data['target'] = data[target_column].apply(lambda x: self.regime_mapping.get(x, 0))
        
        # Extract features and target
        X = data[self.feature_columns].values
        y = data['target'].values
        
        # Scale if needed
        if self.use_scaler and self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Create evaluation metrics
        eval_metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'samples': len(data),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation accuracy: {accuracy:.4f}")
        return eval_metrics
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.feature_importances and self.model is not None:
            self._calculate_feature_importances()
        
        return self.feature_importances


def load_ml_classifier(model_dir: str = 'data/market_regime/models', 
                      model_type: str = 'rf') -> MarketRegimeMLClassifier:
    """
    Helper function to load a trained ML classifier.
    
    Args:
        model_dir: Directory containing model files
        model_type: Type of model to load
        
    Returns:
        Loaded MarketRegimeMLClassifier or None if not found
    """
    config = {
        'model_dir': model_dir,
        'model_type': model_type
    }
    
    classifier = MarketRegimeMLClassifier(config)
    if classifier._load_model():
        return classifier
    
    logger.warning(f"No trained model found for type {model_type} in {model_dir}")
    return classifier  # Return the initialized but untrained classifier
