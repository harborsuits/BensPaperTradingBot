"""
ML Model Training Pipeline

This module provides functionality for training and optimizing ML models
for predicting market movements and generating trading signals.
"""

import numpy as np
import pandas as pd
import logging
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and optimizes ML models for market prediction"""
    
    def __init__(self, config=None):
        """
        Initialize the model trainer
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.models_dir = self.config.get('models_dir', 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set default hyperparameters
        self.default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (100,),
                'max_iter': 1000,
                'random_state': 42
            }
        }
        
        # Override defaults with config values if provided
        for model_type, params in self.config.get('model_params', {}).items():
            if model_type in self.default_params:
                self.default_params[model_type].update(params)
        
        logger.info("Model Trainer initialized")
    
    def prepare_training_data(self, df: pd.DataFrame, label_column: str, 
                             features: List[str] = None, 
                             test_size: float = 0.2) -> Tuple:
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features and labels
            label_column: Name of the column containing labels
            features: List of feature columns to use (if None, use all except label)
            test_size: Percentage of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Drop rows with NaN values
        df = df.dropna()
        
        # If features not specified, use all columns except the label
        if features is None:
            features = [col for col in df.columns if col != label_column]
        
        # Get feature and label data
        X = df[features]
        y = df[label_column]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, features
    
    def train_model(self, X_train, y_train, model_type: str = 'random_forest', 
                   custom_params: Dict = None) -> Any:
        """
        Train a model on the provided data
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to train
            custom_params: Optional custom parameters for the model
            
        Returns:
            Trained model
        """
        # Get model parameters
        params = custom_params or self.default_params.get(model_type, {})
        
        # Create model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(**params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**params)
        elif model_type == 'svm':
            model = SVC(**params)
        elif model_type == 'neural_network':
            model = MLPClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        logger.info(f"Training {model_type} model")
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Log evaluation results
        logger.info(f"Model evaluation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
    
    def optimize_hyperparameters(self, X_train, y_train, model_type: str = 'random_forest', 
                               param_grid: Dict = None) -> Tuple[Dict[str, Any], Any]:
        """
        Optimize model hyperparameters using grid search
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to optimize
            param_grid: Parameter grid for optimization
            
        Returns:
            Tuple of (best parameters, best model)
        """
        # Default parameter grids if none provided
        default_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        param_grid = param_grid or default_grids.get(model_type, {})
        
        # Create base model
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
        elif model_type == 'neural_network':
            model = MLPClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with scaler for applicable models
        if model_type in ['logistic_regression', 'svm', 'neural_network']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            # Adjust param grid keys to include pipeline step
            param_grid = {f'model__{k}': v for k, v in param_grid.items()}
        else:
            pipeline = model
        
        # Perform grid search
        logger.info(f"Performing hyperparameter optimization for {model_type}")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Extract model parameters from pipeline parameters if needed
        if model_type in ['logistic_regression', 'svm', 'neural_network']:
            best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        else:
            best_params = grid_search.best_params_
        
        return best_params, grid_search.best_estimator_
    
    def generate_labels(self, price_data: pd.DataFrame, 
                       label_type: str = 'directional', 
                       horizon: int = 5,
                       threshold: float = 0.0) -> pd.Series:
        """
        Generate labels for supervised learning from price data
        
        Args:
            price_data: DataFrame with price data
            label_type: Type of label to generate ('directional', 'volatility', 'regime')
            horizon: Number of periods to look ahead for label generation
            threshold: Threshold for determining labels (e.g., min return for 'up')
            
        Returns:
            Series with generated labels
        """
        if label_type == 'directional':
            # Generate directional labels (1 for up, 0 for down/sideways)
            future_returns = price_data['close'].shift(-horizon) / price_data['close'] - 1
            return (future_returns > threshold).astype(int)
            
        elif label_type == 'volatility':
            # Generate volatility regime labels
            future_vol = price_data['close'].pct_change().shift(-horizon).rolling(horizon).std() * np.sqrt(252)
            # 1 for high volatility, 0 for low volatility
            vol_threshold = future_vol.median()  # Dynamic threshold
            return (future_vol > vol_threshold).astype(int)
            
        elif label_type == 'regime':
            # Generate market regime labels
            # 0: downtrend, 1: sideways, 2: uptrend
            sma_fast = price_data['close'].rolling(window=20).mean()
            sma_slow = price_data['close'].rolling(window=50).mean()
            
            # Calculate trend direction and strength
            trend_direction = ((sma_fast - sma_slow) / sma_slow * 100)
            
            # Assign regime labels based on trend direction
            labels = pd.Series(1, index=price_data.index)  # Default to sideways
            labels[trend_direction > 1.0] = 2  # Uptrend
            labels[trend_direction < -1.0] = 0  # Downtrend
            
            return labels.shift(-horizon)  # Shift to align with future regime
            
        else:
            raise ValueError(f"Unknown label type: {label_type}")
    
    def save_model(self, model, model_name: str, metadata: Dict = None) -> str:
        """
        Save a trained model to disk
        
        Args:
            model: Trained model to save
            model_name: Name to save the model under
            metadata: Additional model metadata
            
        Returns:
            Path to the saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
        
        # Save the model with pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model
