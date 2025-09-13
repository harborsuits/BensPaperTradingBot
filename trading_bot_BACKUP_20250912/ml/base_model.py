#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for machine learning models in the trading system.
"""

import abc
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import pandas as pd

from trading_bot.data.models import MarketData, TimeFrame
from trading_bot.common.config_utils import setup_directories

logger = logging.getLogger(__name__)


class BaseMLModel(abc.ABC):
    """
    Abstract base class for all machine learning models in the trading system.
    
    This class defines the interface that all ML models must implement.
    Models are responsible for making predictions based on market data and
    can be used standalone or as part of an ensemble.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML model.
        
        Args:
            name: Unique identifier for this model instance
            config: Model configuration
        """
        self.name = name
        self.config = config or {}
        self.is_trained = False
        self.last_training_time = None
        
        # Setup paths for model storage
        self.paths = setup_directories(data_dir=self.config.get("data_dir"), 
                                      component_name=f"ml_models/{name}")
        
    @abc.abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before training or prediction.
        
        Args:
            data: Input data as DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        pass
    
    @abc.abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        """
        Train the model on the given data.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
        """
        pass
    
    @abc.abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the model.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        pass
    
    @abc.abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            filepath: Optional path to save the model
            
        Returns:
            Path where the model was saved
        """
        if not filepath:
            # Use default path
            os.makedirs(self.paths["models_dir"], exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.paths["models_dir"], f"{self.name}_{timestamp}.pkl")
            
        # Create metadata
        metadata = {
            "model_name": self.name,
            "saved_at": datetime.now().isoformat(),
            "is_trained": self.is_trained,
            "last_training_time": self.last_training_time,
            "config": self.config
        }
        
        # Save the model and metadata
        with open(filepath, 'wb') as f:
            pickle.dump({"model": self, "metadata": metadata}, f)
            
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseMLModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        model = data["model"]
        logger.info(f"Model {model.name} loaded from {filepath}")
        return model
    
    def validate(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, float]:
        """
        Validate the model on test data.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            
        Returns:
            Dictionary of validation metrics
        """
        if not self.is_trained:
            logger.warning("Model not trained, cannot validate")
            return {"error": "Model not trained"}
            
        # Make predictions
        predictions = self.predict(features)
        
        # Calculate basic metrics
        metrics = {}
        try:
            # For regression
            metrics["mse"] = np.mean((predictions - targets.values.ravel()) ** 2)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = np.mean(np.abs(predictions - targets.values.ravel()))
            
            # Calculate directional accuracy for trading
            pred_direction = np.sign(predictions)
            true_direction = np.sign(targets.values.ravel())
            metrics["directional_accuracy"] = np.mean(pred_direction == true_direction)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics["error"] = str(e)
            
        return metrics
    
    def explain_prediction(self, features: pd.DataFrame, index: int = 0) -> Dict[str, float]:
        """
        Explain a single prediction using feature importance.
        
        This is a basic implementation. Subclasses can override with 
        more sophisticated methods like SHAP or LIME.
        
        Args:
            features: Feature DataFrame
            index: Index of the prediction to explain
            
        Returns:
            Dictionary mapping features to their contribution
        """
        # Basic implementation - multiply feature values by importance
        if not self.is_trained:
            logger.warning("Model not trained, cannot explain prediction")
            return {}
            
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Get feature values for the specified index
        feature_values = features.iloc[index].to_dict()
        
        # Calculate contribution
        contribution = {}
        for feature in importance:
            if feature in feature_values:
                contribution[feature] = importance[feature] * feature_values[feature]
                
        return contribution 