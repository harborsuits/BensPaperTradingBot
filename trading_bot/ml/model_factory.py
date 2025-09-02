#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Model Factory - Creates and manages different ML model types.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Type

from trading_bot.ml.base_model import BaseMLModel
# Import model implementations
from trading_bot.ml.linear_model import LinearRegressionModel
from trading_bot.ml.random_forest_model import RandomForestModel
from trading_bot.ml.xgboost_model import XGBoostModel
from trading_bot.ml.lstm_model import LSTMModel

# Handle TensorFlow import gracefully
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    logging.warning("TensorFlow not available. Some models will be disabled.")
    HAS_TENSORFLOW = False
    tf = None

logger = logging.getLogger(__name__)

class MLModelFactory:
    """
    Factory class for creating and managing different ML model types.
    Handles model creation, configuration, training, and evaluation.
    """
    
    # Map of model types to their implementation classes
    MODEL_TYPES = {
        "linear": LinearRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lstm": LSTMModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, name: str, **kwargs) -> BaseMLModel:
        """
        Create a new ML model instance based on the specified type.
        
        Args:
            model_type: Type of model to create (e.g., "linear", "random_forest", "lstm")
            name: Unique identifier for the model
            **kwargs: Additional parameters for the model constructor
            
        Returns:
            Instance of the requested model type
        
        Raises:
            ValueError: If the specified model type is not supported
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(cls.MODEL_TYPES.keys())}")
        
        # Special case for TensorFlow-dependent models
        if model_type == "lstm" and not HAS_TENSORFLOW:
            logger.warning(f"Creating a dummy LSTM model because TensorFlow is not available")
            
        model_class = cls.MODEL_TYPES[model_type]
        logger.info(f"Creating new {model_type} model: {name}")
        return model_class(name=name, **kwargs)
    
    @staticmethod
    def list_available_models() -> List[str]:
        """
        List all available model types.
        
        Returns:
            List of available model type names
        """
        models = list(MLModelFactory.MODEL_TYPES.keys())
        if not HAS_TENSORFLOW:
            # Mark TensorFlow-dependent models
            models = [m if m != "lstm" else f"{m} (unavailable - TensorFlow required)" 
                     for m in models]
        return models
    
    @staticmethod
    def get_model_class(model_type: str) -> Type[BaseMLModel]:
        """
        Get the model class for a particular model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model class
            
        Raises:
            ValueError: If the specified model type is not supported
        """
        if model_type not in MLModelFactory.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")
        return MLModelFactory.MODEL_TYPES[model_type]
    
    @staticmethod
    def load_model(model_path: str) -> BaseMLModel:
        """
        Load a serialized model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is corrupted or incompatible
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine model type from filename or metadata
        # This is a simple implementation; a real factory would need more robust detection
        model_type = None
        for model_name in MLModelFactory.MODEL_TYPES.keys():
            if model_name in model_path.lower():
                model_type = model_name
                break
        
        if model_type is None:
            # Default fallback for unknown types - attempt to use BaseMLModel's load
            return BaseMLModel.load(model_path)
        
        # Use the appropriate model class's load method
        model_class = MLModelFactory.get_model_class(model_type)
        logger.info(f"Loading {model_type} model from {model_path}")
        return model_class.load(model_path) 