#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost model implementation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    logging.warning("XGBoost not available. XGBoost model will not work.")
    HAS_XGBOOST = False

from trading_bot.ml.base_model import BaseMLModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseMLModel):
    """
    XGBoost model implementation.
    """
    
    def __init__(
        self, 
        name: str,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        features: Optional[List[str]] = None,
        target_column: str = "close",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            name: Unique identifier for this model
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of the trees
            learning_rate: Learning rate
            features: List of feature columns to use
            target_column: Column to predict
            config: Additional configuration options
        """
        super().__init__(name, config)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.features = features
        self.target_column = target_column
        self.model = None
        self.feature_columns = None
        
        # Check if XGBoost is available
        if not HAS_XGBOOST:
            logger.warning(f"XGBoost is not available, {name} will not work")
    
    def train(self, features: pd.DataFrame, targets: pd.DataFrame = None) -> None:
        """
        Train the XGBoost model.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame (if None, will use target_column from features)
        """
        if not HAS_XGBOOST:
            logger.error("Cannot train model: XGBoost is not available")
            return
            
        logger.info(f"Training XGBoost model {self.name}")
        
        # Determine feature columns if not set
        if self.feature_columns is None:
            if self.features is not None:
                self.feature_columns = [col for col in self.features if col in features.columns]
            else:
                self.feature_columns = features.columns.tolist()
                if self.target_column in self.feature_columns:
                    self.feature_columns.remove(self.target_column)
        
        # Get X and y
        X = features[self.feature_columns].values
        
        if targets is not None:
            y = targets.values
        else:
            y = features[self.target_column].values
        
        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        self.model.fit(X, y)
        
        # Update model state
        self.is_trained = True
        
        logger.info(f"XGBoost model {self.name} trained successfully")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the XGBoost model.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_trained or self.model is None:
            logger.error("Model not trained. Call train() first.")
            return np.array([])
            
        # Get features
        X = features[self.feature_columns].values
        
        # Make predictions
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the XGBoost model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, cannot calculate feature importance")
            return {}
        
        try:    
            # Get feature importance
            importance = self.model.feature_importances_
            
            # Map to feature names
            feature_importance = {
                feature: float(importance[i]) 
                for i, feature in enumerate(self.feature_columns)
            }
            
            return feature_importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {} 