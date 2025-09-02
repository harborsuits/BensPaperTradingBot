#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble model implementation that combines multiple ML models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum, auto

from trading_bot.ml.base_model import BaseMLModel

logger = logging.getLogger(__name__)


class EnsembleMethod(str, Enum):
    """Ensemble methods for combining predictions."""
    AVERAGE = "average"  # Simple averaging
    WEIGHTED = "weighted"  # Weighted average based on model performance
    STACKING = "stacking"  # Meta-model learns how to combine predictions
    MAJORITY_VOTE = "majority_vote"  # For classification, take most common prediction
    RANK = "rank"  # Average rank of predictions


class EnsembleModel(BaseMLModel):
    """
    Ensemble model that combines predictions from multiple base models.
    
    Supports various ensemble methods:
    - Simple averaging
    - Weighted averaging (based on validation performance)
    - Stacking (meta-model)
    - Majority voting (for classification)
    - Rank-based averaging
    """
    
    def __init__(
        self,
        name: str,
        base_models: List[BaseMLModel] = None,
        meta_model: Optional[BaseMLModel] = None,
        ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ensemble model.
        
        Args:
            name: Unique identifier for this model
            base_models: List of base models to use in ensemble
            meta_model: Model for stacking (optional)
            ensemble_method: Method to combine predictions
            weights: Weights for each model (for weighted averaging)
            config: Additional configuration options
        """
        super().__init__(name, config)
        self.base_models = base_models or []
        self.model_names = [model.name for model in self.base_models]
        self.meta_model = meta_model
        self.ensemble_method = ensemble_method
        self.weights = weights or {}
        
        # Initialize performance metrics for each model
        self.model_metrics = {model.name: {} for model in self.base_models}
        
        # Stacking requires a meta-model
        if self.ensemble_method == EnsembleMethod.STACKING and self.meta_model is None:
            raise ValueError("Meta-model must be provided for stacking ensemble")
    
    def add_model(self, model: BaseMLModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add to the ensemble
            weight: Weight for this model in weighted averaging
        """
        if model.name in self.model_names:
            logger.warning(f"Model {model.name} already exists in ensemble. Updating.")
            # Replace existing model
            for i, base_model in enumerate(self.base_models):
                if base_model.name == model.name:
                    self.base_models[i] = model
                    break
        else:
            self.base_models.append(model)
            self.model_names.append(model.name)
            self.model_metrics[model.name] = {}
            
        # Update weight
        self.weights[model.name] = weight
        
        # Reset trained flag
        self.is_trained = False
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the ensemble.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if model was removed, False otherwise
        """
        for i, model in enumerate(self.base_models):
            if model.name == model_name:
                self.base_models.pop(i)
                self.model_names.remove(model_name)
                if model_name in self.weights:
                    del self.weights[model_name]
                if model_name in self.model_metrics:
                    del self.model_metrics[model_name]
                
                # Reset trained flag
                self.is_trained = False
                return True
                
        logger.warning(f"Model {model_name} not found in ensemble")
        return False
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before training or prediction.
        
        This implementation just passes through the data by default.
        Subclasses can override for more sophisticated preprocessing.
        
        Args:
            data: Input data as DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # By default, no preprocessing in the ensemble itself
        # Individual models should handle their own preprocessing
        return data
    
    def train(self, features: pd.DataFrame, targets: pd.DataFrame,
             validation_split: float = 0.2, random_state: int = 42) -> None:
        """
        Train all base models and the meta-model (if using stacking).
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        """
        if not self.base_models:
            logger.error("No base models in ensemble")
            return
            
        # Split data for training and meta-model training (if using stacking)
        if self.ensemble_method == EnsembleMethod.STACKING and validation_split > 0:
            # For stacking, we need to split the data
            np.random.seed(random_state)
            indices = np.random.permutation(len(features))
            val_size = int(validation_split * len(features))
            train_idx, val_idx = indices[val_size:], indices[:val_size]
            
            train_features = features.iloc[train_idx]
            train_targets = targets.iloc[train_idx]
            val_features = features.iloc[val_idx]
            val_targets = targets.iloc[val_idx]
            
            # Train base models on training set
            for model in self.base_models:
                logger.info(f"Training base model: {model.name}")
                model.train(model.preprocess(train_features), train_targets)
                
                # Validate models and store metrics
                self.model_metrics[model.name] = model.validate(
                    model.preprocess(val_features), val_targets
                )
                
            # Generate meta-features (predictions from base models)
            meta_features = pd.DataFrame()
            for model in self.base_models:
                pred = model.predict(model.preprocess(val_features))
                meta_features[model.name] = pred
                
            # Train meta-model on base model predictions
            logger.info(f"Training meta-model: {self.meta_model.name}")
            self.meta_model.train(
                self.meta_model.preprocess(meta_features), 
                val_targets
            )
            
            # Re-train base models on full dataset
            for model in self.base_models:
                logger.info(f"Re-training base model on full dataset: {model.name}")
                model.train(model.preprocess(features), targets)
                
            # Mark meta-model as trained
            self.meta_model.is_trained = True
            self.meta_model.last_training_time = datetime.now()
            
        else:
            # For non-stacking methods, just train each base model
            for model in self.base_models:
                logger.info(f"Training base model: {model.name}")
                model.train(model.preprocess(features), targets)
                
                # If validation split, validate model
                if validation_split > 0:
                    np.random.seed(random_state)
                    indices = np.random.permutation(len(features))
                    val_size = int(validation_split * len(features))
                    val_idx = indices[:val_size]
                    val_features = features.iloc[val_idx]
                    val_targets = targets.iloc[val_idx]
                    
                    self.model_metrics[model.name] = model.validate(
                        model.preprocess(val_features), val_targets
                    )
        
        # Update weights based on model performance for weighted ensemble
        if self.ensemble_method == EnsembleMethod.WEIGHTED:
            self._update_weights()
            
        # Mark ensemble as trained
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        logger.info(f"Ensemble model {self.name} trained successfully")
    
    def _update_weights(self) -> None:
        """
        Update model weights based on validation performance.
        
        Uses inverse error weighting: models with lower error get higher weights.
        """
        # Check if we have validation metrics
        if not all(self.model_metrics.values()):
            logger.warning("Not all models have validation metrics. Using equal weights.")
            for model in self.base_models:
                self.weights[model.name] = 1.0 / len(self.base_models)
            return
            
        # Use directional accuracy if available, otherwise RMSE
        metric = "directional_accuracy"
        inverse = False  # Higher is better for accuracy
        
        if not all(metric in metrics for metrics in self.model_metrics.values()):
            metric = "rmse"
            inverse = True  # Lower is better for error
            
            if not all(metric in metrics for metrics in self.model_metrics.values()):
                logger.warning(f"Not all models have {metric}. Using equal weights.")
                for model in self.base_models:
                    self.weights[model.name] = 1.0 / len(self.base_models)
                return
        
        # Calculate weights based on performance
        metric_values = {model.name: self.model_metrics[model.name].get(metric, 0) 
                        for model in self.base_models}
        
        if inverse:
            # For error metrics (lower is better), use inverse weighting
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            total = sum(1.0 / (val + epsilon) for val in metric_values.values())
            for name, val in metric_values.items():
                self.weights[name] = (1.0 / (val + epsilon)) / total
        else:
            # For accuracy metrics (higher is better), use direct weighting
            total = sum(metric_values.values())
            if total > 0:
                for name, val in metric_values.items():
                    self.weights[name] = val / total
            else:
                # If all zeros, use equal weights
                for name in metric_values:
                    self.weights[name] = 1.0 / len(metric_values)
                    
        logger.info(f"Updated model weights based on {metric}: {self.weights}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the ensemble method.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_trained:
            logger.warning("Ensemble not trained, predictions may be unreliable")
            
        if not self.base_models:
            raise ValueError("No base models in ensemble")
            
        # Get predictions from each base model
        predictions = {}
        for model in self.base_models:
            if not model.is_trained:
                logger.warning(f"Model {model.name} not trained, skipping")
                continue
                
            # Preprocess features for this model
            model_features = model.preprocess(features)
            
            # Get predictions
            try:
                pred = model.predict(model_features)
                predictions[model.name] = pred
            except Exception as e:
                logger.error(f"Error getting predictions from {model.name}: {str(e)}")
                
        if not predictions:
            raise RuntimeError("No valid predictions from base models")
            
        # Combine predictions based on ensemble method
        if self.ensemble_method == EnsembleMethod.STACKING and self.meta_model and self.meta_model.is_trained:
            # For stacking, use meta-model to combine predictions
            meta_features = pd.DataFrame({name: pred for name, pred in predictions.items()})
            return self.meta_model.predict(self.meta_model.preprocess(meta_features))
            
        elif self.ensemble_method == EnsembleMethod.WEIGHTED:
            # For weighted average, use model weights
            weighted_sum = np.zeros_like(list(predictions.values())[0])
            weight_sum = 0
            
            for name, pred in predictions.items():
                weight = self.weights.get(name, 1.0)
                weighted_sum += weight * pred
                weight_sum += weight
                
            if weight_sum > 0:
                return weighted_sum / weight_sum
            else:
                return np.zeros_like(list(predictions.values())[0])
                
        elif self.ensemble_method == EnsembleMethod.MAJORITY_VOTE:
            # For classification, take majority vote
            # Convert predictions to class labels if they're probabilities
            pred_classes = {}
            for name, pred in predictions.items():
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    # For multi-class, take argmax
                    pred_classes[name] = np.argmax(pred, axis=1)
                else:
                    # For binary, threshold at 0.5
                    pred_classes[name] = (pred > 0.5).astype(int)
                    
            # Stack all predictions and take mode (most common) along axis 0
            stacked = np.vstack(list(pred_classes.values()))
            return np.squeeze(np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                             axis=0, arr=stacked.astype(int)))
                                             
        elif self.ensemble_method == EnsembleMethod.RANK:
            # Rank-based ensemble
            all_preds = np.array(list(predictions.values()))
            # Get ranks along axis 0 (across models)
            ranks = np.zeros_like(all_preds)
            for i in range(all_preds.shape[1]):
                ranks[:, i] = all_preds[:, i].argsort().argsort()
            
            # Average ranks
            avg_ranks = np.mean(ranks, axis=0)
            return avg_ranks
            
        else:
            # Default to simple average
            return np.mean(list(predictions.values()), axis=0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from all base models.
        
        For the ensemble, combines feature importances from all models using the
        same weighting as used for predictions.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not self.base_models:
            logger.warning("Ensemble not trained or no base models")
            return {}
            
        # Get feature importance from each model
        importances = {}
        for model in self.base_models:
            if not model.is_trained:
                continue
                
            model_importance = model.get_feature_importance()
            weight = self.weights.get(model.name, 1.0 / len(self.base_models))
            
            for feature, importance in model_importance.items():
                if feature not in importances:
                    importances[feature] = 0
                importances[feature] += weight * importance
                
        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            return {feature: importance / total for feature, importance in importances.items()}
        return importances
    
    def explain_prediction(self, features: pd.DataFrame, index: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Explain a prediction using feature importance from all models.
        
        Args:
            features: Feature DataFrame
            index: Index of the prediction to explain
            
        Returns:
            Dictionary mapping model names to feature contributions
        """
        if not self.is_trained:
            logger.warning("Ensemble not trained, explanations may be unreliable")
            return {}
            
        # Get explanations from each model
        explanations = {}
        for model in self.base_models:
            if not model.is_trained:
                continue
                
            try:
                # Preprocess features for this model
                model_features = model.preprocess(features)
                
                # Get explanation
                model_explanation = model.explain_prediction(model_features, index)
                
                if model_explanation:
                    explanations[model.name] = {
                        'weight': self.weights.get(model.name, 1.0 / len(self.base_models)),
                        'features': model_explanation
                    }
            except Exception as e:
                logger.error(f"Error getting explanation from {model.name}: {str(e)}")
                
        return explanations
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all models in the ensemble.
        
        Returns:
            Dictionary mapping model names to performance metrics
        """
        return self.model_metrics.copy()
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current weights for all models in the ensemble.
        
        Returns:
            Dictionary mapping model names to weights
        """
        return self.weights.copy() 