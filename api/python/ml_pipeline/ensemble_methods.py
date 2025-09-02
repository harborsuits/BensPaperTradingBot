"""
Ensemble Methods

Implements various strategies for combining predictions from multiple models
to create robust trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Available ensemble methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STACKED = "stacked"
    BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"
    MAX_CONFIDENCE = "max_confidence"
    ADAPTIVE = "adaptive"

class EnsembleAggregator:
    """
    Aggregates predictions from multiple models using various
    ensemble methods to create robust trading signals.
    """
    
    def __init__(self, method: Union[str, EnsembleMethod] = EnsembleMethod.WEIGHTED_AVERAGE):
        """
        Initialize the ensemble aggregator
        
        Args:
            method: Ensemble method to use (default: weighted_average)
        """
        if isinstance(method, str):
            try:
                self.method = EnsembleMethod(method)
            except ValueError:
                logger.warning(f"Unknown ensemble method: {method}, using weighted_average")
                self.method = EnsembleMethod.WEIGHTED_AVERAGE
        else:
            self.method = method
        
        logger.info(f"Initialized ensemble aggregator with method: {self.method.value}")
    
    def aggregate(self, 
                 predictions: Dict[str, Any], 
                 weights: Dict[str, float],
                 confidences: Optional[Dict[str, float]] = None,
                 threshold: float = 0.6) -> Dict[str, Any]:
        """
        Aggregate predictions from multiple models
        
        Args:
            predictions: Dictionary of model predictions (model_name -> prediction)
            weights: Dictionary of model weights (model_name -> weight)
            confidences: Optional dictionary of model confidences (model_name -> confidence)
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        # Default method to weighted average if method is not set
        if self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(predictions, weights, threshold)
        elif self.method == EnsembleMethod.MAJORITY_VOTE:
            return self._majority_vote(predictions, weights, threshold)
        elif self.method == EnsembleMethod.CONFIDENCE_WEIGHTED:
            if confidences is None:
                logger.warning("Confidence-weighted method requires confidences, using equal confidences")
                confidences = {name: 0.5 for name in predictions.keys()}
            return self._confidence_weighted(predictions, weights, confidences, threshold)
        elif self.method == EnsembleMethod.STACKED:
            return self._stacked(predictions, weights, confidences, threshold)
        elif self.method == EnsembleMethod.BAYESIAN_MODEL_AVERAGING:
            if confidences is None:
                logger.warning("Bayesian method requires confidences, using equal confidences")
                confidences = {name: 0.5 for name in predictions.keys()}
            return self._bayesian_model_averaging(predictions, weights, confidences, threshold)
        elif self.method == EnsembleMethod.MAX_CONFIDENCE:
            if confidences is None:
                logger.warning("Max confidence method requires confidences, using equal confidences")
                confidences = {name: 0.5 for name in predictions.keys()}
            return self._max_confidence(predictions, confidences, threshold)
        elif self.method == EnsembleMethod.ADAPTIVE:
            if confidences is None:
                logger.warning("Adaptive method requires confidences, using equal confidences")
                confidences = {name: 0.5 for name in predictions.keys()}
            return self._adaptive(predictions, weights, confidences, threshold)
        else:
            logger.warning(f"Unknown ensemble method: {self.method.value}, using weighted_average")
            return self._weighted_average(predictions, weights, threshold)
    
    def _weighted_average(self, 
                         predictions: Dict[str, Any], 
                         weights: Dict[str, float],
                         threshold: float) -> Dict[str, Any]:
        """
        Combine predictions using weighted average
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        if not predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize predictions to be numeric
        numeric_predictions = {}
        valid_model_count = 0
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            # Convert prediction to numeric value if needed
            if isinstance(pred, dict) and 'prediction' in pred:
                # Handle dictionary format
                if isinstance(pred['prediction'], (list, np.ndarray)):
                    numeric_predictions[name] = pred['prediction'][-1]  # Use latest prediction
                else:
                    numeric_predictions[name] = pred['prediction']
            elif isinstance(pred, (list, np.ndarray)):
                numeric_predictions[name] = pred[-1]  # Use latest prediction
            elif isinstance(pred, (int, float)):
                numeric_predictions[name] = pred
            elif isinstance(pred, str):
                # Convert string signals to numeric
                if pred.lower() == 'buy':
                    numeric_predictions[name] = 1.0
                elif pred.lower() == 'sell':
                    numeric_predictions[name] = -1.0
                else:
                    numeric_predictions[name] = 0.0
            else:
                logger.warning(f"Unable to convert prediction from model {name} to numeric value")
                continue
            
            valid_model_count += 1
        
        if valid_model_count == 0:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for name, pred in numeric_predictions.items():
            if name in weights:
                model_weight = weights[name]
            else:
                model_weight = 1.0  # Default weight
                
            weighted_sum += pred * model_weight
            total_weight += model_weight
        
        if total_weight > 0:
            aggregated_prediction = weighted_sum / total_weight
        else:
            aggregated_prediction = 0
        
        # Determine signal
        if aggregated_prediction >= threshold:
            signal = 'buy'
        elif aggregated_prediction <= -threshold:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        # Estimate confidence (absolute value scaled to 0.5-1.0 range)
        confidence = 0.5 + min(0.5, abs(aggregated_prediction) / 2)
        
        return {
            'signal': signal,
            'strength': aggregated_prediction,
            'confidence': confidence,
            'models': {name: {'prediction': pred, 'weight': weights.get(name, 1.0)} 
                      for name, pred in numeric_predictions.items()}
        }
    
    def _majority_vote(self, 
                      predictions: Dict[str, Any], 
                      weights: Dict[str, float],
                      threshold: float) -> Dict[str, Any]:
        """
        Combine predictions using majority vote
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        if not predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize predictions to -1, 0, 1
        votes = {}
        model_signals = {}
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            # Determine vote (-1, 0, 1)
            if isinstance(pred, dict) and 'prediction' in pred:
                # Handle dictionary format
                if isinstance(pred['prediction'], (list, np.ndarray)):
                    value = pred['prediction'][-1]  # Use latest prediction
                else:
                    value = pred['prediction']
            elif isinstance(pred, (list, np.ndarray)):
                value = pred[-1]  # Use latest prediction
            elif isinstance(pred, (int, float)):
                value = pred
            elif isinstance(pred, str):
                # Convert string signals to numeric
                if pred.lower() == 'buy':
                    value = 1.0
                elif pred.lower() == 'sell':
                    value = -1.0
                else:
                    value = 0.0
            else:
                logger.warning(f"Unable to convert prediction from model {name} to numeric value")
                continue
            
            # Convert to vote
            if value >= threshold:
                vote = 1  # Buy
                signal = 'buy'
            elif value <= -threshold:
                vote = -1  # Sell
                signal = 'sell'
            else:
                vote = 0  # Neutral
                signal = 'neutral'
            
            # Apply model weight (integer number of votes)
            model_weight = weights.get(name, 1.0)
            weighted_votes = int(vote * model_weight)
            
            votes[name] = weighted_votes
            model_signals[name] = signal
        
        if not votes:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Count votes
        buy_votes = sum(1 for v in votes.values() if v > 0)
        sell_votes = sum(1 for v in votes.values() if v < 0)
        neutral_votes = sum(1 for v in votes.values() if v == 0)
        
        # Determine signal based on majority
        total_votes = buy_votes + sell_votes + neutral_votes
        
        if buy_votes > sell_votes and buy_votes > neutral_votes:
            signal = 'buy'
            strength = buy_votes / total_votes
        elif sell_votes > buy_votes and sell_votes > neutral_votes:
            signal = 'sell'
            strength = -sell_votes / total_votes
        else:
            signal = 'neutral'
            strength = 0
        
        # Calculate confidence based on vote consistency
        if total_votes > 0:
            consistency = max(buy_votes, sell_votes, neutral_votes) / total_votes
            confidence = 0.5 + (consistency * 0.5)  # Scale to 0.5-1.0 range
        else:
            confidence = 0.5
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'models': {name: {'prediction': votes[name], 'signal': model_signals.get(name, 'neutral')} 
                      for name in votes.keys()}
        }
    
    def _confidence_weighted(self, 
                            predictions: Dict[str, Any], 
                            weights: Dict[str, float],
                            confidences: Dict[str, float],
                            threshold: float) -> Dict[str, Any]:
        """
        Combine predictions using confidence-weighted approach
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            confidences: Dictionary of model confidences
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        if not predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize predictions
        numeric_predictions = {}
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            # Convert prediction to numeric value if needed
            if isinstance(pred, dict) and 'prediction' in pred:
                if isinstance(pred['prediction'], (list, np.ndarray)):
                    numeric_predictions[name] = pred['prediction'][-1]
                else:
                    numeric_predictions[name] = pred['prediction']
            elif isinstance(pred, (list, np.ndarray)):
                numeric_predictions[name] = pred[-1]
            elif isinstance(pred, (int, float)):
                numeric_predictions[name] = pred
            elif isinstance(pred, str):
                if pred.lower() == 'buy':
                    numeric_predictions[name] = 1.0
                elif pred.lower() == 'sell':
                    numeric_predictions[name] = -1.0
                else:
                    numeric_predictions[name] = 0.0
            else:
                logger.warning(f"Unable to convert prediction from model {name} to numeric value")
                continue
        
        if not numeric_predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Calculate confidence-weighted sum
        weighted_sum = 0
        total_weight = 0
        
        for name, pred in numeric_predictions.items():
            # Combined weight: model weight * model confidence
            model_weight = weights.get(name, 1.0)
            model_confidence = confidences.get(name, 0.5)
            combined_weight = model_weight * model_confidence
            
            weighted_sum += pred * combined_weight
            total_weight += combined_weight
        
        if total_weight > 0:
            aggregated_prediction = weighted_sum / total_weight
        else:
            aggregated_prediction = 0
        
        # Determine signal
        if aggregated_prediction >= threshold:
            signal = 'buy'
        elif aggregated_prediction <= -threshold:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        # Use average confidence as overall confidence
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
        
        return {
            'signal': signal,
            'strength': aggregated_prediction,
            'confidence': avg_confidence,
            'models': {name: {
                'prediction': pred,
                'confidence': confidences.get(name, 0.5),
                'weight': weights.get(name, 1.0)
            } for name, pred in numeric_predictions.items()}
        }
    
    def _stacked(self, 
                predictions: Dict[str, Any], 
                weights: Dict[str, float],
                confidences: Optional[Dict[str, float]],
                threshold: float) -> Dict[str, Any]:
        """
        Stacked ensemble (meta-learner approach)
        
        This is a simplified version since we don't have a true stacked model.
        It uses a weighted combination with a non-linear transformation.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            confidences: Dictionary of model confidences
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        # First, get weighted average prediction
        weighted_result = self._weighted_average(predictions, weights, threshold)
        
        # Then, apply a non-linear transformation to enhance strong signals
        # and reduce weak signals (sigmoid-like behavior)
        strength = weighted_result['strength']
        
        # Apply non-linear transformation
        if abs(strength) > threshold:
            # Enhance strong signals
            enhanced_strength = np.sign(strength) * (threshold + (abs(strength) - threshold) * 1.5)
            enhanced_strength = np.clip(enhanced_strength, -1.0, 1.0)
        else:
            # Reduce weak signals
            enhanced_strength = strength * (abs(strength) / threshold)
        
        # Update result
        weighted_result['strength'] = enhanced_strength
        
        # Recalculate signal based on enhanced strength
        if enhanced_strength >= threshold:
            weighted_result['signal'] = 'buy'
        elif enhanced_strength <= -threshold:
            weighted_result['signal'] = 'sell'
        else:
            weighted_result['signal'] = 'neutral'
        
        return weighted_result
    
    def _bayesian_model_averaging(self, 
                                 predictions: Dict[str, Any], 
                                 weights: Dict[str, float],
                                 confidences: Dict[str, float],
                                 threshold: float) -> Dict[str, Any]:
        """
        Bayesian Model Averaging
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            confidences: Dictionary of model confidences
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        if not predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize predictions and confidences
        numeric_predictions = {}
        model_probabilities = {}
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            # Convert prediction to numeric value
            if isinstance(pred, dict) and 'prediction' in pred:
                if isinstance(pred['prediction'], (list, np.ndarray)):
                    numeric_predictions[name] = pred['prediction'][-1]
                else:
                    numeric_predictions[name] = pred['prediction']
            elif isinstance(pred, (list, np.ndarray)):
                numeric_predictions[name] = pred[-1]
            elif isinstance(pred, (int, float)):
                numeric_predictions[name] = pred
            elif isinstance(pred, str):
                if pred.lower() == 'buy':
                    numeric_predictions[name] = 1.0
                elif pred.lower() == 'sell':
                    numeric_predictions[name] = -1.0
                else:
                    numeric_predictions[name] = 0.0
            else:
                logger.warning(f"Unable to convert prediction from model {name} to numeric value")
                continue
            
            # Convert confidence to probability
            model_conf = confidences.get(name, 0.5)
            model_probabilities[name] = model_conf
        
        if not numeric_predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize probabilities (Bayesian model weights)
        total_prob = sum(model_probabilities.values())
        if total_prob > 0:
            normalized_probs = {name: prob/total_prob for name, prob in model_probabilities.items()}
        else:
            normalized_probs = {name: 1.0/len(model_probabilities) for name in model_probabilities}
        
        # Apply Bayesian model averaging
        weighted_sum = 0
        
        for name, pred in numeric_predictions.items():
            # Combine Bayesian weight with model weight
            model_weight = weights.get(name, 1.0)
            bayesian_weight = normalized_probs[name]
            combined_weight = model_weight * bayesian_weight
            
            weighted_sum += pred * combined_weight
        
        # Determine signal
        if weighted_sum >= threshold:
            signal = 'buy'
        elif weighted_sum <= -threshold:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        # Calculate overall confidence based on model agreement
        predictions_array = np.array(list(numeric_predictions.values()))
        agreement = 1.0 - np.std(predictions_array) if len(predictions_array) > 1 else 1.0
        # Scale agreement to confidence range (0.5-1.0)
        confidence = 0.5 + (agreement * 0.5)
        
        return {
            'signal': signal,
            'strength': weighted_sum,
            'confidence': confidence,
            'models': {name: {
                'prediction': pred,
                'weight': weights.get(name, 1.0),
                'bayesian_weight': normalized_probs[name]
            } for name, pred in numeric_predictions.items()}
        }
    
    def _max_confidence(self, 
                       predictions: Dict[str, Any], 
                       confidences: Dict[str, float],
                       threshold: float) -> Dict[str, Any]:
        """
        Use prediction from model with highest confidence
        
        Args:
            predictions: Dictionary of model predictions
            confidences: Dictionary of model confidences
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with selected prediction and signal
        """
        if not predictions or not confidences:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize predictions and find model with max confidence
        numeric_predictions = {}
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            # Convert prediction to numeric value
            if isinstance(pred, dict) and 'prediction' in pred:
                if isinstance(pred['prediction'], (list, np.ndarray)):
                    numeric_predictions[name] = pred['prediction'][-1]
                else:
                    numeric_predictions[name] = pred['prediction']
            elif isinstance(pred, (list, np.ndarray)):
                numeric_predictions[name] = pred[-1]
            elif isinstance(pred, (int, float)):
                numeric_predictions[name] = pred
            elif isinstance(pred, str):
                if pred.lower() == 'buy':
                    numeric_predictions[name] = 1.0
                elif pred.lower() == 'sell':
                    numeric_predictions[name] = -1.0
                else:
                    numeric_predictions[name] = 0.0
            else:
                logger.warning(f"Unable to convert prediction from model {name} to numeric value")
                continue
        
        if not numeric_predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Find model with highest confidence
        max_confidence = 0
        max_confidence_model = None
        
        for name in numeric_predictions.keys():
            model_conf = confidences.get(name, 0)
            if model_conf > max_confidence:
                max_confidence = model_conf
                max_confidence_model = name
        
        if max_confidence_model is None:
            # No model with confidence data, use first model
            max_confidence_model = list(numeric_predictions.keys())[0]
            max_confidence = confidences.get(max_confidence_model, 0.5)
        
        # Get prediction from selected model
        selected_prediction = numeric_predictions[max_confidence_model]
        
        # Determine signal
        if selected_prediction >= threshold:
            signal = 'buy'
        elif selected_prediction <= -threshold:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'strength': selected_prediction,
            'confidence': max_confidence,
            'selected_model': max_confidence_model,
            'models': {name: {
                'prediction': pred,
                'confidence': confidences.get(name, 0),
                'selected': name == max_confidence_model
            } for name, pred in numeric_predictions.items()}
        }
    
    def _adaptive(self, 
                 predictions: Dict[str, Any], 
                 weights: Dict[str, float],
                 confidences: Dict[str, float],
                 threshold: float) -> Dict[str, Any]:
        """
        Adaptive ensemble that selects method based on model agreement
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            confidences: Dictionary of model confidences
            threshold: Threshold for signal generation
            
        Returns:
            Dictionary with aggregated prediction and signal
        """
        if not predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Normalize predictions
        numeric_predictions = {}
        
        for name, pred in predictions.items():
            if pred is None:
                continue
                
            # Convert prediction to numeric value
            if isinstance(pred, dict) and 'prediction' in pred:
                if isinstance(pred['prediction'], (list, np.ndarray)):
                    numeric_predictions[name] = pred['prediction'][-1]
                else:
                    numeric_predictions[name] = pred['prediction']
            elif isinstance(pred, (list, np.ndarray)):
                numeric_predictions[name] = pred[-1]
            elif isinstance(pred, (int, float)):
                numeric_predictions[name] = pred
            elif isinstance(pred, str):
                if pred.lower() == 'buy':
                    numeric_predictions[name] = 1.0
                elif pred.lower() == 'sell':
                    numeric_predictions[name] = -1.0
                else:
                    numeric_predictions[name] = 0.0
            else:
                logger.warning(f"Unable to convert prediction from model {name} to numeric value")
                continue
        
        if not numeric_predictions:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Check model agreement
        predictions_array = np.array(list(numeric_predictions.values()))
        std_dev = np.std(predictions_array)
        
        # Choose method based on agreement
        if std_dev < 0.3:
            # High agreement, use weighted average
            logger.debug("High model agreement, using weighted average")
            return self._weighted_average(predictions, weights, threshold)
        elif std_dev < 0.6:
            # Moderate agreement, use confidence weighted
            logger.debug("Moderate model agreement, using confidence weighted")
            return self._confidence_weighted(predictions, weights, confidences, threshold)
        else:
            # Low agreement, use max confidence
            logger.debug("Low model agreement, using max confidence")
            return self._max_confidence(predictions, confidences, threshold)

# Utility functions for ensemble operations

def create_ensemble_config(method: str = "weighted_average",
                          dynamic_weights: bool = True,
                          performance_metric: str = "sharpe_ratio",
                          weight_update_frequency: str = "daily",
                          min_weight: float = 0.1,
                          max_weight: float = 3.0) -> Dict[str, Any]:
    """
    Create configuration for ensemble methods
    
    Args:
        method: Ensemble method to use
        dynamic_weights: Whether to dynamically adjust weights based on performance
        performance_metric: Metric to use for weight adjustment
        weight_update_frequency: How often to update weights
        min_weight: Minimum model weight
        max_weight: Maximum model weight
        
    Returns:
        Dictionary with ensemble configuration
    """
    return {
        "method": method,
        "dynamic_weights": dynamic_weights,
        "performance_metric": performance_metric,
        "weight_update_frequency": weight_update_frequency,
        "min_weight": min_weight,
        "max_weight": max_weight
    }

def adjust_model_weights(model_weights: Dict[str, float],
                        performance_metrics: Dict[str, Dict[str, float]],
                        metric_name: str = "sharpe_ratio",
                        min_weight: float = 0.1,
                        max_weight: float = 3.0) -> Dict[str, float]:
    """
    Adjust model weights based on performance metrics
    
    Args:
        model_weights: Current model weights
        performance_metrics: Performance metrics for each model
        metric_name: Name of metric to use for adjustment
        min_weight: Minimum weight
        max_weight: Maximum weight
        
    Returns:
        Dictionary with adjusted weights
    """
    adjusted_weights = model_weights.copy()
    
    if not performance_metrics:
        return adjusted_weights
    
    # Extract metrics
    metrics = {}
    for model_name, model_metrics in performance_metrics.items():
        if model_name in adjusted_weights and metric_name in model_metrics:
            metrics[model_name] = model_metrics[metric_name]
    
    if not metrics:
        return adjusted_weights
    
    # Normalize metrics to 0-1 range for comparison
    min_metric = min(metrics.values())
    max_metric = max(metrics.values())
    
    if max_metric == min_metric:
        # All models have same performance, no adjustment needed
        return adjusted_weights
    
    metric_range = max_metric - min_metric
    
    # Adjust weights based on normalized performance
    for model_name, metric_value in metrics.items():
        # Normalize metric to 0-1 range
        normalized_metric = (metric_value - min_metric) / metric_range
        
        # Scale adjustment factor (0.5 to 1.5)
        adjustment_factor = 0.5 + normalized_metric
        
        # Apply adjustment
        current_weight = adjusted_weights[model_name]
        new_weight = current_weight * adjustment_factor
        
        # Enforce weight bounds
        adjusted_weights[model_name] = max(min_weight, min(max_weight, new_weight))
    
    return adjusted_weights
