"""
Regime Classifier

This module provides classification of market regimes based on calculated features
using a combination of rule-based heuristics, statistical methods, and machine learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import pickle
import os
from datetime import datetime
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Import local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeType

logger = logging.getLogger(__name__)

class ClassificationMethod(str, Enum):
    """Classification methods for market regimes."""
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"

class RegimeClassifier:
    """
    Classifies market regimes based on calculated features.
    
    Uses multiple classification methods:
    - Rule-based heuristics
    - Statistical methods
    - Machine learning models
    - Ensemble approach
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize regime classifier.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Configure classification methods to use
        self.enabled_methods = self.config.get("enabled_methods", {
            ClassificationMethod.RULE_BASED: True,
            ClassificationMethod.STATISTICAL: True,
            ClassificationMethod.MACHINE_LEARNING: True,
            ClassificationMethod.ENSEMBLE: True
        })
        
        # Weights for ensemble method
        self.ensemble_weights = self.config.get("ensemble_weights", {
            ClassificationMethod.RULE_BASED: 0.4,
            ClassificationMethod.STATISTICAL: 0.3,
            ClassificationMethod.MACHINE_LEARNING: 0.3
        })
        
        # Thresholds for rule-based classification
        self.adx_threshold = self.config.get("adx_threshold", 25)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.015)
        self.high_volatility_threshold = self.config.get("high_volatility_threshold", 0.025)
        self.trend_slope_threshold = self.config.get("trend_slope_threshold", 0.001)
        self.bollinger_b_threshold = self.config.get("bollinger_b_threshold", 0.2)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        
        # ML model paths
        self.model_dir = self.config.get("model_dir", "models/regime_classifier")
        self.model_path = os.path.join(self.model_dir, "regime_classifier_model.joblib")
        self.scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        # ML model and preprocessing
        self.ml_model = None
        self.scaler = None
        self.feature_importance = {}
        
        # Historical classifications for transition analysis
        self.classification_history: Dict[str, Dict[str, List[Tuple[MarketRegimeType, float]]]] = {}
        
        # Initialize ML model if available
        self._load_ml_model()
        
        logger.info("Regime Classifier initialized")
    
    def _load_ml_model(self) -> bool:
        """
        Load ML model for regime classification.
        
        Returns:
            bool: Success status
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.ml_model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load feature importance if model supports it
                if hasattr(self.ml_model, 'feature_importances_'):
                    importances = self.ml_model.feature_importances_
                    # We'll set feature importance dict when we predict
                    self.feature_importance = {}
                
                logger.info(f"Loaded ML model from {self.model_path}")
                return True
            else:
                logger.warning(f"ML model files not found, creating default model")
                self._create_default_model()
                return False
                
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            self._create_default_model()
            return False
    
    def _create_default_model(self) -> None:
        """Create default ML model when saved model is not available."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Create a default model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Create a default scaler
            self.scaler = StandardScaler()
            
            logger.info("Created default ML model")
            
        except Exception as e:
            logger.error(f"Error creating default model: {str(e)}")
    
    def classify(self, features: Dict[str, float], symbol: str, 
                timeframe: str) -> Tuple[MarketRegimeType, float]:
        """
        Classify market regime based on features.
        
        Args:
            features: Feature values
            symbol: Symbol being classified
            timeframe: Timeframe being classified
            
        Returns:
            Tuple of (regime_type, confidence)
        """
        try:
            results = {}
            
            # Rule-based classification
            if self.enabled_methods.get(ClassificationMethod.RULE_BASED, True):
                rule_regime, rule_confidence = self._rule_based_classification(features)
                results[ClassificationMethod.RULE_BASED] = (rule_regime, rule_confidence)
            
            # Statistical classification
            if self.enabled_methods.get(ClassificationMethod.STATISTICAL, True):
                stat_regime, stat_confidence = self._statistical_classification(features)
                results[ClassificationMethod.STATISTICAL] = (stat_regime, stat_confidence)
            
            # Machine learning classification
            if self.enabled_methods.get(ClassificationMethod.MACHINE_LEARNING, True) and self.ml_model is not None:
                ml_regime, ml_confidence = self._ml_classification(features)
                results[ClassificationMethod.MACHINE_LEARNING] = (ml_regime, ml_confidence)
            
            # Ensemble classification
            if self.enabled_methods.get(ClassificationMethod.ENSEMBLE, True) and len(results) > 1:
                regime, confidence = self._ensemble_classification(results)
            else:
                # Use the first available method
                regime, confidence = next(iter(results.values()))
            
            # Store classification in history
            self._update_classification_history(symbol, timeframe, regime, confidence)
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            return MarketRegimeType.UNCERTAIN, 0.5
    
    def _rule_based_classification(self, features: Dict[str, float]) -> Tuple[MarketRegimeType, float]:
        """
        Perform rule-based classification.
        
        Args:
            features: Feature values
            
        Returns:
            Tuple of (regime_type, confidence)
        """
        try:
            # Extract key features
            adx = features.get("adx", 0)
            volatility = features.get("hist_volatility", 0)
            rsi = features.get("rsi", 50)
            trend_slope = features.get("regression_slope", 0)
            directional_slope = features.get("directional_slope", 0)
            bollinger_b = features.get("bollinger_b", 0.5)
            macd = features.get("macd", 0)
            macd_signal = features.get("macd_signal", 0)
            ma_crossover = features.get("ma_crossover", 0)
            bb_width = features.get("bb_width", 0)
            autocorrelation = features.get("autocorrelation", 0)
            variance_ratio = features.get("variance_ratio", 1.0)
            
            # Check for high volatility regime
            if volatility > self.high_volatility_threshold:
                if rsi > self.rsi_overbought:
                    return MarketRegimeType.BREAKOUT, min(0.6 + volatility * 10, 0.95)
                elif rsi < self.rsi_oversold:
                    return MarketRegimeType.BREAKDOWN, min(0.6 + volatility * 10, 0.95)
                else:
                    return MarketRegimeType.HIGH_VOLATILITY, min(0.5 + volatility * 15, 0.95)
            
            # Check for trending regimes
            if adx > self.adx_threshold:
                # Strong trend
                if trend_slope > self.trend_slope_threshold and directional_slope > 0:
                    # Uptrend
                    confidence = min(0.5 + (adx - self.adx_threshold) / 100 + abs(trend_slope) * 100, 0.95)
                    return MarketRegimeType.TRENDING_UP, confidence
                elif trend_slope < -self.trend_slope_threshold and directional_slope < 0:
                    # Downtrend
                    confidence = min(0.5 + (adx - self.adx_threshold) / 100 + abs(trend_slope) * 100, 0.95)
                    return MarketRegimeType.TRENDING_DOWN, confidence
            
            # Check for range-bound regimes
            if adx < self.adx_threshold and volatility < self.volatility_threshold:
                if bollinger_b < self.bollinger_b_threshold:
                    # Near bottom of range
                    confidence = min(0.5 + (self.adx_threshold - adx) / 100 + (1 - volatility / self.volatility_threshold) * 0.3, 0.9)
                    return MarketRegimeType.RANGE_BOUND, confidence
                elif bollinger_b > (1 - self.bollinger_b_threshold):
                    # Near top of range
                    confidence = min(0.5 + (self.adx_threshold - adx) / 100 + (1 - volatility / self.volatility_threshold) * 0.3, 0.9)
                    return MarketRegimeType.RANGE_BOUND, confidence
                else:
                    # Middle of range
                    confidence = min(0.5 + (self.adx_threshold - adx) / 100 + (1 - volatility / self.volatility_threshold) * 0.3, 0.9)
                    return MarketRegimeType.RANGE_BOUND, confidence
            
            # Check for potential reversals
            if (rsi > self.rsi_overbought and trend_slope < 0 and trend_slope > -self.trend_slope_threshold) or \
               (rsi < self.rsi_oversold and trend_slope > 0 and trend_slope < self.trend_slope_threshold):
                confidence = min(0.5 + abs(rsi - 50) / 100, 0.8)
                return MarketRegimeType.REVERSAL, confidence
            
            # Check for choppy market
            if adx < 15 and volatility > self.volatility_threshold:
                confidence = min(0.5 + (15 - adx) / 30 + (volatility - self.volatility_threshold) * 10, 0.8)
                return MarketRegimeType.CHOPPY, confidence
            
            # Check for low volatility
            if volatility < self.volatility_threshold / 2:
                confidence = min(0.5 + (1 - volatility / (self.volatility_threshold / 2)) * 0.3, 0.8)
                return MarketRegimeType.LOW_VOLATILITY, confidence
            
            # Default to normal market
            return MarketRegimeType.NORMAL, 0.6
            
        except Exception as e:
            logger.error(f"Error in rule-based classification: {str(e)}")
            return MarketRegimeType.UNCERTAIN, 0.5
    
    def _statistical_classification(self, features: Dict[str, float]) -> Tuple[MarketRegimeType, float]:
        """
        Perform statistical classification.
        
        Args:
            features: Feature values
            
        Returns:
            Tuple of (regime_type, confidence)
        """
        try:
            # Extract statistical indicators
            autocorr = features.get("autocorrelation", 0)
            var_ratio = features.get("variance_ratio", 1.0)
            hurst_exp = features.get("hurst_exponent", 0.5)
            mean_reversion = features.get("mean_reversion", 0.5)
            
            # Calculate trend strength based on statistical properties
            trend_strength = 0.0
            mean_reversion_strength = 0.0
            
            # Autocorrelation indicates trending or mean-reverting
            if abs(autocorr) > 0.2:
                if autocorr > 0:
                    trend_strength += autocorr * 2
                else:
                    mean_reversion_strength += abs(autocorr) * 2
            
            # Variance ratio indicates trending or mean-reverting
            # > 1 trends, < 1 mean-reverting
            if abs(var_ratio - 1.0) > 0.2:
                if var_ratio > 1.0:
                    trend_strength += min((var_ratio - 1.0) * 2, 1.0)
                else:
                    mean_reversion_strength += min((1.0 - var_ratio) * 2, 1.0)
            
            # Hurst exponent: > 0.5 trending, < 0.5 mean-reverting
            if abs(hurst_exp - 0.5) > 0.1:
                if hurst_exp > 0.5:
                    trend_strength += (hurst_exp - 0.5) * 4
                else:
                    mean_reversion_strength += (0.5 - hurst_exp) * 4
            
            # Add direct mean reversion score
            mean_reversion_strength += mean_reversion
            
            # Get direction indicators
            regression_slope = features.get("regression_slope", 0)
            price_velocity = features.get("price_velocity", 0)
            
            # Determine regime based on statistical properties
            if trend_strength > 0.6 and mean_reversion_strength < 0.3:
                # Trending market
                if regression_slope > 0:
                    confidence = min(0.5 + trend_strength * 0.4, 0.9)
                    return MarketRegimeType.TRENDING_UP, confidence
                else:
                    confidence = min(0.5 + trend_strength * 0.4, 0.9)
                    return MarketRegimeType.TRENDING_DOWN, confidence
            
            elif mean_reversion_strength > 0.6 and trend_strength < 0.3:
                # Range-bound, mean-reverting market
                confidence = min(0.5 + mean_reversion_strength * 0.4, 0.9)
                return MarketRegimeType.RANGE_BOUND, confidence
            
            elif trend_strength > 0.4 and mean_reversion_strength > 0.4:
                # Choppy market (both trending and mean-reverting)
                confidence = min(0.5 + (trend_strength + mean_reversion_strength) * 0.2, 0.8)
                return MarketRegimeType.CHOPPY, confidence
            
            elif trend_strength < 0.3 and mean_reversion_strength < 0.3:
                # Low conviction, uncertain market
                adx = features.get("adx", 0)
                volatility = features.get("hist_volatility", 0)
                
                if volatility > self.volatility_threshold:
                    return MarketRegimeType.HIGH_VOLATILITY, 0.6
                elif volatility < self.volatility_threshold / 2:
                    return MarketRegimeType.LOW_VOLATILITY, 0.6
                else:
                    return MarketRegimeType.NORMAL, 0.6
            
            # Default to normal market
            return MarketRegimeType.NORMAL, 0.6
            
        except Exception as e:
            logger.error(f"Error in statistical classification: {str(e)}")
            return MarketRegimeType.UNCERTAIN, 0.5
    
    def _ml_classification(self, features: Dict[str, float]) -> Tuple[MarketRegimeType, float]:
        """
        Perform machine learning classification.
        
        Args:
            features: Feature values
            
        Returns:
            Tuple of (regime_type, confidence)
        """
        try:
            if self.ml_model is None:
                # Fall back to rule-based if no model
                return self._rule_based_classification(features)
            
            # Prepare feature vector for model
            # We need to make sure we have the same features the model was trained on
            # For now, just use what we have and handle missing values
            feature_list = list(features.values())
            feature_array = np.array(feature_list).reshape(1, -1)
            
            # Scale features if scaler exists
            if self.scaler is not None:
                try:
                    feature_array = self.scaler.transform(feature_array)
                except:
                    # If scaler fails, just use raw features
                    pass
            
            # Get prediction and probability
            regime_index = self.ml_model.predict(feature_array)[0]
            probabilities = self.ml_model.predict_proba(feature_array)[0]
            
            # Get highest probability
            confidence = float(max(probabilities))
            
            # Map index to regime type
            regime_types = list(MarketRegimeType)
            if isinstance(regime_index, (int, np.integer)) and 0 <= regime_index < len(regime_types):
                regime_type = regime_types[regime_index]
            else:
                # For non-numeric labels
                regime_type = MarketRegimeType(regime_index)
            
            # Update feature importance if available
            if hasattr(self.ml_model, 'feature_importances_'):
                feature_names = list(features.keys())
                importances = self.ml_model.feature_importances_
                if len(feature_names) == len(importances):
                    self.feature_importance = {
                        name: float(importance) for name, importance in zip(feature_names, importances)
                    }
            
            return regime_type, confidence
            
        except Exception as e:
            logger.error(f"Error in ML classification: {str(e)}")
            # Fall back to rule-based
            return self._rule_based_classification(features)
    
    def _ensemble_classification(self, results: Dict[ClassificationMethod, Tuple[MarketRegimeType, float]]) -> Tuple[MarketRegimeType, float]:
        """
        Perform ensemble classification by combining multiple methods.
        
        Args:
            results: Results from different classification methods
            
        Returns:
            Tuple of (regime_type, confidence)
        """
        try:
            # Count votes for each regime type
            regime_votes = {}
            regime_confidences = {}
            
            # Weight each method's vote by its confidence and method weight
            for method, (regime, confidence) in results.items():
                method_weight = self.ensemble_weights.get(method, 1.0 / len(results))
                weighted_vote = confidence * method_weight
                
                if regime not in regime_votes:
                    regime_votes[regime] = 0
                    regime_confidences[regime] = 0
                
                regime_votes[regime] += weighted_vote
                regime_confidences[regime] += confidence
            
            # Find regime with highest weighted vote
            if not regime_votes:
                return MarketRegimeType.UNCERTAIN, 0.5
                
            winning_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            
            # Calculate average confidence for winning regime
            avg_confidence = 0.0
            count = 0
            for method, (regime, confidence) in results.items():
                if regime == winning_regime:
                    avg_confidence += confidence
                    count += 1
            
            if count > 0:
                avg_confidence /= count
            else:
                avg_confidence = 0.7  # Default if no matching confidences
            
            # Adjust confidence based on vote agreement
            vote_agreement = regime_votes[winning_regime] / sum(regime_votes.values())
            
            # Blend confidence and agreement
            final_confidence = 0.7 * avg_confidence + 0.3 * vote_agreement
            
            return winning_regime, final_confidence
            
        except Exception as e:
            logger.error(f"Error in ensemble classification: {str(e)}")
            # Use the first result as fallback
            return next(iter(results.values()))
    
    def _update_classification_history(self, symbol: str, timeframe: str, 
                                      regime: MarketRegimeType, confidence: float) -> None:
        """
        Update classification history.
        
        Args:
            symbol: Symbol being classified
            timeframe: Timeframe being classified
            regime: Classified regime
            confidence: Classification confidence
        """
        try:
            if symbol not in self.classification_history:
                self.classification_history[symbol] = {}
            
            if timeframe not in self.classification_history[symbol]:
                self.classification_history[symbol][timeframe] = []
            
            # Add to history
            history = self.classification_history[symbol][timeframe]
            history.append((regime, confidence))
            
            # Limit history size
            max_history = 100
            if len(history) > max_history:
                self.classification_history[symbol][timeframe] = history[-max_history:]
                
        except Exception as e:
            logger.error(f"Error updating classification history: {str(e)}")
    
    def get_classification_history(self, symbol: str, timeframe: str, 
                                 limit: Optional[int] = None) -> List[Tuple[MarketRegimeType, float]]:
        """
        Get classification history for a symbol and timeframe.
        
        Args:
            symbol: Symbol to get history for
            timeframe: Timeframe to get history for
            limit: Optional limit for number of entries
            
        Returns:
            List of (regime_type, confidence) tuples
        """
        if symbol not in self.classification_history or timeframe not in self.classification_history[symbol]:
            return []
        
        history = self.classification_history[symbol][timeframe]
        
        if limit is not None:
            return history[-limit:]
        return history
    
    def detect_regime_transition(self, symbol: str, timeframe: str) -> Optional[Tuple[MarketRegimeType, MarketRegimeType, float]]:
        """
        Detect if a regime transition has occurred.
        
        Args:
            symbol: Symbol to check
            timeframe: Timeframe to check
            
        Returns:
            Tuple of (from_regime, to_regime, confidence) or None if no transition
        """
        try:
            if symbol not in self.classification_history or timeframe not in self.classification_history[symbol]:
                return None
            
            history = self.classification_history[symbol][timeframe]
            
            if len(history) < 2:
                return None
            
            # Get last two classifications
            prev_regime, prev_confidence = history[-2]
            current_regime, current_confidence = history[-1]
            
            # Check if regime has changed
            if prev_regime != current_regime:
                # Calculate transition confidence
                # Higher if both classifications have high confidence
                transition_confidence = (prev_confidence + current_confidence) / 2
                
                return prev_regime, current_regime, transition_confidence
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting regime transition: {str(e)}")
            return None
    
    def save_model(self, model_path: Optional[str] = None, 
                  scaler_path: Optional[str] = None) -> bool:
        """
        Save ML model to file.
        
        Args:
            model_path: Optional path to save model to
            scaler_path: Optional path to save scaler to
            
        Returns:
            bool: Success status
        """
        try:
            if self.ml_model is None:
                logger.warning("No ML model to save")
                return False
            
            # Use provided paths or defaults
            model_path = model_path or self.model_path
            scaler_path = scaler_path or self.scaler_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.ml_model, model_path)
            
            # Save scaler if available
            if self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Saved ML model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ML model: {str(e)}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for ML model.
        
        Returns:
            Dict mapping feature names to importance values
        """
        return self.feature_importance
