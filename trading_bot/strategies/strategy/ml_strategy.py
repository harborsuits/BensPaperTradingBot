#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-based trading strategy implementation.
Uses machine learning models for prediction and signal generation.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from trading_bot.strategy.strategy_rotator import Strategy
from trading_bot.data.models import MarketData, TimeFrame
from trading_bot.ml.base_model import BaseMLModel
from trading_bot.ml.ensemble_model import EnsembleModel
from trading_bot.data.features.base_feature import FeatureExtractor

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """
    Trading strategy based on machine learning models.
    
    Generates trading signals using predictions from machine learning models.
    Can use individual models or ensembles for signal generation.
    """
    
    def __init__(
        self,
        name: str,
        model: BaseMLModel,
        feature_extractor: Optional[FeatureExtractor] = None,
        signal_threshold: float = 0.0,
        signal_scaling: float = 1.0,
        lookback_periods: int = 20,
        confidence_threshold: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ML-based strategy.
        
        Args:
            name: Strategy name
            model: Machine learning model to use for predictions
            feature_extractor: Optional feature extractor for preprocessing
            signal_threshold: Threshold for considering a signal significant
            signal_scaling: Scaling factor for raw model predictions
            lookback_periods: Number of periods to include in feature calculation
            confidence_threshold: Minimum confidence required for a signal
            config: Strategy configuration
        """
        super().__init__(name, config or {})
        
        self.model = model
        self.feature_extractor = feature_extractor
        self.signal_threshold = signal_threshold
        self.signal_scaling = signal_scaling
        self.lookback_periods = lookback_periods
        self.confidence_threshold = confidence_threshold
        
        # Additional state information
        self.last_prediction = None
        self.last_prediction_confidence = None
        self.last_features = None
        self.prediction_history = []
    
    def _prepare_features(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for the model from market data.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            DataFrame with prepared features
        """
        # Check if we have price data
        if "prices" not in market_data or len(market_data["prices"]) < self.lookback_periods:
            logger.warning(f"Insufficient price data for {self.name}. Need at least {self.lookback_periods} periods.")
            return pd.DataFrame()
            
        # Basic DataFrame with OHLCV data
        data = {}
        
        # Extract price data
        if "prices" in market_data:
            data["close"] = market_data["prices"]
            
        # Extract other OHLCV data if available
        for col in ["open", "high", "low", "volume"]:
            if col in market_data:
                data[col] = market_data[col]
                
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add timestamp column if available
        if "timestamps" in market_data:
            df["timestamp"] = market_data["timestamps"]
            df.set_index("timestamp", inplace=True)
            
        # Apply feature extractor if available
        if self.feature_extractor is not None:
            df = self.feature_extractor.extract_features(df)
            
        # Additional data from market_data dict
        for key, value in market_data.items():
            if key not in ["prices", "timestamps", "open", "high", "low", "close", "volume"] and isinstance(value, (list, np.ndarray)):
                # Check if length matches
                if len(value) == len(df):
                    df[key] = value
                    
        return df
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal between -1.0 and 1.0 using ML model.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        # Prepare features
        features_df = self._prepare_features(market_data)
        
        if features_df.empty:
            logger.warning(f"No features available for {self.name}. Returning neutral signal.")
            return 0.0
            
        try:
            # Store features for explanation
            self.last_features = features_df
            
            # Generate prediction
            prediction = self.model.predict(features_df)
            
            if len(prediction) == 0:
                logger.warning(f"Model {self.model.name} returned empty prediction. Returning neutral signal.")
                return 0.0
                
            # Use the most recent prediction
            latest_prediction = prediction[-1]
            
            # Transform prediction to signal
            signal = self._prediction_to_signal(latest_prediction)
            
            # Store prediction
            self.last_prediction = latest_prediction
            self.prediction_history.append({
                "timestamp": datetime.now().isoformat(),
                "prediction": float(latest_prediction),
                "signal": float(signal)
            })
            
            # Limit history length
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
                
            # Update last signal and time
            self.last_signal = signal
            self.last_update_time = datetime.now()
            
            logger.debug(f"Generated signal {signal:.4f} from prediction {latest_prediction:.4f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return 0.0
    
    def _prediction_to_signal(self, prediction: float) -> float:
        """
        Convert a model prediction to a trading signal.
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Trading signal between -1.0 and 1.0
        """
        # Apply threshold and scaling
        if abs(prediction) < self.signal_threshold:
            return 0.0
            
        # Scale prediction to signal range
        signal = prediction * self.signal_scaling
        
        # Clip to valid range
        return np.clip(signal, -1.0, 1.0)
    
    def get_confidence(self) -> float:
        """
        Get confidence score for the latest prediction.
        
        For ensemble models, this can be derived from model agreement.
        For single models, use a simple heuristic based on signal strength.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.last_prediction is None:
            return 0.0
            
        if isinstance(self.model, EnsembleModel):
            # For ensemble models, get agreement score
            if hasattr(self.model, 'get_model_weights') and hasattr(self.model, 'get_model_performance'):
                weights = self.model.get_model_weights()
                # Higher confidence when good models agree
                return min(1.0, sum(weights.values()) * abs(self.last_signal))
        
        # For single models, use signal strength as a proxy for confidence
        return abs(self.last_signal)
    
    def explain_signal(self) -> Dict[str, Any]:
        """
        Provide an explanation for the latest signal.
        
        Returns:
            Dictionary with signal explanation
        """
        if self.last_prediction is None or self.last_features is None:
            return {"error": "No prediction available"}
            
        explanation = {
            "prediction": float(self.last_prediction),
            "signal": float(self.last_signal),
            "confidence": float(self.get_confidence()),
        }
        
        # Get feature importance
        try:
            # For ensemble models
            if isinstance(self.model, EnsembleModel):
                explanation["model_weights"] = self.model.get_model_weights()
                
            # Get feature contributions if available
            if hasattr(self.model, 'explain_prediction'):
                if self.last_features is not None:
                    feature_contrib = self.model.explain_prediction(self.last_features)
                    explanation["feature_contributions"] = feature_contrib
        except Exception as e:
            logger.error(f"Error explaining signal: {str(e)}")
            
        return explanation


class DeepLearningStrategy(MLStrategy):
    """
    Trading strategy based on deep learning models (LSTM, CNN, Transformer).
    
    Specialization of MLStrategy for deep learning models with additional
    functionality for sequence prediction and uncertainty estimation.
    """
    
    def __init__(
        self,
        name: str,
        model: BaseMLModel,
        feature_extractor: Optional[FeatureExtractor] = None,
        signal_threshold: float = 0.0,
        signal_scaling: float = 1.0,
        lookback_periods: int = 60,  # Longer lookback for sequence models
        confidence_threshold: float = 0.0,
        sequence_length: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the deep learning strategy.
        
        Args:
            name: Strategy name
            model: Deep learning model (LSTM, CNN, Transformer)
            feature_extractor: Feature extractor for preprocessing
            signal_threshold: Threshold for considering a signal significant
            signal_scaling: Scaling factor for raw model predictions
            lookback_periods: Number of periods to include in feature calculation
            confidence_threshold: Minimum confidence required for a signal
            sequence_length: Length of sequences for sequence models
            config: Strategy configuration
        """
        super().__init__(
            name, model, feature_extractor, signal_threshold,
            signal_scaling, lookback_periods, confidence_threshold, config
        )
        
        self.sequence_length = sequence_length
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal using deep learning model.
        
        This overrides the base method to handle sequence models differently.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        # Prepare features (same as base class)
        features_df = self._prepare_features(market_data)
        
        if features_df.empty:
            logger.warning(f"No features available for {self.name}. Returning neutral signal.")
            return 0.0
        
        try:
            # Store features for explanation
            self.last_features = features_df
            
            # For deep learning models, we might need a different approach
            # depending on whether they expect sequences or not
            prediction = self.model.predict(features_df)
            
            if len(prediction) == 0:
                logger.warning(f"Model {self.model.name} returned empty prediction. Returning neutral signal.")
                return 0.0
                
            # Use the most recent prediction
            latest_prediction = prediction[-1]
            
            # For deep learning models, estimate uncertainty
            # (could be done through dropout or ensemble predictions)
            self.last_prediction_confidence = self.get_confidence()
            
            # Apply confidence threshold
            if self.last_prediction_confidence < self.confidence_threshold:
                logger.debug(f"Prediction confidence {self.last_prediction_confidence:.4f} below threshold {self.confidence_threshold:.4f}. Returning neutral signal.")
                signal = 0.0
            else:
                # Transform prediction to signal
                signal = self._prediction_to_signal(latest_prediction)
            
            # Store prediction
            self.last_prediction = latest_prediction
            self.prediction_history.append({
                "timestamp": datetime.now().isoformat(),
                "prediction": float(latest_prediction),
                "signal": float(signal),
                "confidence": float(self.last_prediction_confidence)
            })
            
            # Limit history length
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
                
            # Update last signal and time
            self.last_signal = signal
            self.last_update_time = datetime.now()
            
            logger.debug(f"Generated signal {signal:.4f} from prediction {latest_prediction:.4f} with confidence {self.last_prediction_confidence:.4f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return 0.0
    
    def get_confidence(self) -> float:
        """
        Get confidence score for deep learning prediction.
        
        For deep learning models, confidence can be estimated using:
        - Multiple forward passes with dropout (Monte Carlo Dropout)
        - Ensemble agreement for deep ensemble models
        - Direct uncertainty estimate if the model provides it
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # If we already calculated confidence, return it
        if self.last_prediction_confidence is not None:
            return self.last_prediction_confidence
            
        # For ensemble models, use base class implementation
        if isinstance(self.model, EnsembleModel):
            return super().get_confidence()
            
        # For single deep learning models, use signal strength as a proxy
        # In a real implementation, you could use Monte Carlo Dropout
        # or other uncertainty estimation techniques
        if self.last_prediction is not None:
            # Use signal strength, but with lower confidence than ensemble models
            return 0.7 * abs(self.last_signal)
            
        return 0.0 