#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Fusion Strategy - Combines technical, fundamental, and sentiment signals
using an adaptive weighting model.
"""

import os
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from collections import defaultdict
import pickle

# Import base strategy class and other strategies
from trading_bot.strategy.strategy_rotator import Strategy
from trading_bot.strategy.fundamental_strategy import FundamentalStrategy
from trading_bot.strategy.sentiment_strategy import SentimentStrategy
from trading_bot.common.market_types import MarketRegime
from trading_bot.common.config_utils import setup_directories, load_config, save_state, load_state

# Optional imports for machine learning components
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("scikit-learn not available. Advanced ML features disabled.")

# Setup logging
logger = logging.getLogger("HybridFusionStrategy")

class HybridFusionStrategy(Strategy):
    """
    Strategy that combines technical, fundamental and sentiment signals 
    using an adaptive weighting model.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a hybrid fusion strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Setup paths
        self.paths = setup_directories(
            data_dir=config.get("data_dir"),
            component_name=f"hybrid_fusion_{name}"
        )
        
        # Initialize component strategies
        self._initialize_component_strategies(config)
        
        # Weights for signal fusion
        self.technical_weight = config.get("technical_weight", 0.4)
        self.fundamental_weight = config.get("fundamental_weight", 0.3)
        self.sentiment_weight = config.get("sentiment_weight", 0.3)
        
        # Adaptive weights by market regime
        self.regime_weights = config.get("regime_weights", {
            "BULL": {"technical": 0.4, "fundamental": 0.4, "sentiment": 0.2},
            "BEAR": {"technical": 0.5, "fundamental": 0.3, "sentiment": 0.2},
            "SIDEWAYS": {"technical": 0.3, "fundamental": 0.4, "sentiment": 0.3},
            "HIGH_VOL": {"technical": 0.5, "fundamental": 0.2, "sentiment": 0.3},
            "LOW_VOL": {"technical": 0.3, "fundamental": 0.5, "sentiment": 0.2},
            "UNKNOWN": {"technical": 0.4, "fundamental": 0.3, "sentiment": 0.3}
        })
        
        # Current market regime
        self.current_regime = MarketRegime.UNKNOWN
        
        # History of signals and performance
        self.signal_history = []
        self.performance_history = []
        
        # Machine learning model for adaptive fusion (if scikit-learn available)
        self.use_ml_fusion = config.get("use_ml_fusion", False) and ML_AVAILABLE
        self.ml_model = None
        self.scaler = None
        
        if self.use_ml_fusion:
            self._initialize_ml_model(config)
        
        # Load saved model if available
        self._load_model()
    
    def _initialize_component_strategies(self, config: Dict[str, Any]) -> None:
        """
        Initialize component strategies.
        
        Args:
            config: Configuration dictionary
        """
        # Technical strategies are expected to be passed in via the StrategyRotator
        self.technical_strategies = []
        
        # Create fundamental strategy
        fundamental_config = config.get("fundamental_config", {})
        fundamental_type = fundamental_config.get("type", "default")
        
        if fundamental_type == "dcf":
            from trading_bot.strategy.fundamental_strategy import DCFStrategy
            self.fundamental_strategy = DCFStrategy("DCF_Component", fundamental_config)
        elif fundamental_type == "quality_value":
            from trading_bot.strategy.fundamental_strategy import QualityValueStrategy
            self.fundamental_strategy = QualityValueStrategy("QV_Component", fundamental_config)
        elif fundamental_type == "financial_health":
            from trading_bot.strategy.fundamental_strategy import FinancialHealthStrategy
            self.fundamental_strategy = FinancialHealthStrategy("FinHealth_Component", fundamental_config)
        else:
            self.fundamental_strategy = FundamentalStrategy("Fundamental_Component", fundamental_config)
        
        # Create sentiment strategy
        sentiment_config = config.get("sentiment_config", {})
        sentiment_type = sentiment_config.get("type", "default")
        
        if sentiment_type == "news":
            from trading_bot.strategy.sentiment_strategy import NewsSentimentStrategy
            self.sentiment_strategy = NewsSentimentStrategy("News_Component", sentiment_config)
        elif sentiment_type == "sec_filings":
            from trading_bot.strategy.sentiment_strategy import SECFilingStrategy
            self.sentiment_strategy = SECFilingStrategy("SEC_Component", sentiment_config)
        else:
            self.sentiment_strategy = SentimentStrategy("Sentiment_Component", sentiment_config)
        
        logger.info("Initialized component strategies")
    
    def _initialize_ml_model(self, config: Dict[str, Any]) -> None:
        """
        Initialize machine learning model for adaptive fusion.
        
        Args:
            config: Configuration dictionary
        """
        if not ML_AVAILABLE:
            logger.warning("ML fusion requested but scikit-learn not available")
            self.use_ml_fusion = False
            return
        
        ml_config = config.get("ml_config", {})
        model_type = ml_config.get("model_type", "random_forest")
        
        self.scaler = StandardScaler()
        
        if model_type == "random_forest":
            self.ml_model = RandomForestRegressor(
                n_estimators=ml_config.get("n_estimators", 100),
                max_depth=ml_config.get("max_depth", 10),
                random_state=42
            )
        elif model_type == "gradient_boosting":
            self.ml_model = GradientBoostingRegressor(
                n_estimators=ml_config.get("n_estimators", 100),
                max_depth=ml_config.get("max_depth", 3),
                learning_rate=ml_config.get("learning_rate", 0.1),
                random_state=42
            )
        else:
            # Default to linear regression
            self.ml_model = LinearRegression()
        
        logger.info(f"Initialized ML fusion model: {model_type}")
    
    def _save_model(self) -> None:
        """Save the ML model to disk."""
        if not self.use_ml_fusion or self.ml_model is None:
            return
        
        model_path = os.path.join(self.paths["data_dir"], "ml_model.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.ml_model,
                    'scaler': self.scaler
                }, f)
            logger.info(f"Saved ML fusion model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")
    
    def _load_model(self) -> bool:
        """
        Load the ML model from disk.
        
        Returns:
            bool: True if model was loaded successfully
        """
        if not self.use_ml_fusion:
            return False
        
        model_path = os.path.join(self.paths["data_dir"], "ml_model.pkl")
        
        if not os.path.exists(model_path):
            logger.info("No saved ML model found")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
                self.ml_model = saved['model']
                self.scaler = saved['scaler']
            logger.info(f"Loaded ML fusion model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            return False
    
    def update_market_regime(self, regime: MarketRegime) -> None:
        """
        Update the current market regime and adjust weights.
        
        Args:
            regime: New market regime
        """
        self.current_regime = regime
        
        # Update weights based on regime
        if regime.name in self.regime_weights:
            weights = self.regime_weights[regime.name]
            self.technical_weight = weights.get("technical", self.technical_weight)
            self.fundamental_weight = weights.get("fundamental", self.fundamental_weight)
            self.sentiment_weight = weights.get("sentiment", self.sentiment_weight)
            
            logger.info(f"Updated weights for {regime.name} regime: "
                       f"T={self.technical_weight:.2f}, F={self.fundamental_weight:.2f}, "
                       f"S={self.sentiment_weight:.2f}")
    
    def add_technical_strategy(self, strategy: Strategy) -> None:
        """
        Add a technical strategy to the fusion model.
        
        Args:
            strategy: Strategy to add
        """
        self.technical_strategies.append(strategy)
        logger.info(f"Added technical strategy: {strategy.name}")
    
    def generate_technical_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a combined technical signal from all technical strategies.
        
        Args:
            market_data: Market data for signal generation
            
        Returns:
            float: Combined technical signal
        """
        if not self.technical_strategies:
            logger.warning("No technical strategies available")
            return 0.0
        
        # Get signals from all technical strategies
        signals = []
        for strategy in self.technical_strategies:
            try:
                signal = strategy.generate_signal(market_data)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating technical signal from {strategy.name}: {e}")
        
        # Average signals (simple approach)
        if signals:
            return sum(signals) / len(signals)
        else:
            return 0.0
    
    def generate_fundamental_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a fundamental signal.
        
        Args:
            market_data: Market data for signal generation
            
        Returns:
            float: Fundamental signal
        """
        try:
            return self.fundamental_strategy.generate_signal(market_data)
        except Exception as e:
            logger.error(f"Error generating fundamental signal: {e}")
            return 0.0
    
    def generate_sentiment_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a sentiment signal.
        
        Args:
            market_data: Market data for signal generation
            
        Returns:
            float: Sentiment signal
        """
        try:
            return self.sentiment_strategy.generate_signal(market_data)
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return 0.0
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal by combining technical, fundamental, and sentiment signals.
        
        Args:
            market_data: Market data for signal generation
            
        Returns:
            float: Combined signal between -1.0 and 1.0
        """
        # Generate component signals
        technical_signal = self.generate_technical_signal(market_data)
        fundamental_signal = self.generate_fundamental_signal(market_data)
        sentiment_signal = self.generate_sentiment_signal(market_data)
        
        # Store signals
        self.signal_history.append({
            "timestamp": datetime.now().isoformat(),
            "technical": technical_signal,
            "fundamental": fundamental_signal,
            "sentiment": sentiment_signal,
            "symbol": market_data.get("symbol", "UNKNOWN"),
            "regime": self.current_regime.name
        })
        
        # Use ML model for fusion if available and trained
        if self.use_ml_fusion and hasattr(self.ml_model, 'predict') and len(self.signal_history) > 10:
            try:
                signal = self._ml_fusion_signal(technical_signal, fundamental_signal, sentiment_signal, market_data)
                logger.debug(f"ML fusion signal: {signal:.4f}")
            except Exception as e:
                logger.error(f"Error in ML fusion: {e}")
                signal = self._weighted_fusion_signal(technical_signal, fundamental_signal, sentiment_signal)
        else:
            # Fall back to weighted fusion
            signal = self._weighted_fusion_signal(technical_signal, fundamental_signal, sentiment_signal)
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated hybrid fusion signal: {signal:.4f}")
        logger.debug(f"Component signals: T={technical_signal:.4f}, F={fundamental_signal:.4f}, S={sentiment_signal:.4f}")
        
        return signal
    
    def _weighted_fusion_signal(self, technical: float, fundamental: float, sentiment: float) -> float:
        """
        Combine signals using weighted average.
        
        Args:
            technical: Technical signal
            fundamental: Fundamental signal
            sentiment: Sentiment signal
            
        Returns:
            float: Combined signal
        """
        # Normalize weights
        total_weight = self.technical_weight + self.fundamental_weight + self.sentiment_weight
        
        if total_weight == 0:
            return 0.0
        
        t_weight = self.technical_weight / total_weight
        f_weight = self.fundamental_weight / total_weight
        s_weight = self.sentiment_weight / total_weight
        
        # Weighted average
        signal = (
            technical * t_weight +
            fundamental * f_weight +
            sentiment * s_weight
        )
        
        return np.clip(signal, -1.0, 1.0)
    
    def _ml_fusion_signal(self, technical: float, fundamental: float, sentiment: float, 
                         market_data: Dict[str, Any]) -> float:
        """
        Combine signals using ML model.
        
        Args:
            technical: Technical signal
            fundamental: Fundamental signal
            sentiment: Sentiment signal
            market_data: Additional market data
            
        Returns:
            float: Combined signal
        """
        # Extract additional features from market data
        volume = market_data.get("volume", 0)
        price = market_data.get("price", 0)
        
        # Create feature vector
        features = [
            technical, 
            fundamental, 
            sentiment,
            self.current_regime.value,  # Convert enum to int
            volume if volume else 0,
            price if price else 0
        ]
        
        # Add some interaction terms
        features.append(technical * fundamental)  # Interaction between technical and fundamental
        features.append(technical * sentiment)    # Interaction between technical and sentiment
        features.append(fundamental * sentiment)  # Interaction between fundamental and sentiment
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        signal = self.ml_model.predict(features_scaled)[0]
        
        return np.clip(signal, -1.0, 1.0)
    
    def update_performance(self, performance_data: Dict[str, float]) -> None:
        """
        Update performance metrics and potentially retrain ML model.
        
        Args:
            performance_data: Dictionary of performance metrics
        """
        # Store performance data
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": performance_data,
            "regime": self.current_regime.name
        })
        
        # Retrain ML model if we have enough data
        if self.use_ml_fusion and len(self.performance_history) >= 20:
            self._train_ml_model()
    
    def _train_ml_model(self) -> None:
        """Train ML model using historical signals and performance data."""
        if not self.use_ml_fusion or not ML_AVAILABLE:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            # Match signal history with subsequent performance
            max_lookback = min(len(self.signal_history), len(self.performance_history))
            
            for i in range(max_lookback - 1):
                signal = self.signal_history[i]
                next_perf = self.performance_history[i + 1]
                
                # Create feature vector
                features = [
                    signal["technical"],
                    signal["fundamental"],
                    signal["sentiment"],
                    MarketRegime[signal["regime"]].value,  # Convert string to enum to int
                    # Add more features here as needed
                ]
                
                # Add interaction terms
                features.append(signal["technical"] * signal["fundamental"])
                features.append(signal["technical"] * signal["sentiment"])
                features.append(signal["fundamental"] * signal["sentiment"])
                
                X.append(features)
                y.append(next_perf["performance"].get("return", 0.0))
            
            if len(X) < 10:
                logger.info("Not enough data to train ML model")
                return
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler().fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.ml_model.score(X_train_scaled, y_train)
            test_score = self.ml_model.score(X_test_scaled, y_test)
            
            logger.info(f"Trained ML model: Train R^2={train_score:.4f}, Test R^2={test_score:.4f}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        base_dict = super().to_dict()
        
        # Add HybridFusion-specific fields
        base_dict.update({
            "technical_weight": self.technical_weight,
            "fundamental_weight": self.fundamental_weight,
            "sentiment_weight": self.sentiment_weight,
            "current_regime": self.current_regime.name,
            "signal_history": self.signal_history[-100:] if self.signal_history else [],
            "performance_history": self.performance_history[-100:] if self.performance_history else [],
            "use_ml_fusion": self.use_ml_fusion
        })
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HybridFusionStrategy':
        """Create strategy from dictionary."""
        strategy = super().from_dict(data)
        
        # Restore HybridFusion-specific fields
        strategy.technical_weight = data.get("technical_weight", strategy.technical_weight)
        strategy.fundamental_weight = data.get("fundamental_weight", strategy.fundamental_weight)
        strategy.sentiment_weight = data.get("sentiment_weight", strategy.sentiment_weight)
        strategy.current_regime = MarketRegime[data.get("current_regime", "UNKNOWN")]
        strategy.signal_history = data.get("signal_history", [])
        strategy.performance_history = data.get("performance_history", [])
        strategy.use_ml_fusion = data.get("use_ml_fusion", False)
        
        # Load ML model if needed
        if strategy.use_ml_fusion:
            strategy._load_model()
        
        return strategy


class AdaptiveTimeframeStrategy(HybridFusionStrategy):
    """
    Advanced hybrid strategy that adapts to different timeframes,
    giving more weight to fundamental factors for longer-term decisions
    and more weight to technical/sentiment for short-term decisions.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Adaptive Timeframe strategy"""
        super().__init__(name, config)
        
        # Configure timeframe-specific weights
        self.timeframe_weights = config.get("timeframe_weights", {
            "short_term": {
                "technical": 0.6,
                "fundamental": 0.1,
                "sentiment": 0.3
            },
            "medium_term": {
                "technical": 0.4,
                "fundamental": 0.4,
                "sentiment": 0.2
            },
            "long_term": {
                "technical": 0.2,
                "fundamental": 0.7,
                "sentiment": 0.1
            }
        })
        
        # Current timeframe focus
        self.current_timeframe = config.get("default_timeframe", "medium_term")
    
    def set_timeframe(self, timeframe: str) -> None:
        """
        Set the trading timeframe focus.
        
        Args:
            timeframe: 'short_term', 'medium_term', or 'long_term'
        """
        if timeframe not in self.timeframe_weights:
            logger.warning(f"Invalid timeframe: {timeframe}")
            return
        
        self.current_timeframe = timeframe
        
        # Update weights based on timeframe
        weights = self.timeframe_weights[timeframe]
        self.technical_weight = weights.get("technical", self.technical_weight)
        self.fundamental_weight = weights.get("fundamental", self.fundamental_weight)
        self.sentiment_weight = weights.get("sentiment", self.sentiment_weight)
        
        logger.info(f"Updated weights for {timeframe}: "
                   f"T={self.technical_weight:.2f}, F={self.fundamental_weight:.2f}, "
                   f"S={self.sentiment_weight:.2f}")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate trading signal with timeframe-specific weighting.
        
        Args:
            market_data: Market data
            
        Returns:
            float: Trading signal
        """
        # Check if timeframe is specified in market data
        if "timeframe" in market_data:
            self.set_timeframe(market_data["timeframe"])
        
        # Generate signal using parent method
        return super().generate_signal(market_data)


class BuffettMomentumHybridStrategy(HybridFusionStrategy):
    """
    Specialized hybrid strategy combining Buffett-style fundamentals with 
    momentum and sentiment signals. Particularly effective for long-term 
    value investing with short to medium-term tactical allocations.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Buffett-Momentum Hybrid strategy"""
        super().__init__(name, config)
        
        # Configure the specialized components
        # Override the default weighting to emphasize fundamentals
        self.base_fundamental_weight = config.get("base_fundamental_weight", 0.6)
        self.momentum_weight = config.get("momentum_weight", 0.25)
        self.sentiment_weight = config.get("sentiment_weight", 0.15)
        
        # Apply the base weights
        self.technical_weight = self.momentum_weight
        self.fundamental_weight = self.base_fundamental_weight
        
        # Value metrics that determine when to override momentum
        self.value_threshold = config.get("value_threshold", 0.4)  # Strong fundamental signal
        self.momentum_override_threshold = config.get("momentum_override_threshold", 0.7)  # Strong momentum signal
        
        # Use Quality-Value strategy for fundamentals by default
        from trading_bot.strategy.fundamental_strategy import QualityValueStrategy
        fundamental_config = config.get("fundamental_config", {})
        self.fundamental_strategy = QualityValueStrategy("BuffettQV_Component", fundamental_config)
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate trading signal with Buffett-style decision making.
        
        Args:
            market_data: Market data
            
        Returns:
            float: Trading signal
        """
        # Generate component signals
        technical_signal = self.generate_technical_signal(market_data)
        fundamental_signal = self.generate_fundamental_signal(market_data)
        sentiment_signal = self.generate_sentiment_signal(market_data)
        
        # Store signals in history
        self.signal_history.append({
            "timestamp": datetime.now().isoformat(),
            "technical": technical_signal,
            "fundamental": fundamental_signal,
            "sentiment": sentiment_signal,
            "symbol": market_data.get("symbol", "UNKNOWN"),
            "regime": self.current_regime.name
        })
        
        # Buffett-style decision logic:
        # 1. If fundamentals are very strong (high value), overweight them significantly
        # 2. If fundamentals are very weak (low value), be more conservative
        # 3. If momentum is strong in either direction, consider it more
        
        # Adjust weights based on signal strength
        adjusted_weights = {
            "technical": self.technical_weight,
            "fundamental": self.fundamental_weight,
            "sentiment": self.sentiment_weight
        }
        
        # Strong fundamental signal: increase fundamental weight
        if abs(fundamental_signal) > self.value_threshold:
            # Increase fundamental weight, decrease others proportionally
            strength_factor = min(abs(fundamental_signal) * 1.5, 1.0)
            adjusted_weights["fundamental"] += (1 - adjusted_weights["fundamental"]) * strength_factor * 0.5
            
            # Reduce other weights proportionally
            reduction_factor = (1 - adjusted_weights["fundamental"]) / (adjusted_weights["technical"] + adjusted_weights["sentiment"])
            adjusted_weights["technical"] *= reduction_factor
            adjusted_weights["sentiment"] *= reduction_factor
            
            logger.debug(f"Strong fundamental signal ({fundamental_signal:.2f}): "
                       f"Adjusting weights: F={adjusted_weights['fundamental']:.2f}, "
                       f"T={adjusted_weights['technical']:.2f}, S={adjusted_weights['sentiment']:.2f}")
        
        # Strong momentum signal that contradicts fundamentals: increase momentum consideration
        if abs(technical_signal) > self.momentum_override_threshold and np.sign(technical_signal) != np.sign(fundamental_signal):
            # Increase technical weight, but not as much as we would for fundamentals
            strength_factor = min(abs(technical_signal) * 1.2, 1.0)
            weight_shift = min(adjusted_weights["fundamental"] * 0.3, 0.2) * strength_factor
            
            adjusted_weights["technical"] += weight_shift
            adjusted_weights["fundamental"] -= weight_shift
            
            logger.debug(f"Strong momentum signal contradicting fundamentals: "
                       f"Adjusting weights: F={adjusted_weights['fundamental']:.2f}, "
                       f"T={adjusted_weights['technical']:.2f}")
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        for key in adjusted_weights:
            adjusted_weights[key] /= total_weight
        
        # Calculate weighted signal
        signal = (
            technical_signal * adjusted_weights["technical"] +
            fundamental_signal * adjusted_weights["fundamental"] +
            sentiment_signal * adjusted_weights["sentiment"]
        )
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"Generated Buffett-Momentum signal: {signal:.4f}")
        logger.debug(f"Component signals: T={technical_signal:.2f}, F={fundamental_signal:.2f}, S={sentiment_signal:.2f}")
        logger.debug(f"Used weights: T={adjusted_weights['technical']:.2f}, F={adjusted_weights['fundamental']:.2f}, S={adjusted_weights['sentiment']:.2f}")
        
        return signal


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a hybrid fusion strategy
    config = {
        "technical_weight": 0.4,
        "fundamental_weight": 0.4,
        "sentiment_weight": 0.2,
        "use_ml_fusion": False,  # Set to True for ML-based fusion
    }
    
    hybrid_strategy = HybridFusionStrategy("HybridFusion", config)
    
    # Create a technical strategy for testing
    from trading_bot.strategy.strategy_rotator import MomentumStrategy
    mom_strategy = MomentumStrategy("MomentumStrategy", {"fast_period": 5, "slow_period": 20})
    hybrid_strategy.add_technical_strategy(mom_strategy)
    
    # Create mock market data
    market_data = {
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 1000000,
        "prices": [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
        "volumes": [900000, 950000, 980000, 1020000, 990000, 1000000]
    }
    
    # Generate signal
    signal = hybrid_strategy.generate_signal(market_data)
    print(f"Hybrid fusion signal for AAPL: {signal:.4f}")
    
    # Try adaptive timeframe strategy
    adaptive_strategy = AdaptiveTimeframeStrategy("AdaptiveTimeframe")
    adaptive_strategy.add_technical_strategy(mom_strategy)
    
    # Generate signals with different timeframes
    for timeframe in ["short_term", "medium_term", "long_term"]:
        market_data["timeframe"] = timeframe
        signal = adaptive_strategy.generate_signal(market_data)
        print(f"{timeframe} signal: {signal:.4f}")
    
    # Try Buffett-Momentum strategy
    buffett_strategy = BuffettMomentumHybridStrategy("BuffettMomentum")
    buffett_strategy.add_technical_strategy(mom_strategy)
    signal = buffett_strategy.generate_signal(market_data)
    print(f"Buffett-Momentum signal: {signal:.4f}") 