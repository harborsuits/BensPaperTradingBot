#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-Enhanced Trading Integration Module

This module integrates ML-based signal generation with RL-driven position sizing
to create a complete ML-enhanced trading system.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.ml.enhanced_features import EnhancedFeatureEngineering
from trading_bot.ml.signal_model import SignalModel, create_signal_model
from trading_bot.ml.market_regime_autoencoder import MarketRegimeAutoencoder, create_market_regime_autoencoder
from trading_bot.rl.position_sizer_agent import PositionSizerAgent, create_position_sizer_agent

logger = logging.getLogger(__name__)

class MLEnhancedTradingSystem:
    """
    Integrated ML-Enhanced Trading System
    
    This class combines ML signal generation with RL position sizing to create
    a complete ML-enhanced trading system that can be used by any strategy.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ML-Enhanced Trading System
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._set_default_config()
        
        # Initialize components
        self.feature_engine = None
        self.signal_model = None
        self.regime_detector = None
        self.position_sizer = None
        
        # Initialize tracking
        self.last_update_time = None
        self.current_signals = {}
        self.current_regimes = {}
        self.signal_history = {}
        self.regime_history = {}
        self.sizing_history = {}
        
        # Status flags
        self.initialized = False
        self.models_loaded = False
        
        logger.info("ML-Enhanced Trading System initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # System parameters
        self.config.setdefault("system_name", "ml_enhanced_trading")
        self.config.setdefault("models_dir", "models")
        self.config.setdefault("data_cache_dir", "data/cache")
        
        # Feature engineering config
        self.config.setdefault("feature_config", {})
        
        # Signal model config
        self.config.setdefault("signal_model_config", {
            "prediction_horizons": [1, 3, 5],
            "ensemble_weights": {1: 0.4, 3: 0.4, 5: 0.2},
            "model_dir": os.path.join("models", "signals")
        })
        
        # Regime detector config
        self.config.setdefault("regime_detector_config", {
            "encoding_dim": 3,
            "n_regimes": 5,
            "model_dir": os.path.join("models", "autoencoder")
        })
        
        # Position sizer config
        self.config.setdefault("position_sizer_config", {
            "max_position_size": 0.25,
            "model_dir": os.path.join("models", "position_sizer")
        })
        
        # Integration parameters
        self.config.setdefault("min_confidence_threshold", 0.55)
        self.config.setdefault("position_constraints", {
            "max_total_allocation": 0.9,        # Maximum total allocation to all positions
            "max_correlated_allocation": 0.5,   # Maximum allocation to correlated assets
            "regime_adjustments": {             # Position size adjustments by regime
                0: 1.0,  # Normal
                1: 0.8,  # High volatility
                2: 0.6,  # Crisis
                3: 1.2,  # Strong trend
                4: 0.4   # Choppy market
            }
        })
        
        # Operation modes
        self.config.setdefault("use_ml_signals", True)
        self.config.setdefault("use_regime_detection", True)
        self.config.setdefault("use_rl_sizing", True)
    
    def initialize(self, load_models: bool = True):
        """
        Initialize the system and optionally load models
        
        Args:
            load_models: Whether to load pre-trained models
        """
        logger.info("Initializing ML-Enhanced Trading System")
        
        # Initialize feature engineering
        self.feature_engine = EnhancedFeatureEngineering(self.config["feature_config"])
        
        # Initialize signal model
        self.signal_model = create_signal_model(self.config["signal_model_config"])
        
        # Initialize regime detector
        self.regime_detector = create_market_regime_autoencoder(self.config["regime_detector_config"])
        
        # Initialize position sizer
        self.position_sizer = create_position_sizer_agent(self.config["position_sizer_config"])
        
        # Create necessary directories
        os.makedirs(self.config["models_dir"], exist_ok=True)
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        
        self.initialized = True
        
        # Load models if requested
        if load_models:
            self.load_models()
        
        logger.info("ML-Enhanced Trading System initialization complete")
    
    def load_models(self) -> bool:
        """
        Load pre-trained models
        
        Returns:
            Success flag
        """
        if not self.initialized:
            self.initialize(load_models=False)
        
        logger.info("Loading models for ML-Enhanced Trading System")
        
        success = True
        
        # Load signal model
        try:
            for horizon in self.config["signal_model_config"]["prediction_horizons"]:
                if not self.signal_model.load_model(horizon):
                    logger.warning(f"Failed to load signal model for horizon {horizon}")
                    success = False
        except Exception as e:
            logger.error(f"Error loading signal models: {e}")
            success = False
        
        # Load regime detector
        try:
            if not self.regime_detector.load_model():
                logger.warning("Failed to load regime detector model")
                success = False
        except Exception as e:
            logger.error(f"Error loading regime detector: {e}")
            success = False
        
        # Load position sizer
        try:
            model_dir = self.config["position_sizer_config"]["model_dir"]
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith("position_sizer_") and f.endswith(".zip")]
            
            if model_files:
                # Load latest model
                model_files.sort(reverse=True)
                latest_model = os.path.join(model_dir, model_files[0])
                
                if not self.position_sizer.load(latest_model):
                    logger.warning("Failed to load position sizer model")
                    success = False
            else:
                logger.warning("No position sizer models found")
                success = False
        except Exception as e:
            logger.error(f"Error loading position sizer: {e}")
            success = False
        
        self.models_loaded = success
        logger.info(f"Model loading {'successful' if success else 'failed'}")
        
        return success
    
    def update(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Update the system with new market data
        
        Args:
            market_data: Dictionary of market data frames for each symbol
            
        Returns:
            Dictionary with update results
        """
        if not self.initialized:
            self.initialize()
        
        logger.info("Updating ML-Enhanced Trading System")
        update_time = datetime.now()
        
        # Process each symbol
        results = {}
        for symbol, data in market_data.items():
            # Generate signals
            signals = self._generate_signals(symbol, data)
            
            # Detect regimes
            regimes = self._detect_regimes(symbol, data)
            
            # Store results
            symbol_results = {
                "signals": signals,
                "regimes": regimes
            }
            
            results[symbol] = symbol_results
            
            # Update current state
            self.current_signals[symbol] = signals.iloc[-1] if not signals.empty else None
            self.current_regimes[symbol] = regimes.iloc[-1] if not regimes.empty else None
            
            # Update history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
                
            self.signal_history[symbol].append(self.current_signals[symbol])
            self.regime_history[symbol].append(self.current_regimes[symbol])
        
        self.last_update_time = update_time
        
        # Return update summary
        return {
            "update_time": update_time,
            "num_symbols": len(market_data),
            "symbols_updated": list(market_data.keys()),
            "results": results
        }
    
    def _generate_signals(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for a symbol
        
        Args:
            symbol: Market symbol
            data: OHLCV data frame
            
        Returns:
            DataFrame with signal data
        """
        if not self.config["use_ml_signals"]:
            # Return empty signals if ML signals disabled
            return pd.DataFrame()
        
        try:
            # Get ensemble prediction
            predictions = self.signal_model.predict(data)
            
            if not predictions:
                logger.warning(f"No predictions generated for {symbol}")
                return pd.DataFrame()
            
            # Create ensemble signal
            weights = self.config["signal_model_config"]["ensemble_weights"]
            ensemble = self.signal_model.generate_ensemble_signal(data, weights)
            
            if ensemble.empty:
                logger.warning(f"Empty ensemble signal for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Generated signals for {symbol}")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return pd.DataFrame()
    
    def _detect_regimes(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes for a symbol
        
        Args:
            symbol: Market symbol
            data: OHLCV data frame
            
        Returns:
            DataFrame with regime data
        """
        if not self.config["use_regime_detection"]:
            # Return empty regimes if regime detection disabled
            return pd.DataFrame()
        
        try:
            # Detect anomalies and regimes
            regimes = self.regime_detector.detect_market_regime(data)
            
            if regimes.empty:
                logger.warning(f"No regimes detected for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Detected regimes for {symbol}")
            return regimes
            
        except Exception as e:
            logger.error(f"Error detecting regimes for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_trade_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current trade signal for a symbol
        
        Args:
            symbol: Market symbol
            
        Returns:
            Dictionary with signal data
        """
        if symbol not in self.current_signals or self.current_signals[symbol] is None:
            return {
                "symbol": symbol,
                "signal": "neutral",
                "confidence": 0.5,
                "probability": 0.5,
                "available": False
            }
        
        signal = self.current_signals[symbol]
        
        return {
            "symbol": symbol,
            "signal": signal.get("ensemble_signal", "neutral"),
            "confidence": float(signal.get("weighted_probability", 0.5)),
            "probability": float(signal.get("weighted_probability", 0.5)),
            "available": True,
            "horizons": signal.get("component_horizons", [[]])[0],
            "component_signals": signal.get("component_signals", [[]])[0]
        }
    
    def get_market_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current market regime for a symbol
        
        Args:
            symbol: Market symbol
            
        Returns:
            Dictionary with regime data
        """
        if symbol not in self.current_regimes or self.current_regimes[symbol] is None:
            return {
                "symbol": symbol,
                "regime": 0,
                "regime_name": "unknown",
                "volatility": 0.01,
                "available": False
            }
        
        regime = self.current_regimes[symbol]
        
        # Map regime numbers to names
        regime_names = {
            0: "normal",
            1: "high_volatility",
            2: "crisis",
            3: "strong_trend",
            4: "choppy"
        }
        
        regime_num = int(regime.get("rolling_regime", 0))
        regime_name = regime_names.get(regime_num, "unknown")
        
        return {
            "symbol": symbol,
            "regime": regime_num,
            "regime_name": regime_name,
            "rolling_regime": int(regime.get("rolling_regime", 0)),
            "raw_regime": int(regime.get("regime", 0)),
            "available": True
        }
    
    def get_position_size(self, 
                         symbol: str, 
                         account_state: Dict[str, float],
                         risk_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get recommended position size for a symbol
        
        Args:
            symbol: Market symbol
            account_state: Account metrics (balance, equity, etc.)
            risk_metrics: Optional risk metrics
            
        Returns:
            Dictionary with position sizing data
        """
        if not self.config["use_rl_sizing"]:
            # Return default position size if RL sizing disabled
            return {
                "symbol": symbol,
                "position_size_pct": 0.01,
                "position_size_currency": account_state.get("equity", 0) * 0.01,
                "method": "default",
                "available": False
            }
        
        # Get signal and regime data
        signal_data = self.get_trade_signal(symbol)
        regime_data = self.get_market_regime(symbol)
        
        if not signal_data["available"]:
            return {
                "symbol": symbol,
                "position_size_pct": 0.0,
                "position_size_currency": 0.0,
                "method": "no_signal",
                "available": False
            }
        
        # Check if signal meets minimum confidence threshold
        confidence = signal_data["confidence"]
        min_threshold = self.config["min_confidence_threshold"]
        
        if confidence < min_threshold:
            return {
                "symbol": symbol,
                "position_size_pct": 0.0,
                "position_size_currency": 0.0,
                "confidence": confidence,
                "min_threshold": min_threshold,
                "method": "below_threshold",
                "available": True
            }
        
        try:
            # Get position size from RL agent
            if not self.position_sizer or not hasattr(self.position_sizer, "get_position_size"):
                raise ValueError("Position sizer not initialized or trained")
            
            # Get RL recommendation
            sizing = self.position_sizer.get_position_size(
                signal_confidence=confidence,
                market_regime=regime_data["regime"],
                account_state=account_state,
                risk_metrics=risk_metrics
            )
            
            # Apply regime adjustment
            regime_adjustment = self.config["position_constraints"]["regime_adjustments"].get(
                regime_data["regime"], 1.0
            )
            
            adjusted_size = sizing["position_size_pct"] * regime_adjustment
            
            # Create result
            result = {
                "symbol": symbol,
                "position_size_pct": adjusted_size,
                "position_size_currency": account_state.get("equity", 0) * adjusted_size,
                "raw_size_pct": sizing["position_size_pct"],
                "regime_adjustment": regime_adjustment,
                "confidence": confidence,
                "regime": regime_data["regime"],
                "regime_name": regime_data["regime_name"],
                "method": "rl_agent",
                "available": True
            }
            
            # Store in history
            if symbol not in self.sizing_history:
                self.sizing_history[symbol] = []
            self.sizing_history[symbol].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error determining position size for {symbol}: {e}")
            
            # Fallback to simple sizing
            confidence_adjustment = (confidence - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
            base_size = 0.01  # 1% base position
            size = base_size * (1 + confidence_adjustment)
            
            return {
                "symbol": symbol,
                "position_size_pct": size,
                "position_size_currency": account_state.get("equity", 0) * size,
                "confidence": confidence,
                "method": "fallback",
                "available": True,
                "error": str(e)
            }
    
    def optimize_portfolio(self,
                          signals: Dict[str, Dict[str, Any]],
                          account_state: Dict[str, float],
                          constraints: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Optimize portfolio allocations based on signals and constraints
        
        Args:
            signals: Dictionary of signal data for each symbol
            account_state: Account metrics
            constraints: Optional additional constraints
            
        Returns:
            Dictionary of optimized positions
        """
        # Get position sizes for each symbol
        positions = {}
        for symbol, signal in signals.items():
            position = self.get_position_size(symbol, account_state)
            positions[symbol] = position
        
        # Apply constraints
        total_allocation = sum(p["position_size_pct"] for p in positions.values())
        max_allocation = self.config["position_constraints"]["max_total_allocation"]
        
        # Scale down if exceeding max allocation
        if total_allocation > max_allocation and total_allocation > 0:
            scale_factor = max_allocation / total_allocation
            
            for symbol in positions:
                positions[symbol]["position_size_pct"] *= scale_factor
                positions[symbol]["position_size_currency"] = account_state.get("equity", 0) * positions[symbol]["position_size_pct"]
                positions[symbol]["scaled"] = True
                positions[symbol]["scale_factor"] = scale_factor
        
        return positions
    
    def save_state(self, filepath: Optional[str] = None) -> str:
        """
        Save current system state
        
        Args:
            filepath: Optional path to save state
            
        Returns:
            Path where state was saved
        """
        if filepath is None:
            os.makedirs(self.config["data_cache_dir"], exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.config["data_cache_dir"], 
                f"{self.config['system_name']}_state_{timestamp}.json"
            )
        
        # Create state dictionary
        state = {
            "timestamp": str(datetime.now()),
            "config": self.config,
            "current_signals": {},
            "current_regimes": {}
        }
        
        # Add current signals and regimes (convert to serializable format)
        for symbol, signal in self.current_signals.items():
            if signal is not None:
                state["current_signals"][symbol] = signal.to_dict() if hasattr(signal, 'to_dict') else signal
        
        for symbol, regime in self.current_regimes.items():
            if regime is not None:
                state["current_regimes"][symbol] = regime.to_dict() if hasattr(regime, 'to_dict') else regime
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved system state to {filepath}")
        return filepath
    
    def load_state(self, filepath: str) -> bool:
        """
        Load system state from file
        
        Args:
            filepath: Path to state file
            
        Returns:
            Success flag
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Update config
            if "config" in state:
                self.config.update(state["config"])
            
            # Initialize if needed
            if not self.initialized:
                self.initialize()
            
            # Load signals and regimes
            if "current_signals" in state:
                self.current_signals = state["current_signals"]
            
            if "current_regimes" in state:
                self.current_regimes = state["current_regimes"]
            
            logger.info(f"Loaded system state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            return False


# Utility function to create a complete ML-enhanced trading system
def create_ml_enhanced_trading_system(config: Dict[str, Any] = None) -> MLEnhancedTradingSystem:
    """
    Create a complete ML-enhanced trading system
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized MLEnhancedTradingSystem
    """
    system = MLEnhancedTradingSystem(config)
    system.initialize()
    return system
