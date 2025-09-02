#!/usr/bin/env python3
"""
Machine Learning Integration Module

This module provides classes and functions to integrate machine learning components 
with the multi-asset trading system. It handles:

1. ML model management and lifecycle
2. Feature engineering pipelines for different ML models
3. Integration with trading strategies and risk management
4. Real-time prediction serving
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import threading
import time
import pickle

# Import ML components
from ml.market_anomaly_detector import MarketAnomalyDetector
from ml.price_prediction_model import PricePredictionModel
from ml.market_condition_classifier import MarketConditionClassifier
from ml.parameter_optimizer import ParameterOptimizer

# Import trading components
from multi_asset_adapter import MultiAssetAdapter
from risk_manager import RiskManager
from utils.config_loader import ConfigLoader
from utils.signal_emitter import SignalEmitter

logger = logging.getLogger(__name__)

class MLModelManager:
    """
    Manages the lifecycle of machine learning models:
    - Model initialization and configuration
    - Training and evaluation
    - Model deployment and serving
    - Model versioning and storage
    """
    def __init__(self, 
                 config_path: str = "config/ml_config.json",
                 model_dir: str = "models"):
        """
        Initialize the ML model manager.
        
        Args:
            config_path: Path to ML configuration file
            model_dir: Directory for model storage
        """
        self.config_path = config_path
        self.model_dir = model_dir
        self.config = self._load_config()
        self.models = {}
        self.model_metadata = {}
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Initialized MLModelManager with config from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load ML configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default ML configuration."""
        return {
            "models": {
                "anomaly_detection": {
                    "enabled": True,
                    "lookback_window": 20,
                    "alert_threshold": 0.9,
                    "use_autoencoder": True,
                    "contamination": 0.01,
                    "training_frequency_days": 7
                },
                "price_prediction": {
                    "enabled": True,
                    "forecast_horizon": 5,
                    "confidence_threshold": 0.7,
                    "features": ["price", "volume", "technical"],
                    "training_frequency_days": 14
                },
                "market_classification": {
                    "enabled": True,
                    "num_regimes": 4,
                    "lookback_periods": 30,
                    "training_frequency_days": 30
                },
                "parameter_optimization": {
                    "enabled": True,
                    "n_trials": 100,
                    "optimization_frequency_days": 14,
                    "parameter_update_threshold": 0.1
                }
            },
            "integration": {
                "prediction_frequency_seconds": 300,
                "alert_channels": ["console", "telegram"],
                "save_predictions": True,
                "max_samples_per_symbol": 10000
            },
            "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"]
        }
    
    def initialize_models(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Initialize ML models for the given symbols.
        
        Args:
            symbols: List of symbols to initialize models for (if None, use config)
            
        Returns:
            Dictionary of initialized model count by type
        """
        if symbols is None:
            symbols = self.config.get("symbols", [])
        
        if not symbols:
            logger.warning("No symbols specified for model initialization")
            return {}
        
        model_counts = {
            "anomaly_detection": 0,
            "price_prediction": 0,
            "market_classification": 0,
            "parameter_optimization": 0
        }
        
        for symbol in symbols:
            # Initialize anomaly detection if enabled
            if self.config["models"]["anomaly_detection"]["enabled"]:
                model_id = f"anomaly_detection_{symbol}"
                self.models[model_id] = MarketAnomalyDetector(
                    symbol=symbol,
                    lookback_window=self.config["models"]["anomaly_detection"]["lookback_window"],
                    alert_threshold=self.config["models"]["anomaly_detection"]["alert_threshold"],
                    model_dir=os.path.join(self.model_dir, "anomaly_detection"),
                    use_autoencoder=self.config["models"]["anomaly_detection"]["use_autoencoder"],
                    contamination=self.config["models"]["anomaly_detection"]["contamination"]
                )
                model_counts["anomaly_detection"] += 1
                
                # Try to load existing model
                self.models[model_id].load_models()
                
                # Store metadata
                self.model_metadata[model_id] = {
                    "type": "anomaly_detection",
                    "symbol": symbol,
                    "last_training": None,
                    "version": 1
                }
            
            # Initialize price prediction if enabled
            if self.config["models"]["price_prediction"]["enabled"]:
                model_id = f"price_prediction_{symbol}"
                self.models[model_id] = PricePredictionModel(
                    symbol=symbol,
                    forecast_horizon=self.config["models"]["price_prediction"]["forecast_horizon"],
                    confidence_threshold=self.config["models"]["price_prediction"]["confidence_threshold"],
                    model_dir=os.path.join(self.model_dir, "price_prediction")
                )
                model_counts["price_prediction"] += 1
                
                # Store metadata
                self.model_metadata[model_id] = {
                    "type": "price_prediction",
                    "symbol": symbol,
                    "last_training": None,
                    "version": 1
                }
            
            # Initialize market classification for market-wide symbols
            if self.config["models"]["market_classification"]["enabled"] and symbol in ["SPY", "QQQ", "^VIX"]:
                model_id = f"market_classification_{symbol}"
                self.models[model_id] = MarketConditionClassifier(
                    symbol=symbol,
                    num_regimes=self.config["models"]["market_classification"]["num_regimes"],
                    lookback_periods=self.config["models"]["market_classification"]["lookback_periods"],
                    model_dir=os.path.join(self.model_dir, "market_classification")
                )
                model_counts["market_classification"] += 1
                
                # Store metadata
                self.model_metadata[model_id] = {
                    "type": "market_classification",
                    "symbol": symbol,
                    "last_training": None,
                    "version": 1
                }
        
        # Initialize parameter optimization (one per strategy type)
        if self.config["models"]["parameter_optimization"]["enabled"]:
            strategy_types = ["momentum", "mean_reversion", "trend_following", "volatility"]
            for strategy in strategy_types:
                model_id = f"parameter_optimization_{strategy}"
                self.models[model_id] = ParameterOptimizer(
                    strategy_type=strategy,
                    n_trials=self.config["models"]["parameter_optimization"]["n_trials"],
                    model_dir=os.path.join(self.model_dir, "parameter_optimization")
                )
                model_counts["parameter_optimization"] += 1
                
                # Store metadata
                self.model_metadata[model_id] = {
                    "type": "parameter_optimization",
                    "strategy": strategy,
                    "last_optimization": None,
                    "version": 1
                }
        
        logger.info(f"Initialized {sum(model_counts.values())} ML models: {model_counts}")
        return model_counts
    
    def train_model(self, model_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a specific model with the provided data.
        
        Args:
            model_id: ID of the model to train
            data: Training data
            
        Returns:
            Dictionary with training results
        """
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        
        logger.info(f"Training model {model_id} with {len(data)} samples")
        
        try:
            model = self.models[model_id]
            
            # Call appropriate training method based on model type
            if isinstance(model, MarketAnomalyDetector):
                results = model.train(data, save_model=True)
            elif isinstance(model, PricePredictionModel):
                results = model.train(data, save_model=True)
            elif isinstance(model, MarketConditionClassifier):
                results = model.train(data, save_model=True)
            elif isinstance(model, ParameterOptimizer):
                results = model.optimize_parameters(data, save_results=True)
            else:
                return {"error": f"Unknown model type for {model_id}"}
            
            # Update metadata
            self.model_metadata[model_id]["last_training"] = datetime.now().isoformat()
            self.model_metadata[model_id]["version"] += 1
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Successfully trained model {model_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {str(e)}")
            return {"error": str(e)}
    
    def _save_metadata(self):
        """Save model metadata to file."""
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def needs_training(self, model_id: str) -> bool:
        """
        Check if a model needs retraining based on the last training date.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            Boolean indicating if the model needs training
        """
        if model_id not in self.model_metadata:
            return True
        
        metadata = self.model_metadata[model_id]
        last_training = metadata.get("last_training")
        
        if last_training is None:
            return True
        
        last_date = datetime.fromisoformat(last_training)
        model_type = metadata.get("type")
        
        # Get training frequency from config
        if model_type == "anomaly_detection":
            frequency_days = self.config["models"]["anomaly_detection"]["training_frequency_days"]
        elif model_type == "price_prediction":
            frequency_days = self.config["models"]["price_prediction"]["training_frequency_days"]
        elif model_type == "market_classification":
            frequency_days = self.config["models"]["market_classification"]["training_frequency_days"]
        elif model_type == "parameter_optimization":
            frequency_days = self.config["models"]["parameter_optimization"]["optimization_frequency_days"]
        else:
            frequency_days = 30  # Default
        
        # Check if it's time to retrain
        return datetime.now() - last_date > timedelta(days=frequency_days)
    
    def get_model(self, model_id: str) -> Any:
        """Get a model by ID."""
        return self.models.get(model_id)
    
    def get_models_by_type(self, model_type: str) -> Dict[str, Any]:
        """Get all models of a specific type."""
        return {
            model_id: model 
            for model_id, model in self.models.items() 
            if model_id.startswith(model_type)
        }
    
    def get_model_for_symbol(self, model_type: str, symbol: str) -> Any:
        """Get a specific type of model for a symbol."""
        model_id = f"{model_type}_{symbol}"
        return self.models.get(model_id)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status for all models."""
        status = {}
        for model_id, metadata in self.model_metadata.items():
            status[model_id] = {
                "type": metadata.get("type"),
                "last_training": metadata.get("last_training"),
                "version": metadata.get("version", 1),
                "needs_training": self.needs_training(model_id)
            }
        return status


class MLIntegrationService:
    """
    Service for integrating ML models with the trading system:
    - Processes market data and feeds it to ML models
    - Executes predictions and forwards signals to strategies
    - Manages communication between ML components and trading system
    """
    def __init__(self, 
                 multi_asset_adapter: MultiAssetAdapter = None,
                 risk_manager: RiskManager = None,
                 config_path: str = "config/ml_config.json",
                 model_dir: str = "models"):
        """
        Initialize the ML integration service.
        
        Args:
            multi_asset_adapter: MultiAssetAdapter instance
            risk_manager: RiskManager instance
            config_path: Path to ML configuration file
            model_dir: Directory for model storage
        """
        self.multi_asset_adapter = multi_asset_adapter
        self.risk_manager = risk_manager
        self.model_manager = MLModelManager(config_path, model_dir)
        self.config = self.model_manager.config
        
        # Data cache for each symbol
        self.data_cache = {}
        
        # Prediction results
        self.latest_predictions = {}
        
        # Signal emitter for predictions
        self.signal_emitter = SignalEmitter()
        
        # Threading
        self.running = False
        self.prediction_thread = None
        self.training_thread = None
        
        logger.info("Initialized ML integration service")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the ML integration service.
        
        Returns:
            Dictionary with initialization results
        """
        # Initialize models
        model_counts = self.model_manager.initialize_models()
        
        # Initialize data cache
        symbols = self.config.get("symbols", [])
        for symbol in symbols:
            self.data_cache[symbol] = {
                "ohlcv": pd.DataFrame(),
                "last_update": None,
                "market_data": {}
            }
        
        return {
            "status": "initialized",
            "model_counts": model_counts,
            "symbols": symbols
        }
    
    def update_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update market data for a symbol.
        
        Args:
            symbol: Symbol to update data for
            data: New market data
            
        Returns:
            Dictionary with update results
        """
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {
                "ohlcv": pd.DataFrame(),
                "last_update": None,
                "market_data": {}
            }
        
        # Merge with existing data
        if not self.data_cache[symbol]["ohlcv"].empty:
            # Combine old and new data, removing duplicates
            combined = pd.concat([self.data_cache[symbol]["ohlcv"], data])
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # Sort by index
            combined = combined.sort_index()
            
            # Limit the size
            max_samples = self.config["integration"].get("max_samples_per_symbol", 10000)
            if len(combined) > max_samples:
                combined = combined.iloc[-max_samples:]
            
            self.data_cache[symbol]["ohlcv"] = combined
        else:
            self.data_cache[symbol]["ohlcv"] = data
        
        self.data_cache[symbol]["last_update"] = datetime.now()
        
        return {
            "symbol": symbol,
            "rows": len(self.data_cache[symbol]["ohlcv"]),
            "first_date": self.data_cache[symbol]["ohlcv"].index[0].strftime("%Y-%m-%d") if not self.data_cache[symbol]["ohlcv"].empty else None,
            "last_date": self.data_cache[symbol]["ohlcv"].index[-1].strftime("%Y-%m-%d") if not self.data_cache[symbol]["ohlcv"].empty else None
        }
    
    def start_prediction_service(self, interval_seconds: int = None) -> bool:
        """
        Start the prediction service in a background thread.
        
        Args:
            interval_seconds: Interval between predictions in seconds
            
        Returns:
            Boolean indicating if the service was started
        """
        if self.running:
            logger.warning("Prediction service is already running")
            return False
        
        if interval_seconds is None:
            interval_seconds = self.config["integration"].get("prediction_frequency_seconds", 300)
        
        self.running = True
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.prediction_thread.start()
        
        logger.info(f"Started prediction service with interval {interval_seconds} seconds")
        return True
    
    def _prediction_loop(self, interval_seconds: int):
        """
        Background thread for periodic predictions.
        
        Args:
            interval_seconds: Interval between predictions in seconds
        """
        while self.running:
            try:
                self.run_all_predictions()
            except Exception as e:
                logger.error(f"Error in prediction loop: {str(e)}")
            
            # Sleep until next prediction
            time.sleep(interval_seconds)
    
    def stop_prediction_service(self) -> bool:
        """
        Stop the prediction service.
        
        Returns:
            Boolean indicating if the service was stopped
        """
        if not self.running:
            logger.warning("Prediction service is not running")
            return False
        
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=5.0)
            self.prediction_thread = None
        
        logger.info("Stopped prediction service")
        return True
    
    def run_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """
        Run predictions for all models and symbols.
        
        Returns:
            Dictionary with prediction results by model type and symbol
        """
        results = {
            "anomaly_detection": {},
            "price_prediction": {},
            "market_classification": {}
        }
        
        # Run anomaly detection
        anomaly_models = self.model_manager.get_models_by_type("anomaly_detection")
        for model_id, model in anomaly_models.items():
            symbol = self.model_manager.model_metadata[model_id]["symbol"]
            if symbol in self.data_cache and not self.data_cache[symbol]["ohlcv"].empty:
                data = self.data_cache[symbol]["ohlcv"]
                prediction = model.detect_anomalies(data)
                results["anomaly_detection"][symbol] = prediction
                
                # Store latest prediction
                self.latest_predictions[model_id] = prediction
                
                # Emit signal if anomaly detected
                if prediction.get("num_anomalies", 0) > 0 and prediction.get("latest_score", 0) > model.alert_threshold:
                    alert = model.get_alert_message(data, prediction)
                    self.signal_emitter.emit_signal("anomaly_detected", {
                        "symbol": symbol,
                        "alert": alert,
                        "prediction": prediction
                    })
        
        # Run price prediction
        price_models = self.model_manager.get_models_by_type("price_prediction")
        for model_id, model in price_models.items():
            symbol = self.model_manager.model_metadata[model_id]["symbol"]
            if symbol in self.data_cache and not self.data_cache[symbol]["ohlcv"].empty:
                data = self.data_cache[symbol]["ohlcv"]
                prediction = model.predict(data)
                results["price_prediction"][symbol] = prediction
                
                # Store latest prediction
                self.latest_predictions[model_id] = prediction
                
                # Emit signal if high confidence prediction
                if prediction.get("confidence", 0) > model.confidence_threshold:
                    self.signal_emitter.emit_signal("price_prediction", {
                        "symbol": symbol,
                        "prediction": prediction
                    })
        
        # Run market classification
        market_models = self.model_manager.get_models_by_type("market_classification")
        for model_id, model in market_models.items():
            symbol = self.model_manager.model_metadata[model_id]["symbol"]
            if symbol in self.data_cache and not self.data_cache[symbol]["ohlcv"].empty:
                data = self.data_cache[symbol]["ohlcv"]
                prediction = model.classify_market_condition(data)
                results["market_classification"][symbol] = prediction
                
                # Store latest prediction
                self.latest_predictions[model_id] = prediction
                
                # Emit signal with market condition
                self.signal_emitter.emit_signal("market_condition", {
                    "symbol": symbol,
                    "prediction": prediction
                })
        
        # Save predictions if configured
        if self.config["integration"].get("save_predictions", True):
            self._save_predictions(results)
        
        return results
    
    def _save_predictions(self, results: Dict[str, Dict[str, Any]]):
        """
        Save prediction results to file.
        
        Args:
            results: Prediction results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_dir = os.path.join(self.model_manager.model_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        try:
            filename = os.path.join(predictions_dir, f"predictions_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
    
    def start_training_service(self, interval_hours: int = 6) -> bool:
        """
        Start the training service in a background thread.
        
        Args:
            interval_hours: Interval between training checks in hours
            
        Returns:
            Boolean indicating if the service was started
        """
        if self.training_thread and self.training_thread.is_alive():
            logger.warning("Training service is already running")
            return False
        
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(interval_hours,),
            daemon=True
        )
        self.training_thread.start()
        
        logger.info(f"Started training service with interval {interval_hours} hours")
        return True
    
    def _training_loop(self, interval_hours: int):
        """
        Background thread for periodic training.
        
        Args:
            interval_hours: Interval between training checks in hours
        """
        while True:
            try:
                self.check_and_train_models()
            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}")
            
            # Sleep until next check
            time.sleep(interval_hours * 3600)
    
    def check_and_train_models(self) -> Dict[str, Any]:
        """
        Check which models need training and train them.
        
        Returns:
            Dictionary with training results
        """
        results = {
            "checked": [],
            "trained": [],
            "failed": []
        }
        
        # Check all models
        for model_id in self.model_manager.model_metadata:
            results["checked"].append(model_id)
            
            # Check if model needs training
            if self.model_manager.needs_training(model_id):
                logger.info(f"Model {model_id} needs training")
                
                # Get model type and symbol
                metadata = self.model_manager.model_metadata[model_id]
                model_type = metadata.get("type")
                
                if model_type == "parameter_optimization":
                    # Parameter optimization requires backtest data
                    # This would need integration with the backtesting system
                    logger.info(f"Skipping {model_id} training - requires backtest data")
                    continue
                
                # Get data for the model
                symbol = metadata.get("symbol")
                if symbol not in self.data_cache or self.data_cache[symbol]["ohlcv"].empty:
                    logger.warning(f"No data available for {model_id}")
                    results["failed"].append({
                        "model_id": model_id,
                        "reason": "No data available"
                    })
                    continue
                
                # Train the model
                data = self.data_cache[symbol]["ohlcv"]
                training_result = self.model_manager.train_model(model_id, data)
                
                if "error" in training_result:
                    logger.error(f"Error training {model_id}: {training_result['error']}")
                    results["failed"].append({
                        "model_id": model_id,
                        "reason": training_result["error"]
                    })
                else:
                    logger.info(f"Successfully trained {model_id}")
                    results["trained"].append({
                        "model_id": model_id,
                        "result": training_result
                    })
        
        return results
    
    def get_ml_insights(self) -> Dict[str, Any]:
        """
        Get insights from ML models for all symbols.
        
        Returns:
            Dictionary with ML insights by symbol
        """
        insights = {}
        
        for symbol in self.data_cache:
            symbol_insights = {
                "anomalies": None,
                "price_prediction": None,
                "market_condition": None,
                "optimal_parameters": {}
            }
            
            # Get latest anomaly detection
            anomaly_model_id = f"anomaly_detection_{symbol}"
            if anomaly_model_id in self.latest_predictions:
                prediction = self.latest_predictions[anomaly_model_id]
                symbol_insights["anomalies"] = {
                    "latest_score": prediction.get("latest_score", 0),
                    "alert": prediction.get("latest_score", 0) > self.model_manager.get_model(anomaly_model_id).alert_threshold,
                    "num_anomalies": prediction.get("num_anomalies", 0)
                }
            
            # Get latest price prediction
            price_model_id = f"price_prediction_{symbol}"
            if price_model_id in self.latest_predictions:
                prediction = self.latest_predictions[price_model_id]
                symbol_insights["price_prediction"] = {
                    "direction": prediction.get("direction", "unknown"),
                    "magnitude": prediction.get("magnitude", 0),
                    "confidence": prediction.get("confidence", 0),
                    "horizon": prediction.get("horizon", 0)
                }
            
            # Get market condition (use SPY or symbol specific)
            market_model_ids = [f"market_classification_{symbol}", "market_classification_SPY"]
            for model_id in market_model_ids:
                if model_id in self.latest_predictions:
                    prediction = self.latest_predictions[model_id]
                    symbol_insights["market_condition"] = {
                        "regime": prediction.get("regime", "unknown"),
                        "probabilities": prediction.get("probabilities", {}),
                        "description": prediction.get("description", "")
                    }
                    break
            
            # Get optimal parameters for strategies
            strategy_types = ["momentum", "mean_reversion", "trend_following", "volatility"]
            for strategy in strategy_types:
                model_id = f"parameter_optimization_{strategy}"
                if model_id in self.latest_predictions:
                    prediction = self.latest_predictions[model_id]
                    if "parameters" in prediction:
                        symbol_insights["optimal_parameters"][strategy] = prediction["parameters"]
            
            insights[symbol] = symbol_insights
        
        return insights
    
    def apply_ml_signals_to_risk_manager(self) -> Dict[str, Any]:
        """
        Apply ML signals to the risk manager.
        
        Returns:
            Dictionary with application results
        """
        if self.risk_manager is None:
            return {"error": "No risk manager available"}
        
        results = {
            "anomaly_adjustments": [],
            "market_condition_adjustments": []
        }
        
        # Process anomaly signals
        for symbol in self.data_cache:
            anomaly_model_id = f"anomaly_detection_{symbol}"
            if anomaly_model_id in self.latest_predictions:
                prediction = self.latest_predictions[anomaly_model_id]
                anomaly_score = prediction.get("latest_score", 0)
                alert_threshold = self.model_manager.get_model(anomaly_model_id).alert_threshold
                
                if anomaly_score > alert_threshold:
                    # Apply risk adjustment for anomaly
                    adjustment_factor = min(1.0, anomaly_score / alert_threshold)
                    adjustment = {
                        "symbol": symbol,
                        "factor": 1.0 - (adjustment_factor * 0.5),  # Reduce position size by up to 50%
                        "reason": f"Anomaly score {anomaly_score:.4f} above threshold {alert_threshold:.4f}"
                    }
                    
                    # Apply to risk manager
                    self.risk_manager.apply_ml_adjustment(
                        symbol, "volatility_multiplier", adjustment["factor"], "anomaly_detection"
                    )
                    
                    results["anomaly_adjustments"].append(adjustment)
        
        # Process market condition signals
        market_model_id = "market_classification_SPY"
        if market_model_id in self.latest_predictions:
            prediction = self.latest_predictions[market_model_id]
            regime = prediction.get("regime", "unknown")
            description = prediction.get("description", "")
            
            # Apply global risk adjustment based on market regime
            if regime == "high_volatility" or regime == "bear":
                adjustment = {
                    "regime": regime,
                    "factor": 0.7,  # Reduce overall risk by 30%
                    "reason": f"Market in {description} regime"
                }
                self.risk_manager.apply_ml_adjustment(
                    "global", "risk_multiplier", adjustment["factor"], "market_regime"
                )
                results["market_condition_adjustments"].append(adjustment)
            elif regime == "low_volatility" or regime == "bull":
                adjustment = {
                    "regime": regime,
                    "factor": 1.1,  # Increase overall risk by 10%
                    "reason": f"Market in {description} regime"
                }
                self.risk_manager.apply_ml_adjustment(
                    "global", "risk_multiplier", adjustment["factor"], "market_regime"
                )
                results["market_condition_adjustments"].append(adjustment)
        
        return results

def main():
    """Example usage of the ML integration module."""
    from trading_bot.multi_asset_adapter import MultiAssetAdapter
    from trading_bot.risk_manager import RiskManager
    
    # Initialize components
    adapter = MultiAssetAdapter("config/adapter_config.json")
    risk_manager = RiskManager(adapter, "config/risk_config.json")
    
    # Initialize ML integration
    ml = MLIntegrationService(adapter, risk_manager)
    ml.initialize()
    
    # Train models
    ml.train_models()
    
    # Update with latest data
    ml.update()
    
    # Get trading signals
    signals = ml.get_trading_signals()
    print(f"Trading signals: {signals}")
    
    # Apply insights to trading system
    ml.apply_insights(adapter, risk_manager)
    
    # Get ML insights report
    report = ml.get_ml_insights_report()
    print(f"ML insights report: {report}")

if __name__ == "__main__":
    main() 