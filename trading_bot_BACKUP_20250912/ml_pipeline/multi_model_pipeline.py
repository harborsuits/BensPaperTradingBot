"""
Multi-Model Prediction Pipeline

Creates robust trading signals by:
- Combining multiple prediction models
- Dynamically weighting signal inputs
- Processing parallel prediction streams
"""

import pandas as pd
import numpy as np
import logging
import time
import json
import concurrent.futures
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import joblib
from datetime import datetime, timedelta
import importlib
import os

from trading_bot.ml_pipeline.advanced_feature_generator import AdvancedFeatureGenerator
from trading_bot.ml_pipeline.realtime_analyzer import RealtimeAnalyzer
from trading_bot.ml_pipeline.model_registry import ModelRegistry
from trading_bot.ml_pipeline.ensemble_methods import EnsembleAggregator, EnsembleMethod, adjust_model_weights
from trading_bot.ml_pipeline.freqtrade_adapter import FreqTradeStrategyRegistry, FreqTradeStrategyAdapter

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a prediction model"""
    name: str
    type: str  # ml, statistical, rule_based, hybrid, ensemble
    path: str
    weight: float = 1.0
    performance_window: int = 30  # Days to consider for performance tracking
    metrics: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    method: str = "weighted_average"
    dynamic_weights: bool = True
    performance_metric: str = "sharpe_ratio"
    weight_update_frequency: str = "daily"
    min_weight: float = 0.1
    max_weight: float = 3.0

class MultiModelPipeline:
    """
    Multi-Model Prediction Pipeline that creates robust trading signals
    by combining multiple prediction models with dynamic weighting.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-model prediction pipeline
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides config_path)
        """
        self.config = config or {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize components
        self._initialize_components()
        
        # Ensemble configuration
        ensemble_config = self.config.get('ensemble', {})
        self.ensemble_config = EnsembleConfig(
            method=ensemble_config.get('method', 'weighted_average'),
            dynamic_weights=ensemble_config.get('dynamic_weights', True),
            performance_metric=ensemble_config.get('performance_metric', 'sharpe_ratio'),
            weight_update_frequency=ensemble_config.get('weight_update_frequency', 'daily'),
            min_weight=ensemble_config.get('min_weight', 0.1),
            max_weight=ensemble_config.get('max_weight', 3.0)
        )
        
        # Initialize ensemble aggregator
        self.ensemble_aggregator = EnsembleAggregator(method=self.ensemble_config.method)
        
        # Prediction configuration
        self.prediction_config = self.config.get('prediction', {
            'horizons': [1, 5, 20],
            'targets': ['direction', 'volatility', 'magnitude'],
            'confidence_threshold': 0.62
        })
        
        # Performance tracking
        self.performance_history = {}
        
        # Model configurations from legacy format (backward compatibility)
        self.models = {}
        
        # Load all models
        self._load_models()
        
        logger.info("Multi-Model Prediction Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize required components"""
        # Initialize feature generator if not provided
        feature_config = self.config.get('feature_generator', {})
        self.feature_generator = AdvancedFeatureGenerator(config=feature_config)
        
        # Initialize analyzer if needed for real-time processing
        analyzer_config = self.config.get('analyzer', {})
        self.analyzer = RealtimeAnalyzer(config=analyzer_config)
        
        # Initialize model registry
        registry_path = self.config.get('model_registry', {}).get('path', 'models')
        self.model_registry = ModelRegistry(registry_path=registry_path)
        
        # Initialize FreqTrade strategy registry
        freqtrade_dir = self.config.get('freqtrade', {}).get('strategies_dir', 'strategies/freqtrade')
        self.freqtrade_registry = FreqTradeStrategyRegistry(strategies_dir=freqtrade_dir)
    
    def _load_models(self):
        """Load all models defined in configuration"""
        # First load any models already in the registry
        for model_info in self.model_registry.list_models():
            try:
                model_obj = ModelConfig(
                    name=model_info.name,
                    type=model_info.type,
                    path=model_info.path,
                    weight=1.0,  # Default weight
                    metrics=model_info.metrics,
                    enabled=True,
                    params=model_info.parameters
                )
                
                # Add to models dict for backward compatibility
                self.models[model_obj.name] = model_obj
                
                logger.info(f"Loaded model from registry: {model_obj.name} ({model_obj.type})")
            except Exception as e:
                logger.warning(f"Error loading model from registry: {e}")
        
        # Then load models defined in configuration (may overlap with registry)
        models_config = self.config.get('models', [])
        for model_config in models_config:
            try:
                model_name = model_config['name']
                model_type = model_config['type']
                model_path = model_config['path']
                model_params = model_config.get('params', {})
                
                # Check if model exists in registry
                existing_model = self.model_registry.get_model_info(model_name)
                
                if existing_model is None:
                    # Register the model
                    if model_type == 'ml':
                        # For ML models, load and register
                        model = self._load_ml_model(model_path)
                        if model is not None:
                            self.model_registry.register_model(
                                name=model_name,
                                model=model,
                                model_type=model_type,
                                parameters=model_params
                            )
                    elif model_type == 'freqtrade':
                        # For FreqTrade strategies, register with the adapter
                        self.freqtrade_registry.register_strategy(
                            name=model_name,
                            strategy_path=model_path,
                            strategy_config=model_params
                        )
                
                # Create model config object
                model_obj = ModelConfig(
                    name=model_name,
                    type=model_type,
                    path=model_path,
                    weight=model_config.get('weight', 1.0),
                    performance_window=model_config.get('performance_window', 30),
                    enabled=model_config.get('enabled', True),
                    params=model_params
                )
                
                # Add to models dict
                self.models[model_obj.name] = model_obj
                
                logger.info(f"Registered model from config: {model_obj.name} ({model_obj.type})")
            except Exception as e:
                logger.warning(f"Error loading model config: {e}")
    
    def _load_ml_model(self, path: str):
        """Load machine learning model from file"""
        try:
            return joblib.load(path) if os.path.exists(path) else None
        except Exception as e:
            logger.error(f"Failed to load ML model from {path}: {e}")
            return None
    
    def _load_statistical_model(self, path: str, params: Dict[str, Any]):
        """Load statistical model from module"""
        return self._load_module_class(path, params.get('class_name', 'StatisticalModel'), params)
    
    def _load_rule_based_model(self, path: str, params: Dict[str, Any]):
        """Load rule-based model from module"""
        return self._load_module_class(path, params.get('class_name', 'RuleBasedModel'), params)
    
    def _load_hybrid_model(self, path: str, params: Dict[str, Any]):
        """Load hybrid model from module"""
        return self._load_module_class(path, params.get('class_name', 'HybridModel'), params)
    
    def _load_module_class(self, path: str, class_name: str, params: Dict[str, Any]) -> Any:
        """Load a class from a module and instantiate it"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the class
        model_class = getattr(module, class_name)
        
        # Instantiate with parameters
        init_params = params.get('init_params', {})
        return model_class(**init_params)
    
    def _load_freqtrade_strategy(self, path: str):
        """Load a FreqTrade strategy"""
        try:
            # Use the FreqTrade adapter to load the strategy
            adapter = FreqTradeStrategyAdapter(strategy_path=path)
            return adapter
        except Exception as e:
            logger.error(f"Failed to load FreqTrade strategy: {e}")
            return None
    
    def predict(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Generate predictions from all models and combine them
        
        Args:
            data: DataFrame with market data
            symbol: Optional symbol name for the prediction
            
        Returns:
            Dictionary with combined predictions and signals
        """
        # Check if we should use realtime analyzer for this prediction
        use_realtime = self.config.get('prediction', {}).get('use_realtime_analyzer', False)
        
        if use_realtime:
            # Use realtime analyzer for feature generation and model application
            return self.analyzer.analyze(data, symbol)
        
        # Otherwise use standard pipeline
        # Generate features
        features = self.feature_generator.generate_features(data)
        
        # Apply all models in parallel
        model_predictions = self._apply_models_parallel(features)
        
        # Combine predictions
        combined = self._combine_predictions(model_predictions, symbol)
        
        # Update performance tracking
        self._update_performance_tracking(model_predictions, combined)
        
        # Add timestamp
        combined['timestamp'] = datetime.now()
        combined['symbol'] = symbol
        
        return combined
    
    def _apply_models_parallel(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Apply all models in parallel using ThreadPoolExecutor"""
        model_predictions = {}
        
        # Only apply enabled models
        enabled_models = {name: model for name, model in self.models.items() if model.enabled}
        
        # If no models enabled, return empty predictions
        if not enabled_models:
            logger.warning("No enabled models found for prediction")
            return model_predictions
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(self._apply_model, model_config, features): name 
                              for name, model_config in enabled_models.items()}
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result is not None:
                        model_predictions[model_name] = result
                except Exception as e:
                    logger.error(f"Error applying model {model_name}: {str(e)}")
                    model_predictions[model_name] = None
        
        return model_predictions
    
    def _apply_model(self, model_config: ModelConfig, features: pd.DataFrame) -> Any:
        """Apply a single model to features"""
        try:
            model_name = model_config.name
            model_type = model_config.type
            model_path = model_config.path
            
            if model_type == 'ml':
                # Load ML model from registry
                model = self.model_registry.load_model(model_name)
                if model is None:
                    # Try to load directly if not in registry
                    model = self._load_ml_model(model_path)
                    if model is None:
                        logger.error(f"ML model {model_name} could not be loaded")
                        return None
                
                # ML model: use predict method
                if hasattr(model, 'predict_proba'):
                    # Classification model with probabilities
                    probs = model.predict_proba(features)
                    # Get buy probability (usually class 1 for binary classification)
                    if probs.shape[1] >= 2:
                        prediction = probs[:, 1]
                    else:
                        prediction = probs.flatten()
                elif hasattr(model, 'predict'):
                    # Regression model
                    prediction = model.predict(features)
                else:
                    logger.error(f"Model {model_name} has no predict method")
                    return None
                
                # Format result
                return {
                    'prediction': prediction[-1] if len(prediction) > 0 else 0,
                    'confidence': 0.7,  # Default confidence
                    'type': 'ml'
                }
                
            elif model_type == 'statistical':
                # Load statistical model based on config
                model = self._load_statistical_model(model_path, model_config.params)
                if model is None:
                    logger.error(f"Statistical model {model_name} could not be loaded")
                    return None
                
                # Statistical model: call analyze method
                if hasattr(model, 'analyze'):
                    result = model.analyze(features)
                    return {
                        'prediction': result.get('prediction', 0),
                        'confidence': result.get('confidence', 0.6),
                        'type': 'statistical'
                    }
                else:
                    logger.error(f"Statistical model {model_name} has no analyze method")
                    return None
                    
            elif model_type == 'rule_based':
                # Load rule-based model based on config
                model = self._load_rule_based_model(model_path, model_config.params)
                if model is None:
                    logger.error(f"Rule-based model {model_name} could not be loaded")
                    return None
                
                # Rule-based model: call apply method
                if hasattr(model, 'apply'):
                    result = model.apply(features)
                    return {
                        'prediction': result.get('prediction', 0),
                        'confidence': result.get('confidence', 0.8),
                        'signal': result.get('signal', 'neutral'),
                        'type': 'rule_based'
                    }
                else:
                    logger.error(f"Rule-based model {model_name} has no apply method")
                    return None
                    
            elif model_type == 'hybrid':
                # Load hybrid model based on config
                model = self._load_hybrid_model(model_path, model_config.params)
                if model is None:
                    logger.error(f"Hybrid model {model_name} could not be loaded")
                    return None
                
                # Hybrid model: call predict method
                if hasattr(model, 'predict'):
                    result = model.predict(features)
                    return {
                        'prediction': result.get('prediction', 0),
                        'confidence': result.get('confidence', 0.75),
                        'type': 'hybrid'
                    }
                else:
                    logger.error(f"Hybrid model {model_name} has no predict method")
                    return None
                    
            elif model_type == 'freqtrade':
                # Get FreqTrade strategy from registry
                strategy = self.freqtrade_registry.get_strategy(model_name)
                if strategy is None:
                    # Try loading directly if not in registry
                    strategy = self._load_freqtrade_strategy(model_path)
                    if strategy is None:
                        logger.error(f"FreqTrade strategy {model_name} could not be loaded")
                        return None
                
                # Use adapter's prediction method
                result = strategy.get_prediction(features)
                
                return {
                    'prediction': 1.0 if result['signal'] == 'buy' else (-1.0 if result['signal'] == 'sell' else 0.0),
                    'confidence': result.get('confidence', 0.75),
                    'signal': result['signal'],
                    'strength': result.get('strength', 0),
                    'raw': result,
                    'type': 'freqtrade'
                }
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying model {model_config.name}: {str(e)}")
            return None
    
    def _combine_predictions(self, model_predictions: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """
        Combine predictions from multiple models using the configured ensemble method
        
        Args:
            model_predictions: Dictionary of model predictions
            symbol: Optional symbol name
            
        Returns:
            Combined prediction
        """
        if not model_predictions:
            logger.warning("No predictions to combine")
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {}
            }
        
        # Get model weights
        weights = {name: self._get_model_weight(name, symbol) for name in model_predictions.keys()}
        
        # Get confidences from predictions if available
        confidences = {}
        for name, pred in model_predictions.items():
            if pred is not None and isinstance(pred, dict) and 'confidence' in pred:
                confidences[name] = pred['confidence']
        
        # Use ensemble aggregator to combine predictions
        threshold = self.prediction_config.get('confidence_threshold', 0.62)
        return self.ensemble_aggregator.aggregate(
            predictions=model_predictions,
            weights=weights,
            confidences=confidences if confidences else None,
            threshold=threshold
        )
    
    def _get_model_weight(self, model_name: str, symbol: str = None) -> float:
        """
        Get the current weight for a model, possibly adjusted based on performance
        
        Args:
            model_name: Name of the model
            symbol: Optional symbol name for symbol-specific weights
            
        Returns:
            Current weight for the model
        """
        model_obj = self.models.get(model_name)
        if not model_obj:
            return 0
            
        # Start with configured weight
        base_weight = model_obj.weight
        
        # If dynamic weights are enabled, adjust based on performance
        if self.ensemble_config.dynamic_weights and symbol in self.performance_history:
            model_history = self.performance_history[symbol].get(model_name, [])
            
            if model_history:
                # Get performance metric to use
                metric_name = self.ensemble_config.performance_metric
                
                # Calculate average of the metric over the window
                performance_window = model_obj.performance_window
                recent_history = model_history[-performance_window:]
                
                if recent_history and metric_name in recent_history[0]:
                    metric_values = [x.get(metric_name, 0) for x in recent_history]
                    avg_metric = sum(metric_values) / len(metric_values)
                    
                    # Adjust weight based on metric
                    # Simple scaling: better performers get higher weights
                    if avg_metric > 0:  # Good performance
                        adjustment = min(1 + avg_metric, self.ensemble_config.max_weight / base_weight)
                        adjusted_weight = base_weight * adjustment
                    else:  # Poor performance
                        adjustment = max(0.5, self.ensemble_config.min_weight / base_weight)
                        adjusted_weight = base_weight * adjustment
                        
                    return adjusted_weight
        
        return base_weight
    
    def _update_performance_tracking(self, model_predictions: Dict[str, Any], combined_prediction: Dict[str, Any]):
        """
        Update performance tracking for models
        
        Args:
            model_predictions: Dictionary of model predictions
            combined_prediction: Combined prediction
        """
        # Get symbol
        symbol = combined_prediction.get('symbol', 'default')
        
        # Initialize history for this symbol if not exists
        if symbol not in self.performance_history:
            self.performance_history[symbol] = {}
            
        # Current timestamp
        timestamp = pd.Timestamp.now()
        
        # Update for each model
        for name, result in model_predictions.items():
            if result.get('prediction') is None:
                continue
                
            # Initialize model history if not exists
            if name not in self.performance_history[symbol]:
                self.performance_history[symbol][name] = []
                
            # Basic metrics - can be expanded
            metrics = {
                'timestamp': timestamp,
                'prediction': result.get('prediction'),
                'confidence': result.get('confidence', 0.5),
                'processing_time': result.get('processing_time', 0)
            }
            
            # Add to history
            self.performance_history[symbol][name].append(metrics)
            
            # Limit history length
            max_history = self.models[name].performance_window * 2
            if len(self.performance_history[symbol][name]) > max_history:
                self.performance_history[symbol][name] = self.performance_history[symbol][name][-max_history:]
    
    def update_model_performance(self, symbol: str, actual_outcome: float, timestamp: Optional[datetime] = None):
        """
        Update model performance metrics with actual market outcome
        
        Args:
            symbol: Symbol name
            actual_outcome: Actual market outcome (e.g., return or direction)
            timestamp: Optional timestamp of the prediction (defaults to latest)
        """
        if symbol not in self.performance_history:
            return
            
        # Default to latest timestamp if not provided
        if timestamp is None:
            # Find the most recent prediction timestamp
            latest_timestamp = None
            for model_name in self.performance_history[symbol]:
                if self.performance_history[symbol][model_name]:
                    model_latest = self.performance_history[symbol][model_name][-1]['timestamp']
                    if latest_timestamp is None or model_latest > latest_timestamp:
                        latest_timestamp = model_latest
            
            timestamp = latest_timestamp
        
        if timestamp is None:
            return
        
        # Update metrics for each model
        for model_name, history in self.performance_history[symbol].items():
            # Find prediction closest to the timestamp
            closest_idx = None
            closest_diff = None
            
            for i, entry in enumerate(history):
                diff = abs((entry['timestamp'] - timestamp).total_seconds())
                if closest_diff is None or diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i
            
            # Skip if no prediction found within reasonable time
            if closest_idx is None or closest_diff > 24 * 3600:  # 24 hours
                continue
            
            # Update entry with outcome and performance metrics
            entry = history[closest_idx]
            prediction = entry.get('prediction')
            
            # Calculate performance metrics
            if prediction is not None:
                # Direction accuracy
                direction_match = (prediction > 0 and actual_outcome > 0) or \
                                (prediction < 0 and actual_outcome < 0)
                
                # Prediction error
                if isinstance(prediction, (int, float)):
                    prediction_error = abs(prediction - actual_outcome)
                else:
                    prediction_error = 0
                
                # Update metrics
                entry['actual_outcome'] = actual_outcome
                entry['direction_correct'] = direction_match
                entry['prediction_error'] = prediction_error
                
                # Calculate Sharpe-like metric if we have enough history
                recent_entries = [e for e in history if 'direction_correct' in e]
                if len(recent_entries) >= 5:
                    # Simple Sharpe-like ratio
                    returns = [1.0 if e['direction_correct'] else -1.0 for e in recent_entries]
                    avg_return = sum(returns) / len(returns)
                    std_return = np.std(returns) if len(returns) > 1 else 1.0
                    
                    if std_return > 0:
                        sharpe = avg_return / std_return
                    else:
                        sharpe = 0
                        
                    entry['sharpe_ratio'] = sharpe
                    
                # Update model metrics
                self.models[model_name].metrics = {
                    'direction_accuracy': sum(1 for e in recent_entries if e.get('direction_correct', False)) / len(recent_entries) if recent_entries else 0,
                    'avg_prediction_error': sum(e.get('prediction_error', 0) for e in recent_entries) / len(recent_entries) if recent_entries else 0,
                    'sharpe_ratio': sharpe if 'sharpe' in locals() else 0
                }

# Utility functions for FreqTrade integration
def import_freqtrade_strategy(strategy_path: str) -> Any:
    """Import a FreqTrade strategy as a prediction model"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find strategy class
    strategy_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and hasattr(attr, 'populate_indicators'):
            strategy_class = attr
            break
            
    return strategy_class

def adapt_freqtrade_signals(strategy_instance, dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Adapt FreqTrade signals to the multi-model pipeline format"""
    # Run FreqTrade strategy
    with_indicators = strategy_instance.populate_indicators(dataframe.copy(), {})
    with_buy_sell = strategy_instance.populate_buy_trend(with_indicators, {})
    with_buy_sell = strategy_instance.populate_sell_trend(with_buy_sell, {})
    
    # Extract signals
    buy_signals = with_buy_sell['buy'].astype(int).values
    sell_signals = with_buy_sell['sell'].astype(int).values
    
    # Convert to our signal format (-1 to 1 scale)
    signal_strength = buy_signals - sell_signals
    
    return {
        'signal_strength': signal_strength,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'dataframe': with_buy_sell
    }
