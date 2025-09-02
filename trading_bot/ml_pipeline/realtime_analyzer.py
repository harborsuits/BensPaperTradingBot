"""
Real-Time Analyzer System

This module provides a real-time analysis system with:
- On-the-fly feature computation
- Concurrent model application
- Runtime signal aggregation and processing
"""

import pandas as pd
import numpy as np
import logging
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import joblib
import asyncio
import queue
from threading import Thread

from trading_bot.ml_pipeline.advanced_feature_generator import AdvancedFeatureGenerator

logger = logging.getLogger(__name__)

@dataclass
class AnalysisModel:
    """Configuration for a analysis model"""
    name: str
    type: str  # Type of model (ml, rule-based, hybrid)
    path: str  # Path to model file
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for model
    weight: float = 1.0  # Weight in the final signal
    feature_set: str = ""  # Feature set to use
    target_variable: str = ""  # Target variable for the model
    thresholds: Dict[str, float] = field(default_factory=dict)  # Thresholds for signal generation
    enabled: bool = True

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    name: str
    aggregation: str = "weighted_average"  # How to aggregate signals from multiple models
    thresholds: Dict[str, float] = field(default_factory=dict)  # Thresholds for final signal generation
    time_decay: float = 0.0  # Time decay factor for signal strength
    history_length: int = 10  # Number of historical signals to keep

class RealtimeAnalyzer:
    """
    Real-time analyzer system that processes incoming data streams,
    computes features, applies models, and generates trading signals.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time analyzer
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides config_path)
        """
        self.config = config or {}
        if config_path:
            import json
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
        # Initialize feature generator
        feature_config = self.config.get('feature_generator', {})
        self.feature_generator = AdvancedFeatureGenerator(config=feature_config)
        
        # Initialize models
        self.models = {}
        self._load_models()
        
        # Initialize signal configuration
        signal_config = self.config.get('signal', {})
        self.signal_config = SignalConfig(
            name=signal_config.get('name', 'default'),
            aggregation=signal_config.get('aggregation', 'weighted_average'),
            thresholds=signal_config.get('thresholds', {'buy': 0.7, 'sell': -0.7}),
            time_decay=signal_config.get('time_decay', 0.0),
            history_length=signal_config.get('history_length', 10)
        )
        
        # Signal history
        self.signal_history = {}
        
        # Runtime metrics
        self.metrics = {
            'processing_time': [],
            'model_latency': {},
            'signal_strength': {},
            'data_freshness': {}
        }
        
        # Processing queues for async operation
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.processing_thread = None
        
        logger.info("Real-time Analyzer initialized")
    
    def _load_models(self):
        """Load all models defined in the configuration"""
        models_config = self.config.get('models', [])
        for model_config in models_config:
            model_obj = AnalysisModel(
                name=model_config['name'],
                type=model_config['type'],
                path=model_config['path'],
                params=model_config.get('params', {}),
                weight=model_config.get('weight', 1.0),
                feature_set=model_config.get('feature_set', ''),
                target_variable=model_config.get('target_variable', ''),
                thresholds=model_config.get('thresholds', {}),
                enabled=model_config.get('enabled', True)
            )
            
            try:
                if model_obj.type == 'ml':
                    model_obj.model = joblib.load(model_obj.path)
                elif model_obj.type == 'rule_based':
                    # Import rule-based model from module
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("model_module", model_obj.path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    model_obj.model = getattr(module, model_obj.params.get('class_name', 'RuleBasedModel'))()
                elif model_obj.type == 'hybrid':
                    # Import hybrid model from module
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("model_module", model_obj.path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    model_class = getattr(module, model_obj.params.get('class_name', 'HybridModel'))
                    model_obj.model = model_class(**model_obj.params.get('init_params', {}))
                
                self.models[model_obj.name] = model_obj
                logger.info(f"Loaded model: {model_obj.name}")
            except Exception as e:
                logger.error(f"Error loading model {model_obj.name}: {e}")
    
    def process_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process data and generate signals
        
        Args:
            data_dict: Dictionary of DataFrames with source name as key
            
        Returns:
            Dictionary with signal information
        """
        start_time = time.time()
        
        # Generate features
        features = self.feature_generator.generate_features(data_dict)
        
        feature_time = time.time()
        self.metrics['processing_time'].append({
            'timestamp': pd.Timestamp.now(),
            'feature_generation': feature_time - start_time
        })
        
        # Apply models in parallel
        model_results = self._apply_models_parallel(features)
        
        # Aggregate signals
        signal = self._aggregate_signals(model_results)
        
        # Update signal history
        self._update_signal_history(signal)
        
        # Calculate processing time
        end_time = time.time()
        self.metrics['processing_time'][-1]['total'] = end_time - start_time
        
        return signal
    
    def _apply_models_parallel(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply all enabled models in parallel
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary of model results
        """
        model_results = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for name, model_obj in self.models.items():
                if not model_obj.enabled:
                    continue
                
                futures[executor.submit(self._apply_model, model_obj, features)] = name
            
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    start_time = time.time()
                    result = future.result()
                    end_time = time.time()
                    
                    # Track model latency
                    if name not in self.metrics['model_latency']:
                        self.metrics['model_latency'][name] = []
                    
                    self.metrics['model_latency'][name].append({
                        'timestamp': pd.Timestamp.now(),
                        'latency': end_time - start_time
                    })
                    
                    # Keep limited history
                    if len(self.metrics['model_latency'][name]) > 100:
                        self.metrics['model_latency'][name] = self.metrics['model_latency'][name][-100:]
                    
                    model_results[name] = result
                except Exception as e:
                    logger.error(f"Error applying model {name}: {e}")
                    model_results[name] = None
        
        return model_results
    
    def _apply_model(self, model_obj: AnalysisModel, features: pd.DataFrame) -> Any:
        """
        Apply a single model to features
        
        Args:
            model_obj: Model object
            features: DataFrame with features
            
        Returns:
            Model prediction or signal
        """
        if model_obj.type == 'ml':
            # Select features for this model
            if model_obj.feature_set and hasattr(self.feature_generator, 'feature_sets'):
                feature_set = self.feature_generator.feature_sets.get(model_obj.feature_set)
                if feature_set and hasattr(feature_set, 'features'):
                    # Get available features that match the required features
                    available_features = [f for f in feature_set.features if f in features.columns]
                    X = features[available_features].fillna(0)
                else:
                    X = features.fillna(0)
            else:
                X = features.fillna(0)
            
            # Apply model
            try:
                predictions = model_obj.model.predict(X)
                # If the model has predict_proba method, use it for signal strength
                if hasattr(model_obj.model, 'predict_proba'):
                    probabilities = model_obj.model.predict_proba(X)
                    if probabilities.shape[1] >= 2:  # Binary classification
                        signal_strength = probabilities[:, 1] - probabilities[:, 0]
                    else:
                        signal_strength = probabilities[:, 0]
                else:
                    signal_strength = predictions
                
                return {
                    'predictions': predictions,
                    'signal_strength': signal_strength,
                    'timestamp': pd.Timestamp.now()
                }
            except Exception as e:
                logger.error(f"Error in ML model prediction: {e}")
                return None
                
        elif model_obj.type == 'rule_based':
            # Apply rule-based model
            try:
                result = model_obj.model.apply(features)
                return {
                    'signals': result.get('signals', []),
                    'signal_strength': result.get('signal_strength', 0),
                    'metadata': result.get('metadata', {}),
                    'timestamp': pd.Timestamp.now()
                }
            except Exception as e:
                logger.error(f"Error in rule-based model: {e}")
                return None
                
        elif model_obj.type == 'hybrid':
            # Apply hybrid model
            try:
                result = model_obj.model.analyze(features)
                return {
                    'signals': result.get('signals', []),
                    'signal_strength': result.get('signal_strength', 0),
                    'confidence': result.get('confidence', 0),
                    'metadata': result.get('metadata', {}),
                    'timestamp': pd.Timestamp.now()
                }
            except Exception as e:
                logger.error(f"Error in hybrid model: {e}")
                return None
        
        return None
    
    def _aggregate_signals(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate signals from multiple models
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Aggregated signal
        """
        if not model_results:
            return {
                'signal': 'neutral',
                'strength': 0,
                'confidence': 0,
                'models': {},
                'timestamp': pd.Timestamp.now()
            }
        
        # Extract signal strengths from each model
        signal_strengths = {}
        signal_metadata = {}
        confidences = {}
        
        for name, result in model_results.items():
            if result is None:
                continue
                
            model_obj = self.models[name]
            
            # Extract signal strength
            if 'signal_strength' in result:
                if isinstance(result['signal_strength'], (np.ndarray, list)):
                    # Use the last value for time series data
                    strength = result['signal_strength'][-1]
                else:
                    strength = result['signal_strength']
                
                # Apply model weight
                signal_strengths[name] = strength * model_obj.weight
            
            # Extract confidence if available
            if 'confidence' in result:
                confidences[name] = result['confidence']
            
            # Store metadata
            if 'metadata' in result:
                signal_metadata[name] = result['metadata']
        
        # Aggregation method
        aggregation = self.signal_config.aggregation
        
        if aggregation == 'weighted_average':
            # Weighted average of signal strengths
            total_weight = sum(self.models[name].weight for name in signal_strengths)
            if total_weight > 0:
                aggregated_strength = sum(strength for strength in signal_strengths.values()) / total_weight
            else:
                aggregated_strength = 0
        
        elif aggregation == 'max_confidence':
            # Use signal from model with highest confidence
            if confidences:
                max_conf_model = max(confidences.items(), key=lambda x: x[1])[0]
                aggregated_strength = signal_strengths.get(max_conf_model, 0)
            else:
                aggregated_strength = 0
        
        elif aggregation == 'voting':
            # Simple voting - count positive vs negative signals
            positive_votes = sum(1 for s in signal_strengths.values() if s > 0)
            negative_votes = sum(1 for s in signal_strengths.values() if s < 0)
            
            if positive_votes > negative_votes:
                aggregated_strength = positive_votes / len(signal_strengths)
            elif negative_votes > positive_votes:
                aggregated_strength = -negative_votes / len(signal_strengths)
            else:
                aggregated_strength = 0
        
        else:  # Default to simple average
            if signal_strengths:
                aggregated_strength = sum(signal_strengths.values()) / len(signal_strengths)
            else:
                aggregated_strength = 0
        
        # Determine signal type based on thresholds
        thresholds = self.signal_config.thresholds
        if aggregated_strength >= thresholds.get('buy', 0.7):
            signal_type = 'buy'
        elif aggregated_strength <= thresholds.get('sell', -0.7):
            signal_type = 'sell'
        else:
            signal_type = 'neutral'
        
        # Calculate average confidence
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
        
        # Create signal result
        signal_result = {
            'signal': signal_type,
            'strength': aggregated_strength,
            'confidence': avg_confidence,
            'models': {
                name: {
                    'strength': strength,
                    'confidence': confidences.get(name, 0.5),
                    'metadata': signal_metadata.get(name, {})
                }
                for name, strength in signal_strengths.items()
            },
            'timestamp': pd.Timestamp.now()
        }
        
        # Track signal strength
        symbol = list(model_results.values())[0].get('symbol', 'unknown') if model_results else 'unknown'
        if symbol not in self.metrics['signal_strength']:
            self.metrics['signal_strength'][symbol] = []
        
        self.metrics['signal_strength'][symbol].append({
            'timestamp': pd.Timestamp.now(),
            'strength': aggregated_strength,
            'signal': signal_type
        })
        
        return signal_result
    
    def _update_signal_history(self, signal: Dict[str, Any]):
        """
        Update signal history with new signal
        
        Args:
            signal: Signal dictionary
        """
        symbol = signal.get('symbol', 'default')
        
        # Initialize history for this symbol if not exists
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        # Add new signal to history
        self.signal_history[symbol].append(signal)
        
        # Maintain limited history
        max_history = self.signal_config.history_length
        if len(self.signal_history[symbol]) > max_history:
            self.signal_history[symbol] = self.signal_history[symbol][-max_history:]
    
    def start_async_processing(self):
        """Start asynchronous processing thread"""
        if self.running:
            logger.warning("Async processing is already running")
            return
            
        self.running = True
        self.processing_thread = Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Started async processing thread")
    
    def stop_async_processing(self):
        """Stop asynchronous processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            logger.info("Stopped async processing thread")
    
    def _processing_loop(self):
        """Background processing loop for async operation"""
        while self.running:
            try:
                # Get data from queue with timeout
                try:
                    data = self.data_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process data
                result = self.process_data(data)
                
                # Put result in result queue
                self.result_queue.put(result)
                
                # Mark task as done
                self.data_queue.task_done()
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
    
    def submit_data(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Submit data for asynchronous processing
        
        Args:
            data_dict: Dictionary of DataFrames with source name as key
        """
        self.data_queue.put(data_dict)
    
    def get_signal(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get next available signal from result queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Signal dictionary or None if no signal is available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_signals(self, symbol: str = 'default', count: int = 1) -> List[Dict[str, Any]]:
        """
        Get latest signals for a symbol
        
        Args:
            symbol: Symbol to get signals for
            count: Number of signals to return
            
        Returns:
            List of latest signals
        """
        if symbol not in self.signal_history:
            return []
            
        return self.signal_history[symbol][-count:]
    
    def clear_metrics(self):
        """Clear runtime metrics"""
        self.metrics = {
            'processing_time': [],
            'model_latency': {},
            'signal_strength': {},
            'data_freshness': {}
        }
        
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for models
        
        Returns:
            Dictionary with model performance metrics
        """
        performance = {}
        
        for name, latencies in self.metrics['model_latency'].items():
            if not latencies:
                continue
                
            # Calculate average latency
            avg_latency = sum(item['latency'] for item in latencies) / len(latencies)
            
            # Calculate 95th percentile latency
            sorted_latencies = sorted(item['latency'] for item in latencies)
            p95_index = int(0.95 * len(sorted_latencies))
            p95_latency = sorted_latencies[p95_index]
            
            performance[name] = {
                'avg_latency': avg_latency,
                'p95_latency': p95_latency,
                'call_count': len(latencies)
            }
        
        return performance
