import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set, Callable
from datetime import datetime, timedelta
import json
import os
import pickle
import time
import random
from collections import deque

from trading_bot.core.event_system import EventListener, Event

logger = logging.getLogger(__name__)

class TrainingSession:
    """
    Represents a model training session with metadata about training conditions,
    parameters, and outcomes.
    """
    
    def __init__(self, 
                session_id: str,
                strategy_id: str,
                model_type: str,
                start_time: datetime = None,
                training_data_range: Tuple[datetime, datetime] = None,
                parameters: Dict[str, Any] = None,
                metrics: Dict[str, float] = None,
                market_conditions: Dict[str, Any] = None,
                status: str = "pending"):
        """
        Initialize a training session
        
        Args:
            session_id: Unique identifier for the session
            strategy_id: Strategy ID this training is associated with
            model_type: Type of model being trained
            start_time: When the training started
            training_data_range: Data range used for training (start, end)
            parameters: Model/algorithm parameters used
            metrics: Performance metrics after training
            market_conditions: Market conditions during training
            status: Status of the training (pending, running, completed, failed)
        """
        self.session_id = session_id
        self.strategy_id = strategy_id
        self.model_type = model_type
        self.start_time = start_time or datetime.now()
        self.training_data_range = training_data_range
        self.parameters = parameters or {}
        self.metrics = metrics or {}
        self.market_conditions = market_conditions or {}
        self.status = status
        
        self.end_time = None
        self.duration_seconds = None
        self.notes = []
        self.validation_results = {}
        self.iterations = 0
        self.model_artifact_path = None
        
    def complete(self, metrics: Dict[str, float], model_path: str = None):
        """
        Mark the training session as complete
        
        Args:
            metrics: Performance metrics
            model_path: Path to the model artifact
        """
        self.status = "completed"
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.metrics.update(metrics)
        self.model_artifact_path = model_path
        
    def fail(self, error_message: str):
        """
        Mark the training session as failed
        
        Args:
            error_message: Error message
        """
        self.status = "failed"
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.add_note(f"Failed: {error_message}")
        
    def add_note(self, note: str):
        """
        Add a note to the session
        
        Args:
            note: Note text
        """
        self.notes.append({
            "text": note,
            "timestamp": datetime.now().isoformat()
        })
        
    def add_validation_result(self, result_type: str, data: Dict[str, Any]):
        """
        Add validation result
        
        Args:
            result_type: Type of validation result
            data: Validation data
        """
        self.validation_results[result_type] = {
            **data,
            "timestamp": datetime.now().isoformat()
        }
        
    def increment_iterations(self):
        """Increment the iteration counter"""
        self.iterations += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            "session_id": self.session_id,
            "strategy_id": self.strategy_id,
            "model_type": self.model_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "training_data_range": [d.isoformat() for d in self.training_data_range] if self.training_data_range else None,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "market_conditions": self.market_conditions,
            "status": self.status,
            "notes": self.notes,
            "validation_results": self.validation_results,
            "iterations": self.iterations,
            "model_artifact_path": self.model_artifact_path
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSession':
        """
        Create from dictionary data
        
        Args:
            data: Dictionary data
            
        Returns:
            TrainingSession instance
        """
        session = cls(
            session_id=data["session_id"],
            strategy_id=data["strategy_id"],
            model_type=data["model_type"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            market_conditions=data["market_conditions"],
            status=data["status"]
        )
        
        session.start_time = datetime.fromisoformat(data["start_time"])
        
        if data["end_time"]:
            session.end_time = datetime.fromisoformat(data["end_time"])
            
        session.duration_seconds = data["duration_seconds"]
        
        if data["training_data_range"]:
            session.training_data_range = tuple(datetime.fromisoformat(d) for d in data["training_data_range"])
            
        session.notes = data["notes"]
        session.validation_results = data["validation_results"]
        session.iterations = data["iterations"]
        session.model_artifact_path = data["model_artifact_path"]
        
        return session


class ModelParameter:
    """Represents a model parameter with constraints for optimization"""
    
    def __init__(self, 
                name: str,
                value_type: str,  # 'float', 'int', 'categorical'
                current_value: Any,
                min_value: float = None,
                max_value: float = None,
                step: float = None,
                choices: List[Any] = None,
                description: str = ""):
        """
        Initialize a model parameter
        
        Args:
            name: Parameter name
            value_type: Type of value (float, int, categorical)
            current_value: Current parameter value
            min_value: Minimum value for numerical parameters
            max_value: Maximum value for numerical parameters
            step: Step size for numerical parameters
            choices: Available choices for categorical parameters
            description: Parameter description
        """
        self.name = name
        self.value_type = value_type
        self.current_value = current_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.choices = choices
        self.description = description
        
        # Tuning history
        self.history = []
        
    def sample_value(self) -> Any:
        """
        Sample a random value within constraints
        
        Returns:
            Random value
        """
        if self.value_type == 'float':
            # Continuous value
            return random.uniform(self.min_value, self.max_value)
            
        elif self.value_type == 'int':
            # Discrete value
            return random.randint(self.min_value, self.max_value)
            
        elif self.value_type == 'categorical':
            # Categorical value
            return random.choice(self.choices)
            
        return self.current_value
        
    def update_value(self, new_value: Any):
        """
        Update the parameter value and record in history
        
        Args:
            new_value: New parameter value
        """
        # Record history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "old_value": self.current_value,
            "new_value": new_value
        })
        
        # Update value
        self.current_value = new_value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "value_type": self.value_type,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "choices": self.choices,
            "description": self.description,
            "history": self.history
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParameter':
        """
        Create from dictionary data
        
        Args:
            data: Dictionary data
            
        Returns:
            ModelParameter instance
        """
        param = cls(
            name=data["name"],
            value_type=data["value_type"],
            current_value=data["current_value"],
            min_value=data["min_value"],
            max_value=data["max_value"],
            step=data["step"],
            choices=data["choices"],
            description=data["description"]
        )
        
        param.history = data["history"]
        
        return param


class PerformanceTracker:
    """Tracks performance metrics over time"""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize performance tracker
        
        Args:
            max_history: Maximum number of data points to keep
        """
        self.metrics = {}
        self.max_history = max_history
        
    def add_metric_value(self, metric_name: str, value: float, timestamp: datetime = None):
        """
        Add a metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Timestamp (defaults to now)
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.max_history)
            
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": timestamp or datetime.now().isoformat()
        })
        
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values with timestamps
        """
        return list(self.metrics.get(metric_name, []))
        
    def get_metric_average(self, metric_name: str, window: int = None) -> Optional[float]:
        """
        Get average value for a metric
        
        Args:
            metric_name: Name of the metric
            window: Number of recent values to consider (None = all)
            
        Returns:
            Average value or None if no data
        """
        if metric_name not in self.metrics:
            return None
            
        values = self.metrics[metric_name]
        
        if not values:
            return None
            
        if window and window < len(values):
            values = list(values)[-window:]
            
        return np.mean([v["value"] for v in values])
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            "max_history": self.max_history,
            "metrics": {
                metric_name: list(values)
                for metric_name, values in self.metrics.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceTracker':
        """
        Create from dictionary data
        
        Args:
            data: Dictionary data
            
        Returns:
            PerformanceTracker instance
        """
        tracker = cls(max_history=data["max_history"])
        
        for metric_name, values in data["metrics"].items():
            tracker.metrics[metric_name] = deque(values, maxlen=tracker.max_history)
            
        return tracker


class EnhancedContinuousLearning(EventListener):
    """
    System for continuous learning and adaptation of trading strategies
    through automated experimentation and parameter tuning
    """
    
    def __init__(self, core_context, config: Dict[str, Any]):
        """
        Initialize enhanced continuous learning system
        
        Args:
            core_context: Core context containing references to other systems
            config: Configuration dictionary
        """
        self.core_context = core_context
        self.config = config
        
        # Configuration
        self.data_path = config.get("data_path", "data/continuous_learning")
        self.training_interval_hours = config.get("training_interval_hours", 24)
        self.auto_training_enabled = config.get("auto_training_enabled", True)
        self.model_retention_days = config.get("model_retention_days", 30)
        self.performance_threshold = config.get("performance_threshold", 0.6)
        self.experiment_batch_size = config.get("experiment_batch_size", 5)
        self.max_concurrent_trainings = config.get("max_concurrent_trainings", 2)
        
        # Storage
        self.strategy_parameters: Dict[str, Dict[str, ModelParameter]] = {}
        self.training_sessions: Dict[str, TrainingSession] = {}
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        self.last_training_times: Dict[str, datetime] = {}
        
        # Status tracking
        self.current_trainings: Set[str] = set()  # session_ids of active trainings
        self.training_queue: List[Dict[str, Any]] = []
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Custom functions that can be registered
        self.training_functions: Dict[str, Callable] = {}
        self.evaluation_functions: Dict[str, Callable] = {}
        
        # Initialize
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        
        # Load data from disk
        self.load_data()
        
        # Register event handlers
        self.register_event_handlers()
        
        logger.info("Enhanced Continuous Learning system initialized")
        
    def register_event_handlers(self):
        """Register event handlers"""
        event_system = self.core_context.event_system
        
        event_system.register_handler("strategy_performance_update", self.handle_strategy_performance_update)
        event_system.register_handler("market_regime_change", self.handle_market_regime_change)
        event_system.register_handler("hourly_update", self.handle_hourly_update)
        event_system.register_handler("daily_summary", self.handle_daily_summary)
        event_system.register_handler("training_completed", self.handle_training_completed)
        event_system.register_handler("model_evaluation_completed", self.handle_model_evaluation_completed)
        
    def load_data(self):
        """Load data from disk"""
        self.load_strategy_parameters()
        self.load_training_sessions()
        self.load_performance_trackers()
        
    def load_strategy_parameters(self):
        """Load strategy parameters from disk"""
        params_path = os.path.join(self.data_path, "strategy_parameters.json")
        
        if not os.path.exists(params_path):
            logger.info("No strategy parameters found on disk")
            return
            
        try:
            with open(params_path, 'r') as f:
                params_data = json.load(f)
                
            for strategy_id, parameters in params_data.items():
                self.strategy_parameters[strategy_id] = {}
                
                for param_name, param_data in parameters.items():
                    self.strategy_parameters[strategy_id][param_name] = ModelParameter.from_dict(param_data)
                    
            logger.info(f"Loaded parameters for {len(self.strategy_parameters)} strategies")
                
        except Exception as e:
            logger.error(f"Error loading strategy parameters: {e}")
            
    def load_training_sessions(self):
        """Load training sessions from disk"""
        sessions_path = os.path.join(self.data_path, "training_sessions.json")
        
        if not os.path.exists(sessions_path):
            logger.info("No training sessions found on disk")
            return
            
        try:
            with open(sessions_path, 'r') as f:
                sessions_data = json.load(f)
                
            for session_id, session_data in sessions_data.items():
                self.training_sessions[session_id] = TrainingSession.from_dict(session_data)
                
            logger.info(f"Loaded {len(self.training_sessions)} training sessions")
                
        except Exception as e:
            logger.error(f"Error loading training sessions: {e}")
            
    def load_performance_trackers(self):
        """Load performance trackers from disk"""
        trackers_path = os.path.join(self.data_path, "performance_trackers.json")
        
        if not os.path.exists(trackers_path):
            logger.info("No performance trackers found on disk")
            return
            
        try:
            with open(trackers_path, 'r') as f:
                trackers_data = json.load(f)
                
            for strategy_id, tracker_data in trackers_data.items():
                self.performance_trackers[strategy_id] = PerformanceTracker.from_dict(tracker_data)
                
            logger.info(f"Loaded performance trackers for {len(self.performance_trackers)} strategies")
                
        except Exception as e:
            logger.error(f"Error loading performance trackers: {e}")
            
    def save_data(self):
        """Save all data to disk"""
        self.save_strategy_parameters()
        self.save_training_sessions()
        self.save_performance_trackers()
        
    def save_strategy_parameters(self):
        """Save strategy parameters to disk"""
        params_path = os.path.join(self.data_path, "strategy_parameters.json")
        
        try:
            params_data = {}
            
            for strategy_id, parameters in self.strategy_parameters.items():
                params_data[strategy_id] = {
                    param_name: param.to_dict()
                    for param_name, param in parameters.items()
                }
                
            with open(params_path, 'w') as f:
                json.dump(params_data, f, indent=2)
                
            logger.info(f"Saved parameters for {len(params_data)} strategies")
                
        except Exception as e:
            logger.error(f"Error saving strategy parameters: {e}")
            
    def save_training_sessions(self):
        """Save training sessions to disk"""
        sessions_path = os.path.join(self.data_path, "training_sessions.json")
        
        try:
            sessions_data = {
                session_id: session.to_dict()
                for session_id, session in self.training_sessions.items()
            }
            
            with open(sessions_path, 'w') as f:
                json.dump(sessions_data, f, indent=2)
                
            logger.info(f"Saved {len(sessions_data)} training sessions")
                
        except Exception as e:
            logger.error(f"Error saving training sessions: {e}")
            
    def save_performance_trackers(self):
        """Save performance trackers to disk"""
        trackers_path = os.path.join(self.data_path, "performance_trackers.json")
        
        try:
            trackers_data = {
                strategy_id: tracker.to_dict()
                for strategy_id, tracker in self.performance_trackers.items()
            }
            
            with open(trackers_path, 'w') as f:
                json.dump(trackers_data, f, indent=2)
                
            logger.info(f"Saved performance trackers for {len(trackers_data)} strategies")
                
        except Exception as e:
            logger.error(f"Error saving performance trackers: {e}")
            
    def handle_strategy_performance_update(self, event: Event):
        """
        Handle strategy performance update event
        
        Args:
            event: Strategy performance update event
        """
        performance_data = event.data
        strategy_id = performance_data.get("strategy_id")
        metrics = performance_data.get("metrics", {})
        
        if not strategy_id:
            return
            
        # Initialize performance tracker if needed
        if strategy_id not in self.performance_trackers:
            self.performance_trackers[strategy_id] = PerformanceTracker()
            
        # Add metrics to tracker
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.performance_trackers[strategy_id].add_metric_value(metric_name, value)
                
        # Check if training is needed
        self.check_if_training_needed(strategy_id)
        
    def handle_market_regime_change(self, event: Event):
        """
        Handle market regime change event
        
        Args:
            event: Market regime change event
        """
        regime_data = event.data
        new_regime = regime_data.get("regime")
        
        if new_regime:
            # Consider training models when regime changes
            self.schedule_regime_training(new_regime)
            
    def handle_hourly_update(self, event: Event):
        """
        Handle hourly update event
        
        Args:
            event: Hourly update event
        """
        # Process training queue
        self.process_training_queue()
        
        # Clean up old models
        self.cleanup_old_models()
        
    def handle_daily_summary(self, event: Event):
        """
        Handle daily summary event
        
        Args:
            event: Daily summary event
        """
        # Save data
        self.save_data()
        
        # Generate learning summary
        self.generate_learning_summary()
        
    def handle_training_completed(self, event: Event):
        """
        Handle training completed event
        
        Args:
            event: Training completed event
        """
        training_data = event.data
        session_id = training_data.get("session_id")
        
        if session_id and session_id in self.training_sessions:
            session = self.training_sessions[session_id]
            
            # Mark as completed
            session.complete(
                metrics=training_data.get("metrics", {}),
                model_path=training_data.get("model_path")
            )
            
            # Remove from current trainings
            if session_id in self.current_trainings:
                self.current_trainings.remove(session_id)
                
            # Evaluate new model
            self.schedule_model_evaluation(session)
            
            # Process next in queue
            self.process_training_queue()
            
    def handle_model_evaluation_completed(self, event: Event):
        """
        Handle model evaluation completed event
        
        Args:
            event: Model evaluation completed event
        """
        eval_data = event.data
        session_id = eval_data.get("session_id")
        
        if session_id and session_id in self.training_sessions:
            session = self.training_sessions[session_id]
            
            # Add evaluation results
            result_type = eval_data.get("evaluation_type", "default")
            session.add_validation_result(result_type, eval_data.get("results", {}))
            
            # Check if we should deploy the model
            if self.should_deploy_model(session):
                self.deploy_model(session)
                
    def register_training_function(self, model_type: str, func: Callable):
        """
        Register a training function for a model type
        
        Args:
            model_type: Type of model to be trained
            func: Training function that will be called
        """
        self.training_functions[model_type] = func
        logger.info(f"Registered training function for model type: {model_type}")
        
    def register_evaluation_function(self, model_type: str, func: Callable):
        """
        Register an evaluation function for a model type
        
        Args:
            model_type: Type of model to be evaluated
            func: Evaluation function that will be called
        """
        self.evaluation_functions[model_type] = func
        logger.info(f"Registered evaluation function for model type: {model_type}")
        
    def define_strategy_parameters(self, strategy_id: str, parameters: Dict[str, Dict[str, Any]]):
        """
        Define parameters for a strategy
        
        Args:
            strategy_id: Strategy ID
            parameters: Dictionary mapping parameter names to parameter specs
        """
        if strategy_id not in self.strategy_parameters:
            self.strategy_parameters[strategy_id] = {}
            
        for param_name, param_spec in parameters.items():
            self.strategy_parameters[strategy_id][param_name] = ModelParameter(
                name=param_name,
                value_type=param_spec.get("value_type", "float"),
                current_value=param_spec.get("current_value"),
                min_value=param_spec.get("min_value"),
                max_value=param_spec.get("max_value"),
                step=param_spec.get("step"),
                choices=param_spec.get("choices"),
                description=param_spec.get("description", "")
            )
            
        logger.info(f"Defined {len(parameters)} parameters for strategy {strategy_id}")
        
    def get_strategy_parameter_values(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get current parameter values for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary mapping parameter names to current values
        """
        if strategy_id not in self.strategy_parameters:
            return {}
            
        return {
            param_name: param.current_value
            for param_name, param in self.strategy_parameters[strategy_id].items()
        }
        
    def update_strategy_parameter(self, strategy_id: str, param_name: str, value: Any):
        """
        Update a strategy parameter value
        
        Args:
            strategy_id: Strategy ID
            param_name: Parameter name
            value: New parameter value
        """
        if (strategy_id in self.strategy_parameters and 
            param_name in self.strategy_parameters[strategy_id]):
            
            self.strategy_parameters[strategy_id][param_name].update_value(value)
            logger.info(f"Updated parameter {param_name} for strategy {strategy_id}: {value}")
            
    def check_if_training_needed(self, strategy_id: str):
        """
        Check if training is needed for a strategy
        
        Args:
            strategy_id: Strategy ID
        """
        # Skip if auto-training is disabled
        if not self.auto_training_enabled:
            return
            
        # Skip if no performance data available
        if strategy_id not in self.performance_trackers:
            return
            
        # Check last training time
        last_time = self.last_training_times.get(strategy_id)
        
        if last_time:
            hours_since_last = (datetime.now() - last_time).total_seconds() / 3600
            
            if hours_since_last < self.training_interval_hours:
                # Not enough time has passed since last training
                return
                
        # Check if performance is below threshold
        tracker = self.performance_trackers[strategy_id]
        
        # Different metrics may be used depending on strategy
        for metric_name in ["sharpe_ratio", "win_rate", "pnl"]:
            avg_value = tracker.get_metric_average(metric_name, window=10)
            
            if avg_value is not None:
                if metric_name == "sharpe_ratio" and avg_value < 0.8:
                    self.schedule_training_experiments(strategy_id)
                    return
                    
                if metric_name == "win_rate" and avg_value < 0.55:
                    self.schedule_training_experiments(strategy_id)
                    return
                    
                if metric_name == "pnl" and avg_value < 0:
                    self.schedule_training_experiments(strategy_id)
                    return
                    
    def schedule_regime_training(self, regime: str):
        """
        Schedule training for a specific market regime
        
        Args:
            regime: Market regime name
        """
        # Get strategy factory
        strategy_factory = getattr(self.core_context, "strategy_factory", None)
        
        if not strategy_factory:
            logger.warning("Strategy factory not available")
            return
            
        # Find strategies suitable for this regime
        suitable_strategies = []
        
        for strategy_id, info in strategy_factory.get_available_strategies().items():
            regime_suitability = info.get("regime_suitability", {})
            
            if regime in regime_suitability and regime_suitability[regime] > 0.7:
                suitable_strategies.append(strategy_id)
                
        # Schedule training for suitable strategies
        for strategy_id in suitable_strategies:
            # Only schedule if we have parameters defined
            if strategy_id in self.strategy_parameters:
                self.schedule_training_experiments(strategy_id, 
                                                context={"regime": regime},
                                                priority=1)
                logger.info(f"Scheduled regime-specific training for {strategy_id} (regime: {regime})")
                
    def schedule_training_experiments(self, strategy_id: str, context: Dict[str, Any] = None, priority: int = 0):
        """
        Schedule training experiments for a strategy
        
        Args:
            strategy_id: Strategy ID
            context: Additional context for training
            priority: Priority (higher = more important)
        """
        # Skip if no parameters defined
        if strategy_id not in self.strategy_parameters:
            logger.warning(f"No parameters defined for strategy {strategy_id}")
            return
            
        # Get strategy info
        strategy_factory = getattr(self.core_context, "strategy_factory", None)
        
        if not strategy_factory:
            logger.warning("Strategy factory not available")
            return
            
        strategy_info = strategy_factory.get_strategy_info(strategy_id)
        
        if not strategy_info:
            logger.warning(f"No info available for strategy {strategy_id}")
            return
            
        # Get model type
        model_type = strategy_info.get("model_type", "default")
        
        # Check if we have a training function
        if model_type not in self.training_functions:
            logger.warning(f"No training function for model type {model_type}")
            return
            
        # Generate experiments with parameter variations
        params = self.strategy_parameters[strategy_id]
        
        # Create experiments
        for _ in range(self.experiment_batch_size):
            # Generate session ID
            session_id = f"{strategy_id}_{model_type}_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Create parameter variations
            param_variations = {}
            
            for param_name, param in params.items():
                # Only vary some parameters (random selection)
                if random.random() < 0.7:  # 70% chance to vary each parameter
                    param_variations[param_name] = param.sample_value()
                else:
                    param_variations[param_name] = param.current_value
                    
            # Create training session
            session = TrainingSession(
                session_id=session_id,
                strategy_id=strategy_id,
                model_type=model_type,
                parameters=param_variations,
                market_conditions=context or {}
            )
            
            self.training_sessions[session_id] = session
            
            # Add to queue
            self.training_queue.append({
                "session_id": session_id,
                "priority": priority
            })
            
        # Update last training time
        self.last_training_times[strategy_id] = datetime.now()
        
        logger.info(f"Scheduled {self.experiment_batch_size} training experiments for {strategy_id}")
        
        # Process queue immediately if possible
        self.process_training_queue()
        
    def process_training_queue(self):
        """Process the training queue"""
        # Skip if queue is empty
        if not self.training_queue:
            return
            
        # Check if we can start more trainings
        while (len(self.current_trainings) < self.max_concurrent_trainings and 
              self.training_queue):
            
            # Sort by priority (descending)
            self.training_queue.sort(key=lambda x: x["priority"], reverse=True)
            
            # Get next training
            next_training = self.training_queue.pop(0)
            session_id = next_training["session_id"]
            
            # Skip if session doesn't exist
            if session_id not in self.training_sessions:
                continue
                
            session = self.training_sessions[session_id]
            
            # Skip if already completed or failed
            if session.status in ["completed", "failed"]:
                continue
                
            # Start training
            self.start_training_session(session)
            
    def start_training_session(self, session: TrainingSession):
        """
        Start a training session
        
        Args:
            session: Training session to start
        """
        # Mark as running
        session.status = "running"
        self.current_trainings.add(session.session_id)
        
        # Get training function
        training_func = self.training_functions.get(session.model_type)
        
        if not training_func:
            session.fail("No training function available")
            self.current_trainings.remove(session.session_id)
            return
            
        try:
            # Start training in background (async)
            self.core_context.task_manager.submit_task(
                f"training_{session.session_id}",
                lambda: self._execute_training(training_func, session)
            )
            
            logger.info(f"Started training session: {session.session_id}")
            
        except Exception as e:
            session.fail(str(e))
            self.current_trainings.remove(session.session_id)
            logger.error(f"Failed to start training session: {e}")
            
    def _execute_training(self, training_func: Callable, session: TrainingSession):
        """
        Execute a training function
        
        Args:
            training_func: Training function to call
            session: Training session
        """
        try:
            # Call training function
            result = training_func(
                strategy_id=session.strategy_id,
                parameters=session.parameters,
                context=session.market_conditions
            )
            
            # Process result
            if isinstance(result, dict):
                metrics = result.get("metrics", {})
                model_path = result.get("model_path")
                
                # Emit event
                self.core_context.event_system.emit_event(
                    Event("training_completed", {
                        "session_id": session.session_id,
                        "strategy_id": session.strategy_id,
                        "metrics": metrics,
                        "model_path": model_path
                    })
                )
            else:
                # Failed
                session.fail("Training function returned invalid result")
                
                # Remove from current trainings
                if session.session_id in self.current_trainings:
                    self.current_trainings.remove(session.session_id)
                    
        except Exception as e:
            # Failed
            session.fail(str(e))
            
            # Remove from current trainings
            if session.session_id in self.current_trainings:
                self.current_trainings.remove(session.session_id)
                
            logger.error(f"Training failed: {e}")
            
    def schedule_model_evaluation(self, session: TrainingSession):
        """
        Schedule model evaluation
        
        Args:
            session: Training session with model to evaluate
        """
        # Skip if no model artifact
        if not session.model_artifact_path:
            return
            
        # Get evaluation function
        eval_func = self.evaluation_functions.get(session.model_type)
        
        if not eval_func:
            logger.warning(f"No evaluation function for model type {session.model_type}")
            return
            
        try:
            # Start evaluation in background (async)
            self.core_context.task_manager.submit_task(
                f"eval_{session.session_id}",
                lambda: self._execute_evaluation(eval_func, session)
            )
            
            logger.info(f"Started evaluation for session: {session.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to start evaluation: {e}")
            
    def _execute_evaluation(self, eval_func: Callable, session: TrainingSession):
        """
        Execute an evaluation function
        
        Args:
            eval_func: Evaluation function to call
            session: Training session
        """
        try:
            # Call evaluation function
            result = eval_func(
                strategy_id=session.strategy_id,
                model_path=session.model_artifact_path,
                parameters=session.parameters
            )
            
            # Process result
            if isinstance(result, dict):
                # Emit event
                self.core_context.event_system.emit_event(
                    Event("model_evaluation_completed", {
                        "session_id": session.session_id,
                        "strategy_id": session.strategy_id,
                        "evaluation_type": result.get("evaluation_type", "default"),
                        "results": result.get("results", {})
                    })
                )
            else:
                logger.warning(f"Evaluation function returned invalid result for {session.session_id}")
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            
    def should_deploy_model(self, session: TrainingSession) -> bool:
        """
        Determine if a model should be deployed
        
        Args:
            session: Training session with evaluated model
            
        Returns:
            True if model should be deployed, False otherwise
        """
        # Check if model passed validation
        validation_results = session.validation_results
        
        if not validation_results:
            return False
            
        # Check metrics
        for result_type, result_data in validation_results.items():
            # Don't deploy if any validation failed
            if result_data.get("is_valid") is False:
                return False
                
            # Check if significantly better than baseline
            improvement = result_data.get("improvement_over_baseline")
            
            if improvement is not None and improvement < 0.05:  # 5% improvement threshold
                return False
                
        # Model passed validation checks
        return True
        
    def deploy_model(self, session: TrainingSession):
        """
        Deploy a model
        
        Args:
            session: Training session with model to deploy
        """
        # Update parameters
        strategy_id = session.strategy_id
        
        if strategy_id in self.strategy_parameters:
            for param_name, value in session.parameters.items():
                if param_name in self.strategy_parameters[strategy_id]:
                    self.update_strategy_parameter(strategy_id, param_name, value)
                    
        # Record deployment
        self.deployment_history.append({
            "timestamp": datetime.now().isoformat(),
            "session_id": session.session_id,
            "strategy_id": strategy_id,
            "model_path": session.model_artifact_path,
            "metrics": session.metrics,
            "parameters": session.parameters
        })
        
        # Notify strategy factory
        strategy_factory = getattr(self.core_context, "strategy_factory", None)
        
        if strategy_factory:
            try:
                # Update strategy with new model/parameters
                strategy_factory.update_strategy(
                    strategy_id=strategy_id,
                    model_path=session.model_artifact_path,
                    parameters=session.parameters
                )
            except Exception as e:
                logger.error(f"Failed to update strategy in factory: {e}")
                
        # Emit event
        self.core_context.event_system.emit_event(
            Event("model_deployed", {
                "session_id": session.session_id,
                "strategy_id": strategy_id,
                "model_path": session.model_artifact_path,
                "parameters": session.parameters,
                "timestamp": datetime.now().isoformat()
            })
        )
        
        logger.info(f"Deployed model for strategy {strategy_id} from session {session.session_id}")
        
    def cleanup_old_models(self):
        """Clean up old model artifacts"""
        # Skip if not enabled
        if self.model_retention_days <= 0:
            return
            
        cutoff_date = datetime.now() - timedelta(days=self.model_retention_days)
        models_dir = os.path.join(self.data_path, "models")
        
        try:
            for session_id, session in self.training_sessions.items():
                # Skip recent sessions
                if session.start_time >= cutoff_date:
                    continue
                    
                # Skip sessions without model artifacts
                if not session.model_artifact_path:
                    continue
                    
                # Skip if model is still in use
                model_still_used = False
                
                for deploy_record in self.deployment_history[-10:]:  # Check recent deployments
                    if deploy_record.get("model_path") == session.model_artifact_path:
                        model_still_used = True
                        break
                        
                if model_still_used:
                    continue
                    
                # Delete model file if it exists
                if os.path.exists(session.model_artifact_path):
                    os.remove(session.model_artifact_path)
                    logger.info(f"Deleted old model artifact: {session.model_artifact_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
            
    def generate_learning_summary(self):
        """Generate a summary of the learning system state"""
        now = datetime.now()
        
        summary = {
            "timestamp": now.isoformat(),
            "total_training_sessions": len(self.training_sessions),
            "active_trainings": len(self.current_trainings),
            "queued_trainings": len(self.training_queue),
            "recent_deployments": [
                {
                    "timestamp": d["timestamp"],
                    "strategy_id": d["strategy_id"],
                    "session_id": d["session_id"]
                }
                for d in self.deployment_history[-5:]  # Last 5 deployments
            ],
            "strategy_metrics": {}
        }
        
        # Add strategy metrics
        for strategy_id, tracker in self.performance_trackers.items():
            metrics_summary = {}
            
            for metric_name in ["sharpe_ratio", "win_rate", "pnl"]:
                avg_value = tracker.get_metric_average(metric_name, window=10)
                
                if avg_value is not None:
                    metrics_summary[metric_name] = avg_value
                    
            last_training = self.last_training_times.get(strategy_id)
            
            summary["strategy_metrics"][strategy_id] = {
                "metrics": metrics_summary,
                "last_training": last_training.isoformat() if last_training else None,
                "parameter_count": len(self.strategy_parameters.get(strategy_id, {}))
            }
            
        # Write to summary file
        summary_path = os.path.join(self.data_path, "learning_summary.json")
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Saved learning summary to {summary_path}")
                
        except Exception as e:
            logger.error(f"Error saving learning summary: {e}")
            
        # Also emit event with summary
        self.core_context.event_system.emit_event(
            Event("continuous_learning_summary", summary)
        ) 