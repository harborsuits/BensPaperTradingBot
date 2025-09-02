import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
import joblib
import os
import json
import random
from collections import defaultdict, deque

from trading_bot.core.event_system import EventListener, Event
from trading_bot.strategy_management.strategy_base import Strategy
from trading_bot.market_data.market_context import MarketContext

logger = logging.getLogger(__name__)

class PerformanceMetricsTracker:
    """Tracks and analyzes strategy performance metrics for learning purposes"""
    
    def __init__(self, storage_path: str = None):
        self.performance_data = {}
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        if storage_path and not os.path.exists(storage_path):
            os.makedirs(storage_path)
    
    def update_performance(self, strategy_id: str, metrics: Dict[str, Any]):
        """
        Update performance metrics for a strategy
        
        Args:
            strategy_id: ID of the strategy
            metrics: Dictionary of metrics to update
        """
        if strategy_id not in self.performance_data:
            self.performance_data[strategy_id] = []
            
        # Add timestamp if not provided
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now()
            
        self.performance_data[strategy_id].append(metrics)
        
        # Truncate to keep only most recent 1000 entries
        if len(self.performance_data[strategy_id]) > 1000:
            self.performance_data[strategy_id] = self.performance_data[strategy_id][-1000:]
            
    def get_strategy_performance(self, strategy_id: str, window: int = None) -> pd.DataFrame:
        """
        Get performance data for a strategy as a DataFrame
        
        Args:
            strategy_id: ID of the strategy
            window: Optional window of most recent entries to return
            
        Returns:
            DataFrame with performance metrics
        """
        if strategy_id not in self.performance_data:
            return pd.DataFrame()
            
        data = self.performance_data[strategy_id]
        if window and window < len(data):
            data = data[-window:]
            
        return pd.DataFrame(data)
        
    def save_metrics(self):
        """Save performance metrics to disk"""
        if not self.storage_path:
            logger.warning("No storage path set for performance metrics")
            return
            
        try:
            for strategy_id, metrics in self.performance_data.items():
                file_path = os.path.join(self.storage_path, f"{strategy_id}_metrics.joblib")
                joblib.dump(metrics, file_path)
                logger.debug(f"Saved performance metrics for strategy {strategy_id}")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            
    def load_metrics(self):
        """Load performance metrics from disk"""
        if not self.storage_path:
            logger.warning("No storage path set for performance metrics")
            return
            
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith("_metrics.joblib"):
                    strategy_id = filename.replace("_metrics.joblib", "")
                    file_path = os.path.join(self.storage_path, filename)
                    
                    self.performance_data[strategy_id] = joblib.load(file_path)
                    logger.debug(f"Loaded performance metrics for strategy {strategy_id}")
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            

class StrategyLearningModel:
    """Model to learn and adapt strategy parameters based on performance"""
    
    def __init__(self, 
                strategy_id: str, 
                strategy_config: Dict[str, Any],
                learning_config: Dict[str, Any]):
        """
        Initialize the learning model for a strategy
        
        Args:
            strategy_id: ID of the strategy
            strategy_config: Current configuration of the strategy
            learning_config: Configuration for the learning process
        """
        self.strategy_id = strategy_id
        self.base_config = strategy_config.copy()
        self.current_config = strategy_config.copy()
        self.learning_config = learning_config
        
        # Performance tracking
        self.performance_history = deque(maxlen=learning_config.get("history_size", 100))
        self.parameter_history = deque(maxlen=learning_config.get("history_size", 100))
        
        # Learning parameters
        self.exploration_rate = learning_config.get("initial_exploration_rate", 0.2)
        self.learning_rate = learning_config.get("learning_rate", 0.1)
        self.min_exploration_rate = learning_config.get("min_exploration_rate", 0.05)
        self.exploration_decay = learning_config.get("exploration_decay", 0.995)
        
        # Parameter constraints
        self.parameter_constraints = learning_config.get("parameter_constraints", {})
        
        # Performance baseline
        self.baseline_performance = None
        self.best_performance = float('-inf')
        self.best_config = None
        
        # Last updated
        self.last_update = datetime.now()
        self.iterations = 0
        
    def update_performance(self, 
                          performance_metrics: Dict[str, float], 
                          timestamp: datetime = None):
        """
        Update performance history with new metrics
        
        Args:
            performance_metrics: Dictionary of performance metrics
            timestamp: Timestamp of the performance update
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Calculate composite performance score
        composite_score = self._calculate_composite_score(performance_metrics)
        
        # Add to history
        self.performance_history.append({
            "timestamp": timestamp,
            "metrics": performance_metrics,
            "composite_score": composite_score,
            "config": self.current_config.copy()
        })
        
        # Update baseline if needed
        if self.baseline_performance is None:
            self.baseline_performance = composite_score
            
        # Update best performance
        if composite_score > self.best_performance:
            self.best_performance = composite_score
            self.best_config = self.current_config.copy()
            
        # Log performance
        logger.debug(f"Strategy {self.strategy_id} performance score: {composite_score:.4f}")
        
        # Update iteration count
        self.iterations += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite performance score from metrics
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Composite score as a single float
        """
        # Define weights for different metrics
        weights = {
            "return": 5.0,
            "sharpe_ratio": 3.0,
            "win_rate": 1.0,
            "max_drawdown": -2.0,  # Negative weight since lower is better
            "volatility": -1.0,    # Negative weight since lower is better
            "profit_factor": 2.0
        }
        
        score = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # Special handling for max_drawdown (make positive)
                if metric == "max_drawdown" and value < 0:
                    value = -value
                    
                score += value * weight
                
        return score
        
    def generate_parameter_variation(self) -> Dict[str, Any]:
        """
        Generate a variation of the current parameters for exploration
        
        Returns:
            New parameter configuration
        """
        # Start with current config
        new_config = self.current_config.copy()
        
        # Decide if we should explore or exploit
        if random.random() < self.exploration_rate:
            # Select parameters to modify
            tunable_params = self._get_tunable_parameters()
            
            if not tunable_params:
                return new_config
                
            # Randomly select how many parameters to modify (1 to 3)
            num_params = min(len(tunable_params), random.randint(1, 3))
            params_to_modify = random.sample(tunable_params, num_params)
            
            # Modify selected parameters
            for param in params_to_modify:
                new_config[param] = self._generate_parameter_value(param)
                
            logger.debug(f"Generated parameter variation for {self.strategy_id}: {params_to_modify}")
            
        else:
            # Use best configuration found so far
            if self.best_config is not None:
                new_config = self.best_config.copy()
                
        return new_config
        
    def _get_tunable_parameters(self) -> List[str]:
        """
        Get list of parameters that can be tuned
        
        Returns:
            List of parameter names
        """
        # Parameters that shouldn't be tuned
        excluded = ["id", "name", "type", "description", "enabled", "symbols"]
        
        # Get parameters from constraints if defined
        if self.parameter_constraints:
            return list(self.parameter_constraints.keys())
            
        # Otherwise, find numerical parameters
        tunable = []
        for key, value in self.current_config.items():
            if (key not in excluded and 
                (isinstance(value, (int, float)) or 
                (isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)))):
                tunable.append(key)
                
        return tunable
        
    def _generate_parameter_value(self, param_name: str) -> Any:
        """
        Generate a new value for a parameter
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            New parameter value
        """
        current_value = self.current_config.get(param_name)
        
        # Check if we have constraints for this parameter
        if param_name in self.parameter_constraints:
            constraints = self.parameter_constraints[param_name]
            param_type = constraints.get("type", type(current_value).__name__)
            
            if param_type == "int":
                min_val = constraints.get("min", max(1, int(current_value * 0.5)))
                max_val = constraints.get("max", int(current_value * 2.0))
                return random.randint(min_val, max_val)
                
            elif param_type == "float":
                min_val = constraints.get("min", current_value * 0.5)
                max_val = constraints.get("max", current_value * 2.0)
                return random.uniform(min_val, max_val)
                
            elif param_type == "bool":
                return random.choice([True, False])
                
            elif param_type == "list":
                # For lists of numbers, adjust each element
                if isinstance(current_value, list) and constraints.get("element_type") == "float":
                    new_list = []
                    for val in current_value:
                        min_val = constraints.get("min", val * 0.5)
                        max_val = constraints.get("max", val * 2.0)
                        new_list.append(random.uniform(min_val, max_val))
                    return new_list
                else:
                    return current_value  # Don't modify other list types
                    
            else:
                return current_value  # Don't modify unknown types
                
        else:
            # No constraints, use simple heuristics
            if isinstance(current_value, bool):
                return not current_value
            elif isinstance(current_value, int):
                variation = max(1, int(current_value * 0.3))
                return max(1, current_value + random.randint(-variation, variation))
            elif isinstance(current_value, float):
                variation = current_value * 0.3
                return max(0.0001, current_value + random.uniform(-variation, variation))
            else:
                return current_value  # Don't modify other types
                
    def apply_parameter_update(self, new_config: Dict[str, Any]):
        """
        Apply parameter update and record it
        
        Args:
            new_config: New parameter configuration
        """
        # Record parameter change
        self.parameter_history.append({
            "timestamp": datetime.now(),
            "old_config": self.current_config.copy(),
            "new_config": new_config.copy()
        })
        
        # Apply new configuration
        self.current_config = new_config.copy()
        self.last_update = datetime.now()
        
    def should_update_parameters(self) -> bool:
        """
        Determine if parameters should be updated
        
        Returns:
            True if parameters should be updated, False otherwise
        """
        # Check if we have enough performance history
        if len(self.performance_history) < 5:
            return False
            
        # Check if it's been long enough since last update
        min_hours_between_updates = self.learning_config.get("min_hours_between_updates", 4)
        hours_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        
        if hours_since_update < min_hours_between_updates:
            return False
            
        # Check if performance is declining
        if len(self.performance_history) >= 10:
            recent_scores = [p["composite_score"] for p in list(self.performance_history)[-10:]]
            if sum(recent_scores[:5]) < sum(recent_scores[5:]) * 0.8:
                # Performance declined by more than 20%
                return True
                
        # Regular updates based on configuration
        update_frequency = self.learning_config.get("update_frequency_hours", 24)
        if hours_since_update >= update_frequency:
            return True
            
        return False
        
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current parameter configuration
        
        Returns:
            Current configuration dictionary
        """
        return self.current_config.copy()
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics
        
        Returns:
            Dictionary with learning statistics
        """
        return {
            "iterations": self.iterations,
            "exploration_rate": self.exploration_rate,
            "baseline_performance": self.baseline_performance,
            "best_performance": self.best_performance,
            "last_update": self.last_update.isoformat(),
            "parameter_updates": len(self.parameter_history)
        }
        
    def reset_to_baseline(self):
        """Reset parameters to baseline configuration"""
        self.current_config = self.base_config.copy()
        logger.info(f"Reset strategy {self.strategy_id} parameters to baseline")
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for serialization
        
        Returns:
            Dictionary representation of the model
        """
        return {
            "strategy_id": self.strategy_id,
            "base_config": self.base_config,
            "current_config": self.current_config,
            "learning_config": self.learning_config,
            "best_config": self.best_config,
            "best_performance": self.best_performance,
            "baseline_performance": self.baseline_performance,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "last_update": self.last_update.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyLearningModel':
        """
        Create model from dictionary data
        
        Args:
            data: Dictionary data
            
        Returns:
            StrategyLearningModel instance
        """
        model = cls(
            strategy_id=data["strategy_id"],
            strategy_config=data["base_config"],
            learning_config=data["learning_config"]
        )
        
        model.current_config = data["current_config"]
        model.best_config = data["best_config"]
        model.best_performance = data["best_performance"]
        model.baseline_performance = data["baseline_performance"]
        model.exploration_rate = data["exploration_rate"]
        model.learning_rate = data["learning_rate"]
        model.iterations = data["iterations"]
        
        if "last_update" in data:
            model.last_update = datetime.fromisoformat(data["last_update"])
            
        return model


class EnhancedContinuousLearningSystem(EventListener):
    """
    System that continuously monitors and adapts trading strategies
    based on their performance and market conditions
    """
    
    def __init__(self, core_context, config: Dict[str, Any]):
        """
        Initialize the continuous learning system
        
        Args:
            core_context: Core context containing references to other systems
            config: Configuration dictionary
        """
        self.core_context = core_context
        self.config = config
        
        # System configuration
        self.data_path = config.get("data_path", "data/continuous_learning")
        self.enable_parameter_learning = config.get("enable_parameter_learning", True)
        self.enable_regime_learning = config.get("enable_regime_learning", True)
        self.save_frequency_hours = config.get("save_frequency_hours", 24)
        
        # Learning models
        self.learning_models: Dict[str, StrategyLearningModel] = {}
        
        # Market regime learning data
        self.regime_performance: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Strategy outcome tracking
        self.strategy_outcomes: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Initialize system
        os.makedirs(self.data_path, exist_ok=True)
        
        # Last save timestamp
        self.last_save = datetime.now()
        
        # Load existing learning models
        self.load_learning_models()
        
        # Register event handlers
        self.register_event_handlers()
        
        logger.info("Enhanced Continuous Learning System initialized")
        
    def register_event_handlers(self):
        """Register event handlers"""
        event_system = self.core_context.event_system
        
        event_system.register_handler("trade_executed", self.handle_trade_executed)
        event_system.register_handler("strategy_performance_update", self.handle_performance_update)
        event_system.register_handler("market_regime_change", self.handle_regime_change)
        event_system.register_handler("hourly_update", self.handle_hourly_update)
        event_system.register_handler("daily_summary", self.handle_daily_summary)
        
    def load_learning_models(self):
        """Load learning models from disk"""
        models_path = os.path.join(self.data_path, "learning_models.json")
        
        if not os.path.exists(models_path):
            logger.info("No saved learning models found")
            return
            
        try:
            with open(models_path, 'r') as f:
                models_data = json.load(f)
                
            loaded_count = 0
            for model_data in models_data:
                strategy_id = model_data.get("strategy_id")
                if strategy_id:
                    model = StrategyLearningModel.from_dict(model_data)
                    self.learning_models[strategy_id] = model
                    loaded_count += 1
                    
            logger.info(f"Loaded {loaded_count} learning models from disk")
                
        except Exception as e:
            logger.error(f"Error loading learning models: {e}")
            
    def save_learning_models(self):
        """Save learning models to disk"""
        models_path = os.path.join(self.data_path, "learning_models.json")
        
        try:
            models_data = [model.to_dict() for model in self.learning_models.values()]
            
            with open(models_path, 'w') as f:
                json.dump(models_data, f, indent=2)
                
            self.last_save = datetime.now()
            logger.info(f"Saved {len(models_data)} learning models to disk")
                
        except Exception as e:
            logger.error(f"Error saving learning models: {e}")
            
    def handle_trade_executed(self, event: Event):
        """
        Handle trade executed event
        
        Args:
            event: Trade executed event
        """
        trade_data = event.data
        strategy_id = trade_data.get("strategy_id")
        
        if not strategy_id:
            return
            
        # Get current market regime
        market_context = self.core_context.market_context
        current_regime = market_context.current_regime
        
        # Add to strategy outcomes
        self.strategy_outcomes[strategy_id][current_regime].append({
            "timestamp": datetime.now().isoformat(),
            "trade_data": trade_data
        })
        
        # Limited history
        max_outcomes = 100
        if len(self.strategy_outcomes[strategy_id][current_regime]) > max_outcomes:
            self.strategy_outcomes[strategy_id][current_regime] = \
                self.strategy_outcomes[strategy_id][current_regime][-max_outcomes:]
                
    def handle_performance_update(self, event: Event):
        """
        Handle strategy performance update event
        
        Args:
            event: Performance update event
        """
        performance_data = event.data
        strategy_id = performance_data.get("strategy_id")
        
        if not strategy_id:
            return
            
        metrics = performance_data.get("metrics", {})
        
        # Get current market regime
        market_context = self.core_context.market_context
        current_regime = market_context.current_regime
        
        # Update regime performance tracking
        if "return" in metrics:
            self.regime_performance[strategy_id][current_regime].append(metrics["return"])
            
            # Limited history
            max_history = 100
            if len(self.regime_performance[strategy_id][current_regime]) > max_history:
                self.regime_performance[strategy_id][current_regime] = \
                    self.regime_performance[strategy_id][current_regime][-max_history:]
                    
        # Update learning model if it exists
        if strategy_id in self.learning_models:
            model = self.learning_models[strategy_id]
            model.update_performance(metrics)
            
            # Check if parameters should be updated
            if self.enable_parameter_learning and model.should_update_parameters():
                self._update_strategy_parameters(strategy_id)
        else:
            # Create new learning model if strategy exists
            strategy_factory = self.core_context.strategy_factory
            
            if hasattr(strategy_factory, "available_strategies"):
                available_strategies = strategy_factory.available_strategies
                
                if strategy_id in available_strategies:
                    strategy_info = available_strategies[strategy_id]
                    strategy_config = strategy_info.get("config", {})
                    
                    # Create new learning model
                    model = StrategyLearningModel(
                        strategy_id=strategy_id,
                        strategy_config=strategy_config,
                        learning_config=self.config.get("learning_config", {})
                    )
                    
                    self.learning_models[strategy_id] = model
                    model.update_performance(metrics)
                    
                    logger.info(f"Created new learning model for strategy: {strategy_id}")
                    
    def handle_regime_change(self, event: Event):
        """
        Handle market regime change event
        
        Args:
            event: Regime change event
        """
        regime_data = event.data
        old_regime = regime_data.get("old_regime")
        new_regime = regime_data.get("new_regime")
        
        logger.info(f"Market regime changed from {old_regime} to {new_regime}")
        
        # Skip if regime learning is disabled
        if not self.enable_regime_learning:
            return
            
        # Update strategy regime suitability based on performance
        self._update_regime_suitability(old_regime)
        
    def handle_hourly_update(self, event: Event):
        """
        Handle hourly update event
        
        Args:
            event: Hourly update event
        """
        # Check if we should save models
        hours_since_save = (datetime.now() - self.last_save).total_seconds() / 3600
        
        if hours_since_save >= self.save_frequency_hours:
            self.save_learning_models()
            
    def handle_daily_summary(self, event: Event):
        """
        Handle daily summary event
        
        Args:
            event: Daily summary event
        """
        # Save learning models
        self.save_learning_models()
        
        # Perform parameter updates for strategies that need it
        if self.enable_parameter_learning:
            for strategy_id, model in self.learning_models.items():
                if model.should_update_parameters():
                    self._update_strategy_parameters(strategy_id)
                    
    def _update_strategy_parameters(self, strategy_id: str):
        """
        Update parameters for a strategy based on learning
        
        Args:
            strategy_id: ID of the strategy
        """
        if strategy_id not in self.learning_models:
            return
            
        model = self.learning_models[strategy_id]
        
        # Generate parameter variation
        new_config = model.generate_parameter_variation()
        
        # Apply to model
        model.apply_parameter_update(new_config)
        
        # Apply to actual strategy
        strategy_factory = self.core_context.strategy_factory
        
        if hasattr(strategy_factory, "update_strategy_config"):
            strategy_factory.update_strategy_config(strategy_id, new_config)
            
            logger.info(f"Updated parameters for strategy {strategy_id}")
            
            # Emit event
            self.core_context.event_system.emit_event(
                Event("strategy_parameters_updated", {
                    "strategy_id": strategy_id,
                    "new_config": new_config,
                    "learning_stats": model.get_learning_stats()
                })
            )
            
    def _update_regime_suitability(self, regime: str):
        """
        Update regime suitability scores based on performance
        
        Args:
            regime: Market regime that just ended
        """
        # Dynamic strategy rotator
        strategy_rotator = getattr(self.core_context, "strategy_rotator", None)
        
        if not strategy_rotator:
            return
            
        # Update suitability scores for strategies
        for strategy_id, regime_perf in self.regime_performance.items():
            if regime in regime_perf and regime_perf[regime]:
                # Calculate average performance in this regime
                avg_performance = sum(regime_perf[regime]) / len(regime_perf[regime])
                
                # Calculate performance adjustment
                # Scale: -0.1 to +0.1 adjustment to suitability score
                adjustment = avg_performance * 2.0
                adjustment = max(-0.1, min(0.1, adjustment))
                
                # Apply adjustment to strategy profile
                strategy_rotator.update_strategy_regime_suitability(
                    strategy_id=strategy_id,
                    regime=regime,
                    adjustment=adjustment
                )
                
                logger.debug(f"Updated regime suitability for {strategy_id} in {regime} by {adjustment}")
                
    def get_strategy_learning_stats(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get learning statistics for a strategy
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary with learning statistics
        """
        if strategy_id in self.learning_models:
            return self.learning_models[strategy_id].get_learning_stats()
        return {}
        
    def get_regime_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics by regime
        
        Returns:
            Dictionary with regime performance statistics
        """
        stats = {}
        
        for strategy_id, regimes in self.regime_performance.items():
            strategy_stats = {}
            
            for regime, performances in regimes.items():
                if performances:
                    avg_perf = sum(performances) / len(performances)
                    strategy_stats[regime] = {
                        "avg_return": avg_perf,
                        "samples": len(performances)
                    }
                    
            if strategy_stats:
                stats[strategy_id] = strategy_stats
                
        return stats
        
    def reset_strategy_learning(self, strategy_id: str):
        """
        Reset learning for a strategy to baseline
        
        Args:
            strategy_id: ID of the strategy
        """
        if strategy_id in self.learning_models:
            self.learning_models[strategy_id].reset_to_baseline()
            
            # Apply reset to actual strategy
            strategy_factory = self.core_context.strategy_factory
            
            if hasattr(strategy_factory, "update_strategy_config"):
                base_config = self.learning_models[strategy_id].base_config
                strategy_factory.update_strategy_config(strategy_id, base_config)
                
            logger.info(f"Reset learning for strategy {strategy_id}")
            
    def get_all_learning_models(self) -> Dict[str, StrategyLearningModel]:
        """
        Get all learning models
        
        Returns:
            Dictionary mapping strategy IDs to learning models
        """
        return self.learning_models 