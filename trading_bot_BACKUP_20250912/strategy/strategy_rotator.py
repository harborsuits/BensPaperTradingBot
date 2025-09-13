#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrategyRotator - Component to dynamically switch between trading strategies
based on market conditions and performance metrics.
"""

import os
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict

# Import shared modules
from trading_bot.common.market_types import MarketRegime, MarketRegimeEvent
from trading_bot.common.config_utils import (
    setup_directories, load_config, save_state, load_state
)

# Import strategy classes
# RLStrategy, DQNStrategy, PPOStrategy, MetaLearningStrategy are imported dynamically where needed to avoid circular import
# from trading_bot.strategy.rl_strategy import (
#     RLStrategy, DQNStrategy, PPOStrategy, MetaLearningStrategy
# )

# Import realtime modules
try:
    from trading_bot.realtime.message_queue import MessageBroker
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

# Setup logging
logger = logging.getLogger("StrategyRotator")

class Strategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.performance_history = []
        self.last_signal = 0.0
        self.last_update_time = None
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal between -1.0 and 1.0.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        # Base implementation returns neutral
        return 0.0
    
    def update_performance(self, performance_metric: float) -> None:
        """
        Update strategy performance.
        
        Args:
            performance_metric: Performance metric (e.g., return, profit)
        """
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": performance_metric
        })
        
    def get_average_performance(self, window: int = 10) -> float:
        """
        Get average performance over last n periods.
        
        Args:
            window: Number of periods to average
            
        Returns:
            float: Average performance
        """
        if not self.performance_history:
            return 0.0
            
        # Take last n performance records
        recent = self.performance_history[-window:]
        
        if not recent:
            return 0.0
            
        return np.mean([r["performance"] for r in recent])
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.performance_history = []
        self.last_signal = 0.0
        self.last_update_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        return {
            "name": self.name,
            "config": self.config,
            "enabled": self.enabled,
            "performance_history": self.performance_history,
            "last_signal": self.last_signal,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create strategy from dictionary."""
        strategy = cls(data["name"], data["config"])
        strategy.enabled = data["enabled"]
        strategy.performance_history = data["performance_history"]
        strategy.last_signal = data["last_signal"]
        
        if data["last_update_time"]:
            strategy.last_update_time = datetime.fromisoformat(data["last_update_time"])
            
        return strategy


class MomentumStrategy(Strategy):
    """Simple momentum strategy based on recent price action"""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """Generate momentum-based signal."""
        # Extract data
        prices = market_data.get("prices", [])
        
        if len(prices) < 2:
            return 0.0
        
        # Get config parameters
        fast_period = self.config.get("fast_period", 5)
        slow_period = self.config.get("slow_period", 20)
        
        # Ensure enough data
        if len(prices) < slow_period:
            return 0.0
        
        # Calculate momentum indicators
        fast_momentum = prices[-1] / prices[-min(fast_period, len(prices))] - 1
        slow_momentum = prices[-1] / prices[-min(slow_period, len(prices))] - 1
        
        # Generate signal between -1 and 1
        signal = fast_momentum - slow_momentum
        
        # Scale signal to be between -1 and 1
        scaled_signal = np.clip(signal * 10, -1.0, 1.0)
        
        # Update last signal and time
        self.last_signal = scaled_signal
        self.last_update_time = datetime.now()
        
        return scaled_signal


class TrendFollowingStrategy(Strategy):
    """Trend following strategy based on moving averages"""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """Generate trend-following signal."""
        # Extract data
        prices = market_data.get("prices", [])
        
        if len(prices) < 2:
            return 0.0
        
        # Get config parameters
        short_ma_period = self.config.get("short_ma_period", 10)
        long_ma_period = self.config.get("long_ma_period", 30)
        
        # Ensure enough data
        if len(prices) < long_ma_period:
            return 0.0
        
        # Calculate moving averages
        short_ma = np.mean(prices[-short_ma_period:])
        long_ma = np.mean(prices[-long_ma_period:])
        
        # Calculate trend strength
        trend_strength = (short_ma / long_ma - 1) * 10
        
        # Generate signal between -1 and 1
        signal = np.clip(trend_strength * 5, -1.0, 1.0)
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        return signal


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy looking for oversold/overbought conditions"""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """Generate mean-reversion signal."""
        # Extract data
        prices = market_data.get("prices", [])
        
        if len(prices) < 2:
            return 0.0
        
        # Get config parameters
        period = self.config.get("period", 20)
        std_dev_factor = self.config.get("std_dev_factor", 2.0)
        
        # Ensure enough data
        if len(prices) < period:
            return 0.0
        
        # Calculate mean and standard deviation
        mean_price = np.mean(prices[-period:])
        std_dev = np.std(prices[-period:])
        
        # Calculate z-score (how many std devs from mean)
        z_score = (prices[-1] - mean_price) / std_dev if std_dev > 0 else 0
        
        # Mean reversion signal (negative of z-score, scaled)
        # When price is high (positive z-score), signal is negative (sell)
        # When price is low (negative z-score), signal is positive (buy)
        signal = -z_score / std_dev_factor
        
        # Clip signal to be between -1 and 1
        signal = np.clip(signal, -1.0, 1.0)
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        return signal


class StrategyRotator:
    """
    Component that dynamically switches between trading strategies
    based on market conditions and performance metrics.
    """
    
    def __init__(
        self,
        strategies: List[Strategy] = None,
        config_path: str = None,
        data_dir: str = None,
        performance_window: int = 10,
        regime_adaptation: bool = True,
        use_event_driven: bool = True,
        message_broker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the strategy rotator.
        
        Args:
            strategies: List of Strategy instances
            config_path: Path to configuration file
            data_dir: Directory for data storage
            performance_window: Window for performance calculation
            regime_adaptation: Whether to adapt to market regimes
            use_event_driven: Whether to use event-driven architecture
            message_broker_config: Configuration for message broker
        """
        # Setup paths using shared utilities
        self.paths = setup_directories(
            data_dir=data_dir,
            component_name="strategy_rotator"
        )
        
        # Override config path if provided
        if config_path:
            self.paths["config_path"] = config_path
            
        # Load configuration
        self.config = load_config(
            self.paths["config_path"], 
            default_config_factory=self._get_default_config
        )
        
        # Initialize strategies if provided, otherwise create defaults
        self.strategies = strategies or self._create_default_strategies()
        
        # Store by name for easy lookup
        self.strategies_by_name = {s.name: s for s in self.strategies}
        
        # Performance tracking
        self.performance_window = performance_window
        
        # Market regime adaptation
        self.regime_adaptation = regime_adaptation
        self.current_regime = MarketRegime.UNKNOWN
        
        # Current strategy weights
        self.strategy_weights = self._initialize_weights()
        
        # Performance history
        self.performance_history = []
        
        # Signals history
        self.signals_history = []
        
        # Event-driven mode
        self.use_event_driven = use_event_driven
        self.running = False
        self.event_loop = None
        
        # Price history buffer for strategies
        self.price_history = []
        self.price_history_max_length = 1000  # Maximum length of price history
        
        # Initialize message broker for event-driven mode
        self.message_broker = None
        if self.use_event_driven and REALTIME_AVAILABLE:
            broker_config = message_broker_config or {}
            queue_type = broker_config.get("queue_type", "memory")
            redis_url = broker_config.get("redis_url")
            queue_name = broker_config.get("queue_name", "strategy_rotator_queue")
            max_queue_size = broker_config.get("max_queue_size", 10000)
            
            self.message_broker = MessageBroker(
                queue_type=queue_type,
                redis_url=redis_url,
                queue_name=queue_name,
                max_queue_size=max_queue_size
            )
        
        # Signal handlers for callbacks
        self.signal_handlers = []
        
        # Load state if available
        self.load_state()
        
        logger.info(f"StrategyRotator initialized with {len(self.strategies)} strategies")
        if self.use_event_driven:
            logger.info("Event-driven mode enabled")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "strategy_configs": {
                "MomentumStrategy": {
                    "fast_period": 5,
                    "slow_period": 20
                },
                "TrendFollowingStrategy": {
                    "short_ma_period": 10,
                    "long_ma_period": 30
                },
                "MeanReversionStrategy": {
                    "period": 20,
                    "std_dev_factor": 2.0
                },
                "DQNStrategy": {
                    "window_size": 20,
                    "batch_size": 32,
                    "gamma": 0.99,
                    "eps_start": 1.0,
                    "eps_end": 0.01,
                    "eps_decay": 0.995,
                    "learning_rate": 0.001,
                    "memory_capacity": 10000
                },
                "MetaLearningStrategy": {
                    "window_size": 30,
                    "batch_size": 64,
                    "gamma": 0.99,
                    "learning_rate": 0.0005,
                    "memory_capacity": 20000
                }
            },
            "default_weights": {
                "MomentumStrategy": 0.20,
                "TrendFollowingStrategy": 0.20,
                "MeanReversionStrategy": 0.20,
                "DQNStrategy": 0.30,
                "MetaLearningStrategy": 0.10
            },
            "regime_weights": {
                "UNKNOWN": {
                    "MomentumStrategy": 0.20,
                    "TrendFollowingStrategy": 0.20,
                    "MeanReversionStrategy": 0.20,
                    "DQNStrategy": 0.30,
                    "MetaLearningStrategy": 0.10
                },
                "BULL": {
                    "MomentumStrategy": 0.30,
                    "TrendFollowingStrategy": 0.30,
                    "MeanReversionStrategy": 0.10,
                    "DQNStrategy": 0.25,
                    "MetaLearningStrategy": 0.05
                },
                "BEAR": {
                    "MomentumStrategy": 0.10,
                    "TrendFollowingStrategy": 0.40,
                    "MeanReversionStrategy": 0.15,
                    "DQNStrategy": 0.25,
                    "MetaLearningStrategy": 0.10
                },
                "SIDEWAYS": {
                    "MomentumStrategy": 0.10,
                    "TrendFollowingStrategy": 0.10,
                    "MeanReversionStrategy": 0.40,
                    "DQNStrategy": 0.20,
                    "MetaLearningStrategy": 0.20
                },
                "HIGH_VOL": {
                    "MomentumStrategy": 0.05,
                    "TrendFollowingStrategy": 0.20,
                    "MeanReversionStrategy": 0.35,
                    "DQNStrategy": 0.20,
                    "MetaLearningStrategy": 0.20
                },
                "LOW_VOL": {
                    "MomentumStrategy": 0.20,
                    "TrendFollowingStrategy": 0.20,
                    "MeanReversionStrategy": 0.30,
                    "DQNStrategy": 0.20,
                    "MetaLearningStrategy": 0.10
                },
                "CRISIS": {
                    "MomentumStrategy": 0.05,
                    "TrendFollowingStrategy": 0.50,
                    "MeanReversionStrategy": 0.15,
                    "DQNStrategy": 0.20,
                    "MetaLearningStrategy": 0.10
                }
            },
            "update_frequency": 86400,  # 1 day in seconds
            "minimum_performance_data": 5,  # Minimum data points for adaptive weighting
            "adaptive_weight_strength": 0.5,  # How strongly to weight performance (0-1)
            "enable_rl_strategies": True,  # Whether to enable RL-based strategies
            "rl_training": {
                "enabled": True,  # Whether to enable RL training
                "train_frequency": 604800,  # 1 week in seconds
                "train_epochs": 100,  # Number of epochs per training session
                "min_training_data": 1000,  # Minimum data points for training
                "save_model_interval": 10  # Save model every N epochs
            },
            "event_driven": {
                "process_interval": 0.01,  # Time between processing batches (seconds)
                "batch_size": 10,  # Number of messages to process per batch
                "price_update_interval": 1.0,  # Minimum time between price updates (seconds)
                "auto_save_interval": 300  # Auto-save state every N seconds
            }
        }
    
    def _create_default_strategies(self) -> List[Strategy]:
        """Create default strategies from configuration."""
        default_strategies = []
        
        strategy_configs = self.config.get("strategy_configs", {})
        
        # Create momentum strategy
        if "MomentumStrategy" in strategy_configs:
            default_strategies.append(
                MomentumStrategy("MomentumStrategy", strategy_configs["MomentumStrategy"])
            )
        
        # Create trend following strategy
        if "TrendFollowingStrategy" in strategy_configs:
            default_strategies.append(
                TrendFollowingStrategy("TrendFollowingStrategy", strategy_configs["TrendFollowingStrategy"])
            )
        
        # Create mean reversion strategy
        if "MeanReversionStrategy" in strategy_configs:
            default_strategies.append(
                MeanReversionStrategy("MeanReversionStrategy", strategy_configs["MeanReversionStrategy"])
            )
        
        # Create RL-based strategies if enabled
        if self.config.get("enable_rl_strategies", False):
            # Add DQN strategy
            if "DQNStrategy" in strategy_configs:
                default_strategies.append(
                    DQNStrategy("DQNStrategy", strategy_configs["DQNStrategy"])
                )
            
            # Add Meta-learning strategy
            if "MetaLearningStrategy" in strategy_configs:
                default_strategies.append(
                    MetaLearningStrategy("MetaLearningStrategy", strategy_configs["MetaLearningStrategy"])
                )
        
        return default_strategies
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize strategy weights from configuration."""
        default_weights = self.config.get("default_weights", {})
        
        # If no defaults provided, distribute evenly
        if not default_weights:
            count = len(self.strategies)
            weight = 1.0 / count if count > 0 else 0.0
            default_weights = {s.name: weight for s in self.strategies}
        
        # Ensure all strategies have a weight
        weights = {}
        for strategy in self.strategies:
            weights[strategy.name] = default_weights.get(strategy.name, 0.0)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a new strategy to the rotator.
        
        Args:
            strategy: Strategy instance to add
        """
        # Check if strategy already exists
        if strategy.name in self.strategies_by_name:
            logger.warning(f"Strategy {strategy.name} already exists and will be replaced")
            
            # Remove existing strategy
            self.strategies = [s for s in self.strategies if s.name != strategy.name]
        
        # Add new strategy
        self.strategies.append(strategy)
        self.strategies_by_name[strategy.name] = strategy
        
        # Add weight for new strategy
        if strategy.name not in self.strategy_weights:
            # Initial weight is minimum of other strategies
            min_weight = min(self.strategy_weights.values()) if self.strategy_weights else 0.1
            self.strategy_weights[strategy.name] = min_weight
            
            # Renormalize weights
            self._normalize_weights()
        
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy from the rotator.
        
        Args:
            strategy_name: Name of strategy to remove
            
        Returns:
            bool: True if strategy was removed, False otherwise
        """
        if strategy_name not in self.strategies_by_name:
            logger.warning(f"Strategy {strategy_name} not found")
            return False
        
        # Remove strategy
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        strategy = self.strategies_by_name.pop(strategy_name)
        
        # Remove weight
        if strategy_name in self.strategy_weights:
            del self.strategy_weights[strategy_name]
            
            # Renormalize weights
            self._normalize_weights()
        
        logger.info(f"Removed strategy: {strategy_name}")
        return True
    
    def update_market_regime(self, regime: Union[MarketRegime, str], confidence: float = 1.0) -> None:
        """
        Update the current market regime.
        
        Args:
            regime: New market regime (enum or string name)
            confidence: Confidence in regime detection (0-1)
        """
        # Convert string to enum if needed
        if isinstance(regime, str):
            regime = MarketRegime[regime]
        
        # Check if regime changed
        if regime != self.current_regime:
            logger.info(f"Market regime changed from {self.current_regime.name} to {regime.name}")
            
            # Update current regime
            self.current_regime = regime
            
            # Adjust weights based on regime
            if self.regime_adaptation:
                self._adapt_weights_to_regime(confidence)
                
                # Notify meta-learning strategies of regime change
                self._notify_meta_learning_strategies(regime)
    
    def _notify_meta_learning_strategies(self, regime: MarketRegime) -> None:
        """
        Notify meta-learning strategies of a regime change.
        
        Args:
            regime: New market regime
        """
        for strategy in self.strategies:
            if isinstance(strategy, MetaLearningStrategy):
                try:
                    strategy.adapt_to_regime(regime)
                except Exception as e:
                    logger.error(f"Error adapting strategy {strategy.name} to regime: {str(e)}")
    
    def handle_market_regime_event(self, event: MarketRegimeEvent) -> None:
        """
        Handle a market regime change event.
        
        Args:
            event: Market regime change event
        """
        self.update_market_regime(event.new_regime, event.confidence)
    
    def _adapt_weights_to_regime(self, confidence: float = 1.0) -> None:
        """
        Adapt strategy weights to current market regime.
        
        Args:
            confidence: Confidence in regime detection (0-1)
        """
        # Get regime-specific weights
        regime_weights = self.config.get("regime_weights", {}).get(self.current_regime.name)
        
        if not regime_weights:
            logger.warning(f"No weights defined for regime {self.current_regime.name}")
            return
        
        # Calculate blended weights based on confidence
        for strategy_name, current_weight in self.strategy_weights.items():
            if strategy_name in regime_weights:
                regime_weight = regime_weights[strategy_name]
                
                # Blend weights based on confidence
                self.strategy_weights[strategy_name] = (
                    current_weight * (1 - confidence) + 
                    regime_weight * confidence
                )
        
        # Renormalize weights
        self._normalize_weights()
        
        logger.info(f"Adapted weights to {self.current_regime.name} regime")
    
    def update_strategy_performance(self, performance_data: Dict[str, float]) -> None:
        """
        Update performance metrics for strategies.
        
        Args:
            performance_data: Dict mapping strategy names to performance metrics
        """
        for strategy_name, performance in performance_data.items():
            if strategy_name in self.strategies_by_name:
                strategy = self.strategies_by_name[strategy_name]
                strategy.update_performance(performance)
                
                logger.debug(f"Updated performance for {strategy_name}: {performance}")
        
        # Record timestamp of update
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "data": performance_data
        })
        
        # Adapt weights based on performance if we have enough data
        self._adapt_weights_to_performance()
    
    def _adapt_weights_to_performance(self) -> None:
        """Adapt strategy weights based on performance."""
        # Check if we have enough performance data
        min_data_points = self.config.get("minimum_performance_data", 5)
        
        # Get all strategies with enough performance data
        valid_strategies = {}
        for strategy in self.strategies:
            if len(strategy.performance_history) >= min_data_points:
                avg_perf = strategy.get_average_performance(self.performance_window)
                valid_strategies[strategy.name] = max(0.0001, avg_perf)  # Avoid division by zero
        
        # If no valid strategies, return
        if not valid_strategies:
            return
        
        # Get adaptive weight strength
        strength = self.config.get("adaptive_weight_strength", 0.5)
        
        if strength <= 0:
            return
        
        # Calculate performance-based weights
        total_perf = sum(valid_strategies.values())
        perf_weights = {
            name: perf / total_perf
            for name, perf in valid_strategies.items()
        }
        
        # Blend with current weights
        for strategy_name in self.strategy_weights:
            if strategy_name in perf_weights:
                # Blend current weight with performance-based weight
                self.strategy_weights[strategy_name] = (
                    self.strategy_weights[strategy_name] * (1 - strength) + 
                    perf_weights[strategy_name] * strength
                )
        
        # Renormalize weights
        self._normalize_weights()
        
        logger.debug("Adapted weights based on performance")
    
    def _normalize_weights(self) -> None:
        """Normalize strategy weights to sum to 1.0."""
        total_weight = sum(self.strategy_weights.values())
        
        if total_weight > 0:
            self.strategy_weights = {
                k: v / total_weight
                for k, v in self.strategy_weights.items()
            }
    
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate signals from all strategies.
        
        Args:
            market_data: Market data for signal generation
            
        Returns:
            Dict mapping strategy names to signals
        """
        signals = {}
        
        for strategy in self.strategies:
            if not strategy.enabled:
                signals[strategy.name] = 0.0
                continue
                
            try:
                signal = strategy.generate_signal(market_data)
                signals[strategy.name] = signal
                
                logger.debug(f"Generated signal for {strategy.name}: {signal}")
            except Exception as e:
                logger.error(f"Error generating signal for {strategy.name}: {str(e)}")
                signals[strategy.name] = 0.0
        
        # Record signals
        self.signals_history.append({
            "timestamp": datetime.now().isoformat(),
            "signals": signals.copy()
        })
        
        return signals
    
    def get_combined_signal(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Get combined signal from all strategies weighted by strategy weights.
        
        Args:
            market_data: Optional market data to generate fresh signals
            
        Returns:
            float: Combined signal between -1.0 and 1.0
        """
        # Generate fresh signals if market data provided
        if market_data is not None:
            signals = self.generate_signals(market_data)
        else:
            # Use last signals
            signals = {
                strategy.name: strategy.last_signal
                for strategy in self.strategies
                if strategy.last_signal is not None
            }
        
        # Calculate weighted sum of signals
        combined_signal = 0.0
        for strategy_name, signal in signals.items():
            weight = self.strategy_weights.get(strategy_name, 0.0)
            combined_signal += signal * weight
        
        # Ensure signal is between -1.0 and 1.0
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        
        logger.debug(f"Combined signal: {combined_signal}")
        
        return combined_signal
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights.
        
        Returns:
            Dict mapping strategy names to weights
        """
        return self.strategy_weights.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all strategies.
        
        Returns:
            Dict with performance metrics
        """
        metrics = {}
        
        for strategy in self.strategies:
            avg_perf = strategy.get_average_performance(self.performance_window)
            metrics[strategy.name] = {
                "average_performance": avg_perf,
                "performance_history": strategy.performance_history[-self.performance_window:]
            }
        
        return metrics
    
    def register_signal_handler(self, handler: Callable[[float, Dict[str, float]], None]) -> None:
        """
        Register a callback function to be called when signals are generated.
        
        Args:
            handler: Function to call with combined signal and individual signals
        """
        self.signal_handlers.append(handler)
    
    def _notify_signal_handlers(self, combined_signal: float, signals: Dict[str, float]) -> None:
        """
        Notify all signal handlers of new signals.
        
        Args:
            combined_signal: Combined signal
            signals: Individual strategy signals
        """
        for handler in self.signal_handlers:
            try:
                handler(combined_signal, signals)
            except Exception as e:
                logger.error(f"Error in signal handler: {str(e)}")
    
    def save_state(self) -> None:
        """Save the current state to file."""
        state = {
            "current_regime": self.current_regime.name,
            "strategy_weights": self.strategy_weights,
            "strategies": {
                strategy.name: strategy.to_dict()
                for strategy in self.strategies
            },
            "signals_history": self.signals_history[-100:] if self.signals_history else [],
            "performance_history": self.performance_history[-100:] if self.performance_history else [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Use shared utility for saving state
        save_state(self.paths["state_path"], state)
        
        logger.info("Strategy rotator state saved")
    
    def load_state(self) -> bool:
        """
        Load state from file.
        
        Returns:
            bool: True if state was loaded successfully, False otherwise
        """
        # Use shared utility for loading state
        state = load_state(self.paths["state_path"])
        
        if state is None:
            return False
        
        try:
            # Load strategy state
            for strategy_name, strategy_data in state.get("strategies", {}).items():
                if strategy_name in self.strategies_by_name:
                    # Update existing strategy
                    strategy = self.strategies_by_name[strategy_name]
                    strategy.enabled = strategy_data["enabled"]
                    strategy.performance_history = strategy_data["performance_history"]
                    strategy.last_signal = strategy_data["last_signal"]
                    
                    if strategy_data["last_update_time"]:
                        strategy.last_update_time = datetime.fromisoformat(strategy_data["last_update_time"])
            
            # Load regime
            self.current_regime = MarketRegime[state.get("current_regime", "UNKNOWN")]
            
            # Load weights
            self.strategy_weights = state.get("strategy_weights", self.strategy_weights)
            
            # Load history
            self.signals_history = state.get("signals_history", [])
            self.performance_history = state.get("performance_history", [])
            
            logger.info("Strategy rotator state loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def reset(self) -> None:
        """Reset the strategy rotator to initial state."""
        # Reset all strategies
        for strategy in self.strategies:
            strategy.reset()
        
        # Reset regime
        self.current_regime = MarketRegime.UNKNOWN
        
        # Reset weights
        self.strategy_weights = self._initialize_weights()
        
        # Clear history
        self.signals_history = []
        self.performance_history = []
        self.price_history = []
        
        logger.info("Strategy rotator reset")
    
    def train_rl_strategies(self, market_data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Train RL-based strategies on historical market data.
        
        Args:
            market_data: DataFrame with historical market data
            
        Returns:
            Dict mapping strategy names to training rewards
        """
        if not self.config.get("rl_training", {}).get("enabled", False):
            logger.info("RL training is disabled")
            return {}
        
        # Check if we have enough data
        min_data = self.config.get("rl_training", {}).get("min_training_data", 1000)
        if len(market_data) < min_data:
            logger.warning(f"Not enough data for RL training: {len(market_data)} < {min_data}")
            return {}
        
        epochs = self.config.get("rl_training", {}).get("train_epochs", 100)
        rewards = {}
        
        # Train each RL strategy
        for strategy in self.strategies:
            if isinstance(strategy, RLStrategy):
                logger.info(f"Training {strategy.name}...")
                try:
                    training_rewards = strategy.train(market_data, epochs=epochs)
                    rewards[strategy.name] = training_rewards
                    logger.info(f"Completed training for {strategy.name}")
                except Exception as e:
                    logger.error(f"Error training {strategy.name}: {str(e)}")
        
        return rewards
    
    # Event-driven methods
    
    async def start(self) -> None:
        """Start the event-driven strategy rotator."""
        if not self.use_event_driven:
            logger.warning("Not in event-driven mode, cannot start")
            return
            
        if self.running:
            logger.warning("Already running")
            return
        
        self.running = True
        self.event_loop = asyncio.get_event_loop()
        
        # Start message broker if available
        if self.message_broker:
            await self.message_broker.connect()
            self.message_broker.add_handler(self.process_market_data)
            
            # Start processing messages in the background
            event_config = self.config.get("event_driven", {})
            process_interval = event_config.get("process_interval", 0.01)
            batch_size = event_config.get("batch_size", 10)
            
            await self.message_broker.start_processing(
                interval=process_interval,
                batch_size=batch_size
            )
            
            # Start auto-save task
            auto_save_interval = event_config.get("auto_save_interval", 300)
            if auto_save_interval > 0:
                asyncio.create_task(self._auto_save_task(auto_save_interval))
            
            logger.info("Event-driven strategy rotator started")
    
    async def _auto_save_task(self, interval: float) -> None:
        """
        Periodically save state.
        
        Args:
            interval: Save interval in seconds
        """
        while self.running:
            await asyncio.sleep(interval)
            try:
                self.save_state()
            except Exception as e:
                logger.error(f"Error in auto-save task: {str(e)}")
    
    async def process_market_data(self, data: Dict[str, Any]) -> None:
        """
        Process incoming market data event.
        
        Args:
            data: Market data event
        """
        try:
            # Check if this is a price update
            if "price" in data:
                await self._process_price_update(data)
            # Check if this is a performance update
            elif "performance" in data:
                await self._process_performance_update(data)
            # Check if this is a regime update
            elif "regime" in data:
                await self._process_regime_update(data)
            else:
                logger.warning(f"Unknown data format: {data}")
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
    
    async def _process_price_update(self, data: Dict[str, Any]) -> None:
        """
        Process a price update.
        
        Args:
            data: Price update data
        """
        # Extract price
        price = data.get("price")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        
        if price is None:
            logger.warning("Price update missing price")
            return
            
        # Add to price history
        self.price_history.append(price)
        
        # Limit history length
        if len(self.price_history) > self.price_history_max_length:
            self.price_history = self.price_history[-self.price_history_max_length:]
        
        # Prepare market data for signal generation
        market_data = {
            "prices": self.price_history,
            "timestamp": timestamp,
            **data  # Include any additional data
        }
        
        # Generate signals
        signals = self.generate_signals(market_data)
        
        # Get combined signal
        combined_signal = self.get_combined_signal(market_data=None)  # Use already generated signals
        
        # Notify signal handlers
        self._notify_signal_handlers(combined_signal, signals)
    
    async def _process_performance_update(self, data: Dict[str, Any]) -> None:
        """
        Process a performance update.
        
        Args:
            data: Performance update data
        """
        performance_data = data.get("performance", {})
        self.update_strategy_performance(performance_data)
    
    async def _process_regime_update(self, data: Dict[str, Any]) -> None:
        """
        Process a regime update.
        
        Args:
            data: Regime update data
        """
        regime = data.get("regime")
        confidence = data.get("confidence", 1.0)
        
        if regime is None:
            logger.warning("Regime update missing regime")
            return
            
        self.update_market_regime(regime, confidence)
    
    async def stop(self) -> None:
        """Stop the event-driven strategy rotator."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop message broker
        if self.message_broker:
            self.message_broker.stop()
        
        # Save state before stopping
        self.save_state()
        
        logger.info("Event-driven strategy rotator stopped")
    
    async def publish_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Publish market data to the message broker.
        
        Args:
            data: Market data to publish
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.message_broker:
            logger.warning("No message broker available")
            return False
            
        return await self.message_broker.publish(data)
    
    def process_tradingview_data(self, data: Dict[str, Any]) -> None:
        """
        Process data from TradingView alerts.
        
        Args:
            data: Data from TradingView alert
        """
        try:
            # Determine if we have price information
            if 'price' in data:
                # Convert to market data format expected by strategies
                market_data = {
                    'prices': self.price_history + [data['price']],
                    'timestamp': data.get('timestamp'),
                    'symbol': data.get('symbol')
                }
                
                # Add any indicators from TradingView to market data
                for key, value in data.items():
                    if key not in ['price', 'prices', 'timestamp', 'symbol']:
                        market_data[key] = value
                
                # Update price history
                self.price_history.append(data['price'])
                # Limit history length
                if len(self.price_history) > self.price_history_max_length:
                    self.price_history = self.price_history[-self.price_history_max_length:]
                
                # Generate signals
                signals = self.generate_signals(market_data)
                
                # Get combined signal
                combined_signal = self.get_combined_signal(market_data=None)  # Use already generated signals
                
                # Notify signal handlers
                self._notify_signal_handlers(combined_signal, signals)
                
                logger.info(f"Processed TradingView data for {data.get('symbol')}, "
                           f"generated signal: {combined_signal:.4f}")
            
            # Check for regime information
            if 'regime' in data:
                regime = data.get('regime')
                confidence = data.get('confidence', 1.0)
                
                if regime:
                    self.update_market_regime(regime, confidence)
                    logger.info(f"Updated market regime to {regime} with confidence {confidence}")
            
            # Check for performance metrics
            if 'performance' in data and isinstance(data['performance'], dict):
                self.update_strategy_performance(data['performance'])
                logger.info(f"Updated strategy performance from TradingView data")
                
        except Exception as e:
            logger.error(f"Error processing TradingView data: {str(e)}")


# Simple example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run async example
    async def main():
        # Create rotator with event-driven mode
        rotator = StrategyRotator(use_event_driven=True)
        
        # Example signal handler
        def on_signal(combined_signal, signals):
            print(f"Combined signal: {combined_signal:.4f}")
            for strategy, signal in signals.items():
                print(f"  {strategy}: {signal:.4f}")
        
        # Register signal handler
        rotator.register_signal_handler(on_signal)
        
        # Start rotator
        await rotator.start()
        
        # Simulate some market data
        for i in range(50):
            # Create price data
            price_data = {
                "price": 100 + i * 0.1,
                "volume": 1000 + i,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to message broker
            await rotator.publish_market_data(price_data)
            
            # Wait a bit
            await asyncio.sleep(0.1)
        
        # Stop rotator
        await rotator.stop()
    
    # Run the example
    if __name__ == "__main__":
        asyncio.run(main()) 