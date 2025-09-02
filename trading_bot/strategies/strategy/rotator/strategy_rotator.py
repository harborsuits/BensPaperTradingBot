#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrategyRotator - Component to dynamically switch between trading strategies
based on market conditions and performance metrics.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from collections import defaultdict

# Import shared modules
from trading_bot.common.market_types import MarketRegime, MarketRegimeEvent
from trading_bot.common.config_utils import (
    setup_directories, load_config, save_state, load_state
)
from trading_bot.strategy.base.strategy import Strategy

# Setup logging
logger = logging.getLogger("StrategyRotator")

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
        regime_adaptation: bool = True
    ):
        """
        Initialize the strategy rotator.
        
        Args:
            strategies: List of Strategy instances
            config_path: Path to configuration file
            data_dir: Directory for data storage
            performance_window: Window for performance calculation
            regime_adaptation: Whether to adapt to market regimes
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
        
        # Load state if available
        self.load_state()
        
        logger.info(f"StrategyRotator initialized with {len(self.strategies)} strategies")
    
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
                }
            },
            "default_weights": {
                "MomentumStrategy": 0.33,
                "TrendFollowingStrategy": 0.33,
                "MeanReversionStrategy": 0.34
            },
            "regime_weights": {
                "UNKNOWN": {
                    "MomentumStrategy": 0.33,
                    "TrendFollowingStrategy": 0.33,
                    "MeanReversionStrategy": 0.34
                },
                "BULL": {
                    "MomentumStrategy": 0.4,
                    "TrendFollowingStrategy": 0.4,
                    "MeanReversionStrategy": 0.2
                },
                "BEAR": {
                    "MomentumStrategy": 0.2,
                    "TrendFollowingStrategy": 0.5,
                    "MeanReversionStrategy": 0.3
                },
                "SIDEWAYS": {
                    "MomentumStrategy": 0.2,
                    "TrendFollowingStrategy": 0.2,
                    "MeanReversionStrategy": 0.6
                },
                "HIGH_VOL": {
                    "MomentumStrategy": 0.1,
                    "TrendFollowingStrategy": 0.3,
                    "MeanReversionStrategy": 0.6
                },
                "LOW_VOL": {
                    "MomentumStrategy": 0.3,
                    "TrendFollowingStrategy": 0.3,
                    "MeanReversionStrategy": 0.4
                },
                "CRISIS": {
                    "MomentumStrategy": 0.1,
                    "TrendFollowingStrategy": 0.7,
                    "MeanReversionStrategy": 0.2
                }
            },
            "update_frequency": 86400,  # 1 day in seconds
            "minimum_performance_data": 5,  # Minimum data points for adaptive weighting
            "adaptive_weight_strength": 0.5  # How strongly to weight performance (0-1)
        }
    
    def _create_default_strategies(self) -> List[Strategy]:
        """Create default strategies from configuration."""
        # Import here to avoid circular imports
        from trading_bot.strategy.implementations.standard_strategies import (
            MomentumStrategy, TrendFollowingStrategy, MeanReversionStrategy
        )
        
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
            try:
                regime = MarketRegime[regime]
            except KeyError:
                logger.error(f"Invalid market regime: {regime}")
                return
        
        # Check if regime changed
        if regime != self.current_regime:
            logger.info(f"Market regime changed from {self.current_regime.name} to {regime.name}")
            
            # Update current regime
            self.current_regime = regime
            
            # Adjust weights based on regime
            if self.regime_adaptation:
                self._adapt_weights_to_regime(confidence)
    
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
        
        logger.info("Strategy rotator reset") 