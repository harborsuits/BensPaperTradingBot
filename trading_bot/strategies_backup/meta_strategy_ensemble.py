"""
Meta-Strategy Ensemble Layer

This module implements a dynamic ensemble of trading strategies that adapts
allocation weights based on market regimes and recent performance metrics.

It serves as a "strategy of strategies" that learns when to trust different
types of models under varying market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta
import joblib
import json
import os
from collections import defaultdict

from trading_bot.strategies.strategy_template import StrategyTemplate, Signal, SignalType
from trading_bot.utils.feature_engineering import FeatureEngineering

class MetaStrategyEnsemble:
    """
    A dynamic ensemble of trading strategies with adaptive allocation.
    
    This class manages multiple sub-strategies and uses a meta-learner to:
    1. Track performance of each strategy across different market regimes
    2. Dynamically adjust allocation weights based on recent performance
    3. Combine signals to produce a final trading decision
    4. Provide explainable reasoning for each decision
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the meta-strategy ensemble.
        
        Args:
            config: Configuration dictionary with the following keys:
                - strategies: List of strategy instances or configuration dicts
                - allocation_method: How to allocate between strategies ('performance', 'regime_based', 'ml')
                - lookback_window: How many periods to consider for performance tracking
                - update_frequency: How often to update allocation weights
                - min_weight: Minimum allocation weight per strategy
                - use_regime_tracking: Whether to track performance by market regime
                - meta_features: List of features to use for meta-model learning
                - output_dir: Directory to save performance metrics and models
        """
        self.config = config
        self.strategies = []
        self.strategy_weights = {}
        self.performance_history = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.current_regime = "unknown"
        self.last_weight_update = None
        self.meta_model = None
        self.performance_metrics = ["sharpe", "win_rate", "avg_return", "max_drawdown"]
        
        # Set up output directory
        self.output_dir = config.get("output_dir", "./output/meta_ensemble")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize sub-strategies
        self._initialize_strategies()
        
        # Initialize allocation weights
        self._initialize_weights()
        
        # Initialize feature engineering for market regime detection
        self._initialize_feature_engineering()

    def _initialize_strategies(self):
        """Initialize sub-strategies from configuration."""
        strategy_configs = self.config.get("strategies", [])
        
        for i, strat_config in enumerate(strategy_configs):
            if isinstance(strat_config, StrategyTemplate):
                # Already a strategy instance
                strategy = strat_config
                strategy_id = strategy.name
            else:
                # Config dict to create a new strategy
                strategy_type = strat_config.get("type")
                strategy_id = strat_config.get("id", f"strategy_{i}")
                
                # Import the appropriate strategy class
                if strategy_type == "ml":
                    from trading_bot.strategies.ml_strategy import MLStrategy
                    strategy = MLStrategy(strat_config)
                elif strategy_type == "momentum":
                    from trading_bot.strategies.stocks.momentum import MomentumStrategy
                    strategy = MomentumStrategy(strat_config)
                elif strategy_type == "mean_reversion":
                    from trading_bot.strategies.stocks.mean_reversion import MeanReversionStrategy
                    strategy = MeanReversionStrategy(strat_config)
                elif strategy_type == "breakout":
                    from trading_bot.strategies.stocks.breakout import PriceChannelBreakoutStrategy
                    strategy = PriceChannelBreakoutStrategy(strat_config)
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            self.strategies.append((strategy_id, strategy))
    
    def _initialize_weights(self):
        """Initialize allocation weights for sub-strategies."""
        # Start with equal weights
        n_strategies = len(self.strategies)
        default_weight = 1.0 / max(1, n_strategies)
        
        # Use provided weights or default to equal weighting
        for strategy_id, _ in self.strategies:
            self.strategy_weights[strategy_id] = self.config.get("initial_weights", {}).get(
                strategy_id, default_weight
            )
        
        # Normalize weights to sum to 1
        self._normalize_weights()
        
        # Record initialization time
        self.last_weight_update = datetime.now()
    
    def _initialize_feature_engineering(self):
        """Initialize feature engineering for market regime detection."""
        fe_params = self.config.get("feature_engineering", {})
        fe_params["detect_market_regime"] = True
        self.feature_engineering = FeatureEngineering(fe_params)
    
    def _normalize_weights(self):
        """Ensure strategy weights sum to 1.0 while respecting minimum weights."""
        min_weight = self.config.get("min_weight", 0.05)
        
        # Ensure minimum weights
        for strategy_id in self.strategy_weights:
            self.strategy_weights[strategy_id] = max(min_weight, self.strategy_weights[strategy_id])
        
        # Normalize to sum to 1.0
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy_id in self.strategy_weights:
                self.strategy_weights[strategy_id] /= total_weight
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Detect the current market regime using feature engineering.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Market regime label
        """
        # Use feature engineering to detect market regime
        _ = self.feature_engineering.generate_features(market_data)
        regime = self.feature_engineering.market_regime
        
        # Update current regime
        self.current_regime = regime
        
        return regime
    
    def update_performance_metrics(self, strategy_id: str, performance: Dict[str, float]):
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: Identifier for the strategy
            performance: Dictionary of performance metrics
        """
        # Add timestamp
        performance["timestamp"] = datetime.now().isoformat()
        
        # Add to overall performance history
        self.performance_history[strategy_id].append(performance)
        
        # Add to regime-specific performance if tracking by regime
        if self.config.get("use_regime_tracking", True) and self.current_regime != "unknown":
            self.regime_performance[self.current_regime][strategy_id].append(performance)
        
        # Trim history if it exceeds the lookback window
        lookback = self.config.get("lookback_window", 30)
        if len(self.performance_history[strategy_id]) > lookback:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-lookback:]
    
    def update_allocation_weights(self, market_data: Optional[pd.DataFrame] = None):
        """
        Update allocation weights based on recent performance.
        
        Args:
            market_data: Optional DataFrame with market data for regime detection
        """
        # Check if it's time to update weights
        update_frequency = self.config.get("update_frequency", timedelta(days=1))
        if (self.last_weight_update and
                datetime.now() - self.last_weight_update < update_frequency):
            return
        
        # Detect market regime if market data is provided
        if market_data is not None:
            self.detect_market_regime(market_data)
        
        # Get allocation method
        method = self.config.get("allocation_method", "performance")
        
        if method == "performance":
            self._update_weights_by_performance()
        elif method == "regime_based":
            self._update_weights_by_regime()
        elif method == "ml":
            self._update_weights_by_ml(market_data)
        else:
            # Default to performance-based
            self._update_weights_by_performance()
        
        # Normalize weights
        self._normalize_weights()
        
        # Update timestamp
        self.last_weight_update = datetime.now()
        
        # Log updated weights
        self._log_weights()
    
    def _update_weights_by_performance(self):
        """Update weights based on recent performance metrics."""
        # Performance metric to use for weighting
        metric = self.config.get("performance_metric", "sharpe")
        lookback = self.config.get("lookback_window", 30)
        min_periods = self.config.get("min_periods", 5)
        
        new_weights = {}
        valid_performance = False
        
        for strategy_id, _ in self.strategies:
            history = self.performance_history[strategy_id]
            
            if len(history) >= min_periods:
                valid_performance = True
                recent_history = history[-lookback:]
                
                # Calculate average performance
                if metric in recent_history[0]:
                    avg_metric = np.mean([h[metric] for h in recent_history if metric in h])
                    # Ensure non-negative weights
                    new_weights[strategy_id] = max(0.0, avg_metric)
        
        # Only update if we have valid performance data
        if valid_performance:
            # Handle negative Sharpe ratios or other metrics that can be negative
            for strategy_id in new_weights:
                if new_weights[strategy_id] < 0:
                    new_weights[strategy_id] = self.config.get("min_weight", 0.05)
            
            # Assign new weights
            for strategy_id, _ in self.strategies:
                if strategy_id in new_weights:
                    self.strategy_weights[strategy_id] = new_weights[strategy_id]
                else:
                    # Assign minimum weight for strategies without enough data
                    self.strategy_weights[strategy_id] = self.config.get("min_weight", 0.05)
    
    def _update_weights_by_regime(self):
        """Update weights based on regime-specific performance."""
        if self.current_regime == "unknown":
            # Fall back to overall performance if regime is unknown
            self._update_weights_by_performance()
            return
        
        metric = self.config.get("performance_metric", "sharpe")
        min_periods = self.config.get("min_periods", 3)
        
        # Check if we have enough data for the current regime
        if (self.current_regime in self.regime_performance and
                all(len(history) >= min_periods 
                    for history in self.regime_performance[self.current_regime].values())):
            
            # Calculate weights based on regime-specific performance
            new_weights = {}
            for strategy_id, _ in self.strategies:
                history = self.regime_performance[self.current_regime][strategy_id]
                if history:
                    avg_metric = np.mean([h[metric] for h in history if metric in h])
                    new_weights[strategy_id] = max(0.0, avg_metric)
                else:
                    # Use minimum weight if no history
                    new_weights[strategy_id] = self.config.get("min_weight", 0.05)
            
            # Update weights
            for strategy_id in self.strategy_weights:
                if strategy_id in new_weights:
                    self.strategy_weights[strategy_id] = new_weights[strategy_id]
        else:
            # Fall back to overall performance if not enough regime-specific data
            self._update_weights_by_performance()
    
    def _update_weights_by_ml(self, market_data: pd.DataFrame):
        """
        Update weights using a meta-ML model to predict optimal allocations.
        
        Args:
            market_data: DataFrame with market data
        """
        if market_data is None or market_data.empty:
            # Fall back to performance-based if no market data
            self._update_weights_by_performance()
            return
        
        # Create meta-features for the meta-model
        meta_features = self._create_meta_features(market_data)
        
        # If meta-model doesn't exist or needs retraining
        if self.meta_model is None or self._should_retrain_meta_model():
            self._train_meta_model()
        
        if self.meta_model is not None:
            try:
                # Predict weights using the meta-model
                predicted_weights = self.meta_model.predict(meta_features)
                
                # Update weights with predictions
                for i, (strategy_id, _) in enumerate(self.strategies):
                    self.strategy_weights[strategy_id] = max(
                        self.config.get("min_weight", 0.05),
                        predicted_weights[i]
                    )
            except Exception as e:
                print(f"Error predicting weights with meta-model: {str(e)}")
                # Fall back to performance-based on error
                self._update_weights_by_performance()
        else:
            # Fall back to performance-based if no meta-model
            self._update_weights_by_performance()
    
    def _create_meta_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the meta-model.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with meta-features
        """
        # Generate standard features
        features = self.feature_engineering.generate_features(market_data)
        
        # Select meta-features as specified in config
        meta_feature_list = self.config.get("meta_features", [])
        if meta_feature_list:
            meta_features = features[[col for col in meta_feature_list if col in features.columns]]
        else:
            # Default meta-features
            default_features = [
                "market_regime_numeric", "volatility_20", "rsi_14",
                "adx", "bb_width_20", "atr_percent"
            ]
            meta_features = features[[col for col in default_features if col in features.columns]]
        
        # Add recent performance metrics for each strategy
        for strategy_id, _ in self.strategies:
            history = self.performance_history[strategy_id]
            if history:
                recent = history[-1]
                for metric in self.performance_metrics:
                    if metric in recent:
                        meta_features[f"{strategy_id}_{metric}"] = recent[metric]
        
        return meta_features
    
    def _should_retrain_meta_model(self) -> bool:
        """Check if meta-model should be retrained."""
        # Implement criteria for when to retrain the meta-model
        # For example, retrain every N days or after significant performance changes
        if not hasattr(self, 'last_meta_model_train'):
            return True
        
        retrain_frequency = self.config.get("meta_model_retrain_frequency", timedelta(days=7))
        return datetime.now() - self.last_meta_model_train > retrain_frequency
    
    def _train_meta_model(self):
        """Train a meta-model to predict optimal strategy weights."""
        # This implementation would typically:
        # 1. Collect historical data on strategy performance and market conditions
        # 2. Create a training dataset with meta-features and optimal allocations
        # 3. Train a model (e.g., XGBoost, neural network) to predict allocations
        #
        # For complex models, this would be a separate component.
        # Here we'll outline a simplified version.
        
        # Check if we have enough historical data
        min_history = self.config.get("min_history_for_meta", 30)
        if not all(len(history) >= min_history for history in self.performance_history.values()):
            return
        
        try:
            # Here you would train your meta-model using your ML framework of choice
            # For example, using a simple linear model or gradient boosting
            
            # For demonstration, we'll just log that training would happen here
            print("Training meta-model for strategy allocation...")
            
            # Record training time
            self.last_meta_model_train = datetime.now()
            
            # Save model metadata
            self._log_meta_model_training()
            
        except Exception as e:
            print(f"Error training meta-model: {str(e)}")
    
    def _log_weights(self):
        """Log current strategy weights."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.current_regime,
            "weights": self.strategy_weights.copy()
        }
        
        log_file = os.path.join(self.output_dir, "weight_history.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _log_meta_model_training(self):
        """Log meta-model training metadata."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "num_strategies": len(self.strategies),
            "performance_history_length": {
                strategy_id: len(history)
                for strategy_id, history in self.performance_history.items()
            }
        }
        
        log_file = os.path.join(self.output_dir, "meta_model_training.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_signal(self, market_data: pd.DataFrame) -> Tuple[Signal, Dict[str, Any]]:
        """
        Generate trading signal by combining weighted signals from sub-strategies.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Tuple of (Signal, reasoning dictionary)
        """
        # Update market regime
        self.detect_market_regime(market_data)
        
        # Update allocation weights if needed
        self.update_allocation_weights(market_data)
        
        # Collect signals from each strategy
        signals = []
        reasoning = []
        
        for strategy_id, strategy in self.strategies:
            try:
                # Get signal from strategy
                signal = strategy.generate_signal(market_data)
                weight = self.strategy_weights[strategy_id]
                
                # Store weighted signal info
                signals.append({
                    "strategy_id": strategy_id,
                    "signal": signal,
                    "weight": weight
                })
                
                # Get reasoning if available
                if hasattr(strategy, 'get_signal_reasoning'):
                    signal_reason = strategy.get_signal_reasoning()
                    reasoning.append({
                        "strategy_id": strategy_id,
                        "weight": weight,
                        "reasoning": signal_reason
                    })
                
            except Exception as e:
                print(f"Error getting signal from {strategy_id}: {str(e)}")
        
        # Combine signals
        final_signal = self._combine_signals(signals)
        
        # Create reasoning dictionary
        reasoning_dict = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.current_regime,
            "strategy_weights": self.strategy_weights.copy(),
            "strategy_signals": [
                {
                    "strategy_id": s["strategy_id"],
                    "signal_type": s["signal"].signal_type.name if s["signal"] else "NEUTRAL",
                    "weight": s["weight"]
                }
                for s in signals
            ],
            "detailed_reasoning": reasoning
        }
        
        return final_signal, reasoning_dict
    
    def _combine_signals(self, signals: List[Dict[str, Any]]) -> Signal:
        """
        Combine weighted signals to produce a final signal.
        
        Args:
            signals: List of signal dictionaries with strategy_id, signal, and weight
            
        Returns:
            Combined signal
        """
        if not signals:
            return Signal(SignalType.NEUTRAL, 0.0)
        
        # Calculate weighted signal strength for each type
        signal_strengths = defaultdict(float)
        total_weight = 0.0
        
        for signal_dict in signals:
            if signal_dict["signal"] is None:
                continue
                
            weight = signal_dict["weight"]
            signal = signal_dict["signal"]
            signal_type = signal.signal_type
            strength = signal.strength * weight
            
            signal_strengths[signal_type] += strength
            total_weight += weight
        
        if total_weight == 0:
            return Signal(SignalType.NEUTRAL, 0.0)
        
        # Find dominant signal type
        dominant_type = max(
            signal_strengths.items(),
            key=lambda x: x[1]
        )[0]
        
        # Calculate overall strength (normalized by total weight)
        overall_strength = signal_strengths[dominant_type] / total_weight
        
        # Create final signal
        return Signal(dominant_type, overall_strength)
    
    def get_strategy_performance(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for strategies.
        
        Args:
            strategy_id: Optional ID of specific strategy to get metrics for
            
        Returns:
            Dictionary with performance metrics
        """
        if strategy_id:
            if strategy_id in self.performance_history:
                return {
                    "strategy_id": strategy_id,
                    "history": self.performance_history[strategy_id]
                }
            return {"error": f"Strategy {strategy_id} not found"}
        
        # Return performance for all strategies
        return {
            "overall": {
                strategy_id: history
                for strategy_id, history in self.performance_history.items()
            },
            "by_regime": {
                regime: {
                    strategy_id: history
                    for strategy_id, history in regime_data.items()
                }
                for regime, regime_data in self.regime_performance.items()
            }
        }
    
    def save(self, path: Optional[str] = None):
        """
        Save the meta-strategy ensemble state.
        
        Args:
            path: Optional path to save the state to
        """
        if path is None:
            path = os.path.join(self.output_dir, "meta_ensemble_state.pkl")
        
        # Prepare state dict (excluding non-serializable objects)
        state = {
            "strategy_weights": self.strategy_weights,
            "performance_history": dict(self.performance_history),
            "regime_performance": {
                regime: dict(strategy_data)
                for regime, strategy_data in self.regime_performance.items()
            },
            "current_regime": self.current_regime,
            "last_weight_update": self.last_weight_update,
            "config": self.config
        }
        
        # Save to file
        joblib.dump(state, path)
        
        # Also save a more readable JSON version of key metrics
        json_path = os.path.join(self.output_dir, "meta_ensemble_state.json")
        json_state = {
            "timestamp": datetime.now().isoformat(),
            "current_regime": self.current_regime,
            "strategy_weights": self.strategy_weights,
            "strategy_count": len(self.strategies)
        }
        
        with open(json_path, "w") as f:
            json.dump(json_state, f, indent=2, default=str)
    
    def load(self, path: str):
        """
        Load the meta-strategy ensemble state.
        
        Args:
            path: Path to load the state from
        """
        try:
            state = joblib.load(path)
            
            # Restore state
            self.strategy_weights = state["strategy_weights"]
            self.performance_history = defaultdict(list, state["performance_history"])
            
            # Restore regime performance with defaultdict structure
            self.regime_performance = defaultdict(lambda: defaultdict(list))
            for regime, strategy_data in state["regime_performance"].items():
                for strategy_id, history in strategy_data.items():
                    self.regime_performance[regime][strategy_id] = history
            
            self.current_regime = state["current_regime"]
            self.last_weight_update = state["last_weight_update"]
            
            # Update config with loaded values
            self.config.update(state.get("config", {}))
            
            return True
        except Exception as e:
            print(f"Error loading meta-strategy state: {str(e)}")
            return False 