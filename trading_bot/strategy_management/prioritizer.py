import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta

from trading_bot.strategy_management.interfaces import StrategyPrioritizer, MarketContext, Strategy

logger = logging.getLogger(__name__)

class WeightedContextStrategyPrioritizer(StrategyPrioritizer):
    """
    Strategy prioritizer that uses a weighted scoring system based on:
    - Market regime compatibility
    - Historical performance
    - Recency bias
    - Drawdown recovery
    - Volatility adaptation
    """
    
    def __init__(
        self,
        core_context,
        config_path: Optional[str] = None,
        min_strategy_allocation: float = 0.05,
        max_strategy_allocation: float = 0.40,
        matrix_path: Optional[str] = None
    ):
        """
        Initialize the prioritizer
        
        Args:
            core_context: Trading system core context
            config_path: Path to configuration file
            min_strategy_allocation: Minimum allocation per strategy
            max_strategy_allocation: Maximum allocation per strategy
            matrix_path: Path to strategy-regime compatibility matrix
        """
        self.core_context = core_context
        self.config_path = config_path
        self.min_allocation = min_strategy_allocation
        self.max_allocation = max_strategy_allocation
        self.matrix_path = matrix_path
        
        # Scoring weights
        self.weights = {
            "regime_compatibility": 0.35,
            "performance": 0.25,
            "drawdown_recovery": 0.15,
            "volatility_adaptation": 0.15,
            "recency": 0.10,
        }
        
        # Performance metrics timeframes
        self.timeframes = {
            "short_term": {"days": 7, "weight": 0.5},
            "medium_term": {"days": 30, "weight": 0.3},
            "long_term": {"days": 90, "weight": 0.2}
        }
        
        # Strategy-regime compatibility matrix
        self.strategy_regime_matrix = {}
        
        # Last run timestamp
        self.last_run_time = None
        
        # Cache for strategy scores
        self.strategy_scores_cache = {}
        
        # Rotation parameters
        self.rotation_cooldown_days = 5
        self.last_rotation_date = None
        
        # Load configuration
        self._set_default_config()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
            
        # Load strategy-regime matrix
        if matrix_path and os.path.exists(matrix_path):
            self._load_strategy_regime_matrix(matrix_path)
        else:
            self._initialize_strategy_regime_matrix()
            
        logger.info(f"Initialized WeightedContextStrategyPrioritizer with {len(self.weights)} factors")
    
    def _set_default_config(self):
        """Set default configuration"""
        self.config = {
            "weights": self.weights,
            "timeframes": self.timeframes,
            "min_allocation": self.min_allocation,
            "max_allocation": self.max_allocation,
            "rotation_cooldown_days": self.rotation_cooldown_days,
            "performance_metrics": {
                "sharpe_ratio": {"weight": 0.3, "higher_is_better": True},
                "sortino_ratio": {"weight": 0.3, "higher_is_better": True},
                "win_rate": {"weight": 0.2, "higher_is_better": True},
                "profit_factor": {"weight": 0.2, "higher_is_better": True},
                "max_drawdown": {"weight": 0.2, "higher_is_better": False}
            }
        }
    
    def _load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update configuration
            self.config.update(config)
            
            # Update instance variables
            self.weights = self.config.get("weights", self.weights)
            self.timeframes = self.config.get("timeframes", self.timeframes)
            self.min_allocation = self.config.get("min_allocation", self.min_allocation)
            self.max_allocation = self.config.get("max_allocation", self.max_allocation)
            self.rotation_cooldown_days = self.config.get("rotation_cooldown_days", self.rotation_cooldown_days)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _initialize_strategy_regime_matrix(self):
        """Initialize the strategy-regime compatibility matrix with default values"""
        # Get all strategy IDs
        strategy_ids = self.core_context.get_strategy_ids()
        
        # Default regimes
        regimes = ["bullish", "bearish", "sideways", "volatile", "crisis"]
        
        # Initialize matrix with neutral values (0.5)
        self.strategy_regime_matrix = {
            strategy_id: {regime: 0.5 for regime in regimes}
            for strategy_id in strategy_ids
        }
        
        # Save matrix
        if self.matrix_path:
            self._save_strategy_regime_matrix(self.matrix_path)
    
    def _load_strategy_regime_matrix(self, matrix_path: str):
        """
        Load strategy-regime compatibility matrix
        
        Args:
            matrix_path: Path to matrix file
        """
        try:
            with open(matrix_path, 'r') as f:
                self.strategy_regime_matrix = json.load(f)
            logger.info(f"Loaded strategy-regime matrix from {matrix_path}")
        except Exception as e:
            logger.error(f"Error loading strategy-regime matrix: {str(e)}")
            self._initialize_strategy_regime_matrix()
    
    def _save_strategy_regime_matrix(self, matrix_path: str):
        """
        Save strategy-regime compatibility matrix
        
        Args:
            matrix_path: Path to save matrix
        """
        try:
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            with open(matrix_path, 'w') as f:
                json.dump(self.strategy_regime_matrix, f, indent=2)
            logger.info(f"Saved strategy-regime matrix to {matrix_path}")
        except Exception as e:
            logger.error(f"Error saving strategy-regime matrix: {str(e)}")
    
    def update_strategy_regime_compatibility(self, strategy_id: str, regime: str, performance: float):
        """
        Update the strategy-regime compatibility matrix based on performance
        
        Args:
            strategy_id: Strategy ID
            regime: Market regime
            performance: Performance metric (0-1 scale)
        """
        if strategy_id not in self.strategy_regime_matrix:
            self.strategy_regime_matrix[strategy_id] = {}
        
        # Update compatibility with exponential moving average
        current = self.strategy_regime_matrix.get(strategy_id, {}).get(regime, 0.5)
        alpha = 0.2  # Learning rate
        updated = (1 - alpha) * current + alpha * performance
        
        # Ensure value is between 0 and 1
        updated = max(0.0, min(1.0, updated))
        
        # Update matrix
        if strategy_id not in self.strategy_regime_matrix:
            self.strategy_regime_matrix[strategy_id] = {}
        self.strategy_regime_matrix[strategy_id][regime] = updated
        
        # Save matrix
        if self.matrix_path:
            self._save_strategy_regime_matrix(self.matrix_path)
            
        logger.debug(f"Updated compatibility for {strategy_id} in {regime} regime: {current:.2f} -> {updated:.2f}")
    
    def update_performance_data(self, strategy_id: str, performance_data: Dict[str, Any]):
        """
        Update performance data for a strategy
        
        Args:
            strategy_id: Strategy ID
            performance_data: Performance metrics
        """
        strategy = self.core_context.get_strategy(strategy_id)
        if strategy:
            strategy.update_performance(performance_data)
            logger.debug(f"Updated performance data for strategy {strategy_id}")
            
            # Clear cache
            if strategy_id in self.strategy_scores_cache:
                del self.strategy_scores_cache[strategy_id]
    
    def _calculate_regime_compatibility_score(self, strategy_id: str, market_context: MarketContext) -> float:
        """
        Calculate regime compatibility score for a strategy
        
        Args:
            strategy_id: Strategy ID
            market_context: Market context
            
        Returns:
            Compatibility score (0-1)
        """
        # Get current regime and probabilities
        current_regime = market_context.get_value("market_regime")
        regime_probs = market_context.get_value("regime_probabilities", {})
        
        if not current_regime or not regime_probs:
            # Default to neutral score if no regime data
            return 0.5
        
        # Get strategy compatibility for current regime
        strategy_compatibilities = self.strategy_regime_matrix.get(strategy_id, {})
        
        # Calculate weighted score across all regimes
        score = 0.0
        total_weight = 0.0
        
        for regime, probability in regime_probs.items():
            compatibility = strategy_compatibilities.get(regime, 0.5)
            score += compatibility * probability
            total_weight += probability
        
        # Normalize score
        if total_weight > 0:
            score /= total_weight
        else:
            score = 0.5
            
        return score
    
    def _calculate_performance_score(self, strategy: Strategy) -> float:
        """
        Calculate performance score based on multiple metrics and timeframes
        
        Args:
            strategy: Strategy object
            
        Returns:
            Performance score (0-1)
        """
        performance_metrics = strategy.get_performance_metrics()
        if not performance_metrics:
            return 0.5  # Neutral score if no data
            
        metric_scores = []
        metric_weights = []
        
        # Calculate scores for each performance metric and timeframe
        for metric_name, metric_config in self.config["performance_metrics"].items():
            weight = metric_config["weight"]
            higher_is_better = metric_config["higher_is_better"]
            
            for timeframe_name, timeframe_config in self.timeframes.items():
                # Get metric value for specific timeframe
                timeframe_key = f"{metric_name}_{timeframe_name}"
                if timeframe_key not in performance_metrics:
                    continue
                    
                value = performance_metrics[timeframe_key]
                
                # Normalize to 0-1 scale based on reasonable bounds
                # These bounds should be calibrated for each metric
                normalized_value = self._normalize_metric(metric_name, value)
                
                # Invert if lower is better
                if not higher_is_better:
                    normalized_value = 1.0 - normalized_value
                
                # Apply timeframe weight
                timeframe_weight = timeframe_config["weight"]
                total_weight = weight * timeframe_weight
                
                metric_scores.append(normalized_value)
                metric_weights.append(total_weight)
        
        # Calculate weighted average
        if sum(metric_weights) > 0:
            weighted_score = sum(s * w for s, w in zip(metric_scores, metric_weights)) / sum(metric_weights)
            return weighted_score
        else:
            return 0.5  # Neutral score if no data
    
    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """
        Normalize a metric value to 0-1 scale
        
        Args:
            metric_name: Name of the metric
            value: Raw metric value
            
        Returns:
            Normalized value (0-1)
        """
        # Define reasonable bounds for each metric
        bounds = {
            "sharpe_ratio": {"min": -2.0, "max": 4.0},
            "sortino_ratio": {"min": -2.0, "max": 4.0},
            "win_rate": {"min": 0.0, "max": 1.0},
            "profit_factor": {"min": 0.0, "max": 3.0},
            "max_drawdown": {"min": 0.0, "max": 0.5}
        }
        
        if metric_name not in bounds:
            return 0.5  # Default to neutral for unknown metrics
            
        min_val = bounds[metric_name]["min"]
        max_val = bounds[metric_name]["max"]
        
        # Clamp value to bounds
        clamped = max(min_val, min(max_val, value))
        
        # Normalize to 0-1
        return (clamped - min_val) / (max_val - min_val)
    
    def _calculate_drawdown_recovery_score(self, strategy: Strategy, market_context: MarketContext) -> float:
        """
        Calculate score based on drawdown recovery capability
        
        Args:
            strategy: Strategy object
            market_context: Market context
            
        Returns:
            Drawdown recovery score (0-1)
        """
        # Get market stress level
        market_stress = market_context.get_value("market_stress_index", 0.5)
        
        # Get strategy's drawdown recovery metrics
        metrics = strategy.get_performance_metrics()
        recovery_speed = metrics.get("drawdown_recovery_speed", 0.5)
        past_recoveries = metrics.get("past_recovery_success", 0.5)
        
        # Calculate score: higher values mean better recovery during stress
        if market_stress > 0.7:  # High stress
            # Weight recovery capabilities more heavily
            score = 0.7 * recovery_speed + 0.3 * past_recoveries
        else:  # Normal or low stress
            # Weight past recoveries more
            score = 0.3 * recovery_speed + 0.7 * past_recoveries
            
        return score
    
    def _calculate_volatility_adaptation_score(self, strategy: Strategy, market_context: MarketContext) -> float:
        """
        Calculate score based on volatility adaptation
        
        Args:
            strategy: Strategy object
            market_context: Market context
            
        Returns:
            Volatility adaptation score (0-1)
        """
        # Get current volatility level
        volatility = market_context.get_value("market_volatility", 0.5)
        
        # Get strategy's volatility adaptation metrics
        metrics = strategy.get_performance_metrics()
        high_vol_performance = metrics.get("high_volatility_performance", 0.5)
        low_vol_performance = metrics.get("low_volatility_performance", 0.5)
        
        # Score based on current volatility environment
        if volatility > 0.7:  # High volatility
            return high_vol_performance
        elif volatility < 0.3:  # Low volatility
            return low_vol_performance
        else:  # Medium volatility
            # Linear interpolation between low and high
            normalized_vol = (volatility - 0.3) / 0.4
            return low_vol_performance * (1 - normalized_vol) + high_vol_performance * normalized_vol
    
    def _calculate_recency_score(self, strategy: Strategy) -> float:
        """
        Calculate score based on recent performance
        
        Args:
            strategy: Strategy object
            
        Returns:
            Recency score (0-1)
        """
        metrics = strategy.get_performance_metrics()
        
        # Get performance for different timeframes
        recent_perf = metrics.get("sharpe_ratio_short_term", 0.0)
        medium_perf = metrics.get("sharpe_ratio_medium_term", 0.0)
        
        # Normalize
        recent_score = self._normalize_metric("sharpe_ratio", recent_perf)
        medium_score = self._normalize_metric("sharpe_ratio", medium_perf)
        
        # Weight recent performance more heavily
        score = 0.7 * recent_score + 0.3 * medium_score
        
        return score
    
    def _should_rotate_strategies(self, market_context: MarketContext) -> bool:
        """
        Determine if strategies should be rotated
        
        Args:
            market_context: Market context
            
        Returns:
            True if rotation should occur
        """
        current_date = datetime.now().date()
        
        # Check cooldown period
        if self.last_rotation_date and (current_date - self.last_rotation_date).days < self.rotation_cooldown_days:
            logger.debug("Strategy rotation in cooldown period")
            return False
            
        # Check if market conditions have changed significantly
        regime_change = market_context.get_value("regime_changed_recently", False)
        volatility_spike = market_context.get_value("volatility_spike", False)
        
        if regime_change or volatility_spike:
            logger.info(f"Strategy rotation triggered by {'regime change' if regime_change else 'volatility spike'}")
            return True
            
        # Regular scheduled rotation
        return True
    
    def prioritize_strategies(
        self, 
        market_context: MarketContext,
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Prioritize strategies based on current market context
        
        Args:
            market_context: Current market context
            include_reasoning: Whether to include reasoning
            
        Returns:
            Dictionary with strategy allocations and optional reasoning
        """
        # Get all strategy IDs
        strategy_ids = self.core_context.get_strategy_ids()
        
        # Check if we should rotate strategies
        should_rotate = self._should_rotate_strategies(market_context)
        
        # Prepare result
        result = {
            "allocations": {},
            "timestamp": datetime.now().isoformat(),
            "rotation_occurred": should_rotate
        }
        
        if include_reasoning:
            result["reasoning"] = {}
            
        # If no rotation, return current allocations
        if not should_rotate and self.last_run_time:
            for strategy_id in strategy_ids:
                strategy = self.core_context.get_strategy(strategy_id)
                if strategy and strategy.is_enabled():
                    allocation = strategy.get_portfolio_allocation()
                    result["allocations"][strategy_id] = allocation
            
            return result
            
        # Calculate scores for each strategy
        strategy_scores = {}
        total_score = 0.0
        
        for strategy_id in strategy_ids:
            strategy = self.core_context.get_strategy(strategy_id)
            
            # Skip disabled strategies
            if not strategy or not strategy.is_enabled():
                continue
                
            # Calculate component scores
            regime_score = self._calculate_regime_compatibility_score(strategy_id, market_context)
            performance_score = self._calculate_performance_score(strategy)
            drawdown_score = self._calculate_drawdown_recovery_score(strategy, market_context)
            volatility_score = self._calculate_volatility_adaptation_score(strategy, market_context)
            recency_score = self._calculate_recency_score(strategy)
            
            # Calculate weighted score
            weighted_score = (
                self.weights["regime_compatibility"] * regime_score +
                self.weights["performance"] * performance_score +
                self.weights["drawdown_recovery"] * drawdown_score +
                self.weights["volatility_adaptation"] * volatility_score +
                self.weights["recency"] * recency_score
            )
            
            # Store scores
            strategy_scores[strategy_id] = {
                "total": weighted_score,
                "components": {
                    "regime_compatibility": regime_score,
                    "performance": performance_score,
                    "drawdown_recovery": drawdown_score,
                    "volatility_adaptation": volatility_score,
                    "recency": recency_score
                }
            }
            
            total_score += weighted_score
            
        # Calculate allocations
        if total_score > 0:
            # Get number of strategies to allocate to
            num_strategies = len(strategy_scores)
            
            # Normalize scores and calculate initial allocations
            raw_allocations = {}
            for strategy_id, score_data in strategy_scores.items():
                raw_allocations[strategy_id] = score_data["total"] / total_score
                
            # Adjust allocations to respect min/max constraints
            adjusted_allocations = self._adjust_allocations(raw_allocations, num_strategies)
            
            # Update result
            result["allocations"] = adjusted_allocations
            
            if include_reasoning:
                # Include scores and reasoning
                for strategy_id in adjusted_allocations.keys():
                    score_data = strategy_scores[strategy_id]
                    result["reasoning"][strategy_id] = {
                        "scores": score_data["components"],
                        "total_score": score_data["total"],
                        "raw_allocation": raw_allocations[strategy_id],
                        "adjusted_allocation": adjusted_allocations[strategy_id]
                    }
        
        # Update last rotation date if rotation occurred
        if should_rotate:
            self.last_rotation_date = datetime.now().date()
            
        # Update last run time
        self.last_run_time = datetime.now()
        
        # Apply allocations to strategies
        for strategy_id, allocation in result["allocations"].items():
            strategy = self.core_context.get_strategy(strategy_id)
            if strategy:
                strategy.set_portfolio_allocation(allocation)
                
        return result
    
    def _adjust_allocations(self, raw_allocations: Dict[str, float], num_strategies: int) -> Dict[str, float]:
        """
        Adjust allocations to respect min/max constraints
        
        Args:
            raw_allocations: Initial allocations
            num_strategies: Number of strategies
            
        Returns:
            Adjusted allocations
        """
        # Sort strategies by raw allocation (descending)
        sorted_allocations = sorted(
            raw_allocations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        adjusted_allocations = {}
        remaining_allocation = 1.0
        remaining_strategies = num_strategies
        
        # First pass: apply minimum allocations
        for strategy_id, raw_allocation in sorted_allocations:
            if remaining_allocation <= 0 or remaining_strategies <= 0:
                adjusted_allocations[strategy_id] = 0.0
                continue
                
            # Apply minimum allocation
            allocation = max(self.min_allocation, raw_allocation)
            
            # Ensure we don't exceed remaining allocation
            allocation = min(allocation, remaining_allocation)
            
            adjusted_allocations[strategy_id] = allocation
            remaining_allocation -= allocation
            remaining_strategies -= 1
            
        # Reset for second pass
        remaining_allocation = 1.0 - sum(adjusted_allocations.values())
        
        # Second pass: apply maximum allocations and redistribute excess
        if remaining_allocation > 0:
            # Find strategies below max allocation
            below_max = {
                strategy_id: alloc for strategy_id, alloc in adjusted_allocations.items()
                if alloc < self.max_allocation and alloc > 0
            }
            
            while remaining_allocation > 0.001 and below_max:
                # Distribute remaining allocation proportionally
                total_below_max = sum(below_max.values())
                
                # Calculate additional allocations
                additional_allocations = {}
                for strategy_id, alloc in below_max.items():
                    weight = alloc / total_below_max
                    additional = min(
                        remaining_allocation * weight,
                        self.max_allocation - adjusted_allocations[strategy_id]
                    )
                    additional_allocations[strategy_id] = additional
                    
                # Apply additional allocations
                total_additional = sum(additional_allocations.values())
                if total_additional <= 0:
                    break
                    
                for strategy_id, additional in additional_allocations.items():
                    adjusted_allocations[strategy_id] += additional
                    
                # Update remaining
                remaining_allocation -= total_additional
                
                # Update below_max strategies
                below_max = {
                    strategy_id: alloc for strategy_id, alloc in adjusted_allocations.items()
                    if alloc < self.max_allocation and alloc > 0
                }
                
        # Normalize to ensure sum is 1.0
        total = sum(adjusted_allocations.values())
        if total > 0:
            for strategy_id in adjusted_allocations:
                adjusted_allocations[strategy_id] /= total
                
        return adjusted_allocations 