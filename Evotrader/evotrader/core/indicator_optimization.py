"""
Automatic indicator optimization during evolutionary cycles.

This module enhances the EvoTrader system with capabilities to automatically
adjust indicator parameters during the evolutionary process, allowing for
more robust and adaptive trading systems.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
import copy
import numpy as np
from collections import defaultdict

from ..core.strategy import Strategy, StrategyParameter
from ..utils.indicator_system import Indicator, IndicatorFactory
from ..strategies.enhanced_strategy import EnhancedStrategy
from ..strategies.multi_timeframe import MultiTimeFrameStrategy

logger = logging.getLogger(__name__)


class IndicatorOptimizationMixin:
    """
    Mixin class for strategies to enable indicator parameter optimization.
    
    This class provides methods for strategies to automatically adjust
    their indicator parameters based on performance feedback, both during
    initialization and between evolutionary cycles.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        # Call the parent's __init__ if it exists
        super().__init__(*args, **kwargs)
        
        # Optimization tracking
        self.indicator_performance_history = {}
        self.optimization_cycles = 0
        
        # Mutation rates
        self.mutation_probability = 0.3  # 30% chance of parameter mutation
        self.mutation_strength = 0.2  # How much to adjust parameters
        
        # Success tracking
        self.signals_success = []  # List of (signal, was_profitable)
        self.indicator_success = defaultdict(list)  # indicator_name -> list of success rates
    
    def optimize_indicators(self, performance_data: Dict[str, Any]) -> None:
        """
        Optimize indicator parameters based on performance data.
        
        Args:
            performance_data: Dictionary of performance metrics
        """
        if not hasattr(self, 'indicators'):
            logger.warning("Strategy does not have indicators attribute")
            return
            
        # Record performance for history
        self.indicator_performance_history[self.optimization_cycles] = copy.deepcopy(performance_data)
        
        # Extract key performance metrics
        profit = performance_data.get('profit', 0)
        sharpe = performance_data.get('sharpe_ratio', 0)
        max_drawdown = performance_data.get('max_drawdown', 100)
        win_rate = performance_data.get('win_rate', 0)
        
        # Combine metrics into a single score
        # Higher is better
        performance_score = (
            0.4 * profit + 
            0.3 * sharpe + 
            0.1 * (100 - max_drawdown) + 
            0.2 * win_rate
        )
        
        # Analyze signal success rates
        self._analyze_signal_success()
        
        # Based on performance and signal analysis, adjust indicators
        self._adjust_indicator_parameters(performance_score)
        
        # Increment the optimization cycle counter
        self.optimization_cycles += 1
        
        logger.info(f"Completed indicator optimization cycle {self.optimization_cycles}")
    
    def _analyze_signal_success(self) -> None:
        """Analyze the success rates of signals by indicator."""
        # Skip if no signals recorded
        if not self.signals_success:
            return
            
        # Calculate success rate by indicator
        indicator_counts = defaultdict(int)
        indicator_success_counts = defaultdict(int)
        
        for signal, was_profitable in self.signals_success:
            # Extract indicators from signal params
            indicators_used = signal.params.get('indicators_used', [])
            
            for indicator_name in indicators_used:
                indicator_counts[indicator_name] += 1
                if was_profitable:
                    indicator_success_counts[indicator_name] += 1
        
        # Calculate success rates
        for indicator_name, count in indicator_counts.items():
            if count > 0:
                success_rate = indicator_success_counts[indicator_name] / count
                self.indicator_success[indicator_name].append(success_rate)
                
                logger.debug(f"Indicator {indicator_name} success rate: {success_rate:.2f}")
    
    def _adjust_indicator_parameters(self, performance_score: float) -> None:
        """
        Adjust indicator parameters based on performance score.
        
        Args:
            performance_score: Overall performance score
        """
        # Apply different strategies based on performance
        if performance_score > 50:
            # Good performance - subtle refinement
            self._refine_parameters()
        elif performance_score > 20:
            # Mediocre performance - balanced approach
            self._balanced_parameter_adjustment()
        else:
            # Poor performance - significant changes
            self._significant_parameter_changes()
    
    def _refine_parameters(self) -> None:
        """Make subtle refinements to parameters for already good performance."""
        # Small, targeted adjustments
        mutation_prob = self.mutation_probability * 0.5  # Lower probability
        mutation_strength = self.mutation_strength * 0.5  # Gentler adjustments
        
        # Only adjust parameters of indicators with below average success
        self._adjust_underperforming_indicators(mutation_prob, mutation_strength)
    
    def _balanced_parameter_adjustment(self) -> None:
        """Make balanced adjustments to parameters."""
        # Standard mutations
        for symbol, indicators in getattr(self, 'indicators', {}).items():
            for indicator_name, indicator in indicators.items():
                if hasattr(indicator, 'params'):
                    self._mutate_indicator_params(
                        indicator, 
                        self.mutation_probability, 
                        self.mutation_strength
                    )
    
    def _significant_parameter_changes(self) -> None:
        """Make significant changes to parameters for poor performance."""
        # Higher mutations
        mutation_prob = min(0.7, self.mutation_probability * 2)  # Higher probability
        mutation_strength = min(0.5, self.mutation_strength * 2)  # Stronger adjustments
        
        # Adjust all indicators
        for symbol, indicators in getattr(self, 'indicators', {}).items():
            for indicator_name, indicator in indicators.items():
                if hasattr(indicator, 'params'):
                    self._mutate_indicator_params(indicator, mutation_prob, mutation_strength)
    
    def _adjust_underperforming_indicators(
        self, 
        mutation_prob: float, 
        mutation_strength: float
    ) -> None:
        """
        Adjust parameters for underperforming indicators.
        
        Args:
            mutation_prob: Probability of mutation
            mutation_strength: Strength of mutation
        """
        # Calculate average success rate
        all_rates = [rate for rates in self.indicator_success.values() for rate in rates]
        avg_success = sum(all_rates) / len(all_rates) if all_rates else 0.5
        
        # Identify underperforming indicators
        underperforming = []
        for indicator_name, rates in self.indicator_success.items():
            if rates and sum(rates) / len(rates) < avg_success:
                underperforming.append(indicator_name)
        
        # Adjust parameters for underperforming indicators
        for symbol, indicators in getattr(self, 'indicators', {}).items():
            for indicator_name, indicator in indicators.items():
                if hasattr(indicator, 'params') and indicator_name in underperforming:
                    self._mutate_indicator_params(indicator, mutation_prob, mutation_strength)
    
    def _mutate_indicator_params(
        self, 
        indicator: Indicator, 
        mutation_prob: float, 
        mutation_strength: float
    ) -> None:
        """
        Mutate the parameters of an indicator.
        
        Args:
            indicator: Indicator to mutate
            mutation_prob: Probability of mutation
            mutation_strength: Strength of mutation
        """
        if not hasattr(indicator, 'params'):
            return
            
        for param_name, param_value in indicator.params.items():
            # Skip non-numeric parameters
            if not isinstance(param_value, (int, float)):
                continue
                
            # Decide whether to mutate
            if random.random() < mutation_prob:
                # Determine mutation bounds based on type
                if isinstance(param_value, int):
                    min_val = max(1, int(param_value * (1 - mutation_strength)))
                    max_val = int(param_value * (1 + mutation_strength))
                    new_value = random.randint(min_val, max_val)
                else:  # float
                    min_val = param_value * (1 - mutation_strength)
                    max_val = param_value * (1 + mutation_strength)
                    new_value = random.uniform(min_val, max_val)
                
                # Update the parameter
                indicator.params[param_name] = new_value
                
                logger.debug(f"Mutated {indicator.__class__.__name__} parameter {param_name}: {param_value} -> {new_value}")
    
    def record_signal_result(self, signal: Any, was_profitable: bool) -> None:
        """
        Record the result of a signal for future optimization.
        
        Args:
            signal: Trading signal
            was_profitable: Whether the signal resulted in profit
        """
        self.signals_success.append((signal, was_profitable))


class OptimizedStrategy(EnhancedStrategy, IndicatorOptimizationMixin):
    """
    Strategy that supports automatic indicator optimization.
    
    This strategy class combines the enhanced indicator system with
    automatic parameter optimization during evolution.
    """
    
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the optimized strategy."""
        EnhancedStrategy.__init__(self, strategy_id, parameters)
        IndicatorOptimizationMixin.__init__(self)
        
        # Set up optimization parameters
        self.enable_optimization = parameters.get('enable_optimization', True)
        
        # Add metadata to each indicator to track its performance
        self.indicator_metadata = {}
    
    def setup_indicators(self, symbol: str) -> None:
        """
        Set up indicators with automatic optimization capabilities.
        Override in subclasses.
        """
        super().setup_indicators(symbol)
        
        # Add metadata to track indicators
        for indicator_name, indicator in self.indicators[symbol].items():
            if indicator_name not in self.indicator_metadata:
                self.indicator_metadata[indicator_name] = {
                    'success_rate': 0.5,  # Initial neutral success rate
                    'usage_count': 0,
                    'win_count': 0,
                    'loss_count': 0
                }
    
    def on_generation_end(self, performance_data: Dict[str, Any]) -> None:
        """
        Process feedback at the end of an evolutionary generation.
        
        Args:
            performance_data: Dictionary of performance metrics
        """
        if not self.enable_optimization:
            return
            
        # Call the optimization method
        self.optimize_indicators(performance_data)
    
    def on_order_filled(self, order_data: Dict[str, Any]) -> None:
        """
        Update strategy state when an order is filled.
        Track order results for optimization.
        """
        # Call parent implementation
        super().on_order_filled(order_data)
        
        # Extract information for optimization
        order = order_data.get("order")
        if not order:
            return
            
        # Record the result for optimization
        signal = order_data.get("signal")
        pnl = order_data.get("pnl", 0)
        
        if signal and pnl is not None:
            was_profitable = pnl > 0
            
            # Record signal result
            self.record_signal_result(signal, was_profitable)
            
            # Update indicator metadata
            indicators_used = signal.params.get('indicators_used', [])
            for indicator_name in indicators_used:
                if indicator_name in self.indicator_metadata:
                    metadata = self.indicator_metadata[indicator_name]
                    metadata['usage_count'] += 1
                    
                    if was_profitable:
                        metadata['win_count'] += 1
                    else:
                        metadata['loss_count'] += 1
                        
                    # Update success rate
                    if metadata['usage_count'] > 0:
                        metadata['success_rate'] = metadata['win_count'] / metadata['usage_count']


class OptimizedMultiTimeFrameStrategy(MultiTimeFrameStrategy, IndicatorOptimizationMixin):
    """
    Multi-timeframe strategy with automatic indicator optimization.
    
    This strategy optimizes indicators across multiple timeframes based on
    performance feedback during evolution.
    """
    
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the optimized multi-timeframe strategy."""
        MultiTimeFrameStrategy.__init__(self, strategy_id, parameters)
        IndicatorOptimizationMixin.__init__(self)
        
        # Set up optimization parameters
        self.enable_optimization = parameters.get('enable_optimization', True)
        
        # Track performance by timeframe
        self.timeframe_performance = {}
    
    def setup_timeframe_indicators(self, timeframe: str, symbol: str) -> None:
        """
        Set up indicators for a specific timeframe with optimization capabilities.
        Override in subclasses.
        """
        super().setup_timeframe_indicators(timeframe, symbol)
    
    def on_generation_end(self, performance_data: Dict[str, Any]) -> None:
        """
        Process feedback at the end of an evolutionary generation.
        
        Args:
            performance_data: Dictionary of performance metrics
        """
        if not self.enable_optimization:
            return
            
        # Call the optimization method
        self.optimize_indicators(performance_data)
        
        # Analyze timeframe performance
        self._analyze_timeframe_performance()
        
        # Adjust timeframe weights based on performance
        self._optimize_timeframe_weights()
    
    def _analyze_timeframe_performance(self) -> None:
        """Analyze the performance contribution from each timeframe."""
        # Skip if no signals recorded
        if not self.signals_success:
            return
            
        # Initialize timeframe success counters
        timeframe_signals = defaultdict(int)
        timeframe_success = defaultdict(int)
        
        for signal, was_profitable in self.signals_success:
            # Extract timeframe from signal params
            timeframe = signal.params.get('timeframe')
            if timeframe:
                timeframe_signals[timeframe] += 1
                if was_profitable:
                    timeframe_success[timeframe] += 1
        
        # Calculate success rates by timeframe
        for timeframe, count in timeframe_signals.items():
            if count > 0:
                success_rate = timeframe_success[timeframe] / count
                
                if timeframe not in self.timeframe_performance:
                    self.timeframe_performance[timeframe] = []
                    
                self.timeframe_performance[timeframe].append(success_rate)
                
                logger.debug(f"Timeframe {timeframe} success rate: {success_rate:.2f}")
    
    def _optimize_timeframe_weights(self) -> None:
        """Adjust timeframe weights based on performance."""
        # Skip if we don't have enough data
        if not self.timeframe_performance:
            return
            
        # Calculate average success rates
        avg_rates = {}
        for timeframe, rates in self.timeframe_performance.items():
            if rates:
                avg_rates[timeframe] = sum(rates) / len(rates)
        
        # Skip if we don't have rates for all timeframes
        primary_tf = self.parameters.get("primary_timeframe")
        secondary_tf = self.parameters.get("secondary_timeframe")
        tertiary_tf = self.parameters.get("tertiary_timeframe")
        
        if not all(tf in avg_rates for tf in [primary_tf, secondary_tf, tertiary_tf]):
            return
            
        # Calculate total success rate
        total_success = sum(avg_rates.values())
        
        if total_success > 0:
            # Assign weights proportional to success rates
            # With some minimum weight to ensure all timeframes contribute
            min_weight = 0.1
            remaining_weight = 1.0 - (min_weight * 3)  # 3 timeframes
            
            # Calculate normalized weights
            primary_weight = min_weight + (avg_rates[primary_tf] / total_success) * remaining_weight
            secondary_weight = min_weight + (avg_rates[secondary_tf] / total_success) * remaining_weight
            tertiary_weight = min_weight + (avg_rates[tertiary_tf] / total_success) * remaining_weight
            
            # Normalize to ensure they sum to 1.0
            total_weight = primary_weight + secondary_weight + tertiary_weight
            primary_weight /= total_weight
            secondary_weight /= total_weight
            tertiary_weight /= total_weight
            
            # Update the parameters
            self.parameters["primary_weight"] = primary_weight
            self.parameters["secondary_weight"] = secondary_weight
            self.parameters["tertiary_weight"] = tertiary_weight
            
            logger.info(f"Optimized timeframe weights: P={primary_weight:.2f}, S={secondary_weight:.2f}, T={tertiary_weight:.2f}")
    
    def record_signal_result(self, signal: Any, was_profitable: bool) -> None:
        """
        Record the result of a signal for future optimization.
        Also record the timeframe that generated the signal.
        
        Args:
            signal: Trading signal
            was_profitable: Whether the signal resulted in profit
        """
        # Call parent implementation
        super().record_signal_result(signal, was_profitable)
        
        # Extract timeframe information
        timeframe = signal.params.get('timeframe')
        if timeframe:
            # We could add additional timeframe-specific tracking here
            pass


# Helper functions to integrate with the evolutionary framework

def apply_indicator_optimization(strategy: Strategy, fitness_data: Dict[str, Any]) -> None:
    """
    Apply indicator optimization to a strategy based on fitness data.
    
    Args:
        strategy: Strategy to optimize
        fitness_data: Dictionary of fitness metrics
    """
    # Check if the strategy supports indicator optimization
    if hasattr(strategy, 'on_generation_end'):
        strategy.on_generation_end(fitness_data)


def register_optimization_callbacks(evolution_manager: Any) -> None:
    """
    Register callbacks for indicator optimization with the evolution manager.
    
    Args:
        evolution_manager: The evolution manager instance
    """
    if hasattr(evolution_manager, 'add_post_generation_callback'):
        evolution_manager.add_post_generation_callback(apply_indicator_optimization)
        logger.info("Registered indicator optimization callbacks with evolution manager")
    else:
        logger.warning("Evolution manager does not support callbacks, indicator optimization disabled")


def optimize_strategy_parameters(
    strategy: Strategy, 
    performance_history: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Analyze performance history and suggest optimized strategy parameters.
    
    Args:
        strategy: Strategy to optimize
        performance_history: List of performance dictionaries
        
    Returns:
        Dictionary of optimized parameters or None
    """
    if not performance_history:
        return None
        
    # Extract strategy parameters
    if not hasattr(strategy, 'parameters'):
        return None
        
    # Create a copy of current parameters
    current_params = copy.deepcopy(strategy.parameters)
    
    # Analyze which parameters correlate with better performance
    param_performance = {}
    
    # Extract performance metrics from history
    profits = [perf.get('profit', 0) for perf in performance_history]
    sharpes = [perf.get('sharpe_ratio', 0) for perf in performance_history]
    drawdowns = [perf.get('max_drawdown', 100) for perf in performance_history]
    win_rates = [perf.get('win_rate', 0) for perf in performance_history]
    
    # Calculate performance scores
    scores = [
        0.4 * profit + 0.3 * sharpe + 0.1 * (100 - drawdown) + 0.2 * win_rate
        for profit, sharpe, drawdown, win_rate in zip(profits, sharpes, drawdowns, win_rates)
    ]
    
    # Find best parameter set
    best_idx = scores.index(max(scores)) if scores else 0
    best_performance = performance_history[best_idx]
    
    # Get parameters associated with best performance
    if 'parameters' in best_performance:
        best_params = best_performance['parameters']
        
        # Filter out non-mutable parameters
        if hasattr(strategy, 'get_parameters'):
            mutable_params = {
                param.name: param
                for param in strategy.get_parameters()
                if hasattr(param, 'is_mutable') and param.is_mutable
            }
            
            # Only keep mutable parameters
            best_params = {
                k: v for k, v in best_params.items()
                if k in mutable_params
            }
            
        return best_params
    
    return None
