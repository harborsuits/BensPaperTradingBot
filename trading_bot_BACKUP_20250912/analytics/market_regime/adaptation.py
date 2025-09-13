"""
Parameter Optimizer for Market Regimes

This module provides functionality to optimize strategy parameters for different
market regimes and adapt strategies in real-time to changing market conditions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
from datetime import datetime
import threading

# Import local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeType

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Optimizes strategy parameters for different market regimes and provides
    dynamic adaptation as market conditions change.
    
    Features:
    - Maintains optimal parameter sets for each strategy and regime
    - Tracks parameter performance across regimes
    - Smoothly transitions parameters between regimes
    - Learns from historical performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parameter optimizer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Directory to store parameter sets
        self.parameter_dir = self.config.get("parameter_dir", "data/regime_parameters")
        
        # Optimal parameters by strategy and regime
        self.optimal_parameters: Dict[str, Dict[MarketRegimeType, Dict[str, Any]]] = {}
        
        # Parameter performance history
        self.parameter_performance: Dict[str, Dict[MarketRegimeType, List[Dict[str, Any]]]] = {}
        
        # Parameter transition settings
        self.transition_smoothing = self.config.get("transition_smoothing", True)
        self.smoothing_factor = self.config.get("smoothing_factor", 0.3)
        
        # Current transition state (for smooth parameter changes)
        self.current_transitions: Dict[str, Dict[str, Any]] = {}
        
        # Recent parameters by strategy and symbol
        self.recent_parameters: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._load_parameter_sets()
        
        logger.info("Parameter Optimizer initialized")
    
    def _load_parameter_sets(self) -> None:
        """Load parameter sets from disk."""
        try:
            # Create parameter directory if it doesn't exist
            os.makedirs(self.parameter_dir, exist_ok=True)
            
            # Check for parameter files
            for strategy_file in os.listdir(self.parameter_dir):
                if strategy_file.endswith(".json") and strategy_file.startswith("strategy_"):
                    strategy_id = strategy_file[9:-5]  # Remove "strategy_" prefix and ".json" suffix
                    
                    file_path = os.path.join(self.parameter_dir, strategy_file)
                    with open(file_path, 'r') as f:
                        strategy_params = json.load(f)
                        
                    # Convert string keys to MarketRegimeType
                    regime_params = {}
                    for regime_str, params in strategy_params.items():
                        try:
                            regime_type = MarketRegimeType(regime_str)
                            regime_params[regime_type] = params
                        except ValueError:
                            logger.warning(f"Unknown regime type in parameter file: {regime_str}")
                    
                    self.optimal_parameters[strategy_id] = regime_params
            
            loaded_count = len(self.optimal_parameters)
            if loaded_count > 0:
                logger.info(f"Loaded parameter sets for {loaded_count} strategies")
            else:
                logger.info("No parameter sets found, using defaults")
                
        except Exception as e:
            logger.error(f"Error loading parameter sets: {str(e)}")
    
    def _save_parameter_sets(self) -> None:
        """Save parameter sets to disk."""
        try:
            # Create parameter directory if it doesn't exist
            os.makedirs(self.parameter_dir, exist_ok=True)
            
            # Save each strategy's parameters
            for strategy_id, regime_params in self.optimal_parameters.items():
                file_path = os.path.join(self.parameter_dir, f"strategy_{strategy_id}.json")
                
                # Convert MarketRegimeType to strings for JSON serialization
                serializable_params = {regime.value: params for regime, params in regime_params.items()}
                
                with open(file_path, 'w') as f:
                    json.dump(serializable_params, f, indent=2)
            
            logger.info(f"Saved parameter sets for {len(self.optimal_parameters)} strategies")
            
        except Exception as e:
            logger.error(f"Error saving parameter sets: {str(e)}")
    
    def get_optimal_parameters(
        self, strategy_id: str, regime_type: MarketRegimeType, 
        symbol: str, timeframe: str, confidence: float
    ) -> Dict[str, Any]:
        """
        Get optimal parameters for a strategy in a specific market regime.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Current market regime type
            symbol: Symbol being traded
            timeframe: Timeframe being traded
            confidence: Confidence level of regime classification
            
        Returns:
            Dict of optimal parameters
        """
        with self._lock:
            try:
                # Check if we have optimal parameters for this strategy and regime
                if strategy_id in self.optimal_parameters and regime_type in self.optimal_parameters[strategy_id]:
                    # Get optimal parameters for the regime
                    regime_params = self.optimal_parameters[strategy_id][regime_type]
                    
                    # If transition smoothing is enabled, apply it
                    if self.transition_smoothing:
                        return self._get_smoothed_parameters(
                            strategy_id=strategy_id,
                            regime_type=regime_type,
                            regime_params=regime_params,
                            symbol=symbol,
                            timeframe=timeframe,
                            confidence=confidence
                        )
                    else:
                        # Store as recent parameters
                        self._update_recent_parameters(strategy_id, symbol, regime_params)
                        return regime_params
                
                # If no optimal parameters for this regime, try default regime
                if strategy_id in self.optimal_parameters and MarketRegimeType.NORMAL in self.optimal_parameters[strategy_id]:
                    default_params = self.optimal_parameters[strategy_id][MarketRegimeType.NORMAL]
                    
                    # Store as recent parameters
                    self._update_recent_parameters(strategy_id, symbol, default_params)
                    return default_params
                
                # If still no parameters, return empty dict
                logger.warning(f"No parameters found for strategy {strategy_id} in regime {regime_type}")
                return {}
                
            except Exception as e:
                logger.error(f"Error getting optimal parameters: {str(e)}")
                return {}
    
    def _get_smoothed_parameters(
        self, strategy_id: str, regime_type: MarketRegimeType, regime_params: Dict[str, Any],
        symbol: str, timeframe: str, confidence: float
    ) -> Dict[str, Any]:
        """
        Get smoothed parameters for transition between regimes.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Current market regime type
            regime_params: Target regime parameters
            symbol: Symbol being traded
            timeframe: Timeframe being traded
            confidence: Confidence level of regime classification
            
        Returns:
            Dict of smoothed parameters
        """
        try:
            # Create transition key
            transition_key = f"{strategy_id}_{symbol}_{timeframe}"
            
            # Get recent parameters as the source
            source_params = self._get_recent_parameters(strategy_id, symbol)
            
            # If no recent parameters or same regime, just use target params
            if not source_params or transition_key not in self.current_transitions:
                self._update_recent_parameters(strategy_id, symbol, regime_params)
                return regime_params
            
            # Check if regime has changed
            current_transition = self.current_transitions.get(transition_key, {})
            current_regime = current_transition.get('regime_type')
            
            # If regime changed, start new transition
            if current_regime != regime_type:
                self.current_transitions[transition_key] = {
                    'regime_type': regime_type,
                    'progress': 0.0,
                    'source_params': source_params,
                    'target_params': regime_params,
                    'start_time': datetime.now()
                }
                current_transition = self.current_transitions[transition_key]
            
            # Calculate smoothed parameters
            smoothed_params = {}
            progress = current_transition.get('progress', 0.0)
            
            # Adjust progress based on confidence
            confidence_factor = max(0.1, confidence)  # Ensure minimum progress
            progress += self.smoothing_factor * confidence_factor
            progress = min(progress, 1.0)  # Cap at 1.0
            
            # Update progress
            self.current_transitions[transition_key]['progress'] = progress
            
            # Calculate smoothed values for each parameter
            source_params = current_transition.get('source_params', {})
            target_params = current_transition.get('target_params', {})
            
            for param_name, target_value in target_params.items():
                if param_name in source_params:
                    source_value = source_params[param_name]
                    
                    # Handle different parameter types
                    if isinstance(target_value, (int, float)) and isinstance(source_value, (int, float)):
                        # Numeric parameter
                        smoothed_value = source_value + progress * (target_value - source_value)
                        
                        # Convert to int if both source and target are ints
                        if isinstance(target_value, int) and isinstance(source_value, int):
                            smoothed_value = int(round(smoothed_value))
                        
                        smoothed_params[param_name] = smoothed_value
                    else:
                        # Non-numeric parameter, use target once we're halfway through transition
                        smoothed_params[param_name] = target_value if progress > 0.5 else source_value
                else:
                    # Parameter doesn't exist in source, use target
                    smoothed_params[param_name] = target_value
            
            # Add any source parameters not in target
            for param_name, source_value in source_params.items():
                if param_name not in target_params:
                    smoothed_params[param_name] = source_value
            
            # If transition complete, update recent parameters and clean up
            if progress >= 1.0:
                self._update_recent_parameters(strategy_id, symbol, target_params)
                del self.current_transitions[transition_key]
                return target_params
            
            # Store intermediate smoothed parameters
            self._update_recent_parameters(strategy_id, symbol, smoothed_params)
            
            return smoothed_params
            
        except Exception as e:
            logger.error(f"Error calculating smoothed parameters: {str(e)}")
            return regime_params
    
    def _update_recent_parameters(self, strategy_id: str, symbol: str, params: Dict[str, Any]) -> None:
        """
        Update record of recently used parameters.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Symbol being traded
            params: Parameter values
        """
        if strategy_id not in self.recent_parameters:
            self.recent_parameters[strategy_id] = {}
        
        self.recent_parameters[strategy_id][symbol] = params.copy()
    
    def _get_recent_parameters(self, strategy_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get recently used parameters.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Symbol being traded
            
        Returns:
            Dict of recent parameters or empty dict if none
        """
        if strategy_id in self.recent_parameters and symbol in self.recent_parameters[strategy_id]:
            return self.recent_parameters[strategy_id][symbol].copy()
        
        return {}
    
    def update_optimal_parameters(
        self, strategy_id: str, regime_type: MarketRegimeType, 
        parameters: Dict[str, Any], performance_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update optimal parameters for a strategy in a specific market regime.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Market regime type
            parameters: Parameter values
            performance_metrics: Optional performance metrics for these parameters
        """
        with self._lock:
            try:
                # Initialize if needed
                if strategy_id not in self.optimal_parameters:
                    self.optimal_parameters[strategy_id] = {}
                
                if strategy_id not in self.parameter_performance:
                    self.parameter_performance[strategy_id] = {}
                
                # Update optimal parameters
                self.optimal_parameters[strategy_id][regime_type] = parameters.copy()
                
                # Record performance if provided
                if performance_metrics:
                    if regime_type not in self.parameter_performance[strategy_id]:
                        self.parameter_performance[strategy_id][regime_type] = []
                    
                    # Add performance record
                    performance_record = {
                        'parameters': parameters.copy(),
                        'metrics': performance_metrics.copy(),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.parameter_performance[strategy_id][regime_type].append(performance_record)
                    
                    # Limit history size
                    max_history = self.config.get("max_performance_history", 100)
                    if len(self.parameter_performance[strategy_id][regime_type]) > max_history:
                        self.parameter_performance[strategy_id][regime_type] = self.parameter_performance[strategy_id][regime_type][-max_history:]
                
                # Save to disk
                self._save_parameter_sets()
                
                logger.info(f"Updated optimal parameters for strategy {strategy_id} in regime {regime_type}")
                
            except Exception as e:
                logger.error(f"Error updating optimal parameters: {str(e)}")
    
    def get_parameter_performance(
        self, strategy_id: str, regime_type: Optional[MarketRegimeType] = None
    ) -> Dict[MarketRegimeType, List[Dict[str, Any]]]:
        """
        Get parameter performance history.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Optional specific regime to get performance for
            
        Returns:
            Dict mapping regime types to performance records
        """
        if strategy_id not in self.parameter_performance:
            return {}
        
        if regime_type is not None:
            if regime_type in self.parameter_performance[strategy_id]:
                return {regime_type: self.parameter_performance[strategy_id][regime_type]}
            return {}
        
        return self.parameter_performance[strategy_id]
    
    def clear_transitions(self, strategy_id: Optional[str] = None, 
                         symbol: Optional[str] = None) -> None:
        """
        Clear parameter transitions.
        
        Args:
            strategy_id: Optional strategy to clear transitions for
            symbol: Optional symbol to clear transitions for
        """
        with self._lock:
            if strategy_id is None and symbol is None:
                # Clear all transitions
                self.current_transitions = {}
            else:
                # Clear specific transitions
                keys_to_remove = []
                
                for key in self.current_transitions:
                    parts = key.split('_')
                    
                    if len(parts) >= 3:
                        key_strategy = parts[0]
                        key_symbol = parts[1]
                        
                        if (strategy_id is None or key_strategy == strategy_id) and \
                           (symbol is None or key_symbol == symbol):
                            keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.current_transitions[key]
    
    def optimize_parameters(
        self, strategy_id: str, regime_type: MarketRegimeType,
        parameter_ranges: Dict[str, Tuple[float, float, float]],
        evaluation_func: callable,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize parameters for a strategy in a specific market regime.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Market regime type
            parameter_ranges: Dict mapping parameter names to (min, max, step) tuples
            evaluation_func: Function to evaluate parameter sets (higher is better)
            optimization_config: Optional configuration for optimization
            
        Returns:
            Tuple of (optimal_parameters, score)
        """
        try:
            config = optimization_config or {}
            
            # Optimization settings
            max_iterations = config.get("max_iterations", 100)
            population_size = config.get("population_size", 20)
            mutation_rate = config.get("mutation_rate", 0.1)
            
            # Implementation of simple genetic algorithm for parameter optimization
            
            # Generate initial population
            population = []
            for _ in range(population_size):
                params = {}
                for param_name, (min_val, max_val, step) in parameter_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                        # Integer parameter
                        value = np.random.randint(min_val, max_val + 1)
                        # Ensure step size is respected
                        if step > 1:
                            value = min_val + ((value - min_val) // step) * step
                    else:
                        # Float parameter
                        value = np.random.uniform(min_val, max_val)
                        # Ensure step size is respected
                        if step > 0:
                            value = min_val + round((value - min_val) / step) * step
                    
                    params[param_name] = value
                
                # Evaluate parameter set
                score = evaluation_func(params)
                population.append((params, score))
            
            # Optimization iterations
            for iteration in range(max_iterations):
                # Sort population by score (descending)
                population.sort(key=lambda x: x[1], reverse=True)
                
                # Check for early convergence
                best_score = population[0][1]
                worst_score = population[-1][1]
                
                if best_score == worst_score or iteration == max_iterations - 1:
                    # Return best parameters
                    return population[0][0], population[0][1]
                
                # Create new population
                new_population = []
                
                # Keep top performers
                elite_count = max(1, population_size // 10)
                new_population.extend(population[:elite_count])
                
                # Create offspring
                while len(new_population) < population_size:
                    # Select parents (tournament selection)
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Create child through crossover
                    child_params = {}
                    for param_name in parameter_ranges:
                        # 50% chance of inheriting from each parent
                        if np.random.random() < 0.5:
                            child_params[param_name] = parent1[0][param_name]
                        else:
                            child_params[param_name] = parent2[0][param_name]
                    
                    # Apply mutation
                    for param_name, (min_val, max_val, step) in parameter_ranges.items():
                        if np.random.random() < mutation_rate:
                            if isinstance(min_val, int) and isinstance(max_val, int):
                                # Integer parameter
                                value = np.random.randint(min_val, max_val + 1)
                                # Ensure step size is respected
                                if step > 1:
                                    value = min_val + ((value - min_val) // step) * step
                            else:
                                # Float parameter
                                value = np.random.uniform(min_val, max_val)
                                # Ensure step size is respected
                                if step > 0:
                                    value = min_val + round((value - min_val) / step) * step
                            
                            child_params[param_name] = value
                    
                    # Evaluate child
                    child_score = evaluation_func(child_params)
                    new_population.append((child_params, child_score))
                
                # Replace population
                population = new_population
            
            # Return best parameters
            population.sort(key=lambda x: x[1], reverse=True)
            return population[0][0], population[0][1]
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            
            # Return default parameters
            default_params = {
                param_name: (min_val + max_val) / 2
                for param_name, (min_val, max_val, _) in parameter_ranges.items()
            }
            return default_params, 0.0
    
    def _tournament_selection(self, population, tournament_size=3):
        """
        Tournament selection for genetic algorithm.
        
        Args:
            population: List of (params, score) tuples
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual (params, score)
        """
        # Randomly select tournament_size individuals
        tournament = [population[np.random.randint(0, len(population))] for _ in range(tournament_size)]
        
        # Return the best individual in the tournament
        return max(tournament, key=lambda x: x[1])
    
    def learn_from_performance(self, strategy_id: str, regime_type: MarketRegimeType) -> bool:
        """
        Learn optimal parameters from historical performance.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Market regime type
            
        Returns:
            bool: Success status
        """
        try:
            if strategy_id not in self.parameter_performance or \
               regime_type not in self.parameter_performance[strategy_id]:
                logger.warning(f"No performance data for strategy {strategy_id} in regime {regime_type}")
                return False
            
            # Get performance records
            performance_records = self.parameter_performance[strategy_id][regime_type]
            
            if not performance_records:
                return False
            
            # Sort by performance (assuming higher is better)
            # We'll use the 'profit_factor' metric if available, otherwise the first metric
            def get_performance_score(record):
                metrics = record.get('metrics', {})
                if 'profit_factor' in metrics:
                    return metrics['profit_factor']
                elif 'sharpe_ratio' in metrics:
                    return metrics['sharpe_ratio']
                elif 'net_profit' in metrics:
                    return metrics['net_profit']
                elif 'win_rate' in metrics:
                    return metrics['win_rate']
                elif len(metrics) > 0:
                    return metrics[list(metrics.keys())[0]]
                return 0.0
            
            sorted_records = sorted(performance_records, key=get_performance_score, reverse=True)
            
            # Get top performing parameters
            if sorted_records:
                best_params = sorted_records[0]['parameters']
                
                # Update optimal parameters
                self.update_optimal_parameters(strategy_id, regime_type, best_params)
                
                logger.info(f"Learned optimal parameters for strategy {strategy_id} in regime {regime_type}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error learning from performance: {str(e)}")
            return False
    
    def get_parameter_transition_status(self, strategy_id: str, symbol: str, 
                                     timeframe: str) -> Dict[str, Any]:
        """
        Get current parameter transition status.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Symbol being traded
            timeframe: Timeframe being traded
            
        Returns:
            Dict with transition status
        """
        transition_key = f"{strategy_id}_{symbol}_{timeframe}"
        return self.current_transitions.get(transition_key, {})
    
    def get_all_regime_parameters(self, strategy_id: str) -> Dict[MarketRegimeType, Dict[str, Any]]:
        """
        Get all parameter sets for all regimes for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict mapping regime types to parameter sets
        """
        return self.optimal_parameters.get(strategy_id, {})
