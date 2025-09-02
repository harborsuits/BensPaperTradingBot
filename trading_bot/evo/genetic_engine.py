#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoTrader Genetic Engine

This module implements the genetic algorithm engine for EvoTrader,
handling strategy mutation, crossover, and evolution.
"""

import os
import sys
import random
import logging
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from copy import deepcopy

from trading_bot.evo.architecture import (
    StrategyCandidate, StrategyStage, FitnessMetric,
    create_strategy_candidate
)

logger = logging.getLogger(__name__)


class GeneticEngine:
    """
    Genetic algorithm engine for evolving trading strategies
    
    This engine handles the creation of new strategy variants through 
    mutation and crossover operations, as well as the selection of 
    strategies for evolution.
    """
    
    def __init__(self, core, config: Dict[str, Any] = None):
        """
        Initialize the genetic engine
        
        Args:
            core: Reference to the EvoTrader core
            config: Configuration dictionary
        """
        self.core = core
        self.config = config or {}
        
        if not self.config and hasattr(core, 'config') and 'genetic' in core.config:
            self.config = core.config.get('genetic', {})
            
        self._set_default_config()
        
        # Track evolutionary statistics
        self.stats = {
            "generations": 0,
            "mutations": 0,
            "crossovers": 0,
            "created_candidates": 0,
            "fitness_improvements": 0
        }
        
        logger.info("Genetic engine initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Population parameters
        self.config.setdefault("population_size", 100)
        self.config.setdefault("tournament_size", 20)
        self.config.setdefault("elite_count", 5)
        
        # Genetic operators
        self.config.setdefault("mutation_rate", 0.05)
        self.config.setdefault("crossover_rate", 0.7)
        self.config.setdefault("generations", 50)
        
        # Mutation parameters
        self.config.setdefault("mutation_params", {
            "small_change_prob": 0.7,  # Probability of small parameter changes
            "medium_change_prob": 0.2,  # Probability of medium parameter changes
            "large_change_prob": 0.1,   # Probability of large parameter changes
            "add_param_prob": 0.05,     # Probability of adding a new parameter
            "remove_param_prob": 0.05   # Probability of removing a parameter
        })
        
        # Strategy types for generation
        self.config.setdefault("strategy_types", [
            "momentum",
            "mean_reversion",
            "trend_following",
            "breakout",
            "volatility",
            "ml_enhanced",
            "options_income",
            "options_spread"
        ])
        
        # Common parameter ranges
        self.config.setdefault("parameter_ranges", {
            "timeframe": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            "rsi_period": {"min": 2, "max": 30, "step": 1},
            "ema_fast": {"min": 3, "max": 50, "step": 1},
            "ema_slow": {"min": 10, "max": 200, "step": 1},
            "atr_period": {"min": 5, "max": 30, "step": 1},
            "stop_loss_atr": {"min": 1.0, "max": 5.0, "step": 0.1},
            "take_profit_atr": {"min": 1.0, "max": 10.0, "step": 0.1},
            "position_size_pct": {"min": 0.01, "max": 0.2, "step": 0.01}
        })
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Run a full genetic cycle
        
        This method creates a new generation of strategies through
        selection, crossover, and mutation.
        
        Returns:
            Results dictionary
        """
        logger.info("Starting genetic cycle")
        
        # Get current population
        current_population = self._get_current_population()
        
        # Check if we have enough candidates
        if len(current_population) < 2:
            # Generate initial population if needed
            logger.info("Not enough candidates, generating initial population")
            self._generate_initial_population()
            current_population = self._get_current_population()
        
        # Evaluate fitness (if not already evaluated)
        self._ensure_fitness_evaluated(current_population)
        
        # Create new generation
        new_candidates = self._create_new_generation(current_population)
        
        # Register new candidates
        for candidate in new_candidates:
            self.core.register_candidate(candidate)
            
            # Promote to tournament directly
            self.core.promote_to_tournament(candidate.id)
        
        # Update stats
        self.stats["generations"] += 1
        self.stats["created_candidates"] += len(new_candidates)
        
        logger.info(f"Genetic cycle completed. {len(new_candidates)} new candidates created")
        
        return {
            "generation": self.stats["generations"],
            "created_candidates": len(new_candidates),
            "new_candidate_ids": [c.id for c in new_candidates],
            "total_candidates": self.stats["created_candidates"],
            "stats": self.stats
        }
    
    def _get_current_population(self) -> List[StrategyCandidate]:
        """
        Get the current population of candidates
        
        Returns:
            List of candidates
        """
        # Combine candidates and tournament pool
        population = list(self.core.candidates.values())
        population.extend(list(self.core.tournament_pool.values()))
        
        # Filter out candidates without fitness scores
        population = [c for c in population if c.fitness_scores]
        
        return population
    
    def _ensure_fitness_evaluated(self, population: List[StrategyCandidate]):
        """
        Ensure all candidates have fitness scores
        
        Args:
            population: List of candidates to evaluate
        """
        # In a real implementation, this would trigger backtests
        # for any candidates that don't have fitness scores.
        # For now, we'll just log a warning.
        
        unevaluated = [c for c in population if not c.fitness_scores]
        if unevaluated:
            logger.warning(f"{len(unevaluated)} candidates have no fitness scores")
    
    def _generate_initial_population(self) -> List[StrategyCandidate]:
        """
        Generate an initial population of strategy candidates
        
        Returns:
            List of generated candidates
        """
        logger.info(f"Generating initial population of {self.config['population_size']} candidates")
        
        # Create candidates
        candidates = []
        
        for i in range(self.config['population_size']):
            # Choose a random strategy type
            strategy_type = random.choice(self.config['strategy_types'])
            
            # Generate parameters for this strategy type
            parameters = self._generate_parameters(strategy_type)
            
            # Create a new candidate
            candidate = create_strategy_candidate(
                name=f"{strategy_type.capitalize()}_Strategy_{i+1}",
                strategy_type=strategy_type,
                parameters=parameters
            )
            
            # Register with core
            self.core.register_candidate(candidate)
            
            candidates.append(candidate)
        
        logger.info(f"Generated {len(candidates)} initial candidates")
        
        return candidates
    
    def _generate_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """
        Generate parameters for a specific strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Dictionary of parameters
        """
        parameters = {}
        
        # Common parameters for all strategy types
        parameters["timeframe"] = random.choice(self.config["parameter_ranges"]["timeframe"])
        parameters["position_size_pct"] = self._random_from_range(self.config["parameter_ranges"]["position_size_pct"])
        
        # Add type-specific parameters
        if strategy_type == "momentum":
            parameters.update({
                "rsi_period": self._random_from_range(self.config["parameter_ranges"]["rsi_period"]),
                "rsi_overbought": random.randint(65, 85),
                "rsi_oversold": random.randint(15, 35),
                "momentum_period": random.randint(3, 20),
                "confirmation_candles": random.randint(1, 3),
                "stop_loss_atr": self._random_from_range(self.config["parameter_ranges"]["stop_loss_atr"]),
                "take_profit_atr": self._random_from_range(self.config["parameter_ranges"]["take_profit_atr"])
            })
            
        elif strategy_type == "mean_reversion":
            parameters.update({
                "bollinger_period": random.randint(10, 30),
                "bollinger_std": random.uniform(1.5, 3.0),
                "rsi_period": self._random_from_range(self.config["parameter_ranges"]["rsi_period"]),
                "rsi_threshold": random.randint(20, 35),
                "max_position_duration": random.randint(2, 10),
                "stop_loss_atr": self._random_from_range(self.config["parameter_ranges"]["stop_loss_atr"])
            })
            
        elif strategy_type == "trend_following":
            parameters.update({
                "ema_fast": self._random_from_range(self.config["parameter_ranges"]["ema_fast"]),
                "ema_slow": self._random_from_range(self.config["parameter_ranges"]["ema_slow"]),
                "atr_period": self._random_from_range(self.config["parameter_ranges"]["atr_period"]),
                "stop_loss_atr": self._random_from_range(self.config["parameter_ranges"]["stop_loss_atr"]),
                "trailing_stop_atr": random.uniform(1.0, 3.0),
                "profit_protect_threshold": random.uniform(1.0, 3.0)
            })
            
        elif strategy_type == "breakout":
            parameters.update({
                "breakout_periods": random.randint(10, 50),
                "volume_confirm_factor": random.uniform(1.2, 2.5),
                "stop_loss_pct": random.uniform(0.5, 3.0),
                "take_profit_pct": random.uniform(1.0, 5.0),
                "max_holding_periods": random.randint(3, 20)
            })
            
        elif strategy_type == "volatility":
            parameters.update({
                "atr_period": self._random_from_range(self.config["parameter_ranges"]["atr_period"]),
                "volatility_entry_threshold": random.uniform(0.5, 2.0),
                "position_scaling": random.uniform(0.5, 1.5),
                "stop_loss_atr": self._random_from_range(self.config["parameter_ranges"]["stop_loss_atr"]),
                "target_atr": random.uniform(1.0, 3.0)
            })
            
        elif strategy_type == "ml_enhanced":
            parameters.update({
                "min_confidence": random.uniform(0.55, 0.75),
                "signal_expiry_bars": random.randint(1, 5),
                "max_position_size": random.uniform(0.05, 0.25),
                "max_portfolio_allocation": random.uniform(0.5, 0.9),
                "regime_adjustment_factor": random.uniform(0.7, 1.3)
            })
            
        elif strategy_type == "options_income":
            parameters.update({
                "delta_target": random.uniform(0.2, 0.4),
                "days_to_expiration": random.randint(20, 60),
                "profit_target_pct": random.uniform(25, 75),
                "max_loss_pct": random.uniform(50, 100),
                "position_size_pct": random.uniform(0.01, 0.05),
                "iv_rank_min": random.uniform(30, 50)
            })
            
        elif strategy_type == "options_spread":
            parameters.update({
                "spread_width": random.randint(2, 10),
                "days_to_expiration": random.randint(20, 60),
                "delta_short": random.uniform(0.2, 0.4),
                "profit_target_pct": random.uniform(25, 75),
                "max_loss_pct": random.uniform(50, 100),
                "position_size_pct": random.uniform(0.01, 0.05)
            })
            
        return parameters
    
    def _random_from_range(self, range_dict: Dict[str, Any]) -> Union[int, float]:
        """
        Generate a random value from a range
        
        Args:
            range_dict: Range dictionary with min, max, step
            
        Returns:
            Random value from range
        """
        if "min" not in range_dict or "max" not in range_dict:
            return 0
            
        min_val = range_dict["min"]
        max_val = range_dict["max"]
        step = range_dict.get("step", 1)
        
        if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
            # Integer range
            steps = int((max_val - min_val) / step) + 1
            return min_val + random.randint(0, steps - 1) * step
        else:
            # Float range
            steps = int((max_val - min_val) / step) + 1
            return min_val + random.randint(0, steps - 1) * step
    
    def _create_new_generation(self, population: List[StrategyCandidate]) -> List[StrategyCandidate]:
        """
        Create a new generation of strategies
        
        Args:
            population: Current population
            
        Returns:
            List of new candidates
        """
        # Sort by fitness
        sorted_population = sorted(
            population, 
            key=lambda x: x.calculate_composite_fitness(), 
            reverse=True
        )
        
        # Keep elite candidates
        elite_count = min(self.config["elite_count"], len(sorted_population))
        elite_candidates = sorted_population[:elite_count]
        
        # Create new generation through crossover and mutation
        new_candidates = []
        
        # Number of candidates to create
        num_to_create = self.config["population_size"] - elite_count
        
        for i in range(num_to_create):
            if random.random() < self.config["crossover_rate"] and len(sorted_population) > 1:
                # Crossover
                parent1 = self._tournament_selection(sorted_population)
                parent2 = self._tournament_selection(sorted_population)
                
                child = self._crossover(parent1, parent2)
                new_candidates.append(child)
                self.stats["crossovers"] += 1
            else:
                # Mutation
                parent = self._tournament_selection(sorted_population)
                child = self._mutate(parent)
                new_candidates.append(child)
                self.stats["mutations"] += 1
        
        return new_candidates
    
    def _tournament_selection(self, population: List[StrategyCandidate]) -> StrategyCandidate:
        """
        Select a candidate using tournament selection
        
        Args:
            population: Population to select from
            
        Returns:
            Selected candidate
        """
        # Tournament size
        tournament_size = min(self.config["tournament_size"], len(population))
        
        # Select random participants
        participants = random.sample(population, tournament_size)
        
        # Return the best
        return max(participants, key=lambda x: x.calculate_composite_fitness())
    
    def _crossover(self, parent1: StrategyCandidate, parent2: StrategyCandidate) -> StrategyCandidate:
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child candidate
        """
        # Determine strategy type (can be from either parent)
        if random.random() < 0.5:
            strategy_type = parent1.strategy_type
        else:
            strategy_type = parent2.strategy_type
        
        # Create parameter set from both parents
        parameters = {}
        
        # Get all parameter keys from both parents
        all_keys = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        for key in all_keys:
            # Choose which parent to inherit from for this parameter
            if key in parent1.parameters and key in parent2.parameters:
                # Both parents have this parameter
                if random.random() < 0.5:
                    parameters[key] = parent1.parameters[key]
                else:
                    parameters[key] = parent2.parameters[key]
            elif key in parent1.parameters:
                # Only parent1 has this parameter
                if random.random() < 0.8:  # 80% chance to inherit
                    parameters[key] = parent1.parameters[key]
            elif key in parent2.parameters:
                # Only parent2 has this parameter
                if random.random() < 0.8:  # 80% chance to inherit
                    parameters[key] = parent2.parameters[key]
        
        # Create child candidate
        child = create_strategy_candidate(
            name=f"Hybrid_{strategy_type.capitalize()}_{parent1.id[-4:]}_{parent2.id[-4:]}",
            strategy_type=strategy_type,
            parameters=parameters,
            parent_ids=[parent1.id, parent2.id]
        )
        
        # Inherit generation number
        child.generation = max(parent1.generation, parent2.generation) + 1
        
        # Record crossover points for reference
        child.crossover_points = list(parameters.keys())
        
        return child
    
    def _mutate(self, parent: StrategyCandidate) -> StrategyCandidate:
        """
        Perform mutation on a parent
        
        Args:
            parent: Parent candidate
            
        Returns:
            Mutated child candidate
        """
        # Clone parameters
        parameters = deepcopy(parent.parameters)
        
        # Choose parameters to mutate
        mutation_count = max(1, int(len(parameters) * self.config["mutation_rate"]))
        params_to_mutate = random.sample(list(parameters.keys()), min(mutation_count, len(parameters)))
        
        for param in params_to_mutate:
            # Get current value
            current_value = parameters[param]
            
            # Get parameter range if available
            param_range = self.config["parameter_ranges"].get(param, None)
            
            if param_range:
                # Parameter has a defined range
                parameters[param] = self._random_from_range(param_range)
            elif isinstance(current_value, (int, float)):
                # Numeric parameter without range
                # Determine mutation magnitude
                mutation_probs = self.config["mutation_params"]
                
                if random.random() < mutation_probs["small_change_prob"]:
                    # Small change (±10%)
                    change_factor = random.uniform(0.9, 1.1)
                elif random.random() < mutation_probs["medium_change_prob"] / (1 - mutation_probs["small_change_prob"]):
                    # Medium change (±25%)
                    change_factor = random.uniform(0.75, 1.25)
                else:
                    # Large change (±50%)
                    change_factor = random.uniform(0.5, 1.5)
                
                # Apply mutation
                if isinstance(current_value, int):
                    new_value = int(current_value * change_factor)
                    # Ensure at least ±1 change
                    if new_value == current_value:
                        new_value += random.choice([-1, 1])
                    parameters[param] = new_value
                else:
                    parameters[param] = current_value * change_factor
                    
            elif isinstance(current_value, str) and param == "timeframe":
                # Timeframe parameter
                parameters[param] = random.choice(self.config["parameter_ranges"]["timeframe"])
            elif isinstance(current_value, bool):
                # Boolean parameter - flip it
                parameters[param] = not current_value
                
        # Add/remove parameters with low probability
        if random.random() < self.config["mutation_params"]["add_param_prob"]:
            # Add a new parameter relevant to the strategy type
            new_params = self._generate_parameters(parent.strategy_type)
            
            # Find a parameter not already in the set
            for key, value in new_params.items():
                if key not in parameters:
                    parameters[key] = value
                    break
        
        if random.random() < self.config["mutation_params"]["remove_param_prob"] and len(parameters) > 3:
            # Remove a parameter (ensure we keep at least 3)
            param_to_remove = random.choice(list(parameters.keys()))
            del parameters[param_to_remove]
        
        # Create child candidate
        child = create_strategy_candidate(
            name=f"Mutated_{parent.strategy_type.capitalize()}_{parent.id[-4:]}",
            strategy_type=parent.strategy_type,
            parameters=parameters,
            parent_ids=[parent.id]
        )
        
        # Inherit generation number
        child.generation = parent.generation + 1
        
        # Inherit and adjust mutation rate
        child.mutation_rate = parent.mutation_rate
        
        # Occasionally adjust mutation rate
        if random.random() < 0.1:
            # Adjust mutation rate ±20%
            child.mutation_rate *= random.uniform(0.8, 1.2)
            # Ensure it stays within reasonable bounds
            child.mutation_rate = max(0.01, min(0.2, child.mutation_rate))
        
        return child


# Factory function
def create_genetic_engine(core, config: Dict[str, Any] = None) -> GeneticEngine:
    """
    Create a genetic engine instance
    
    Args:
        core: Reference to EvoTrader core
        config: Optional configuration dictionary
        
    Returns:
        Initialized genetic engine
    """
    return GeneticEngine(core, config)
