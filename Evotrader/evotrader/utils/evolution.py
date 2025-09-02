"""Evolutionary algorithm utilities for EvoTrader."""

import random
import copy
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Type

from ..core.strategy import Strategy
from ..core.challenge_bot import ChallengeBot


def apply_mutation(strategy: Strategy, mutation_rate: float = 0.1) -> Strategy:
    """
    Apply parameter mutations to a strategy based on mutation rate.
    
    Args:
        strategy: Strategy instance to mutate
        mutation_rate: Probability of each parameter mutating
        
    Returns:
        Mutated strategy (same instance, modified in-place)
    """
    # Get parameter definitions to understand constraints
    param_defs = {p.name: p for p in strategy.get_parameters()}
    
    # Apply mutations to parameters
    for name, value in strategy.parameters.items():
        # Skip parameters that aren't mutable
        if name in param_defs and not param_defs[name].is_mutable:
            continue
            
        # Apply mutation based on parameter type
        if isinstance(value, (int, float)):
            # Get mutation constraints
            if name in param_defs:
                min_val = param_defs[name].min_value
                max_val = param_defs[name].max_value
                mutation_factor = param_defs[name].mutation_factor
            else:
                # Default constraints
                min_val = value * 0.5 if value > 0 else value * 2.0
                max_val = value * 2.0 if value > 0 else value * 0.5
                mutation_factor = 0.2
            
            # Apply mutation
            if random.random() < mutation_rate:
                # Determine the mutation size
                max_change = (max_val - min_val) * mutation_factor
                change = random.uniform(-max_change, max_change)
                
                # Apply the change and clamp to valid range
                new_val = value + change
                if min_val is not None:
                    new_val = max(min_val, new_val)
                if max_val is not None:
                    new_val = min(max_val, new_val)
                    
                # Round to int if the original was int
                if isinstance(value, int):
                    new_val = int(round(new_val))
                    
                strategy.parameters[name] = new_val
        
        elif isinstance(value, bool) and random.random() < mutation_rate:
            # Flip boolean values
            strategy.parameters[name] = not value
            
        elif isinstance(value, str) and random.random() < mutation_rate:
            # Handle string parameters (e.g., selecting from options)
            # This is a stub - would need to know valid options for each string parameter
            pass
            
    return strategy


def crossover_strategies(parent_a: Strategy, parent_b: Strategy, 
                         crossover_rate: float = 0.3) -> Strategy:
    """
    Create a hybrid strategy by combining parameters from two parent strategies.
    
    Args:
        parent_a: First parent strategy (primary parent)
        parent_b: Second parent strategy
        crossover_rate: Probability of inheriting from parent_b for each parameter
        
    Returns:
        New strategy instance with combined parameters
    """
    # Create a new strategy of the same type as parent_a
    strategy_class = parent_a.__class__
    strategy_id = f"{strategy_class.__name__}_{str(uuid.uuid4())[:8]}"
    child_strategy = strategy_class(strategy_id=strategy_id)
    
    # Get parameters from both parents
    params_a = parent_a.parameters
    params_b = parent_b.parameters
    
    # Crossover parameters
    combined_params = {}
    
    # Get parameter definitions to validate values
    param_defs = {p.name: p for p in parent_a.get_parameters()}
    
    # Perform parameter crossover
    for param_name in params_a.keys():
        if param_name in params_b:
            # For each parameter, decide which parent to inherit from
            if random.random() < crossover_rate:
                combined_params[param_name] = params_b[param_name]
            else:
                combined_params[param_name] = params_a[param_name]
        else:
            # Parameter only in parent_a
            combined_params[param_name] = params_a[param_name]
    
    # Set the combined parameters
    child_strategy.parameters = combined_params
    
    return child_strategy


def create_hybrid_bot(parent_a: ChallengeBot, parent_b: ChallengeBot, 
                      bot_id: Optional[str] = None, 
                      crossover_rate: float = 0.3,
                      mutation_rate: float = 0.1) -> ChallengeBot:
    """
    Create a hybrid bot by combining strategies from two parent bots.
    
    Args:
        parent_a: First parent bot (primary parent)
        parent_b: Second parent bot
        bot_id: Optional ID for the new bot
        crossover_rate: Probability of inheriting from parent_b for each parameter
        mutation_rate: Probability of each parameter mutating
        
    Returns:
        New bot with hybrid strategy
    """
    if bot_id is None:
        bot_id = f"hybrid_{str(uuid.uuid4())[:8]}"
    
    # Strategy crossover
    hybrid_strategy = crossover_strategies(
        parent_a.strategy, 
        parent_b.strategy, 
        crossover_rate
    )
    
    # Apply mutations to prevent local minima
    apply_mutation(hybrid_strategy, mutation_rate)
    
    # Create new bot with the hybrid strategy
    new_bot = ChallengeBot(bot_id, hybrid_strategy, parent_a.initial_balance)
    
    # Set up evolution tracking
    new_bot.generation = max(parent_a.generation, parent_b.generation) + 1
    new_bot.parent_ids = [parent_a.bot_id, parent_b.bot_id]
    
    return new_bot


def calculate_population_diversity(strategies: List[Strategy]) -> float:
    """
    Calculate the diversity of a population of strategies based on parameter variance.
    
    Args:
        strategies: List of strategy instances
        
    Returns:
        Diversity score from 0 (identical) to 1 (maximum diversity)
    """
    if not strategies or len(strategies) < 2:
        return 0.0
    
    # Group strategies by type
    strategy_groups = {}
    for s in strategies:
        class_name = s.__class__.__name__
        if class_name not in strategy_groups:
            strategy_groups[class_name] = []
        strategy_groups[class_name].append(s)
    
    # Calculate diversity across groups and within groups
    strategy_type_diversity = len(strategy_groups) / len(strategies)
    
    # Calculate parameter diversity within each group
    param_diversities = []
    
    for group_name, group_strategies in strategy_groups.items():
        if len(group_strategies) < 2:
            continue
            
        # Get all parameter names across all strategies in the group
        all_params = set()
        for s in group_strategies:
            all_params.update(s.parameters.keys())
            
        # Calculate variance for each parameter
        for param_name in all_params:
            values = []
            for s in group_strategies:
                if param_name in s.parameters:
                    param_value = s.parameters[param_name]
                    if isinstance(param_value, (int, float)):
                        values.append(param_value)
            
            if values:
                # Calculate normalized variance
                mean_value = sum(values) / len(values)
                if mean_value != 0:
                    variance = sum((v - mean_value)**2 for v in values) / len(values)
                    normalized_variance = min(variance / abs(mean_value), 1.0)
                    param_diversities.append(normalized_variance)
    
    # Combine strategy type diversity with parameter diversity
    if param_diversities:
        avg_param_diversity = sum(param_diversities) / len(param_diversities)
        overall_diversity = 0.7 * strategy_type_diversity + 0.3 * avg_param_diversity
    else:
        overall_diversity = strategy_type_diversity
        
    return overall_diversity


def adaptive_crossover_rate(population_diversity: float, 
                           min_rate: float = 0.1, 
                           max_rate: float = 0.7) -> float:
    """
    Calculate adaptive crossover rate based on population diversity.
    
    Uses higher crossover rate when diversity is low to encourage exploration.
    
    Args:
        population_diversity: Diversity measure from 0-1
        min_rate: Minimum crossover rate
        max_rate: Maximum crossover rate
        
    Returns:
        Adjusted crossover rate
    """
    # Inverse relationship: as diversity decreases, crossover rate increases
    return max_rate - (population_diversity * (max_rate - min_rate))
