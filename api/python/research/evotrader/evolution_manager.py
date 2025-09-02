"""
Evolution Manager for BensBot-EvoTrader Integration

This module implements the genetic algorithm based strategy evolution system,
handling population management, selection, crossover, mutation, and evaluation.
"""

import os
import json
import random
import logging
import datetime
import copy
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger("benbot.research.evotrader.evolution")

class EvolutionManager:
    """
    Manages the evolutionary process for trading strategies.
    
    This class handles:
    - Creation of initial strategy populations
    - Selection of parent strategies
    - Crossover (breeding) of strategies
    - Mutation of strategies
    - Evaluation and fitness calculation
    - Generation management and tracking
    """
    
    def __init__(self,
                asset_class: str = "forex",
                symbols: List[str] = None,
                timeframes: List[str] = None,
                config: Dict[str, Any] = None):
        """
        Initialize the evolution manager.
        
        Args:
            asset_class: "forex" or "crypto"
            symbols: List of symbols to focus on
            timeframes: List of timeframes to use
            config: Evolution configuration
        """
        self.asset_class = asset_class
        self.symbols = symbols or self._get_default_symbols(asset_class)
        self.timeframes = timeframes or self._get_default_timeframes(asset_class)
        
        # Set default config
        default_config = {
            "population_size": 50,
            "generations": 10,
            "selection_rate": 0.3,
            "mutation_rate": 0.2,
            "crossover_rate": 0.3,
            "elitism": 2,
            "tournament_size": 3,
            "fitness_metrics": ["sharpe_ratio", "profit_factor", "win_rate"],
            "fitness_weights": [0.5, 0.3, 0.2]
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
            
        # Create output directory
        self.output_dir = self.config.get("output_dir", "evolution_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize evaluator (lazy-loaded when needed)
        self.evaluator = None
        
        logger.info(f"Evolution manager initialized for {asset_class}")
    
    def _get_default_symbols(self, asset_class: str) -> List[str]:
        """Get default symbols for the asset class."""
        if asset_class == "forex":
            return ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        elif asset_class == "crypto":
            return ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
        else:
            return ["EURUSD", "BTC/USD"]  # Default to one of each
    
    def _get_default_timeframes(self, asset_class: str) -> List[str]:
        """Get default timeframes for the asset class."""
        if asset_class == "forex":
            return ["1h", "4h", "1d"]
        elif asset_class == "crypto":
            return ["15m", "1h", "4h", "1d"]
        else:
            return ["1h", "4h"]  # Default
    
    def create_initial_population(self, template_strategy=None):
        """
        Create an initial population of strategies.
        
        Args:
            template_strategy: Optional template to base population on
            
        Returns:
            List of strategies
        """
        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
        from trading_bot.research.evotrader.strategy_templates import create_strategy_template
        
        population_size = self.config.get("population_size", 50)
        population = []
        
        # If no template provided, create one for the asset class
        if not template_strategy:
            template_strategy = create_strategy_template(self.asset_class)
        
        # Extract parameter ranges from template
        if isinstance(template_strategy, BensBotStrategyAdapter):
            param_metadata = template_strategy.param_metadata
            base_params = template_strategy.parameters
        else:
            # If it's a raw BensBot strategy, adapt it first
            adapter = BensBotStrategyAdapter(benbot_strategy=template_strategy)
            param_metadata = adapter.param_metadata
            base_params = adapter.parameters
        
        # Generate the population
        for i in range(population_size):
            # For each strategy, randomize parameters
            new_params = {}
            
            for param_name, param_meta in param_metadata.items():
                if not param_meta.is_mutable:
                    # Use original value for immutable parameters
                    new_params[param_name] = base_params.get(param_name, param_meta.default_value)
                    continue
                
                # Randomize based on parameter type
                if param_meta.param_type == int:
                    if param_meta.min_value is not None and param_meta.max_value is not None:
                        new_params[param_name] = random.randint(
                            param_meta.min_value, 
                            param_meta.max_value
                        )
                    else:
                        # No range specified, use base value +/- 50%
                        base_value = base_params.get(param_name, param_meta.default_value)
                        min_val = max(1, int(base_value * 0.5))
                        max_val = int(base_value * 1.5)
                        new_params[param_name] = random.randint(min_val, max_val)
                        
                elif param_meta.param_type == float:
                    if param_meta.min_value is not None and param_meta.max_value is not None:
                        new_params[param_name] = random.uniform(
                            param_meta.min_value, 
                            param_meta.max_value
                        )
                    else:
                        # No range specified, use base value +/- 50%
                        base_value = base_params.get(param_name, param_meta.default_value)
                        min_val = max(0.001, base_value * 0.5)
                        max_val = base_value * 1.5
                        new_params[param_name] = random.uniform(min_val, max_val)
                        
                elif param_meta.param_type == bool:
                    # Randomly True or False
                    new_params[param_name] = random.choice([True, False])
                    
                elif param_meta.param_type == str:
                    # Strings are not typically randomized, use original
                    new_params[param_name] = base_params.get(param_name, param_meta.default_value)
            
            # Create new strategy with randomized parameters
            strategy_id = f"gen0_strat{i}_{str(uuid.uuid4())[:8]}"
            
            if isinstance(template_strategy, BensBotStrategyAdapter):
                # Create a new adapter with same benbot_strategy but different parameters
                new_strategy = BensBotStrategyAdapter(
                    benbot_strategy=template_strategy.benbot_strategy,
                    strategy_id=strategy_id,
                    parameters=new_params,
                    metadata={"generation": 0, "parent_ids": []}
                )
            else:
                # Create a new adapter from scratch
                new_strategy = BensBotStrategyAdapter(
                    benbot_strategy=template_strategy,
                    strategy_id=strategy_id,
                    parameters=new_params,
                    metadata={"generation": 0, "parent_ids": []}
                )
            
            population.append(new_strategy)
        
        logger.info(f"Created initial population of {len(population)} strategies")
        return population
    
    def run_evolution(self, initial_population):
        """
        Run the full evolutionary process.
        
        Args:
            initial_population: Starting population of strategies
            
        Returns:
            Dictionary with evolution results
        """
        # Initialize results tracking
        results = {
            "config": self.config,
            "asset_class": self.asset_class,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "start_time": datetime.datetime.now().isoformat(),
            "generations": []
        }
        
        # Get config values
        num_generations = self.config.get("generations", 10)
        population_size = self.config.get("population_size", 50)
        
        # Start with initial population
        population = initial_population
        
        # For each generation
        for gen_idx in range(num_generations):
            logger.info(f"Starting generation {gen_idx + 1}/{num_generations}")
            
            # Evaluate current population
            logger.info(f"Evaluating population of {len(population)} strategies")
            evaluation_results = self._evaluate_population(population)
            
            # Track best strategy of this generation
            best_strategy_idx = max(range(len(evaluation_results)), 
                                   key=lambda i: evaluation_results[i]["fitness"])
            
            best_strategy = population[best_strategy_idx]
            best_fitness = evaluation_results[best_strategy_idx]["fitness"]
            
            logger.info(f"Generation {gen_idx + 1} best fitness: {best_fitness:.4f} "
                       f"(Strategy {best_strategy.strategy_id})")
            
            # Record generation results
            generation_results = {
                "generation_number": gen_idx + 1,
                "best_strategy_id": best_strategy.strategy_id,
                "best_fitness": best_fitness,
                "best_strategy_params": best_strategy.parameters,
                "best_strategy_metrics": evaluation_results[best_strategy_idx],
                "strategies": [
                    {
                        "strategy_id": population[i].strategy_id,
                        "fitness": evaluation_results[i]["fitness"],
                        "metrics": evaluation_results[i],
                        "parameters": population[i].parameters
                    }
                    for i in range(len(population))
                ]
            }
            
            results["generations"].append(generation_results)
            
            # If this is the last generation, stop here
            if gen_idx == num_generations - 1:
                break
                
            # Create next generation
            new_population = self._create_next_generation(population, evaluation_results)
            population = new_population
        
        # Record end time
        results["end_time"] = datetime.datetime.now().isoformat()
        
        # Save results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"evolution_{self.asset_class}_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evolution completed with {num_generations} generations")
        logger.info(f"Results saved to {results_file}")
        
        return results
    
    def _evaluate_population(self, population):
        """
        Evaluate all strategies in the population.
        
        Args:
            population: List of strategies to evaluate
            
        Returns:
            List of evaluation results with fitness scores
        """
        # Lazy-load evaluator if not already loaded
        if self.evaluator is None:
            from trading_bot.research.evotrader.strategy_evaluator import StrategyEvaluator
            self.evaluator = StrategyEvaluator(
                symbols=self.symbols,
                timeframes=self.timeframes,
                asset_class=self.asset_class
            )
        
        evaluation_results = []
        
        # Evaluate each strategy
        for strategy in population:
            try:
                # Run evaluation
                metrics = self.evaluator.evaluate(strategy)
                
                # Calculate fitness based on configured metrics and weights
                fitness_metrics = self.config.get("fitness_metrics", ["sharpe_ratio", "profit_factor", "win_rate"])
                fitness_weights = self.config.get("fitness_weights", [0.5, 0.3, 0.2])
                
                # Calculate weighted fitness
                fitness = 0
                for metric_name, weight in zip(fitness_metrics, fitness_weights):
                    # Normalize metrics to 0-1 range when needed
                    if metric_name == "win_rate" or metric_name == "win_rate_pct":
                        # Convert percentage to 0-1
                        value = metrics.get(metric_name, 0) / 100 if metrics.get(metric_name, 0) > 1 else metrics.get(metric_name, 0)
                    elif metric_name == "max_drawdown" or metric_name == "max_drawdown_pct":
                        # Smaller drawdown is better, invert and normalize
                        max_drawdown = metrics.get(metric_name, 0)
                        if max_drawdown > 1:  # Percentage
                            max_drawdown /= 100
                        value = 1 - min(max_drawdown, 1)
                    else:
                        # Use raw value for other metrics
                        value = metrics.get(metric_name, 0)
                    
                    # Apply weight
                    fitness += value * weight
                
                # Add fitness to metrics
                metrics["fitness"] = fitness
                evaluation_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating strategy {strategy.strategy_id}: {e}")
                # Assign minimal fitness
                evaluation_results.append({"fitness": 0, "error": str(e)})
        
        return evaluation_results
    
    def _create_next_generation(self, current_population, evaluation_results):
        """
        Create the next generation through selection, crossover, and mutation.
        
        Args:
            current_population: Current generation strategies
            evaluation_results: Evaluation metrics with fitness scores
            
        Returns:
            New population of strategies
        """
        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
        
        # Get configuration
        population_size = self.config.get("population_size", 50)
        elitism = self.config.get("elitism", 2)
        mutation_rate = self.config.get("mutation_rate", 0.2)
        crossover_rate = self.config.get("crossover_rate", 0.3)
        generation_number = current_population[0].metadata.get("generation", 0) + 1
        
        # New population starts empty
        new_population = []
        
        # Sort current population by fitness
        sorted_indices = sorted(range(len(evaluation_results)), 
                              key=lambda i: evaluation_results[i]["fitness"],
                              reverse=True)
        
        # Apply elitism - keep best strategies unchanged
        for i in range(min(elitism, len(sorted_indices))):
            elite_idx = sorted_indices[i]
            elite_strategy = current_population[elite_idx]
            
            # Create a copy of the elite strategy
            elite_copy = BensBotStrategyAdapter(
                benbot_strategy=elite_strategy.benbot_strategy,
                strategy_id=f"gen{generation_number}_elite{i}_{str(uuid.uuid4())[:8]}",
                parameters=copy.deepcopy(elite_strategy.parameters),
                metadata={
                    "generation": generation_number,
                    "parent_ids": [elite_strategy.strategy_id],
                    "elite": True
                }
            )
            
            new_population.append(elite_copy)
            logger.debug(f"Added elite strategy {elite_copy.strategy_id} with fitness {evaluation_results[elite_idx]['fitness']:.4f}")
        
        # Fill the rest through selection, crossover, and mutation
        while len(new_population) < population_size:
            # Determine operation: crossover or mutation
            if random.random() < crossover_rate and len(current_population) >= 2:
                # Select two parents using tournament selection
                parent1 = self._tournament_selection(current_population, evaluation_results)
                parent2 = self._tournament_selection(current_population, evaluation_results)
                
                # Perform crossover
                child = self._crossover(parent1, parent2, generation_number)
                
                # Add to new population
                new_population.append(child)
                logger.debug(f"Added crossover child {child.strategy_id}")
                
            else:
                # Select parent
                parent = self._tournament_selection(current_population, evaluation_results)
                
                # Perform mutation
                child = self._mutate(parent, mutation_rate, generation_number)
                
                # Add to new population
                new_population.append(child)
                logger.debug(f"Added mutation child {child.strategy_id}")
        
        # Ensure population size is exactly as specified
        if len(new_population) > population_size:
            new_population = new_population[:population_size]
            
        logger.info(f"Created new generation {generation_number} with {len(new_population)} strategies")
        return new_population
    
    def _tournament_selection(self, population, evaluation_results):
        """
        Select a strategy using tournament selection.
        
        Args:
            population: List of strategies
            evaluation_results: Evaluation metrics with fitness scores
            
        Returns:
            Selected strategy
        """
        tournament_size = self.config.get("tournament_size", 3)
        tournament_size = min(tournament_size, len(population))
        
        # Select random strategies for tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Find the best one
        best_idx = max(tournament_indices, 
                      key=lambda i: evaluation_results[i]["fitness"])
        
        return population[best_idx]
    
    def _crossover(self, parent1, parent2, generation_number):
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            generation_number: Current generation number
            
        Returns:
            Child strategy
        """
        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
        
        # Generate child ID
        child_id = f"gen{generation_number}_cross_{str(uuid.uuid4())[:8]}"
        
        # Create new parameters dict
        child_params = {}
        
        # Get all parameter names from both parents
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        # For each parameter, randomly select from either parent
        for param_name in all_params:
            if param_name in parent1.parameters and param_name in parent2.parameters:
                # Parameter exists in both parents
                # For numerical parameters, can also do interpolation
                p1_value = parent1.parameters[param_name]
                p2_value = parent2.parameters[param_name]
                
                if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                    # Randomly interpolate between values with bias towards better parent
                    alpha = random.uniform(0.25, 0.75)
                    
                    if isinstance(p1_value, int):
                        # For integers, round to nearest int
                        child_params[param_name] = int(p1_value * alpha + p2_value * (1 - alpha))
                    else:
                        # For floats, keep as float
                        child_params[param_name] = p1_value * alpha + p2_value * (1 - alpha)
                else:
                    # For non-numerical, randomly select
                    child_params[param_name] = random.choice([p1_value, p2_value])
            
            elif param_name in parent1.parameters:
                # Parameter only in parent1
                child_params[param_name] = parent1.parameters[param_name]
                
            elif param_name in parent2.parameters:
                # Parameter only in parent2
                child_params[param_name] = parent2.parameters[param_name]
        
        # Create child strategy
        child = BensBotStrategyAdapter(
            benbot_strategy=parent1.benbot_strategy,  # Use first parent's strategy type
            strategy_id=child_id,
            parameters=child_params,
            metadata={
                "generation": generation_number,
                "parent_ids": [parent1.strategy_id, parent2.strategy_id],
                "crossover": True
            }
        )
        
        return child
    
    def _mutate(self, parent, mutation_rate, generation_number):
        """
        Perform mutation on a parent strategy.
        
        Args:
            parent: Parent strategy
            mutation_rate: Probability of each parameter mutating
            generation_number: Current generation number
            
        Returns:
            Mutated child strategy
        """
        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
        
        # Generate child ID
        child_id = f"gen{generation_number}_mut_{str(uuid.uuid4())[:8]}"
        
        # Create new parameters dict (copy parent's)
        child_params = copy.deepcopy(parent.parameters)
        
        # Track mutations
        mutations = []
        
        # Check each parameter for mutation
        for param_name, param_value in parent.parameters.items():
            # Check if parameter should mutate
            if random.random() < mutation_rate:
                # Get parameter metadata
                param_meta = parent.param_metadata.get(param_name)
                
                if not param_meta or not param_meta.is_mutable:
                    # Skip immutable parameters
                    continue
                
                # Mutate based on parameter type
                if isinstance(param_value, int):
                    # For integers, adjust by a percentage
                    mutation_amount = int(param_value * param_meta.mutation_factor)
                    mutation_amount = max(1, mutation_amount)  # At least 1
                    
                    # Randomly increase or decrease
                    if random.random() < 0.5:
                        new_value = param_value + mutation_amount
                    else:
                        new_value = param_value - mutation_amount
                    
                    # Apply constraints
                    if param_meta.min_value is not None:
                        new_value = max(param_meta.min_value, new_value)
                    if param_meta.max_value is not None:
                        new_value = min(param_meta.max_value, new_value)
                    
                    # Record mutation
                    mutations.append({
                        "parameter": param_name,
                        "original": param_value,
                        "mutated": new_value
                    })
                    
                    # Update child parameter
                    child_params[param_name] = new_value
                    
                elif isinstance(param_value, float):
                    # For floats, adjust by a percentage
                    mutation_factor = param_meta.mutation_factor
                    mutation_amount = param_value * mutation_factor
                    
                    # Randomly increase or decrease
                    if random.random() < 0.5:
                        new_value = param_value + mutation_amount
                    else:
                        new_value = param_value - mutation_amount
                    
                    # Apply constraints
                    if param_meta.min_value is not None:
                        new_value = max(param_meta.min_value, new_value)
                    if param_meta.max_value is not None:
                        new_value = min(param_meta.max_value, new_value)
                    
                    # Record mutation
                    mutations.append({
                        "parameter": param_name,
                        "original": param_value,
                        "mutated": new_value
                    })
                    
                    # Update child parameter
                    child_params[param_name] = new_value
                    
                elif isinstance(param_value, bool):
                    # For booleans, flip the value
                    new_value = not param_value
                    
                    # Record mutation
                    mutations.append({
                        "parameter": param_name,
                        "original": param_value,
                        "mutated": new_value
                    })
                    
                    # Update child parameter
                    child_params[param_name] = new_value
        
        # Create child strategy
        child = BensBotStrategyAdapter(
            benbot_strategy=parent.benbot_strategy,
            strategy_id=child_id,
            parameters=child_params,
            metadata={
                "generation": generation_number,
                "parent_ids": [parent.strategy_id],
                "mutation": True,
                "mutation_details": mutations
            }
        )
        
        return child
