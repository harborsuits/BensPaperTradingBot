"""
Strategy hybridization system for EvoTrader.

This module enables the creation of new, potentially more effective
strategies by combining elements from successful parent strategies.
Inspired by genetic algorithms and breeding techniques from EA31337.
"""

import logging
import random
import copy
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Type, Callable
from collections import defaultdict

from ..core.strategy import Strategy, StrategyParameter, Signal, SignalType
from ..strategies.enhanced_strategy import EnhancedStrategy 
from ..strategies.multi_timeframe import MultiTimeFrameStrategy
from ..utils.indicator_system import Indicator, IndicatorFactory

logger = logging.getLogger(__name__)


class StrategyGene:
    """
    Represents a specific characteristic or parameter of a strategy.
    
    This class encapsulates a strategy's characteristic that can be inherited
    during hybridization, such as indicator parameters, timeframes, or
    decision thresholds.
    """
    
    def __init__(self, 
                 name: str, 
                 value: Any, 
                 gene_type: str = "parameter",
                 weight: float = 1.0,
                 performance_correlation: float = 0.0):
        """
        Initialize a strategy gene.
        
        Args:
            name: Gene name (e.g., 'rsi_period', 'overbought_threshold')
            value: Gene value
            gene_type: Type of gene (parameter, indicator, logic)
            weight: Importance weight of this gene (higher = more important)
            performance_correlation: Correlation with strategy performance
        """
        self.name = name
        self.value = value
        self.gene_type = gene_type
        self.weight = weight
        self.performance_correlation = performance_correlation
    
    def __repr__(self) -> str:
        """String representation of the gene."""
        return f"StrategyGene(name='{self.name}', value={self.value}, type='{self.gene_type}')"


class StrategyGenome:
    """
    Collection of genes that define a strategy's behavior.
    
    This class extracts and manages the complete set of parameters,
    indicators, and logic elements that make up a strategy.
    """
    
    def __init__(self, strategy: Strategy):
        """
        Initialize a genome from a strategy.
        
        Args:
            strategy: Source strategy to extract genes from
        """
        self.strategy_id = strategy.id if hasattr(strategy, 'id') else None
        self.strategy_class = strategy.__class__
        self.strategy_name = strategy.__class__.__name__
        
        # Extract genes from the strategy
        self.genes: Dict[str, StrategyGene] = {}
        self.extract_genes(strategy)
    
    def extract_genes(self, strategy: Strategy) -> None:
        """
        Extract all genes from a strategy.
        
        Args:
            strategy: Strategy to extract genes from
        """
        # Extract parameter genes
        if hasattr(strategy, 'parameters'):
            for param_name, param_value in strategy.parameters.items():
                self.genes[param_name] = StrategyGene(
                    name=param_name,
                    value=param_value,
                    gene_type="parameter"
                )
        
        # Extract indicator genes if it's an EnhancedStrategy
        if isinstance(strategy, EnhancedStrategy) and hasattr(strategy, 'indicators'):
            for symbol, indicators in strategy.indicators.items():
                for ind_name, indicator in indicators.items():
                    if hasattr(indicator, 'params'):
                        for param_name, param_value in indicator.params.items():
                            gene_name = f"indicator_{ind_name}_{param_name}"
                            self.genes[gene_name] = StrategyGene(
                                name=gene_name,
                                value=param_value,
                                gene_type="indicator_param"
                            )
        
        # Extract timeframe genes if it's a MultiTimeFrameStrategy
        if isinstance(strategy, MultiTimeFrameStrategy):
            for tf_name in ['primary_timeframe', 'secondary_timeframe', 'tertiary_timeframe']:
                if hasattr(strategy, 'parameters') and tf_name in strategy.parameters:
                    self.genes[tf_name] = StrategyGene(
                        name=tf_name,
                        value=strategy.parameters.get(tf_name),
                        gene_type="timeframe"
                    )
    
    def update_performance_correlation(self, performance_data: Dict[str, float]) -> None:
        """
        Update the performance correlation of genes based on strategy performance.
        
        Args:
            performance_data: Dictionary of performance metrics
        """
        # Simple example, in practice would use more sophisticated correlation analysis
        score = self._calculate_performance_score(performance_data)
        
        # Update the performance correlation for all genes
        # In a real implementation, you'd want to analyze which genes
        # actually contributed to the performance
        for gene_name, gene in self.genes.items():
            # For simplicity, we're just setting all genes to the same correlation
            gene.performance_correlation = score
    
    def _calculate_performance_score(self, performance_data: Dict[str, float]) -> float:
        """
        Calculate a single performance score from multiple metrics.
        
        Args:
            performance_data: Dictionary of performance metrics
            
        Returns:
            Combined performance score
        """
        # Extract key performance metrics
        profit = performance_data.get('profit', 0)
        sharpe = performance_data.get('sharpe_ratio', 0)
        max_drawdown = performance_data.get('max_drawdown', 100)
        win_rate = performance_data.get('win_rate', 0)
        
        # Combine metrics into a single score
        # Higher is better
        score = (
            0.4 * profit + 
            0.3 * sharpe + 
            0.1 * (100 - max_drawdown) + 
            0.2 * win_rate
        )
        
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert genome to dictionary.
        
        Returns:
            Dictionary representation of the genome
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_class': self.strategy_class.__name__,
            'strategy_name': self.strategy_name,
            'genes': {
                name: {
                    'name': gene.name,
                    'value': gene.value,
                    'gene_type': gene.gene_type,
                    'weight': gene.weight,
                    'performance_correlation': gene.performance_correlation
                }
                for name, gene in self.genes.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], strategy_registry: Dict[str, Type[Strategy]]) -> 'StrategyGenome':
        """
        Create a genome from a dictionary.
        
        Args:
            data: Dictionary representation of a genome
            strategy_registry: Dictionary mapping strategy names to classes
            
        Returns:
            StrategyGenome instance
        """
        # Get the strategy class
        strategy_class_name = data.get('strategy_class')
        strategy_class = strategy_registry.get(strategy_class_name)
        
        if not strategy_class:
            raise ValueError(f"Strategy class {strategy_class_name} not found in registry")
        
        # Create a dummy strategy instance
        strategy = strategy_class()
        
        # Create a genome
        genome = cls(strategy)
        
        # Update the genome with the data
        genome.strategy_id = data.get('strategy_id')
        genome.strategy_name = data.get('strategy_name')
        
        # Update the genes
        for gene_name, gene_data in data.get('genes', {}).items():
            genome.genes[gene_name] = StrategyGene(
                name=gene_data.get('name'),
                value=gene_data.get('value'),
                gene_type=gene_data.get('gene_type'),
                weight=gene_data.get('weight'),
                performance_correlation=gene_data.get('performance_correlation')
            )
        
        return genome


class StrategyHybridizer:
    """
    Creates new strategies by combining aspects of successful parent strategies.
    
    This class implements various breeding techniques to combine strategies
    based on their performance and genetic makeup.
    """
    
    def __init__(self, 
                 strategy_registry: Dict[str, Type[Strategy]],
                 crossover_probability: float = 0.7,
                 mutation_probability: float = 0.2,
                 elite_percentage: float = 0.1):
        """
        Initialize the hybridizer.
        
        Args:
            strategy_registry: Dictionary mapping strategy names to classes
            crossover_probability: Probability of crossover between parents
            mutation_probability: Probability of mutation after crossover
            elite_percentage: Percentage of top strategies to keep unchanged
        """
        self.strategy_registry = strategy_registry
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elite_percentage = elite_percentage
        self.performance_history = {}
    
    def hybridize(self, 
                  parent_strategies: List[Strategy], 
                  performance_data: List[Dict[str, float]],
                  num_offspring: int) -> List[Strategy]:
        """
        Generate new strategies by combining elements of parent strategies.
        
        Args:
            parent_strategies: List of parent strategies
            performance_data: List of performance metrics for each parent
            num_offspring: Number of new strategies to generate
            
        Returns:
            List of new hybrid strategies
        """
        # Create genomes for all parent strategies
        parent_genomes = []
        for i, strategy in enumerate(parent_strategies):
            genome = StrategyGenome(strategy)
            if i < len(performance_data):
                genome.update_performance_correlation(performance_data[i])
            parent_genomes.append(genome)
        
        # Sort parent genomes by performance
        sorted_parent_indices = sorted(
            range(len(performance_data)),
            key=lambda i: self._calculate_performance_score(performance_data[i]),
            reverse=True
        )
        
        sorted_parent_genomes = [parent_genomes[i] for i in sorted_parent_indices]
        
        # Keep elite strategies unchanged
        num_elite = max(1, int(len(parent_strategies) * self.elite_percentage))
        elite_genomes = sorted_parent_genomes[:num_elite]
        
        # Generate offspring
        offspring_genomes = []
        
        # First, add the elite genomes
        offspring_genomes.extend(elite_genomes)
        
        # Then generate the rest through crossover and mutation
        while len(offspring_genomes) < num_offspring:
            # Select two parents using tournament selection
            parent1 = self._tournament_selection(sorted_parent_genomes)
            parent2 = self._tournament_selection(sorted_parent_genomes)
            
            # Perform crossover if random value is less than crossover probability
            if random.random() < self.crossover_probability:
                child_genome = self._crossover(parent1, parent2)
            else:
                # Otherwise just clone one of the parents
                child_genome = copy.deepcopy(parent1 if random.random() < 0.5 else parent2)
            
            # Perform mutation if random value is less than mutation probability
            if random.random() < self.mutation_probability:
                self._mutate(child_genome)
            
            offspring_genomes.append(child_genome)
        
        # Convert genomes back to strategies
        offspring_strategies = [
            self._genome_to_strategy(genome)
            for genome in offspring_genomes[:num_offspring]
        ]
        
        return offspring_strategies
    
    def _calculate_performance_score(self, performance_data: Dict[str, float]) -> float:
        """
        Calculate a single performance score from multiple metrics.
        
        Args:
            performance_data: Dictionary of performance metrics
            
        Returns:
            Combined performance score
        """
        # Extract key performance metrics
        profit = performance_data.get('profit', 0)
        sharpe = performance_data.get('sharpe_ratio', 0)
        max_drawdown = performance_data.get('max_drawdown', 100)
        win_rate = performance_data.get('win_rate', 0)
        
        # Combine metrics into a single score
        # Higher is better
        score = (
            0.4 * profit + 
            0.3 * sharpe + 
            0.1 * (100 - max_drawdown) + 
            0.2 * win_rate
        )
        
        return score
    
    def _tournament_selection(self, 
                            sorted_genomes: List[StrategyGenome], 
                            tournament_size: int = 3) -> StrategyGenome:
        """
        Select a parent using tournament selection.
        
        Args:
            sorted_genomes: List of genomes sorted by performance
            tournament_size: Number of genomes to include in the tournament
            
        Returns:
            Selected parent genome
        """
        # Randomly select tournament_size genomes
        tournament = random.sample(
            sorted_genomes,
            min(tournament_size, len(sorted_genomes))
        )
        
        # Return the best one
        return tournament[0]
    
    def _crossover(self, 
                  parent1: StrategyGenome, 
                  parent2: StrategyGenome) -> StrategyGenome:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Child genome
        """
        # Check if parents are compatible for crossover
        if parent1.strategy_class != parent2.strategy_class:
            # For simplicity, if they're not the same class, just return a copy of parent1
            logger.warning(f"Cannot crossover different strategy classes: {parent1.strategy_name} and {parent2.strategy_name}")
            return copy.deepcopy(parent1)
        
        # Create a child genome as a copy of parent1
        child = copy.deepcopy(parent1)
        
        # For each gene in parent2, decide whether to inherit it
        for gene_name, gene in parent2.genes.items():
            # If the gene is also in parent1, decide which one to keep
            if gene_name in child.genes:
                # Use performance correlation to decide which gene to inherit
                # Higher correlation = more likely to inherit
                p1_correlation = child.genes[gene_name].performance_correlation
                p2_correlation = gene.performance_correlation
                
                # Normalize correlations to probabilities
                total_correlation = max(0.01, abs(p1_correlation) + abs(p2_correlation))
                p2_probability = abs(p2_correlation) / total_correlation
                
                # Inherit from parent2 with probability based on correlation
                if random.random() < p2_probability:
                    child.genes[gene_name] = copy.deepcopy(gene)
            else:
                # If the gene is only in parent2, add it to the child
                child.genes[gene_name] = copy.deepcopy(gene)
        
        return child
    
    def _mutate(self, genome: StrategyGenome) -> None:
        """
        Mutate a genome.
        
        Args:
            genome: Genome to mutate
        """
        # Randomly select genes to mutate
        for gene_name, gene in genome.genes.items():
            # Each gene has a chance to mutate
            if random.random() < self.mutation_probability:
                # Different mutation strategies based on gene type
                if gene.gene_type == "parameter":
                    # For numeric parameters, adjust by a small amount
                    if isinstance(gene.value, (int, float)):
                        # Determine mutation range based on type
                        if isinstance(gene.value, int):
                            # For integers, mutate by ±10%
                            mutation_range = max(1, int(abs(gene.value) * 0.1))
                            gene.value += random.randint(-mutation_range, mutation_range)
                        else:
                            # For floats, mutate by ±10%
                            mutation_factor = random.uniform(0.9, 1.1)
                            gene.value *= mutation_factor
                elif gene.gene_type == "indicator_param":
                    # Similar to parameter, but might have different ranges
                    if isinstance(gene.value, (int, float)):
                        # For indicator periods, ensure they stay positive
                        if isinstance(gene.value, int):
                            mutation_range = max(1, int(abs(gene.value) * 0.1))
                            gene.value = max(1, gene.value + random.randint(-mutation_range, mutation_range))
                        else:
                            mutation_factor = random.uniform(0.9, 1.1)
                            gene.value = max(0.01, gene.value * mutation_factor)
                elif gene.gene_type == "timeframe":
                    # For timeframes, we might want to switch to an adjacent timeframe
                    timeframes = [
                        "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
                    ]
                    
                    if gene.value in timeframes:
                        current_index = timeframes.index(gene.value)
                        # Move up or down one timeframe
                        new_index = max(0, min(len(timeframes) - 1, 
                                            current_index + random.choice([-1, 1])))
                        gene.value = timeframes[new_index]
    
    def _genome_to_strategy(self, genome: StrategyGenome) -> Strategy:
        """
        Convert a genome back to a strategy.
        
        Args:
            genome: Genome to convert
            
        Returns:
            Strategy instance
        """
        # Get the strategy class
        strategy_class = genome.strategy_class
        
        # Extract parameters
        parameters = {}
        for gene_name, gene in genome.genes.items():
            if gene.gene_type == "parameter":
                parameters[gene_name] = gene.value
        
        # Create a new strategy with the parameters
        strategy = strategy_class(parameters=parameters)
        
        # For more complex strategies like EnhancedStrategy,
        # we would need to set up indicators with the right parameters
        if isinstance(strategy, EnhancedStrategy) and hasattr(strategy, 'indicators'):
            indicator_params = {}
            for gene_name, gene in genome.genes.items():
                if gene.gene_type == "indicator_param":
                    # Extract indicator name and param from gene name
                    # Format is "indicator_<name>_<param>"
                    parts = gene_name.split('_', 2)
                    if len(parts) >= 3:
                        ind_name = parts[1]
                        param_name = parts[2]
                        
                        if ind_name not in indicator_params:
                            indicator_params[ind_name] = {}
                        
                        indicator_params[ind_name][param_name] = gene.value
            
            # We'll need to recreate the indicators with these parameters
            # in the strategy's setup_indicators method
            strategy.indicator_params = indicator_params
        
        return strategy


class StrategyEvolutionManager:
    """
    Manages the evolutionary process for trading strategies.
    
    This class handles the full lifecycle of strategy evolution, including
    selection, hybridization, mutation, and performance tracking.
    """
    
    def __init__(self, 
                 strategy_registry: Dict[str, Type[Strategy]],
                 population_size: int = 100,
                 generations: int = 10,
                 elite_percentage: float = 0.1,
                 selection_percentage: float = 0.5,
                 crossover_probability: float = 0.7,
                 mutation_probability: float = 0.2):
        """
        Initialize the evolution manager.
        
        Args:
            strategy_registry: Dictionary mapping strategy names to classes
            population_size: Number of strategies in the population
            generations: Number of generations to evolve
            elite_percentage: Percentage of top strategies to keep unchanged
            selection_percentage: Percentage of strategies to use for breeding
            crossover_probability: Probability of crossover between parents
            mutation_probability: Probability of mutation after crossover
        """
        self.strategy_registry = strategy_registry
        self.population_size = population_size
        self.generations = generations
        self.elite_percentage = elite_percentage
        self.selection_percentage = selection_percentage
        
        # Create the hybridizer
        self.hybridizer = StrategyHybridizer(
            strategy_registry=strategy_registry,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            elite_percentage=elite_percentage
        )
        
        # Track current population and performance
        self.current_population: List[Strategy] = []
        self.current_performance: List[Dict[str, float]] = []
        
        # Track evolution history
        self.evolution_history = []
    
    def initialize_population(self, 
                             strategy_classes: List[Type[Strategy]],
                             initial_params: Optional[List[Dict[str, Any]]] = None) -> List[Strategy]:
        """
        Initialize the initial population of strategies.
        
        Args:
            strategy_classes: List of strategy classes to initialize
            initial_params: Optional list of initial parameters for strategies
            
        Returns:
            List of initialized strategies
        """
        population = []
        
        # If no initial params provided, create random ones
        if not initial_params:
            initial_params = [None] * self.population_size
        
        # Create strategies up to population_size
        while len(population) < self.population_size:
            for strategy_class in strategy_classes:
                # Skip if we've reached the population size
                if len(population) >= self.population_size:
                    break
                
                # Get parameters if available
                params = initial_params[len(population)] if len(population) < len(initial_params) else None
                
                # Create a new strategy
                strategy = strategy_class(parameters=params)
                
                # Add to population
                population.append(strategy)
        
        # Store as current population
        self.current_population = population
        
        return population
    
    def evolve_generation(self, 
                         performance_data: List[Dict[str, float]]) -> Tuple[List[Strategy], List[Dict[str, Any]]]:
        """
        Evolve a single generation based on performance data.
        
        Args:
            performance_data: List of performance metrics for each strategy
            
        Returns:
            Tuple of (new_population, evolution_stats)
        """
        # Store current performance
        self.current_performance = performance_data
        
        # Calculate number of strategies to select
        num_selected = max(2, int(self.population_size * self.selection_percentage))
        
        # Sort strategies by performance
        sorted_indices = sorted(
            range(len(performance_data)),
            key=lambda i: self.hybridizer._calculate_performance_score(performance_data[i]),
            reverse=True
        )
        
        # Select the top performing strategies
        selected_strategies = [self.current_population[i] for i in sorted_indices[:num_selected]]
        selected_performance = [performance_data[i] for i in sorted_indices[:num_selected]]
        
        # Generate new population
        new_population = self.hybridizer.hybridize(
            selected_strategies,
            selected_performance,
            self.population_size
        )
        
        # Record evolution statistics
        evolution_stats = {
            'generation': len(self.evolution_history) + 1,
            'best_score': self.hybridizer._calculate_performance_score(performance_data[sorted_indices[0]]),
            'avg_score': sum(self.hybridizer._calculate_performance_score(p) for p in performance_data) / len(performance_data),
            'best_strategy': self.current_population[sorted_indices[0]].__class__.__name__,
            'selected_strategies': [s.__class__.__name__ for s in selected_strategies]
        }
        
        # Update current population
        self.current_population = new_population
        
        # Add to history
        self.evolution_history.append(evolution_stats)
        
        return new_population, evolution_stats
    
    def run_evolution(self, 
                     evaluate_function: Callable[[List[Strategy]], List[Dict[str, float]]],
                     initial_population: Optional[List[Strategy]] = None) -> Tuple[List[Strategy], List[Dict[str, Any]]]:
        """
        Run the full evolutionary process.
        
        Args:
            evaluate_function: Function to evaluate a population of strategies
            initial_population: Optional initial population of strategies
            
        Returns:
            Tuple of (final_population, evolution_history)
        """
        # Initialize population if not provided
        if initial_population:
            self.current_population = initial_population
        elif not self.current_population:
            raise ValueError("Population not initialized. Either provide initial_population or call initialize_population first.")
        
        # Run evolution for specified number of generations
        for generation in range(self.generations):
            # Evaluate current population
            performance_data = evaluate_function(self.current_population)
            
            # Evolve to next generation
            self.current_population, stats = self.evolve_generation(performance_data)
            
            logger.info(f"Generation {generation + 1} complete. Best score: {stats['best_score']:.2f}, Avg score: {stats['avg_score']:.2f}")
        
        # Return final population and evolution history
        return self.current_population, self.evolution_history
