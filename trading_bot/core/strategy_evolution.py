#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Evolution System

This module implements a genetic algorithm-based evolution system for discovering
and optimizing trading strategies across diverse market conditions.
"""

import logging
import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import uuid
import json

from trading_bot.core.constants import EventType, MarketRegime
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger(__name__)

class StrategyEvolution:
    """
    Genetic algorithm-based evolution system for trading strategies.
    
    Features:
    - Multi-regime optimization (trending, ranging, volatile, etc.)
    - Parameter-based strategy evolution
    - Population-based approach with crossover and mutation
    - Fitness evaluation across diverse market conditions
    - Tracking of evolution history and performance metrics
    - Integration with strategy registry for deployment
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 market_conditions: List[str] = None,
                 persistence: Optional[PersistenceManager] = None,
                 fitness_evaluator: Optional[Callable] = None):
        """
        Initialize the evolution system.
        
        Args:
            population_size: Size of each generation's population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation (0.0-1.0)
            crossover_rate: Probability of crossover (0.0-1.0)
            market_conditions: List of market regimes to evaluate against
            persistence: Persistence manager for saving/loading strategies
            fitness_evaluator: Optional custom fitness evaluation function
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Default market conditions if none provided
        if not market_conditions:
            self.market_conditions = [
                'trending_up', 'trending_down', 'ranging', 
                'volatile', 'breakout', 'low_volatility'
            ]
        else:
            self.market_conditions = market_conditions
            
        self.strategy_registry = {}
        self.fitness_history = []
        self.current_generation = 0
        self.best_fitness_per_generation = []
        self.persistence = persistence
        self.fitness_evaluator = fitness_evaluator
        
        # Evolution status tracking
        self.evolution_status = {
            'running': False,
            'current_generation': 0,
            'best_fitness': 0.0,
            'start_time': None,
            'end_time': None,
            'elapsed_time': None
        }
    
    def initialize_population(self, strategy_templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate initial population with random parameters within defined ranges.
        
        Args:
            strategy_templates: Templates defining parameter ranges for strategies
            
        Returns:
            List of initialized strategy instances with randomized parameters
        """
        population = []
        
        for _ in range(self.population_size):
            # Choose a random template
            template = random.choice(strategy_templates)
            
            # Create a new strategy based on the template
            strategy = {
                'strategy_id': str(uuid.uuid4()),
                'strategy_name': f"{template['base_name']}_{len(population)}",
                'strategy_type': template['strategy_type'],
                'parameters': {},
                'template_id': template['template_id'],
                'creation_time': datetime.now().isoformat(),
                'generation': 0,
                'parent_ids': [],
                'metadata': {
                    'evolved': True,
                    'generation': 0,
                    'mutation_count': 0,
                    'crossover_count': 0,
                }
            }
            
            # Initialize parameters within the defined ranges
            for param_name, param_range in template['parameter_ranges'].items():
                if isinstance(param_range, list) and len(param_range) == 2:
                    # Numerical parameter
                    min_val, max_val = param_range
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        strategy['parameters'][param_name] = random.randint(min_val, max_val)
                    else:
                        # Float parameter
                        strategy['parameters'][param_name] = random.uniform(min_val, max_val)
                        
                elif isinstance(param_range, list):
                    # Categorical parameter, choose one from the list
                    strategy['parameters'][param_name] = random.choice(param_range)
                    
                elif isinstance(param_range, dict) and 'type' in param_range:
                    # Special parameter types
                    if param_range['type'] == 'boolean':
                        strategy['parameters'][param_name] = random.choice([True, False])
                    elif param_range['type'] == 'weighted_choice' and 'choices' in param_range:
                        choices = param_range['choices']
                        weights = param_range.get('weights', [1.0/len(choices)] * len(choices))
                        strategy['parameters'][param_name] = random.choices(
                            choices, weights=weights, k=1
                        )[0]
            
            # Add any fixed parameters from the template
            if 'fixed_parameters' in template:
                for param_name, param_value in template['fixed_parameters'].items():
                    strategy['parameters'][param_name] = param_value
            
            # Add to population
            population.append(strategy)
            
        return population
    
    def evaluate_fitness(self, strategy: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Evaluate the fitness of a strategy across different market conditions.
        
        Args:
            strategy: Strategy configuration to evaluate
            market_data: Dictionary of market data for different conditions
            
        Returns:
            Fitness score (higher is better)
        """
        if self.fitness_evaluator:
            # Use custom fitness evaluator if provided
            return self.fitness_evaluator(strategy, market_data)
        
        # Default fitness evaluation
        fitness_scores = []
        
        for condition, data in market_data.items():
            if data.empty:
                continue
                
            # Simulate the strategy on this market condition
            try:
                # Simple backtest implementation - in practice, you'd use your
                # actual backtester with position sizing and proper execution
                returns, dd, sharpe, win_rate = self._quick_backtest(strategy, data)
                
                # Calculate fitness for this condition (example formula)
                # Weight by Sharpe ratio and win rate, penalize by drawdown
                condition_fitness = (sharpe * 0.4) + (win_rate * 0.3) - (dd * 0.3)
                
                # Add to scores
                fitness_scores.append((condition, condition_fitness))
                
            except Exception as e:
                logger.error(f"Error evaluating strategy {strategy['strategy_id']}: {str(e)}")
                fitness_scores.append((condition, -1.0))  # Penalty for error
        
        if not fitness_scores:
            return -1.0  # Could not evaluate
            
        # Calculate overall fitness - can be weighted by regime importance
        # Here we use a simple average, but you could weight some regimes higher
        overall_fitness = sum(score for _, score in fitness_scores) / len(fitness_scores)
        
        # Add a bonus for strategies that work well across all conditions
        min_fitness = min(score for _, score in fitness_scores)
        if min_fitness > 0:
            robustness_bonus = min_fitness * 0.2
            overall_fitness += robustness_bonus
            
        # Record fitness details in strategy
        strategy['fitness'] = {
            'overall': overall_fitness,
            'by_condition': {cond: score for cond, score in fitness_scores},
            'evaluation_time': datetime.now().isoformat()
        }
        
        return overall_fitness
    
    def _quick_backtest(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        Perform a quick backtest of the strategy on given data.
        
        Args:
            strategy: Strategy configuration
            data: Market data as DataFrame
            
        Returns:
            Tuple of (returns, max_drawdown, sharpe_ratio, win_rate)
        """
        # This is a simplified example - you would replace this with your actual backtester
        # In practice, this would use your trading bot's backtesting engine
        
        # Mock implementation for example purposes
        params = strategy['parameters']
        
        # Extract some common parameters (if they exist)
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        
        # Mock strategy - moving average crossover
        data = data.copy()
        data['fast_ma'] = data['close'].rolling(window=fast_period).mean()
        data['slow_ma'] = data['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1
        
        # Calculate returns (simplified)
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['signal'].shift(1) * data['returns']
        
        # Clean up NaN values
        data = data.dropna()
        
        if len(data) < 2:
            return 0.0, 0.0, 0.0, 0.0
            
        # Calculate metrics
        cumulative_returns = (1 + data['strategy_returns']).cumprod().iloc[-1] - 1
        
        # Calculate drawdown
        cum_returns = (1 + data['strategy_returns']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = (data['strategy_returns'].mean() / data['strategy_returns'].std()) * (252 ** 0.5)
        
        # Calculate win rate
        wins = (data['strategy_returns'] > 0).sum()
        total_trades = (data['signal'] != data['signal'].shift(1)).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return cumulative_returns, abs(max_drawdown), sharpe_ratio, win_rate
    
    def select_parents(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Select parents for the next generation using tournament selection.
        
        Args:
            population: Current population of strategies
            fitness_scores: Corresponding fitness scores
            
        Returns:
            Selected parents
        """
        parents = []
        
        # Create population indices with their fitness scores
        population_with_fitness = list(zip(population, fitness_scores))
        
        # Number of parents to select (same as population size)
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = min(5, len(population_with_fitness))
            tournament = random.sample(population_with_fitness, tournament_size)
            
            # Select the best from the tournament
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
            
        return parents
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a child strategy by combining parameters from two parents.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Child strategy
        """
        # Only crossover if parents are of the same template
        if parent1.get('template_id') != parent2.get('template_id'):
            # If not compatible, return a copy of the first parent
            child = copy.deepcopy(parent1)
            child['strategy_id'] = str(uuid.uuid4())
            child['parent_ids'] = [parent1['strategy_id']]
            child['metadata']['crossover_count'] = 0
            return child
            
        # Create a new child with a unique ID
        child = {
            'strategy_id': str(uuid.uuid4()),
            'strategy_name': f"Child_{parent1['strategy_name']}_{parent2['strategy_name']}",
            'strategy_type': parent1['strategy_type'],
            'parameters': {},
            'template_id': parent1['template_id'],
            'creation_time': datetime.now().isoformat(),
            'generation': max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1,
            'parent_ids': [parent1['strategy_id'], parent2['strategy_id']],
            'metadata': {
                'evolved': True,
                'generation': max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1,
                'mutation_count': 0,
                'crossover_count': 1 + parent1.get('metadata', {}).get('crossover_count', 0) 
                                     + parent2.get('metadata', {}).get('crossover_count', 0),
            }
        }
        
        # For each parameter, randomly choose from either parent
        # or do a weighted average for numerical parameters
        for param_name in set(parent1['parameters'].keys()) | set(parent2['parameters'].keys()):
            # If parameter only exists in one parent, use that value
            if param_name not in parent1['parameters']:
                child['parameters'][param_name] = parent2['parameters'][param_name]
                continue
                
            if param_name not in parent2['parameters']:
                child['parameters'][param_name] = parent1['parameters'][param_name]
                continue
                
            # Both parents have this parameter
            p1_value = parent1['parameters'][param_name]
            p2_value = parent2['parameters'][param_name]
            
            # Handle different parameter types
            if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                # Numerical parameter - do interpolation or random choice
                if random.random() < 0.5:
                    # Random choice between parents
                    child['parameters'][param_name] = random.choice([p1_value, p2_value])
                else:
                    # Interpolation between parents
                    alpha = random.random()  # Interpolation factor
                    interpolated = p1_value * alpha + p2_value * (1 - alpha)
                    
                    # Convert to int if both parents have int values
                    if isinstance(p1_value, int) and isinstance(p2_value, int):
                        interpolated = int(round(interpolated))
                        
                    child['parameters'][param_name] = interpolated
            else:
                # Non-numerical parameter - just choose one
                child['parameters'][param_name] = random.choice([p1_value, p2_value])
                
        return child
    
    def mutate(self, strategy: Dict[str, Any], strategy_templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mutate a strategy by randomly adjusting its parameters.
        
        Args:
            strategy: Strategy to mutate
            strategy_templates: Templates with parameter ranges
            
        Returns:
            Mutated strategy
        """
        # Find the template for this strategy
        template = None
        for t in strategy_templates:
            if t['template_id'] == strategy.get('template_id'):
                template = t
                break
                
        if not template:
            # Can't mutate without template information
            logger.warning(f"Could not find template for strategy {strategy['strategy_id']}")
            return strategy
            
        # Create a copy of the strategy
        mutated = copy.deepcopy(strategy)
        mutated['strategy_id'] = str(uuid.uuid4())
        mutated['strategy_name'] = f"Mutated_{strategy['strategy_name']}"
        mutated['parent_ids'] = [strategy['strategy_id']]
        
        # Update metadata
        mutated['metadata']['mutation_count'] = strategy.get('metadata', {}).get('mutation_count', 0) + 1
        
        # Randomly select parameters to mutate
        for param_name, param_range in template['parameter_ranges'].items():
            # Only mutate some parameters (based on mutation rate)
            if random.random() > self.mutation_rate:
                continue
                
            if isinstance(param_range, list) and len(param_range) == 2:
                # Numerical parameter
                min_val, max_val = param_range
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    mutated['parameters'][param_name] = random.randint(min_val, max_val)
                else:
                    # Float parameter
                    mutated['parameters'][param_name] = random.uniform(min_val, max_val)
                    
            elif isinstance(param_range, list):
                # Categorical parameter, choose one from the list
                mutated['parameters'][param_name] = random.choice(param_range)
                
            elif isinstance(param_range, dict) and 'type' in param_range:
                # Special parameter types
                if param_range['type'] == 'boolean':
                    mutated['parameters'][param_name] = random.choice([True, False])
                elif param_range['type'] == 'weighted_choice' and 'choices' in param_range:
                    choices = param_range['choices']
                    weights = param_range.get('weights', [1.0/len(choices)] * len(choices))
                    mutated['parameters'][param_name] = random.choices(
                        choices, weights=weights, k=1
                    )[0]
                    
        return mutated
    
    def evolve(self, 
               strategy_templates: List[Dict[str, Any]], 
               market_data: Dict[str, pd.DataFrame],
               max_time_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full evolution process.
        
        Args:
            strategy_templates: Templates defining parameter ranges
            market_data: Market data for different conditions
            max_time_seconds: Maximum time to run evolution in seconds
            
        Returns:
            Evolution results with best strategies
        """
        # Initialize evolution tracking
        self.evolution_status['running'] = True
        self.evolution_status['start_time'] = datetime.now()
        self.evolution_status['best_fitness'] = 0.0
        
        # Initialize population
        population = self.initialize_population(strategy_templates)
        
        start_time = datetime.now()
        self.current_generation = 0
        
        # Main evolution loop
        for generation in range(self.generations):
            self.current_generation = generation
            self.evolution_status['current_generation'] = generation
            
            # Check time limit
            if max_time_seconds and (datetime.now() - start_time).total_seconds() > max_time_seconds:
                logger.info(f"Evolution stopped due to time limit after {generation} generations")
                break
                
            # Evaluate fitness for each strategy
            fitness_scores = []
            for idx, strategy in enumerate(population):
                fitness = self.evaluate_fitness(strategy, market_data)
                fitness_scores.append(fitness)
                
                # Log progress occasionally
                if idx % 10 == 0:
                    logger.info(f"Generation {generation}, evaluated {idx}/{len(population)} strategies")
            
            # Track best fitness
            best_idx = fitness_scores.index(max(fitness_scores))
            best_strategy = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            self.best_fitness_per_generation.append(best_fitness)
            
            if best_fitness > self.evolution_status['best_fitness']:
                self.evolution_status['best_fitness'] = best_fitness
            
            logger.info(f"Generation {generation} complete. Best fitness: {best_fitness:.4f}")
            
            # Add best strategy to registry
            self.strategy_registry[best_strategy['strategy_id']] = best_strategy
            
            # Create next generation (except for last iteration)
            if generation < self.generations - 1:
                # Select parents
                parents = self.select_parents(population, fitness_scores)
                
                # Create new population through crossover and mutation
                new_population = []
                
                while len(new_population) < self.population_size:
                    # Select two parents
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    
                    # Crossover with probability
                    if random.random() < self.crossover_rate:
                        child = self.crossover(parent1, parent2)
                    else:
                        # No crossover, just copy a parent
                        child = copy.deepcopy(random.choice([parent1, parent2]))
                        child['strategy_id'] = str(uuid.uuid4())
                        child['parent_ids'] = [parent1['strategy_id']]
                    
                    # Mutate with probability
                    if random.random() < self.mutation_rate:
                        child = self.mutate(child, strategy_templates)
                        
                    # Add to new population
                    new_population.append(child)
                    
                # Replace old population
                population = new_population
                
        # Evolution complete
        self.evolution_status['running'] = False
        self.evolution_status['end_time'] = datetime.now()
        self.evolution_status['elapsed_time'] = (self.evolution_status['end_time'] - 
                                               self.evolution_status['start_time']).total_seconds()
        
        # Final results
        results = {
            'generations_completed': self.current_generation + 1,
            'best_fitness_per_generation': self.best_fitness_per_generation,
            'elapsed_time': self.evolution_status['elapsed_time'],
            'top_strategies': self.get_top_strategies(10),
            'total_strategies_evaluated': (self.current_generation + 1) * self.population_size,
        }
        
        return results
    
    def get_top_strategies(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top performing strategies from the registry.
        
        Args:
            top_n: Number of top strategies to return
            
        Returns:
            List of top strategies
        """
        if not self.strategy_registry:
            return []
            
        # Sort strategies by fitness
        strategies = list(self.strategy_registry.values())
        strategies.sort(key=lambda s: s.get('fitness', {}).get('overall', -float('inf')), reverse=True)
        
        return strategies[:top_n]
    
    def register_top_strategies(self, top_n: int = 10) -> List[str]:
        """
        Save the best strategies to the registry.
        
        Args:
            top_n: Number of top strategies to register
            
        Returns:
            List of registered strategy IDs
        """
        top_strategies = self.get_top_strategies(top_n)
        
        # Save to persistence
        if self.persistence:
            for strategy in top_strategies:
                try:
                    self.persistence.save_strategy(strategy)
                    logger.info(f"Registered evolved strategy: {strategy['strategy_name']}")
                except Exception as e:
                    logger.error(f"Failed to register strategy {strategy['strategy_id']}: {str(e)}")
        
        return [s['strategy_id'] for s in top_strategies]
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get the current status of the evolution process."""
        return self.evolution_status
    
    def load_market_data(self, symbol: str, timeframe: str, 
                         start_date: datetime, end_date: datetime,
                         data_provider: Any) -> Dict[str, pd.DataFrame]:
        """
        Load market data for different market conditions.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            data_provider: Data provider to load data
            
        Returns:
            Dictionary of market data for different conditions
        """
        # This is a template function - implement based on your actual data provider
        market_data = {}
        
        # Load the base dataset
        try:
            full_data = data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if full_data.empty:
                logger.error(f"No data found for {symbol} {timeframe} from {start_date} to {end_date}")
                return market_data
                
            # Identify different market regimes/conditions in the data
            # This is a simplified example - you would use your actual market regime detection
            for condition in self.market_conditions:
                # Filter data for this condition
                condition_data = self._filter_data_by_condition(full_data, condition)
                
                if not condition_data.empty:
                    market_data[condition] = condition_data
                else:
                    logger.warning(f"No data found for condition: {condition}")
                    
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            
        return market_data
    
    def _filter_data_by_condition(self, data: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Filter data to include only periods matching a specific market condition.
        
        Args:
            data: Full market data
            condition: Market condition to filter for
            
        Returns:
            Filtered dataframe containing only the specific condition
        """
        # This is a simplified example - you would use your actual regime detection logic
        # This mock implementation divides the data into chunks by condition
        
        if condition == 'trending_up':
            # Find periods of uptrends (example: price higher than 50-day MA and rising)
            data['ma50'] = data['close'].rolling(window=50).mean()
            trending_up = (data['close'] > data['ma50']) & (data['close'].diff() > 0)
            return data[trending_up]
            
        elif condition == 'trending_down':
            # Find periods of downtrends
            data['ma50'] = data['close'].rolling(window=50).mean()
            trending_down = (data['close'] < data['ma50']) & (data['close'].diff() < 0)
            return data[trending_down]
            
        elif condition == 'ranging':
            # Find periods of range-bound markets (example: price within 5% of 20-day MA)
            data['ma20'] = data['close'].rolling(window=20).mean()
            ranging = (data['close'] > data['ma20'] * 0.95) & (data['close'] < data['ma20'] * 1.05)
            return data[ranging]
            
        elif condition == 'volatile':
            # Find periods of high volatility (example: ATR > 2x its moving average)
            data['tr'] = self._calculate_tr(data)
            data['atr14'] = data['tr'].rolling(window=14).mean()
            data['atr_ma'] = data['atr14'].rolling(window=30).mean()
            volatile = data['atr14'] > (data['atr_ma'] * 2)
            return data[volatile]
            
        elif condition == 'breakout':
            # Find breakout periods (example: price breaking out of a 20-day range)
            data['high20'] = data['high'].rolling(window=20).max()
            data['low20'] = data['low'].rolling(window=20).min()
            breakout_up = data['close'] > data['high20'].shift(1)
            breakout_down = data['close'] < data['low20'].shift(1)
            return data[breakout_up | breakout_down]
            
        elif condition == 'low_volatility':
            # Find periods of low volatility
            data['tr'] = self._calculate_tr(data)
            data['atr14'] = data['tr'].rolling(window=14).mean()
            data['atr_ma'] = data['atr14'].rolling(window=30).mean()
            low_vol = data['atr14'] < (data['atr_ma'] * 0.5)
            return data[low_vol]
            
        else:
            # Unknown condition, return empty dataframe
            logger.warning(f"Unknown market condition: {condition}")
            return pd.DataFrame()
    
    def _calculate_tr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range for volatility measurement."""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift(1)).abs()
        low_close = (data['low'] - data['close'].shift(1)).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range
