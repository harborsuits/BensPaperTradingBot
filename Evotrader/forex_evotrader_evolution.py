#!/usr/bin/env python3
"""
Forex EvoTrader Evolution - Strategy Evolution Module

This module handles the evolution of forex trading strategies with:
- Session-aware fitness evaluation
- Pip-based metrics
- News event filtering
- BenBot integration for decision making
"""

import os
import sys
import yaml
import json
import logging
import datetime
import pandas as pd
import numpy as np
import uuid
import random
from typing import Dict, List, Tuple, Union, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_evotrader_evolution')


class ForexStrategyEvolution:
    """
    Manages the evolution of forex trading strategies with session awareness and pip-based metrics.
    """
    
    def __init__(self, forex_evotrader, config: Dict[str, Any] = None):
        """
        Initialize the forex strategy evolution module.
        
        Args:
            forex_evotrader: Parent ForexEvoTrader instance
            config: Evolution configuration (if None, uses parent config)
        """
        self.forex_evotrader = forex_evotrader
        
        # Use parent config if none provided
        if config is None and hasattr(forex_evotrader, 'config'):
            config = forex_evotrader.config
        
        self.config = config or {}
        self.evolution_config = self.config.get('evolution', {})
        
        # Access parent components
        self.pair_manager = getattr(forex_evotrader, 'pair_manager', None)
        self.news_guard = getattr(forex_evotrader, 'news_guard', None)
        self.session_manager = getattr(forex_evotrader, 'session_manager', None)
        self.pip_logger = getattr(forex_evotrader, 'pip_logger', None)
        self.performance_tracker = getattr(forex_evotrader, 'performance_tracker', None)
        
        # Templates
        self.strategy_templates = self._load_strategy_templates()
        
        logger.info("Forex Strategy Evolution module initialized")
    
    def _load_strategy_templates(self) -> Dict[str, Any]:
        """Load strategy templates from the configuration file."""
        template_path = self.config.get('strategy_templates', 'forex_strategy_templates.yaml')
        
        try:
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    templates = yaml.safe_load(f)
                logger.info(f"Loaded {len(templates)} strategy templates from {template_path}")
                return templates
            else:
                logger.warning(f"Strategy templates file not found: {template_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading strategy templates: {e}")
            return {}
    
    def evolve_strategy(self,
                       pair: str,
                       timeframe: str,
                       template_name: str,
                       session_focus: Optional[str] = None,
                       max_generations: Optional[int] = None,
                       data: Optional[pd.DataFrame] = None,
                       strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evolve a forex trading strategy based on the specified template.
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe for the strategy (e.g., '1h', '4h')
            template_name: Name of the strategy template to use
            session_focus: Specific session to optimize for (optional)
            max_generations: Maximum generations to run (optional)
            data: OHLCV data for backtest (optional, will load if not provided)
            strategy_name: Custom name for the strategy (optional)
            
        Returns:
            Dictionary with evolution results
        """
        # Validate inputs
        if template_name not in self.strategy_templates:
            logger.error(f"Template not found: {template_name}")
            return {'error': f"Template not found: {template_name}"}
        
        # Get template
        template = self.strategy_templates[template_name]
        logger.info(f"Using template: {template_name} - {template.get('description', 'No description')}")
        
        # Check pair validity
        if self.pair_manager and not self.pair_manager.get_pair(pair):
            logger.error(f"Invalid pair: {pair}")
            return {'error': f"Invalid pair: {pair}"}
        
        # Check if timeframe is supported by the template
        if 'timeframes' in template and timeframe not in template['timeframes']:
            logger.warning(f"Timeframe {timeframe} not in recommended timeframes for template: {template['timeframes']}")
        
        # Generate strategy ID and name
        if not strategy_name:
            strategy_name = f"{template_name.capitalize()}_{pair}_{timeframe}"
        
        strategy_id = self.forex_evotrader.get_strategy_id(strategy_name, template_name, pair, timeframe)
        
        # Set evolution parameters
        population_size = self.evolution_config.get('population_size', 50)
        generations = max_generations or self.evolution_config.get('generations', 20)
        crossover_rate = self.evolution_config.get('crossover_rate', 0.7)
        mutation_rate = self.evolution_config.get('mutation_rate', 0.3)
        elite_size = self.evolution_config.get('elite_size', 5)
        
        # Set session parameters
        if not session_focus:
            session_focus = self.config.get('sessions', {}).get('default_focus', 'London-NewYork')
            
        logger.info(f"Starting evolution for {strategy_name} ({strategy_id})")
        logger.info(f"Parameters: {pair} {timeframe}, pop={population_size}, gens={generations}, session={session_focus}")
        
        # Check with BenBot if available
        if hasattr(self.forex_evotrader, 'benbot_available') and self.forex_evotrader.benbot_available:
            benbot_response = self.forex_evotrader._consult_benbot('evolve', {
                'pair': pair,
                'timeframe': timeframe,
                'template': template_name,
                'session_focus': session_focus,
                'strategy_id': strategy_id
            })
            
            if not benbot_response.get('proceed', True):
                logger.warning(f"BenBot rejected evolution: {benbot_response.get('message', 'No reason')}")
                return {'error': f"BenBot rejected evolution: {benbot_response.get('message', 'No reason')}"}
        
        # Load or verify data
        if data is None:
            data = self._load_market_data(pair, timeframe)
            if data is None or len(data) < 1000:  # Minimum data requirement
                return {'error': f"Insufficient data for {pair} {timeframe}"}
        
        # Label data with sessions
        if self.session_manager and 'session_London' not in data.columns:
            data = self.session_manager.label_dataframe_sessions(data)
            logger.info(f"Added session labels to data: {[col for col in data.columns if 'session_' in col]}")
        
        # Initialize population
        initial_population = self._create_initial_population(
            template, population_size, pair, timeframe
        )
        
        # Run evolution
        evolution_results = self._run_evolution(
            initial_population=initial_population,
            data=data,
            pair=pair,
            timeframe=timeframe,
            template=template,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            session_focus=session_focus,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size
        )
        
        # Store results
        results_file = self.forex_evotrader.save_evolution_results(evolution_results, strategy_id)
        
        # Register top strategies with session performance tracker
        if self.performance_tracker and 'top_strategies' in evolution_results:
            for i, strat in enumerate(evolution_results['top_strategies'][:3]):  # Register top 3
                if 'session_performance' in strat:
                    for session, metrics in strat['session_performance'].items():
                        self.performance_tracker.db.update_session_performance(
                            strategy_id=f"{strategy_id}_{i}",
                            strategy_name=f"{strategy_name} Gen {strat.get('generation', 'Final')}",
                            session=session,
                            metrics=metrics
                        )
        
        logger.info(f"Evolution completed: {strategy_name} ({strategy_id})")
        return evolution_results
    
    def _load_market_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load market data for the specified pair and timeframe.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        data_dir = self.config.get('data_dir', 'data')
        filepath = f"{data_dir}/{pair}_{timeframe}.csv"
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                logger.info(f"Loaded {len(data)} rows of {pair} {timeframe} data from {filepath}")
                return data
            except Exception as e:
                logger.error(f"Error loading data: {e}")
        else:
            logger.warning(f"Data file not found: {filepath}")
            
        return None
    
    def _create_initial_population(self, 
                                 template: Dict[str, Any], 
                                 population_size: int,
                                 pair: str,
                                 timeframe: str) -> List[Dict[str, Any]]:
        """
        Create initial population of strategies based on template.
        
        Args:
            template: Strategy template dictionary
            population_size: Size of population to create
            pair: Currency pair
            timeframe: Timeframe
            
        Returns:
            List of strategy parameter dictionaries
        """
        population = []
        
        # Get parameter ranges from template
        param_ranges = template.get('ranges', {})
        default_params = template.get('parameters', {})
        
        # Generate random strategies
        for i in range(population_size):
            strategy_params = default_params.copy()
            
            # Randomize parameters within ranges
            for param, range_values in param_ranges.items():
                if param in strategy_params:
                    if isinstance(range_values, list) and len(range_values) == 2:
                        min_val, max_val = range_values
                        
                        # Handle different parameter types
                        if isinstance(default_params[param], int):
                            strategy_params[param] = random.randint(min_val, max_val)
                        elif isinstance(default_params[param], float):
                            strategy_params[param] = min_val + random.random() * (max_val - min_val)
                        elif isinstance(default_params[param], bool):
                            strategy_params[param] = random.choice([True, False])
                        elif isinstance(default_params[param], str):
                            strategy_params[param] = random.choice(range_values)
            
            # Add metadata
            strategy = {
                'id': f"{i}",
                'template': template.get('type', 'unknown'),
                'pair': pair,
                'timeframe': timeframe,
                'parameters': strategy_params,
                'fitness': 0.0,
                'generation': 0
            }
            
            population.append(strategy)
        
        logger.info(f"Created initial population of {len(population)} strategies")
        return population
    
    def _run_evolution(self,
                      initial_population: List[Dict[str, Any]],
                      data: pd.DataFrame,
                      pair: str,
                      timeframe: str,
                      template: Dict[str, Any],
                      strategy_id: str,
                      strategy_name: str,
                      session_focus: str,
                      generations: int,
                      crossover_rate: float,
                      mutation_rate: float,
                      elite_size: int) -> Dict[str, Any]:
        """
        Run the evolutionary process.
        
        Args:
            initial_population: List of initial strategies
            data: Market data
            pair: Currency pair
            timeframe: Timeframe
            template: Strategy template
            strategy_id: Strategy ID
            strategy_name: Strategy name
            session_focus: Session to focus optimization on
            generations: Number of generations to run
            crossover_rate: Crossover rate
            mutation_rate: Mutation rate
            elite_size: Number of elites to preserve
            
        Returns:
            Dictionary with evolution results
        """
        # Initialize tracking
        population = initial_population
        best_fitness_history = []
        avg_fitness_history = []
        best_strategy = None
        
        # Split data for training and validation
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        validation_data = data.iloc[train_size:]
        
        # Track when to stop
        early_stopping = self.evolution_config.get('early_stopping_generations', 5)
        generations_without_improvement = 0
        
        # Evolution loop
        for generation in range(generations):
            logger.info(f"Generation {generation+1}/{generations}")
            
            # Evaluate fitness in parallel
            population = self._evaluate_population_fitness(
                population, train_data, pair, template, session_focus
            )
            
            # Sort by fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Track best and average fitness
            best_fitness = population[0]['fitness']
            avg_fitness = sum(s['fitness'] for s in population) / len(population)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            # Check for improvement
            if generation > 0 and best_fitness <= best_fitness_history[-2]:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0
                best_strategy = population[0].copy()
                best_strategy['generation'] = generation + 1
            
            logger.info(f"Gen {generation+1}: Best fitness = {best_fitness:.4f}, Avg fitness = {avg_fitness:.4f}")
            
            # Early stopping check
            if generations_without_improvement >= early_stopping:
                logger.info(f"Early stopping after {generation+1} generations (no improvement for {early_stopping} gens)")
                break
            
            # Select parents, crossover, and mutation for next generation
            if generation < generations - 1:
                next_generation = []
                
                # Keep elites
                elites = population[:elite_size]
                for elite in elites:
                    next_generation.append(elite.copy())
                
                # Create offspring
                while len(next_generation) < len(population):
                    # Tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover with probability
                    if random.random() < crossover_rate:
                        child = self._crossover(parent1, parent2)
                    else:
                        child = parent1.copy()
                    
                    # Mutation with probability
                    if random.random() < mutation_rate:
                        child = self._mutate(child, template)
                    
                    # Update metadata
                    child['id'] = f"{generation+1}_{len(next_generation)}"
                    child['generation'] = generation + 1
                    child['fitness'] = 0.0  # Reset fitness
                    
                    next_generation.append(child)
                
                # Update population
                population = next_generation
        
        # Final evaluation on validation data
        logger.info("Evaluating top strategies on validation data...")
        top_strategies = population[:10]  # Evaluate top 10
        
        for strat in top_strategies:
            validation_result = self._evaluate_strategy(
                strat, validation_data, pair, template, session_focus
            )
            strat['validation'] = validation_result
        
        # Sort by validation fitness
        top_strategies.sort(key=lambda x: x.get('validation', {}).get('fitness', 0), reverse=True)
        
        # Compile results
        results = {
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'pair': pair,
            'timeframe': timeframe,
            'template': template.get('type', 'unknown'),
            'session_focus': session_focus,
            'generations_ran': generation + 1,
            'population_size': len(initial_population),
            'early_stopping': early_stopping,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'best_strategy': best_strategy,
            'top_strategies': top_strategies,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return results
    
    def _evaluate_population_fitness(self,
                                   population: List[Dict[str, Any]],
                                   data: pd.DataFrame,
                                   pair: str,
                                   template: Dict[str, Any],
                                   session_focus: str) -> List[Dict[str, Any]]:
        """
        Evaluate fitness for all strategies in population.
        
        Args:
            population: List of strategies
            data: Market data
            pair: Currency pair
            template: Strategy template
            session_focus: Session to focus optimization on
            
        Returns:
            List of strategies with updated fitness
        """
        max_workers = self.evolution_config.get('max_concurrent_processes', 8)
        results = []
        
        # Use parallel processing if available
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._evaluate_strategy, strat, data, pair, template, session_focus
                    )
                    for strat in population
                ]
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        population[i]['fitness'] = result['fitness']
                        population[i].update(result)
                    except Exception as e:
                        logger.error(f"Error evaluating strategy {population[i]['id']}: {e}")
                        population[i]['fitness'] = 0.0
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing")
            
            # Sequential fallback
            for i, strat in enumerate(population):
                try:
                    result = self._evaluate_strategy(strat, data, pair, template, session_focus)
                    strat['fitness'] = result['fitness']
                    strat.update(result)
                except Exception as e:
                    logger.error(f"Error evaluating strategy {strat['id']}: {e}")
                    strat['fitness'] = 0.0
        
        return population
    
    def _evaluate_strategy(self,
                         strategy: Dict[str, Any],
                         data: pd.DataFrame,
                         pair: str,
                         template: Dict[str, Any],
                         session_focus: str) -> Dict[str, Any]:
        """
        Evaluate a single strategy and calculate fitness.
        
        Args:
            strategy: Strategy dictionary
            data: Market data
            pair: Currency pair
            template: Strategy template
            session_focus: Session to focus optimization on
            
        Returns:
            Dictionary with evaluation results including fitness
        """
        # This method would typically instantiate the strategy class based on template
        # and run a backtest. For simplicity, we'll use a stub implementation.
        
        # In a real implementation, this would:
        # 1. Create strategy instance with parameters
        # 2. Run backtest with session awareness
        # 3. Calculate pip-based metrics
        # 4. Calculate session-specific performance
        # 5. Apply prop firm compliance checks
        # 6. Calculate final fitness score
        
        # For now, we'll simulate this with random fitness scores
        backtest_result = {
            'total_trades': random.randint(50, 200),
            'win_rate': random.uniform(0.4, 0.7),
            'profit_factor': random.uniform(1.0, 3.0),
            'sharpe_ratio': random.uniform(0.5, 2.5),
            'max_drawdown': random.uniform(0.05, 0.25),
            'total_pips': random.uniform(100, 1000),
            'avg_pips_per_trade': random.uniform(5, 30),
            'session_performance': {
                'London': {
                    'trades': random.randint(20, 80),
                    'win_rate': random.uniform(0.4, 0.7),
                    'profit_factor': random.uniform(1.0, 3.0),
                    'total_pips': random.uniform(50, 400)
                },
                'NewYork': {
                    'trades': random.randint(20, 80),
                    'win_rate': random.uniform(0.4, 0.7),
                    'profit_factor': random.uniform(1.0, 3.0),
                    'total_pips': random.uniform(50, 400)
                },
                'Asia': {
                    'trades': random.randint(20, 80),
                    'win_rate': random.uniform(0.4, 0.7),
                    'profit_factor': random.uniform(1.0, 3.0),
                    'total_pips': random.uniform(50, 400)
                }
            }
        }
        
        # Calculate fitness with emphasis on session focus
        session_fitness_weight = self.evolution_config.get('session_fitness_weight', 0.3)
        pip_fitness_weight = self.evolution_config.get('pip_fitness_weight', 0.7)
        
        # Base fitness components
        win_rate_component = backtest_result['win_rate'] * 0.3
        profit_factor_component = min(backtest_result['profit_factor'] / 3.0, 1.0) * 0.3
        drawdown_component = (1.0 - backtest_result['max_drawdown'] / 0.25) * 0.2
        sharpe_component = min(backtest_result['sharpe_ratio'] / 2.5, 1.0) * 0.2
        
        base_fitness = win_rate_component + profit_factor_component + drawdown_component + sharpe_component
        
        # Session-specific fitness
        session_fitness = 0.0
        if session_focus in backtest_result['session_performance']:
            session_data = backtest_result['session_performance'][session_focus]
            session_fitness = (
                (session_data['win_rate'] * 0.4) +
                (min(session_data['profit_factor'] / 3.0, 1.0) * 0.6)
            )
        elif '-' in session_focus:  # Handle overlap sessions like "London-NewYork"
            sessions = session_focus.split('-')
            if all(s in backtest_result['session_performance'] for s in sessions):
                # Average of both sessions
                session_values = [backtest_result['session_performance'][s] for s in sessions]
                session_fitness = (
                    (sum(s['win_rate'] for s in session_values) / len(sessions) * 0.4) +
                    (min(sum(s['profit_factor'] for s in session_values) / len(sessions) / 3.0, 1.0) * 0.6)
                )
        
        # Combine base and session fitness
        fitness = (base_fitness * (1.0 - session_fitness_weight)) + (session_fitness * session_fitness_weight)
        
        # Add pip-based component
        pip_fitness = min(backtest_result['total_pips'] / 1000.0, 1.0)
        fitness = (fitness * (1.0 - pip_fitness_weight)) + (pip_fitness * pip_fitness_weight)
        
        # Store all results
        result = backtest_result.copy()
        result['fitness'] = fitness
        
        return result
    
    def _tournament_selection(self, population: List[Dict[str, Any]], tournament_size: int = 3) -> Dict[str, Any]:
        """
        Select a parent using tournament selection.
        
        Args:
            population: List of strategies
            tournament_size: Number of strategies to compare
            
        Returns:
            Selected strategy
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create child strategy by combining parameters from two parents.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Child strategy
        """
        child = parent1.copy()
        child_params = child['parameters'].copy()
        
        # Randomly select parameters from each parent
        for param in child_params:
            if param in parent2['parameters'] and random.random() < 0.5:
                child_params[param] = parent2['parameters'][param]
        
        child['parameters'] = child_params
        return child
    
    def _mutate(self, strategy: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate strategy parameters.
        
        Args:
            strategy: Strategy to mutate
            template: Strategy template with parameter ranges
            
        Returns:
            Mutated strategy
        """
        mutated = strategy.copy()
        params = mutated['parameters'].copy()
        param_ranges = template.get('ranges', {})
        
        # Select random parameter to mutate
        param_to_mutate = random.choice(list(params.keys()))
        
        if param_to_mutate in param_ranges:
            range_values = param_ranges[param_to_mutate]
            
            if isinstance(range_values, list) and len(range_values) == 2:
                min_val, max_val = range_values
                
                # Mutate based on parameter type
                if isinstance(params[param_to_mutate], int):
                    params[param_to_mutate] = random.randint(min_val, max_val)
                elif isinstance(params[param_to_mutate], float):
                    params[param_to_mutate] = min_val + random.random() * (max_val - min_val)
                elif isinstance(params[param_to_mutate], bool):
                    params[param_to_mutate] = random.choice([True, False])
                elif isinstance(params[param_to_mutate], str):
                    params[param_to_mutate] = random.choice(range_values)
        
        mutated['parameters'] = params
        return mutated
