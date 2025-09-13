#!/usr/bin/env python
"""
Regime Parameter Optimizer

Script for optimizing strategy parameters for different market regimes 
using genetic algorithms or grid search.
"""

import os
import sys
import argparse
import logging
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import regime components and backtest engine
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.metrics import calculate_performance_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("regime_optimizer")

# Default parameter ranges for different strategy types
DEFAULT_PARAM_RANGES = {
    'trend_following': {
        'entry_threshold': (0.1, 5.0),
        'exit_threshold': (0.1, 5.0),
        'lookback_period': (5, 200),
        'stop_loss': (0.5, 10.0),
        'take_profit': (0.5, 15.0),
        'trailing_stop': (0.0, 10.0)
    },
    'mean_reversion': {
        'overbought_level': (65, 90),
        'oversold_level': (10, 35),
        'lookback_period': (2, 50),
        'stop_loss': (0.5, 10.0),
        'take_profit': (0.5, 10.0),
        'entry_threshold': (0.5, 5.0)
    },
    'breakout': {
        'breakout_period': (5, 100),
        'confirmation_bars': (1, 5),
        'stop_loss': (0.5, 8.0),
        'take_profit': (1.0, 12.0),
        'atr_multiplier': (0.5, 5.0)
    },
    'support_resistance': {
        'sr_lookback': (10, 200),
        'zone_width': (0.1, 2.0),
        'min_touches': (2, 5),
        'stop_loss': (0.5, 8.0),
        'take_profit': (1.0, 12.0)
    },
    'volatility_breakout': {
        'atr_period': (5, 50),
        'atr_multiplier': (0.5, 5.0),
        'lookback_period': (5, 100),
        'stop_loss': (0.5, 8.0),
        'take_profit': (1.0, 12.0)
    }
}

# Fitness metrics weights (higher means more important)
FITNESS_WEIGHTS = {
    'sharpe_ratio': 3.0,     # Risk-adjusted returns
    'profit_factor': 2.0,    # Ratio of gross profits to gross losses
    'win_rate': 1.0,         # Percentage of winning trades
    'max_drawdown': -2.0,    # Maximum drawdown (negative weight means lower is better)
    'recovery_factor': 1.0,  # Net profit divided by max drawdown
    'expectancy': 2.0,       # Average profit per trade with respect to risk
    'avg_trade': 1.0,        # Average profit per trade
    'trades_per_day': 0.5    # Number of trades per day (moderate weight)
}

class ParameterOptimizer:
    """
    Optimizes strategy parameters for different market regimes 
    using genetic algorithms.
    """
    
    def __init__(self, strategy_id: str, data_dir: str, output_dir: str, config: Optional[Dict] = None):
        """
        Initialize the parameter optimizer.
        
        Args:
            strategy_id: ID of the strategy to optimize
            data_dir: Directory containing historical data
            output_dir: Directory to save optimized parameters
            config: Configuration parameters
        """
        self.strategy_id = strategy_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(max_workers=self.config.get('max_workers', 4))
        
        # Set parameter ranges based on strategy type
        self.param_ranges = DEFAULT_PARAM_RANGES.get(strategy_id, {})
        if not self.param_ranges:
            logger.warning(f"No default parameter ranges found for strategy {strategy_id}")
            self.param_ranges = {}
        
        # Override with custom ranges if provided
        if 'param_ranges' in self.config:
            self.param_ranges.update(self.config['param_ranges'])
        
        # Genetic algorithm settings
        self.generations = self.config.get('generations', 10)
        self.population_size = self.config.get('population_size', 20)
        self.crossover_prob = self.config.get('crossover_prob', 0.7)
        self.mutation_prob = self.config.get('mutation_prob', 0.2)
        self.tournament_size = self.config.get('tournament_size', 3)
        
        # Fitness metrics
        self.fitness_weights = self.config.get('fitness_weights', FITNESS_WEIGHTS)
        
        logger.info(f"Initialized Parameter Optimizer for strategy {strategy_id}")
        logger.info(f"Parameter ranges: {self.param_ranges}")
    
    def load_regime_data(self, regime_type: MarketRegimeType, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data for a specific market regime.
        
        Args:
            regime_type: Type of market regime
            symbol: Symbol to load data for, or None for all symbols
            
        Returns:
            DataFrame with historical data for the regime
        """
        logger.info(f"Loading data for regime {regime_type.value}")
        
        # Get regime data directory
        regime_dir = os.path.join(self.data_dir, "regime_data")
        
        # Get list of regime data files
        if symbol:
            regime_files = [os.path.join(regime_dir, f) for f in os.listdir(regime_dir) 
                           if f.startswith(f"{symbol}_") and f.endswith("_regimes.csv")]
        else:
            regime_files = [os.path.join(regime_dir, f) for f in os.listdir(regime_dir) 
                           if f.endswith("_regimes.csv")]
        
        if not regime_files:
            raise ValueError(f"No regime data found for {regime_type.value}")
        
        # Load regimes and filter by regime type
        all_data = []
        
        for regime_file in regime_files:
            # Get corresponding price data
            filename = os.path.basename(regime_file)
            symbol_tf = filename.replace("_regimes.csv", "")
            price_file = os.path.join(self.data_dir, "historical_data", f"{symbol_tf}.csv")
            
            if not os.path.exists(price_file):
                logger.warning(f"Price data not found for {symbol_tf}")
                continue
            
            # Load regime data
            try:
                regime_df = pd.read_csv(regime_file, index_col=0, parse_dates=True)
                
                # Filter by regime type
                regime_df = regime_df[regime_df['regime'] == regime_type.value]
                
                if regime_df.empty:
                    logger.debug(f"No {regime_type.value} regime data for {symbol_tf}")
                    continue
                
                # Load price data
                price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
                
                # Extract date ranges for this regime
                segments = []
                
                for _, row in regime_df.iterrows():
                    # Find start and end index in price data
                    try:
                        date = pd.to_datetime(row.name)
                        # Go back 200 bars to provide enough data for indicators
                        idx = price_df.index.get_indexer([date], method='nearest')[0]
                        start_idx = max(0, idx - 200)
                        
                        # Get at least 50 bars after the regime point
                        end_idx = min(len(price_df), idx + 50)
                        
                        # Extract segment
                        segment = price_df.iloc[start_idx:end_idx].copy()
                        segment['symbol'] = symbol_tf.split('_')[0]
                        segment['timeframe'] = symbol_tf.split('_')[1]
                        
                        segments.append(segment)
                    except Exception as e:
                        logger.error(f"Error extracting segment for {date}: {str(e)}")
                
                # Combine segments
                if segments:
                    all_data.extend(segments)
            
            except Exception as e:
                logger.error(f"Error loading {regime_file}: {str(e)}")
        
        if not all_data:
            raise ValueError(f"No data segments found for regime {regime_type.value}")
        
        # Combine all segments
        combined_df = pd.concat(all_data)
        logger.info(f"Loaded {len(combined_df)} data points across {len(all_data)} segments for {regime_type.value}")
        
        return combined_df
    
    def create_individual(self) -> List[float]:
        """
        Create a random individual (parameter set) based on parameter ranges.
        
        Returns:
            List of parameter values
        """
        individual = []
        
        for param_name, (min_val, max_val) in self.param_ranges.items():
            # Integer parameters
            if param_name in ['lookback_period', 'breakout_period', 'sr_lookback', 
                             'confirmation_bars', 'min_touches', 'atr_period']:
                value = random.randint(min_val, max_val)
            # Float parameters with higher precision
            elif param_name in ['entry_threshold', 'exit_threshold', 'atr_multiplier', 'zone_width']:
                value = round(random.uniform(min_val, max_val), 2)
            # Other float parameters
            else:
                value = round(random.uniform(min_val, max_val), 1)
            
            individual.append(value)
        
        return individual
    
    def individual_to_params(self, individual: List[float]) -> Dict[str, Any]:
        """
        Convert an individual (list of values) to a parameter dictionary.
        
        Args:
            individual: List of parameter values
            
        Returns:
            Dictionary mapping parameter names to values
        """
        return {name: individual[i] for i, name in enumerate(self.param_ranges.keys())}
    
    def run_backtest(self, params: Dict[str, Any], data: pd.DataFrame) -> Dict[str, float]:
        """
        Run a backtest with the given parameters.
        
        Args:
            params: Strategy parameters
            data: Historical data
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Create strategy instance
            # In a real implementation, this would call your backtest engine
            # For now, we'll simulate it with random metrics
            
            # Configure backtest
            config = {
                'strategy_id': self.strategy_id,
                'params': params,
                'initial_capital': 10000.0,
                'commission': 0.001
            }
            
            # Run backtest
            results = self.backtest_engine.run_backtest(data, config)
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(results)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            # Return poor fitness values
            return {
                'sharpe_ratio': -1.0,
                'profit_factor': 0.5,
                'win_rate': 0.3,
                'max_drawdown': -20.0,
                'recovery_factor': 0.1,
                'expectancy': -0.5,
                'avg_trade': -1.0,
                'trades_per_day': 0.1
            }
    
    def calculate_fitness(self, individual: List[float], data: pd.DataFrame) -> Tuple[float,]:
        """
        Calculate fitness of an individual.
        
        Args:
            individual: List of parameter values
            data: Historical data
            
        Returns:
            Tuple with single fitness value
        """
        # Convert individual to parameter dictionary
        params = self.individual_to_params(individual)
        
        # Run backtest
        metrics = self.run_backtest(params, data)
        
        # Calculate weighted fitness
        fitness = 0.0
        for metric, weight in self.fitness_weights.items():
            if metric in metrics:
                fitness += metrics[metric] * weight
        
        return (fitness,)
    
    def optimize_parameters(self, regime_type: MarketRegimeType, 
                           data: Optional[pd.DataFrame] = None,
                           symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize parameters for a specific market regime using genetic algorithm.
        
        Args:
            regime_type: Type of market regime
            data: Historical data, or None to load from disk
            symbols: List of symbols to optimize for, or None for all
            
        Returns:
            Dictionary with optimized parameters
        """
        logger.info(f"Optimizing parameters for {self.strategy_id} in {regime_type.value} regime")
        
        # Load data if not provided
        if data is None:
            data = self.load_regime_data(regime_type, symbols[0] if symbols else None)
        
        # Setup genetic algorithm
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=self.population_size)
        toolbox.register("evaluate", self.calculate_fitness, data=data)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Create initial population
        population = toolbox.population()
        
        # Track best individual
        hof = tools.HallOfFame(1)
        
        # Stats to track
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Run genetic algorithm
        logger.info(f"Running genetic algorithm with {self.generations} generations and {self.population_size} individuals")
        
        population, logbook = algorithms.eaSimple(
            population, 
            toolbox, 
            cxpb=self.crossover_prob, 
            mutpb=self.mutation_prob, 
            ngen=self.generations, 
            stats=stats, 
            halloffame=hof, 
            verbose=True
        )
        
        # Get best parameters
        best_individual = hof[0]
        best_params = self.individual_to_params(best_individual)
        best_fitness = best_individual.fitness.values[0]
        
        logger.info(f"Best fitness: {best_fitness}")
        logger.info(f"Best parameters: {best_params}")
        
        # Run final backtest with best parameters
        metrics = self.run_backtest(best_params, data)
        
        # Generate results
        result = {
            'strategy_id': self.strategy_id,
            'regime': regime_type.value,
            'parameters': best_params,
            'fitness': best_fitness,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Plot convergence
        self._plot_convergence(logbook, regime_type)
        
        return result
    
    def _plot_convergence(self, logbook, regime_type):
        """Plot the convergence of the genetic algorithm."""
        try:
            gen = logbook.select("gen")
            fit_max = logbook.select("max")
            fit_avg = logbook.select("avg")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(gen, fit_max, 'b-', label='Maximum Fitness')
            ax.plot(gen, fit_avg, 'r-', label='Average Fitness')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title(f'Fitness Convergence - {self.strategy_id} - {regime_type.value}')
            ax.legend()
            ax.grid(True)
            
            # Save plot
            plot_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'{self.strategy_id}_{regime_type.value}_convergence.png')
            plt.savefig(plot_path)
            plt.close(fig)
            
            logger.info(f"Convergence plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating convergence plot: {str(e)}")
    
    def optimize_all_regimes(self, regimes: Optional[List[MarketRegimeType]] = None, 
                            symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize parameters for all market regimes (or specified regimes).
        
        Args:
            regimes: List of regimes to optimize for, or None for all
            symbols: List of symbols to optimize for, or None for all
            
        Returns:
            Dictionary mapping regimes to optimized parameters
        """
        logger.info(f"Optimizing parameters for {self.strategy_id} across all regimes")
        
        # Use all regimes if not specified
        if regimes is None:
            regimes = list(MarketRegimeType)
        
        # Optimize for each regime
        results = {}
        
        for regime in regimes:
            try:
                result = self.optimize_parameters(regime, symbols=symbols)
                results[regime.value] = result
            except Exception as e:
                logger.error(f"Error optimizing for {regime.value}: {str(e)}")
        
        # Save results
        self.save_parameters(results)
        
        return results
    
    def save_parameters(self, results: Dict[str, Any]) -> str:
        """
        Save optimized parameters to a JSON file.
        
        Args:
            results: Dictionary mapping regimes to optimized parameters
            
        Returns:
            Path to the saved file
        """
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{self.strategy_id}_parameters.json")
        
        # Extract just the parameters for each regime
        parameters = {}
        
        for regime, result in results.items():
            parameters[regime] = result['parameters']
        
        # Save parameters
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        logger.info(f"Saved optimized parameters to {output_file}")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"{self.strategy_id}_optimization_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved detailed optimization results to {results_file}")
        
        return output_file


def optimize_parameters(strategy_id: str, regime_type: MarketRegimeType, 
                       historical_data: pd.DataFrame, param_ranges: Dict[str, Tuple[float, float]],
                       generations: int = 10, population_size: int = 20) -> Dict[str, Any]:
    """
    Uses genetic algorithm to optimize strategy parameters for a specific regime.
    
    Args:
        strategy_id: ID of the strategy to optimize
        regime_type: Market regime type
        historical_data: Historical price data
        param_ranges: Dictionary mapping parameter names to (min, max) ranges
        generations: Number of generations for genetic algorithm
        population_size: Population size for genetic algorithm
        
    Returns:
        Dictionary with optimized parameters
    """
    # Create optimizer with config
    config = {
        'param_ranges': param_ranges,
        'generations': generations,
        'population_size': population_size
    }
    
    # Create temporary directories
    temp_data_dir = 'data/temp_optimization'
    os.makedirs(temp_data_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_data_dir, 'historical_data'), exist_ok=True)
    os.makedirs(os.path.join(temp_data_dir, 'regime_data'), exist_ok=True)
    
    # Create optimizer
    optimizer = ParameterOptimizer(strategy_id, temp_data_dir, 'data/market_regime/parameters', config)
    
    # Run optimization
    result = optimizer.optimize_parameters(regime_type, data=historical_data)
    
    # Extract parameters
    optimized_params = result['parameters']
    
    return optimized_params


def optimize_all_regimes(strategy_id: str, data_dir: str, output_dir: str, 
                        regimes: Optional[List[str]] = None,
                        symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Optimize parameters for all market regimes for a given strategy.
    
    Args:
        strategy_id: ID of the strategy to optimize
        data_dir: Directory with historical data
        output_dir: Output directory for parameters
        regimes: List of regime types to optimize for, or None for all
        symbols: List of symbols to optimize for, or None for all
        
    Returns:
        Dictionary mapping regimes to optimized parameters
    """
    # Create optimizer
    optimizer = ParameterOptimizer(strategy_id, data_dir, output_dir)
    
    # Convert regime strings to enum values
    if regimes:
        regime_types = [MarketRegimeType(r) for r in regimes]
    else:
        regime_types = None
    
    # Run optimization
    results = optimizer.optimize_all_regimes(regime_types, symbols)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Regime Parameter Optimizer')
    parser.add_argument('--strategy', required=True, help='Strategy ID to optimize')
    parser.add_argument('--data-dir', default='data/market_regime', help='Directory with historical data')
    parser.add_argument('--output-dir', default='data/market_regime/parameters', help='Output directory for parameters')
    parser.add_argument('--regimes', help='Comma-separated list of regimes to optimize')
    parser.add_argument('--symbols', help='Comma-separated list of symbols to optimize for')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations for genetic algorithm')
    parser.add_argument('--population', type=int, default=20, help='Population size for genetic algorithm')
    parser.add_argument('--param-ranges', help='JSON file with parameter ranges')
    
    args = parser.parse_args()
    
    # Parse regimes if provided
    regimes = None
    if args.regimes:
        regimes = args.regimes.split(',')
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = args.symbols.split(',')
    
    # Load custom parameter ranges if provided
    param_ranges = None
    if args.param_ranges:
        try:
            with open(args.param_ranges, 'r') as f:
                param_ranges = json.load(f)
        except Exception as e:
            logger.error(f"Error loading parameter ranges: {str(e)}")
    
    # Create config
    config = {
        'generations': args.generations,
        'population_size': args.population
    }
    
    if param_ranges:
        config['param_ranges'] = param_ranges
    
    # Create optimizer
    optimizer = ParameterOptimizer(args.strategy, args.data_dir, args.output_dir, config)
    
    # Convert regime strings to enum values if provided
    regime_types = None
    if regimes:
        try:
            regime_types = [MarketRegimeType(r) for r in regimes]
        except ValueError as e:
            logger.error(f"Invalid regime: {str(e)}")
            print(f"Valid regimes: {[r.value for r in MarketRegimeType]}")
            return 1
    
    # Run optimization
    optimizer.optimize_all_regimes(regime_types, symbols)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
