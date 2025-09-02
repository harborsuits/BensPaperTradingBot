#!/usr/bin/env python3
"""
Practical Optimization Example

This script demonstrates how different optimization algorithms (genetic,
simulated annealing, multi-objective) can be applied to trading strategy
parameter optimization across different market regimes.

It compares the performance of each approach and visualizes the tradeoffs.
"""

import os
import sys
import json
import math
import random
import time
import datetime
from typing import Dict, List, Any, Tuple, Callable, Optional
from enum import Enum
from copy import deepcopy

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed for reproducibility
random.seed(42)

# Define market regimes
class MarketRegime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


# Define parameter types
class ParameterType(str, Enum):
    REAL = "real"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


# Parameter space definition
class ParameterSpace:
    """Parameter space for optimization."""
    
    def __init__(self):
        """Initialize an empty parameter space."""
        self.parameters = []
        self.param_names = []
        self.param_types = {}
        self.bounds = {}
        self.defaults = {}
        self.categories = {}
    
    def add_real_parameter(self, name, lower_bound, upper_bound, default=None):
        """Add a real-valued parameter."""
        if default is None:
            default = (lower_bound + upper_bound) / 2.0
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.REAL
        self.bounds[name] = (lower_bound, upper_bound)
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.REAL.value,
            'bounds': (lower_bound, upper_bound),
            'default': default
        })
        
        return self
    
    def add_integer_parameter(self, name, lower_bound, upper_bound, default=None):
        """Add an integer parameter."""
        if default is None:
            default = (lower_bound + upper_bound) // 2
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.INTEGER
        self.bounds[name] = (lower_bound, upper_bound)
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.INTEGER.value,
            'bounds': (lower_bound, upper_bound),
            'default': default
        })
        
        return self
    
    def add_categorical_parameter(self, name, categories, default=None):
        """Add a categorical parameter."""
        if default is None and categories:
            default = categories[0]
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.CATEGORICAL
        self.categories[name] = categories
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.CATEGORICAL.value,
            'categories': categories,
            'default': default
        })
        
        return self
    
    def add_boolean_parameter(self, name, default=False):
        """Add a boolean parameter."""
        self.param_names.append(name)
        self.param_types[name] = ParameterType.BOOLEAN
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.BOOLEAN.value,
            'default': default
        })
        
        return self
    
    def get_default_parameters(self):
        """Get default parameters as a dictionary."""
        return {name: self.defaults[name] for name in self.param_names}
    
    def get_random_parameters(self, count=1):
        """Generate random parameter combinations."""
        result = []
        
        for _ in range(count):
            params = {}
            
            for param in self.parameters:
                name = param['name']
                param_type = param['type']
                
                if param_type == ParameterType.REAL.value:
                    lower, upper = param['bounds']
                    params[name] = lower + random.random() * (upper - lower)
                    
                elif param_type == ParameterType.INTEGER.value:
                    lower, upper = param['bounds']
                    params[name] = random.randint(lower, upper)
                    
                elif param_type == ParameterType.CATEGORICAL.value:
                    categories = param['categories']
                    params[name] = random.choice(categories)
                    
                elif param_type == ParameterType.BOOLEAN.value:
                    params[name] = random.choice([True, False])
            
            result.append(params)
        
        return result[0] if count == 1 else result
    
    def __len__(self):
        """Get number of parameters."""
        return len(self.param_names)


# Synthetic trading history generator
class SyntheticMarket:
    """Simple synthetic market generator for optimization testing."""
    
    def __init__(self, seed=None):
        """Initialize the market generator with optional seed."""
        if seed is not None:
            random.seed(seed)
        
        self.regime_generators = {
            MarketRegime.BULLISH: self._generate_bullish,
            MarketRegime.BEARISH: self._generate_bearish,
            MarketRegime.SIDEWAYS: self._generate_sideways,
            MarketRegime.VOLATILE: self._generate_volatile
        }
    
    def _generate_bullish(self, days=252, base_price=100.0):
        """Generate a bullish price series."""
        drift = 0.0008  # Higher positive drift
        volatility = 0.012  # Moderate volatility
        return self._generate_random_walk(days, base_price, volatility, drift)
    
    def _generate_bearish(self, days=252, base_price=100.0):
        """Generate a bearish price series."""
        drift = -0.0007  # Negative drift
        volatility = 0.015  # Slightly higher volatility
        return self._generate_random_walk(days, base_price, volatility, drift)
    
    def _generate_sideways(self, days=252, base_price=100.0):
        """Generate a sideways/ranging price series."""
        drift = 0.0001  # Very small drift
        volatility = 0.008  # Lower volatility
        return self._generate_random_walk(days, base_price, volatility, drift)
    
    def _generate_volatile(self, days=252, base_price=100.0):
        """Generate a volatile price series."""
        drift = 0.0002  # Small drift
        volatility = 0.025  # High volatility
        return self._generate_random_walk(days, base_price, volatility, drift)
    
    def _generate_random_walk(self, days, base_price, volatility, drift):
        """Generate a basic random walk price series."""
        daily_returns = [random.normalvariate(drift, volatility) for _ in range(days)]
        prices = [base_price]
        
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        return prices
    
    def generate_price_series(self, regime=MarketRegime.BULLISH, days=252, base_price=100.0):
        """Generate a price series for the specified market regime."""
        if regime in self.regime_generators:
            return self.regime_generators[regime](days, base_price)
        else:
            # Default to sideways if regime not recognized
            return self._generate_sideways(days, base_price)


# Strategy simulation
class TradingStrategy:
    """Trading strategy to optimize."""
    
    def __init__(self, parameters):
        """Initialize the trading strategy with parameters."""
        self.parameters = parameters
        
        # Extract parameters
        self.trend_lookback = parameters.get('trend_lookback', 20)
        self.volatility_lookback = parameters.get('volatility_lookback', 10)
        self.entry_threshold = parameters.get('entry_threshold', 0.5)
        self.exit_threshold = parameters.get('exit_threshold', -0.2)
        self.stop_loss_pct = parameters.get('stop_loss_pct', 2.0)
        self.position_sizing = parameters.get('position_sizing', 0.1)
        self.trail_stop = parameters.get('trail_stop', True)
        self.filter_type = parameters.get('filter_type', 'sma')
        self.indicator_weight = parameters.get('indicator_weight', 0.5)
        
        # Internal state
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0
        self.stop_price = 0
        self.equity_curve = []
        self.trades = []
    
    def reset(self):
        """Reset the strategy state."""
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0
        self.stop_price = 0
        self.equity_curve = []
        self.trades = []
    
    def calculate_trend(self, prices):
        """Calculate trend indicator."""
        if len(prices) < self.trend_lookback + 1:
            return 0
        
        if self.filter_type == 'sma':
            # Simple Moving Average crossover
            short_ma = sum(prices[-10:]) / 10
            long_ma = sum(prices[-self.trend_lookback:]) / self.trend_lookback
            return (short_ma / long_ma - 1) * 100
        
        elif self.filter_type == 'ema':
            # Exponential Moving Average (simplified)
            alpha_short = 2 / (10 + 1)
            alpha_long = 2 / (self.trend_lookback + 1)
            
            short_ema = prices[-10]
            for i in range(9):
                short_ema = alpha_short * prices[-9+i] + (1 - alpha_short) * short_ema
            
            long_ema = prices[-self.trend_lookback]
            for i in range(self.trend_lookback - 1):
                long_ema = alpha_long * prices[-self.trend_lookback+1+i] + (1 - alpha_long) * long_ema
            
            return (short_ema / long_ema - 1) * 100
        
        elif self.filter_type == 'momentum':
            # Simple momentum
            return (prices[-1] / prices[-self.trend_lookback] - 1) * 100
        
        else:
            # Default to price change if filter type not recognized
            return (prices[-1] / prices[-2] - 1) * 100
    
    def calculate_volatility(self, prices):
        """Calculate historical volatility."""
        if len(prices) < self.volatility_lookback + 1:
            return 0
        
        # Calculate daily returns
        returns = [(prices[i] / prices[i-1] - 1) for i in range(-self.volatility_lookback, 0)]
        
        # Calculate standard deviation of returns (volatility)
        mean_return = sum(returns) / len(returns)
        variance = sum([(r - mean_return) ** 2 for r in returns]) / len(returns)
        
        return (variance ** 0.5) * 100  # Convert to percentage
    
    def generate_signal(self, prices):
        """Generate trading signal based on strategy parameters."""
        trend = self.calculate_trend(prices)
        volatility = self.calculate_volatility(prices)
        
        # Combine indicators
        signal = trend * self.indicator_weight + (1 - trend/volatility) * (1 - self.indicator_weight)
        
        return signal
    
    def update_stops(self, current_price):
        """Update stop loss levels."""
        if not self.position:
            return
        
        # Update highest seen price if in a long position
        if self.position > 0 and current_price > self.highest_price:
            self.highest_price = current_price
            
            # Update trailing stop if enabled
            if self.trail_stop:
                self.stop_price = self.highest_price * (1 - self.stop_loss_pct / 100)
    
    def execute_trade(self, signal, price, cash):
        """Execute a trade based on signal and update equity."""
        if self.position == 0 and signal > self.entry_threshold:
            # Enter long position
            self.position = cash * self.position_sizing / price
            self.entry_price = price
            self.highest_price = price
            self.stop_price = price * (1 - self.stop_loss_pct / 100)
            
            self.trades.append({
                'type': 'entry',
                'price': price,
                'size': self.position,
                'value': self.position * price
            })
            
        elif self.position > 0:
            # Check for exit conditions
            stop_triggered = price <= self.stop_price
            signal_exit = signal < self.exit_threshold
            
            if stop_triggered or signal_exit:
                # Exit position
                exit_value = self.position * price
                entry_value = self.position * self.entry_price
                pnl = exit_value - entry_value
                
                self.trades.append({
                    'type': 'exit',
                    'price': price,
                    'size': self.position,
                    'value': exit_value,
                    'pnl': pnl,
                    'pnl_pct': (price / self.entry_price - 1) * 100,
                    'reason': 'stop_loss' if stop_triggered else 'signal'
                })
                
                self.position = 0
                self.entry_price = 0
                self.highest_price = 0
    
    def calculate_equity(self, price, initial_cash=10000):
        """Calculate current equity value."""
        if self.position == 0:
            if not self.equity_curve:
                return initial_cash
            return self.equity_curve[-1]
        
        position_value = self.position * price
        
        if not self.equity_curve:
            # First equity point is initial cash + position value - cost
            cost = self.position * self.entry_price
            return initial_cash - cost + position_value
        
        # Return previous equity plus change in position value
        prev_price = self.trades[-1]['price'] if self.trades else self.entry_price
        price_change = price / prev_price - 1
        position_change = self.position * prev_price * price_change
        
        return self.equity_curve[-1] + position_change
    
    def run_backtest(self, prices, initial_cash=10000):
        """Run a backtest of the strategy on price series."""
        self.reset()
        cash = initial_cash
        self.equity_curve = [cash]
        
        # Need enough bars for lookback
        min_bars = max(self.trend_lookback, self.volatility_lookback) + 1
        
        for i in range(min_bars, len(prices)):
            price_window = prices[:i]
            current_price = prices[i-1]
            
            # Generate signal
            signal = self.generate_signal(price_window)
            
            # Update stops for existing positions
            self.update_stops(current_price)
            
            # Execute any trades
            self.execute_trade(signal, current_price, cash)
            
            # Calculate and record equity
            equity = self.calculate_equity(current_price, initial_cash)
            self.equity_curve.append(equity)
        
        # Ensure we close any open positions at the end
        if self.position > 0:
            final_price = prices[-1]
            exit_value = self.position * final_price
            entry_value = self.position * self.entry_price
            pnl = exit_value - entry_value
            
            self.trades.append({
                'type': 'exit',
                'price': final_price,
                'size': self.position,
                'value': exit_value,
                'pnl': pnl,
                'pnl_pct': (final_price / self.entry_price - 1) * 100,
                'reason': 'end_of_period'
            })
            
            self.position = 0
            
            # Update final equity point
            self.equity_curve[-1] = self.calculate_equity(final_price, initial_cash)
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self):
        """Calculate performance metrics from backtest results."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trade_count': 0
            }
        
        # Calculate returns
        initial_equity = self.equity_curve[0]
        final_equity = self.equity_curve[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        # Calculate Sharpe ratio (simplified, assuming 0% risk-free rate)
        daily_returns = [(self.equity_curve[i] / self.equity_curve[i-1] - 1) 
                         for i in range(1, len(self.equity_curve))]
        
        if not daily_returns:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trade_count': 0
            }
        
        mean_return = sum(daily_returns) / len(daily_returns)
        std_return = math.sqrt(sum([(r - mean_return) ** 2 for r in daily_returns]) / len(daily_returns))
        
        sharpe_ratio = mean_return / std_return * math.sqrt(252) if std_return > 0 else 0
        
        # Calculate maximum drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0
        
        for equity in self.equity_curve:
            # Update peak
            if equity > peak:
                peak = equity
            
            # Calculate drawdown
            drawdown = (peak - equity) / peak * 100
            
            # Update max drawdown
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate win rate and profit factor
        win_count = sum(1 for trade in self.trades if trade.get('type') == 'exit' and trade.get('pnl', 0) > 0)
        loss_count = sum(1 for trade in self.trades if trade.get('type') == 'exit' and trade.get('pnl', 0) <= 0)
        
        total_profit = sum(trade.get('pnl', 0) for trade in self.trades 
                           if trade.get('type') == 'exit' and trade.get('pnl', 0) > 0)
        total_loss = abs(sum(trade.get('pnl', 0) for trade in self.trades 
                             if trade.get('type') == 'exit' and trade.get('pnl', 0) <= 0))
        
        trade_count = win_count + loss_count
        win_rate = win_count / trade_count * 100 if trade_count > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_count': trade_count
        }


# Optimization algorithms
class GeneticOptimizer:
    """Genetic algorithm optimizer."""
    
    def __init__(
        self,
        parameter_space,
        population_size=30,
        generations=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism=True,
        tournament_size=3,
        minimize=False
    ):
        """Initialize the genetic algorithm optimizer."""
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.minimize = minimize
        
        # Population and fitness tracking
        self.population = []
        self.fitness = []
        self.best_individual = None
        self.best_fitness = float('-inf') if not minimize else float('inf')
    
    def initialize_population(self):
        """Initialize population with random individuals."""
        self.population = []
        for _ in range(self.population_size):
            individual = self.parameter_space.get_random_parameters()
            self.population.append(individual)
    
    def evaluate_population(self, fitness_function):
        """Evaluate fitness of all individuals in the population."""
        self.fitness = []
        
        for individual in self.population:
            try:
                fitness = fitness_function(individual)
            except Exception:
                # Assign worst possible fitness if evaluation fails
                fitness = float('-inf') if not self.minimize else float('inf')
            
            self.fitness.append(fitness)
            
            # Update best individual if better
            is_better = fitness > self.best_fitness if not self.minimize else fitness < self.best_fitness
            if is_better:
                self.best_individual = deepcopy(individual)
                self.best_fitness = fitness
    
    def select_parent(self):
        """Select parent using tournament selection."""
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
        tournament_fitness = [self.fitness[i] for i in tournament_indices]
        
        # Get best individual in tournament
        if not self.minimize:
            best_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        else:
            best_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        
        return deepcopy(self.population[best_idx])
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create two offspring."""
        # Decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Initialize offspring with parent parameters
        offspring1 = {}
        offspring2 = {}
        
        # For each parameter, decide source parent
        for param_name in self.parameter_space.param_names:
            if random.random() < 0.5:
                # Inherit from opposite parents
                offspring1[param_name] = parent1[param_name]
                offspring2[param_name] = parent2[param_name]
            else:
                # Inherit from opposite parents
                offspring1[param_name] = parent2[param_name]
                offspring2[param_name] = parent1[param_name]
        
        return offspring1, offspring2
    
    def mutate(self, individual):
        """Apply mutation to an individual."""
        mutated = deepcopy(individual)
        
        for param_name in self.parameter_space.param_names:
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                param_type = self.parameter_space.param_types[param_name]
                
                if param_type == ParameterType.REAL:
                    # Mutate real parameter
                    lower, upper = self.parameter_space.bounds[param_name]
                    range_size = upper - lower
                    mutation_amount = random.gauss(0, range_size * 0.1)  # 10% of range as std dev
                    mutated[param_name] += mutation_amount
                    # Ensure within bounds
                    mutated[param_name] = max(lower, min(upper, mutated[param_name]))
                
                elif param_type == ParameterType.INTEGER:
                    # Mutate integer parameter
                    lower, upper = self.parameter_space.bounds[param_name]
                    range_size = upper - lower
                    mutation_amount = random.randint(-max(1, int(range_size * 0.1)), 
                                                   max(1, int(range_size * 0.1)))
                    mutated[param_name] += mutation_amount
                    # Ensure within bounds
                    mutated[param_name] = max(lower, min(upper, mutated[param_name]))
                
                elif param_type == ParameterType.CATEGORICAL:
                    # Mutate categorical parameter
                    categories = self.parameter_space.categories[param_name]
                    # Only mutate if more than one category
                    if len(categories) > 1:
                        current = mutated[param_name]
                        # Get all categories except current
                        other_categories = [c for c in categories if c != current]
                        # Select a random different category
                        mutated[param_name] = random.choice(other_categories)
                
                elif param_type == ParameterType.BOOLEAN:
                    # Flip boolean parameter
                    mutated[param_name] = not mutated[param_name]
        
        return mutated
    
    def create_next_generation(self):
        """Create the next generation through selection, crossover and mutation."""
        new_population = []
        
        # Apply elitism if enabled (keep best individual)
        if self.elitism and self.best_individual is not None:
            new_population.append(deepcopy(self.best_individual))
        
        # Create the rest of the new population
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            # Perform crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)
            
            # Perform mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # Update population
        self.population = new_population[:self.population_size]
    
    def optimize(self, fitness_function, callback=None):
        """Run the genetic algorithm optimization process.
        
        Args:
            fitness_function: Function to evaluate fitness of individuals
            callback: Optional callback function called after each generation
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        # Track optimization progress
        best_history = []
        avg_history = []
        
        # Main optimization loop
        for generation in range(self.generations):
            # Evaluate current population
            self.evaluate_population(fitness_function)
            
            # Track progress
            if not self.minimize:
                avg_fitness = sum(self.fitness) / len(self.fitness)
                best_fitness = max(self.fitness)
            else:
                avg_fitness = sum(self.fitness) / len(self.fitness)
                best_fitness = min(self.fitness)
                
            best_history.append(best_fitness)
            avg_history.append(avg_fitness)
            
            # Report progress
            print(f"Generation {generation+1}/{self.generations}, "
                  f"Best Fitness: {best_fitness:.4f}, "
                  f"Avg Fitness: {avg_fitness:.4f}")
            
            # Call callback if provided
            if callback:
                callback(generation, self.best_individual, self.best_fitness)
            
            # Create next generation (except for last generation)
            if generation < self.generations - 1:
                self.create_next_generation()
        
        # Return results
        elapsed_time = time.time() - start_time
        
        return {
            'best_params': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations': self.generations,
            'best_history': best_history,
            'avg_history': avg_history,
            'runtime': elapsed_time
        }


class SimulatedAnnealingOptimizer:
    """Simulated annealing optimizer."""
    
    def __init__(
        self,
        parameter_space,
        initial_temp=100.0,
        cooling_rate=0.95,
        n_steps_per_temp=10,
        min_temp=1e-10,
        adaptive_step_size=True,
        neighborhood_size=0.1,
        minimize=False,
        cooling_schedule="exponential"
    ):
        """Initialize simulated annealing optimizer."""
        self.parameter_space = parameter_space
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_steps_per_temp = n_steps_per_temp
        self.min_temp = min_temp
        self.adaptive_step_size = adaptive_step_size
        self.neighborhood_size = neighborhood_size
        self.minimize = minimize
        self.cooling_schedule = cooling_schedule
        
        # Current state
        self.current_solution = None
        self.current_value = float('-inf') if not minimize else float('inf')
        self.best_solution = None
        self.best_value = float('-inf') if not minimize else float('inf')
    
    def initialize(self):
        """Initialize the optimizer."""
        # Start from default parameters
        self.current_solution = self.parameter_space.get_default_parameters()
        
        # Reset temperature
        self.current_temp = self.initial_temp
    
    def _generate_neighbor(self, solution):
        """Generate a neighboring solution."""
        neighbor = deepcopy(solution)
        
        # Choose a random parameter to modify
        param_name = random.choice(self.parameter_space.param_names)
        param_type = self.parameter_space.param_types[param_name]
        
        if param_type == ParameterType.REAL:
            # Modify real parameter
            lower, upper = self.parameter_space.bounds[param_name]
            range_size = upper - lower
            
            # Determine step size
            if self.adaptive_step_size:
                # Step size decreases with temperature
                step_size = range_size * self.neighborhood_size * (self.current_temp / self.initial_temp)
            else:
                step_size = range_size * self.neighborhood_size
            
            # Apply Gaussian perturbation
            new_value = neighbor[param_name] + random.gauss(0, step_size)
            neighbor[param_name] = max(lower, min(upper, new_value))
            
        elif param_type == ParameterType.INTEGER:
            # Modify integer parameter
            lower, upper = self.parameter_space.bounds[param_name]
            range_size = upper - lower
            
            # Determine step size
            if self.adaptive_step_size:
                # Step size decreases with temperature
                step_size = max(1, int(range_size * self.neighborhood_size * 
                                      (self.current_temp / self.initial_temp)))
            else:
                step_size = max(1, int(range_size * self.neighborhood_size))
            
            # Apply random step
            step = random.randint(-step_size, step_size)
            new_value = neighbor[param_name] + step
            neighbor[param_name] = max(lower, min(upper, new_value))
            
        elif param_type == ParameterType.CATEGORICAL:
            # Modify categorical parameter
            categories = self.parameter_space.categories[param_name]
            if len(categories) > 1:
                # Select a different category
                current_category = neighbor[param_name]
                available = [c for c in categories if c != current_category]
                if available:
                    neighbor[param_name] = random.choice(available)
                
        elif param_type == ParameterType.BOOLEAN:
            # Flip boolean parameter
            neighbor[param_name] = not neighbor[param_name]
        
        return neighbor
    
    def _acceptance_probability(self, current_value, new_value):
        """Calculate acceptance probability."""
        # For maximization, we want to move to higher values
        # For minimization, we want to move to lower values
        if (not self.minimize and new_value > current_value) or \
           (self.minimize and new_value < current_value):
            # Always accept better solutions
            return 1.0
        else:
            # Calculate difference (always positive)
            if self.minimize:
                delta = new_value - current_value
            else:
                delta = current_value - new_value
            
            # Calculate acceptance probability
            # Lower temperatures make accepting worse solutions less likely
            return math.exp(-delta / self.current_temp)
    
    def _cool_temperature(self, iteration, n_iterations):
        """Update temperature according to cooling schedule."""
        if self.cooling_schedule == "exponential":
            self.current_temp *= self.cooling_rate
        elif self.cooling_schedule == "linear":
            self.current_temp = self.initial_temp - \
                              (self.initial_temp - self.min_temp) * \
                              (iteration / n_iterations)
        elif self.cooling_schedule == "logarithmic":
            self.current_temp = self.initial_temp / (1 + math.log(1 + iteration))
        else:
            # Default to exponential
            self.current_temp *= self.cooling_rate
        
        # Ensure temperature doesn't go below minimum
        self.current_temp = max(self.current_temp, self.min_temp)
    
    def optimize(self, fitness_function, n_iterations=100, callback=None):
        """Run the optimization process."""
        start_time = time.time()
        
        # Initialize
        self.initialize()
        
        # Evaluate initial solution
        try:
            self.current_value = fitness_function(self.current_solution)
        except Exception:
            if self.minimize:
                self.current_value = float('inf')
            else:
                self.current_value = float('-inf')
        
        # Set initial solution as best solution
        self.best_solution = deepcopy(self.current_solution)
        self.best_value = self.current_value
        
        # Track progress
        value_history = [self.current_value]
        temp_history = [self.current_temp]
        
        # Main annealing loop
        iteration = 0
        accepted_count = 0
        total_evaluations = 0
        
        while iteration < n_iterations and self.current_temp > self.min_temp:
            accepted_at_this_temp = 0
            
            # Steps at current temperature
            for step in range(self.n_steps_per_temp):
                # Generate neighbor
                neighbor = self._generate_neighbor(self.current_solution)
                
                # Evaluate neighbor
                try:
                    neighbor_value = fitness_function(neighbor)
                except Exception:
                    if self.minimize:
                        neighbor_value = float('inf')
                    else:
                        neighbor_value = float('-inf')
                
                total_evaluations += 1
                
                # Determine if we should accept the neighbor
                accept_prob = self._acceptance_probability(self.current_value, neighbor_value)
                
                if random.random() < accept_prob:
                    # Accept neighbor
                    self.current_solution = neighbor
                    self.current_value = neighbor_value
                    accepted_at_this_temp += 1
                    accepted_count += 1
                    
                    # Update best solution if needed
                    if (not self.minimize and neighbor_value > self.best_value) or \
                       (self.minimize and neighbor_value < self.best_value):
                        self.best_solution = deepcopy(neighbor)
                        self.best_value = neighbor_value
            
            # Track progress
            value_history.append(self.best_value)
            temp_history.append(self.current_temp)
            
            # Report progress
            acceptance_rate = accepted_at_this_temp / self.n_steps_per_temp
            print(f"Iteration {iteration+1}/{n_iterations}, "
                  f"Temp: {self.current_temp:.4f}, "
                  f"Best: {self.best_value:.4f}, "
                  f"Accept Rate: {acceptance_rate:.2f}")
            
            # Call callback if provided
            if callback:
                callback(iteration, self.best_solution, self.best_value, self.current_temp)
            
            # Cool temperature
            self._cool_temperature(iteration, n_iterations)
            
            iteration += 1
        
        # Final verification of best solution
        try:
            final_value = fitness_function(self.best_solution)
            if (not self.minimize and final_value > self.best_value) or \
               (self.minimize and final_value < self.best_value):
                self.best_value = final_value
        except Exception:
            pass
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        return {
            'best_params': self.best_solution,
            'best_fitness': self.best_value,
            'iterations': iteration,
            'evaluations': total_evaluations,
            'acceptance_rate': accepted_count / total_evaluations if total_evaluations > 0 else 0,
            'fitness_history': value_history,
            'temperature_history': temp_history,
            'runtime': elapsed_time
        }


# Multi-objective optimization support functions
def dominates(solution1, solution2, objectives):
    """Check if solution1 dominates solution2.
    
    Args:
        solution1: First solution with objective values
        solution2: Second solution with objective values
        objectives: List of objective names
        
    Returns:
        True if solution1 dominates solution2, False otherwise
    """
    better_in_any = False
    worse_in_any = False
    
    for obj in objectives:
        val1 = solution1['objectives'][obj]
        val2 = solution2['objectives'][obj]
        
        if val1 < val2:
            worse_in_any = True
        elif val1 > val2:
            better_in_any = True
    
    return better_in_any and not worse_in_any


def fast_non_dominated_sort(population, objectives):
    """Perform fast non-dominated sorting as in NSGA-II.
    
    Args:
        population: List of solutions with objective values
        objectives: List of objective names
        
    Returns:
        List of fronts, where each front is a list of solution indices
    """
    n = len(population)
    if n == 0:
        return [[]]
        
    S = [[] for _ in range(n)]
    n_dominated = [0 for _ in range(n)]
    fronts = [[]]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            if dominates(population[i], population[j], objectives):
                S[i].append(j)
            elif dominates(population[j], population[i], objectives):
                n_dominated[i] += 1
        
        if n_dominated[i] == 0:
            fronts[0].append(i)
    
    if not fronts[0]:
        fronts[0] = list(range(n))
        return fronts
    
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        
        for j in fronts[i]:
            for k in S[j]:
                n_dominated[k] -= 1
                
                if n_dominated[k] == 0:
                    next_front.append(k)
        
        i += 1
        
        if next_front:
            fronts.append(next_front)
    
    return fronts


def calculate_crowding_distance(front, population, objectives):
    """Calculate crowding distance for solutions in a Pareto front.
    
    Args:
        front: List of solution indices in the front
        population: List of solutions with objective values
        objectives: List of objective names
        
    Returns:
        List of crowding distances for each solution in the front
    """
    n = len(front)
    if n <= 2:
        return [float('inf') for _ in range(n)]
    
    distances = [0.0 for _ in range(n)]
    
    for obj in objectives:
        front_sorted = sorted(front, key=lambda i: population[i]['objectives'][obj])
        
        distances[front.index(front_sorted[0])] = float('inf')
        distances[front.index(front_sorted[-1])] = float('inf')
        
        obj_min = population[front_sorted[0]]['objectives'][obj]
        obj_max = population[front_sorted[-1]]['objectives'][obj]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        for i in range(1, n - 1):
            idx = front.index(front_sorted[i])
            prev_val = population[front_sorted[i - 1]]['objectives'][obj]
            next_val = population[front_sorted[i + 1]]['objectives'][obj]
            
            distance = (next_val - prev_val) / obj_range
            distances[idx] += distance
    
    return distances


def tournament_selection(population, tournament_size, fronts, crowding_distances):
    """Select a solution using tournament selection based on Pareto ranking and crowding distance."""
    candidates = random.sample(range(len(population)), tournament_size)
    
    candidate_fronts = []
    for candidate in candidates:
        for i, front in enumerate(fronts):
            if candidate in front:
                candidate_fronts.append((candidate, i))
                break
    
    candidate_fronts.sort(key=lambda x: x[1])
    
    best_front = candidate_fronts[0][1]
    best_front_candidates = [c for c, f in candidate_fronts if f == best_front]
    
    if len(best_front_candidates) == 1:
        return best_front_candidates[0]
    
    best_candidate = max(best_front_candidates, key=lambda x: crowding_distances[x])
    
    return best_candidate


class MultiObjectiveOptimizer:
    """Multi-objective optimizer based on NSGA-II algorithm."""
    
    def __init__(
        self,
        parameter_space,
        population_size=50,
        generations=30,
        tournament_size=3,
        crossover_prob=0.9,
        mutation_prob=0.1,
        objectives=None
    ):
        """Initialize the multi-objective optimizer."""
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.objectives = objectives if objectives else []
        
        # Solutions and history
        self.population = []
        self.pareto_history = []
        self.best_solutions = []
    
    def initialize_population(self):
        """Initialize the population with random solutions."""
        self.population = []
        params_list = self.parameter_space.get_random_parameters(self.population_size)
        
        if not isinstance(params_list, list):
            params_list = [params_list]
        
        for params in params_list:
            solution = {
                'parameters': params,
                'objectives': {obj: 0.0 for obj in self.objectives}
            }
            self.population.append(solution)
    
    def evaluate_objectives(self, objective_functions):
        """Evaluate all objectives for the entire population."""
        for solution in self.population:
            try:
                for obj_name, obj_func in objective_functions.items():
                    solution['objectives'][obj_name] = obj_func(solution['parameters'])
            except Exception:
                for obj_name in self.objectives:
                    solution['objectives'][obj_name] = float('-inf')
    
    def select_parents(self, fronts, crowding_distances):
        """Select parents for crossover using tournament selection."""
        parent1 = tournament_selection(self.population, self.tournament_size, fronts, crowding_distances)
        parent2 = tournament_selection(self.population, self.tournament_size, fronts, crowding_distances)
        
        while parent2 == parent1:
            parent2 = tournament_selection(self.population, self.tournament_size, fronts, crowding_distances)
        
        return parent1, parent2
    
    def crossover(self, parent1_idx, parent2_idx):
        """Perform crossover between two parents."""
        parent1_params = deepcopy(self.population[parent1_idx]['parameters'])
        parent2_params = deepcopy(self.population[parent2_idx]['parameters'])
        
        offspring1_params = deepcopy(parent1_params)
        offspring2_params = deepcopy(parent2_params)
        
        if random.random() < self.crossover_prob:
            for param_name in self.parameter_space.param_names:
                if random.random() < 0.5:
                    offspring1_params[param_name] = parent2_params[param_name]
                    offspring2_params[param_name] = parent1_params[param_name]
        
        offspring1 = {
            'parameters': offspring1_params,
            'objectives': {obj: 0.0 for obj in self.objectives}
        }
        offspring2 = {
            'parameters': offspring2_params,
            'objectives': {obj: 0.0 for obj in self.objectives}
        }
        
        return offspring1, offspring2
    
    def mutate(self, solution):
        """Mutate a solution."""
        mutated_solution = deepcopy(solution)
        
        if random.random() < self.mutation_prob:
            param_name = random.choice(self.parameter_space.param_names)
            param_type = self.parameter_space.param_types[param_name]
            
            if param_type == ParameterType.REAL:
                lower, upper = self.parameter_space.bounds[param_name]
                range_size = upper - lower
                mutation = random.gauss(0, range_size * 0.1)
                current_value = mutated_solution['parameters'][param_name]
                new_value = current_value + mutation
                mutated_solution['parameters'][param_name] = max(lower, min(upper, new_value))
                
            elif param_type == ParameterType.INTEGER:
                lower, upper = self.parameter_space.bounds[param_name]
                range_size = upper - lower
                step = random.randint(-max(1, int(range_size * 0.1)), max(1, int(range_size * 0.1)))
                current_value = mutated_solution['parameters'][param_name]
                new_value = current_value + step
                mutated_solution['parameters'][param_name] = max(lower, min(upper, new_value))
                
            elif param_type == ParameterType.CATEGORICAL:
                categories = self.parameter_space.categories[param_name]
                if len(categories) > 1:
                    current_category = mutated_solution['parameters'][param_name]
                    available = [c for c in categories if c != current_category]
                    if available:
                        mutated_solution['parameters'][param_name] = random.choice(available)
                        
            elif param_type == ParameterType.BOOLEAN:
                mutated_solution['parameters'][param_name] = not mutated_solution['parameters'][param_name]
        
        return mutated_solution
    
    def create_next_generation(self, fronts, crowding_distances):
        """Create the next generation using selection, crossover, and mutation."""
        new_population = []
        
        while len(new_population) < self.population_size:
            parent1_idx, parent2_idx = self.select_parents(fronts, crowding_distances)
            
            offspring1, offspring2 = self.crossover(parent1_idx, parent2_idx)
            
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        return new_population
    
    def get_pareto_front(self):
        """Get the current Pareto front (non-dominated solutions)."""
        fronts = fast_non_dominated_sort(self.population, self.objectives)
        
        return [self.population[idx] for idx in fronts[0]]
    
    def optimize(self, objective_functions, callback=None):
        """Run the optimization process."""
        start_time = time.time()
        
        self.objectives = list(objective_functions.keys())
        
        self.initialize_population()
        
        self.evaluate_objectives(objective_functions)
        
        self.pareto_history = [self.get_pareto_front()]
        
        for generation in range(self.generations):
            fronts = fast_non_dominated_sort(self.population, self.objectives)
            
            all_crowding_distances = [0] * len(self.population)
            for front in fronts:
                front_crowding_distances = calculate_crowding_distance(
                    front, self.population, self.objectives
                )
                for i, idx in enumerate(front):
                    all_crowding_distances[idx] = front_crowding_distances[i]
            
            self.population = self.create_next_generation(fronts, all_crowding_distances)
            
            self.evaluate_objectives(objective_functions)
            
            self.pareto_history.append(self.get_pareto_front())
            
            if callback:
                callback(generation, self.get_pareto_front())
            
            front_size = len(fronts[0])
            avg_objectives = {}
            for obj in self.objectives:
                avg_objectives[obj] = sum(sol['objectives'][obj] for sol in self.population) / len(self.population)
            
            if generation % 5 == 0 or generation == self.generations - 1:
                print(f"Generation {generation+1}/{self.generations}, "
                      f"Pareto Front Size: {front_size}, "
                      f"Avg Objectives: {', '.join([f'{k}: {v:.4f}' for k, v in avg_objectives.items()])}")
        
        final_pareto_front = self.get_pareto_front()
        
        self.best_solutions = final_pareto_front
        
        optimization_time = time.time() - start_time
        
        return {
            'pareto_front': final_pareto_front,
            'pareto_history': self.pareto_history,
            'generations': self.generations,
            'runtime': optimization_time
        }


def main():
    """Main function to run the practical optimization example."""
    print("\n=== Practical Trading Strategy Optimization Across Different Market Regimes ===")
    
    # Create parameter space for the trading strategy
    param_space = ParameterSpace()
    param_space.add_integer_parameter('trend_lookback', 5, 50, 20)
    param_space.add_integer_parameter('volatility_lookback', 5, 30, 10)
    param_space.add_real_parameter('entry_threshold', 0.1, 2.0, 0.5)
    param_space.add_real_parameter('exit_threshold', -2.0, 0.0, -0.2)
    param_space.add_real_parameter('stop_loss_pct', 0.5, 5.0, 2.0)
    param_space.add_real_parameter('position_sizing', 0.01, 0.5, 0.1)
    param_space.add_boolean_parameter('trail_stop', True)
    param_space.add_categorical_parameter('filter_type', ['sma', 'ema', 'momentum'], 'sma')
    param_space.add_real_parameter('indicator_weight', 0.1, 0.9, 0.5)
    
    # Initialize synthetic market generator
    market = SyntheticMarket(seed=42)  # Use fixed seed for reproducibility
    
    # Generate price series for different market regimes
    print("\nGenerating synthetic price data for different market regimes...")
    bullish_prices = market.generate_price_series(MarketRegime.BULLISH, days=252)
    bearish_prices = market.generate_price_series(MarketRegime.BEARISH, days=252)
    sideways_prices = market.generate_price_series(MarketRegime.SIDEWAYS, days=252)
    volatile_prices = market.generate_price_series(MarketRegime.VOLATILE, days=252)
    
    print(f"Generated price series lengths: Bullish={len(bullish_prices)}, "
          f"Bearish={len(bearish_prices)}, Sideways={len(sideways_prices)}, "
          f"Volatile={len(volatile_prices)}")
    
    # Define fitness function for single-objective optimization
    def fitness_function(params):
        """Calculate combined fitness across all market regimes."""
        strategy = TradingStrategy(params)
        
        # Run backtests for different regimes
        bullish_metrics = strategy.run_backtest(bullish_prices)
        bearish_metrics = strategy.run_backtest(bearish_prices)
        sideways_metrics = strategy.run_backtest(sideways_prices)
        volatile_metrics = strategy.run_backtest(volatile_prices)
        
        # Calculate overall fitness (weighted by regime importance)
        # Here we balance:  returns (positive), sharpe ratio (positive), and drawdown (negative)
        fitness = (
            0.3 * (bullish_metrics['total_return'] + 
                  bearish_metrics['total_return'] + 
                  sideways_metrics['total_return'] + 
                  volatile_metrics['total_return']) / 4 +
            0.4 * (bullish_metrics['sharpe_ratio'] + 
                  bearish_metrics['sharpe_ratio'] + 
                  sideways_metrics['sharpe_ratio'] + 
                  volatile_metrics['sharpe_ratio']) / 4 -
            0.3 * (bullish_metrics['max_drawdown'] + 
                  bearish_metrics['max_drawdown'] + 
                  sideways_metrics['max_drawdown'] + 
                  volatile_metrics['max_drawdown']) / 4
        )
        
        return fitness
    
    # Define multi-objective functions for multi-objective optimization
    def bull_return(params):
        strategy = TradingStrategy(params)
        metrics = strategy.run_backtest(bullish_prices)
        return metrics['total_return']
    
    def bear_return(params):
        strategy = TradingStrategy(params)
        metrics = strategy.run_backtest(bearish_prices)
        return metrics['total_return']
    
    def drawdown(params):
        # Average drawdown across all regimes (negated for maximization)
        strategy = TradingStrategy(params)
        
        bull_dd = strategy.run_backtest(bullish_prices)['max_drawdown']
        bear_dd = strategy.run_backtest(bearish_prices)['max_drawdown']
        sideways_dd = strategy.run_backtest(sideways_prices)['max_drawdown']
        volatile_dd = strategy.run_backtest(volatile_prices)['max_drawdown']
        
        # Negate because we want to minimize drawdown (but optimizer maximizes)
        return -((bull_dd + bear_dd + sideways_dd + volatile_dd) / 4)
    
    # Define objective functions for multi-objective optimization
    multi_obj_functions = {
        'bull_return': bull_return,
        'bear_return': bear_return,
        'drawdown': drawdown  # already negated for maximization
    }
    
    # Run genetic algorithm optimization
    print("\n=== Genetic Algorithm Optimization ===\n")
    ga_optimizer = GeneticOptimizer(
        parameter_space=param_space,
        population_size=20,  # Smaller population for faster example
        generations=15,      # Fewer generations for faster example
        mutation_rate=0.2,
        crossover_rate=0.7,
        elitism=True,
        tournament_size=3,
        minimize=False       # We want to maximize fitness
    )
    
    ga_results = ga_optimizer.optimize(fitness_function)
    
    # Display GA results
    best_ga_params = ga_results['best_params']
    best_ga_fitness = ga_results['best_fitness']
    
    print(f"\nGA Best Fitness: {best_ga_fitness:.4f}")
    print(f"GA Best Parameters: {best_ga_params}")
    
    # Run simulated annealing optimization
    print("\n=== Simulated Annealing Optimization ===\n")
    sa_optimizer = SimulatedAnnealingOptimizer(
        parameter_space=param_space,
        initial_temp=100.0,
        cooling_rate=0.9,
        n_steps_per_temp=5,  # Fewer steps for faster example
        adaptive_step_size=True,
        minimize=False        # We want to maximize fitness
    )
    
    sa_results = sa_optimizer.optimize(fitness_function, n_iterations=20)  # Fewer iterations for faster example
    
    # Display SA results
    best_sa_params = sa_results['best_params']
    best_sa_fitness = sa_results['best_fitness']
    
    print(f"\nSA Best Fitness: {best_sa_fitness:.4f}")
    print(f"SA Best Parameters: {best_sa_params}")
    
    # Run multi-objective optimization
    print("\n=== Multi-Objective Optimization ===\n")
    mo_optimizer = MultiObjectiveOptimizer(
        parameter_space=param_space,
        population_size=20,    # Smaller population for faster example
        generations=15,        # Fewer generations for faster example
        tournament_size=3,
        crossover_prob=0.9,
        mutation_prob=0.2,
        objectives=list(multi_obj_functions.keys())
    )
    
    mo_results = mo_optimizer.optimize(multi_obj_functions)
    
    # Display multi-objective results
    pareto_front = mo_results['pareto_front']
    print(f"\nMulti-Objective Pareto Front Size: {len(pareto_front)}")
    
    print("\nSample solutions from Pareto front:")
    # Display a few diverse solutions from the Pareto front
    if pareto_front:
        # Find extreme solutions
        best_bull = max(pareto_front, key=lambda s: s['objectives']['bull_return'])
        best_bear = max(pareto_front, key=lambda s: s['objectives']['bear_return'])
        best_dd = max(pareto_front, key=lambda s: s['objectives']['drawdown'])
        
        # Display extreme solutions
        print("\nBest Bullish Solution:")
        print(f"Parameters: {best_bull['parameters']}")
        print(f"Objectives: Bull Return={best_bull['objectives']['bull_return']:.2f}%, "
              f"Bear Return={best_bull['objectives']['bear_return']:.2f}%, "
              f"Drawdown={-best_bull['objectives']['drawdown']:.2f}%")
        
        print("\nBest Bearish Solution:")
        print(f"Parameters: {best_bear['parameters']}")
        print(f"Objectives: Bull Return={best_bear['objectives']['bull_return']:.2f}%, "
              f"Bear Return={best_bear['objectives']['bear_return']:.2f}%, "
              f"Drawdown={-best_bear['objectives']['drawdown']:.2f}%")
        
        print("\nBest Low Drawdown Solution:")
        print(f"Parameters: {best_dd['parameters']}")
        print(f"Objectives: Bull Return={best_dd['objectives']['bull_return']:.2f}%, "
              f"Bear Return={best_dd['objectives']['bear_return']:.2f}%, "
              f"Drawdown={-best_dd['objectives']['drawdown']:.2f}%")
    
    # Compare performance of best solutions
    print("\n=== Performance Comparison Across Market Regimes ===\n")
    
    # Function to evaluate a set of parameters across all regimes
    def evaluate_across_regimes(params, label):
        strategy = TradingStrategy(params)
        
        bull_metrics = strategy.run_backtest(bullish_prices)
        bear_metrics = strategy.run_backtest(bearish_prices)
        sideways_metrics = strategy.run_backtest(sideways_prices)
        volatile_metrics = strategy.run_backtest(volatile_prices)
        
        print(f"{label} Performance:")
        print(f"  Bullish: Return={bull_metrics['total_return']:.2f}%, "
              f"Sharpe={bull_metrics['sharpe_ratio']:.2f}, "
              f"Drawdown={bull_metrics['max_drawdown']:.2f}%")
        print(f"  Bearish: Return={bear_metrics['total_return']:.2f}%, "
              f"Sharpe={bear_metrics['sharpe_ratio']:.2f}, "
              f"Drawdown={bear_metrics['max_drawdown']:.2f}%")
        print(f"  Sideways: Return={sideways_metrics['total_return']:.2f}%, "
              f"Sharpe={sideways_metrics['sharpe_ratio']:.2f}, "
              f"Drawdown={sideways_metrics['max_drawdown']:.2f}%")
        print(f"  Volatile: Return={volatile_metrics['total_return']:.2f}%, "
              f"Sharpe={volatile_metrics['sharpe_ratio']:.2f}, "
              f"Drawdown={volatile_metrics['max_drawdown']:.2f}%")
    
    # Evaluate the best solutions from each approach
    evaluate_across_regimes(best_ga_params, "GA Best")
    print()
    evaluate_across_regimes(best_sa_params, "SA Best")
    print()
    
    if pareto_front:
        evaluate_across_regimes(best_bull['parameters'], "MO Bullish Optimal")
        print()
        evaluate_across_regimes(best_bear['parameters'], "MO Bearish Optimal")
        print()
        evaluate_across_regimes(best_dd['parameters'], "MO Low Drawdown Optimal")
    
    # Summary of algorithmic approach comparison
    print("\n=== Optimization Approach Comparison ===\n")
    print(f"{'Approach':<25} {'Runtime':<10} {'Best Fitness/Note':<20}")
    print("-" * 55)
    print(f"{'Genetic Algorithm':<25} {ga_results['runtime']:.2f}s {best_ga_fitness:.4f}")
    print(f"{'Simulated Annealing':<25} {sa_results['runtime']:.2f}s {best_sa_fitness:.4f}")
    print(f"{'Multi-Objective':<25} {mo_results['runtime']:.2f}s {'Pareto front solutions'}")
    
    print("\nConclusion:")
    print("The multi-objective approach provides a range of solutions with different trade-offs,")
    print("while the single-objective approaches find solutions that maximize a combined fitness metric.")
    print("Each approach has strengths depending on the specific requirements of the trading system.")


if __name__ == "__main__":
    main()
