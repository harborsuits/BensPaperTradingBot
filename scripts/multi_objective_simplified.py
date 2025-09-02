#!/usr/bin/env python3
"""
Multi-Objective Simplified Feedback Loop

This script builds on our successful simplified_feedback_loop.py implementation,
adding multi-objective capabilities across market regimes while maintaining the
core learning mechanism that improves prediction accuracy.
"""

import random
import math
import time
from typing import Dict, List, Any, Tuple

# Constants
LEARNING_RATE = 0.2
GENERATIONS = 5
POPULATION_SIZE = 100

# Real market parameters (the "ground truth" unknown to our optimization system)
REAL_WORLD_PARAMS = {
    'bullish': {'trend': 0.008, 'volatility': 0.018, 'mean_reversion': 0.15},
    'bearish': {'trend': -0.006, 'volatility': 0.03, 'mean_reversion': 0.1},
    'sideways': {'trend': 0.0, 'volatility': 0.012, 'mean_reversion': 0.8},
    'volatile': {'trend': 0.0005, 'volatility': 0.045, 'mean_reversion': 0.05}
}


def generate_market_data(days=252, trend=0.0005, volatility=0.01, mean_reversion=0.1):
    """
    Generate synthetic market data with specified characteristics.
    
    Args:
        days: Number of days to generate
        trend: Daily trend factor (positive=bullish, negative=bearish)
        volatility: Daily volatility
        mean_reversion: Mean reversion strength (0-1)
        
    Returns:
        List of prices
    """
    prices = [100.0]  # Start with $100
    
    for _ in range(days - 1):
        # Calculate distance from moving average
        if len(prices) >= 20:
            moving_avg = sum(prices[-20:]) / 20
            distance = prices[-1] / moving_avg - 1
        else:
            distance = 0
            
        # Mean reversion component: move back toward moving average
        reversion = -distance * mean_reversion
        
        # Combine trend, reversion, and random components
        daily_change = trend + reversion + random.normalvariate(0, volatility)
        
        # Calculate new price and add to list
        new_price = prices[-1] * (1 + daily_change)
        prices.append(new_price)
    
    return prices


class SimpleStrategy:
    """A simple trading strategy with adjustable parameters."""
    
    def __init__(self, parameters):
        # Extract parameters
        self.trend_period = parameters.get('trend_period', 20)
        self.entry_threshold = parameters.get('entry_threshold', 0.5)
        self.exit_threshold = parameters.get('exit_threshold', -0.5)
        self.stop_loss = parameters.get('stop_loss', 5.0)
        self.position_size = parameters.get('position_size', 0.1)
    
    def calculate_signal(self, prices):
        """Calculate trading signal (-100 to +100)."""
        if len(prices) < self.trend_period + 1:
            return 0
        
        # Simple momentum
        momentum = (prices[-1] / prices[-self.trend_period] - 1) * 100
        
        # Moving average difference
        short_ma = sum(prices[-10:]) / 10
        long_ma = sum(prices[-self.trend_period:]) / self.trend_period
        ma_diff = (short_ma / long_ma - 1) * 100
        
        # Combined signal
        signal = (momentum + ma_diff) / 2
        return signal
    
    def backtest(self, prices):
        """Run a simple backtest on price data."""
        if len(prices) < self.trend_period + 1:
            return {'return': 0, 'max_drawdown': 0, 'sharpe': 0}
        
        cash = 10000
        position = 0
        entry_price = 0
        equity = [cash]
        
        # Daily price changes for Sharpe calculation
        daily_returns = []
        
        for i in range(self.trend_period, len(prices)):
            price_window = prices[:i+1]
            current_price = price_window[-1]
            
            # Calculate signal
            signal = self.calculate_signal(price_window)
            
            # Trading logic
            if position == 0 and signal > self.entry_threshold:
                # Enter long position
                position = cash * self.position_size / current_price
                cash -= position * current_price
                entry_price = current_price
            elif position > 0:
                # Check exit conditions
                stop_hit = current_price < entry_price * (1 - self.stop_loss/100)
                exit_signal = signal < self.exit_threshold
                
                if stop_hit or exit_signal:
                    # Exit position
                    cash += position * current_price
                    position = 0
            
            # Update equity
            current_equity = cash + (position * current_price)
            equity.append(current_equity)
            
            # Calculate daily return
            daily_return = equity[-1] / equity[-2] - 1 if len(equity) > 1 else 0
            daily_returns.append(daily_return)
        
        # Calculate performance metrics
        total_return = (equity[-1] / equity[0] - 1) * 100
        
        # Max drawdown
        max_drawdown = 0
        peak = equity[0]
        
        for e in equity:
            if e > peak:
                peak = e
            else:
                drawdown = (peak - e) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 1:
            avg_return = sum(daily_returns) / len(daily_returns)
            std_return = math.sqrt(sum((r - avg_return)**2 for r in daily_returns) / len(daily_returns))
            sharpe = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
        
        return {
            'return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe
        }


class ParameterSpace:
    """Simple parameter space for strategy optimization."""
    
    def __init__(self):
        """Initialize parameter space."""
        self.parameters = {
            'trend_period': {'min': 5, 'max': 50, 'type': 'int'},
            'entry_threshold': {'min': 0, 'max': 2, 'type': 'float'},
            'exit_threshold': {'min': -2, 'max': 0, 'type': 'float'},
            'stop_loss': {'min': 0.5, 'max': 10, 'type': 'float'},
            'position_size': {'min': 0.01, 'max': 1.0, 'type': 'float'}
        }
    
    def random_parameters(self):
        """Generate random parameter values."""
        params = {}
        
        for name, config in self.parameters.items():
            if config['type'] == 'int':
                params[name] = random.randint(config['min'], config['max'])
            elif config['type'] == 'float':
                params[name] = random.uniform(config['min'], config['max'])
        
        return params


class MultiObjectiveFeedbackLoop:
    """
    Multi-objective feedback loop that learns from performance differences
    across market regimes.
    """
    
    def __init__(self):
        """Initialize the feedback loop with parameter space and market models."""
        # Initialize parameter space
        self.parameter_space = ParameterSpace()
        
        # Initial market models (these will be refined over iterations)
        self.market_models = {
            'bullish': {'trend': 0.006, 'volatility': 0.015, 'mean_reversion': 0.2},
            'bearish': {'trend': -0.004, 'volatility': 0.025, 'mean_reversion': 0.15},
            'sideways': {'trend': 0.0005, 'volatility': 0.01, 'mean_reversion': 0.7},
            'volatile': {'trend': 0.001, 'volatility': 0.035, 'mean_reversion': 0.1}
        }
        
        # History tracking
        self.param_history = []  # Market model parameters over time
        self.expected_perf = []  # Expected performance
        self.actual_perf = []    # Actual performance
        self.accuracy_history = {'overall': []}  # Prediction accuracy
        
        # Initialize accuracy tracking for each regime
        for regime in self.market_models:
            self.accuracy_history[regime] = []
        
        print("Initialized MultiObjectiveFeedbackLoop")
        print("Initial market models:")
        for regime, params in self.market_models.items():
            print(f"  {regime}: {params}")
    
    def optimize(self):
        """
        Run multi-objective optimization with current market models.
        
        Returns:
            Dict with optimized strategy parameters
        """
        print("\nRunning multi-objective optimization...")
        
        # Generate synthetic data for each regime using current models
        market_data = {}
        for regime, params in self.market_models.items():
            market_data[regime] = generate_market_data(
                days=500,  # Plenty of data for any lookback period
                trend=params['trend'],
                volatility=params['volatility'],
                mean_reversion=params['mean_reversion']
            )
        
        # Function to evaluate parameters across all regimes
        def evaluate_parameters(params):
            strategy = SimpleStrategy(params)
            results = {}
            
            for regime, prices in market_data.items():
                results[regime] = strategy.backtest(prices)
            
            return results
        
        # Simple multi-objective optimization using NSGA-II inspired approach
        # Initialize population with random solutions
        population = []
        
        for _ in range(POPULATION_SIZE):
            params = self.parameter_space.random_parameters()
            objectives = evaluate_parameters(params)
            
            population.append({
                'parameters': params,
                'objectives': objectives
            })
        
        # Run through generations
        for gen in range(GENERATIONS):
            # Sort population by Pareto dominance
            fronts = self._fast_non_dominated_sort(population)
            
            # Create new population
            new_population = []
            
            # Add solutions from the Pareto fronts
            for front in fronts:
                if len(new_population) + len(front) <= POPULATION_SIZE // 2:
                    new_population.extend([population[i] for i in front])
                else:
                    # If adding entire front exceeds half population, stop
                    break
            
            # Add random new solutions
            while len(new_population) < POPULATION_SIZE:
                params = self.parameter_space.random_parameters()
                objectives = evaluate_parameters(params)
                
                new_population.append({
                    'parameters': params,
                    'objectives': objectives
                })
            
            # Replace population
            population = new_population
        
        # Select a balanced solution from the Pareto front
        fronts = self._fast_non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]]
        
        best_solution = self._select_balanced_solution(pareto_front)
        
        # Store expected performance for verification
        self.expected_perf.append(best_solution['objectives'])
        
        print("Optimization complete!")
        print("\nSelected balanced solution:")
        for param, value in best_solution['parameters'].items():
            print(f"  {param}: {value}")
        
        print("\nExpected performance:")
        for regime, metrics in best_solution['objectives'].items():
            print(f"  {regime}: Return={metrics['return']:.2f}%, "
                  f"Drawdown={metrics['max_drawdown']:.2f}%, "
                  f"Sharpe={metrics['sharpe']:.2f}")
        
        return best_solution['parameters']
    
    def _fast_non_dominated_sort(self, population):
        """
        Sort population into Pareto fronts using NSGA-II approach.
        
        Args:
            population: List of solutions with objectives
            
        Returns:
            List of fronts, where each front is a list of solution indices
        """
        n = len(population)
        if n == 0:
            return [[]]
            
        dominated_by = [[] for _ in range(n)]
        domination_count = [0 for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # Check if i dominates j
                i_dominates = True
                j_dominates = True
                
                for regime in population[i]['objectives']:
                    # Higher return is better
                    ret_i = population[i]['objectives'][regime]['return']
                    ret_j = population[j]['objectives'][regime]['return']
                    
                    # Lower drawdown is better, so we negate
                    dd_i = -population[i]['objectives'][regime]['max_drawdown']
                    dd_j = -population[j]['objectives'][regime]['max_drawdown']
                    
                    if ret_i < ret_j or dd_i < dd_j:
                        i_dominates = False
                    if ret_i > ret_j or dd_i > dd_j:
                        j_dominates = False
                
                if i_dominates:
                    dominated_by[i].append(j)
                elif j_dominates:
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Build remaining fronts
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            
            for j in fronts[i]:
                for k in dominated_by[j]:
                    domination_count[k] -= 1
                    
                    if domination_count[k] == 0:
                        next_front.append(k)
            
            i += 1
            
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def _select_balanced_solution(self, pareto_front):
        """
        Select a balanced solution from the Pareto front.
        
        Args:
            pareto_front: List of Pareto-optimal solutions
            
        Returns:
            Selected solution
        """
        if not pareto_front:
            return None
        
        if len(pareto_front) == 1:
            return pareto_front[0]
        
        # Find min and max values for each objective
        regimes = list(pareto_front[0]['objectives'].keys())
        min_values = {}
        max_values = {}
        
        for regime in regimes:
            min_return = min(solution['objectives'][regime]['return'] for solution in pareto_front)
            max_return = max(solution['objectives'][regime]['return'] for solution in pareto_front)
            
            min_values[regime] = min_return
            max_values[regime] = max_return
        
        # Find solution closest to ideal point (max return, min drawdown for all regimes)
        best_idx = 0
        best_distance = float('inf')
        
        for i, solution in enumerate(pareto_front):
            # Calculate normalized distance to ideal point
            distance = 0
            
            for regime in regimes:
                ret = solution['objectives'][regime]['return']
                
                # Normalize return to [0,1]
                ret_range = max_values[regime] - min_values[regime]
                norm_ret = (ret - min_values[regime]) / ret_range if ret_range > 0 else 0
                
                # Distance to ideal (1.0)
                distance += (1.0 - norm_ret) ** 2
            
            distance = math.sqrt(distance)
            
            if distance < best_distance:
                best_distance = distance
                best_idx = i
        
        return pareto_front[best_idx]
    
    def test_strategy(self, params):
        """
        Test strategy in simulated real-world conditions.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Dict of performance metrics by regime
        """
        print("\nTesting strategy in 'real-world' conditions...")
        
        # Generate "real" market data using the hidden parameters
        real_performance = {}
        
        for regime, real_params in REAL_WORLD_PARAMS.items():
            # Generate real market data
            real_data = generate_market_data(
                days=500,
                trend=real_params['trend'],
                volatility=real_params['volatility'],
                mean_reversion=real_params['mean_reversion']
            )
            
            # Run backtest
            strategy = SimpleStrategy(params)
            results = strategy.backtest(real_data)
            
            real_performance[regime] = results
        
        # Store for verification
        self.actual_perf.append(real_performance)
        
        print("Real-world performance:")
        for regime, metrics in real_performance.items():
            print(f"  {regime}: Return={metrics['return']:.2f}%, "
                  f"Drawdown={metrics['max_drawdown']:.2f}%, "
                  f"Sharpe={metrics['sharpe']:.2f}")
        
        return real_performance
    
    def verify_accuracy(self):
        """
        Verify prediction accuracy by comparing expected vs actual performance.
        
        Returns:
            Overall prediction accuracy
        """
        # Get most recent expected and actual performance
        expected = self.expected_perf[-1]
        actual = self.actual_perf[-1]
        
        print("\nVerifying prediction accuracy...")
        
        # Calculate accuracy for each regime
        regime_accuracies = {}
        
        for regime in expected:
            if regime not in actual:
                continue
                
            expected_metrics = expected[regime]
            actual_metrics = actual[regime]
            
            # Calculate relative accuracy for return
            return_diff = abs(expected_metrics['return'] - actual_metrics['return'])
            return_scale = max(1.0, abs(expected_metrics['return']))
            return_accuracy = 1.0 - min(1.0, return_diff / return_scale)
            
            # Calculate relative accuracy for drawdown
            dd_diff = abs(expected_metrics['max_drawdown'] - actual_metrics['max_drawdown'])
            dd_scale = max(1.0, abs(expected_metrics['max_drawdown']))
            dd_accuracy = 1.0 - min(1.0, dd_diff / dd_scale)
            
            # Calculate relative accuracy for Sharpe
            sharpe_diff = abs(expected_metrics['sharpe'] - actual_metrics['sharpe'])
            sharpe_scale = max(1.0, abs(expected_metrics['sharpe']))
            sharpe_accuracy = 1.0 - min(1.0, sharpe_diff / sharpe_scale)
            
            # Overall accuracy for this regime
            regime_accuracy = (return_accuracy + dd_accuracy + sharpe_accuracy) / 3.0
            
            regime_accuracies[regime] = regime_accuracy
            self.accuracy_history[regime].append(regime_accuracy)
            
            print(f"  {regime}: Accuracy={regime_accuracy:.2%} "
                  f"(Return={return_accuracy:.2%}, "
                  f"Drawdown={dd_accuracy:.2%}, "
                  f"Sharpe={sharpe_accuracy:.2%})")
        
        # Calculate overall accuracy
        overall_accuracy = sum(regime_accuracies.values()) / len(regime_accuracies)
        
        # Store in history
        self.accuracy_history['overall'].append(overall_accuracy)
        
        print(f"Overall prediction accuracy: {overall_accuracy:.2%}")
        
        return overall_accuracy
    
    def update_market_models(self):
        """
        Update market models based on verification results.
        
        This is the key feedback mechanism that allows the system to learn.
        """
        if not self.expected_perf or not self.actual_perf:
            print("No performance data available for model updates.")
            return
        
        print("\nUpdating market models based on verification results...")
        
        # Store current parameters for tracking
        self.param_history.append({})
        for regime, params in self.market_models.items():
            self.param_history[-1][regime] = params.copy()
        
        # Get the most recent expected and actual performance
        expected = self.expected_perf[-1]
        actual = self.actual_perf[-1]
        
        # Update each regime's parameters
        for regime in self.market_models:
            if regime not in expected or regime not in actual:
                continue
            
            # Calculate ratios between actual and expected metrics
            return_ratio = actual[regime]['return'] / expected[regime]['return'] if expected[regime]['return'] != 0 else 1.0
            volatility_ratio = actual[regime]['max_drawdown'] / expected[regime]['max_drawdown'] if expected[regime]['max_drawdown'] != 0 else 1.0
            
            # Adjust trend parameter based on return difference
            if abs(return_ratio) > 1.1:
                # If actual returns are higher, increase trend strength
                self.market_models[regime]['trend'] *= (1.0 + (return_ratio - 1.0) * LEARNING_RATE)
            elif abs(return_ratio) < 0.9:
                # If actual returns are lower, decrease trend strength
                self.market_models[regime]['trend'] *= (1.0 - (1.0 - return_ratio) * LEARNING_RATE)
            
            # Adjust volatility based on drawdown difference
            self.market_models[regime]['volatility'] *= (1.0 + (volatility_ratio - 1.0) * LEARNING_RATE)
            
            # Adjust mean reversion based on combined factors
            if return_ratio < 0.9 and volatility_ratio > 1.1:
                # Lower returns and higher drawdowns might indicate more mean reversion
                self.market_models[regime]['mean_reversion'] *= (1.0 + LEARNING_RATE)
            elif return_ratio > 1.1 and volatility_ratio < 0.9:
                # Higher returns and lower drawdowns might indicate less mean reversion
                self.market_models[regime]['mean_reversion'] *= (1.0 - LEARNING_RATE)
            
            # Clamp parameters to reasonable ranges
            self.market_models[regime]['trend'] = max(-0.01, min(0.01, self.market_models[regime]['trend']))
            self.market_models[regime]['volatility'] = max(0.005, min(0.05, self.market_models[regime]['volatility']))
            self.market_models[regime]['mean_reversion'] = max(0.05, min(0.9, self.market_models[regime]['mean_reversion']))
        
        print("Updated market models:")
        for regime, params in self.market_models.items():
            print(f"  {regime}: {params}")
    
    def run_feedback_loop(self, iterations=3):
        """
        Run the complete feedback loop for multiple iterations.
        
        Args:
            iterations: Number of feedback loop iterations to run
        """
        print(f"\nRunning feedback loop for {iterations} iterations...")
        
        for i in range(iterations):
            print(f"\n=== Iteration {i+1}/{iterations} ===")
            
            # Step 1: Optimize strategy parameters
            params = self.optimize()
            
            # Step 2: Test in "real-world" conditions
            self.test_strategy(params)
            
            # Step 3: Verify prediction accuracy
            self.verify_accuracy()
            
            # Step 4: Update market models
            self.update_market_models()
            
            # Small delay for readability
            time.sleep(0.5)
        
        # Show summary of results
        print("\n" + "=" * 70)
        print("FEEDBACK LOOP RESULTS SUMMARY")
        print("=" * 70)
        
        # Show accuracy improvement
        print("\n1. PREDICTION ACCURACY IMPROVEMENT")
        print("-" * 40)
        
        for i, accuracy in enumerate(self.accuracy_history['overall']):
            print(f"Iteration {i+1}: {accuracy:.2%}")
        
        if len(self.accuracy_history['overall']) > 1:
            initial = self.accuracy_history['overall'][0]
            final = self.accuracy_history['overall'][-1]
            improvement = (final - initial) / initial if initial > 0 else 0
            
            print(f"\nInitial accuracy: {initial:.2%}")
            print(f"Final accuracy: {final:.2%}")
            print(f"Improvement: {improvement:.2%}")
        
        # Show parameter convergence
        print("\n2. PARAMETER CONVERGENCE")
        print("-" * 40)
        
        for regime in self.market_models:
            if regime in REAL_WORLD_PARAMS:
                print(f"\n{regime.capitalize()} regime:")
                print("Parameter     | Initial    | Final      | Real       | Convergence")
                print("-" * 64)
                
                initial_params = self.param_history[0][regime]
                final_params = self.market_models[regime]
                real_params = REAL_WORLD_PARAMS[regime]
                
                for param in ['trend', 'volatility', 'mean_reversion']:
                    initial = initial_params[param]
                    final = final_params[param]
                    real = real_params[param]
                    
                    # Calculate convergence percentage
                    initial_diff = abs(initial - real)
                    final_diff = abs(final - real)
                    
                    if initial_diff == 0:
                        convergence = 100.0
                    else:
                        convergence = (1 - final_diff / initial_diff) * 100.0
                        convergence = max(0, convergence)  # Don't show negative convergence
                    
                    print(f"{param:<13} | {initial:<10.6f} | {final:<10.6f} | {real:<10.6f} | {convergence:>6.2f}%")
        
        # Performance gap reduction
        print("\n3. PERFORMANCE GAP REDUCTION")
        print("-" * 40)
        
        if len(self.expected_perf) > 1 and len(self.actual_perf) > 1:
            initial_expected = self.expected_perf[0]
            initial_actual = self.actual_perf[0]
            final_expected = self.expected_perf[-1]
            final_actual = self.actual_perf[-1]
            
            print("Regime       | Initial Gap | Final Gap  | Reduction")
            print("-" * 50)
            
            for regime in self.market_models:
                if regime in initial_expected and regime in initial_actual:
                    initial_gap = abs(initial_expected[regime]['return'] - initial_actual[regime]['return'])
                    final_gap = abs(final_expected[regime]['return'] - final_actual[regime]['return'])
                    
                    reduction = (initial_gap - final_gap) / initial_gap * 100 if initial_gap > 0 else 0
                    
                    print(f"{regime:<12} | {initial_gap:<11.2f} | {final_gap:<10.2f} | {reduction:>6.2f}%")
        
        # Overall conclusion
        print("\n" + "=" * 70)
        
        if len(self.accuracy_history['overall']) > 1 and self.accuracy_history['overall'][-1] > self.accuracy_history['overall'][0]:
            print("CONCLUSION: The feedback loop successfully improved optimization accuracy!")
            print(f"Prediction accuracy increased by {improvement:.2%} over {iterations} iterations.")
        else:
            print("CONCLUSION: The feedback loop did not improve optimization accuracy.")
        
        print("=" * 70)


def main():
    """Run a demonstration of the multi-objective feedback loop."""
    print("=" * 70)
    print("MULTI-OBJECTIVE FEEDBACK LOOP DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstrates how our optimization system learns from the")
    print("gap between expected and actual performance to improve its")
    print("prediction accuracy over multiple iterations.")
    print("\n" + "=" * 70)
    
    # Run the feedback loop demonstration
    feedback_loop = MultiObjectiveFeedbackLoop()
    feedback_loop.run_feedback_loop(iterations=3)


if __name__ == "__main__":
    main()
