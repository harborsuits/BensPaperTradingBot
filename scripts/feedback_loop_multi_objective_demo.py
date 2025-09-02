#!/usr/bin/env python3
"""
Multi-Objective Feedback Loop Demonstration

This script implements a simplified version of the optimization feedback loop
that uses multi-objective optimization to find strategies that perform well
across different market regimes.

It builds upon our successful practical_optimization_example.py and
simplified_feedback_loop.py implementations, adding the ability to learn
and improve over iterations.
"""

import random
import math
import time
from typing import Dict, List, Any, Tuple

# Import from our successful examples
from practical_optimization_example import ParameterSpace, TradingStrategy
from simplified_feedback_loop import generate_market_data

# Import visualization utilities
try:
    import feedback_loop_visualization as viz
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    print("Visualization module not available. Skipping visualizations.")
    VISUALIZATIONS_AVAILABLE = False


class MultiObjectiveFeedbackLoop:
    """
    Implements a feedback loop for multi-objective strategy optimization.
    
    This class demonstrates how the system can learn from the gap between
    expected and actual performance to improve its prediction accuracy.
    """
    
    def __init__(self):
        """Initialize the feedback loop with parameter space and market models."""
        # Initialize parameter space (based on practical_optimization_example.py)
        self.parameter_space = ParameterSpace()
        self.parameter_space.add_integer_parameter('trend_lookback', 5, 50)
        self.parameter_space.add_integer_parameter('volatility_lookback', 5, 50)
        self.parameter_space.add_real_parameter('entry_threshold', 0.0, 2.0)
        self.parameter_space.add_real_parameter('exit_threshold', -2.0, 0.0)
        self.parameter_space.add_real_parameter('stop_loss_pct', 0.5, 5.0)
        self.parameter_space.add_real_parameter('position_sizing', 0.01, 0.5)
        self.parameter_space.add_boolean_parameter('trail_stop')
        self.parameter_space.add_categorical_parameter('filter_type', ['sma', 'ema', 'none'])
        self.parameter_space.add_real_parameter('indicator_weight', 0.1, 0.9)
        
        # Initial market models by regime - these will be refined through feedback
        self.market_models = {
            'bullish': {'trend': 0.006, 'volatility': 0.015, 'reversion': 0.2},
            'bearish': {'trend': -0.004, 'volatility': 0.025, 'reversion': 0.15},
            'sideways': {'trend': 0.0005, 'volatility': 0.01, 'reversion': 0.7},
            'volatile': {'trend': 0.001, 'volatility': 0.035, 'reversion': 0.1}
        }
        
        # The "real" market parameters (unknown to the optimization system)
        # In a real system, this would be the actual market behavior
        self.real_world_params = {
            'bullish': {'trend': 0.008, 'volatility': 0.018, 'reversion': 0.15},
            'bearish': {'trend': -0.006, 'volatility': 0.03, 'reversion': 0.1},
            'sideways': {'trend': 0.0, 'volatility': 0.012, 'reversion': 0.8},
            'volatile': {'trend': 0.0005, 'volatility': 0.045, 'reversion': 0.05}
        }
        
        # Tracking for feedback loop
        self.param_history = []
        self.expected_perf = []
        self.actual_perf = []
        self.accuracy = {'overall': []}
        
        for regime in self.market_models:
            self.accuracy[regime] = []
        
        print("Initialized MultiObjectiveFeedbackLoop")
        print("Initial market models:")
        for regime, params in self.market_models.items():
            print(f"  {regime}: {params}")
    
    def optimize(self):
        """
        Run multi-objective optimization with current market models.
        
        Returns:
            Dict containing optimized strategy parameters
        """
        print("\nRunning multi-objective optimization...")
        
        # Generate synthetic data for each regime using current models
        market_data = {}
        # Use 500 days to ensure we have enough data for any lookback period
        for regime, params in self.market_models.items():
            market_data[regime] = generate_market_data(
                days=500,  # Increased from 252 to provide more history
                trend=params['trend'],
                volatility=params['volatility'],
                mean_reversion=params['reversion']
            )
        
        # Define fitness functions for each regime
        def evaluate_in_regime(params, regime):
            strategy = TradingStrategy(params)
            
            # Simple backtest on the synthetic data
            prices = market_data[regime]
            
            # Skip evaluation if we don't have enough price history
            if len(prices) <= max(strategy.trend_lookback, strategy.volatility_lookback) + 10:
                return {'return': 0, 'drawdown': 0, 'sharpe': 0}
            
            position = 0
            entry_price = 0
            equity = [100.0]  # Start with $100
            
            for i in range(strategy.trend_lookback, len(prices)):
                price_window = prices[:i+1]
                current_price = price_window[-1]
                
                # Generate signal using strategy logic
                try:
                    signal = strategy.generate_signal(price_window)
                except ZeroDivisionError:
                    # Handle division by zero in signal calculation
                    signal = 0
                
                # Execute trading logic (simplified)
                if position == 0 and signal > strategy.entry_threshold:
                    # Enter position
                    position = 1
                    entry_price = current_price
                elif position > 0:
                    # Check exit conditions
                    if (signal < strategy.exit_threshold or 
                            current_price < entry_price * (1 - strategy.stop_loss_pct/100)):
                        # Exit position
                        position = 0
                
                # Update equity (simplified)
                if position > 0:
                    equity.append(equity[0] * (current_price / prices[strategy.trend_lookback]))
                else:
                    equity.append(equity[-1])
            
            # Calculate metrics
            if len(equity) < 2:
                return {'return': 0, 'drawdown': 0, 'sharpe': 0}
            
            # Return percentage
            total_return = (equity[-1] / equity[0] - 1) * 100
            
            # Maximum drawdown
            max_drawdown = 0
            peak = equity[0]
            
            for e in equity:
                if e > peak:
                    peak = e
                else:
                    drawdown = (peak - e) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Sharpe ratio (annualized, simplified)
            returns = [(equity[i] / equity[i-1] - 1) for i in range(1, len(equity))]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(sum((r - avg_return)**2 for r in returns) / len(returns))
                sharpe = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
            else:
                sharpe = 0
            
            return {
                'return': total_return,
                'drawdown': max_drawdown,
                'sharpe': sharpe
            }
        
        # Simple multi-objective optimization (random search with Pareto selection)
        population_size = 100
        generations = 5
        population = []
        
        # Generate initial population
        for _ in range(population_size):
            params = self.parameter_space.get_random_parameters()
            
            # Evaluate in each regime
            objectives = {}
            for regime in market_data:
                objectives[regime] = evaluate_in_regime(params, regime)
            
            population.append({
                'params': params,
                'objectives': objectives
            })
        
        # Run through generations
        for gen in range(generations):
            # Sort population by Pareto dominance
            pareto_fronts = self._fast_non_dominated_sort(population)
            
            # Create new population - keep best solutions and add new ones
            new_population = []
            for front in pareto_fronts:
                if len(new_population) + len(front) <= population_size // 2:
                    new_population.extend([population[i] for i in front])
                else:
                    break
            
            # Add new random solutions
            while len(new_population) < population_size:
                params = self.parameter_space.get_random_parameters()
                
                # Evaluate in each regime
                objectives = {}
                for regime in market_data:
                    objectives[regime] = evaluate_in_regime(params, regime)
                
                new_population.append({
                    'params': params,
                    'objectives': objectives
                })
            
            population = new_population
        
        # Select a balanced solution from the first Pareto front
        pareto_fronts = self._fast_non_dominated_sort(population)
        pareto_front = [population[i] for i in pareto_fronts[0]]
        
        best_solution = self._select_balanced_solution(pareto_front)
        
        # Store expected performance for verification
        expected_performance = {}
        for regime, metrics in best_solution['objectives'].items():
            expected_performance[regime] = metrics
        
        self.expected_perf.append(expected_performance)
        
        print("Optimization complete! Selected balanced solution:")
        for param, value in best_solution['params'].items():
            print(f"  {param}: {value}")
        
        print("\nExpected performance:")
        for regime, metrics in expected_performance.items():
            print(f"  {regime}: Return={metrics['return']:.2f}%, "
                  f"Drawdown={metrics['drawdown']:.2f}%, "
                  f"Sharpe={metrics['sharpe']:.2f}")
        
        return best_solution['params']
    
    def _fast_non_dominated_sort(self, population):
        """
        Sort population into Pareto fronts.
        
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
                    obj_i = population[i]['objectives'][regime]['return']
                    obj_j = population[j]['objectives'][regime]['return']
                    
                    if obj_i < obj_j:
                        i_dominates = False
                    if obj_i > obj_j:
                        j_dominates = False
                
                if i_dominates:
                    dominated_by[i].append(j)
                elif j_dominates:
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
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
        
        # Normalize objectives
        regimes = list(pareto_front[0]['objectives'].keys())
        
        # Find min and max values for each objective
        min_values = {}
        max_values = {}
        
        for regime in regimes:
            min_return = min(solution['objectives'][regime]['return'] for solution in pareto_front)
            max_return = max(solution['objectives'][regime]['return'] for solution in pareto_front)
            
            min_values[regime] = min_return
            max_values[regime] = max_return
        
        # Calculate distance to the ideal point
        best_idx = 0
        best_distance = float('inf')
        
        for i, solution in enumerate(pareto_front):
            # Calculate normalized distance to the ideal point (1,1,1,...)
            distance = 0
            
            for regime in regimes:
                obj_return = solution['objectives'][regime]['return']
                
                # Normalize to [0,1] range
                range_val = max_values[regime] - min_values[regime]
                normalized = (obj_return - min_values[regime]) / range_val if range_val > 0 else 0
                
                # Distance to ideal (1.0)
                distance += (1.0 - normalized) ** 2
            
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
            Dictionary of performance metrics by regime
        """
        print("\nTesting strategy in 'real-world' conditions...")
        
        # We'll use our hidden real-world parameters to simulate reality
        real_performance = {}
        
        # Evaluate in each regime
        for regime, real_params in self.real_world_params.items():
            # Generate "real" market data
            real_data = generate_market_data(
                days=252,
                trend=real_params['trend'],
                volatility=real_params['volatility'],
                mean_reversion=real_params['reversion']
            )
            
            # Evaluate strategy
            strategy = TradingStrategy(params)
            
            # Simple backtest
            position = 0
            entry_price = 0
            equity = [100.0]  # Start with $100
            
            for i in range(strategy.trend_lookback, len(real_data)):
                price_window = real_data[:i+1]
                current_price = price_window[-1]
                
                # Generate signal using strategy logic
                try:
                    signal = strategy.generate_signal(price_window)
                except ZeroDivisionError:
                    # Handle division by zero in signal calculation
                    signal = 0
                
                # Execute trading logic (simplified)
                if position == 0 and signal > strategy.entry_threshold:
                    # Enter position
                    position = 1
                    entry_price = current_price
                elif position > 0:
                    # Check exit conditions
                    if (signal < strategy.exit_threshold or 
                            current_price < entry_price * (1 - strategy.stop_loss_pct/100)):
                        # Exit position
                        position = 0
                
                # Update equity (simplified)
                if position > 0:
                    equity.append(equity[0] * (current_price / real_data[strategy.trend_lookback]))
                else:
                    equity.append(equity[-1])
            
            # Calculate metrics
            if len(equity) < 2:
                real_performance[regime] = {'return': 0, 'drawdown': 0, 'sharpe': 0}
                continue
            
            # Return percentage
            total_return = (equity[-1] / equity[0] - 1) * 100
            
            # Maximum drawdown
            max_drawdown = 0
            peak = equity[0]
            
            for e in equity:
                if e > peak:
                    peak = e
                else:
                    drawdown = (peak - e) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Sharpe ratio (annualized, simplified)
            returns = [(equity[i] / equity[i-1] - 1) for i in range(1, len(equity))]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(sum((r - avg_return)**2 for r in returns) / len(returns))
                sharpe = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
            else:
                sharpe = 0
            
            real_performance[regime] = {
                'return': total_return,
                'drawdown': max_drawdown,
                'sharpe': sharpe
            }
        
        # Store for verification
        self.actual_perf.append(real_performance)
        
        print("Real-world performance:")
        for regime, metrics in real_performance.items():
            print(f"  {regime}: Return={metrics['return']:.2f}%, "
                  f"Drawdown={metrics['drawdown']:.2f}%, "
                  f"Sharpe={metrics['sharpe']:.2f}")
        
        return real_performance
    
    def verify_accuracy(self, expected=None, actual=None):
        """
        Verify prediction accuracy by comparing expected vs actual performance.
        
        Args:
            expected: Expected performance (optional)
            actual: Actual performance (optional)
            
        Returns:
            Overall prediction accuracy
        """
        if expected is None:
            expected = self.expected_perf[-1]
        
        if actual is None:
            actual = self.actual_perf[-1]
        
        print("\nVerifying prediction accuracy...")
        
        # Calculate accuracy for each regime and metric
        regime_accuracies = {}
        
        for regime in expected:
            expected_metrics = expected[regime]
            actual_metrics = actual[regime]
            
            # Calculate relative accuracy for return
            return_diff = abs(expected_metrics['return'] - actual_metrics['return'])
            return_scale = max(1.0, abs(expected_metrics['return']))
            return_accuracy = 1.0 - min(1.0, return_diff / return_scale)
            
            # Calculate relative accuracy for drawdown
            drawdown_diff = abs(expected_metrics['drawdown'] - actual_metrics['drawdown'])
            drawdown_scale = max(1.0, abs(expected_metrics['drawdown']))
            drawdown_accuracy = 1.0 - min(1.0, drawdown_diff / drawdown_scale)
            
            # Calculate relative accuracy for Sharpe ratio
            sharpe_diff = abs(expected_metrics['sharpe'] - actual_metrics['sharpe'])
            sharpe_scale = max(1.0, abs(expected_metrics['sharpe']))
            sharpe_accuracy = 1.0 - min(1.0, sharpe_diff / sharpe_scale)
            
            # Overall accuracy for this regime
            regime_accuracy = (return_accuracy + drawdown_accuracy + sharpe_accuracy) / 3.0
            
            regime_accuracies[regime] = regime_accuracy
            
            # Store in history
            self.accuracy[regime].append(regime_accuracy)
            
            print(f"  {regime}: Accuracy={regime_accuracy:.2%} "
                  f"(Return={return_accuracy:.2%}, "
                  f"Drawdown={drawdown_accuracy:.2%}, "
                  f"Sharpe={sharpe_accuracy:.2%})")
        
        # Calculate overall accuracy across all regimes
        overall_accuracy = sum(regime_accuracies.values()) / len(regime_accuracies)
        
        # Store in history
        self.accuracy['overall'].append(overall_accuracy)
        
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
        
        # Learning rate for parameter adjustments
        learning_rate = 0.2
        
        # Update each regime's parameters
        for regime in self.market_models:
            if regime not in expected or regime not in actual:
                continue
            
            # Calculate ratios between actual and expected metrics
            return_ratio = actual[regime]['return'] / expected[regime]['return'] if expected[regime]['return'] != 0 else 1.0
            volatility_ratio = actual[regime]['drawdown'] / expected[regime]['drawdown'] if expected[regime]['drawdown'] != 0 else 1.0
            
            # Adjust trend parameter based on return difference
            if abs(return_ratio) > 1.1:
                # If actual returns are higher, increase trend strength
                self.market_models[regime]['trend'] *= (1.0 + (return_ratio - 1.0) * learning_rate)
            elif abs(return_ratio) < 0.9:
                # If actual returns are lower, decrease trend strength
                self.market_models[regime]['trend'] *= (1.0 - (1.0 - return_ratio) * learning_rate)
            
            # Adjust volatility based on drawdown difference
            self.market_models[regime]['volatility'] *= (1.0 + (volatility_ratio - 1.0) * learning_rate)
            
            # Adjust mean reversion based on combined factors
            if return_ratio < 0.9 and volatility_ratio > 1.1:
                # Lower returns and higher drawdowns might indicate more mean reversion
                self.market_models[regime]['reversion'] *= (1.0 + learning_rate)
            elif return_ratio > 1.1 and volatility_ratio < 0.9:
                # Higher returns and lower drawdowns might indicate less mean reversion
                self.market_models[regime]['reversion'] *= (1.0 - learning_rate)
            
            # Clamp parameters to reasonable ranges
            self.market_models[regime]['trend'] = max(-0.01, min(0.01, self.market_models[regime]['trend']))
            self.market_models[regime]['volatility'] = max(0.005, min(0.05, self.market_models[regime]['volatility']))
            self.market_models[regime]['reversion'] = max(0.05, min(0.5, self.market_models[regime]['reversion']))
        
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
            real_perf = self.test_strategy(params)
            
            # Step 3: Verify prediction accuracy
            self.verify_accuracy()
            
            # Step 4: Update market models
            self.update_market_models()
            
            # Small delay for readability
            time.sleep(0.5)
        
        # Generate visualization report if available
        if VISUALIZATIONS_AVAILABLE:
            print("\nGenerating visualization report...")
            viz.generate_feedback_loop_report(
                self.param_history,
                self.real_world_params,
                self.accuracy,
                self.expected_perf,
                self.actual_perf
            )
        
        # Show summary of results
        print("\n" + "=" * 70)
        print("FEEDBACK LOOP RESULTS SUMMARY")
        print("=" * 70)
        
        # Show accuracy improvement
        print("\n1. PREDICTION ACCURACY IMPROVEMENT")
        print("-" * 40)
        
        for i, accuracy in enumerate(self.accuracy['overall']):
            print(f"Iteration {i+1}: {accuracy:.2%}")
        
        if len(self.accuracy['overall']) > 1:
            initial = self.accuracy['overall'][0]
            final = self.accuracy['overall'][-1]
            improvement = (final - initial) / initial if initial > 0 else 0
            
            print(f"\nInitial accuracy: {initial:.2%}")
            print(f"Final accuracy: {final:.2%}")
            print(f"Improvement: {improvement:.2%}")
        
        # Overall conclusion
        print("\n" + "=" * 70)
        
        if len(self.accuracy['overall']) > 1 and self.accuracy['overall'][-1] > self.accuracy['overall'][0]:
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
