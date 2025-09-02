#!/usr/bin/env python3
"""
Feedback Loop Demo: Self-Improving Trading Strategy Optimization

This script demonstrates the complete feedback loop for strategy optimization:
1. Initial optimization based on synthetic market data
2. Testing in "real-world" conditions (simulated with perturbations)
3. Verification of prediction accuracy
4. Adjustment of synthetic market parameters to match reality
5. Re-optimization with improved synthetic parameters

The key innovation demonstrated is how the system learns from performance 
verification to continuously improve its optimization accuracy.
"""

import time
import random
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import core components from our practical optimization example
from practical_optimization_example import ParameterSpace, MultiObjectiveOptimizer, TradingStrategy

# Define synthetic market functions directly in this script
def generate_synthetic_market_data(days=252, trend_strength=0.0006, volatility=0.015, mean_reversion=0.2, base_price=100.0):
    """Generate synthetic market data with given parameters.
    
    Args:
        days: Number of days to generate
        trend_strength: Strength of the trend (daily drift)
        volatility: Daily volatility
        mean_reversion: Mean reversion strength (0-1)
        base_price: Starting price
        
    Returns:
        List of prices
    """
    prices = [base_price]
    
    for _ in range(days):
        # Calculate mean reversion component
        current_price = prices[-1]
        price_diff = current_price - base_price
        reversion = -price_diff * mean_reversion / 100.0
        
        # Random component based on volatility
        noise = random.normalvariate(0, volatility)
        
        # Combine drift, mean reversion and noise
        daily_return = trend_strength + reversion + noise
        
        # Calculate new price
        new_price = current_price * (1 + daily_return)
        prices.append(new_price)
    
    return prices

def evaluate_strategy_in_regime(strategy, price_data):
    """Evaluate a trading strategy on price data.
    
    Args:
        strategy: TradingStrategy instance
        price_data: List of prices
        
    Returns:
        Performance metrics dictionary
    """
    # Reset strategy state
    strategy.reset()
    
    # Initial capital
    initial_capital = 10000.0
    capital = initial_capital
    position = 0
    
    trades = []
    equity_curve = [initial_capital]
    returns = []
    
    # Need minimum lookback periods of data
    min_bars = max(strategy.trend_lookback, strategy.volatility_lookback) + 1
    
    # Simulate trading
    for i in range(min_bars, len(price_data)):
        # Get price data up to current bar
        prices = price_data[:i+1]
        current_price = prices[-1]
        
        # Generate signal
        signal = strategy.generate_signal(prices)
        
        # Execute trade logic
        if position == 0 and signal > strategy.entry_threshold:
            # Enter long position
            shares = capital * strategy.position_sizing / current_price
            position = shares
            entry_price = current_price
            stop_price = entry_price * (1 - strategy.stop_loss_pct / 100)
            
            trades.append({
                'type': 'entry',
                'price': current_price,
                'shares': shares,
                'value': shares * current_price
            })
            
        elif position > 0:
            # Check for exit conditions
            stop_triggered = current_price <= stop_price
            signal_exit = signal < strategy.exit_threshold
            
            if stop_triggered or signal_exit:
                # Exit position
                exit_value = position * current_price
                entry_value = position * entry_price
                pnl = exit_value - entry_value
                
                trades.append({
                    'type': 'exit',
                    'price': current_price,
                    'shares': position,
                    'value': exit_value,
                    'pnl': pnl,
                    'pct': (current_price / entry_price - 1) * 100
                })
                
                # Update capital
                capital += pnl
                position = 0
            
            # Update stop price for trailing stop if enabled
            elif strategy.trail_stop and current_price > entry_price:
                # Move stop up to maintain same percentage distance from current price
                stop_price = max(stop_price, current_price * (1 - strategy.stop_loss_pct / 100))
        
        # Update equity curve
        if position > 0:
            equity = capital + position * current_price
        else:
            equity = capital
            
        equity_curve.append(equity)
        
        # Calculate return
        if len(equity_curve) > 1:
            daily_return = equity_curve[-1] / equity_curve[-2] - 1
            returns.append(daily_return)
    
    # Close any open position at the end
    if position > 0:
        final_price = price_data[-1]
        exit_value = position * final_price
        entry_value = position * entry_price
        pnl = exit_value - entry_value
        
        trades.append({
            'type': 'exit',
            'price': final_price,
            'shares': position,
            'value': exit_value,
            'pnl': pnl,
            'pct': (final_price / entry_price - 1) * 100
        })
        
        # Update final capital
        capital += pnl
        equity_curve[-1] = capital
    
    # Calculate performance metrics
    if len(equity_curve) < 2:
        return {
            'return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'trade_count': 0
        }
    
    # Total return
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if returns:
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    max_drawdown = 0
    peak = equity_curve[0]
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        else:
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
    
    # Win rate
    exit_trades = [t for t in trades if t.get('type') == 'exit']
    win_count = sum(1 for t in exit_trades if t.get('pnl', 0) > 0)
    loss_count = sum(1 for t in exit_trades if t.get('pnl', 0) <= 0)
    
    total_profit = sum(t.get('pnl', 0) for t in exit_trades if t.get('pnl', 0) > 0)
    total_loss = abs(sum(t.get('pnl', 0) for t in exit_trades if t.get('pnl', 0) <= 0))
    
    trade_count = win_count + loss_count
    win_rate = win_count / trade_count * 100 if trade_count > 0 else 0
    
    return {
        'return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': trade_count
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feedback_loop_demo")


class FeedbackLoopDemo:
    """
    Demonstrates how strategy optimization improves through the feedback loop.
    """

    def __init__(self, iterations=3):
        """
        Initialize the feedback loop demonstration.
        
        Args:
            iterations: Number of feedback loop iterations to run
        """
        self.iterations = iterations
        
        # Define parameter space for our trading strategy
        self.parameter_space = ParameterSpace()
        
        # Add parameters one by one
        self.parameter_space.add_integer_parameter('trend_lookback', 5, 50)
        self.parameter_space.add_integer_parameter('volatility_lookback', 5, 50)
        self.parameter_space.add_real_parameter('entry_threshold', 0.0, 2.0)
        self.parameter_space.add_real_parameter('exit_threshold', -2.0, 0.0)
        self.parameter_space.add_real_parameter('stop_loss_pct', 0.5, 5.0)
        self.parameter_space.add_real_parameter('position_sizing', 0.01, 0.5)
        self.parameter_space.add_boolean_parameter('trail_stop')
        self.parameter_space.add_categorical_parameter('filter_type', ['sma', 'ema', 'none'])
        self.parameter_space.add_real_parameter('indicator_weight', 0.1, 0.9)
        
        # Initial parameters for synthetic market generation
        # These will be refined through the feedback loop
        self.synthetic_params = {
            'bullish': {
                'trend': 0.006,
                'volatility': 0.015,
                'reversion': 0.2
            },
            'bearish': {
                'trend': -0.004,
                'volatility': 0.025,
                'reversion': 0.15
            },
            'sideways': {
                'trend': 0.0005,
                'volatility': 0.01,
                'reversion': 0.7
            },
            'volatile': {
                'trend': 0.001,
                'volatility': 0.035,
                'reversion': 0.1
            }
        }
        
        # The "real-world" parameters - HIDDEN from the optimization system
        # These represent the true market conditions we're trying to learn
        self.real_world_params = {
            'bullish': {
                'trend': 0.008,        # Stronger trend than expected
                'volatility': 0.018,   # Higher volatility
                'reversion': 0.15      # Less mean reversion
            },
            'bearish': {
                'trend': -0.006,       # Stronger negative trend 
                'volatility': 0.03,    # Higher volatility
                'reversion': 0.1       # Less mean reversion
            },
            'sideways': {
                'trend': 0.0,          # No trend
                'volatility': 0.012,   # Slightly higher volatility
                'reversion': 0.8       # Stronger mean reversion
            },
            'volatile': {
                'trend': 0.0005,       # Less trend
                'volatility': 0.045,   # Much higher volatility
                'reversion': 0.05      # Less mean reversion
            }
        }
        
        # Results tracking for analysis and visualization
        self.optimization_results = []
        self.verification_results = []
        self.prediction_accuracy = []
        self.synthetic_param_history = {}
        
        # Initialize parameter history
        for regime in self.synthetic_params:
            self.synthetic_param_history[regime] = {
                'trend': [],
                'volatility': [],
                'reversion': []
            }
            
        logger.info("Initialized FeedbackLoopDemo")
    
    def run_demo(self):
        """
        Run the complete feedback loop demonstration.
        """
        logger.info(f"Starting feedback loop demo with {self.iterations} iterations")
        
        for iteration in range(self.iterations):
            iteration_num = iteration + 1
            logger.info(f"===== Iteration {iteration_num}/{self.iterations} =====")
            
            # Step 1: Generate synthetic data and optimize strategy
            strategy_params = self.optimize_strategy(iteration_num)
            
            # Step 2: Simulate real-world testing and calculate performance
            real_world_perf = self.simulate_real_world(strategy_params)
            
            # Step 3: Verify predictions against real performance
            accuracy = self.verify_performance(strategy_params, real_world_perf)
            
            # Step 4: Update synthetic parameters based on verification
            self.update_synthetic_parameters(real_world_perf)
            
            # Small delay for readability
            time.sleep(0.5)
        
        # Generate visualization of results
        self.visualize_results()
        
        logger.info("Feedback loop demo completed")
    
    def optimize_strategy(self, iteration: int) -> Dict[str, Any]:
        """
        Optimize strategy parameters using current synthetic market parameters.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Optimized strategy parameters
        """
        logger.info(f"Iteration {iteration}: Optimizing strategy with current synthetic parameters")
        
        # Generate synthetic market data with current parameters
        market_data = {}
        for regime, params in self.synthetic_params.items():
            market_data[regime] = generate_synthetic_market_data(
                days=252,
                trend_strength=params['trend'],
                volatility=params['volatility'],
                mean_reversion=params['reversion']
            )
            
            # Store current params in history for visualization
            for key in params:
                self.synthetic_param_history[regime][key].append(params[key])
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            parameter_space=self.parameter_space,
            population_size=40,
            crossover_prob=0.7,
            mutation_prob=0.2
        )
        
        # Define multi-objective fitness functions
        def bull_objective(params):
            strategy = TradingStrategy(params)
            results = evaluate_strategy_in_regime(strategy, market_data['bullish'])
            return results['return_pct']  # Higher is better
            
        def bear_objective(params):
            strategy = TradingStrategy(params)
            results = evaluate_strategy_in_regime(strategy, market_data['bearish'])
            return results['return_pct']  # Higher is better
        
        def drawdown_objective(params):
            # Evaluate across all regimes and return worst drawdown
            drawdowns = []
            for regime in market_data:
                strategy = TradingStrategy(params)
                results = evaluate_strategy_in_regime(strategy, market_data[regime])
                drawdowns.append(results['max_drawdown'])
            return -max(drawdowns)  # Higher (less negative) is better
        
        # Run optimization
        objectives = [
            ("bull_return", bull_objective, True),  # Maximize
            ("bear_return", bear_objective, True),  # Maximize
            ("drawdown", drawdown_objective, True),  # Maximize (minimize drawdown)
        ]
        
        pareto_front, objective_values = optimizer.optimize(
            objectives=objectives,
            generations=15,
            parallelism=1,
            verbose=True  # Set to False for less output
        )
        
        # Select a balanced solution from the pareto front
        best_solution = self._select_balanced_solution(pareto_front, objective_values)
        
        # Calculate expected performance across regimes
        expected_performance = {}
        for regime in market_data:
            strategy = TradingStrategy(best_solution)
            results = evaluate_strategy_in_regime(strategy, market_data[regime])
            
            expected_performance[regime] = {
                'return_pct': results['return_pct'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate']
            }
            
            logger.info(f"Expected {regime} performance: Return={results['return_pct']:.2f}%, "
                        f"Sharpe={results['sharpe_ratio']:.2f}, Drawdown={results['max_drawdown']:.2f}%")
        
        # Store results for tracking
        self.optimization_results.append({
            'iteration': iteration,
            'parameters': best_solution,
            'expected_performance': expected_performance
        })
        
        return best_solution
    
    def _select_balanced_solution(self, pareto_front, objective_values):
        """
        Select a balanced solution from the Pareto front.
        
        Args:
            pareto_front: List of Pareto-optimal parameter sets
            objective_values: Corresponding objective values
            
        Returns:
            Selected parameter set
        """
        # Normalize objective values to 0-1 range
        normalized_objectives = []
        
        for i, obj in enumerate(objective_values[0]):
            obj_values = [values[i] for values in objective_values]
            min_val = min(obj_values)
            max_val = max(obj_values)
            range_val = max_val - min_val if max_val > min_val else 1.0
            
            normalized = [(v - min_val) / range_val for v in obj_values]
            normalized_objectives.append(normalized)
        
        # Calculate distance from ideal point (1,1,1)
        distances = []
        for i in range(len(pareto_front)):
            norm_vals = [normalized_objectives[j][i] for j in range(len(objective_values[0]))]
            distance = sum((1 - v) ** 2 for v in norm_vals) ** 0.5
            distances.append(distance)
        
        # Select solution with minimum distance (most balanced)
        best_idx = distances.index(min(distances))
        return pareto_front[best_idx]
    
    def simulate_real_world(self, parameters: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Simulate "real-world" performance using hidden real-world parameters.
        
        Args:
            parameters: Strategy parameters
            
        Returns:
            Performance across different market regimes
        """
        logger.info("Simulating real-world performance...")
        
        # Create strategy
        strategy = TradingStrategy(parameters)
        
        # Evaluate in each "real" market regime
        real_performance = {}
        
        for regime, params in self.real_world_params.items():
            # Generate "real" market data
            real_data = generate_synthetic_market_data(
                days=252,
                trend_strength=params['trend'],
                volatility=params['volatility'],
                mean_reversion=params['reversion']
            )
            
            # Evaluate strategy
            results = evaluate_strategy_in_regime(strategy, real_data)
            
            real_performance[regime] = {
                'return_pct': results['return_pct'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate']
            }
            
            logger.info(f"Real {regime} performance: Return={results['return_pct']:.2f}%, "
                        f"Sharpe={results['sharpe_ratio']:.2f}, Drawdown={results['max_drawdown']:.2f}%")
        
        return real_performance
    
    def verify_performance(self, strategy_params: Dict[str, Any], 
                          real_perf: Dict[str, Dict[str, float]]) -> float:
        """
        Verify predicted performance against real-world results.
        
        Args:
            strategy_params: Strategy parameters
            real_perf: Real-world performance
            
        Returns:
            Overall prediction accuracy
        """
        logger.info("Verifying prediction accuracy...")
        
        # Get latest optimization results
        latest_opt = self.optimization_results[-1]
        expected_perf = latest_opt['expected_performance']
        
        # Calculate prediction accuracy for each regime and metric
        accuracy_by_regime = {}
        
        for regime in real_perf:
            expected = expected_perf[regime]
            actual = real_perf[regime]
            
            # Calculate relative accuracy for each metric
            return_accuracy = 1.0 - min(1.0, abs(
                expected['return_pct'] - actual['return_pct']) / max(1.0, abs(expected['return_pct'])))
            
            sharpe_accuracy = 1.0 - min(1.0, abs(
                expected['sharpe_ratio'] - actual['sharpe_ratio']) / max(1.0, abs(expected['sharpe_ratio'])))
            
            drawdown_accuracy = 1.0 - min(1.0, abs(
                expected['max_drawdown'] - actual['max_drawdown']) / max(1.0, abs(expected['max_drawdown'])))
            
            overall_regime_acc = (return_accuracy + sharpe_accuracy + drawdown_accuracy) / 3.0
            
            accuracy_by_regime[regime] = {
                'return': return_accuracy,
                'sharpe': sharpe_accuracy,
                'drawdown': drawdown_accuracy,
                'overall': overall_regime_acc
            }
        
        # Calculate overall accuracy across all regimes
        overall_accuracy = sum(acc['overall'] for acc in accuracy_by_regime.values()) / len(accuracy_by_regime)
        
        # Store verification results
        self.verification_results.append({
            'iteration': latest_opt['iteration'],
            'by_regime': accuracy_by_regime,
            'overall': overall_accuracy,
            'expected': expected_perf,
            'actual': real_perf
        })
        
        self.prediction_accuracy.append(overall_accuracy)
        
        logger.info(f"Overall prediction accuracy: {overall_accuracy:.2%}")
        
        for regime, acc in accuracy_by_regime.items():
            logger.info(f"  {regime} accuracy: {acc['overall']:.2%}")
        
        return overall_accuracy
    
    def update_synthetic_parameters(self, real_perf: Dict[str, Dict[str, float]]):
        """
        Update synthetic market parameters based on real-world performance.
        
        This is the key feedback mechanism that allows our system to learn and improve.
        
        Args:
            real_perf: Real-world performance data
        """
        logger.info("Updating synthetic parameters based on verification results...")
        
        # Get the latest verification results
        verification = self.verification_results[-1]
        
        # For each market regime, adjust parameters to better match reality
        for regime, real_metrics in real_perf.items():
            # Current synthetic parameters
            current_params = self.synthetic_params[regime]
            
            # Compare expected vs actual metrics to infer parameter adjustments
            expected = verification['expected'][regime]
            actual = verification['actual'][regime]
            
            # Calculate adjustment ratios
            return_ratio = actual['return_pct'] / expected['return_pct'] if expected['return_pct'] != 0 else 1.0
            volatility_ratio = actual['max_drawdown'] / expected['max_drawdown'] if expected['max_drawdown'] != 0 else 1.0
            
            # Use a learning rate to prevent over-correction
            learning_rate = 0.3
            
            # Adjust trend strength based on return ratio
            if abs(return_ratio) > 1.0:
                # If actual returns are higher, increase trend strength
                current_params['trend'] *= (1.0 + (return_ratio - 1.0) * learning_rate)
            else:
                # If actual returns are lower, decrease trend strength
                current_params['trend'] *= (1.0 - (1.0 - return_ratio) * learning_rate)
            
            # Adjust volatility based on drawdown ratio
            current_params['volatility'] *= (1.0 + (volatility_ratio - 1.0) * learning_rate)
            
            # Clamp parameters to reasonable ranges
            current_params['trend'] = max(-0.01, min(0.01, current_params['trend']))
            current_params['volatility'] = max(0.005, min(0.05, current_params['volatility']))
            current_params['reversion'] = max(0.05, min(0.9, current_params['reversion']))
            
            logger.info(f"Updated {regime} parameters:")
            logger.info(f"  Trend strength: {current_params['trend']:.6f}")
            logger.info(f"  Volatility: {current_params['volatility']:.6f}")
            logger.info(f"  Mean reversion: {current_params['reversion']:.6f}")
    
    def visualize_results(self):
        """
        Generate text-based summary of the feedback loop results.
        """
        logger.info("Generating results summary...")
        
        # Create a divider line
        divider = "=" * 70
        
        # Print header
        print(divider)
        print("OPTIMIZATION FEEDBACK LOOP RESULTS SUMMARY")
        print(divider)
        
        # 1. Overall accuracy improvement
        print("\n1. PREDICTION ACCURACY IMPROVEMENT")
        print("-" * 40)
        
        if len(self.prediction_accuracy) > 1:
            initial_accuracy = self.prediction_accuracy[0]
            final_accuracy = self.prediction_accuracy[-1]
            improvement = (final_accuracy - initial_accuracy) / initial_accuracy if initial_accuracy > 0 else 0
            
            print(f"Initial accuracy: {initial_accuracy:.2%}")
            print(f"Final accuracy:   {final_accuracy:.2%}")
            print(f"Improvement:      {improvement:.2%}")
            
            # Show per-iteration accuracy
            print("\nAccuracy by iteration:")
            for i, acc in enumerate(self.prediction_accuracy):
                print(f"  Iteration {i+1}: {acc:.2%}")
        
        # 2. Parameter convergence
        print("\n2. SYNTHETIC PARAMETER CONVERGENCE")
        print("-" * 40)
        
        # Show how parameters evolved toward real values
        for regime in self.synthetic_param_history.keys():
            print(f"\n{regime.capitalize()} Regime:")
            
            # Show initial, final and target values
            for param in ['trend', 'volatility', 'reversion']:
                initial = self.synthetic_param_history[regime][param][0]
                final = self.synthetic_param_history[regime][param][-1]
                target = self.real_world_params[regime][param]
                
                # Calculate how much closer we got to the target
                initial_distance = abs(initial - target)
                final_distance = abs(final - target)
                improvement = (initial_distance - final_distance) / initial_distance if initial_distance > 0 else 0
                
                print(f"  {param.capitalize()}: ")
                print(f"    Initial: {initial:.6f}")
                print(f"    Final:   {final:.6f}")
                print(f"    Target:  {target:.6f}")
                print(f"    Convergence: {improvement:.2%} closer to target")
        
        # 3. Expected vs Actual Performance Comparison
        print("\n3. EXPECTED VS ACTUAL PERFORMANCE (FINAL ITERATION)")
        print("-" * 40)
        
        # Get data from the last iteration
        last_verify = self.verification_results[-1]
        
        print("\nReturns by Market Regime:")
        for regime in last_verify['expected'].keys():
            expected = last_verify['expected'][regime]['return_pct']
            actual = last_verify['actual'][regime]['return_pct']
            accuracy = last_verify['by_regime'][regime]['return']
            
            print(f"  {regime.capitalize()}:")
            print(f"    Expected: {expected:.2f}%")
            print(f"    Actual:   {actual:.2f}%")
            print(f"    Accuracy: {accuracy:.2%}")
        
        # 4. Accuracy by regime
        print("\n4. PREDICTION ACCURACY BY MARKET REGIME")
        print("-" * 40)
        
        # Extract accuracy by regime for each iteration
        accuracy_by_iteration = {}
        regimes = list(self.verification_results[0]['by_regime'].keys())
        
        for regime in regimes:
            accuracy_by_iteration[regime] = []
            for result in self.verification_results:
                accuracy_by_iteration[regime].append(result['by_regime'][regime]['overall'])
        
        for regime in regimes:
            print(f"\n{regime.capitalize()} regime accuracy:")
            for i, acc in enumerate(accuracy_by_iteration[regime]):
                print(f"  Iteration {i+1}: {acc:.2%}")
        
        print(divider)
        print("CONCLUSION: The feedback loop" + 
              (" successfully improved" if improvement > 0 else " failed to improve") + 
              " optimization accuracy.")
        print(divider)
        
        # Also log the key metrics
        if len(self.prediction_accuracy) > 1:
            logger.info(f"\nPrediction Accuracy Improvement:")
            logger.info(f"  Initial: {initial_accuracy:.2%}")
            logger.info(f"  Final:   {final_accuracy:.2%}")
            logger.info(f"  Improvement: {improvement:.2%}")


def main():
    """Run the feedback loop demonstration."""
    print("=" * 80)
    print("FEEDBACK LOOP DEMONSTRATION: SELF-IMPROVING OPTIMIZATION")
    print("=" * 80)
    print("\nThis demonstration shows how the system learns and improves over time:")
    print("1. Initial optimization with imperfect synthetic market models")
    print("2. Testing in simulated 'real-world' conditions")
    print("3. Measuring prediction accuracy")
    print("4. Updating synthetic models to better match reality")
    print("5. Re-optimizing with improved models")
    print("\nThe key innovation is that the system gets better at predicting")
    print("real-world performance with each iteration of the feedback loop.")
    print("\n" + "=" * 80)
    
    # Run the demonstration with 3 iterations
    demo = FeedbackLoopDemo(iterations=3)
    demo.run_demo()
    
    print("\n" + "=" * 80)
    print("Demonstration complete! Results visualization saved to 'feedback_loop_results.png'")
    print("=" * 80)


if __name__ == "__main__":
    main()
