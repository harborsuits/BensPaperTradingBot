#!/usr/bin/env python3
"""
Simplified Feedback Loop Demonstration

This script demonstrates the core concept of a self-improving optimization system:
1. Initial optimization with imperfect market models
2. Real-world testing and verification
3. Parameter adjustment based on real-world feedback
4. Re-optimization with improved parameters
5. Measurement of prediction accuracy improvement over iterations

The key innovation is a system that gets better at predicting real-world performance
by continuously learning from the gap between predictions and actual results.
"""

import random
import time
import math
from typing import Dict, List, Any, Tuple

# Set random seed for reproducibility
random.seed(42)

class SimpleStrategy:
    """A simplified trading strategy with parameters to optimize."""
    
    def __init__(self, params: Dict[str, float]):
        """Initialize strategy with parameters."""
        # Extract parameters with defaults if not specified
        self.lookback = int(params.get('lookback', 20))
        self.threshold = params.get('threshold', 0.5)
        self.stop_loss = params.get('stop_loss', 2.0)
        self.position_size = params.get('position_size', 0.1)
        
        # Strategy state
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
    
    def reset(self):
        """Reset strategy state."""
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
    
    def calculate_signal(self, prices: List[float]) -> float:
        """Generate a trading signal based on price data."""
        if len(prices) < self.lookback:
            return 0
        
        # Calculate short-term momentum
        short_term = prices[-1] / prices[-5] - 1
        
        # Calculate long-term trend
        long_term = prices[-1] / prices[-self.lookback] - 1
        
        # Combine signals
        signal = short_term * 0.7 + long_term * 0.3
        
        return signal * 100  # Convert to percentage
    
    def backtest(self, prices: List[float]) -> Dict[str, float]:
        """Run backtest on price data and return performance metrics."""
        self.reset()
        capital = 10000.0
        self.equity_curve = [capital]
        
        min_bars = self.lookback + 1
        
        for i in range(min_bars, len(prices)):
            price_window = prices[:i+1]
            current_price = price_window[-1]
            
            # Calculate signal
            signal = self.calculate_signal(price_window)
            
            # Execute trading logic
            if self.position == 0 and signal > self.threshold:
                # Enter long position
                shares = capital * self.position_size / current_price
                self.position = shares
                self.entry_price = current_price
                
                self.trades.append({
                    'type': 'entry',
                    'price': current_price,
                    'shares': shares
                })
                
            elif self.position > 0:
                # Check stop loss
                if current_price < self.entry_price * (1 - self.stop_loss/100):
                    # Exit at stop loss
                    exit_value = self.position * current_price
                    entry_value = self.position * self.entry_price
                    pnl = exit_value - entry_value
                    
                    self.trades.append({
                        'type': 'exit',
                        'price': current_price,
                        'shares': self.position,
                        'pnl': pnl
                    })
                    
                    capital += pnl
                    self.position = 0
                    
                # Check for exit signal
                elif signal < -self.threshold:
                    # Exit on signal
                    exit_value = self.position * current_price
                    entry_value = self.position * self.entry_price
                    pnl = exit_value - entry_value
                    
                    self.trades.append({
                        'type': 'exit',
                        'price': current_price,
                        'shares': self.position,
                        'pnl': pnl
                    })
                    
                    capital += pnl
                    self.position = 0
            
            # Update equity
            if self.position > 0:
                equity = capital + self.position * current_price
            else:
                equity = capital
                
            self.equity_curve.append(equity)
        
        # Close any open position at the end
        if self.position > 0:
            final_price = prices[-1]
            exit_value = self.position * final_price
            entry_value = self.position * self.entry_price
            pnl = exit_value - entry_value
            
            self.trades.append({
                'type': 'exit',
                'price': final_price,
                'shares': self.position,
                'pnl': pnl
            })
            
            capital += pnl
        
        # Calculate metrics
        if len(self.equity_curve) < 2:
            return {'return': 0, 'max_drawdown': 0, 'sharpe': 0, 'trade_count': 0}
        
        # Return percentage
        total_return = (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = self.equity_curve[0]
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate daily returns for Sharpe
        daily_returns = [(self.equity_curve[i] / self.equity_curve[i-1] - 1) 
                         for i in range(1, len(self.equity_curve))]
        
        # Sharpe ratio (annualized)
        if daily_returns:
            avg_return = sum(daily_returns) / len(daily_returns)
            std_return = math.sqrt(sum((r - avg_return)**2 for r in daily_returns) / len(daily_returns))
            sharpe = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Count trades
        exit_trades = [t for t in self.trades if t.get('type') == 'exit']
        trade_count = len(exit_trades)
        
        # Count winning trades
        win_count = sum(1 for t in exit_trades if t.get('pnl', 0) > 0)
        win_rate = win_count / trade_count * 100 if trade_count > 0 else 0
        
        return {
            'return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'trade_count': trade_count,
            'win_rate': win_rate
        }


def generate_market_data(days=252, trend=0.0005, volatility=0.01, mean_reversion=0.1):
    """Generate synthetic price data with given parameters.
    
    Args:
        days: Number of trading days to simulate
        trend: Daily drift (trend strength)
        volatility: Daily volatility
        mean_reversion: Strength of mean reversion (0-1)
        
    Returns:
        List of prices
    """
    prices = [100.0]  # Start with $100
    
    for _ in range(days):
        # Current price
        current = prices[-1]
        
        # Mean reversion component (pull toward starting price)
        reversion = (100 - current) * mean_reversion / 100
        
        # Random component based on volatility
        noise = random.normalvariate(0, volatility)
        
        # Calculate price move
        move = trend + reversion + noise
        
        # Add new price
        new_price = current * (1 + move)
        prices.append(new_price)
    
    return prices


def simple_optimizer(param_ranges, fitness_fn, iterations=100, population_size=20):
    """A simple random search optimizer.
    
    Args:
        param_ranges: Dictionary of parameter ranges (min, max)
        fitness_fn: Function that evaluates a parameter set
        iterations: Number of iterations to run
        population_size: Number of parameter sets to try per iteration
        
    Returns:
        Tuple of (best parameters, best fitness)
    """
    best_params = None
    best_fitness = float('-inf')
    
    for _ in range(iterations):
        # Generate random parameters
        for _ in range(population_size):
            params = {}
            for param, (min_val, max_val) in param_ranges.items():
                params[param] = random.uniform(min_val, max_val)
            
            # Evaluate fitness
            fitness = fitness_fn(params)
            
            # Update best if better
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = params.copy()
    
    return best_params, best_fitness


class FeedbackLoopDemo:
    """Demonstration of the optimization feedback loop."""
    
    def __init__(self, iterations=3):
        """Initialize the feedback loop demo.
        
        Args:
            iterations: Number of feedback loop iterations to run
        """
        self.iterations = iterations
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'lookback': (10, 50),
            'threshold': (0.2, 2.0),
            'stop_loss': (1.0, 5.0),
            'position_size': (0.05, 0.3)
        }
        
        # Synthetic market parameters (our model of the market)
        self.synthetic_params = {
            'trend': 0.0005,      # Initial estimate
            'volatility': 0.01,   # Initial estimate
            'mean_reversion': 0.1 # Initial estimate
        }
        
        # "Real" market parameters (this would be the actual market, unknown to us)
        # In this demo, we simulate this with different parameters
        self.real_params = {
            'trend': 0.0008,      # Stronger trend than we estimated
            'volatility': 0.012,  # Higher volatility than we estimated
            'mean_reversion': 0.2 # Stronger mean reversion than we estimated
        }
        
        # Track metrics for analysis
        self.parameter_history = []
        self.prediction_accuracy = []
        self.expected_performance = []
        self.actual_performance = []
        
        print("Feedback Loop Demo initialized")
        print("Initial synthetic market parameters:")
        for param, value in self.synthetic_params.items():
            print(f"  {param}: {value:.6f}")
        print("\nReal market parameters (unknown to the optimizer):")
        for param, value in self.real_params.items():
            print(f"  {param}: {value:.6f}")
        print("\n" + "=" * 70)
    
    def run_demo(self):
        """Run the feedback loop demonstration."""
        print("\nStarting feedback loop demonstration...")
        
        for i in range(self.iterations):
            iteration = i + 1
            print(f"\n=== Iteration {iteration}/{self.iterations} ===\n")
            
            # Step 1: Optimize strategy using current synthetic parameters
            strategy_params = self.optimize_strategy()
            
            # Step 2: Test in "real-world" conditions
            real_performance = self.test_real_world(strategy_params)
            
            # Step 3: Verify predictions vs actual performance
            accuracy = self.verify_predictions(real_performance)
            
            # Step 4: Update synthetic parameters
            self.update_synthetic_parameters()
            
            # Store parameters for tracking
            self.parameter_history.append(self.synthetic_params.copy())
            
            # Small delay for readability
            time.sleep(0.5)
        
        # Show summary of results
        self.show_results()
    
    def optimize_strategy(self):
        """Optimize strategy parameters using current synthetic market model.
        
        Returns:
            Optimized strategy parameters
        """
        print("Optimizing strategy with current synthetic market parameters...")
        
        # Generate synthetic market data
        market_data = generate_market_data(
            days=252,
            trend=self.synthetic_params['trend'],
            volatility=self.synthetic_params['volatility'],
            mean_reversion=self.synthetic_params['mean_reversion']
        )
        
        # Define fitness function for optimization
        def fitness_function(params):
            strategy = SimpleStrategy(params)
            metrics = strategy.backtest(market_data)
            
            # Combine metrics into single fitness score
            # Balance return, drawdown, and sharpe ratio
            fitness = (
                metrics['return'] / 2 -        # Want high returns
                metrics['max_drawdown'] / 2 +  # Want low drawdown
                metrics['sharpe'] * 5          # Want high Sharpe
            )
            
            return fitness
        
        # Run optimization
        best_params, best_fitness = simple_optimizer(
            self.param_ranges,
            fitness_function,
            iterations=50,
            population_size=20
        )
        
        # Evaluate performance of best parameters
        best_strategy = SimpleStrategy(best_params)
        expected_metrics = best_strategy.backtest(market_data)
        
        print("Optimization complete!")
        print("Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.4f}")
            
        print("\nExpected performance in synthetic market:")
        print(f"  Return: {expected_metrics['return']:.2f}%")
        print(f"  Max Drawdown: {expected_metrics['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {expected_metrics['sharpe']:.2f}")
        print(f"  Trade Count: {expected_metrics['trade_count']}")
        print(f"  Win Rate: {expected_metrics['win_rate']:.2f}%")
        
        # Store for comparison
        self.expected_performance.append(expected_metrics)
        
        return best_params
    
    def test_real_world(self, strategy_params):
        """Test strategy in "real-world" conditions.
        
        In a real system, this would be live trading or out-of-sample testing.
        For this demo, we simulate it using different market parameters.
        
        Args:
            strategy_params: Optimized strategy parameters
            
        Returns:
            Performance metrics
        """
        print("\nTesting strategy in 'real-world' conditions...")
        
        # Generate "real" market data
        real_market_data = generate_market_data(
            days=252,
            trend=self.real_params['trend'],
            volatility=self.real_params['volatility'],
            mean_reversion=self.real_params['mean_reversion']
        )
        
        # Evaluate strategy on real data
        strategy = SimpleStrategy(strategy_params)
        real_metrics = strategy.backtest(real_market_data)
        
        print("Real-world performance:")
        print(f"  Return: {real_metrics['return']:.2f}%")
        print(f"  Max Drawdown: {real_metrics['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {real_metrics['sharpe']:.2f}")
        print(f"  Trade Count: {real_metrics['trade_count']}")
        print(f"  Win Rate: {real_metrics['win_rate']:.2f}%")
        
        # Store for comparison
        self.actual_performance.append(real_metrics)
        
        return real_metrics
    
    def verify_predictions(self, real_metrics):
        """Verify prediction accuracy by comparing expected vs actual performance.
        
        Args:
            real_metrics: Real-world performance metrics
            
        Returns:
            Overall prediction accuracy
        """
        print("\nVerifying prediction accuracy...")
        
        # Get expected metrics from most recent optimization
        expected = self.expected_performance[-1]
        
        # Calculate accuracy for key metrics
        return_accuracy = 1.0 - min(1.0, abs(expected['return'] - real_metrics['return']) 
                                   / max(1.0, abs(expected['return'])))
        
        drawdown_accuracy = 1.0 - min(1.0, abs(expected['max_drawdown'] - real_metrics['max_drawdown']) 
                                     / max(1.0, abs(expected['max_drawdown'])))
        
        sharpe_accuracy = 1.0 - min(1.0, abs(expected['sharpe'] - real_metrics['sharpe']) 
                                   / max(1.0, abs(expected['sharpe'])))
        
        # Overall accuracy (average of key metrics)
        overall_accuracy = (return_accuracy + drawdown_accuracy + sharpe_accuracy) / 3
        
        print("Prediction accuracy:")
        print(f"  Return accuracy: {return_accuracy:.2%}")
        print(f"  Drawdown accuracy: {drawdown_accuracy:.2%}")
        print(f"  Sharpe accuracy: {sharpe_accuracy:.2%}")
        print(f"  Overall accuracy: {overall_accuracy:.2%}")
        
        # Store for tracking
        self.prediction_accuracy.append(overall_accuracy)
        
        return overall_accuracy
    
    def update_synthetic_parameters(self):
        """Update synthetic market parameters based on verification results.
        
        This is the key feedback mechanism that allows the system to learn.
        """
        print("\nUpdating synthetic market parameters...")
        
        # Get the most recent expected and actual performance
        expected = self.expected_performance[-1]
        actual = self.actual_performance[-1]
        
        # Calculate adjustment ratios
        return_ratio = actual['return'] / expected['return'] if expected['return'] != 0 else 1.0
        volatility_ratio = actual['max_drawdown'] / expected['max_drawdown'] if expected['max_drawdown'] != 0 else 1.0
        
        # Use a learning rate to prevent over-correction
        learning_rate = 0.3
        
        # Update trend strength based on return difference
        if abs(return_ratio) > 1.0:
            # If actual returns are higher, increase trend strength
            self.synthetic_params['trend'] *= (1.0 + (return_ratio - 1.0) * learning_rate)
        else:
            # If actual returns are lower, decrease trend strength
            self.synthetic_params['trend'] *= (1.0 - (1.0 - return_ratio) * learning_rate)
        
        # Update volatility based on drawdown difference
        self.synthetic_params['volatility'] *= (1.0 + (volatility_ratio - 1.0) * learning_rate)
        
        # Adjust mean reversion based on both return and drawdown
        if return_ratio < 1.0 and volatility_ratio > 1.0:
            # Lower returns and higher drawdowns might indicate more mean reversion
            self.synthetic_params['mean_reversion'] *= (1.0 + learning_rate)
        elif return_ratio > 1.0 and volatility_ratio < 1.0:
            # Higher returns and lower drawdowns might indicate less mean reversion
            self.synthetic_params['mean_reversion'] *= (1.0 - learning_rate)
        
        # Clamp parameters to reasonable ranges
        self.synthetic_params['trend'] = max(-0.001, min(0.002, self.synthetic_params['trend']))
        self.synthetic_params['volatility'] = max(0.005, min(0.03, self.synthetic_params['volatility']))
        self.synthetic_params['mean_reversion'] = max(0.05, min(0.5, self.synthetic_params['mean_reversion']))
        
        print("Updated synthetic parameters:")
        for param, value in self.synthetic_params.items():
            print(f"  {param}: {value:.6f}")
    
    def show_results(self):
        """Show summary of results from the feedback loop iterations."""
        print("\n" + "=" * 70)
        print("FEEDBACK LOOP RESULTS SUMMARY")
        print("=" * 70)
        
        # Show accuracy improvement
        print("\n1. PREDICTION ACCURACY IMPROVEMENT")
        print("-" * 40)
        
        for i, accuracy in enumerate(self.prediction_accuracy):
            print(f"Iteration {i+1}: {accuracy:.2%}")
        
        if len(self.prediction_accuracy) > 1:
            initial = self.prediction_accuracy[0]
            final = self.prediction_accuracy[-1]
            improvement = (final - initial) / initial if initial > 0 else 0
            
            print(f"\nInitial accuracy: {initial:.2%}")
            print(f"Final accuracy: {final:.2%}")
            print(f"Improvement: {improvement:.2%}")
        
        # Show parameter convergence
        print("\n2. SYNTHETIC PARAMETER CONVERGENCE")
        print("-" * 40)
        
        for param in ['trend', 'volatility', 'mean_reversion']:
            print(f"\n{param.capitalize()} parameter:")
            
            # Initial and target values
            initial = self.parameter_history[0][param]
            target = self.real_params[param]
            final = self.parameter_history[-1][param]
            
            # Calculate convergence
            initial_distance = abs(initial - target)
            final_distance = abs(final - target)
            convergence = (initial_distance - final_distance) / initial_distance if initial_distance > 0 else 0
            
            print(f"  Initial value: {initial:.6f}")
            print(f"  Final value: {final:.6f}")
            print(f"  Target value: {target:.6f}")
            print(f"  Convergence: {convergence:.2%}")
        
        # Overall conclusion
        print("\n" + "=" * 70)
        
        if self.prediction_accuracy[-1] > self.prediction_accuracy[0]:
            print("CONCLUSION: The feedback loop successfully improved optimization accuracy!")
            print(f"Prediction accuracy increased by {improvement:.2%} over {len(self.prediction_accuracy)} iterations.")
        else:
            print("CONCLUSION: The feedback loop did not improve optimization accuracy.")
        
        print("=" * 70)


def main():
    """Run the feedback loop demonstration."""
    print("=" * 70)
    print("OPTIMIZATION FEEDBACK LOOP DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration shows how a trading strategy optimization")
    print("system can improve itself by learning from real-world feedback.")
    print("\nThe key innovation is that our system gets better at predicting")
    print("real-world performance with each iteration of the feedback loop.")
    print("\n" + "=" * 70)
    
    # Run the demo with 3 iterations
    demo = FeedbackLoopDemo(iterations=3)
    demo.run_demo()


if __name__ == "__main__":
    main()
