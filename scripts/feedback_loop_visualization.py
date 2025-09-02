#!/usr/bin/env python3
"""
Feedback Loop Visualization Utilities

This module provides visualization tools for monitoring the performance of
the optimization feedback loop. It focuses on three key aspects:
1. Parameter convergence - How synthetic parameters evolve toward real values
2. Prediction accuracy - How accuracy improves over iterations
3. Performance comparison - How the gap between expected and actual performance closes

These visualizations help to validate the effectiveness of the feedback loop
and provide insights into the learning process.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def plot_parameter_convergence(param_history: List[Dict[str, Dict[str, float]]], 
                              real_params: Dict[str, Dict[str, float]],
                              output_file: str = "parameter_convergence.png"):
    """
    Plot convergence of synthetic parameters toward real values.
    
    Args:
        param_history: List of parameter dictionaries by iteration
        real_params: Dictionary of real parameter values to converge toward
        output_file: Path to save the output figure
    """
    if not param_history:
        print("No parameter history available for plotting")
        return
    
    # Extract regimes and parameters from the history
    regimes = list(param_history[0].keys())
    params = list(param_history[0][regimes[0]].keys())
    
    # Create figure with subplots - one row per parameter, one column per regime
    fig, axes = plt.subplots(len(params), len(regimes), figsize=(5*len(regimes), 4*len(params)))
    
    # Add super title
    fig.suptitle("Convergence of Synthetic Market Parameters", fontsize=20)
    
    # If only one parameter, wrap axes in a list
    if len(params) == 1:
        if len(regimes) == 1:
            axes = np.array([[axes]])
        else:
            axes = np.array([axes])
    
    # For each parameter and regime, plot convergence
    for i, param in enumerate(params):
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            
            # Extract parameter values across iterations
            values = [hist[regime][param] for hist in param_history]
            iterations = list(range(1, len(values) + 1))
            
            # Plot parameter convergence
            ax.plot(iterations, values, 'o-', color='blue', label='Synthetic')
            
            # Add target line for real parameter
            target = real_params[regime][param]
            ax.axhline(y=target, color='red', linestyle='--', label='Real')
            
            # Calculate convergence percentage
            initial_distance = abs(values[0] - target)
            final_distance = abs(values[-1] - target)
            
            if initial_distance > 0:
                convergence = (initial_distance - final_distance) / initial_distance
                convergence_text = f"Convergence: {convergence:.1%}"
            else:
                convergence_text = "Perfect initial guess"
            
            # Add annotation
            ax.annotate(convergence_text, xy=(0.05, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Set title and labels
            ax.set_title(f"{regime.capitalize()}: {param}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"{param} Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"Parameter convergence plot saved to {output_file}")


def plot_accuracy_improvement(accuracy_history: Dict[str, List[float]],
                             output_file: str = "accuracy_improvement.png"):
    """
    Plot prediction accuracy improvement over iterations.
    
    Args:
        accuracy_history: Dictionary of accuracy lists by regime/metric
        output_file: Path to save the output figure
    """
    if not accuracy_history:
        print("No accuracy history available for plotting")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add overall accuracy if it exists
    if 'overall' in accuracy_history:
        overall = accuracy_history['overall']
        iterations = list(range(1, len(overall) + 1))
        ax.plot(iterations, [a * 100 for a in overall], 'o-', linewidth=3, 
                color='black', label='Overall')
    
    # Add regime-specific accuracies
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    color_idx = 0
    for metric, values in sorted(accuracy_history.items()):
        if metric != 'overall':
            iterations = list(range(1, len(values) + 1))
            ax.plot(iterations, [a * 100 for a in values], 'o--', 
                    color=colors[color_idx % len(colors)], label=metric.capitalize())
            color_idx += 1
    
    # Calculate improvement if we have at least two points
    if 'overall' in accuracy_history and len(accuracy_history['overall']) >= 2:
        initial = accuracy_history['overall'][0]
        final = accuracy_history['overall'][-1]
        improvement = (final - initial) / initial if initial > 0 else 0
        
        # Add annotation about improvement
        ax.annotate(f"Improvement: {improvement:.1%}", 
                   xy=(0.75, 0.05), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set title and labels
    ax.set_title("Prediction Accuracy Improvement", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim([0, 105])  # Set y-axis range from 0 to 105%
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"Accuracy improvement plot saved to {output_file}")


def plot_performance_comparison(expected_performance: List[Dict[str, Dict[str, float]]],
                               actual_performance: List[Dict[str, Dict[str, float]]],
                               metrics: List[str] = ['return', 'sharpe', 'max_drawdown'],
                               output_file: str = "performance_comparison.png"):
    """
    Compare expected vs actual performance metrics over iterations.
    
    Args:
        expected_performance: List of expected performance dictionaries by iteration
        actual_performance: List of actual performance dictionaries by iteration
        metrics: List of metrics to plot
        output_file: Path to save the output figure
    """
    if not expected_performance or not actual_performance:
        print("No performance data available for plotting")
        return
    
    # Number of iterations and regimes
    iterations = list(range(1, len(expected_performance) + 1))
    regimes = sorted(list(expected_performance[0].keys()))
    
    # Create figure with subplots - one row per metric, one column per regime
    fig, axes = plt.subplots(len(metrics), len(regimes), figsize=(5*len(regimes), 4*len(metrics)))
    
    # Add super title
    fig.suptitle("Expected vs Actual Performance Comparison", fontsize=20)
    
    # If only one metric, wrap axes in a list
    if len(metrics) == 1:
        if len(regimes) == 1:
            axes = np.array([[axes]])
        else:
            axes = np.array([axes])
    
    # For each metric and regime, plot comparison
    for i, metric in enumerate(metrics):
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            
            # Extract expected and actual values
            expected_values = [perf[regime][metric] for perf in expected_performance]
            actual_values = [perf[regime][metric] for perf in actual_performance]
            
            # Calculate gap closing
            initial_gap = abs(expected_values[0] - actual_values[0])
            final_gap = abs(expected_values[-1] - actual_values[-1])
            
            gap_reduction = (initial_gap - final_gap) / initial_gap if initial_gap > 0 else 0
            
            # Plot expected vs actual
            ax.plot(iterations, expected_values, 'o-', color='blue', label='Expected')
            ax.plot(iterations, actual_values, 'o-', color='red', label='Actual')
            
            # Fill area between curves
            ax.fill_between(iterations, expected_values, actual_values, 
                           color='gray', alpha=0.2, label='Gap')
            
            # Add annotation
            ax.annotate(f"Gap reduction: {gap_reduction:.1%}", 
                       xy=(0.05, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Set title and labels
            ax.set_title(f"{regime.capitalize()}: {metric}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"{metric}")
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"Performance comparison plot saved to {output_file}")


def generate_feedback_loop_report(param_history: List[Dict[str, Dict[str, float]]],
                                 real_params: Dict[str, Dict[str, float]],
                                 accuracy_history: Dict[str, List[float]],
                                 expected_performance: List[Dict[str, Dict[str, float]]],
                                 actual_performance: List[Dict[str, Dict[str, float]]],
                                 output_dir: str = "."):
    """
    Generate a comprehensive visual report of the feedback loop performance.
    
    Args:
        param_history: List of parameter dictionaries by iteration
        real_params: Dictionary of real parameter values to converge toward
        accuracy_history: Dictionary of accuracy lists by regime/metric
        expected_performance: List of expected performance dictionaries by iteration
        actual_performance: List of actual performance dictionaries by iteration
        output_dir: Directory to save the output figures
    """
    # Create all three visualization types
    param_file = f"{output_dir}/parameter_convergence.png"
    accuracy_file = f"{output_dir}/accuracy_improvement.png"
    perf_file = f"{output_dir}/performance_comparison.png"
    
    plot_parameter_convergence(param_history, real_params, param_file)
    plot_accuracy_improvement(accuracy_history, accuracy_file)
    plot_performance_comparison(expected_performance, actual_performance, output_file=perf_file)
    
    print(f"Feedback loop report generated in {output_dir}")


if __name__ == "__main__":
    # Simple test with dummy data
    # This would be replaced with actual data from the feedback loop
    
    # Parameter history
    param_history = [
        {'bullish': {'trend': 0.0005, 'volatility': 0.01}, 
         'bearish': {'trend': -0.0004, 'volatility': 0.015}},
        {'bullish': {'trend': 0.0006, 'volatility': 0.012}, 
         'bearish': {'trend': -0.0005, 'volatility': 0.018}},
        {'bullish': {'trend': 0.0007, 'volatility': 0.013}, 
         'bearish': {'trend': -0.0006, 'volatility': 0.020}}
    ]
    
    # Real parameters
    real_params = {
        'bullish': {'trend': 0.0008, 'volatility': 0.014},
        'bearish': {'trend': -0.0007, 'volatility': 0.022}
    }
    
    # Accuracy history
    accuracy_history = {
        'overall': [0.50, 0.70, 0.85],
        'bullish': [0.55, 0.75, 0.90],
        'bearish': [0.45, 0.65, 0.80]
    }
    
    # Performance data
    expected_performance = [
        {'bullish': {'return': 10.0, 'sharpe': 1.0, 'max_drawdown': 5.0},
         'bearish': {'return': -5.0, 'sharpe': 0.5, 'max_drawdown': 10.0}},
        {'bullish': {'return': 12.0, 'sharpe': 1.2, 'max_drawdown': 6.0},
         'bearish': {'return': -4.0, 'sharpe': 0.6, 'max_drawdown': 9.0}},
        {'bullish': {'return': 15.0, 'sharpe': 1.5, 'max_drawdown': 7.0},
         'bearish': {'return': -3.0, 'sharpe': 0.7, 'max_drawdown': 8.0}}
    ]
    
    actual_performance = [
        {'bullish': {'return': 5.0, 'sharpe': 0.5, 'max_drawdown': 8.0},
         'bearish': {'return': -10.0, 'sharpe': 0.2, 'max_drawdown': 15.0}},
        {'bullish': {'return': 9.0, 'sharpe': 0.9, 'max_drawdown': 7.0},
         'bearish': {'return': -6.0, 'sharpe': 0.4, 'max_drawdown': 11.0}},
        {'bullish': {'return': 14.0, 'sharpe': 1.4, 'max_drawdown': 7.5},
         'bearish': {'return': -3.5, 'sharpe': 0.6, 'max_drawdown': 8.5}}
    ]
    
    # Generate the report
    generate_feedback_loop_report(
        param_history, real_params, accuracy_history,
        expected_performance, actual_performance
    )
