#!/usr/bin/env python3
"""
Feedback Loop with Visualization

This script integrates our enhanced feedback loop with visualization capabilities,
creating a complete system that:
1. Optimizes strategies using multi-objective methods
2. Learns from real-world performance using adaptive techniques
3. Visualizes parameter convergence and accuracy improvement

It builds directly on our successful implementations:
- multi_objective_simplified.py (core feedback loop)
- feedback_loop_enhancements.py (adaptive learning)
- feedback_loop_visualization.py (visualization utilities)
"""

import math
import time
import os
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

# Import our successful components
from multi_objective_simplified import MultiObjectiveFeedbackLoop, REAL_WORLD_PARAMS
from feedback_loop_enhancements import patch_feedback_loop

# Create output directory for visualizations
VISUALIZATION_DIR = "visualization_output"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


class VisualizationTools:
    """
    Visualization tools for feedback loop analysis.
    
    This class provides methods to visualize parameter convergence,
    prediction accuracy improvement, and performance gap reduction.
    """
    
    @staticmethod
    def plot_parameter_convergence(param_history, real_params, filename=None):
        """
        Plot parameter convergence over iterations.
        
        Args:
            param_history: List of parameter dictionaries by iteration
            real_params: Dictionary of real parameter values
            filename: Optional filename to save the plot
        """
        if not param_history:
            print("No parameter history available for visualization.")
            return None
        
        # Create figure with subplots for each regime
        regimes = param_history[0].keys()
        fig, axes = plt.subplots(len(regimes), 3, figsize=(15, 4 * len(regimes)))
        fig.suptitle('Parameter Convergence to Real Values', fontsize=16)
        
        # For single regime case
        if len(regimes) == 1:
            axes = [axes]
        
        for i, regime in enumerate(regimes):
            if regime not in real_params:
                continue
                
            # Extract parameter values across iterations
            iterations = list(range(1, len(param_history) + 1))
            trend_values = [params[regime]['trend'] for params in param_history]
            vol_values = [params[regime]['volatility'] for params in param_history]
            mr_values = [params[regime]['mean_reversion'] for params in param_history]
            
            # Real values
            real_trend = real_params[regime]['trend']
            real_vol = real_params[regime]['volatility']
            real_mr = real_params[regime]['mean_reversion']
            
            # Plot trend
            axes[i][0].plot(iterations, trend_values, 'b-o', label='Synthetic')
            axes[i][0].axhline(y=real_trend, color='r', linestyle='--', label='Real')
            axes[i][0].set_title(f'{regime.capitalize()} - Trend')
            axes[i][0].set_xlabel('Iteration')
            axes[i][0].set_ylabel('Value')
            axes[i][0].legend()
            
            # Plot volatility
            axes[i][1].plot(iterations, vol_values, 'b-o', label='Synthetic')
            axes[i][1].axhline(y=real_vol, color='r', linestyle='--', label='Real')
            axes[i][1].set_title(f'{regime.capitalize()} - Volatility')
            axes[i][1].set_xlabel('Iteration')
            axes[i][1].set_ylabel('Value')
            axes[i][1].legend()
            
            # Plot mean reversion
            axes[i][2].plot(iterations, mr_values, 'b-o', label='Synthetic')
            axes[i][2].axhline(y=real_mr, color='r', linestyle='--', label='Real')
            axes[i][2].set_title(f'{regime.capitalize()} - Mean Reversion')
            axes[i][2].set_xlabel('Iteration')
            axes[i][2].set_ylabel('Value')
            axes[i][2].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if filename:
            plt.savefig(filename)
            print(f"Parameter convergence plot saved as {filename}")
            
        return fig
    
    @staticmethod
    def plot_accuracy_improvement(accuracy_history, filename=None):
        """
        Plot prediction accuracy improvement over iterations.
        
        Args:
            accuracy_history: Dictionary of accuracy values by regime
            filename: Optional filename to save the plot
        """
        if not accuracy_history or 'overall' not in accuracy_history:
            print("No accuracy history available for visualization.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Prediction Accuracy Improvement', fontsize=16)
        
        # Plot overall accuracy
        iterations = list(range(1, len(accuracy_history['overall']) + 1))
        ax.plot(iterations, accuracy_history['overall'], 'b-o', linewidth=2, label='Overall')
        
        # Plot regime-specific accuracies
        for regime, values in accuracy_history.items():
            if regime != 'overall' and values:
                ax.plot(iterations, values, '-o', linewidth=1.5, label=regime.capitalize())
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('Accuracy Improvement Over Iterations')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Formatting
        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if filename:
            plt.savefig(filename)
            print(f"Accuracy improvement plot saved as {filename}")
            
        return fig
    
    @staticmethod
    def plot_performance_gap(expected_perf, actual_perf, filename=None):
        """
        Plot expected vs actual performance gap reduction.
        
        Args:
            expected_perf: List of expected performance dictionaries
            actual_perf: List of actual performance dictionaries
            filename: Optional filename to save the plot
        """
        if not expected_perf or not actual_perf:
            print("No performance data available for visualization.")
            return None
        
        # Create a figure with subplots for each regime
        regimes = set()
        for perf in expected_perf:
            regimes.update(perf.keys())
        
        fig, axes = plt.subplots(len(regimes), 1, figsize=(12, 5 * len(regimes)))
        fig.suptitle('Expected vs Actual Performance', fontsize=16)
        
        # For single regime case
        if len(regimes) == 1:
            axes = [axes]
        
        for i, regime in enumerate(sorted(regimes)):
            # Extract return values
            iterations = list(range(1, len(expected_perf) + 1))
            expected_returns = []
            actual_returns = []
            
            for iter_idx in range(len(expected_perf)):
                if regime in expected_perf[iter_idx] and regime in actual_perf[iter_idx]:
                    expected_returns.append(expected_perf[iter_idx][regime]['return'])
                    actual_returns.append(actual_perf[iter_idx][regime]['return'])
            
            # Calculate gap for each iteration
            gaps = [abs(e - a) for e, a in zip(expected_returns, actual_returns)]
            
            # Plot
            axes[i].plot(iterations, expected_returns, 'b-o', label='Expected Return')
            axes[i].plot(iterations, actual_returns, 'g-o', label='Actual Return')
            axes[i].bar(iterations, gaps, alpha=0.3, color='r', label='Gap')
            
            axes[i].set_title(f'{regime.capitalize()} - Performance Gap Reduction')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel('Return (%)')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Add text showing gap reduction
            if len(gaps) > 1:
                gap_reduction = (gaps[0] - gaps[-1]) / gaps[0] * 100 if gaps[0] > 0 else 0
                gap_text = f"Gap Reduction: {gap_reduction:.2f}%"
                axes[i].annotate(gap_text, xy=(0.7, 0.05), xycoords='axes fraction',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if filename:
            plt.savefig(filename)
            print(f"Performance gap plot saved as {filename}")
            
        return fig
    
    @staticmethod
    def generate_summary_report(feedback_loop, filename_prefix=None):
        """
        Generate a comprehensive visual report from feedback loop results.
        
        Args:
            feedback_loop: MultiObjectiveFeedbackLoop instance
            filename_prefix: Optional prefix for saved files
        """
        if filename_prefix is None:
            filename_prefix = f"{VISUALIZATION_DIR}/feedback_loop_report_{int(time.time())}"
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
        
        # Generate plots
        param_plot = VisualizationTools.plot_parameter_convergence(
            feedback_loop.param_history,
            REAL_WORLD_PARAMS,
            f"{filename_prefix}_parameters.png"
        )
        
        accuracy_plot = VisualizationTools.plot_accuracy_improvement(
            feedback_loop.accuracy_history,
            f"{filename_prefix}_accuracy.png"
        )
        
        perf_gap_plot = VisualizationTools.plot_performance_gap(
            feedback_loop.expected_perf,
            feedback_loop.actual_perf,
            f"{filename_prefix}_performance_gap.png"
        )
        
        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feedback Loop Visualization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                .section { margin-bottom: 30px; }
                .image-container { text-align: center; margin: 20px 0; }
                img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .convergence-good { color: green; }
                .convergence-medium { color: orange; }
                .convergence-poor { color: red; }
            </style>
        </head>
        <body>
            <h1>Feedback Loop Visualization Report</h1>
            
            <div class="section">
                <h2>Parameter Convergence</h2>
                <div class="image-container">
                    <img src="parameters.png" alt="Parameter Convergence">
                </div>
                <h3>Convergence Analysis</h3>
                <table>
                    <tr>
                        <th>Regime</th>
                        <th>Parameter</th>
                        <th>Initial Value</th>
                        <th>Final Value</th>
                        <th>Real Value</th>
                        <th>Convergence</th>
                    </tr>
        """
        
        # Add parameter convergence data
        for regime in feedback_loop.param_history[0]:
            if regime in REAL_WORLD_PARAMS:
                for param in ['trend', 'volatility', 'mean_reversion']:
                    initial = feedback_loop.param_history[0][regime][param]
                    final = feedback_loop.market_models[regime][param]
                    real = REAL_WORLD_PARAMS[regime][param]
                    
                    # Calculate convergence
                    initial_diff = abs(initial - real)
                    final_diff = abs(final - real)
                    
                    if initial_diff == 0:
                        convergence = 100.0
                    else:
                        convergence = (1 - final_diff / initial_diff) * 100.0
                        convergence = max(0, convergence)
                    
                    # Determine convergence class
                    if convergence > 70:
                        conv_class = "convergence-good"
                    elif convergence > 30:
                        conv_class = "convergence-medium"
                    else:
                        conv_class = "convergence-poor"
                    
                    html_content += f"""
                    <tr>
                        <td>{regime.capitalize()}</td>
                        <td>{param}</td>
                        <td>{initial:.6f}</td>
                        <td>{final:.6f}</td>
                        <td>{real:.6f}</td>
                        <td class="{conv_class}">{convergence:.2f}%</td>
                    </tr>
                    """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Prediction Accuracy Improvement</h2>
                <div class="image-container">
                    <img src="accuracy.png" alt="Accuracy Improvement">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Gap Reduction</h2>
                <div class="image-container">
                    <img src="performance_gap.png" alt="Performance Gap Reduction">
                </div>
                <h3>Gap Analysis</h3>
                <table>
                    <tr>
                        <th>Regime</th>
                        <th>Initial Gap</th>
                        <th>Final Gap</th>
                        <th>Reduction</th>
                    </tr>
        """
        
        # Add performance gap data
        for regime in sorted(feedback_loop.expected_perf[0].keys()):
            if (regime in feedback_loop.expected_perf[0] and 
                regime in feedback_loop.actual_perf[0] and
                regime in feedback_loop.expected_perf[-1] and
                regime in feedback_loop.actual_perf[-1]):
                
                initial_gap = abs(feedback_loop.expected_perf[0][regime]['return'] - 
                                feedback_loop.actual_perf[0][regime]['return'])
                final_gap = abs(feedback_loop.expected_perf[-1][regime]['return'] - 
                               feedback_loop.actual_perf[-1][regime]['return'])
                
                if initial_gap > 0:
                    reduction = (initial_gap - final_gap) / initial_gap * 100
                else:
                    reduction = 0
                
                # Determine class
                if reduction > 30:
                    red_class = "convergence-good"
                elif reduction > 0:
                    red_class = "convergence-medium"
                else:
                    red_class = "convergence-poor"
                
                html_content += f"""
                <tr>
                    <td>{regime.capitalize()}</td>
                    <td>{initial_gap:.2f}%</td>
                    <td>{final_gap:.2f}%</td>
                    <td class="{red_class}">{reduction:.2f}%</td>
                </tr>
                """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML report
        html_filename = f"{filename_prefix}.html"
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive report generated at {html_filename}")
        
        # Copy the images to make relative paths work
        import shutil
        shutil.copy(f"{filename_prefix}_parameters.png", 
                   os.path.join(os.path.dirname(filename_prefix), "parameters.png"))
        shutil.copy(f"{filename_prefix}_accuracy.png", 
                   os.path.join(os.path.dirname(filename_prefix), "accuracy.png"))
        shutil.copy(f"{filename_prefix}_performance_gap.png", 
                   os.path.join(os.path.dirname(filename_prefix), "performance_gap.png"))
        
        return html_filename


class VisualizedFeedbackLoop(MultiObjectiveFeedbackLoop):
    """
    Extension of MultiObjectiveFeedbackLoop with visualization capabilities.
    
    This class adds visualization to our successful feedback loop implementation,
    allowing us to track and analyze the learning process.
    """
    
    def __init__(self):
        """Initialize the visualized feedback loop."""
        super().__init__()
        
        # Apply enhanced learning
        patch_feedback_loop(self)
        
        # Visualization settings
        self.visualization_dir = VISUALIZATION_DIR
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Store reference to real-world params for visualization
        self.real_world_params = REAL_WORLD_PARAMS
        
        print("Initialized VisualizedFeedbackLoop with enhanced learning")
    
    def run_feedback_loop(self, iterations=5, generate_visuals=True):
        """
        Run the complete feedback loop with visualization.
        
        Args:
            iterations: Number of feedback loop iterations
            generate_visuals: Whether to generate visualization during run
        """
        print(f"\nRunning visualized feedback loop for {iterations} iterations...")
        
        # Create a live plot for parameter convergence if enabled
        if generate_visuals:
            plt.ion()  # Turn on interactive mode
            param_fig, param_axes = plt.subplots(4, 3, figsize=(15, 16))
            param_fig.suptitle('Parameter Convergence (Live)', fontsize=16)
            
            # Set up the axes
            regimes = ['bullish', 'bearish', 'sideways', 'volatile']
            params = ['trend', 'volatility', 'mean_reversion']
            
            for i, regime in enumerate(regimes):
                for j, param in enumerate(params):
                    param_axes[i, j].set_title(f'{regime.capitalize()} - {param}')
                    param_axes[i, j].set_xlabel('Iteration')
                    param_axes[i, j].set_ylabel('Value')
                    
                    # Add real value line if available
                    if regime in self.real_world_params and param in self.real_world_params[regime]:
                        param_axes[i, j].axhline(
                            y=self.real_world_params[regime][param],
                            color='r', linestyle='--', label='Real'
                        )
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            param_fig.canvas.draw()
            plt.pause(0.1)
        
        # Run the feedback loop
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
            
            # Update live visualization if enabled
            if generate_visuals:
                self.update_live_visualization(param_fig, param_axes, i+1)
            
            # Small delay for readability
            time.sleep(0.5)
        
        # Generate final visualization report
        if generate_visuals:
            plt.ioff()  # Turn off interactive mode
            timestamp = int(time.time())
            report_path = VisualizationTools.generate_summary_report(
                self, f"{self.visualization_dir}/feedback_report_{timestamp}"
            )
            print(f"\nFinal visualization report generated: {report_path}")
        
        # Show text summary
        self.print_results_summary()
    
    def update_live_visualization(self, fig, axes, iteration):
        """
        Update the live visualization with current parameter values.
        
        Args:
            fig: Figure object
            axes: Axes array
            iteration: Current iteration number
        """
        regimes = ['bullish', 'bearish', 'sideways', 'volatile']
        params = ['trend', 'volatility', 'mean_reversion']
        
        # Collect all parameter values up to current iteration
        iterations = list(range(1, iteration + 1))
        
        for i, regime in enumerate(regimes):
            # Skip if regime not in our models
            if regime not in self.market_models:
                continue
                
            for j, param in enumerate(params):
                # Skip if parameter not in this regime's model
                if param not in self.market_models[regime]:
                    continue
                    
                # Extract parameter values across iterations
                param_values = [ph[regime][param] for ph in self.param_history]
                param_values.append(self.market_models[regime][param])
                
                # Clear axis and replot
                axes[i, j].clear()
                axes[i, j].plot(iterations, param_values, 'b-o', label='Synthetic')
                
                # Add real value line
                if regime in self.real_world_params and param in self.real_world_params[regime]:
                    axes[i, j].axhline(
                        y=self.real_world_params[regime][param],
                        color='r', linestyle='--', label='Real'
                    )
                
                # Labels and legend
                axes[i, j].set_title(f'{regime.capitalize()} - {param}')
                axes[i, j].set_xlabel('Iteration')
                axes[i, j].set_ylabel('Value')
                axes[i, j].legend()
                
                # Add convergence percentage if real values available
                if regime in self.real_world_params and param in self.real_world_params[regime]:
                    initial = self.param_history[0][regime][param]
                    current = self.market_models[regime][param]
                    real = self.real_world_params[regime][param]
                    
                    initial_diff = abs(initial - real)
                    current_diff = abs(current - real)
                    
                    if initial_diff > 0:
                        convergence = (1 - current_diff / initial_diff) * 100
                        convergence = max(0, convergence)
                        axes[i, j].set_title(
                            f'{regime.capitalize()} - {param} (Conv: {convergence:.1f}%)'
                        )
        
        # Update the figure
        fig.canvas.draw()
        plt.pause(0.1)
    
    def print_results_summary(self):
        """Print a text summary of the feedback loop results."""
        print("\n" + "=" * 70)
        print("VISUALIZED FEEDBACK LOOP RESULTS SUMMARY")
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
            if regime in self.real_world_params:
                print(f"\n{regime.capitalize()} regime:")
                print("Parameter     | Initial    | Final      | Real       | Convergence")
                print("-" * 64)
                
                initial_params = self.param_history[0][regime]
                final_params = self.market_models[regime]
                real_params = self.real_world_params[regime]
                
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
        
        # Overall conclusion
        print("\n" + "=" * 70)
        
        if len(self.accuracy_history['overall']) > 1 and self.accuracy_history['overall'][-1] > self.accuracy_history['overall'][0]:
            print("CONCLUSION: The feedback loop successfully improved optimization accuracy!")
            print(f"Prediction accuracy increased by {improvement:.2%} over {len(self.accuracy_history['overall'])} iterations.")
        else:
            print("CONCLUSION: The feedback loop did not improve optimization accuracy.")
        
        print("=" * 70)


def run_visualized_demo(iterations=5):
    """
    Run a demonstration of the visualized feedback loop.
    
    Args:
        iterations: Number of feedback loop iterations to run
    """
    print("=" * 70)
    print("VISUALIZED FEEDBACK LOOP DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstrates our trading strategy optimization system with:")
    print("1. Multi-objective optimization across market regimes")
    print("2. Enhanced adaptive learning mechanism")
    print("3. Live parameter convergence visualization")
    print("4. Comprehensive performance analysis")
    print("\n" + "=" * 70)
    
    # Run the visualized feedback loop
    feedback_loop = VisualizedFeedbackLoop()
    feedback_loop.run_feedback_loop(iterations=iterations)
    
    # Return the path to the output report
    return os.path.join(os.getcwd(), VISUALIZATION_DIR)


if __name__ == "__main__":
    output_dir = run_visualized_demo(iterations=7)
    print(f"\nVisualization outputs are available in: {output_dir}")
