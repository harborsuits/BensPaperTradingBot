#!/usr/bin/env python3
"""
Feedback Loop with Text Visualization

This script integrates our enhanced feedback loop with text-based visualization,
creating a complete system that:
1. Optimizes strategies using multi-objective methods
2. Learns from real-world performance using adaptive techniques
3. Visualizes parameter convergence and accuracy improvement using text-based charts

It builds directly on our successful implementations:
- multi_objective_simplified.py (core feedback loop)
- feedback_loop_enhancements.py (adaptive learning)
"""

import math
import time
import os
import json
from typing import Dict, List, Any, Tuple

# Import our successful components
from multi_objective_simplified import MultiObjectiveFeedbackLoop, REAL_WORLD_PARAMS
from feedback_loop_enhancements import patch_feedback_loop

# Create output directory for results
RESULTS_DIR = "feedback_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class TextVisualization:
    """
    Text-based visualization tools for feedback loop analysis.
    
    This class provides methods to visualize parameter convergence,
    prediction accuracy improvement, and performance gap reduction
    using text-based charts and tables.
    """
    
    @staticmethod
    def create_text_chart(values, width=50, title=None, labels=None, reference_value=None):
        """
        Create a simple text-based chart.
        
        Args:
            values: List of values to plot
            width: Width of the chart
            title: Optional chart title
            labels: Optional x-axis labels
            reference_value: Optional reference value to show as a line
            
        Returns:
            Text chart as a string
        """
        if not values:
            return "No data to visualize."
        
        # Find min and max for scaling
        min_val = min(values)
        max_val = max(values)
        
        # Include reference value in min/max if provided
        if reference_value is not None:
            min_val = min(min_val, reference_value)
            max_val = max(max_val, reference_value)
        
        # Ensure we have a range to prevent division by zero
        value_range = max_val - min_val
        if value_range == 0:
            value_range = 1
        
        # Create the chart
        chart = []
        
        # Add title if provided
        if title:
            chart.append(title)
            chart.append("=" * len(title))
        
        # Create chart body (10 lines high)
        lines = []
        for i in range(10):
            level = max_val - (i / 9) * value_range
            
            # Mark reference value with a line
            if reference_value is not None and level >= reference_value > (level - value_range/9):
                level_str = f"{level:.6f} ★" 
            else:
                level_str = f"{level:.6f}"
                
            line = f"{level_str:10} |"
            lines.append(line)
        
        # Add data points
        for i, value in enumerate(values):
            # Calculate position
            line_idx = min(9, max(0, int(9 * (max_val - value) / value_range)))
            pos = i * (width - 12) // (len(values) - 1) if len(values) > 1 else width // 2
            
            # Add point marker
            lines[line_idx] = lines[line_idx][:12 + pos] + "o" + lines[line_idx][13 + pos:]
        
        # Finalize lines by filling gaps
        for i, line in enumerate(lines):
            if len(line) < width:
                lines[i] = line + " " * (width - len(line))
        
        chart.extend(lines)
        
        # Add x-axis
        chart.append("-" * width)
        
        # Add labels if provided
        if labels:
            label_line = ""
            spacing = (width - 12) // (len(values) - 1) if len(values) > 1 else width // 2
            for i, label in enumerate(labels):
                pos = i * spacing
                label_line = label_line + " " * (pos - len(label_line)) + label
            chart.append(label_line)
        
        return "\n".join(chart)
    
    @staticmethod
    def create_text_trend_chart(values, references=None, title=None, width=60):
        """
        Create a text-based trend chart comparing values to references.
        
        Args:
            values: List of values to plot
            references: List of reference values
            title: Optional chart title
            width: Chart width
            
        Returns:
            Text chart as a string
        """
        if not values:
            return "No data to visualize."
        
        iterations = list(range(1, len(values) + 1))
        
        chart = []
        if title:
            chart.append(title)
            chart.append("=" * len(title))
        
        values_chart = TextVisualization.create_text_chart(
            values, 
            width=width,
            labels=[str(i) for i in iterations],
            reference_value=references[0] if references else None
        )
        
        chart.append(values_chart)
        
        # Add reference line label if applicable
        if references and len(references) == 1:
            chart.append(f"★ Reference value: {references[0]:.6f}")
        
        # Calculate convergence if reference is provided
        if references and len(references) == 1:
            initial_diff = abs(values[0] - references[0])
            final_diff = abs(values[-1] - references[0])
            
            if initial_diff > 0:
                convergence = (1 - final_diff / initial_diff) * 100
                convergence = max(0, convergence)
                chart.append(f"Convergence: {convergence:.2f}%")
            else:
                chart.append("Perfect initial match!")
        
        return "\n".join(chart)
    
    @staticmethod
    def create_accuracy_table(accuracy_history):
        """
        Create a text table showing accuracy changes over iterations.
        
        Args:
            accuracy_history: Dictionary of accuracy values by regime
            
        Returns:
            Formatted text table
        """
        if not accuracy_history or 'overall' not in accuracy_history:
            return "No accuracy history available."
        
        # Table header
        table = ["Iteration | " + " | ".join(r.capitalize() for r in accuracy_history.keys())]
        table.append("-" * (10 + sum(12 for _ in accuracy_history)))
        
        # Table rows
        iterations = len(accuracy_history['overall'])
        for i in range(iterations):
            row = [f"{i+1:9}"]
            for regime in accuracy_history:
                if i < len(accuracy_history[regime]):
                    value = accuracy_history[regime][i] * 100  # Convert to percentage
                    row.append(f"{value:10.2f}%")
                else:
                    row.append(" " * 11)
            table.append(" | ".join(row))
        
        # Add improvement row if possible
        if iterations > 1:
            table.append("-" * (10 + sum(12 for _ in accuracy_history)))
            row = ["Improvement"]
            
            for regime in accuracy_history:
                if iterations <= len(accuracy_history[regime]):
                    initial = accuracy_history[regime][0]
                    final = accuracy_history[regime][-1]
                    
                    if initial > 0:
                        improvement = (final - initial) / initial * 100
                        row.append(f"{improvement:10.2f}%")
                    else:
                        row.append("    N/A    ")
                else:
                    row.append(" " * 11)
            
            table.append(" | ".join(row))
        
        return "\n".join(table)
    
    @staticmethod
    def create_parameter_table(param_history, real_params):
        """
        Create a text table showing parameter convergence.
        
        Args:
            param_history: List of parameter dictionaries by iteration
            real_params: Dictionary of real parameter values
            
        Returns:
            Formatted text table
        """
        if not param_history:
            return "No parameter history available."
        
        table = []
        
        # For each regime
        for regime in param_history[0]:
            if regime in real_params:
                table.append(f"\n{regime.capitalize()} Regime")
                table.append("=" * (len(regime) + 7))
                
                # Table header
                table.append("Parameter     | Initial    | Final      | Real       | Convergence")
                table.append("-" * 64)
                
                # Add parameter rows
                for param in ['trend', 'volatility', 'mean_reversion']:
                    initial = param_history[0][regime][param]
                    final = param_history[-1][regime][param]
                    real = real_params[regime][param]
                    
                    # Calculate convergence
                    initial_diff = abs(initial - real)
                    final_diff = abs(final - real)
                    
                    if initial_diff == 0:
                        convergence = 100.0
                    else:
                        convergence = (1 - final_diff / initial_diff) * 100.0
                        convergence = max(0, convergence)
                    
                    table.append(f"{param:<13} | {initial:<10.6f} | {final:<10.6f} | {real:<10.6f} | {convergence:>6.2f}%")
        
        return "\n".join(table)
    
    @staticmethod
    def create_performance_gap_table(expected_perf, actual_perf):
        """
        Create a text table showing performance gap reduction.
        
        Args:
            expected_perf: List of expected performance dictionaries
            actual_perf: List of actual performance dictionaries
            
        Returns:
            Formatted text table
        """
        if not expected_perf or not actual_perf:
            return "No performance data available."
        
        table = ["\nPerformance Gap Analysis"]
        table.append("=" * 22)
        
        # Table header
        table.append("Regime       | Initial Gap | Final Gap  | Reduction")
        table.append("-" * 50)
        
        # Add rows for each regime
        for regime in sorted(expected_perf[0].keys()):
            if (regime in expected_perf[0] and 
                regime in actual_perf[0] and
                regime in expected_perf[-1] and
                regime in actual_perf[-1]):
                
                initial_gap = abs(expected_perf[0][regime]['return'] - 
                               actual_perf[0][regime]['return'])
                final_gap = abs(expected_perf[-1][regime]['return'] - 
                              actual_perf[-1][regime]['return'])
                
                if initial_gap > 0:
                    reduction = (initial_gap - final_gap) / initial_gap * 100
                else:
                    reduction = 0
                
                table.append(f"{regime:<12} | {initial_gap:<11.2f} | {final_gap:<10.2f} | {reduction:>6.2f}%")
        
        return "\n".join(table)
    
    @staticmethod
    def generate_text_report(feedback_loop, output_file=None):
        """
        Generate a comprehensive text report from feedback loop results.
        
        Args:
            feedback_loop: MultiObjectiveFeedbackLoop instance
            output_file: Optional file to save the report
            
        Returns:
            Report text
        """
        report = []
        
        # Title
        report.append("=" * 70)
        report.append("FEEDBACK LOOP VISUALIZATION REPORT")
        report.append("=" * 70)
        
        # Parameter convergence charts
        report.append("\n\nPARAMETER CONVERGENCE CHARTS")
        report.append("-" * 26)
        
        for regime in feedback_loop.param_history[0]:
            if regime in REAL_WORLD_PARAMS:
                report.append(f"\n{regime.capitalize()} Regime")
                
                # Create chart for each parameter
                for param in ['trend', 'volatility', 'mean_reversion']:
                    values = [params[regime][param] for params in feedback_loop.param_history]
                    values.append(feedback_loop.market_models[regime][param])
                    
                    real_value = REAL_WORLD_PARAMS[regime][param]
                    
                    chart = TextVisualization.create_text_trend_chart(
                        values,
                        references=[real_value],
                        title=f"{param.capitalize()}"
                    )
                    report.append(chart)
                    report.append("")
        
        # Accuracy table
        report.append("\n\nPREDICTION ACCURACY IMPROVEMENT")
        report.append("-" * 30)
        
        accuracy_table = TextVisualization.create_accuracy_table(feedback_loop.accuracy_history)
        report.append(accuracy_table)
        
        # Parameter convergence table
        report.append("\n\nPARAMETER CONVERGENCE SUMMARY")
        report.append("-" * 26)
        
        param_table = TextVisualization.create_parameter_table(
            feedback_loop.param_history,
            REAL_WORLD_PARAMS
        )
        report.append(param_table)
        
        # Performance gap table
        report.append("\n\nPERFORMANCE GAP REDUCTION")
        report.append("-" * 24)
        
        gap_table = TextVisualization.create_performance_gap_table(
            feedback_loop.expected_perf,
            feedback_loop.actual_perf
        )
        report.append(gap_table)
        
        # Overall conclusion
        report.append("\n\n" + "=" * 70)
        
        if len(feedback_loop.accuracy_history['overall']) > 1:
            initial = feedback_loop.accuracy_history['overall'][0]
            final = feedback_loop.accuracy_history['overall'][-1]
            improvement = (final - initial) / initial if initial > 0 else 0
            
            if final > initial:
                report.append("CONCLUSION: The feedback loop successfully improved optimization accuracy!")
                report.append(f"Prediction accuracy increased by {improvement:.2%} over {len(feedback_loop.accuracy_history['overall'])} iterations.")
            else:
                report.append("CONCLUSION: The feedback loop did not improve optimization accuracy.")
        
        report.append("=" * 70)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("\n".join(report))
            print(f"Report saved to {output_file}")
        
        return "\n".join(report)


class TextVisualizedFeedbackLoop(MultiObjectiveFeedbackLoop):
    """
    Extension of MultiObjectiveFeedbackLoop with text visualization capabilities.
    
    This class adds text-based visualization to our successful feedback loop implementation,
    allowing us to track and analyze the learning process without external dependencies.
    """
    
    def __init__(self):
        """Initialize the text visualized feedback loop."""
        super().__init__()
        
        # Apply enhanced learning
        patch_feedback_loop(self)
        
        # Output settings
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Store reference to real-world params for visualization
        self.real_world_params = REAL_WORLD_PARAMS
        
        print("Initialized TextVisualizedFeedbackLoop with enhanced learning")
    
    def run_feedback_loop(self, iterations=5, generate_report=True):
        """
        Run the complete feedback loop with text visualization.
        
        Args:
            iterations: Number of feedback loop iterations
            generate_report: Whether to generate a report at the end
        """
        print(f"\nRunning text-visualized feedback loop for {iterations} iterations...")
        
        # For JSON export
        results_data = {
            "iterations": iterations,
            "accuracy_history": {},
            "param_history": [],
            "expected_perf": [],
            "actual_perf": []
        }
        
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
            
            # Show live parameter convergence
            if i % 2 == 1 or i == iterations - 1:  # Show every other iteration to save space
                self.show_live_convergence(i+1)
            
            # Save data for export
            results_data["param_history"].append({})
            for regime, params in self.market_models.items():
                results_data["param_history"][-1][regime] = params.copy()
            
            results_data["expected_perf"].append({})
            for regime, metrics in self.expected_perf[-1].items():
                results_data["expected_perf"][-1][regime] = {
                    "return": metrics["return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe": metrics["sharpe"]
                }
            
            results_data["actual_perf"].append({})
            for regime, metrics in self.actual_perf[-1].items():
                results_data["actual_perf"][-1][regime] = {
                    "return": metrics["return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe": metrics["sharpe"]
                }
            
            # Small delay for readability
            time.sleep(0.5)
        
        # Save accuracy history
        for regime, values in self.accuracy_history.items():
            results_data["accuracy_history"][regime] = values
        
        # Export results to JSON
        timestamp = int(time.time())
        json_file = os.path.join(self.results_dir, f"feedback_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults exported to {json_file}")
        
        # Generate final report
        if generate_report:
            report_file = os.path.join(self.results_dir, f"feedback_report_{timestamp}.txt")
            report = TextVisualization.generate_text_report(self, report_file)
            print(f"\nFinal report generated: {report_file}")
        
        # Show text summary
        self.print_results_summary()
        
        return os.path.join(os.getcwd(), self.results_dir)
    
    def show_live_convergence(self, iteration):
        """
        Show live parameter convergence in text format.
        
        Args:
            iteration: Current iteration number
        """
        print("\nCurrent Parameter Convergence:")
        print("-" * 30)
        
        # Get parameters from history and current models
        params_by_iteration = {}
        
        for regime in self.market_models:
            if regime not in self.real_world_params:
                continue
                
            params_by_iteration[regime] = {}
            
            for param in ['trend', 'volatility', 'mean_reversion']:
                values = [ph[regime][param] for ph in self.param_history]
                values.append(self.market_models[regime][param])
                
                params_by_iteration[regime][param] = values
        
        # Show convergence for key parameters
        shown_params = 0
        
        for regime in ['bullish', 'sideways']:  # Show only selected regimes to save space
            if regime not in params_by_iteration:
                continue
                
            for param in ['trend', 'volatility']:  # Show only key parameters
                if param not in params_by_iteration[regime]:
                    continue
                    
                values = params_by_iteration[regime][param]
                real_value = self.real_world_params[regime][param]
                
                # Calculate convergence
                initial_diff = abs(values[0] - real_value)
                current_diff = abs(values[-1] - real_value)
                
                if initial_diff > 0:
                    convergence = (1 - current_diff / initial_diff) * 100
                    convergence = max(0, convergence)
                else:
                    convergence = 100.0
                
                # Show simple chart
                print(f"\n{regime.capitalize()} {param.capitalize()}")
                print(f"Current: {values[-1]:.6f}, Real: {real_value:.6f}, Convergence: {convergence:.2f}%")
                
                # Simple ASCII chart
                chart_width = 40
                symbols = ['░', '▒', '▓', '█']
                
                chart = "["
                conv_int = min(100, max(0, int(convergence)))
                filled = int(chart_width * conv_int / 100)
                
                if filled > 0:
                    chart += symbols[3] * filled
                
                chart += " " * (chart_width - filled)
                chart += f"] {conv_int}%"
                
                print(chart)
                shown_params += 1
        
        if shown_params == 0:
            print("No parameters with real values available to show convergence.")
    
    def print_results_summary(self):
        """Print a text summary of the feedback loop results."""
        print("\n" + "=" * 70)
        print("TEXT VISUALIZED FEEDBACK LOOP RESULTS SUMMARY")
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


def run_text_visualized_demo(iterations=5):
    """
    Run a demonstration of the text visualized feedback loop.
    
    Args:
        iterations: Number of feedback loop iterations to run
    """
    print("=" * 70)
    print("TEXT VISUALIZED FEEDBACK LOOP DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstrates our trading strategy optimization system with:")
    print("1. Multi-objective optimization across market regimes")
    print("2. Enhanced adaptive learning mechanism")
    print("3. Text-based visualization of parameter convergence")
    print("4. Comprehensive performance analysis without external dependencies")
    print("\n" + "=" * 70)
    
    # Run the text visualized feedback loop
    feedback_loop = TextVisualizedFeedbackLoop()
    output_dir = feedback_loop.run_feedback_loop(iterations=iterations)
    
    # Return the path to the output directory
    return output_dir


if __name__ == "__main__":
    output_dir = run_text_visualized_demo(iterations=7)
    print(f"\nResults and reports are available in: {output_dir}")
