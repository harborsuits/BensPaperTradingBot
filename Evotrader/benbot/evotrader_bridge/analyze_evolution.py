"""
Analysis tools for the evolutionary trading system.

This script provides functions to analyze the results of the evolutionary process,
visualize performance improvements, and suggest parameter optimizations.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from main import EvoTraderBridge


class EvolutionAnalyzer:
    """Analyzer for evolution results and strategy performance."""
    
    def __init__(self, test_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            test_dir: Directory containing test results
        """
        self.test_dir = test_dir
        
        # Load configuration if available
        self.config = self._load_config()
        
        # Initialize bridge to access data
        self.bridge = EvoTraderBridge(config=self.config)
        
        # Set up output directory
        self.output_dir = os.path.join(test_dir, "analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Initialized analyzer for test directory: {test_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration if available."""
        config_path = os.path.join(self.test_dir, "test_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {str(e)}")
        
        return {"output_dir": self.test_dir}
    
    def load_evolution_history(self) -> Dict[str, Any]:
        """
        Load evolution history from the results.
        
        Returns:
            Evolution history dictionary
        """
        # Get evolution history from the bridge
        history = self.bridge.evolution_manager.get_evolutionary_history()
        
        # If history is empty, try to load from files
        if not history or "generations" not in history or history["generations"] == 0:
            evolution_dir = os.path.join(self.test_dir, "evolution")
            if os.path.exists(evolution_dir):
                # Look for generation stats
                history = {"generations": 0, "generation_stats": {}}
                
                gen_stats_dir = os.path.join(evolution_dir, "generation_stats")
                if os.path.exists(gen_stats_dir):
                    for file in os.listdir(gen_stats_dir):
                        if file.startswith("generation_") and file.endswith(".json"):
                            try:
                                with open(os.path.join(gen_stats_dir, file), 'r') as f:
                                    gen_data = json.load(f)
                                    gen_num = int(file.split("_")[1].split(".")[0])
                                    history["generation_stats"][gen_num] = gen_data
                                    history["generations"] = max(history["generations"], gen_num + 1)
                            except Exception as e:
                                print(f"Error loading generation file {file}: {str(e)}")
        
        return history
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends across generations.
        
        Returns:
            Analysis results dictionary
        """
        # Load evolution history
        history = self.load_evolution_history()
        
        # Initialize results
        results = {
            "generations": history.get("generations", 0),
            "metrics": {},
            "strategy_distribution": {},
            "improvement_rates": {}
        }
        
        # Extract metrics from each generation
        gen_stats = history.get("generation_stats", {})
        
        # Track metrics across generations
        metrics_to_track = ["avg_fitness", "max_fitness", "avg_performance", "win_rate", "sharpe_ratio"]
        for metric in metrics_to_track:
            results["metrics"][metric] = []
        
        # Process each generation
        for gen in range(results["generations"]):
            if gen in gen_stats:
                stats = gen_stats[gen]
                
                # Extract metrics
                for metric in metrics_to_track:
                    value = stats.get(metric, 0)
                    results["metrics"][metric].append(value)
                
                # Extract strategy distribution
                if "strategy_distribution" in stats:
                    results["strategy_distribution"][gen] = stats["strategy_distribution"]
        
        # Calculate improvement rates
        for metric in metrics_to_track:
            values = results["metrics"].get(metric, [])
            if len(values) >= 2:
                first_value = values[0]
                last_value = values[-1]
                
                if first_value != 0:
                    improvement = (last_value - first_value) / abs(first_value) * 100
                    results["improvement_rates"][metric] = improvement
                else:
                    results["improvement_rates"][metric] = 0
        
        return results
    
    def analyze_strategy_parameters(self) -> Dict[str, Any]:
        """
        Analyze how strategy parameters evolve and converge.
        
        Returns:
            Parameter analysis dictionary
        """
        # Get all strategies from the bridge
        strategies = self.bridge.evolution_manager.strategies
        
        # Group strategies by type and generation
        strategy_groups = {}
        
        for strategy_id, strategy in strategies.items():
            # Get strategy type
            strategy_type = "Unknown"
            if hasattr(strategy, "benbot_strategy") and strategy.benbot_strategy:
                strategy_type = strategy.benbot_strategy.__class__.__name__
            
            # Get generation
            generation = strategy.metadata.get("generation", 0)
            
            # Create group key
            group_key = f"{strategy_type}_{generation}"
            
            if group_key not in strategy_groups:
                strategy_groups[group_key] = []
                
            strategy_groups[group_key].append(strategy)
        
        # Analyze parameters for each group
        results = {
            "parameter_trends": {},
            "parameter_convergence": {},
            "optimal_ranges": {}
        }
        
        # Process each strategy type separately
        for group_key, group_strategies in strategy_groups.items():
            strategy_type, generation = group_key.split("_")
            generation = int(generation)
            
            # Skip if no strategies
            if not group_strategies:
                continue
                
            # Collect parameters
            all_params = {}
            
            for strategy in group_strategies:
                for param_name, param_value in strategy.parameters.items():
                    if isinstance(param_value, (int, float)):
                        if param_name not in all_params:
                            all_params[param_name] = []
                        all_params[param_name].append(param_value)
            
            # Calculate statistics for each parameter
            for param_name, values in all_params.items():
                # Create key for this parameter
                param_key = f"{strategy_type}_{param_name}"
                
                # Initialize if not exists
                if param_key not in results["parameter_trends"]:
                    results["parameter_trends"][param_key] = {
                        "generations": [],
                        "mean": [],
                        "std": [],
                        "min": [],
                        "max": []
                    }
                
                # Record this generation's stats
                stats = results["parameter_trends"][param_key]
                stats["generations"].append(generation)
                stats["mean"].append(np.mean(values))
                stats["std"].append(np.std(values))
                stats["min"].append(min(values))
                stats["max"].append(max(values))
        
        # Calculate parameter convergence
        for param_key, stats in results["parameter_trends"].items():
            if len(stats["std"]) >= 2:
                initial_std = stats["std"][0]
                final_std = stats["std"][-1]
                
                if initial_std > 0:
                    convergence = (initial_std - final_std) / initial_std
                    results["parameter_convergence"][param_key] = convergence
                else:
                    results["parameter_convergence"][param_key] = 0
                    
            # Estimate optimal range based on the best performing strategies
            strategy_type = param_key.split("_")[0]
            param_name = "_".join(param_key.split("_")[1:])
            
            best_strategies = self.bridge.get_best_strategies(count=5)
            best_values = []
            
            for strategy in best_strategies:
                if hasattr(strategy, "benbot_strategy") and strategy.benbot_strategy:
                    if strategy.benbot_strategy.__class__.__name__ == strategy_type:
                        if param_name in strategy.parameters:
                            best_values.append(strategy.parameters[param_name])
            
            if best_values:
                results["optimal_ranges"][param_key] = {
                    "mean": np.mean(best_values),
                    "std": np.std(best_values),
                    "min": min(best_values),
                    "max": max(best_values)
                }
        
        return results
    
    def generate_performance_charts(self):
        """Generate charts visualizing performance trends."""
        # Analyze performance trends
        performance = self.analyze_performance_trends()
        
        # Create output directory
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('ggplot')
        
        # 1. Fitness progression chart
        self._plot_fitness_progression(performance, charts_dir)
        
        # 2. Strategy distribution chart
        self._plot_strategy_distribution(performance, charts_dir)
        
        # 3. Performance metrics chart
        self._plot_performance_metrics(performance, charts_dir)
        
        print(f"Generated performance charts in {charts_dir}")
    
    def _plot_fitness_progression(self, performance: Dict[str, Any], charts_dir: str):
        """Plot fitness progression across generations."""
        plt.figure(figsize=(10, 6))
        
        generations = list(range(performance["generations"]))
        avg_fitness = performance["metrics"].get("avg_fitness", [])
        max_fitness = performance["metrics"].get("max_fitness", [])
        
        if not generations or not avg_fitness:
            print("Not enough data to plot fitness progression")
            return
            
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.plot(generations, max_fitness, 'g-', label='Max Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('Evolution of Strategy Fitness Across Generations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add improvement annotations
        if len(avg_fitness) >= 2:
            improvement = performance["improvement_rates"].get("avg_fitness", 0)
            plt.annotate(f"Improvement: {improvement:.1f}%", 
                     xy=(generations[-1], avg_fitness[-1]),
                     xytext=(0, 10), 
                     textcoords="offset points",
                     ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "fitness_progression.png"), dpi=300)
        plt.close()
    
    def _plot_strategy_distribution(self, performance: Dict[str, Any], charts_dir: str):
        """Plot strategy type distribution across generations."""
        strategy_distribution = performance.get("strategy_distribution", {})
        
        if not strategy_distribution:
            print("No strategy distribution data available")
            return
            
        # Prepare data for stacked bar chart
        generations = sorted(strategy_distribution.keys())
        strategy_types = set()
        
        for gen_data in strategy_distribution.values():
            strategy_types.update(gen_data.keys())
            
        strategy_types = sorted(strategy_types)
        
        # Create matrix of values
        data = []
        for strategy_type in strategy_types:
            values = []
            for gen in generations:
                values.append(strategy_distribution[gen].get(strategy_type, 0))
            data.append(values)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        bottom = np.zeros(len(generations))
        for i, d in enumerate(data):
            plt.bar(generations, d, bottom=bottom, label=strategy_types[i])
            bottom += d
            
        plt.xlabel('Generation')
        plt.ylabel('Strategy Count')
        plt.title('Strategy Type Distribution Across Generations')
        plt.legend(title='Strategy Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "strategy_distribution.png"), dpi=300)
        plt.close()
    
    def _plot_performance_metrics(self, performance: Dict[str, Any], charts_dir: str):
        """Plot various performance metrics across generations."""
        metrics = performance.get("metrics", {})
        generations = list(range(performance["generations"]))
        
        if not generations:
            print("No generation data available")
            return
            
        # Select metrics to plot
        metrics_to_plot = ["avg_performance", "win_rate", "sharpe_ratio"]
        available_metrics = [m for m in metrics_to_plot if m in metrics and metrics[m]]
        
        if not available_metrics:
            print("No performance metrics available to plot")
            return
            
        # Create subplot for each metric
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4*len(available_metrics)))
        
        # Handle case with only one metric
        if len(available_metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            values = metrics[metric]
            
            ax.plot(generations, values, 'b-')
            ax.set_xlabel('Generation')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Across Generations')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add improvement annotation
            if len(values) >= 2:
                improvement = performance["improvement_rates"].get(metric, 0)
                ax.annotate(f"Improvement: {improvement:.1f}%", 
                         xy=(generations[-1], values[-1]),
                         xytext=(0, 10), 
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "performance_metrics.png"), dpi=300)
        plt.close()
    
    def generate_parameter_analysis(self):
        """Generate analysis of parameter evolution and convergence."""
        # Analyze parameter trends
        param_analysis = self.analyze_strategy_parameters()
        
        # Create output directory
        param_dir = os.path.join(self.output_dir, "parameters")
        os.makedirs(param_dir, exist_ok=True)
        
        # Save parameter analysis to file
        with open(os.path.join(param_dir, "parameter_analysis.json"), 'w') as f:
            json.dump(param_analysis, f, indent=2)
        
        # Generate charts for parameter trends
        self._plot_parameter_trends(param_analysis, param_dir)
        
        # Generate convergence chart
        self._plot_parameter_convergence(param_analysis, param_dir)
        
        # Generate optimal parameter ranges
        self._generate_optimal_parameters(param_analysis, param_dir)
        
        print(f"Generated parameter analysis in {param_dir}")
    
    def _plot_parameter_trends(self, param_analysis: Dict[str, Any], param_dir: str):
        """Plot parameter trends across generations."""
        parameter_trends = param_analysis.get("parameter_trends", {})
        
        for param_key, trends in parameter_trends.items():
            if not trends["generations"] or len(trends["generations"]) < 2:
                continue
                
            # Create plot
            plt.figure(figsize=(10, 6))
            
            generations = trends["generations"]
            mean_values = trends["mean"]
            std_values = trends["std"]
            min_values = trends["min"]
            max_values = trends["max"]
            
            # Plot mean with error bars
            plt.errorbar(generations, mean_values, yerr=std_values, fmt='o-', 
                      capsize=5, label='Mean ± Std Dev')
            
            # Plot min/max range
            plt.fill_between(generations, min_values, max_values, alpha=0.2, 
                          label='Min/Max Range')
            
            # Clean up parameter name for display
            display_name = param_key.replace('_', ' ').title()
            
            plt.xlabel('Generation')
            plt.ylabel('Parameter Value')
            plt.title(f'Evolution of {display_name} Parameter')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save plot
            safe_key = param_key.replace('/', '_')
            plt.tight_layout()
            plt.savefig(os.path.join(param_dir, f"param_{safe_key}.png"), dpi=300)
            plt.close()
    
    def _plot_parameter_convergence(self, param_analysis: Dict[str, Any], param_dir: str):
        """Plot parameter convergence rates."""
        convergence = param_analysis.get("parameter_convergence", {})
        
        if not convergence:
            print("No convergence data available")
            return
            
        # Sort parameters by convergence rate
        sorted_params = sorted(convergence.items(), key=lambda x: x[1], reverse=True)
        
        param_names = [p[0].replace('_', ' ').title() for p in sorted_params]
        convergence_rates = [p[1] for p in sorted_params]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        y_pos = np.arange(len(param_names))
        plt.barh(y_pos, convergence_rates, align='center')
        plt.yticks(y_pos, param_names)
        
        plt.xlabel('Convergence Rate')
        plt.title('Parameter Convergence Rates')
        
        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, "parameter_convergence.png"), dpi=300)
        plt.close()
    
    def _generate_optimal_parameters(self, param_analysis: Dict[str, Any], param_dir: str):
        """Generate report of optimal parameter ranges."""
        optimal_ranges = param_analysis.get("optimal_ranges", {})
        
        if not optimal_ranges:
            print("No optimal parameter ranges available")
            return
            
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "parameter_recommendations": {}
        }
        
        # Group by strategy type
        strategy_params = {}
        
        for param_key, range_data in optimal_ranges.items():
            parts = param_key.split('_')
            strategy_type = parts[0]
            param_name = '_'.join(parts[1:])
            
            if strategy_type not in strategy_params:
                strategy_params[strategy_type] = {}
                
            strategy_params[strategy_type][param_name] = range_data
        
        # Format report
        for strategy_type, params in strategy_params.items():
            report["parameter_recommendations"][strategy_type] = {
                "strategy_type": strategy_type,
                "parameters": {}
            }
            
            for param_name, range_data in params.items():
                report["parameter_recommendations"][strategy_type]["parameters"][param_name] = {
                    "optimal_value": range_data["mean"],
                    "recommended_range": [
                        max(0, range_data["mean"] - range_data["std"]),
                        range_data["mean"] + range_data["std"]
                    ],
                    "observed_range": [range_data["min"], range_data["max"]]
                }
        
        # Save report
        with open(os.path.join(param_dir, "optimal_parameters.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also create a human-readable version
        with open(os.path.join(param_dir, "parameter_recommendations.md"), 'w') as f:
            f.write("# Strategy Parameter Recommendations\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for strategy_type, data in report["parameter_recommendations"].items():
                f.write(f"## {strategy_type}\n\n")
                f.write("| Parameter | Optimal Value | Recommended Range | Observed Range |\n")
                f.write("|-----------|--------------|------------------|---------------|\n")
                
                for param_name, param_data in data["parameters"].items():
                    f.write(f"| {param_name} | {param_data['optimal_value']:.2f} | ")
                    f.write(f"{param_data['recommended_range'][0]:.2f} - {param_data['recommended_range'][1]:.2f} | ")
                    f.write(f"{param_data['observed_range'][0]:.2f} - {param_data['observed_range'][1]:.2f} |\n")
                
                f.write("\n")


def main():
    """Run the analysis process."""
    parser = argparse.ArgumentParser(description='Analyze evolution results')
    parser.add_argument('--test-dir', type=str, required=True, help='Directory containing test results')
    
    args = parser.parse_args()
    
    print(f"Analyzing results in directory: {args.test_dir}")
    
    # Initialize analyzer
    analyzer = EvolutionAnalyzer(args.test_dir)
    
    # Generate performance charts
    analyzer.generate_performance_charts()
    
    # Generate parameter analysis
    analyzer.generate_parameter_analysis()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
