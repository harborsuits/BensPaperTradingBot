"""
A/B Testing Framework for BensBot-EvoTrader Integration

This module provides tools for conducting A/B tests between original strategies
and evolved variants to measure the effectiveness of the evolution system.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import random
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter
from benbot.evotrader_bridge.testing_framework import SimulationEnvironment

logger = logging.getLogger(__name__)


class ABTest:
    """Class for conducting A/B tests between original and evolved strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the A/B testing framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "output_dir": "ab_test_results",
            "test_count": 10,  # Number of tests to run per strategy
            "bootstrap_samples": 100,  # For statistical significance
            "significance_level": 0.05,  # For p-value calculation
            "metrics_to_compare": ["profit", "win_rate", "max_drawdown", "sharpe_ratio"],
            "simulation_config": {
                "data_source": "historical",
                "initial_balance": 10000,
                "fee_rate": 0.001,
                "slippage": 0.0005
            }
        }
        
        # Create output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize simulation environment
        sim_config = self.config["simulation_config"]
        sim_config["output_dir"] = os.path.join(self.output_dir, "simulations")
        self.simulation_env = SimulationEnvironment(sim_config)
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.ab_test")
        self.setup_logger()
        
        self.logger.info(f"A/B testing framework initialized with config: {json.dumps(self.config, indent=2)}")
    
    def setup_logger(self):
        """Configure logging for the A/B testing framework."""
        log_file = os.path.join(self.output_dir, "ab_test.log")
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
    
    def compare_strategies(
        self,
        original_strategy: BensBotStrategyAdapter,
        evolved_strategy: BensBotStrategyAdapter,
        test_name: str = None
    ) -> Dict[str, Any]:
        """
        Compare an original strategy with an evolved variant.
        
        Args:
            original_strategy: Original strategy
            evolved_strategy: Evolved strategy variant
            test_name: Optional name for this test
            
        Returns:
            Comparison results dictionary
        """
        if test_name is None:
            test_name = f"abtest_{str(uuid.uuid4())[:8]}"
            
        self.logger.info(f"Starting A/B test '{test_name}' comparing strategies "
                        f"{original_strategy.strategy_id} and {evolved_strategy.strategy_id}")
        
        # Create test directory
        test_dir = os.path.join(self.output_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)
        
        # Run multiple tests with different data/conditions
        original_results = []
        evolved_results = []
        
        for i in range(self.config["test_count"]):
            test_id = f"{test_name}_run_{i+1}"
            self.logger.info(f"Running test iteration {i+1}/{self.config['test_count']}")
            
            # Test original strategy
            try:
                orig_metrics = self.simulation_env.test_strategy(
                    original_strategy, 
                    test_id=f"{test_id}_original"
                )
                original_results.append(orig_metrics)
                
                # Test evolved strategy
                evolved_metrics = self.simulation_env.test_strategy(
                    evolved_strategy, 
                    test_id=f"{test_id}_evolved"
                )
                evolved_results.append(evolved_metrics)
                
            except Exception as e:
                self.logger.error(f"Error in test iteration {i+1}: {str(e)}")
                continue
        
        # Calculate comparison statistics
        comparison = self._calculate_comparison_stats(original_results, evolved_results)
        
        # Save results
        self._save_test_results(test_dir, test_name, original_results, evolved_results, comparison)
        
        # Generate visualization
        self._generate_comparison_charts(test_dir, test_name, original_results, evolved_results, comparison)
        
        self.logger.info(f"A/B test '{test_name}' completed with {len(original_results)} valid test runs")
        
        return comparison
    
    def _calculate_comparison_stats(self, original_results: List[Dict[str, Any]], evolved_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistical comparison between original and evolved strategy results.
        
        Args:
            original_results: List of performance metrics for original strategy
            evolved_results: List of performance metrics for evolved strategy
            
        Returns:
            Comparison statistics dictionary
        """
        if not original_results or not evolved_results:
            return {
                "error": "Insufficient data for comparison",
                "is_improvement": False
            }
            
        # Get metrics to compare
        metrics = self.config["metrics_to_compare"]
        
        # Initialize comparison results
        comparison = {
            "metrics": {},
            "overall": {
                "improvement_score": 0,
                "statistically_significant": False,
                "is_improvement": False
            }
        }
        
        # Calculate improvement for each metric
        improvement_count = 0
        significant_count = 0
        
        for metric in metrics:
            # Extract metric values
            orig_values = [r.get(metric, 0) for r in original_results]
            evol_values = [r.get(metric, 0) for r in evolved_results]
            
            # Calculate basic statistics
            orig_mean = np.mean(orig_values)
            evol_mean = np.mean(evol_values)
            orig_std = np.std(orig_values) if len(orig_values) > 1 else 0
            evol_std = np.std(evol_values) if len(evol_values) > 1 else 0
            
            # Calculate absolute and percentage difference
            abs_diff = evol_mean - orig_mean
            pct_diff = (abs_diff / orig_mean) * 100 if orig_mean != 0 else 0
            
            # Determine improvement direction
            # For most metrics higher is better, but for some (like drawdown) lower is better
            higher_is_better = True
            if metric == "max_drawdown":
                higher_is_better = False
                
            is_improvement = (higher_is_better and abs_diff > 0) or (not higher_is_better and abs_diff < 0)
            
            # Calculate statistical significance using bootstrap
            p_value = self._calculate_p_value(orig_values, evol_values, higher_is_better)
            is_significant = p_value < self.config["significance_level"]
            
            # Store metric comparison
            comparison["metrics"][metric] = {
                "original_mean": orig_mean,
                "evolved_mean": evol_mean,
                "original_std": orig_std,
                "evolved_std": evol_std,
                "absolute_diff": abs_diff,
                "percentage_diff": pct_diff,
                "is_improvement": is_improvement,
                "p_value": p_value,
                "is_significant": is_significant
            }
            
            # Count improvements
            if is_improvement:
                improvement_count += 1
                
            # Count significant improvements
            if is_improvement and is_significant:
                significant_count += 1
        
        # Calculate overall improvement score
        improvement_score = improvement_count / len(metrics) if metrics else 0
        significant_score = significant_count / len(metrics) if metrics else 0
        
        # Set overall results
        comparison["overall"] = {
            "improvement_score": improvement_score,
            "significant_score": significant_score,
            "metrics_compared": len(metrics),
            "metrics_improved": improvement_count,
            "metrics_significantly_improved": significant_count,
            "is_improvement": improvement_score > 0.5,  # Improved in majority of metrics
            "is_significant_improvement": significant_score > 0.3  # Significantly improved in at least 30% of metrics
        }
        
        return comparison
    
    def _calculate_p_value(self, original_values: List[float], evolved_values: List[float], higher_is_better: bool) -> float:
        """
        Calculate p-value for the difference between two sets of values using bootstrap resampling.
        
        Args:
            original_values: Values from original strategy
            evolved_values: Values from evolved strategy
            higher_is_better: Whether higher values are better for this metric
            
        Returns:
            p-value indicating statistical significance
        """
        # If not enough samples, return inconclusive p-value
        if len(original_values) < 2 or len(evolved_values) < 2:
            return 0.5
            
        # Calculate observed difference in means
        orig_mean = np.mean(original_values)
        evol_mean = np.mean(evolved_values)
        observed_diff = evol_mean - orig_mean
        
        # If not higher is better, invert the difference
        if not higher_is_better:
            observed_diff = -observed_diff
        
        # Combine samples for bootstrapping
        combined = original_values + evolved_values
        n_orig = len(original_values)
        n_evol = len(evolved_values)
        n_combined = len(combined)
        
        # Bootstrap to estimate p-value
        n_samples = self.config["bootstrap_samples"]
        count_greater_diff = 0
        
        for _ in range(n_samples):
            # Shuffle and resample
            np.random.shuffle(combined)
            boot_orig = combined[:n_orig]
            boot_evol = combined[n_orig:n_orig+n_evol]
            
            # Calculate bootstrapped difference
            boot_orig_mean = np.mean(boot_orig)
            boot_evol_mean = np.mean(boot_evol)
            boot_diff = boot_evol_mean - boot_orig_mean
            
            # If not higher is better, invert the difference
            if not higher_is_better:
                boot_diff = -boot_diff
            
            # Count how many bootstrapped differences exceed observed
            if boot_diff >= observed_diff:
                count_greater_diff += 1
        
        # Calculate p-value
        p_value = count_greater_diff / n_samples
        return p_value
    
    def _save_test_results(self, test_dir: str, test_name: str, original_results: List[Dict[str, Any]], 
                          evolved_results: List[Dict[str, Any]], comparison: Dict[str, Any]):
        """
        Save A/B test results to files.
        
        Args:
            test_dir: Directory to save results
            test_name: Name of the test
            original_results: Results from original strategy
            evolved_results: Results from evolved strategy
            comparison: Comparison statistics
        """
        # Save raw results
        with open(os.path.join(test_dir, "original_results.json"), 'w') as f:
            json.dump(original_results, f, indent=2)
            
        with open(os.path.join(test_dir, "evolved_results.json"), 'w') as f:
            json.dump(evolved_results, f, indent=2)
            
        # Save comparison
        with open(os.path.join(test_dir, "comparison.json"), 'w') as f:
            json.dump(comparison, f, indent=2)
            
        # Save summary
        summary = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "test_count": len(original_results),
            "metrics_compared": self.config["metrics_to_compare"],
            "overall_result": comparison["overall"],
            "key_metrics": {}
        }
        
        # Include key metrics in summary
        for metric in self.config["metrics_to_compare"]:
            if metric in comparison["metrics"]:
                metric_result = comparison["metrics"][metric]
                summary["key_metrics"][metric] = {
                    "original": metric_result["original_mean"],
                    "evolved": metric_result["evolved_mean"],
                    "improvement": metric_result["percentage_diff"],
                    "is_significant": metric_result["is_significant"]
                }
        
        with open(os.path.join(test_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Saved test results to {test_dir}")
    
    def _generate_comparison_charts(self, test_dir: str, test_name: str, original_results: List[Dict[str, Any]], 
                                   evolved_results: List[Dict[str, Any]], comparison: Dict[str, Any]):
        """
        Generate charts comparing original and evolved strategy performance.
        
        Args:
            test_dir: Directory to save charts
            test_name: Name of the test
            original_results: Results from original strategy
            evolved_results: Results from evolved strategy
            comparison: Comparison statistics
        """
        try:
            # Create charts directory
            charts_dir = os.path.join(test_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # 1. Bar chart comparing key metrics
            self._generate_metrics_comparison_chart(charts_dir, comparison)
            
            # 2. Box plots for distribution of results
            self._generate_distribution_chart(charts_dir, original_results, evolved_results)
            
            # 3. Statistical significance visualization
            self._generate_significance_chart(charts_dir, comparison)
            
            self.logger.info(f"Generated comparison charts in {charts_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {str(e)}")
    
    def _generate_metrics_comparison_chart(self, charts_dir: str, comparison: Dict[str, Any]):
        """
        Generate bar chart comparing metrics between original and evolved strategies.
        
        Args:
            charts_dir: Directory to save chart
            comparison: Comparison statistics
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = []
        orig_values = []
        evol_values = []
        colors = []
        
        for metric, data in comparison["metrics"].items():
            metrics.append(metric)
            orig_values.append(data["original_mean"])
            evol_values.append(data["evolved_mean"])
            
            # Green for improvement, red for regression
            colors.append('green' if data["is_improvement"] else 'red')
        
        # Set up bar positions
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, orig_values, width, label='Original', color='blue', alpha=0.7)
        ax.bar(x + width/2, evol_values, width, label='Evolved', color=colors, alpha=0.7)
        
        # Add labels and titles
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Comparison of Strategy Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add improvement percentage annotations
        for i, metric in enumerate(metrics):
            data = comparison["metrics"][metric]
            pct_diff = data["percentage_diff"]
            sign = '+' if pct_diff > 0 else ''
            
            # Only show if significant
            if data["is_significant"]:
                ax.annotate(f"{sign}{pct_diff:.1f}%*", 
                          xy=(i + width/2, evol_values[i]),
                          xytext=(0, 3), 
                          textcoords="offset points",
                          ha='center', va='bottom')
            else:
                ax.annotate(f"{sign}{pct_diff:.1f}%", 
                          xy=(i + width/2, evol_values[i]),
                          xytext=(0, 3), 
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        # Add a note about significance
        fig.text(0.5, 0.01, "* Indicates statistically significant difference (p < 0.05)", 
                ha='center', fontsize=10)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "metrics_comparison.png"), dpi=300)
        plt.close()
    
    def _generate_distribution_chart(self, charts_dir: str, original_results: List[Dict[str, Any]], 
                                    evolved_results: List[Dict[str, Any]]):
        """
        Generate box plots showing distribution of results.
        
        Args:
            charts_dir: Directory to save chart
            original_results: Results from original strategy
            evolved_results: Results from evolved strategy
        """
        metrics = self.config["metrics_to_compare"]
        
        # Create subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
        
        # Handle case with only one metric
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract values
            orig_values = [r.get(metric, 0) for r in original_results]
            evol_values = [r.get(metric, 0) for r in evolved_results]
            
            # Create box plots
            box_data = [orig_values, evol_values]
            box = ax.boxplot(box_data, patch_artist=True, labels=['Original', 'Evolved'])
            
            # Set colors
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric} Distribution')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean line
            for j, data in enumerate(box_data):
                mean = np.mean(data)
                ax.axhline(mean, color='r', linestyle='--', alpha=0.3, 
                           xmin=j/len(box_data), xmax=(j+1)/len(box_data))
                ax.text(j+1, mean, f'Mean: {mean:.2f}', 
                         ha='center', va='bottom', color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "distributions.png"), dpi=300)
        plt.close()
    
    def _generate_significance_chart(self, charts_dir: str, comparison: Dict[str, Any]):
        """
        Generate chart visualizing statistical significance of improvements.
        
        Args:
            charts_dir: Directory to save chart
            comparison: Comparison statistics
        """
        metrics = list(comparison["metrics"].keys())
        p_values = [comparison["metrics"][m]["p_value"] for m in metrics]
        improvements = [comparison["metrics"][m]["percentage_diff"] for m in metrics]
        significant = [comparison["metrics"][m]["is_significant"] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        scatter = ax.scatter(p_values, improvements, 
                           c=['green' if s else 'red' for s in significant], 
                           alpha=0.7, s=100)
        
        # Add labels for each point
        for i, metric in enumerate(metrics):
            ax.annotate(metric, (p_values[i], improvements[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        # Add threshold line
        ax.axvline(x=self.config["significance_level"], color='r', linestyle='--')
        ax.text(self.config["significance_level"], ax.get_ylim()[0], 
                f'p={self.config["significance_level"]}', 
                ha='center', va='bottom', color='red')
        
        # Add zero improvement line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel('p-value (lower is more significant)')
        ax.set_ylabel('Improvement Percentage')
        ax.set_title('Statistical Significance of Strategy Improvements')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Significant', 
                      markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Not Significant', 
                      markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements)
        
        # Add quadrant labels
        ax.text(0.1, ax.get_ylim()[1]*0.9, "Significant Improvement", 
                ha='left', va='top', color='green', alpha=0.7)
        ax.text(0.9, ax.get_ylim()[1]*0.9, "Possible Improvement\n(needs more data)", 
                ha='right', va='top', color='black', alpha=0.7)
        ax.text(0.1, ax.get_ylim()[0]*0.9, "Significant Regression", 
                ha='left', va='bottom', color='red', alpha=0.7)
        ax.text(0.9, ax.get_ylim()[0]*0.9, "Possible Regression\n(needs more data)", 
                ha='right', va='bottom', color='black', alpha=0.7)
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "significance.png"), dpi=300)
        plt.close()


class ABTestBatch:
    """Run a batch of A/B tests on multiple strategy pairs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the A/B test batch runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "output_dir": "ab_test_batch",
            "parallel_tests": 1,  # Number of tests to run in parallel
            "abtest_config": {}  # Passed to ABTest instances
        }
        
        # Create output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.ab_test_batch")
        self.setup_logger()
        
        self.logger.info(f"A/B test batch runner initialized with config: {json.dumps(self.config, indent=2)}")
    
    def setup_logger(self):
        """Configure logging for the A/B test batch runner."""
        log_file = os.path.join(self.output_dir, "batch.log")
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
    
    def run_batch(self, strategy_pairs: List[Tuple[BensBotStrategyAdapter, BensBotStrategyAdapter]]) -> Dict[str, Any]:
        """
        Run A/B tests on multiple pairs of strategies.
        
        Args:
            strategy_pairs: List of (original, evolved) strategy pairs to test
            
        Returns:
            Dictionary with batch results
        """
        self.logger.info(f"Starting batch of {len(strategy_pairs)} A/B tests")
        
        batch_results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(strategy_pairs),
            "successful_tests": 0,
            "improvement_count": 0,
            "significant_improvement_count": 0,
            "test_results": {}
        }
        
        for i, (original, evolved) in enumerate(strategy_pairs):
            test_name = f"test_{i+1}_{original.strategy_id}_vs_{evolved.strategy_id}"
            
            self.logger.info(f"Running test {i+1}/{len(strategy_pairs)}: {test_name}")
            
            try:
                # Create AB test instance
                ab_test = ABTest(self.config.get("abtest_config", {}))
                
                # Run test
                comparison = ab_test.compare_strategies(original, evolved, test_name)
                
                # Record results
                batch_results["test_results"][test_name] = {
                    "original_id": original.strategy_id,
                    "evolved_id": evolved.strategy_id,
                    "overall": comparison["overall"]
                }
                
                # Update counts
                batch_results["successful_tests"] += 1
                if comparison["overall"].get("is_improvement", False):
                    batch_results["improvement_count"] += 1
                if comparison["overall"].get("is_significant_improvement", False):
                    batch_results["significant_improvement_count"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error running test {test_name}: {str(e)}")
                batch_results["test_results"][test_name] = {
                    "original_id": original.strategy_id,
                    "evolved_id": evolved.strategy_id,
                    "error": str(e)
                }
        
        # Calculate summary statistics
        if batch_results["successful_tests"] > 0:
            batch_results["improvement_rate"] = batch_results["improvement_count"] / batch_results["successful_tests"]
            batch_results["significant_improvement_rate"] = batch_results["significant_improvement_count"] / batch_results["successful_tests"]
        else:
            batch_results["improvement_rate"] = 0
            batch_results["significant_improvement_rate"] = 0
            
        # Save batch results
        results_file = os.path.join(self.output_dir, "batch_results.json")
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
            
        self.logger.info(f"Completed batch with {batch_results['successful_tests']} successful tests")
        self.logger.info(f"Improvement rate: {batch_results['improvement_rate']:.2f}, "  
                      f"Significant improvement rate: {batch_results['significant_improvement_rate']:.2f}")
        
        return batch_results
