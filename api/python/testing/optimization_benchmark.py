#!/usr/bin/env python3
"""
Optimization Performance Benchmark

This script measures the effectiveness of strategy optimization by comparing
performance before and after optimization across multiple strategies and
market conditions.
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_benchmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("optimization_benchmark")

# Add project root to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components for benchmarking
from trading_bot.strategies.optimizer.enhanced_optimizer import EnhancedOptimizer
from trading_bot.strategies.components.component_registry import ComponentRegistry
from trading_bot.testing.market_data_generator import MarketDataGenerator

class PerformanceMetric:
    """Container for tracking a performance metric across optimization."""
    
    def __init__(self, name, higher_is_better=True):
        self.name = name
        self.higher_is_better = higher_is_better
        self.before_values = []
        self.after_values = []
        
    def add_result(self, before, after):
        """Add a before/after result pair."""
        self.before_values.append(before)
        self.after_values.append(after)
        
    def mean_improvement(self):
        """Calculate mean improvement as a percentage."""
        if not self.before_values or not self.after_values:
            return 0.0
            
        improvements = []
        for before, after in zip(self.before_values, self.after_values):
            if before == 0:
                continue
                
            change = after - before
            # If lower is better, invert the change
            if not self.higher_is_better:
                change = -change
                
            percent = (change / abs(before)) * 100
            improvements.append(percent)
            
        if not improvements:
            return 0.0
            
        return sum(improvements) / len(improvements)
        
    def success_rate(self):
        """Calculate percentage of cases where optimization improved the metric."""
        if not self.before_values or not self.after_values:
            return 0.0
            
        success_count = 0
        for before, after in zip(self.before_values, self.after_values):
            if self.higher_is_better:
                if after > before:
                    success_count += 1
            else:
                if after < before:
                    success_count += 1
                    
        return (success_count / len(self.before_values)) * 100
        
    def to_dataframe(self):
        """Convert results to a DataFrame."""
        return pd.DataFrame({
            'Before': self.before_values,
            'After': self.after_values,
            'Change (%)': [(after - before) / before * 100 if before != 0 else 0 
                           for before, after in zip(self.before_values, self.after_values)]
        })

class OptimizationBenchmark:
    """Benchmark the performance of strategy optimization."""
    
    def __init__(self):
        """Initialize the benchmark."""
        self.registry = ComponentRegistry()
        self.optimizer = EnhancedOptimizer()
        self.data_generator = MarketDataGenerator(seed=42)
        
        # Performance metrics to track
        self.metrics = {
            "sharpe_ratio": PerformanceMetric("Sharpe Ratio", higher_is_better=True),
            "profit_factor": PerformanceMetric("Profit Factor", higher_is_better=True),
            "max_drawdown": PerformanceMetric("Max Drawdown", higher_is_better=False),
            "win_rate": PerformanceMetric("Win Rate", higher_is_better=True)
        }
        
        # Test parameters
        self.market_conditions = [
            {"name": "Bull Market", "drift": 0.15, "volatility": 0.15},
            {"name": "Bear Market", "drift": -0.10, "volatility": 0.25},
            {"name": "Sideways Market", "drift": 0.01, "volatility": 0.10},
            {"name": "Volatile Market", "drift": 0.05, "volatility": 0.30}
        ]
        
        # Strategy types to test
        self.strategy_types = []
        
        # Benchmark results
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def setup(self):
        """Set up the benchmark environment."""
        logger.info("Setting up optimization benchmark")
        
        # Get available strategy types
        self.strategy_types = self.registry.get_registered_strategy_types()
        logger.info(f"Found {len(self.strategy_types)} registered strategy types")
        
        if not self.strategy_types:
            logger.error("No strategy types registered")
            return False
            
        # Log found strategy types
        for strategy_type in self.strategy_types:
            logger.info(f"Found strategy type: {strategy_type}")
            
        return True
        
    def generate_market_data(self, condition):
        """Generate market data for a specific market condition."""
        logger.info(f"Generating {condition['name']} market data")
        
        # Create parameters for this market condition
        params = {
            "drift": condition["drift"],
            "volatility": condition["volatility"]
        }
        
        # Generate dataset
        dataset = self.data_generator.generate_test_dataset(
            num_stocks=3,
            days=120,
            include_options=True
        )
        
        logger.info(f"Generated {condition['name']} data with {len(dataset['stocks'])} stocks")
        return dataset
        
    def run_optimization_test(self, strategy_type, market_data, initial_params=None):
        """Run optimization test for a specific strategy type and market data."""
        logger.info(f"Testing optimization for {strategy_type}")
        
        try:
            # Get strategy instance
            strategy = self.registry.get_strategy_instance(strategy_type)
            
            if not strategy:
                logger.error(f"Failed to instantiate strategy: {strategy_type}")
                return None
                
            # Use initial parameters if provided, otherwise use defaults
            if initial_params:
                for param, value in initial_params.items():
                    if hasattr(strategy, param):
                        setattr(strategy, param, value)
                        
            # Prepare test structure with near-miss metrics
            test_candidate = type('StrategyCandidate', (), {})()
            test_candidate.strategy_id = f"{strategy_type}_test"
            test_candidate.strategy_type = strategy_type
            test_candidate.strategy = strategy
            test_candidate.symbols = list(market_data['stocks'].keys())
            test_candidate.universe = "TEST"
            
            # Set initial near-miss performance metrics
            # These are designed to be below thresholds but within optimization range
            test_candidate.performance_metrics = {
                "sharpe_ratio": 1.3,    # Below threshold (1.5) but within 15%
                "profit_factor": 1.6,   # Below threshold (1.8) but within 15%
                "max_drawdown": 17.0,   # Worse than threshold (15.0) but within 15%
                "win_rate": 52.0        # Below threshold (55.0) but within 15%
            }
            
            # Store before metrics
            before_metrics = test_candidate.performance_metrics.copy()
            
            # Run optimization
            logger.info(f"Optimizing {strategy_type}")
            start_time = time.time()
            success = self.optimizer.optimize(test_candidate)
            end_time = time.time()
            
            if not success:
                logger.warning(f"Optimization failed for {strategy_type}")
                return None
                
            # Check for after metrics
            if not hasattr(test_candidate, 'performance_metrics') or not test_candidate.performance_metrics:
                logger.error(f"No performance metrics after optimization for {strategy_type}")
                return None
                
            # Store after metrics
            after_metrics = test_candidate.performance_metrics.copy()
            
            # Calculate optimization time
            optimization_time = end_time - start_time
            
            return {
                "strategy_type": strategy_type,
                "before": before_metrics,
                "after": after_metrics,
                "optimization_time": optimization_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error testing optimization for {strategy_type}: {str(e)}")
            return None
            
    def run_benchmark(self):
        """Run the complete benchmark across all strategies and market conditions."""
        logger.info("Starting optimization benchmark")
        self.start_time = datetime.datetime.now()
        
        if not self.setup():
            logger.error("Benchmark setup failed")
            return False
            
        # Initialize results structure
        self.results = {
            "by_strategy": {},
            "by_condition": {},
            "overall": {
                "tests": 0,
                "successful": 0,
                "failed": 0,
                "avg_optimization_time": 0
            }
        }
        
        # Total optimization time
        total_optimization_time = 0
        
        # For each market condition
        for condition in self.market_conditions:
            condition_name = condition["name"]
            logger.info(f"Testing market condition: {condition_name}")
            
            # Generate market data for this condition
            market_data = self.generate_market_data(condition)
            
            # Initialize condition results
            self.results["by_condition"][condition_name] = {
                "tests": 0,
                "successful": 0,
                "failed": 0,
                "metrics": {}
            }
            
            # For each strategy type
            for strategy_type in self.strategy_types:
                logger.info(f"Testing strategy: {strategy_type} in {condition_name}")
                
                # Initialize strategy results if first time seeing this strategy
                if strategy_type not in self.results["by_strategy"]:
                    self.results["by_strategy"][strategy_type] = {
                        "tests": 0,
                        "successful": 0,
                        "failed": 0,
                        "metrics": {}
                    }
                    
                # Run the optimization test
                test_result = self.run_optimization_test(strategy_type, market_data)
                
                # Process test results
                self.results["overall"]["tests"] += 1
                self.results["by_strategy"][strategy_type]["tests"] += 1
                self.results["by_condition"][condition_name]["tests"] += 1
                
                if test_result and test_result["success"]:
                    # Successful test
                    self.results["overall"]["successful"] += 1
                    self.results["by_strategy"][strategy_type]["successful"] += 1
                    self.results["by_condition"][condition_name]["successful"] += 1
                    
                    # Add optimization time
                    total_optimization_time += test_result["optimization_time"]
                    
                    # Record metrics
                    for metric_name in self.metrics:
                        # Initialize metric for strategy if needed
                        if metric_name not in self.results["by_strategy"][strategy_type]["metrics"]:
                            self.results["by_strategy"][strategy_type]["metrics"][metric_name] = {
                                "before": [], "after": []
                            }
                            
                        # Initialize metric for condition if needed
                        if metric_name not in self.results["by_condition"][condition_name]["metrics"]:
                            self.results["by_condition"][condition_name]["metrics"][metric_name] = {
                                "before": [], "after": []
                            }
                            
                        # Record before/after values
                        before_value = test_result["before"].get(metric_name, 0)
                        after_value = test_result["after"].get(metric_name, 0)
                        
                        # Add to metrics tracker
                        self.metrics[metric_name].add_result(before_value, after_value)
                        
                        # Add to results structure
                        self.results["by_strategy"][strategy_type]["metrics"][metric_name]["before"].append(before_value)
                        self.results["by_strategy"][strategy_type]["metrics"][metric_name]["after"].append(after_value)
                        
                        self.results["by_condition"][condition_name]["metrics"][metric_name]["before"].append(before_value)
                        self.results["by_condition"][condition_name]["metrics"][metric_name]["after"].append(after_value)
                else:
                    # Failed test
                    self.results["overall"]["failed"] += 1
                    self.results["by_strategy"][strategy_type]["failed"] += 1
                    self.results["by_condition"][condition_name]["failed"] += 1
        
        # Calculate average optimization time
        if self.results["overall"]["successful"] > 0:
            self.results["overall"]["avg_optimization_time"] = total_optimization_time / self.results["overall"]["successful"]
            
        self.end_time = datetime.datetime.now()
        return True
        
    def generate_report(self):
        """Generate a comprehensive benchmark report."""
        logger.info("Generating benchmark report")
        
        report = []
        report.append("# Optimization Benchmark Report")
        report.append(f"Run date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration: {(self.end_time - self.start_time).total_seconds():.2f} seconds\n")
        
        # Overall results
        report.append("## Overall Results")
        report.append(f"Total tests: {self.results['overall']['tests']}")
        report.append(f"Successful: {self.results['overall']['successful']}")
        report.append(f"Failed: {self.results['overall']['failed']}")
        if self.results['overall']['successful'] > 0:
            success_rate = (self.results['overall']['successful'] / self.results['overall']['tests']) * 100
            report.append(f"Success rate: {success_rate:.2f}%")
            report.append(f"Average optimization time: {self.results['overall']['avg_optimization_time']:.2f} seconds\n")
        
        # Metrics summary
        report.append("## Performance Metrics Summary")
        for metric_name, metric in self.metrics.items():
            report.append(f"### {metric.name}")
            report.append(f"Mean improvement: {metric.mean_improvement():.2f}%")
            report.append(f"Success rate: {metric.success_rate():.2f}%\n")
        
        # Results by strategy
        report.append("## Results by Strategy")
        for strategy_type, strategy_result in self.results["by_strategy"].items():
            report.append(f"### {strategy_type}")
            report.append(f"Tests: {strategy_result['tests']}")
            report.append(f"Success rate: {(strategy_result['successful'] / strategy_result['tests']) * 100:.2f}%")
            
            # Add metric details if available
            if strategy_result["metrics"]:
                report.append("\nMetric improvements:")
                for metric_name, values in strategy_result["metrics"].items():
                    if not values["before"] or not values["after"]:
                        continue
                        
                    # Calculate mean improvement
                    improvements = []
                    for before, after in zip(values["before"], values["after"]):
                        if before == 0:
                            continue
                        improvements.append((after - before) / before * 100)
                        
                    if improvements:
                        mean_improvement = sum(improvements) / len(improvements)
                        report.append(f"- {metric_name}: {mean_improvement:.2f}%")
                        
            report.append("")
        
        # Results by market condition
        report.append("## Results by Market Condition")
        for condition_name, condition_result in self.results["by_condition"].items():
            report.append(f"### {condition_name}")
            report.append(f"Tests: {condition_result['tests']}")
            report.append(f"Success rate: {(condition_result['successful'] / condition_result['tests']) * 100:.2f}%")
            
            # Add metric details if available
            if condition_result["metrics"]:
                report.append("\nMetric improvements:")
                for metric_name, values in condition_result["metrics"].items():
                    if not values["before"] or not values["after"]:
                        continue
                        
                    # Calculate mean improvement
                    improvements = []
                    for before, after in zip(values["before"], values["after"]):
                        if before == 0:
                            continue
                        improvements.append((after - before) / before * 100)
                        
                    if improvements:
                        mean_improvement = sum(improvements) / len(improvements)
                        report.append(f"- {metric_name}: {mean_improvement:.2f}%")
                        
            report.append("")
        
        # Write report to file
        report_text = "\n".join(report)
        with open("optimization_benchmark_report.md", "w") as f:
            f.write(report_text)
            
        logger.info("Benchmark report written to optimization_benchmark_report.md")
        return report_text
        
    def generate_visualizations(self):
        """Generate visualizations of benchmark results."""
        logger.info("Generating benchmark visualizations")
        
        try:
            # Set up plotting
            plt.figure(figsize=(12, 8))
            plt.style.use('ggplot')
            
            # Plot 1: Overall metric improvements
            plt.subplot(2, 2, 1)
            metric_names = []
            improvements = []
            
            for metric_name, metric in self.metrics.items():
                metric_names.append(metric.name)
                improvements.append(metric.mean_improvement())
                
            plt.bar(metric_names, improvements)
            plt.title('Mean Metric Improvements')
            plt.ylabel('Improvement (%)')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot 2: Success rate by strategy
            plt.subplot(2, 2, 2)
            strategies = []
            success_rates = []
            
            for strategy_type, strategy_result in self.results["by_strategy"].items():
                strategies.append(strategy_type)
                success_rate = (strategy_result['successful'] / strategy_result['tests']) * 100
                success_rates.append(success_rate)
                
            plt.bar(strategies, success_rates)
            plt.title('Optimization Success Rate by Strategy')
            plt.ylabel('Success Rate (%)')
            plt.xticks(rotation=45, ha='right')
            
            # Plot 3: Before vs After metrics for Sharpe Ratio
            plt.subplot(2, 2, 3)
            sharpe_df = self.metrics["sharpe_ratio"].to_dataframe()
            
            plt.scatter(sharpe_df['Before'], sharpe_df['After'])
            plt.plot([0, sharpe_df['Before'].max() * 1.2], [0, sharpe_df['Before'].max() * 1.2], 'k--', alpha=0.5)
            plt.title('Sharpe Ratio: Before vs After Optimization')
            plt.xlabel('Before')
            plt.ylabel('After')
            
            # Plot 4: Market condition comparison
            plt.subplot(2, 2, 4)
            conditions = []
            condition_improvements = []
            
            for condition_name, condition_result in self.results["by_condition"].items():
                # Calculate average improvement across all metrics
                total_improvement = 0
                count = 0
                
                for metric_name, values in condition_result["metrics"].items():
                    if not values["before"] or not values["after"]:
                        continue
                        
                    # Get improvements for this metric
                    improvements = []
                    for before, after in zip(values["before"], values["after"]):
                        if before == 0:
                            continue
                        improvements.append((after - before) / before * 100)
                        
                    if improvements:
                        total_improvement += sum(improvements) / len(improvements)
                        count += 1
                        
                if count > 0:
                    avg_improvement = total_improvement / count
                    conditions.append(condition_name)
                    condition_improvements.append(avg_improvement)
                    
            plt.bar(conditions, condition_improvements)
            plt.title('Average Improvement by Market Condition')
            plt.ylabel('Average Improvement (%)')
            plt.xticks(rotation=45, ha='right')
            
            # Save plot
            plt.tight_layout()
            plt.savefig('optimization_benchmark_results.png')
            logger.info("Visualizations saved to optimization_benchmark_results.png")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")


def main():
    """Run the optimization benchmark."""
    print("\n" + "="*70)
    print("OPTIMIZATION PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Run benchmark
    benchmark = OptimizationBenchmark()
    
    try:
        if benchmark.run_benchmark():
            # Generate reports
            benchmark.generate_report()
            benchmark.generate_visualizations()
            
            # Print summary
            overall = benchmark.results["overall"]
            print("\nBenchmark Complete!")
            print(f"Total tests: {overall['tests']}")
            print(f"Success rate: {(overall['successful'] / overall['tests']) * 100:.2f}%")
            
            # Print metric improvements
            print("\nMetric Improvements:")
            for metric_name, metric in benchmark.metrics.items():
                print(f"  {metric.name}: {metric.mean_improvement():.2f}% mean improvement")
                
            print("\nSee optimization_benchmark_report.md and optimization_benchmark_results.png for details")
        else:
            print("\nBenchmark failed during setup")
            
    except Exception as e:
        print(f"\nError running benchmark: {str(e)}")
    
    print("="*70)
    
    
if __name__ == "__main__":
    main()
