#!/usr/bin/env python3
"""
Strategy Registry - Tracks, analyzes, and stores evolved trading strategies
"""

import os
import json
import hashlib
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class StrategyRegistry:
    """
    Registry for tracking, analyzing and storing evolved trading strategies.
    
    Features:
    - Fingerprinting strategies based on parameters and behavior
    - Tracking performance metrics across market conditions
    - Clustering similar strategies
    - Providing insights on strategy evolution
    """
    
    def __init__(self, registry_dir: str = "strategy_registry"):
        """
        Initialize the strategy registry.
        
        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = registry_dir
        self.strategies_path = os.path.join(registry_dir, "strategies")
        self.performance_path = os.path.join(registry_dir, "performance")
        self.analysis_path = os.path.join(registry_dir, "analysis")
        
        # Create directories if they don't exist
        for path in [self.registry_dir, self.strategies_path, 
                    self.performance_path, self.analysis_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Load existing strategies
        self.strategies = self._load_strategies()
        self.strategy_performance = self._load_performance()
    
    def _load_strategies(self) -> Dict[str, Any]:
        """Load all registered strategies"""
        strategies = {}
        
        if not os.path.exists(self.strategies_path):
            return strategies
            
        for filename in os.listdir(self.strategies_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.strategies_path, filename)
                try:
                    with open(filepath, 'r') as file:
                        strategy_data = json.load(file)
                        strategy_id = strategy_data.get('id')
                        if strategy_id:
                            strategies[strategy_id] = strategy_data
                except Exception as e:
                    print(f"Error loading strategy {filename}: {e}")
        
        return strategies
    
    def _load_performance(self) -> Dict[str, Dict[str, Any]]:
        """Load performance data for all strategies"""
        performance = {}
        
        if not os.path.exists(self.performance_path):
            return performance
            
        for filename in os.listdir(self.performance_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.performance_path, filename)
                try:
                    with open(filepath, 'r') as file:
                        perf_data = json.load(file)
                        strategy_id = filename.split('.')[0]
                        performance[strategy_id] = perf_data
                except Exception as e:
                    print(f"Error loading performance {filename}: {e}")
        
        return performance
    
    def _compute_strategy_fingerprint(self, strategy: Any) -> str:
        """
        Compute a unique fingerprint for a strategy based on its parameters.
        
        Args:
            strategy: A strategy object with parameters
            
        Returns:
            Unique hash identifier
        """
        # Get strategy type
        strategy_type = strategy.__class__.__name__
        
        # Get strategy parameters as sorted items for consistent hashing
        params = getattr(strategy, 'parameters', {})
        if not params:
            params = {}
            
        # Convert params to a string representation
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash from type and parameters
        base_hash = hashlib.md5(f"{strategy_type}:{param_str}".encode()).hexdigest()
        
        return base_hash
    
    def register_strategy(self, strategy: Any, generation: int = 0, 
                         parent_ids: List[str] = None) -> str:
        """
        Register a strategy in the registry.
        
        Args:
            strategy: Strategy object to register
            generation: Generation number
            parent_ids: IDs of parent strategies if this is a result of crossover
            
        Returns:
            Strategy ID
        """
        # Compute fingerprint to identify the strategy
        strategy_id = self._compute_strategy_fingerprint(strategy)
        
        # If strategy already exists, update attributes and return existing ID
        if strategy_id in self.strategies:
            return strategy_id
        
        # Extract strategy attributes
        strategy_type = strategy.__class__.__name__
        parameters = getattr(strategy, 'parameters', {})
        
        # Create strategy metadata
        timestamp = datetime.datetime.now().isoformat()
        strategy_data = {
            'id': strategy_id,
            'type': strategy_type,
            'parameters': parameters,
            'registered_at': timestamp,
            'generation': generation,
            'parent_ids': parent_ids or [],
            'lineage': [],  # Will be computed later if parents exist
            'mutation_history': []  # Track parameter changes over time
        }
        
        # Compute lineage if parents exist
        if parent_ids:
            lineage = self._compute_lineage(parent_ids)
            strategy_data['lineage'] = lineage
        
        # Save strategy data
        self.strategies[strategy_id] = strategy_data
        strategy_file = os.path.join(self.strategies_path, f"{strategy_id}.json")
        with open(strategy_file, 'w') as file:
            json.dump(strategy_data, file, indent=2)
        
        return strategy_id
    
    def _compute_lineage(self, parent_ids: List[str]) -> List[Dict[str, Any]]:
        """Compute the evolutionary lineage of a strategy"""
        lineage = []
        
        for parent_id in parent_ids:
            if parent_id in self.strategies:
                parent = self.strategies[parent_id]
                lineage.append({
                    'id': parent_id,
                    'type': parent['type'],
                    'generation': parent['generation']
                })
                # Add parent's lineage if it exists
                parent_lineage = parent.get('lineage', [])
                lineage.extend(parent_lineage)
        
        return lineage
    
    def record_performance(self, strategy_id: str, market_condition: str, 
                          metrics: Dict[str, Any]) -> None:
        """
        Record performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            market_condition: Label for market condition (e.g., 'bull', 'bear', etc.)
            metrics: Dictionary of performance metrics
        """
        if strategy_id not in self.strategies:
            print(f"Strategy {strategy_id} not found in registry")
            return
        
        # Initialize performance record if it doesn't exist
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'market_conditions': {}
            }
        
        # Add timestamp to metrics
        metrics['timestamp'] = datetime.datetime.now().isoformat()
        
        # Store metrics for this market condition
        self.strategy_performance[strategy_id]['market_conditions'][market_condition] = metrics
        
        # Calculate overall performance
        self._update_overall_performance(strategy_id)
        
        # Save performance data
        perf_file = os.path.join(self.performance_path, f"{strategy_id}.json")
        with open(perf_file, 'w') as file:
            json.dump(self.strategy_performance[strategy_id], file, indent=2)
    
    def _update_overall_performance(self, strategy_id: str) -> None:
        """Update the overall performance metrics for a strategy"""
        if strategy_id not in self.strategy_performance:
            return
            
        performance = self.strategy_performance[strategy_id]
        condition_metrics = performance.get('market_conditions', {})
        
        if not condition_metrics:
            return
            
        # Calculate aggregated statistics
        total_return = []
        win_rates = []
        drawdowns = []
        sharpe_ratios = []
        
        for condition, metrics in condition_metrics.items():
            if isinstance(metrics, dict):
                if 'total_return' in metrics:
                    total_return.append(metrics['total_return'])
                if 'win_rate' in metrics:
                    win_rates.append(metrics['win_rate'])
                if 'max_drawdown' in metrics:
                    drawdowns.append(metrics['max_drawdown'])
                if 'sharpe_ratio' in metrics:
                    sharpe_ratios.append(metrics['sharpe_ratio'])
        
        # Store overall performance
        overall = {}
        
        if total_return:
            overall['avg_return'] = np.mean(total_return)
            overall['return_std'] = np.std(total_return)
            overall['min_return'] = min(total_return)
            overall['max_return'] = max(total_return)
        
        if win_rates:
            overall['avg_win_rate'] = np.mean(win_rates)
        
        if drawdowns:
            overall['avg_drawdown'] = np.mean(drawdowns)
            overall['max_drawdown'] = max(drawdowns)
        
        if sharpe_ratios:
            overall['avg_sharpe'] = np.mean(sharpe_ratios)
        
        # Calculate robustness score
        if total_return and drawdowns:
            consistency = 1 - (np.std(total_return) / (np.mean(total_return) + 1e-6))
            consistency = max(0, min(1, consistency))
            
            avg_return = np.mean(total_return)
            normalized_return = 1 / (1 + np.exp(-avg_return/10))  # Sigmoid to normalize
            
            risk_factor = 1 - (np.mean(drawdowns) / 100)
            risk_factor = max(0, min(1, risk_factor))
            
            robustness = (0.4 * normalized_return + 
                         0.3 * consistency + 
                         0.3 * risk_factor)
            
            overall['robustness_score'] = robustness
        
        self.strategy_performance[strategy_id]['overall'] = overall
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy data by ID"""
        return self.strategies.get(strategy_id)
    
    def get_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get performance data by strategy ID"""
        return self.strategy_performance.get(strategy_id)
    
    def generate_performance_report(self, strategy_id: str) -> Dict[str, Any]:
        """
        Generate a detailed performance report for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Report data dictionary
        """
        if strategy_id not in self.strategies:
            return {"error": f"Strategy {strategy_id} not found"}
            
        if strategy_id not in self.strategy_performance:
            return {"error": f"No performance data for strategy {strategy_id}"}
        
        strategy = self.strategies[strategy_id]
        performance = self.strategy_performance[strategy_id]
        
        # Generate report data
        report = {
            "strategy_id": strategy_id,
            "strategy_type": strategy["type"],
            "parameters": strategy["parameters"],
            "generation": strategy["generation"],
            "performance": performance,
            "timestamp": datetime.datetime.now().isoformat(),
            "lineage_depth": len(strategy.get("lineage", [])),
            "parent_count": len(strategy.get("parent_ids", [])),
        }
        
        # Generate report filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.analysis_path, 
                                 f"{strategy_id}_{timestamp_str}_report.json")
        
        with open(report_file, 'w') as file:
            json.dump(report, file, indent=2)
        
        return report
    
    def plot_strategy_performance(self, strategy_id: str, 
                                 save_path: Optional[str] = None) -> None:
        """
        Create performance visualization for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            save_path: Optional path to save the visualization
        """
        if strategy_id not in self.strategy_performance:
            print(f"No performance data for strategy {strategy_id}")
            return
        
        performance = self.strategy_performance[strategy_id]
        conditions = performance.get('market_conditions', {})
        
        if not conditions:
            print(f"No market condition data for strategy {strategy_id}")
            return
        
        # Set up the visualization
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        condition_names = []
        returns = []
        drawdowns = []
        win_rates = []
        
        for condition, metrics in conditions.items():
            condition_names.append(condition)
            returns.append(metrics.get('total_return', 0))
            drawdowns.append(metrics.get('max_drawdown', 0))
            win_rates.append(metrics.get('win_rate', 0))
        
        # Create subplots
        plt.subplot(3, 1, 1)
        plt.bar(condition_names, returns)
        plt.title('Returns by Market Condition')
        plt.ylabel('Return %')
        
        plt.subplot(3, 1, 2)
        plt.bar(condition_names, drawdowns)
        plt.title('Max Drawdown by Market Condition')
        plt.ylabel('Drawdown %')
        
        plt.subplot(3, 1, 3)
        plt.bar(condition_names, win_rates)
        plt.title('Win Rate by Market Condition')
        plt.ylabel('Win Rate %')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Performance visualization saved to {save_path}")
        else:
            # Default save location in analysis directory
            strategy_name = self.strategies[strategy_id]['type']
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = os.path.join(
                self.analysis_path, 
                f"{strategy_id}_{strategy_name}_{timestamp_str}.png"
            )
            plt.savefig(default_path)
            print(f"Performance visualization saved to {default_path}")
        
        plt.close()
    
    def find_similar_strategies(self, strategy_id: str, 
                              max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find strategies similar to the given strategy.
        
        Args:
            strategy_id: Strategy identifier
            max_results: Maximum number of similar strategies to return
            
        Returns:
            List of similar strategy data
        """
        if strategy_id not in self.strategies:
            return []
        
        target_strategy = self.strategies[strategy_id]
        target_type = target_strategy['type']
        target_params = target_strategy['parameters']
        
        # Only compare with strategies of the same type
        similar_strategies = []
        
        for sid, strategy in self.strategies.items():
            if sid == strategy_id:
                continue
                
            if strategy['type'] != target_type:
                continue
            
            # Calculate parameter similarity
            params = strategy['parameters']
            similarity_score = self._calculate_parameter_similarity(target_params, params)
            
            if similarity_score > 0.5:  # Minimum similarity threshold
                similar_strategies.append({
                    'id': sid,
                    'similarity': similarity_score,
                    'strategy': strategy
                })
        
        # Sort by similarity (descending) and limit results
        similar_strategies.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_strategies[:max_results]
    
    def _calculate_parameter_similarity(self, params1: Dict, params2: Dict) -> float:
        """Calculate similarity between two parameter sets"""
        # Get all unique parameter keys
        all_keys = set(params1.keys()).union(set(params2.keys()))
        
        if not all_keys:
            return 0.0
            
        match_count = 0
        similarity_sum = 0.0
        
        for key in all_keys:
            if key in params1 and key in params2:
                val1 = params1[key]
                val2 = params2[key]
                
                # Handle different parameter types
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalize numerical difference
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarity = 1.0  # Both zero
                    else:
                        diff = abs(val1 - val2) / max_val
                        similarity = 1.0 - min(1.0, diff)
                elif val1 == val2:
                    similarity = 1.0  # Exact match for non-numeric
                else:
                    similarity = 0.0  # No match for non-numeric
                    
                similarity_sum += similarity
                match_count += 1
        
        if match_count == 0:
            return 0.0
            
        return similarity_sum / match_count
    
    def get_evolution_insights(self) -> Dict[str, Any]:
        """
        Generate insights about the evolution of strategies.
        
        Returns:
            Dictionary of insights
        """
        if not self.strategies:
            return {"error": "No strategies in registry"}
            
        # Get strategies by generation
        strategies_by_gen = defaultdict(list)
        for sid, strategy in self.strategies.items():
            gen = strategy.get('generation', 0)
            strategies_by_gen[gen].append(strategy)
        
        # Strategy type distribution by generation
        type_distribution = {}
        for gen, strats in strategies_by_gen.items():
            type_counts = defaultdict(int)
            for s in strats:
                type_counts[s['type']] += 1
            type_distribution[gen] = dict(type_counts)
        
        # Parameter evolution for common strategy types
        common_types = set()
        for strats in strategies_by_gen.values():
            for s in strats:
                common_types.add(s['type'])
        
        parameter_evolution = {}
        for stype in common_types:
            param_by_gen = {}
            for gen in sorted(strategies_by_gen.keys()):
                gen_strats = [s for s in strategies_by_gen[gen] if s['type'] == stype]
                if not gen_strats:
                    continue
                    
                # Collect all parameter values
                param_values = defaultdict(list)
                for s in gen_strats:
                    for param, value in s.get('parameters', {}).items():
                        if isinstance(value, (int, float)):
                            param_values[param].append(value)
                
                # Calculate statistics for each parameter
                param_stats = {}
                for param, values in param_values.items():
                    if values:
                        param_stats[param] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': min(values),
                            'max': max(values)
                        }
                
                param_by_gen[gen] = param_stats
            
            parameter_evolution[stype] = param_by_gen
        
        # Performance evolution
        perf_by_gen = {}
        for gen in sorted(strategies_by_gen.keys()):
            gen_strats = strategies_by_gen[gen]
            perf_values = defaultdict(list)
            
            for s in gen_strats:
                sid = s.get('id')
                if sid in self.strategy_performance:
                    perf = self.strategy_performance[sid].get('overall', {})
                    for metric, value in perf.items():
                        if isinstance(value, (int, float)):
                            perf_values[metric].append(value)
            
            # Calculate statistics
            perf_stats = {}
            for metric, values in perf_values.items():
                if values:
                    perf_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values)
                    }
            
            perf_by_gen[gen] = perf_stats
        
        # Put together the insights
        insights = {
            'total_strategies': len(self.strategies),
            'generations': len(strategies_by_gen),
            'type_distribution': type_distribution,
            'parameter_evolution': parameter_evolution,
            'performance_evolution': perf_by_gen,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save insights
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        insights_file = os.path.join(self.analysis_path, f"evolution_insights_{timestamp_str}.json")
        with open(insights_file, 'w') as file:
            json.dump(insights, file, indent=2)
        
        return insights
    
    def plot_evolution_insights(self, save_path: Optional[str] = None) -> None:
        """
        Create visualizations of evolution insights.
        
        Args:
            save_path: Optional directory to save visualizations
        """
        insights = self.get_evolution_insights()
        
        if "error" in insights:
            print(f"Error generating insights: {insights['error']}")
            return
            
        if save_path is None:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.analysis_path, f"evolution_plots_{timestamp_str}")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Plot strategy type distribution by generation
        type_dist = insights.get('type_distribution', {})
        if type_dist:
            plt.figure(figsize=(12, 8))
            generations = sorted(type_dist.keys())
            
            bottom = np.zeros(len(generations))
            for stype in set().union(*[set(type_dist[g].keys()) for g in generations]):
                values = [type_dist[g].get(stype, 0) for g in generations]
                plt.bar(generations, values, bottom=bottom, label=stype)
                bottom += np.array(values)
            
            plt.title('Strategy Type Distribution by Generation')
            plt.xlabel('Generation')
            plt.ylabel('Count')
            plt.legend()
            
            plt.savefig(os.path.join(save_path, "type_distribution.png"))
            plt.close()
        
        # Plot parameter evolution for each strategy type
        param_evolution = insights.get('parameter_evolution', {})
        for stype, param_by_gen in param_evolution.items():
            if not param_by_gen:
                continue
                
            # Get all parameter names across generations
            all_params = set()
            for gen_stats in param_by_gen.values():
                all_params.update(gen_stats.keys())
            
            # Plot evolution of each parameter
            for param in all_params:
                plt.figure(figsize=(10, 6))
                
                generations = sorted([int(g) for g in param_by_gen.keys()])
                means = []
                stds = []
                
                for gen in generations:
                    gen_str = str(gen)
                    if gen_str in param_by_gen and param in param_by_gen[gen_str]:
                        means.append(param_by_gen[gen_str][param]['mean'])
                        stds.append(param_by_gen[gen_str][param]['std'])
                    else:
                        means.append(None)
                        stds.append(None)
                
                # Filter out None values for plotting
                valid_gens = []
                valid_means = []
                valid_stds = []
                
                for i, (gen, mean, std) in enumerate(zip(generations, means, stds)):
                    if mean is not None:
                        valid_gens.append(gen)
                        valid_means.append(mean)
                        valid_stds.append(std)
                
                if valid_means:
                    plt.errorbar(valid_gens, valid_means, yerr=valid_stds, 
                               marker='o', linestyle='-')
                    
                    plt.title(f'Evolution of {param} for {stype}')
                    plt.xlabel('Generation')
                    plt.ylabel(param)
                    
                    plt.savefig(os.path.join(save_path, f"{stype}_{param}_evolution.png"))
                
                plt.close()
        
        # Plot performance evolution
        perf_evolution = insights.get('performance_evolution', {})
        if perf_evolution:
            metrics = set()
            for gen_perf in perf_evolution.values():
                metrics.update(gen_perf.keys())
            
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                
                generations = sorted([int(g) for g in perf_evolution.keys()])
                means = []
                stds = []
                
                for gen in generations:
                    gen_str = str(gen)
                    if gen_str in perf_evolution and metric in perf_evolution[gen_str]:
                        means.append(perf_evolution[gen_str][metric]['mean'])
                        stds.append(perf_evolution[gen_str][metric]['std'])
                    else:
                        means.append(None)
                        stds.append(None)
                
                # Filter out None values
                valid_gens = []
                valid_means = []
                valid_stds = []
                
                for i, (gen, mean, std) in enumerate(zip(generations, means, stds)):
                    if mean is not None:
                        valid_gens.append(gen)
                        valid_means.append(mean)
                        valid_stds.append(std)
                
                if valid_means:
                    plt.errorbar(valid_gens, valid_means, yerr=valid_stds, 
                               marker='o', linestyle='-')
                    
                    plt.title(f'Evolution of {metric}')
                    plt.xlabel('Generation')
                    plt.ylabel(metric)
                    
                    plt.savefig(os.path.join(save_path, f"{metric}_evolution.png"))
                
                plt.close()
        
        print(f"Evolution plots saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    registry = StrategyRegistry()
    # ... register strategies and record performance
