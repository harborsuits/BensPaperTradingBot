"""
Parameter Sweep for Adaptive Strategy Backtesting

This module provides functionality for running parameter sweeps and sensitivity analysis
to identify optimal parameter configurations for the adaptive trading system.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import itertools
from tqdm import tqdm

from trading_bot.backtest.adaptive_backtest_engine import AdaptiveBacktestEngine
from trading_bot.backtest.market_data_generator import MarketDataGenerator, MarketRegimeType

logger = logging.getLogger(__name__)

class ParameterSweep:
    """
    Performs parameter sweeps and sensitivity analysis for the adaptive strategy system.
    
    This class allows testing multiple parameter combinations to identify which parameters
    have the most impact on performance and which settings are most robust across
    different market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the parameter sweep.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize engines
        self.backtest_engine = AdaptiveBacktestEngine(config=self.config.get('backtest_engine', {}))
        self.market_data_generator = MarketDataGenerator()
        
        # Output directory
        self.output_dir = self.config.get('output_dir', './parameter_sweep_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.sweep_results = {}
        self.sensitivity_analysis = {}
        
        logger.info("Initialized ParameterSweep")
    
    def run_sweep(self,
                 parameter_grid: Dict[str, List[Any]],
                 base_config: Dict[str, Any],
                 market_data: Dict[str, pd.DataFrame],
                 strategies: Dict[str, Dict[str, Any]],
                 simulation_days: int = 252,
                 name: str = "parameter_sweep") -> Dict[str, Any]:
        """
        Run a parameter sweep using the specified parameter grid.
        
        Args:
            parameter_grid: Dict mapping parameter paths to lists of values to test
                            Paths are dot-separated, e.g., 'snowball_allocator.reinvestment_ratio'
            base_config: Base configuration for the controller
            market_data: Dictionary mapping symbols to market data DataFrames
            strategies: Dictionary mapping strategy IDs to strategy metadata
            simulation_days: Number of days to simulate
            name: Name for this parameter sweep
            
        Returns:
            Dictionary with sweep results
        """
        # Convert parameter grid to list of parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # Generate all possible combinations of parameter values
        param_combinations = list(itertools.product(*param_values))
        
        # Store all results
        all_results = []
        best_config = None
        best_return = -float('inf')
        
        logger.info(f"Starting parameter sweep '{name}' with {len(param_combinations)} combinations")
        
        # Run backtests for each parameter combination
        for i, combination in enumerate(tqdm(param_combinations, desc="Running parameter sweep")):
            # Create configuration for this combination
            config = base_config.copy()
            param_dict = {}
            
            # Set parameters in config
            for param_name, param_value in zip(param_names, combination):
                # Handle nested parameters using dot notation
                parts = param_name.split('.')
                
                # Build nested dict representation
                current = param_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set leaf value
                current[parts[-1]] = param_value
            
            # Update config with parameter values
            self._update_nested_dict(config, param_dict)
            
            # Run backtest with this config
            backtest_name = f"{name}_{i+1}"
            try:
                results = self.backtest_engine.run_backtest(
                    controller_config=config,
                    market_data=market_data,
                    strategies=strategies,
                    simulation_days=simulation_days,
                    name=backtest_name
                )
                
                # Extract key metrics
                metrics = {
                    'backtest_name': backtest_name,
                    'total_return': results['total_return'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'max_drawdown': results['max_drawdown']
                }
                
                # Add parameter values to metrics
                for param_name, param_value in zip(param_names, combination):
                    metrics[param_name] = param_value
                
                all_results.append(metrics)
                
                # Check if this is the best result so far
                if results['sharpe_ratio'] > best_return:
                    best_return = results['sharpe_ratio']
                    best_config = config.copy()
                    
            except Exception as e:
                logger.error(f"Error running backtest {backtest_name}: {str(e)}")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_results)
        
        # Save results
        os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)
        results_df.to_csv(os.path.join(self.output_dir, name, 'sweep_results.csv'), index=False)
        
        if best_config:
            with open(os.path.join(self.output_dir, name, 'best_config.json'), 'w') as f:
                json.dump(best_config, f, indent=2)
        
        # Generate sensitivity analysis
        sensitivity = self._analyze_sensitivity(results_df, param_names)
        self.sensitivity_analysis[name] = sensitivity
        
        # Store results
        sweep_results = {
            'name': name,
            'results_df': results_df,
            'best_config': best_config,
            'sensitivity': sensitivity,
            'parameter_grid': parameter_grid
        }
        
        self.sweep_results[name] = sweep_results
        
        # Generate analysis visualizations
        self._generate_heatmaps(results_df, param_names, name)
        self._generate_parameter_impact_plots(results_df, param_names, name)
        
        logger.info(f"Completed parameter sweep '{name}' with {len(param_combinations)} combinations")
        logger.info(f"Best Sharpe ratio: {best_return:.2f}")
        
        return sweep_results
    
    def run_regime_robustness_test(self,
                                 config_to_test: Dict[str, Any],
                                 strategies: Dict[str, Dict[str, Any]],
                                 symbols: List[str],
                                 regimes: List[MarketRegimeType],
                                 days_per_regime: int = 252,
                                 name: str = "regime_robustness") -> Dict[str, Any]:
        """
        Test a configuration's robustness across different market regimes.
        
        Args:
            config_to_test: Configuration to test
            strategies: Dictionary mapping strategy IDs to strategy metadata
            symbols: List of symbols to test
            regimes: List of market regimes to test
            days_per_regime: Number of days to simulate per regime
            name: Name for this robustness test
            
        Returns:
            Dictionary with robustness test results
        """
        results = {}
        
        # Test each regime
        for regime in regimes:
            regime_name = regime.value
            logger.info(f"Testing {regime_name} market")
            
            # Generate data for this regime
            market_data = {}
            for symbol in symbols:
                market_data[symbol] = self.market_data_generator.generate_regime_data(
                    symbol=symbol,
                    regime=regime,
                    days=days_per_regime
                )
            
            # Run backtest for this regime
            test_name = f"{name}_{regime_name}"
            try:
                regime_results = self.backtest_engine.run_backtest(
                    controller_config=config_to_test,
                    market_data=market_data,
                    strategies=strategies,
                    simulation_days=days_per_regime,
                    name=test_name
                )
                
                results[regime_name] = {
                    'total_return': regime_results['total_return'],
                    'sharpe_ratio': regime_results['sharpe_ratio'],
                    'max_drawdown': regime_results['max_drawdown'],
                    'equity_curve': regime_results['equity_curve']
                }
            except Exception as e:
                logger.error(f"Error running {regime_name} market test: {str(e)}")
                results[regime_name] = {
                    'error': str(e)
                }
        
        # Calculate overall robustness score
        robustness_scores = {}
        
        if results:
            # Sharpe ratio across regimes
            sharpe_values = [r.get('sharpe_ratio', 0) for r in results.values() if 'sharpe_ratio' in r]
            
            if sharpe_values:
                # Minimum Sharpe across regimes
                min_sharpe = min(sharpe_values)
                # Average Sharpe across regimes
                avg_sharpe = sum(sharpe_values) / len(sharpe_values)
                # Sharpe consistency (std dev of Sharpe ratios)
                sharpe_std = np.std(sharpe_values)
                # Robustness score: avg_sharpe * (1 - sharpe_std/avg_sharpe) * (min_sharpe/avg_sharpe)
                # This rewards high average Sharpe, low variability, and high minimum Sharpe
                if avg_sharpe > 0:
                    robustness_score = avg_sharpe * (1 - min(1, sharpe_std/avg_sharpe)) * (min_sharpe/avg_sharpe)
                else:
                    robustness_score = 0
                
                robustness_scores = {
                    'min_sharpe': min_sharpe,
                    'avg_sharpe': avg_sharpe,
                    'sharpe_std': sharpe_std,
                    'robustness_score': robustness_score
                }
        
        # Save results
        os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)
        
        # Save summary
        with open(os.path.join(self.output_dir, name, 'robustness_results.json'), 'w') as f:
            # Convert numpy values to float for JSON serialization
            json_results = {}
            for regime, regime_results in results.items():
                json_results[regime] = {k: float(v) if isinstance(v, np.floating) else v 
                                       for k, v in regime_results.items() if k != 'equity_curve'}
            
            json.dump({
                'results': json_results,
                'robustness_scores': {k: float(v) if isinstance(v, np.floating) else v for k, v in robustness_scores.items()}
            }, f, indent=2)
        
        # Generate visualization
        self._plot_regime_comparison(results, name)
        
        logger.info(f"Completed regime robustness test '{name}'")
        if robustness_scores:
            logger.info(f"Robustness score: {robustness_scores['robustness_score']:.2f}")
        
        return {
            'name': name,
            'results': results,
            'robustness_scores': robustness_scores
        }
    
    def _update_nested_dict(self, 
                          target_dict: Dict[str, Any], 
                          source_dict: Dict[str, Any]) -> None:
        """Update nested dictionary with values from source dict"""
        for key, value in source_dict.items():
            if isinstance(value, dict):
                # If key doesn't exist in target, create it
                if key not in target_dict:
                    target_dict[key] = {}
                # Recursively update nested dict
                self._update_nested_dict(target_dict[key], value)
            else:
                # Set value directly
                target_dict[key] = value
    
    def _analyze_sensitivity(self, 
                           results_df: pd.DataFrame, 
                           param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze parameter sensitivity"""
        sensitivity = {}
        
        # Metrics to analyze
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        
        for param in param_names:
            param_sensitivity = {}
            
            # Check if parameter has multiple values
            if len(results_df[param].unique()) <= 1:
                continue
                
            for metric in metrics:
                # Group by parameter and calculate average metric
                param_impact = results_df.groupby(param)[metric].agg(['mean', 'std', 'min', 'max'])
                
                # Normalize values for comparison
                if param_impact['mean'].max() != param_impact['mean'].min():
                    normalized_impact = (param_impact['mean'] - param_impact['mean'].min()) / (param_impact['mean'].max() - param_impact['mean'].min())
                    
                    # Calculate sensitivity score:
                    # Higher score = parameter has more impact on this metric
                    sensitivity_score = normalized_impact.max() - normalized_impact.min()
                else:
                    sensitivity_score = 0.0
                
                param_sensitivity[metric] = sensitivity_score
            
            sensitivity[param] = param_sensitivity
        
        return sensitivity
    
    def _generate_heatmaps(self, 
                         results_df: pd.DataFrame, 
                         param_names: List[str],
                         sweep_name: str) -> None:
        """Generate heatmaps for parameter combinations"""
        # Only proceed if we have at least 2 parameters
        if len(param_names) < 2:
            return
        
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, sweep_name, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate heatmaps for each pair of parameters
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown']:
            # Limit to parameter pairs with reasonable numbers of unique values
            valid_params = [p for p in param_names if 2 <= len(results_df[p].unique()) <= 20]
            
            for i, param1 in enumerate(valid_params):
                for param2 in valid_params[i+1:]:
                    # Create pivot table
                    try:
                        pivot = results_df.pivot_table(
                            index=param1, 
                            columns=param2, 
                            values=metric,
                            aggfunc='mean'
                        )
                        
                        # Plot heatmap
                        plt.figure(figsize=(10, 8))
                        
                        # Adjust colormap based on metric
                        if metric == 'max_drawdown':
                            # Lower drawdown is better
                            cmap = 'RdYlGn_r'
                        else:
                            # Higher values are better
                            cmap = 'RdYlGn'
                            
                        sns.heatmap(pivot, annot=True, cmap=cmap, fmt='.2f')
                        plt.title(f'{metric.replace("_", " ").title()} by {param1} and {param2}')
                        plt.tight_layout()
                        
                        # Save figure
                        plt.savefig(os.path.join(plots_dir, f'heatmap_{metric}_{param1}_{param2}.png'))
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Error generating heatmap for {param1} and {param2}: {str(e)}")
    
    def _generate_parameter_impact_plots(self, 
                                      results_df: pd.DataFrame, 
                                      param_names: List[str],
                                      sweep_name: str) -> None:
        """Generate plots showing parameter impact on performance"""
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, sweep_name, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Metrics to plot
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        
        # For each parameter, plot its impact on metrics
        for param in param_names:
            # Skip if parameter has only one value
            if len(results_df[param].unique()) <= 1:
                continue
                
            plt.figure(figsize=(12, 8))
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(len(metrics), 1, i)
                
                # Group by parameter and calculate mean and std for the metric
                grouped = results_df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
                
                # Plot mean with error bars
                plt.errorbar(
                    grouped[param],
                    grouped['mean'],
                    yerr=grouped['std'],
                    marker='o',
                    linestyle='-',
                    capsize=5
                )
                
                plt.title(f'{metric.replace("_", " ").title()} vs {param}')
                plt.grid(True)
                
                # Format y-axis for percentage metrics
                if metric in ['total_return', 'max_drawdown']:
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'impact_{param}.png'))
            plt.close()
        
        # Generate sensitivity comparison bar chart
        if self.sensitivity_analysis and sweep_name in self.sensitivity_analysis:
            sensitivity = self.sensitivity_analysis[sweep_name]
            
            # Extract data for bar chart
            params = []
            sharpe_sensitivity = []
            return_sensitivity = []
            drawdown_sensitivity = []
            
            for param, metrics in sensitivity.items():
                params.append(param)
                sharpe_sensitivity.append(metrics.get('sharpe_ratio', 0))
                return_sensitivity.append(metrics.get('total_return', 0))
                drawdown_sensitivity.append(metrics.get('max_drawdown', 0))
            
            # Sort by sharpe ratio sensitivity
            sort_idx = np.argsort(sharpe_sensitivity)[::-1]  # Descending
            params = [params[i] for i in sort_idx]
            sharpe_sensitivity = [sharpe_sensitivity[i] for i in sort_idx]
            return_sensitivity = [return_sensitivity[i] for i in sort_idx]
            drawdown_sensitivity = [drawdown_sensitivity[i] for i in sort_idx]
            
            # Plot
            plt.figure(figsize=(12, 8))
            x = np.arange(len(params))
            width = 0.25
            
            plt.bar(x - width, sharpe_sensitivity, width, label='Sharpe Ratio')
            plt.bar(x, return_sensitivity, width, label='Total Return')
            plt.bar(x + width, drawdown_sensitivity, width, label='Max Drawdown')
            
            plt.xlabel('Parameter')
            plt.ylabel('Sensitivity Score')
            plt.title('Parameter Sensitivity Comparison')
            plt.xticks(x, params, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_dir, 'sensitivity_comparison.png'))
            plt.close()
    
    def _plot_regime_comparison(self, 
                             regime_results: Dict[str, Dict[str, Any]],
                             test_name: str) -> None:
        """Plot performance comparison across different market regimes"""
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, test_name, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract metrics for each regime
        regimes = []
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for regime, results in regime_results.items():
            if 'total_return' in results:
                regimes.append(regime)
                returns.append(results['total_return'])
                sharpe_ratios.append(results['sharpe_ratio'])
                max_drawdowns.append(results['max_drawdown'])
        
        # Plot metrics comparison
        plt.figure(figsize=(14, 10))
        
        # Plot returns
        plt.subplot(3, 1, 1)
        bars = plt.bar(regimes, returns)
        
        # Color bars based on return (green for positive, red for negative)
        for i, bar in enumerate(bars):
            bar.set_color('green' if returns[i] >= 0 else 'red')
            
        plt.title('Total Return by Market Regime')
        plt.ylabel('Return')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot Sharpe ratios
        plt.subplot(3, 1, 2)
        bars = plt.bar(regimes, sharpe_ratios)
        
        # Color bars based on Sharpe (green for positive, red for negative)
        for i, bar in enumerate(bars):
            bar.set_color('green' if sharpe_ratios[i] >= 0 else 'red')
            
        plt.title('Sharpe Ratio by Market Regime')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        # Plot max drawdowns
        plt.subplot(3, 1, 3)
        bars = plt.bar(regimes, max_drawdowns)
        
        # Color bars based on drawdown (lighter red is better)
        for i, bar in enumerate(bars):
            # Scale color intensity by drawdown magnitude
            intensity = min(1.0, max_drawdowns[i] / 0.5)  # Scale to max of 50% drawdown
            bar.set_color((1.0, 0.4 * (1 - intensity), 0.4 * (1 - intensity)))
            
        plt.title('Maximum Drawdown by Market Regime')
        plt.ylabel('Drawdown')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'regime_comparison.png'))
        plt.close()
        
        # Plot equity curves if available
        plt.figure(figsize=(12, 8))
        
        for regime, results in regime_results.items():
            if 'equity_curve' in results:
                # Normalize to percentage of initial equity
                equity = results['equity_curve']
                initial = equity[0]
                normalized = [(e / initial - 1) * 100 for e in equity]
                plt.plot(normalized, label=regime)
        
        plt.title('Equity Curves by Market Regime')
        plt.xlabel('Trading Day')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'equity_curves.png'))
        plt.close()
