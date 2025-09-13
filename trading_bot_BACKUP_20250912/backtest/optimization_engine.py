"""
Hyperparameter Optimization for Adaptive Strategy System

Uses Optuna for Bayesian optimization to find optimal hyperparameters
for the adaptive trading system across different market regimes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

from trading_bot.backtest.adaptive_backtest_engine import AdaptiveBacktestEngine
from trading_bot.backtest.market_data_generator import MarketDataGenerator, MarketRegimeType

logger = logging.getLogger(__name__)

class OptimizationEngine:
    """
    Performs hyperparameter optimization for the adaptive strategy system.
    
    Uses Optuna to efficiently search the parameter space and find robust
    parameter combinations that perform well across different market regimes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the optimization engine.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize engines
        self.backtest_engine = AdaptiveBacktestEngine(config=self.config.get('backtest_engine', {}))
        self.market_data_generator = MarketDataGenerator()
        
        # Output directory
        self.output_dir = self.config.get('output_dir', './optimization_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default optimization parameters
        self.n_trials = self.config.get('n_trials', 100)
        self.timeout = self.config.get('timeout', 7200)  # 2 hours default timeout
        
        # Results storage
        self.studies = {}
        self.best_parameters = {}
        
        logger.info("Initialized OptimizationEngine")
    
    def optimize(self,
               parameter_ranges: Dict[str, Tuple[float, float]],
               base_config: Dict[str, Any],
               market_data: Dict[str, pd.DataFrame],
               strategies: Dict[str, Dict[str, Any]],
               simulation_days: int = 252,
               n_trials: Optional[int] = None,
               timeout: Optional[int] = None,
               name: str = "optimization") -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna.
        
        Args:
            parameter_ranges: Dict mapping parameter paths to (min, max) tuples
                              Paths are dot-separated, e.g., 'snowball_allocator.reinvestment_ratio'
            base_config: Base configuration for the controller
            market_data: Dictionary mapping symbols to market data DataFrames
            strategies: Dictionary mapping strategy IDs to strategy metadata
            simulation_days: Number of days to simulate
            n_trials: Number of optimization trials (default from config)
            timeout: Timeout in seconds (default from config)
            name: Name for this optimization run
            
        Returns:
            Dictionary with optimization results
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        # Create study directory
        study_dir = os.path.join(self.output_dir, name)
        os.makedirs(study_dir, exist_ok=True)
        
        # Define objective function for Optuna
        def objective(trial):
            # Create configuration for this trial
            config = base_config.copy()
            param_dict = {}
            
            # Sample parameters
            for param_name, (param_min, param_max) in parameter_ranges.items():
                # Handle different parameter types based on range values
                if isinstance(param_min, int) and isinstance(param_max, int):
                    param_value = trial.suggest_int(param_name, param_min, param_max)
                elif param_min == 0 and param_max == 1 and isinstance(param_min, int) and isinstance(param_max, int):
                    # Categorical boolean parameter
                    param_value = trial.suggest_categorical(param_name, [False, True])
                elif param_min == 0 and param_max == 1:
                    # Continuous parameter between 0 and 1
                    param_value = trial.suggest_float(param_name, param_min, param_max)
                else:
                    # Default to float
                    param_value = trial.suggest_float(param_name, param_min, param_max)
                
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
            backtest_name = f"{name}_trial_{trial.number}"
            try:
                results = self.backtest_engine.run_backtest(
                    controller_config=config,
                    market_data=market_data,
                    strategies=strategies,
                    simulation_days=simulation_days,
                    name=backtest_name
                )
                
                # Calculate objective value
                # We use a robust objective that balances return and risk
                # Sharpe ratio with penalty for large drawdowns
                sharpe = results['sharpe_ratio']
                drawdown = results['max_drawdown']
                
                # Penalize large drawdowns more heavily
                # This creates a more conservative optimization
                drawdown_penalty = 0
                if drawdown > 0.15:  # 15% drawdown threshold
                    drawdown_penalty = (drawdown - 0.15) * 10
                
                # Final objective: sharpe ratio with drawdown penalty
                objective_value = sharpe - drawdown_penalty
                
                # Store trial results for analysis
                trial.set_user_attr('total_return', float(results['total_return']))
                trial.set_user_attr('sharpe_ratio', float(sharpe))
                trial.set_user_attr('max_drawdown', float(drawdown))
                
                return objective_value
                
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {str(e)}")
                # Return very low value to indicate failure
                return -100.0
        
        # Create Optuna study
        study_name = f"adaptive_strategy_{name}"
        storage_name = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
        
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
        except:
            # If failed to load, create new study
            study = optuna.create_study(
                study_name=study_name,
                storage=None,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
        
        # Run optimization
        logger.info(f"Starting optimization '{name}' with {n_trials} trials (timeout: {timeout}s)")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        
        # Convert best parameters to nested config structure
        best_config = base_config.copy()
        best_param_dict = {}
        
        for param_name, param_value in best_params.items():
            # Handle nested parameters
            parts = param_name.split('.')
            
            # Build nested dict
            current = best_param_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set leaf value
            current[parts[-1]] = param_value
        
        # Update config with best parameters
        self._update_nested_dict(best_config, best_param_dict)
        
        # Save best parameters
        with open(os.path.join(study_dir, 'best_parameters.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save best config
        with open(os.path.join(study_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Save study statistics
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(study_dir, 'trials.csv'), index=False)
        
        # Generate optimization visualizations
        try:
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(study_dir, 'optimization_history.png'))
            
            # Parameter importances
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(study_dir, 'param_importances.png'))
            
            # Parameter slice plots for each parameter
            for param_name in parameter_ranges.keys():
                try:
                    fig = plot_slice(study, params=[param_name])
                    fig.write_image(os.path.join(study_dir, f'slice_{param_name.replace(".", "_")}.png'))
                except:
                    logger.warning(f"Could not generate slice plot for {param_name}")
        except Exception as e:
            logger.warning(f"Error generating optimization visualizations: {str(e)}")
        
        # Store study and results
        self.studies[name] = study
        self.best_parameters[name] = best_params
        
        # Log results
        logger.info(f"Completed optimization '{name}'")
        logger.info(f"Best objective value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Extract metrics from best trial
        best_metrics = {
            'total_return': best_trial.user_attrs.get('total_return', 0),
            'sharpe_ratio': best_trial.user_attrs.get('sharpe_ratio', 0),
            'max_drawdown': best_trial.user_attrs.get('max_drawdown', 0)
        }
        
        return {
            'name': name,
            'best_value': best_value,
            'best_params': best_params,
            'best_config': best_config,
            'best_metrics': best_metrics,
            'study': study
        }
    
    def cross_validate_optimization(self,
                                  parameter_ranges: Dict[str, Tuple[float, float]],
                                  base_config: Dict[str, Any],
                                  strategies: Dict[str, Dict[str, Any]],
                                  regimes: List[MarketRegimeType],
                                  symbols: List[str],
                                  days_per_regime: int = 252,
                                  n_trials: Optional[int] = None,
                                  name: str = "cross_validation") -> Dict[str, Any]:
        """
        Run cross-validation optimization across multiple market regimes.
        
        Args:
            parameter_ranges: Dict mapping parameter paths to (min, max) tuples
            base_config: Base configuration for the controller
            strategies: Dictionary mapping strategy IDs to strategy metadata
            regimes: List of market regimes to cross-validate against
            symbols: List of symbols to use
            days_per_regime: Number of days to simulate per regime
            n_trials: Number of optimization trials
            name: Name for this optimization run
            
        Returns:
            Dictionary with cross-validation results
        """
        n_trials = n_trials or self.n_trials
        
        # Create directory for this cross-validation
        cv_dir = os.path.join(self.output_dir, name)
        os.makedirs(cv_dir, exist_ok=True)
        
        # Run optimization for each regime
        regime_results = {}
        regime_best_params = {}
        
        for regime in regimes:
            regime_name = regime.value
            logger.info(f"Running optimization for {regime_name} market")
            
            # Generate market data for this regime
            market_data = {}
            for symbol in symbols:
                market_data[symbol] = self.market_data_generator.generate_regime_data(
                    symbol=symbol,
                    regime=regime,
                    days=days_per_regime
                )
            
            # Run optimization for this regime
            try:
                results = self.optimize(
                    parameter_ranges=parameter_ranges,
                    base_config=base_config,
                    market_data=market_data,
                    strategies=strategies,
                    simulation_days=days_per_regime,
                    n_trials=n_trials // len(regimes),  # Divide trials among regimes
                    name=f"{name}_{regime_name}"
                )
                
                # Store results
                regime_results[regime_name] = results
                regime_best_params[regime_name] = results['best_params']
                
            except Exception as e:
                logger.error(f"Error in {regime_name} optimization: {str(e)}")
        
        # Find robust parameters across regimes
        # We'll use an ensemble approach that averages the parameters
        if not regime_best_params:
            logger.error("No successful regime optimizations")
            return {
                'name': name,
                'error': "No successful regime optimizations"
            }
        
        # Collect parameters from all regimes
        all_params = {}
        for regime, params in regime_best_params.items():
            for param_name, value in params.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(value)
        
        # Average parameters across regimes
        robust_params = {}
        for param_name, values in all_params.items():
            # For boolean parameters, use majority vote
            if all(isinstance(v, bool) for v in values):
                robust_params[param_name] = sum(values) > len(values) / 2
            else:
                # For numeric parameters, use median for robustness against outliers
                robust_params[param_name] = float(np.median(values))
        
        # Create robust config
        robust_config = base_config.copy()
        robust_param_dict = {}
        
        for param_name, param_value in robust_params.items():
            # Handle nested parameters
            parts = param_name.split('.')
            
            # Build nested dict
            current = robust_param_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set leaf value
            current[parts[-1]] = param_value
        
        # Update config with robust parameters
        self._update_nested_dict(robust_config, robust_param_dict)
        
        # Save robust parameters
        with open(os.path.join(cv_dir, 'robust_parameters.json'), 'w') as f:
            json.dump(robust_params, f, indent=2)
        
        # Save robust config
        with open(os.path.join(cv_dir, 'robust_config.json'), 'w') as f:
            json.dump(robust_config, f, indent=2)
        
        # Generate parameter comparison visualization
        self._plot_parameter_comparison(regime_best_params, robust_params, name)
        
        # Test robust parameters across all regimes
        test_results = {}
        
        for regime in regimes:
            regime_name = regime.value
            logger.info(f"Testing robust parameters on {regime_name} market")
            
            # Generate market data for this regime
            market_data = {}
            for symbol in symbols:
                market_data[symbol] = self.market_data_generator.generate_regime_data(
                    symbol=symbol,
                    regime=regime,
                    days=days_per_regime
                )
            
            # Run backtest with robust parameters
            try:
                results = self.backtest_engine.run_backtest(
                    controller_config=robust_config,
                    market_data=market_data,
                    strategies=strategies,
                    simulation_days=days_per_regime,
                    name=f"{name}_robust_{regime_name}"
                )
                
                # Store results
                test_results[regime_name] = {
                    'total_return': results['total_return'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'max_drawdown': results['max_drawdown']
                }
                
            except Exception as e:
                logger.error(f"Error testing robust parameters on {regime_name}: {str(e)}")
                test_results[regime_name] = {'error': str(e)}
        
        # Save test results
        with open(os.path.join(cv_dir, 'robust_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Calculate robustness metrics
        robustness_metrics = {}
        
        sharpe_values = [r.get('sharpe_ratio', 0) for r in test_results.values() if 'sharpe_ratio' in r]
        if sharpe_values:
            robustness_metrics['avg_sharpe'] = float(np.mean(sharpe_values))
            robustness_metrics['min_sharpe'] = float(np.min(sharpe_values))
            robustness_metrics['sharpe_std'] = float(np.std(sharpe_values))
        
        returns = [r.get('total_return', 0) for r in test_results.values() if 'total_return' in r]
        if returns:
            robustness_metrics['avg_return'] = float(np.mean(returns))
            robustness_metrics['min_return'] = float(np.min(returns))
            robustness_metrics['return_std'] = float(np.std(returns))
        
        drawdowns = [r.get('max_drawdown', 0) for r in test_results.values() if 'max_drawdown' in r]
        if drawdowns:
            robustness_metrics['avg_drawdown'] = float(np.mean(drawdowns))
            robustness_metrics['max_drawdown'] = float(np.max(drawdowns))
            robustness_metrics['drawdown_std'] = float(np.std(drawdowns))
        
        # Calculate overall robustness score
        if 'avg_sharpe' in robustness_metrics and robustness_metrics['avg_sharpe'] > 0:
            sharpe_consistency = 1 - min(1, robustness_metrics['sharpe_std'] / robustness_metrics['avg_sharpe'])
            min_ratio = robustness_metrics['min_sharpe'] / robustness_metrics['avg_sharpe']
            robustness_metrics['robustness_score'] = robustness_metrics['avg_sharpe'] * sharpe_consistency * min_ratio
        else:
            robustness_metrics['robustness_score'] = 0
        
        # Save robustness metrics
        with open(os.path.join(cv_dir, 'robustness_metrics.json'), 'w') as f:
            json.dump(robustness_metrics, f, indent=2)
        
        # Return results
        return {
            'name': name,
            'regime_results': regime_results,
            'regime_best_params': regime_best_params,
            'robust_params': robust_params,
            'robust_config': robust_config,
            'test_results': test_results,
            'robustness_metrics': robustness_metrics
        }
    
    def _update_nested_dict(self, target_dict: Dict[str, Any], source_dict: Dict[str, Any]) -> None:
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
    
    def _plot_parameter_comparison(self, 
                                 regime_params: Dict[str, Dict[str, Any]],
                                 robust_params: Dict[str, Any],
                                 name: str) -> None:
        """Plot parameter comparison across different regimes"""
        if not regime_params:
            return
            
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, name, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Collect all parameter names
        all_params = set()
        for params in regime_params.values():
            all_params.update(params.keys())
        
        # Filter out non-numeric parameters
        numeric_params = []
        for param in all_params:
            is_numeric = True
            for regime, params in regime_params.items():
                if param in params and not isinstance(params[param], (int, float)):
                    is_numeric = False
                    break
            if is_numeric:
                numeric_params.append(param)
        
        # Create comparison plot for numeric parameters
        if numeric_params:
            # Determine how many parameters to plot per figure
            params_per_fig = 5
            num_figs = (len(numeric_params) + params_per_fig - 1) // params_per_fig
            
            for fig_idx in range(num_figs):
                start_idx = fig_idx * params_per_fig
                end_idx = min(start_idx + params_per_fig, len(numeric_params))
                fig_params = numeric_params[start_idx:end_idx]
                
                plt.figure(figsize=(12, 8))
                
                for i, param in enumerate(fig_params):
                    plt.subplot(len(fig_params), 1, i+1)
                    
                    # Get parameter values for each regime
                    regimes = []
                    values = []
                    
                    for regime, params in regime_params.items():
                        if param in params:
                            regimes.append(regime)
                            values.append(params[param])
                    
                    # Plot parameter values
                    bars = plt.bar(regimes, values)
                    
                    # Add robust value as a horizontal line
                    if param in robust_params:
                        plt.axhline(y=robust_params[param], color='r', linestyle='-', label='Robust Value')
                        
                    plt.title(f'Parameter: {param}')
                    plt.ylabel('Value')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend()
                    plt.grid(axis='y')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'param_comparison_{fig_idx+1}.png'))
                plt.close()
