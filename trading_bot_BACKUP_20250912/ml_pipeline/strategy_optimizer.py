"""
Strategy Optimizer

This module provides advanced strategy optimization capabilities
adapted from Freqtrade to find optimal parameters for trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import itertools
import json
import os
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Advanced strategy optimizer
    
    Finds optimal parameters for trading strategies using
    various optimization methods including grid search,
    random search, and Bayesian optimization.
    """
    
    def __init__(self, config=None):
        """
        Initialize the strategy optimizer
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        
        # Optimization parameters
        self.results_dir = self.config.get('results_dir', 'optimization_results')
        self.max_workers = self.config.get('max_workers', multiprocessing.cpu_count())
        self.optimization_method = self.config.get('optimization_method', 'grid')
        self.n_trials = self.config.get('n_trials', 100)
        self.random_state = self.config.get('random_state', 42)
        
        # Test duration, defaults to 6 months
        self.days = self.config.get('days', 180)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Store optimization results
        self.optimization_results = []
        
        logger.info(f"Strategy Optimizer initialized using {self.optimization_method} method")
    
    def optimize(self, 
                 strategy_class, 
                 param_space: Dict[str, Union[List, Tuple]], 
                 historical_data: Dict[str, pd.DataFrame],
                 metric: str = 'total_profit',
                 metric_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        
        Args:
            strategy_class: Class of the strategy to optimize
            param_space: Dictionary of parameter names and possible values
            historical_data: Dictionary of symbol -> DataFrame with historical data
            metric: Metric to optimize ('total_profit', 'sortino', 'sharpe', etc.)
            metric_function: Optional custom function to calculate metric
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting {self.optimization_method} optimization for {strategy_class.__name__}")
        
        start_time = datetime.now()
        np.random.seed(self.random_state)
        
        # Select optimization method
        if self.optimization_method == 'grid':
            results = self._grid_search(strategy_class, param_space, historical_data, metric, metric_function)
        elif self.optimization_method == 'random':
            results = self._random_search(strategy_class, param_space, historical_data, metric, metric_function)
        elif self.optimization_method == 'bayesian':
            # We'll add Bayesian optimization later if needed
            logger.warning("Bayesian optimization not yet implemented, falling back to random search")
            results = self._random_search(strategy_class, param_space, historical_data, metric, metric_function)
        else:
            logger.error(f"Unknown optimization method: {self.optimization_method}")
            results = {"error": f"Unknown optimization method: {self.optimization_method}"}
            return results
        
        # Calculate optimization time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Prepare final results
        final_results = {
            'strategy': strategy_class.__name__,
            'optimization_method': self.optimization_method,
            'best_params': results['best_params'],
            'best_metrics': results['best_metrics'],
            'total_combinations': results['total_combinations'],
            'combinations_tested': results['combinations_tested'],
            'elapsed_time': elapsed_time,
            'timestamp': end_time.isoformat()
        }
        
        # Store full results
        self.optimization_results.append(final_results)
        
        # Save results to disk
        self._save_results(final_results)
        
        logger.info(f"Optimization completed in {elapsed_time:.1f} seconds")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best {metric}: {results['best_metrics'].get(metric, 'N/A')}")
        
        return final_results
    
    def _grid_search(self, 
                    strategy_class, 
                    param_space: Dict[str, Union[List, Tuple]], 
                    historical_data: Dict[str, pd.DataFrame],
                    metric: str,
                    metric_function: Optional[Callable]) -> Dict[str, Any]:
        """
        Perform grid search optimization
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space to search
            historical_data: Historical price data
            metric: Metric to optimize
            metric_function: Custom metric function
            
        Returns:
            Dict with optimization results
        """
        # Create parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        combinations = list(itertools.product(*param_values))
        total_combinations = len(combinations)
        
        logger.info(f"Grid search with {total_combinations} parameter combinations")
        
        # Prepare for parallel processing
        combinations_to_test = []
        for combination in combinations:
            params = dict(zip(param_names, combination))
            combinations_to_test.append((strategy_class, params, historical_data, metric, metric_function))
        
        # Run evaluations in parallel
        results = self._run_parallel_evaluations(combinations_to_test)
        
        # Find best parameters
        best_result = self._find_best_result(results, metric)
        
        return {
            'best_params': best_result['params'],
            'best_metrics': best_result['metrics'],
            'all_results': results,
            'total_combinations': total_combinations,
            'combinations_tested': len(results)
        }
    
    def _random_search(self, 
                      strategy_class, 
                      param_space: Dict[str, Union[List, Tuple]], 
                      historical_data: Dict[str, pd.DataFrame],
                      metric: str,
                      metric_function: Optional[Callable]) -> Dict[str, Any]:
        """
        Perform random search optimization
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space to search
            historical_data: Historical price data
            metric: Metric to optimize
            metric_function: Custom metric function
            
        Returns:
            Dict with optimization results
        """
        # Get total possible combinations
        param_values = list(param_space.values())
        total_combinations = np.prod([len(values) for values in param_values])
        
        # Number of trials is capped by total combinations
        trials = min(self.n_trials, total_combinations)
        
        logger.info(f"Random search with {trials} trials out of {total_combinations} possible combinations")
        
        # Generate random parameter combinations
        param_names = list(param_space.keys())
        combinations_to_test = []
        
        for _ in range(trials):
            # Generate a random combination
            random_params = {}
            for name, values in param_space.items():
                random_params[name] = np.random.choice(values)
            
            combinations_to_test.append((strategy_class, random_params, historical_data, metric, metric_function))
        
        # Run evaluations in parallel
        results = self._run_parallel_evaluations(combinations_to_test)
        
        # Find best parameters
        best_result = self._find_best_result(results, metric)
        
        return {
            'best_params': best_result['params'],
            'best_metrics': best_result['metrics'],
            'all_results': results,
            'total_combinations': total_combinations,
            'combinations_tested': len(results)
        }
    
    def _run_parallel_evaluations(self, combinations_to_test: List[Tuple]) -> List[Dict[str, Any]]:
        """
        Run strategy evaluations in parallel
        
        Args:
            combinations_to_test: List of (strategy_class, params, data, metric, metric_fn) tuples
            
        Returns:
            List of results
        """
        results = []
        
        # Show progress in batches
        total_combinations = len(combinations_to_test)
        progress_step = max(1, total_combinations // 10)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._evaluate_strategy, *combo): combo[1]
                for combo in combinations_to_test
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append({
                        'params': params,
                        'metrics': result
                    })
                    
                    # Log progress
                    completed += 1
                    if completed % progress_step == 0:
                        logger.info(f"Progress: {completed}/{total_combinations} combinations evaluated")
                        
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
        
        return results
    
    def _evaluate_strategy(self, 
                          strategy_class, 
                          params: Dict[str, Any], 
                          historical_data: Dict[str, pd.DataFrame],
                          metric: str,
                          metric_function: Optional[Callable]) -> Dict[str, float]:
        """
        Evaluate a strategy with specific parameters
        
        Args:
            strategy_class: Strategy class to evaluate
            params: Parameters for the strategy
            historical_data: Historical price data
            metric: Primary metric to track
            metric_function: Custom metric function
            
        Returns:
            Dict with evaluation metrics
        """
        try:
            # Initialize strategy with parameters
            strategy = strategy_class(parameters=params)
            
            # Calculate metrics for each symbol
            symbol_results = []
            for symbol, data in historical_data.items():
                # Skip if data is empty
                if data.empty:
                    continue
                
                # Generate signals
                signals = []
                for i in range(len(data) - 1):  # Leave last row for position exit
                    df_subset = data.iloc[:i+1].copy()
                    try:
                        signal = strategy.generate_signals(df_subset)
                        if signal:
                            signal['timestamp'] = data.index[i]
                            signal['symbol'] = symbol
                            signals.append(signal)
                    except Exception as e:
                        logger.debug(f"Error generating signal for {symbol} at index {i}: {e}")
                
                # If no signals generated, skip this symbol
                if not signals:
                    continue
                
                # Convert signals to dataframe
                signals_df = pd.DataFrame(signals)
                
                # Calculate performance metrics
                symbol_metrics = self._calculate_metrics(signals_df, data, symbol)
                symbol_results.append(symbol_metrics)
            
            # Calculate aggregate metrics
            if not symbol_results:
                return {"error": "No signals generated for any symbol"}
            
            # Combine results across symbols
            combined_metrics = self._combine_symbol_metrics(symbol_results)
            
            # Use custom metric function if provided
            if metric_function:
                try:
                    custom_metric_value = metric_function(symbol_results, combined_metrics)
                    combined_metrics['custom_metric'] = custom_metric_value
                except Exception as e:
                    logger.error(f"Error calculating custom metric: {e}")
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            return {"error": str(e)}
    
    def _calculate_metrics(self, signals_df: pd.DataFrame, price_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Calculate performance metrics for a strategy
        
        Args:
            signals_df: DataFrame with trading signals
            price_data: DataFrame with price data
            symbol: Symbol being analyzed
            
        Returns:
            Dict with performance metrics
        """
        # If no signals, return empty metrics
        if signals_df.empty:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0
            }
        
        # Extract actions and timestamps
        actions = signals_df['action'].tolist()
        timestamps = signals_df['timestamp'].tolist()
        
        # Track virtual portfolio
        portfolio_value = 1000.0  # Start with $1000
        initial_value = portfolio_value
        position = 0  # 0 = no position, 1 = long, -1 = short
        trades = []
        equity_curve = []
        
        # Track all trades
        for i, (action, timestamp) in enumerate(zip(actions, timestamps)):
            time_idx = price_data.index.get_indexer([timestamp], method='nearest')[0]
            
            # Skip if index is out of bounds
            if time_idx >= len(price_data) - 1:
                continue
                
            price = price_data.iloc[time_idx]['close']
            next_price = price_data.iloc[time_idx + 1]['close']
            
            # Process action
            if action == 'buy' and position <= 0:
                # Enter long position
                position = 1
                entry_price = price
                entry_time = timestamp
            elif action == 'sell' and position >= 0:
                # Enter short position
                position = -1
                entry_price = price
                entry_time = timestamp
            elif action == 'exit' and position != 0:
                # Exit position
                exit_price = price
                pnl = 0.0
                
                if position == 1:  # Long position
                    pnl = (exit_price / entry_price - 1) * 100
                    position = 0
                elif position == -1:  # Short position
                    pnl = (1 - exit_price / entry_price) * 100
                    position = 0
                
                # Update portfolio value
                portfolio_value *= (1 + pnl/100)
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': 'long' if position == 1 else 'short',
                    'pnl_pct': pnl,
                    'portfolio_value': portfolio_value
                })
            
            # Record equity point
            equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value
            })
        
        # Close any open position at the end
        if position != 0:
            last_price = price_data.iloc[-1]['close']
            exit_time = price_data.index[-1]
            
            pnl = 0.0
            if position == 1:  # Long position
                pnl = (last_price / entry_price - 1) * 100
            elif position == -1:  # Short position
                pnl = (1 - last_price / entry_price) * 100
            
            # Update portfolio value
            portfolio_value *= (1 + pnl/100)
            
            # Record trade
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': last_price,
                'position': 'long' if position == 1 else 'short',
                'pnl_pct': pnl,
                'portfolio_value': portfolio_value
            })
            
            # Record final equity point
            equity_curve.append({
                'timestamp': exit_time,
                'portfolio_value': portfolio_value
            })
        
        # Calculate metrics
        metrics = {
            'symbol': symbol,
            'total_trades': len(trades),
            'total_profit': portfolio_value / initial_value - 1,
            'ending_portfolio_value': portfolio_value
        }
        
        # Win rate
        if trades:
            profitable_trades = sum(1 for trade in trades if trade['pnl_pct'] > 0)
            metrics['win_rate'] = profitable_trades / len(trades)
            
            # Average profit per trade
            metrics['avg_profit_per_trade'] = np.mean([trade['pnl_pct'] for trade in trades])
            
            # Average holding period
            holding_periods = [(trade['exit_time'] - trade['entry_time']).total_seconds() / 3600 for trade in trades]
            metrics['avg_holding_period_hours'] = np.mean(holding_periods)
        else:
            metrics['win_rate'] = 0.0
            metrics['avg_profit_per_trade'] = 0.0
            metrics['avg_holding_period_hours'] = 0.0
        
        # Create equity curve dataframe
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate drawdown
            equity_df['peak'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
            metrics['max_drawdown'] = equity_df['drawdown'].min()
            
            # Calculate Sharpe ratio (assuming daily data)
            if len(equity_df) > 1:
                equity_df['return'] = equity_df['portfolio_value'].pct_change().fillna(0)
                sharpe = equity_df['return'].mean() / equity_df['return'].std() * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe
                
                # Calculate Sortino ratio (downside deviation only)
                negative_returns = equity_df['return'][equity_df['return'] < 0]
                if len(negative_returns) > 0:
                    sortino = equity_df['return'].mean() / negative_returns.std() * np.sqrt(252)
                    metrics['sortino_ratio'] = sortino
                else:
                    metrics['sortino_ratio'] = float('inf')  # No negative returns
            else:
                metrics['sharpe_ratio'] = 0.0
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['max_drawdown'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
        
        # Calculate Calmar ratio (return / max drawdown)
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['total_profit'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = float('inf')  # No drawdown
        
        return metrics
    
    def _combine_symbol_metrics(self, symbol_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine metrics from multiple symbols
        
        Args:
            symbol_results: List of per-symbol metric dictionaries
            
        Returns:
            Dict with combined metrics
        """
        if not symbol_results:
            return {}
        
        # Calculate portfolio level metrics
        combined = {}
        
        # Sum portfolio values
        total_portfolio_value = sum(result.get('ending_portfolio_value', 0) for result in symbol_results)
        initial_value = 1000 * len(symbol_results)
        
        # Calculate total profit
        combined['total_profit'] = (total_portfolio_value / initial_value) - 1
        
        # Average metrics
        for metric in ['win_rate', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'avg_profit_per_trade']:
            values = [result.get(metric, 0) for result in symbol_results]
            combined[metric] = np.mean(values) if values else 0
        
        # Max drawdown is the worst across all symbols
        max_drawdowns = [result.get('max_drawdown', 0) for result in symbol_results]
        combined['max_drawdown'] = min(max_drawdowns) if max_drawdowns else 0
        
        # Total trades
        combined['total_trades'] = sum(result.get('total_trades', 0) for result in symbol_results)
        
        # Number of symbols traded
        combined['symbols_traded'] = len(symbol_results)
        
        return combined
    
    def _find_best_result(self, results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        """
        Find the best result based on the metric
        
        Args:
            results: List of optimization results
            metric: Metric to optimize
            
        Returns:
            Dict with best parameters and metrics
        """
        if not results:
            return {'params': {}, 'metrics': {}}
        
        # Filter out results with errors
        valid_results = [r for r in results if 'error' not in r['metrics']]
        
        if not valid_results:
            return {'params': {}, 'metrics': {}}
        
        # Sort by metric (higher is better by default)
        maximize = True
        if metric in ['max_drawdown']:
            maximize = False  # Lower is better for these metrics
        
        sorted_results = sorted(
            valid_results,
            key=lambda x: x['metrics'].get(metric, float('-inf') if maximize else float('inf')),
            reverse=maximize
        )
        
        return sorted_results[0]
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save optimization results to disk
        
        Args:
            results: Optimization results to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = results['strategy']
        filename = f"{strategy_name}_optimization_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Filter non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key == 'best_metrics':
                # Ensure all metrics are JSON serializable
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        logger.info(f"Optimization results saved to {filepath}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all optimization runs
        
        Returns:
            List of optimization result dictionaries
        """
        return self.optimization_results
    
    def load_optimization_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load optimization results from disk
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Dict with optimization results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded optimization results from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Error loading optimization results: {e}")
            return {}
