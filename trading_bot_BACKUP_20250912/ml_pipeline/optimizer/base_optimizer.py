"""
Base Optimizer Module

Provides the base class for all optimization methods.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

import pandas as pd
import numpy as np

from trading_bot.ml_pipeline.optimizer.metrics import StrategyMetrics

logger = logging.getLogger(__name__)

class BaseOptimizer:
    """
    Base class for all optimization methods
    
    This class defines the common interface and functionality
    for all strategy optimization methods.
    """
    
    def __init__(self, config=None):
        """
        Initialize the base optimizer
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        
        # Common configuration
        self.results_dir = self.config.get('results_dir', 'optimization_results')
        self.max_workers = self.config.get('max_workers', os.cpu_count())
        self.n_trials = self.config.get('n_trials', 100)
        self.random_state = self.config.get('random_state', 42)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics = StrategyMetrics()
        
        # Store optimization results
        self.optimization_results = []
    
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
        raise NotImplementedError("Subclasses must implement this method")
    
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
        
        # Call the comprehensive metrics calculator
        return self.metrics.calculate_all_metrics(trades, equity_curve, symbol, portfolio_value, initial_value)
    
    def _combine_symbol_metrics(self, symbol_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine metrics from multiple symbols
        
        Args:
            symbol_results: List of per-symbol metric dictionaries
            
        Returns:
            Dict with combined metrics
        """
        return self.metrics.combine_metrics(symbol_results)
    
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
