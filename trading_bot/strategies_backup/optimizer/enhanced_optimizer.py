"""
Enhanced Strategy Optimizer

Provides advanced optimization capabilities for trading strategy components
with multiple optimization methods and parameter space exploration.
Inspired by professional trading frameworks.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import concurrent.futures
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import itertools
import random
from tqdm import tqdm
import time

from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, 
    SignalGeneratorComponent, FilterComponent, 
    PositionSizerComponent, ExitManagerComponent
)
from trading_bot.strategies.components.component_registry import get_component_registry
from trading_bot.strategies.base_strategy import SignalType

logger = logging.getLogger(__name__)

class OptimizationResult:
    """Container for optimization results"""
    
    def __init__(self):
        self.parameter_sets = []
        self.performance_metrics = []
        self.best_parameters = {}
        self.best_performance = {}
        self.optimization_time = 0
        self.component_id = ""
        self.component_type = None
        self.optimization_method = ""
        self.status = "not_started"  # not_started, running, completed, failed
        self.error_message = ""
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.parameter_importance = {}  # For tracking which parameters matter most
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'id': self.id,
            'component_id': self.component_id,
            'component_type': self.component_type.name if self.component_type else None,
            'optimization_method': self.optimization_method,
            'parameter_sets': self.parameter_sets[:min(100, len(self.parameter_sets))],  # Limit to prevent large payloads
            'best_parameters': self.best_parameters,
            'best_performance': self.best_performance,
            'optimization_time': self.optimization_time,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'parameter_importance': self.parameter_importance
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'OptimizationResult':
        """Create from dictionary"""
        result = OptimizationResult()
        
        result.id = data.get('id', str(uuid.uuid4()))
        result.component_id = data.get('component_id', "")
        
        # Handle component_type
        component_type_str = data.get('component_type')
        if component_type_str:
            try:
                result.component_type = ComponentType[component_type_str]
            except (KeyError, TypeError):
                result.component_type = None
        
        result.optimization_method = data.get('optimization_method', "")
        result.parameter_sets = data.get('parameter_sets', [])
        result.best_parameters = data.get('best_parameters', {})
        result.best_performance = data.get('best_performance', {})
        result.optimization_time = data.get('optimization_time', 0)
        result.status = data.get('status', "completed")
        result.error_message = data.get('error_message', "")
        
        # Handle created_at
        created_at_str = data.get('created_at')
        if created_at_str:
            try:
                result.created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                result.created_at = datetime.now()
        
        result.parameter_importance = data.get('parameter_importance', {})
        
        return result


class BaseOptimizer:
    """Base class for strategy component optimizers"""
    
    def __init__(self):
        self.name = "base_optimizer"
        self.description = "Base optimizer class"
        self.component = None
        self.historical_data = {}
        self.parameter_ranges = {}
        self.evaluation_metric = "profit"  # Default metric
        self.max_evaluations = 100
        self.current_result = None
        self.parallel_jobs = 4  # Default parallel jobs
        self.save_path = None
    
    def set_component(self, component: StrategyComponent) -> 'BaseOptimizer':
        """Set the component to optimize"""
        self.component = component
        return self
    
    def with_data(self, symbol: str, data: pd.DataFrame) -> 'BaseOptimizer':
        """Add historical data for optimization"""
        self.historical_data[symbol] = data
        return self
    
    def with_parameter_ranges(self, parameter_ranges: Dict[str, List[Any]]) -> 'BaseOptimizer':
        """Set parameter ranges for optimization"""
        self.parameter_ranges = parameter_ranges
        return self
    
    def with_metric(self, metric: str) -> 'BaseOptimizer':
        """Set evaluation metric"""
        valid_metrics = ["profit", "sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor", "max_drawdown"]
        
        if metric.lower() not in valid_metrics:
            logger.warning(f"Metric '{metric}' not recognized. Using default: {self.evaluation_metric}")
            return self
        
        self.evaluation_metric = metric.lower()
        return self
    
    def with_max_evaluations(self, max_evals: int) -> 'BaseOptimizer':
        """Set maximum number of evaluations"""
        self.max_evaluations = max(1, max_evals)
        return self
    
    def with_parallel_jobs(self, jobs: int) -> 'BaseOptimizer':
        """Set number of parallel jobs"""
        self.parallel_jobs = max(1, jobs)
        return self
    
    def save_results_to(self, path: str) -> 'BaseOptimizer':
        """Set path to save optimization results"""
        self.save_path = path
        return self
    
    def optimize(self) -> OptimizationResult:
        """
        Optimize component parameters
        
        Returns:
            Optimization result
        """
        if not self.component:
            raise ValueError("Component not set")
        
        if not self.historical_data:
            raise ValueError("No historical data provided")
        
        if not self.parameter_ranges:
            raise ValueError("No parameter ranges provided")
        
        # Initialize result object
        result = OptimizationResult()
        result.component_id = self.component.component_id
        result.component_type = self.component.component_type
        result.optimization_method = self.name
        result.status = "running"
        
        self.current_result = result
        
        # Implementation in derived classes
        return result
    
    def _evaluate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a set of parameters
        
        Args:
            parameters: Parameter set to evaluate
            
        Returns:
            Performance metrics
        """
        # Apply parameters to component
        original_params = {}
        
        # Store original parameters and apply new ones
        for param_name, param_value in parameters.items():
            if hasattr(self.component, param_name):
                original_params[param_name] = getattr(self.component, param_name)
                setattr(self.component, param_name, param_value)
        
        try:
            # Evaluate based on component type
            if self.component.component_type == ComponentType.SIGNAL_GENERATOR:
                metrics = self._evaluate_signal_generator()
            elif self.component.component_type == ComponentType.FILTER:
                metrics = self._evaluate_filter()
            elif self.component.component_type == ComponentType.POSITION_SIZER:
                metrics = self._evaluate_position_sizer()
            elif self.component.component_type == ComponentType.EXIT_MANAGER:
                metrics = self._evaluate_exit_manager()
            else:
                metrics = {'error': 'Unsupported component type'}
        
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            metrics = {'error': str(e)}
        
        # Restore original parameters
        for param_name, param_value in original_params.items():
            setattr(self.component, param_name, param_value)
        
        return metrics
    
    def _evaluate_signal_generator(self) -> Dict[str, float]:
        """Evaluate signal generator performance"""
        results = {}
        
        total_profit = 0
        trades = 0
        win_count = 0
        loss_count = 0
        returns = []
        
        for symbol, data in self.historical_data.items():
            # Skip if not enough data
            if len(data) < 10:
                continue
            
            try:
                # Generate signals
                signals = []
                for i in range(len(data) - 1):
                    # Create a slice up to the current point to simulate real-time
                    slice_data = data.iloc[:i+1].copy()
                    
                    # Skip if not enough data for indicators
                    if len(slice_data) < 5:
                        signals.append(SignalType.HOLD)
                        continue
                    
                    # Get signal from component
                    signal = self.component.generate_signal(symbol, slice_data)
                    signals.append(signal)
                
                # Calculate simple returns (not compounded)
                symbol_returns = []
                for i in range(len(signals)):
                    # Get next day's return
                    if i+1 >= len(data):
                        continue
                    
                    daily_return = (data.iloc[i+1]['close'] - data.iloc[i]['close']) / data.iloc[i]['close']
                    
                    # Apply signal direction
                    if signals[i] == SignalType.BUY:
                        signal_return = daily_return
                        trades += 1
                    elif signals[i] == SignalType.SELL:
                        signal_return = -daily_return  # Short position
                        trades += 1
                    else:  # HOLD
                        signal_return = 0
                    
                    symbol_returns.append(signal_return)
                    returns.append(signal_return)
                    
                    # Track win/loss
                    if signal_return > 0:
                        win_count += 1
                    elif signal_return < 0:
                        loss_count += 1
                
                # Calculate symbol profit
                symbol_profit = sum(symbol_returns)
                total_profit += symbol_profit
                
                # Store symbol-specific metrics
                results[f"{symbol}_profit"] = symbol_profit
                results[f"{symbol}_trade_count"] = sum(1 for s in signals if s != SignalType.HOLD)
                
            except Exception as e:
                logger.error(f"Error evaluating signal generator for {symbol}: {e}")
                results[f"{symbol}_error"] = str(e)
        
        # Overall metrics
        results['total_profit'] = total_profit
        results['trade_count'] = trades
        
        if trades > 0:
            results['win_rate'] = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
        else:
            results['win_rate'] = 0
        
        # Calculate more advanced metrics if we have returns
        if returns:
            results['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            results['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            results['max_drawdown'] = self._calculate_max_drawdown(returns)
            results['profit_factor'] = self._calculate_profit_factor(returns)
        
        return results
    
    def _evaluate_filter(self) -> Dict[str, float]:
        """Evaluate filter performance"""
        results = {}
        
        # This will be similar to _evaluate_signal_generator but focused on filter metrics
        # For a filter, we want to measure how well it improves signal quality
        
        pass_count = 0
        block_count = 0
        total_profit_with_filter = 0
        total_profit_without_filter = 0
        
        for symbol, data in self.historical_data.items():
            # Skip if not enough data
            if len(data) < 10:
                continue
            
            try:
                # Generate mock signals (alternating buy/sell)
                signals = []
                
                # We'll use a simple rule to generate test signals
                for i in range(len(data) - 1):
                    if i % 5 == 0:  # Every 5th bar
                        mock_signal = SignalType.BUY
                    elif i % 5 == 2:  # Every 5th+2 bar
                        mock_signal = SignalType.SELL
                    else:
                        mock_signal = SignalType.HOLD
                    
                    signals.append(mock_signal)
                
                # Apply filter and calculate metrics
                filtered_signals = []
                for i in range(len(signals)):
                    if signals[i] == SignalType.HOLD:
                        filtered_signals.append(SignalType.HOLD)
                        continue
                    
                    # Apply filter
                    slice_data = data.iloc[:i+1].copy()
                    
                    # Skip if not enough data for indicators
                    if len(slice_data) < 5:
                        filtered_signals.append(SignalType.HOLD)
                        continue
                    
                    passed = self.component.apply_filter(symbol, slice_data, signals[i])
                    
                    if passed:
                        filtered_signals.append(signals[i])
                        pass_count += 1
                    else:
                        filtered_signals.append(SignalType.HOLD)
                        block_count += 1
                
                # Calculate returns for both original and filtered signals
                unfiltered_returns = []
                filtered_returns = []
                
                for i in range(len(signals)):
                    # Get next day's return
                    if i+1 >= len(data):
                        continue
                    
                    daily_return = (data.iloc[i+1]['close'] - data.iloc[i]['close']) / data.iloc[i]['close']
                    
                    # Unfiltered signal return
                    if signals[i] == SignalType.BUY:
                        unfiltered_return = daily_return
                    elif signals[i] == SignalType.SELL:
                        unfiltered_return = -daily_return  # Short position
                    else:  # HOLD
                        unfiltered_return = 0
                    
                    unfiltered_returns.append(unfiltered_return)
                    
                    # Filtered signal return
                    if filtered_signals[i] == SignalType.BUY:
                        filtered_return = daily_return
                    elif filtered_signals[i] == SignalType.SELL:
                        filtered_return = -daily_return  # Short position
                    else:  # HOLD
                        filtered_return = 0
                    
                    filtered_returns.append(filtered_return)
                
                # Calculate profit
                symbol_profit_unfiltered = sum(unfiltered_returns)
                symbol_profit_filtered = sum(filtered_returns)
                
                total_profit_without_filter += symbol_profit_unfiltered
                total_profit_with_filter += symbol_profit_filtered
                
                # Store symbol-specific metrics
                results[f"{symbol}_profit_improvement"] = symbol_profit_filtered - symbol_profit_unfiltered
                results[f"{symbol}_pass_rate"] = pass_count / (pass_count + block_count) if (pass_count + block_count) > 0 else 0
                
            except Exception as e:
                logger.error(f"Error evaluating filter for {symbol}: {e}")
                results[f"{symbol}_error"] = str(e)
        
        # Overall metrics
        results['profit_improvement'] = total_profit_with_filter - total_profit_without_filter
        results['pass_rate'] = pass_count / (pass_count + block_count) if (pass_count + block_count) > 0 else 0
        results['block_rate'] = block_count / (pass_count + block_count) if (pass_count + block_count) > 0 else 0
        
        # Quality metrics
        if total_profit_without_filter != 0:
            results['quality_improvement'] = (total_profit_with_filter / total_profit_without_filter) - 1
        else:
            results['quality_improvement'] = 0
        
        return results
    
    def _evaluate_position_sizer(self) -> Dict[str, float]:
        """Evaluate position sizer performance"""
        # For position sizers, we need to evaluate how well they allocate capital
        # This is more complex and requires simulating a portfolio
        
        # This is a simplified implementation - a real one would use proper portfolio simulation
        
        results = {}
        
        for symbol, data in self.historical_data.items():
            # Skip if not enough data
            if len(data) < 10:
                continue
            
            try:
                # Generate mock signals
                signals = []
                for i in range(len(data)):
                    if i % 10 == 0:  # Every 10th bar
                        signals.append(SignalType.BUY)
                    elif i % 10 == 5:  # Every 10th+5 bar
                        signals.append(SignalType.SELL)
                    else:
                        signals.append(SignalType.HOLD)
                
                # Calculate position sizes
                position_sizes = []
                
                for i in range(len(data)):
                    if signals[i] == SignalType.HOLD:
                        position_sizes.append(0)
                        continue
                    
                    # Context for position sizing
                    context = {
                        'account_value': 100000,
                        'available_funds': 50000,
                        'risk_level': 'Moderate',
                        'max_position_size': 10000
                    }
                    
                    # Get slice data
                    slice_data = data.iloc[:i+1].copy()
                    
                    # Skip if not enough data for indicators
                    if len(slice_data) < 5:
                        position_sizes.append(0)
                        continue
                    
                    # Get position size
                    size = self.component.get_position_size(symbol, slice_data, signals[i], context)
                    position_sizes.append(size)
                
                # Calculate metrics
                avg_size = np.mean(position_sizes) if position_sizes else 0
                max_size = np.max(position_sizes) if position_sizes else 0
                min_size = np.min([p for p in position_sizes if p > 0]) if any(p > 0 for p in position_sizes) else 0
                size_volatility = np.std(position_sizes) if position_sizes else 0
                
                # Store metrics
                results[f"{symbol}_avg_size"] = avg_size
                results[f"{symbol}_max_size"] = max_size
                results[f"{symbol}_min_size"] = min_size
                results[f"{symbol}_size_volatility"] = size_volatility
                
                # Calculate position size efficiency
                # This is a simplified metric - a real implementation would use portfolio simulation
                # The idea is to measure how well position sizes align with profitable trades
                
                # Get next-day returns
                returns = []
                for i in range(len(data) - 1):
                    daily_return = (data.iloc[i+1]['close'] - data.iloc[i]['close']) / data.iloc[i]['close']
                    returns.append(daily_return)
                
                # Last day has no next-day return
                returns.append(0)
                
                # Calculate weighted returns based on position sizes
                weighted_returns = []
                for i in range(len(returns)):
                    if signals[i] == SignalType.BUY:
                        w_return = returns[i] * position_sizes[i]
                    elif signals[i] == SignalType.SELL:
                        w_return = -returns[i] * position_sizes[i]
                    else:
                        w_return = 0
                    
                    weighted_returns.append(w_return)
                
                # Calculate efficiency
                total_weighted_return = sum(weighted_returns)
                results[f"{symbol}_weighted_return"] = total_weighted_return
                
            except Exception as e:
                logger.error(f"Error evaluating position sizer for {symbol}: {e}")
                results[f"{symbol}_error"] = str(e)
        
        # Overall metrics
        results['total_weighted_return'] = sum(results.get(f"{symbol}_weighted_return", 0) 
                                           for symbol in self.historical_data.keys())
        
        return results
    
    def _evaluate_exit_manager(self) -> Dict[str, float]:
        """Evaluate exit manager performance"""
        # For exit managers, we need to evaluate how well they time exits
        
        results = {}
        
        total_profit = 0
        avg_hold_time = []
        
        for symbol, data in self.historical_data.items():
            # Skip if not enough data
            if len(data) < 10:
                continue
            
            try:
                # Simulate trades with entries at regular intervals
                in_position = False
                entry_price = 0
                entry_time = 0
                
                profits = []
                hold_times = []
                
                for i in range(len(data)):
                    # Entry logic
                    if not in_position and i % 10 == 0:  # Enter every 10th bar
                        in_position = True
                        entry_price = data.iloc[i]['close']
                        entry_time = i
                        continue
                    
                    # Skip if not in position
                    if not in_position:
                        continue
                    
                    # Exit logic
                    slice_data = data.iloc[:i+1].copy()
                    
                    # Create position info
                    position = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'current_price': data.iloc[i]['close'],
                        'position_size': 100,
                        'entry_time': data.index[entry_time],
                        'current_time': data.index[i]
                    }
                    
                    # Check exit
                    should_exit = self.component.check_exit(symbol, slice_data, position)
                    
                    if should_exit:
                        # Calculate profit
                        exit_price = data.iloc[i]['close']
                        profit = (exit_price - entry_price) / entry_price
                        profits.append(profit)
                        
                        # Calculate hold time
                        hold_time = i - entry_time
                        hold_times.append(hold_time)
                        
                        # Reset
                        in_position = False
                    
                # Calculate metrics
                symbol_profit = sum(profits)
                total_profit += symbol_profit
                
                avg_symbol_hold_time = np.mean(hold_times) if hold_times else 0
                avg_hold_time.extend(hold_times)
                
                # Store metrics
                results[f"{symbol}_profit"] = symbol_profit
                results[f"{symbol}_avg_hold_time"] = avg_symbol_hold_time
                results[f"{symbol}_win_rate"] = len([p for p in profits if p > 0]) / len(profits) if profits else 0
                
            except Exception as e:
                logger.error(f"Error evaluating exit manager for {symbol}: {e}")
                results[f"{symbol}_error"] = str(e)
        
        # Overall metrics
        results['total_profit'] = total_profit
        results['avg_hold_time'] = np.mean(avg_hold_time) if avg_hold_time else 0
        
        return results
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        daily_sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        # Annualize (assuming daily returns)
        annual_sharpe = daily_sharpe * np.sqrt(252)
        
        return annual_sharpe
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio from returns"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        # Calculate downside deviation (standard deviation of negative returns only)
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
            
        daily_sortino = np.mean(excess_returns) / np.std(negative_returns)
        
        # Annualize (assuming daily returns)
        annual_sortino = daily_sortino * np.sqrt(252)
        
        return annual_sortino
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + np.array(returns))
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = (running_max - cum_returns) / running_max
        
        return np.max(drawdown)
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor from returns"""
        if not returns:
            return 0.0
        
        # Separate gains and losses
        gains = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        total_gain = sum(gains)
        total_loss = sum(losses)
        
        if total_loss == 0:
            return float('inf') if total_gain > 0 else 0.0
            
        return total_gain / total_loss
