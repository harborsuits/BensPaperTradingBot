"""
Enhanced Component Testing Framework

Provides comprehensive utilities for testing individual strategy components
with advanced features inspired by professional trading systems.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import unittest
import tempfile
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps

# Import existing component test framework
from trading_bot.strategies.testing.component_tester import (
    ComponentTestCase, ComponentTestSuite, ComponentTestLoader
)

from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, 
    SignalGeneratorComponent, FilterComponent, 
    PositionSizerComponent, ExitManagerComponent
)
from trading_bot.strategies.components.component_registry import get_component_registry
from trading_bot.strategies.base_strategy import SignalType

logger = logging.getLogger(__name__)

# Performance metrics calculation helper functions
def calculate_win_rate(signals, actual_returns):
    """Calculate win rate for a series of signals and returns"""
    if len(signals) == 0 or len(actual_returns) == 0:
        return 0.0
    
    # Match signals with returns
    wins = sum(1 for s, r in zip(signals, actual_returns) if 
               (s == SignalType.BUY and r > 0) or 
               (s == SignalType.SELL and r < 0))
    
    return wins / len(signals) if len(signals) > 0 else 0.0

def calculate_profit_factor(signals, actual_returns):
    """Calculate profit factor for a series of signals and returns"""
    if len(signals) == 0 or len(actual_returns) == 0:
        return 0.0
    
    # Calculate gross profit and gross loss
    gross_profit = sum(r for s, r in zip(signals, actual_returns) if 
                     (s == SignalType.BUY and r > 0) or 
                     (s == SignalType.SELL and r < 0) or
                     (s == SignalType.HOLD and r > 0))
    
    gross_loss = abs(sum(r for s, r in zip(signals, actual_returns) if 
                       (s == SignalType.BUY and r < 0) or 
                       (s == SignalType.SELL and r > 0) or
                       (s == SignalType.HOLD and r < 0)))
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio for a series of returns"""
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    if len(equity_curve) == 0:
        return 0.0
    
    # Convert to numpy array for efficiency
    equity = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity)
    
    # Calculate drawdown in percentage terms
    drawdown = (running_max - equity) / running_max
    
    return drawdown.max()

class EnhancedComponentTestCase(ComponentTestCase):
    """Enhanced test case with advanced metrics and visualizations"""
    
    def __init__(self, component: StrategyComponent, test_name: str = ""):
        """Initialize enhanced test case"""
        super().__init__(component, test_name)
        
        # Additional test properties
        self.historical_data = {}  # Historical data for backtesting
        self.benchmark_data = {}   # Benchmark data for comparison
        self.performance_metrics = {}  # Performance metrics
        self.edge_case_tests = []  # Edge case tests
        self.visualization_results = {}  # Visualization results
        
    def with_historical_data(self, symbol: str, data: pd.DataFrame) -> 'EnhancedComponentTestCase':
        """Add historical data for backtesting"""
        self.historical_data[symbol] = data
        return self
    
    def with_benchmark(self, symbol: str, data: pd.DataFrame) -> 'EnhancedComponentTestCase':
        """Add benchmark data for comparison"""
        self.benchmark_data[symbol] = data
        return self
        
    def add_edge_case(self, name: str, data: Dict[str, Any], expected_outcome: Any) -> 'EnhancedComponentTestCase':
        """Add edge case test"""
        self.edge_case_tests.append({
            'name': name,
            'data': data,
            'expected_outcome': expected_outcome
        })
        return self
        
    def run_with_metrics(self) -> bool:
        """Run test case and calculate performance metrics"""
        # First run the standard test
        result = self.run()
        
        # Calculate performance metrics if we have historical data
        if self.historical_data:
            self._calculate_performance_metrics()
            
        # Run edge case tests
        self._run_edge_case_tests()
        
        return result
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for this component"""
        if not self.historical_data:
            return
        
        metrics = {}
        
        # Process based on component type
        if self.component_type == ComponentType.SIGNAL_GENERATOR:
            metrics = self._calculate_signal_generator_metrics()
        elif self.component_type == ComponentType.FILTER:
            metrics = self._calculate_filter_metrics()
        elif self.component_type == ComponentType.POSITION_SIZER:
            metrics = self._calculate_position_sizer_metrics()
        elif self.component_type == ComponentType.EXIT_MANAGER:
            metrics = self._calculate_exit_manager_metrics()
            
        self.performance_metrics = metrics
    
    def _calculate_signal_generator_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for signal generators"""
        metrics = {}
        
        for symbol, data in self.historical_data.items():
            # Skip if we don't have enough data
            if len(data) < 2:
                continue
                
            # Generate signals
            signals = []
            try:
                for i in range(len(data)):
                    # Create a slice up to the current point to simulate real-time
                    slice_data = data.iloc[:i+1].copy()
                    
                    # We need at least 2 data points for most indicators
                    if len(slice_data) < 2:
                        signals.append(SignalType.HOLD)
                        continue
                        
                    # Get signal from component
                    signal = self.component.generate_signal(symbol, slice_data)
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                continue
                
            # Calculate returns
            returns = data['close'].pct_change().dropna().values
            signals = signals[1:]  # Align with returns (skip first)
            
            # Calculate metrics
            symbol_metrics = {
                'win_rate': calculate_win_rate(signals, returns),
                'profit_factor': calculate_profit_factor(signals, returns),
                'signal_count': {
                    'buy': signals.count(SignalType.BUY),
                    'sell': signals.count(SignalType.SELL),
                    'hold': signals.count(SignalType.HOLD)
                }
            }
            
            metrics[symbol] = symbol_metrics
            
        return metrics
    
    def _calculate_filter_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for filters"""
        metrics = {}
        
        for symbol, data in self.historical_data.items():
            # Skip if we don't have enough data
            if len(data) < 2:
                continue
                
            # Test filter pass/block rates
            passed = 0
            blocked = 0
            
            try:
                for i in range(len(data)):
                    # Create a slice up to the current point to simulate real-time
                    slice_data = data.iloc[:i+1].copy()
                    
                    # We need at least 2 data points for most indicators
                    if len(slice_data) < 2:
                        continue
                        
                    # Create a mock signal
                    mock_signal = SignalType.BUY if i % 2 == 0 else SignalType.SELL
                    
                    # Check if filter passes the signal
                    if self.component.apply_filter(symbol, slice_data, mock_signal):
                        passed += 1
                    else:
                        blocked += 1
            except Exception as e:
                logger.error(f"Error testing filter for {symbol}: {e}")
                continue
                
            total = passed + blocked
            
            symbol_metrics = {
                'pass_rate': passed / total if total > 0 else 0,
                'block_rate': blocked / total if total > 0 else 0,
                'total_signals': total
            }
            
            metrics[symbol] = symbol_metrics
            
        return metrics
    
    def _calculate_position_sizer_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for position sizers"""
        metrics = {}
        
        for symbol, data in self.historical_data.items():
            # Skip if we don't have enough data
            if len(data) < 2:
                continue
                
            # Track position sizes
            position_sizes = []
            
            try:
                for i in range(len(data)):
                    # Create a slice up to the current point to simulate real-time
                    slice_data = data.iloc[:i+1].copy()
                    
                    # We need at least 2 data points for most indicators
                    if len(slice_data) < 2:
                        continue
                        
                    # Use a mock context with basic account info
                    context = {
                        'account_value': 100000,
                        'available_funds': 50000,
                        'risk_level': 'Moderate',
                        'max_position_size': 10000
                    }
                    
                    # Get position size
                    size = self.component.get_position_size(symbol, slice_data, SignalType.BUY, context)
                    position_sizes.append(size)
            except Exception as e:
                logger.error(f"Error calculating position sizes for {symbol}: {e}")
                continue
                
            # Calculate metrics
            symbol_metrics = {
                'avg_position_size': np.mean(position_sizes) if position_sizes else 0,
                'max_position_size': max(position_sizes) if position_sizes else 0,
                'min_position_size': min(position_sizes) if position_sizes else 0,
                'position_size_volatility': np.std(position_sizes) if position_sizes else 0
            }
            
            metrics[symbol] = symbol_metrics
            
        return metrics
        
    def _calculate_exit_manager_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for exit managers"""
        metrics = {}
        
        for symbol, data in self.historical_data.items():
            # Skip if we don't have enough data
            if len(data) < 2:
                continue
                
            # Track exit decisions
            exit_decisions = []
            holding_periods = []
            current_holding_period = 0
            in_position = False
            
            try:
                for i in range(len(data)):
                    # Create a slice up to the current point to simulate real-time
                    slice_data = data.iloc[:i+1].copy()
                    
                    # We need at least 2 data points for most indicators
                    if len(slice_data) < 2:
                        continue
                        
                    # Use a mock position
                    position = {
                        'symbol': symbol,
                        'entry_price': data.iloc[max(0, i-5)]['close'],  # 5 bars ago or first bar
                        'current_price': data.iloc[i]['close'],
                        'position_size': 100,
                        'entry_time': data.index[max(0, i-5)],  # 5 bars ago or first bar
                        'current_time': data.index[i]
                    }
                    
                    # Check exit decision
                    should_exit = self.component.check_exit(symbol, slice_data, position)
                    exit_decisions.append(should_exit)
                    
                    # Track holding periods
                    if should_exit and in_position:
                        holding_periods.append(current_holding_period)
                        current_holding_period = 0
                        in_position = False
                    elif not should_exit and not in_position:
                        in_position = True
                        current_holding_period = 0
                    elif in_position:
                        current_holding_period += 1
            except Exception as e:
                logger.error(f"Error testing exit manager for {symbol}: {e}")
                continue
                
            # Calculate metrics
            symbol_metrics = {
                'exit_rate': sum(exit_decisions) / len(exit_decisions) if exit_decisions else 0,
                'avg_holding_period': np.mean(holding_periods) if holding_periods else 0,
                'max_holding_period': max(holding_periods) if holding_periods else 0,
                'min_holding_period': min(holding_periods) if holding_periods else 0
            }
            
            metrics[symbol] = symbol_metrics
            
        return metrics
    
    def _run_edge_case_tests(self):
        """Run all edge case tests"""
        edge_case_results = {}
        
        for edge_case in self.edge_case_tests:
            name = edge_case['name']
            data = edge_case['data']
            expected = edge_case['expected_outcome']
            
            try:
                # Create appropriate test context based on component type
                if self.component_type == ComponentType.SIGNAL_GENERATOR:
                    symbol = data.get('symbol', 'EDGE_TEST')
                    test_data = data.get('data', pd.DataFrame())
                    actual = self.component.generate_signal(symbol, test_data)
                elif self.component_type == ComponentType.FILTER:
                    symbol = data.get('symbol', 'EDGE_TEST')
                    test_data = data.get('data', pd.DataFrame())
                    signal = data.get('signal', SignalType.BUY)
                    actual = self.component.apply_filter(symbol, test_data, signal)
                elif self.component_type == ComponentType.POSITION_SIZER:
                    symbol = data.get('symbol', 'EDGE_TEST')
                    test_data = data.get('data', pd.DataFrame())
                    signal = data.get('signal', SignalType.BUY)
                    context = data.get('context', {})
                    actual = self.component.get_position_size(symbol, test_data, signal, context)
                elif self.component_type == ComponentType.EXIT_MANAGER:
                    symbol = data.get('symbol', 'EDGE_TEST')
                    test_data = data.get('data', pd.DataFrame())
                    position = data.get('position', {})
                    actual = self.component.check_exit(symbol, test_data, position)
                else:
                    actual = None
                
                # Compare with expected
                passed = actual == expected
                
                edge_case_results[name] = {
                    'passed': passed,
                    'expected': expected,
                    'actual': actual
                }
            except Exception as e:
                # Log and mark as failed
                logger.error(f"Error running edge case '{name}': {e}")
                edge_case_results[name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        self.edge_case_results = edge_case_results
        
    def visualize_performance(self, output_dir: str = None) -> Dict[str, str]:
        """
        Generate visualization of component performance
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dict of visualization file paths
        """
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="component_test_viz_")
            
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Create visualizations based on component type
        if self.component_type == ComponentType.SIGNAL_GENERATOR:
            results = self._visualize_signal_generator(output_dir)
        elif self.component_type == ComponentType.FILTER:
            results = self._visualize_filter(output_dir)
        elif self.component_type == ComponentType.POSITION_SIZER:
            results = self._visualize_position_sizer(output_dir)
        elif self.component_type == ComponentType.EXIT_MANAGER:
            results = self._visualize_exit_manager(output_dir)
            
        self.visualization_results = results
        return results
    
    def _visualize_signal_generator(self, output_dir: str) -> Dict[str, str]:
        """Visualize signal generator performance"""
        results = {}
        
        for symbol, data in self.historical_data.items():
            if len(data) < 10:  # Need enough data for visualization
                continue
                
            # Generate signals for entire dataset
            try:
                signals = self.component.generate_signals_batch(symbol, data)
            except AttributeError:
                # Fallback if batch method not available
                signals = []
                for i in range(len(data)):
                    try:
                        signal = self.component.generate_signal(symbol, data.iloc[:i+1])
                        signals.append(signal)
                    except Exception as e:
                        logger.error(f"Error generating signal: {e}")
                        signals.append(SignalType.HOLD)
            
            # Create signal plot
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot price
            ax.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
            
            # Plot buy/sell signals
            buy_signals = [data.iloc[i]['close'] for i in range(len(signals)) if signals[i] == SignalType.BUY]
            buy_dates = [data.index[i] for i in range(len(signals)) if signals[i] == SignalType.BUY]
            
            sell_signals = [data.iloc[i]['close'] for i in range(len(signals)) if signals[i] == SignalType.SELL]
            sell_dates = [data.index[i] for i in range(len(signals)) if signals[i] == SignalType.SELL]
            
            ax.scatter(buy_dates, buy_signals, marker='^', color='green', s=100, label='Buy Signal')
            ax.scatter(sell_dates, sell_signals, marker='v', color='red', s=100, label='Sell Signal')
            
            # Add metrics to plot
            if symbol in self.performance_metrics:
                metrics = self.performance_metrics[symbol]
                metrics_text = f"Win Rate: {metrics.get('win_rate', 0):.2f}\n"
                metrics_text += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                metrics_text += f"Buy Signals: {metrics.get('signal_count', {}).get('buy', 0)}\n"
                metrics_text += f"Sell Signals: {metrics.get('signal_count', {}).get('sell', 0)}"
                
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f"{self.component.component_id} Signals for {symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{self.component.component_id}_{symbol}_signals.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            results[f"{symbol}_signals"] = fig_path
            
        return results
    
    def _visualize_filter(self, output_dir: str) -> Dict[str, str]:
        """Visualize filter component performance"""
        results = {}
        
        # Visualize pass/block rates
        if self.performance_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            symbols = list(self.performance_metrics.keys())
            pass_rates = [self.performance_metrics[s].get('pass_rate', 0) for s in symbols]
            block_rates = [self.performance_metrics[s].get('block_rate', 0) for s in symbols]
            
            x = np.arange(len(symbols))
            width = 0.35
            
            ax.bar(x - width/2, pass_rates, width, label='Pass Rate')
            ax.bar(x + width/2, block_rates, width, label='Block Rate')
            
            ax.set_ylabel('Rate')
            ax.set_title(f'{self.component.component_id} Pass/Block Rates')
            ax.set_xticks(x)
            ax.set_xticklabels(symbols)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{self.component.component_id}_filter_rates.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            results['filter_rates'] = fig_path
        
        # Visualize filter decisions over time for each symbol
        for symbol, data in self.historical_data.items():
            if len(data) < 10:  # Need enough data for visualization
                continue
                
            # Generate filter decisions
            buy_decisions = []
            sell_decisions = []
            
            for i in range(len(data)):
                slice_data = data.iloc[:i+1]
                if len(slice_data) < 2:
                    continue
                
                try:
                    buy_pass = self.component.apply_filter(symbol, slice_data, SignalType.BUY)
                    sell_pass = self.component.apply_filter(symbol, slice_data, SignalType.SELL)
                    
                    buy_decisions.append(buy_pass)
                    sell_decisions.append(sell_pass)
                except Exception as e:
                    logger.error(f"Error applying filter: {e}")
                    buy_decisions.append(False)
                    sell_decisions.append(False)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Plot price
            ax1.plot(data.index[-len(buy_decisions):], data['close'][-len(buy_decisions):], 
                    label='Price', color='blue', alpha=0.6)
            
            # Plot buy filter decisions
            buy_pass_dates = [data.index[-len(buy_decisions):][i] for i in range(len(buy_decisions)) if buy_decisions[i]]
            buy_pass_values = [1 for _ in range(len(buy_pass_dates))]
            
            ax2.scatter(buy_pass_dates, buy_pass_values, marker='o', color='green', label='Buy Passed')
            
            # Plot sell filter decisions
            sell_pass_dates = [data.index[-len(sell_decisions):][i] for i in range(len(sell_decisions)) if sell_decisions[i]]
            sell_pass_values = [0 for _ in range(len(sell_pass_dates))]
            
            ax2.scatter(sell_pass_dates, sell_pass_values, marker='o', color='red', label='Sell Passed')
            
            ax1.set_title(f"{self.component.component_id} Filter for {symbol}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel("Date")
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Sell', 'Buy'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{self.component.component_id}_{symbol}_filter.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            results[f"{symbol}_filter"] = fig_path
            
        return results
    
    def _visualize_position_sizer(self, output_dir: str) -> Dict[str, str]:
        """Visualize position sizer performance"""
        results = {}
        
        # Visualize position size distribution
        for symbol, data in self.historical_data.items():
            if len(data) < 10:  # Need enough data for visualization
                continue
                
            # Calculate position sizes
            position_sizes = []
            dates = []
            
            for i in range(len(data)):
                slice_data = data.iloc[:i+1]
                if len(slice_data) < 2:
                    continue
                
                try:
                    # Use a mock context with basic account info
                    context = {
                        'account_value': 100000,
                        'available_funds': 50000,
                        'risk_level': 'Moderate',
                        'max_position_size': 10000
                    }
                    
                    size = self.component.get_position_size(symbol, slice_data, SignalType.BUY, context)
                    position_sizes.append(size)
                    dates.append(data.index[i])
                except Exception as e:
                    logger.error(f"Error calculating position size: {e}")
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Plot price
            ax1.plot(dates, data['close'][-len(dates):], label='Price', color='blue', alpha=0.6)
            
            # Plot position sizes
            ax2.bar(dates, position_sizes, alpha=0.6, color='purple')
            
            # Add metrics
            if symbol in self.performance_metrics:
                metrics = self.performance_metrics[symbol]
                metrics_text = f"Avg Position: ${metrics.get('avg_position_size', 0):.2f}\n"
                metrics_text += f"Max Position: ${metrics.get('max_position_size', 0):.2f}\n"
                metrics_text += f"Min Position: ${metrics.get('min_position_size', 0):.2f}\n"
                metrics_text += f"Volatility: ${metrics.get('position_size_volatility', 0):.2f}"
                
                ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax1.set_title(f"{self.component.component_id} for {symbol}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Position Size ($)")
            ax2.grid(True, alpha=0.3)
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{self.component.component_id}_{symbol}_position_sizes.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            results[f"{symbol}_position_sizes"] = fig_path
            
            # Create histogram of position sizes
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(position_sizes, bins=20, alpha=0.7, color='blue')
            ax.set_title(f"{self.component.component_id} Position Size Distribution for {symbol}")
            ax.set_xlabel("Position Size ($)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{self.component.component_id}_{symbol}_position_hist.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            results[f"{symbol}_position_hist"] = fig_path
            
        return results
    
    def _visualize_exit_manager(self, output_dir: str) -> Dict[str, str]:
        """Visualize exit manager performance"""
        results = {}
        
        for symbol, data in self.historical_data.items():
            if len(data) < 10:  # Need enough data for visualization
                continue
                
            # Calculate exit decisions
            exit_decisions = []
            dates = []
            holding_periods = []
            current_holding = 0
            in_position = False
            
            for i in range(len(data)):
                slice_data = data.iloc[:i+1]
                if len(slice_data) < 2:
                    continue
                
                try:
                    # Use a mock position
                    position = {
                        'symbol': symbol,
                        'entry_price': data.iloc[max(0, i-5)]['close'],  # 5 bars ago or first bar
                        'current_price': data.iloc[i]['close'],
                        'position_size': 100,
                        'entry_time': data.index[max(0, i-5)],  # 5 bars ago or first bar
                        'current_time': data.index[i]
                    }
                    
                    # Check exit decision
                    should_exit = self.component.check_exit(symbol, slice_data, position)
                    exit_decisions.append(should_exit)
                    dates.append(data.index[i])
                    
                    # Track holding periods
                    if should_exit and in_position:
                        holding_periods.append(current_holding)
                        current_holding = 0
                        in_position = False
                    elif not should_exit and not in_position:
                        in_position = True
                        current_holding = 0
                    elif in_position:
                        current_holding += 1
                except Exception as e:
                    logger.error(f"Error checking exit: {e}")
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Plot price
            ax1.plot(dates, data['close'][-len(dates):], label='Price', color='blue', alpha=0.6)
            
            # Plot exit decisions
            exit_dates = [dates[i] for i in range(len(exit_decisions)) if exit_decisions[i]]
            exit_values = [data['close'][-len(dates):][i] for i in range(len(exit_decisions)) if exit_decisions[i]]
            
            ax1.scatter(exit_dates, exit_values, marker='X', color='red', s=100, label='Exit')
            
            # Plot holding periods histogram
            if holding_periods:
                ax2.hist(holding_periods, bins=min(20, len(holding_periods)), alpha=0.7, color='green')
                ax2.set_xlabel("Holding Period (bars)")
                ax2.set_ylabel("Frequency")
            else:
                ax2.text(0.5, 0.5, "No completed holdings", ha='center', va='center')
            
            # Add metrics
            if symbol in self.performance_metrics:
                metrics = self.performance_metrics[symbol]
                metrics_text = f"Exit Rate: {metrics.get('exit_rate', 0):.2f}\n"
                metrics_text += f"Avg Holding: {metrics.get('avg_holding_period', 0):.1f} bars\n"
                metrics_text += f"Max Holding: {metrics.get('max_holding_period', 0)} bars\n"
                metrics_text += f"Min Holding: {metrics.get('min_holding_period', 0)} bars"
                
                ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax1.set_title(f"{self.component.component_id} Exits for {symbol}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{self.component.component_id}_{symbol}_exits.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            results[f"{symbol}_exits"] = fig_path
            
        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report for this component"""
        report = {
            'component_id': self.component.component_id,
            'component_type': self.component_type.name,
            'performance_metrics': self.performance_metrics,
            'edge_case_results': getattr(self, 'edge_case_results', {}),
            'visualization_files': self.visualization_results,
            'test_passed': self.passed,
            'error_message': self.error_message
        }
        
        return report
