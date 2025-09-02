"""
Component Performance Analytics

This module provides analytics and performance tracking for strategy components.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import uuid
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.strategies.modular_strategy import ModularStrategy
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, 
    SignalGeneratorComponent, FilterComponent, 
    PositionSizerComponent, ExitManagerComponent
)
from trading_bot.strategies.components.component_registry import get_component_registry

logger = logging.getLogger(__name__)

class ComponentMetrics:
    """Calculate and track performance metrics for strategy components."""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        """
        Initialize component metrics
        
        Args:
            component_id: Component ID
            component_type: Component type
        """
        self.component_id = component_id
        self.component_type = component_type
        self.signals_history = []
        self.trades_history = []
        self.performance_metrics = {}
        self.cumulative_metrics = {}
    
    def add_signal(self, timestamp: datetime, signal: Dict[str, Any], 
                  result: Optional[Dict[str, Any]] = None) -> None:
        """
        Add signal to history
        
        Args:
            timestamp: Signal timestamp
            signal: Signal data
            result: Signal result (if available)
        """
        self.signals_history.append({
            'timestamp': timestamp,
            'signal': signal,
            'result': result
        })
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Add trade to history
        
        Args:
            trade_data: Trade data
        """
        self.trades_history.append(trade_data)
    
    def calculate_signal_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for signals
        
        Returns:
            Signal metrics
        """
        if not self.signals_history:
            return {}
        
        metrics = {}
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.signals_history)
        
        # Add timestamps as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Signal count
        metrics['signal_count'] = len(df)
        
        # Analyze signals based on component type
        if self.component_type == ComponentType.SIGNAL_GENERATOR:
            # Count signal types
            signal_types = {}
            for entry in self.signals_history:
                signal = entry.get('signal', {})
                
                for symbol, signal_type in signal.items():
                    if symbol not in signal_types:
                        signal_types[symbol] = {}
                    
                    signal_type_str = str(signal_type)
                    signal_types[symbol][signal_type_str] = signal_types[symbol].get(signal_type_str, 0) + 1
            
            metrics['signal_types'] = signal_types
            
            # Calculate correct signal ratio if results available
            correct_signals = 0
            total_signals_with_result = 0
            
            for entry in self.signals_history:
                signal = entry.get('signal', {})
                result = entry.get('result', {})
                
                if not result:
                    continue
                
                for symbol, signal_type in signal.items():
                    if symbol in result:
                        total_signals_with_result += 1
                        
                        # Check if signal was correct
                        if result[symbol].get('correct', False):
                            correct_signals += 1
            
            if total_signals_with_result > 0:
                metrics['correct_signal_ratio'] = correct_signals / total_signals_with_result
            
        elif self.component_type == ComponentType.FILTER:
            # Count filtered signals
            filtered_count = 0
            for entry in self.signals_history:
                signal = entry.get('signal', {})
                result = entry.get('result', {})
                
                if not signal or not result:
                    continue
                
                for symbol, before in signal.items():
                    if symbol in result:
                        after = result[symbol]
                        
                        if before != after:
                            filtered_count += 1
            
            metrics['filtered_count'] = filtered_count
            if len(self.signals_history) > 0:
                metrics['filter_ratio'] = filtered_count / len(self.signals_history)
        
        elif self.component_type == ComponentType.POSITION_SIZER:
            # Analyze position sizes
            position_sizes = {}
            for entry in self.signals_history:
                result = entry.get('result', {})
                
                if not result:
                    continue
                
                for symbol, size in result.items():
                    if symbol not in position_sizes:
                        position_sizes[symbol] = []
                    
                    position_sizes[symbol].append(size)
            
            # Calculate size statistics
            size_stats = {}
            for symbol, sizes in position_sizes.items():
                size_stats[symbol] = {
                    'mean': np.mean(sizes),
                    'median': np.median(sizes),
                    'min': np.min(sizes),
                    'max': np.max(sizes),
                    'std': np.std(sizes)
                }
            
            metrics['position_size_stats'] = size_stats
        
        elif self.component_type == ComponentType.EXIT_MANAGER:
            # Count exit signals
            exit_count = 0
            for entry in self.signals_history:
                result = entry.get('result', {})
                
                if not result:
                    continue
                
                for symbol, exit_flag in result.items():
                    if exit_flag:
                        exit_count += 1
            
            metrics['exit_count'] = exit_count
            if len(self.signals_history) > 0:
                metrics['exit_ratio'] = exit_count / len(self.signals_history)
        
        return metrics
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for trades
        
        Returns:
            Trade metrics
        """
        if not self.trades_history:
            return {}
        
        metrics = {}
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.trades_history)
        
        # Add timestamps if available
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Trade count
        metrics['trade_count'] = len(df)
        
        # Win rate
        if 'profit_pct' in df.columns:
            winning_trades = df[df['profit_pct'] > 0]
            metrics['win_count'] = len(winning_trades)
            metrics['win_rate'] = len(winning_trades) / len(df) if len(df) > 0 else 0
            
            # Calculate profit metrics
            if len(winning_trades) > 0:
                metrics['avg_win_pct'] = winning_trades['profit_pct'].mean()
                metrics['max_win_pct'] = winning_trades['profit_pct'].max()
            
            # Calculate loss metrics
            losing_trades = df[df['profit_pct'] < 0]
            if len(losing_trades) > 0:
                metrics['avg_loss_pct'] = losing_trades['profit_pct'].mean()
                metrics['max_loss_pct'] = losing_trades['profit_pct'].min()
            
            # Calculate overall metrics
            metrics['avg_profit_pct'] = df['profit_pct'].mean()
            metrics['median_profit_pct'] = df['profit_pct'].median()
            metrics['total_profit_pct'] = df['profit_pct'].sum()
            
            # Calculate profit factor
            gross_profit = winning_trades['profit_pct'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['profit_pct'].sum()) if len(losing_trades) > 0 else 0
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Hold time
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['hold_time'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600  # in hours
            metrics['avg_hold_time'] = df['hold_time'].mean()
            metrics['max_hold_time'] = df['hold_time'].max()
            metrics['min_hold_time'] = df['hold_time'].min()
        
        return metrics
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate all metrics
        
        Returns:
            All metrics
        """
        signal_metrics = self.calculate_signal_metrics()
        trade_metrics = self.calculate_trade_metrics()
        
        self.performance_metrics = {**signal_metrics, **trade_metrics}
        return self.performance_metrics
    
    def calculate_cumulative_metrics(self, window_size: int = 20) -> Dict[str, Any]:
        """
        Calculate cumulative metrics over time
        
        Args:
            window_size: Window size for rolling metrics
            
        Returns:
            Cumulative metrics
        """
        if not self.trades_history:
            return {}
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.trades_history)
        
        # Add timestamps if available
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df.sort_values('exit_time', inplace=True)
        
        cumulative = {}
        
        # Cumulative profit
        if 'profit_pct' in df.columns:
            df['cumulative_profit'] = df['profit_pct'].cumsum()
            cumulative['cumulative_profit'] = df['cumulative_profit'].tolist()
            
            # Rolling win rate
            df['win'] = df['profit_pct'] > 0
            df['rolling_win_rate'] = df['win'].rolling(window=window_size).mean()
            cumulative['rolling_win_rate'] = df['rolling_win_rate'].tolist()
            
            # Rolling profit
            df['rolling_profit'] = df['profit_pct'].rolling(window=window_size).mean()
            cumulative['rolling_profit'] = df['rolling_profit'].tolist()
        
        # Timestamps
        if 'exit_time' in df.columns:
            cumulative['timestamps'] = df['exit_time'].tolist()
        
        self.cumulative_metrics = cumulative
        return self.cumulative_metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.name,
            'metrics': self.performance_metrics,
            'cumulative_metrics': self.cumulative_metrics,
            'last_updated': datetime.now().isoformat()
        }
    
    def save(self, file_path: str) -> bool:
        """
        Save metrics to file
        
        Args:
            file_path: File path to save to
            
        Returns:
            Success flag
        """
        try:
            # Calculate metrics
            self.calculate_metrics()
            self.calculate_cumulative_metrics()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> Optional['ComponentMetrics']:
        """
        Load metrics from file
        
        Args:
            file_path: File path to load from
            
        Returns:
            ComponentMetrics instance or None if loading failed
        """
        try:
            # Load from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create instance
            component_id = data.get('component_id', '')
            component_type = ComponentType[data.get('component_type', 'SIGNAL_GENERATOR')]
            
            metrics = cls(component_id, component_type)
            metrics.performance_metrics = data.get('metrics', {})
            metrics.cumulative_metrics = data.get('cumulative_metrics', {})
            
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return None

class ComponentAnalytics:
    """Analytics engine for strategy components."""
    
    def __init__(self, metrics_dir: str = None):
        """
        Initialize component analytics
        
        Args:
            metrics_dir: Directory for storing metrics
        """
        self.metrics_dir = metrics_dir or os.path.join(
            os.path.dirname(__file__), 
            '../../../data/component_metrics'
        )
        
        # Ensure directory exists
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Component metrics
        self.component_metrics = {}
    
    def get_metrics_file_path(self, component_id: str) -> str:
        """
        Get file path for component metrics
        
        Args:
            component_id: Component ID
            
        Returns:
            Metrics file path
        """
        return os.path.join(self.metrics_dir, f"{component_id}.json")
    
    def get_component_metrics(self, component_id: str) -> Optional[ComponentMetrics]:
        """
        Get metrics for component
        
        Args:
            component_id: Component ID
            
        Returns:
            ComponentMetrics instance or None if not found
        """
        # Check if already loaded
        if component_id in self.component_metrics:
            return self.component_metrics[component_id]
        
        # Check if metrics file exists
        file_path = self.get_metrics_file_path(component_id)
        if os.path.exists(file_path):
            # Load from file
            metrics = ComponentMetrics.load(file_path)
            if metrics:
                self.component_metrics[component_id] = metrics
                return metrics
        
        return None
    
    def create_component_metrics(self, component_id: str, 
                               component_type: ComponentType) -> ComponentMetrics:
        """
        Create metrics for component
        
        Args:
            component_id: Component ID
            component_type: Component type
            
        Returns:
            ComponentMetrics instance
        """
        metrics = ComponentMetrics(component_id, component_type)
        self.component_metrics[component_id] = metrics
        return metrics
    
    def record_signal(self, component_id: str, component_type: ComponentType,
                    timestamp: datetime, signal: Dict[str, Any], 
                    result: Optional[Dict[str, Any]] = None) -> None:
        """
        Record signal for component
        
        Args:
            component_id: Component ID
            component_type: Component type
            timestamp: Signal timestamp
            signal: Signal data
            result: Signal result (if available)
        """
        # Get or create metrics
        metrics = self.get_component_metrics(component_id)
        if not metrics:
            metrics = self.create_component_metrics(component_id, component_type)
        
        # Add signal
        metrics.add_signal(timestamp, signal, result)
        
        # Save periodically
        if len(metrics.signals_history) % 100 == 0:
            file_path = self.get_metrics_file_path(component_id)
            metrics.save(file_path)
    
    def record_trade(self, component_id: str, component_type: ComponentType,
                    trade_data: Dict[str, Any]) -> None:
        """
        Record trade for component
        
        Args:
            component_id: Component ID
            component_type: Component type
            trade_data: Trade data
        """
        # Get or create metrics
        metrics = self.get_component_metrics(component_id)
        if not metrics:
            metrics = self.create_component_metrics(component_id, component_type)
        
        # Add trade
        metrics.add_trade(trade_data)
        
        # Save periodically
        if len(metrics.trades_history) % 10 == 0:
            file_path = self.get_metrics_file_path(component_id)
            metrics.save(file_path)
    
    def save_all_metrics(self) -> None:
        """Save all component metrics."""
        for component_id, metrics in self.component_metrics.items():
            file_path = self.get_metrics_file_path(component_id)
            metrics.save(file_path)
    
    def get_all_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all components
        
        Returns:
            Dictionary of component metrics
        """
        result = {}
        
        # Scan metrics directory
        for filename in os.listdir(self.metrics_dir):
            if filename.endswith('.json'):
                component_id = filename[:-5]  # Remove '.json'
                
                # Load metrics
                metrics = self.get_component_metrics(component_id)
                if metrics:
                    result[component_id] = metrics.to_dict()
        
        return result
    
    def generate_performance_report(self, component_id: str) -> Dict[str, Any]:
        """
        Generate performance report for component
        
        Args:
            component_id: Component ID
            
        Returns:
            Performance report
        """
        # Get metrics
        metrics = self.get_component_metrics(component_id)
        if not metrics:
            return {'error': f"No metrics found for component {component_id}"}
        
        # Calculate metrics
        metrics.calculate_metrics()
        metrics.calculate_cumulative_metrics()
        
        # Create report
        report = metrics.to_dict()
        
        # Add summary
        if 'metrics' in report:
            perf_metrics = report['metrics']
            
            summary = {}
            
            # Signal metrics
            if 'signal_count' in perf_metrics:
                summary['signal_count'] = perf_metrics['signal_count']
            
            # Trade metrics
            if 'trade_count' in perf_metrics:
                summary['trade_count'] = perf_metrics['trade_count']
            
            if 'win_rate' in perf_metrics:
                summary['win_rate'] = f"{perf_metrics['win_rate']:.2%}"
            
            if 'avg_profit_pct' in perf_metrics:
                summary['avg_profit'] = f"{perf_metrics['avg_profit_pct']:.2f}%"
            
            if 'profit_factor' in perf_metrics:
                summary['profit_factor'] = f"{perf_metrics['profit_factor']:.2f}"
            
            report['summary'] = summary
        
        return report
    
    def compare_components(self, component_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance of multiple components
        
        Args:
            component_ids: List of component IDs
            
        Returns:
            Comparison report
        """
        # Get metrics for all components
        components = {}
        for component_id in component_ids:
            metrics = self.get_component_metrics(component_id)
            if metrics:
                metrics.calculate_metrics()
                components[component_id] = metrics
        
        if not components:
            return {'error': "No metrics found for any component"}
        
        # Create comparison report
        comparison = {
            'components': {},
            'metrics_comparison': {}
        }
        
        # Add component info
        for component_id, metrics in components.items():
            comparison['components'][component_id] = {
                'component_id': component_id,
                'component_type': metrics.component_type.name
            }
        
        # Compare key metrics
        key_metrics = [
            'win_rate', 'profit_factor', 'avg_profit_pct', 
            'trade_count', 'signal_count'
        ]
        
        for metric in key_metrics:
            comparison['metrics_comparison'][metric] = {}
            
            for component_id, metrics in components.items():
                if metric in metrics.performance_metrics:
                    comparison['metrics_comparison'][metric][component_id] = metrics.performance_metrics[metric]
        
        return comparison
    
    def plot_component_performance(self, component_id: str, 
                                 output_file: Optional[str] = None) -> Any:
        """
        Plot component performance
        
        Args:
            component_id: Component ID
            output_file: Output file path for plot
            
        Returns:
            Plot figure
        """
        # Get metrics
        metrics = self.get_component_metrics(component_id)
        if not metrics:
            return None
        
        # Calculate cumulative metrics
        metrics.calculate_cumulative_metrics()
        
        # Get data
        cumulative = metrics.cumulative_metrics
        
        if not cumulative or 'timestamps' not in cumulative or 'cumulative_profit' not in cumulative:
            return None
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Cumulative Profit (%)", "Rolling Win Rate"),
            vertical_spacing=0.1
        )
        
        # Add traces
        timestamps = pd.to_datetime(cumulative['timestamps'])
        
        # Cumulative profit
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cumulative['cumulative_profit'],
                mode='lines',
                name='Cumulative Profit'
            ),
            row=1, col=1
        )
        
        # Rolling win rate
        if 'rolling_win_rate' in cumulative:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cumulative['rolling_win_rate'],
                    mode='lines',
                    name='Win Rate',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Component Performance: {component_id}",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save to file if specified
        if output_file:
            fig.write_image(output_file)
        
        return fig
    
    def analyze_component_portfolio(self, component_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze portfolio of components
        
        Args:
            component_ids: List of component IDs
            
        Returns:
            Portfolio analysis
        """
        # Get metrics for all components
        components = {}
        for component_id in component_ids:
            metrics = self.get_component_metrics(component_id)
            if metrics:
                metrics.calculate_metrics()
                metrics.calculate_cumulative_metrics()
                components[component_id] = metrics
        
        if not components:
            return {'error': "No metrics found for any component"}
        
        # Analyze portfolio
        portfolio = {
            'summary': {},
            'components': {},
            'correlation': {}
        }
        
        # Process each component
        for component_id, metrics in components.items():
            if metrics.component_type != ComponentType.SIGNAL_GENERATOR:
                continue
                
            # Get cumulative profit
            if 'cumulative_profit' in metrics.cumulative_metrics:
                portfolio['components'][component_id] = {
                    'profit': metrics.cumulative_metrics['cumulative_profit'][-1] if metrics.cumulative_metrics['cumulative_profit'] else 0,
                    'win_rate': metrics.performance_metrics.get('win_rate', 0),
                    'profit_factor': metrics.performance_metrics.get('profit_factor', 0),
                    'trade_count': metrics.performance_metrics.get('trade_count', 0)
                }
        
        # Calculate correlations
        profit_series = {}
        
        for component_id, metrics in components.items():
            if metrics.component_type != ComponentType.SIGNAL_GENERATOR:
                continue
                
            # Get trade profits as series
            trades = pd.DataFrame(metrics.trades_history)
            if 'profit_pct' in trades.columns and 'exit_time' in trades.columns:
                trades['exit_time'] = pd.to_datetime(trades['exit_time'])
                trades.set_index('exit_time', inplace=True)
                trades.sort_index(inplace=True)
                
                # Resample to daily
                daily = trades['profit_pct'].resample('D').sum()
                profit_series[component_id] = daily
        
        # Calculate correlation matrix
        if profit_series:
            # Combine series
            df = pd.DataFrame(profit_series)
            df.fillna(0, inplace=True)
            
            # Calculate correlation
            corr = df.corr()
            
            # Convert to dictionary
            for comp1 in corr.index:
                portfolio['correlation'][comp1] = {}
                for comp2 in corr.columns:
                    portfolio['correlation'][comp1][comp2] = corr.loc[comp1, comp2]
        
        # Calculate portfolio summary
        portfolio['summary'] = {
            'component_count': len(portfolio['components']),
            'avg_win_rate': np.mean([comp.get('win_rate', 0) for comp in portfolio['components'].values()]),
            'avg_profit_factor': np.mean([comp.get('profit_factor', 0) for comp in portfolio['components'].values()]),
            'total_trades': sum(comp.get('trade_count', 0) for comp in portfolio['components'].values())
        }
        
        return portfolio
