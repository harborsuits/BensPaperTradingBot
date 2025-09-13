"""
Strategy Signal Visualization Dashboard

This module provides a dashboard that visualizes:
1. Trading positions plotted against regime detector signals
2. Strategy performance metrics alongside market regimes
3. Snowball weight evolution charts plotted with P&L curves
"""

import logging
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.execution.adaptive_paper_integration import get_paper_trading_instance
from trading_bot.execution.realtime_data_feed import get_realtime_feed_instance
from trading_bot.dashboard.enhanced_alerting import get_alert_monitor

logger = logging.getLogger(__name__)

class StrategySignalVisualizer:
    """
    Creates visualizations that overlay trading positions with regime detector signals
    and plot snowball weight evolution alongside P&L performance.
    """
    
    def __init__(self, output_dir: str = './dashboard/visualizations'):
        """Initialize the strategy signal visualizer"""
        self.output_dir = output_dir
        self.controller = None
        self.position_history = []
        self.regime_history = []
        self.weight_history = []
        self.performance_history = []
        self.signal_history = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def set_controller(self, controller: AdaptiveStrategyController):
        """Set the adaptive strategy controller reference"""
        self.controller = controller
        logger.info(f"StrategySignalVisualizer now monitoring AdaptiveStrategyController")
    
    def record_current_state(self):
        """Record the current state of the system for visualization"""
        if not self.controller:
            logger.warning("No controller set, cannot record state")
            return
        
        # Get current timestamp
        now = datetime.now()
        
        # Get paper trading instance
        paper_trading = get_paper_trading_instance()
        positions = None
        
        if paper_trading and paper_trading.paper_adapter:
            # Get current positions
            positions = paper_trading.paper_adapter.get_positions()
        
        # Record position data
        if positions:
            for symbol, position in positions.items():
                position_data = {
                    'timestamp': now,
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'market_value': position.market_value,
                    'average_price': position.avg_price,
                    'unrealized_pnl': position.unrealized_pnl
                }
                self.position_history.append(position_data)
        
        # Record regime data
        if hasattr(self.controller, 'market_regime_detector'):
            regime_data = self.controller.market_regime_detector.get_current_regime()
            if regime_data:
                regime_record = {
                    'timestamp': now,
                    'regime_type': regime_data.get('regime_type'),
                    'confidence': regime_data.get('confidence', 0),
                    'volatility': regime_data.get('volatility', 0),
                    'trend_strength': regime_data.get('trend_strength', 0)
                }
                self.regime_history.append(regime_record)
        
        # Record allocation weights
        if hasattr(self.controller, 'snowball_allocator'):
            weights = self.controller.snowball_allocator.get_current_allocation()
            if weights:
                weight_record = {
                    'timestamp': now,
                    'weights': weights.copy()
                }
                self.weight_history.append(weight_record)
        
        # Record performance metrics
        if hasattr(self.controller, 'performance_tracker'):
            metrics = self.controller.performance_tracker.get_metrics()
            if metrics:
                performance_record = {
                    'timestamp': now,
                    'equity': metrics.get('equity', 0),
                    'cash': metrics.get('cash', 0),
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0)
                }
                self.performance_history.append(performance_record)
        
        # Record strategy signals
        strategies = self.controller.get_active_strategies()
        for strategy_id, strategy in strategies.items():
            # Check if strategy has signal generation method
            if hasattr(strategy, 'get_current_signals'):
                signals = strategy.get_current_signals()
                if signals:
                    if strategy_id not in self.signal_history:
                        self.signal_history[strategy_id] = []
                    
                    signal_record = {
                        'timestamp': now,
                        'signals': signals.copy()
                    }
                    self.signal_history[strategy_id].append(signal_record)
    
    def generate_position_vs_regime_chart(self, 
                                       symbol: str, 
                                       days: int = 30,
                                       show_signals: bool = True) -> str:
        """
        Generate a chart showing positions vs market regime for a specific symbol
        
        Args:
            symbol: Trading symbol to visualize
            days: Number of days of history to show
            show_signals: Whether to overlay strategy signals
            
        Returns:
            Path to the generated chart file
        """
        # Filter position history for this symbol
        start_date = datetime.now() - timedelta(days=days)
        
        symbol_positions = [
            p for p in self.position_history
            if p['symbol'] == symbol and p['timestamp'] >= start_date
        ]
        
        regime_data = [
            r for r in self.regime_history
            if r['timestamp'] >= start_date
        ]
        
        if not symbol_positions or not regime_data:
            logger.warning(f"Insufficient data to generate position vs regime chart for {symbol}")
            return None
        
        # Create DataFrames
        position_df = pd.DataFrame(symbol_positions)
        regime_df = pd.DataFrame(regime_data)
        
        # Setup the plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot 1: Position Market Value
        position_df.plot(
            x='timestamp', 
            y='market_value',
            ax=axes[0],
            color='blue',
            label=f'{symbol} Position'
        )
        
        # Add regime background coloring
        prev_timestamp = None
        prev_regime = None
        regime_colors = {
            'bull': 'lightgreen',
            'bear': 'lightcoral',
            'neutral': 'lightyellow',
            'volatile': 'lightblue',
            'trending': 'lightgreen',
            'ranging': 'lightyellow',
            'unknown': 'lightgray'
        }
        
        for _, row in regime_df.iterrows():
            timestamp = row['timestamp']
            regime = row['regime_type']
            confidence = row['confidence']
            
            if prev_timestamp is not None and prev_regime is not None:
                color = regime_colors.get(prev_regime.lower(), 'lightgray')
                alpha = min(1.0, max(0.2, prev_confidence))
                axes[0].axvspan(prev_timestamp, timestamp, color=color, alpha=alpha, zorder=0)
            
            prev_timestamp = timestamp
            prev_regime = regime
            prev_confidence = confidence
        
        # Plot 2: Regime Confidence
        regime_df.plot(
            x='timestamp',
            y='confidence',
            ax=axes[1],
            color='purple',
            label='Regime Confidence'
        )
        
        # Plot 3: Strategy signals if requested
        if show_signals:
            # Find strategies that have signals for this symbol
            symbol_signals = {}
            
            for strategy_id, signals in self.signal_history.items():
                filtered_signals = []
                
                for signal_record in signals:
                    timestamp = signal_record['timestamp']
                    if timestamp < start_date:
                        continue
                    
                    # Check if signal contains data for this symbol
                    signal_data = signal_record['signals'].get(symbol)
                    if signal_data:
                        filtered_signals.append({
                            'timestamp': timestamp,
                            'signal': signal_data.get('signal', 0),
                            'strength': signal_data.get('strength', 0),
                            'direction': signal_data.get('direction', 'neutral')
                        })
                
                if filtered_signals:
                    symbol_signals[strategy_id] = pd.DataFrame(filtered_signals)
            
            # Plot signals from each strategy
            for strategy_id, signal_df in symbol_signals.items():
                signal_df.plot(
                    x='timestamp',
                    y='signal',
                    ax=axes[2],
                    label=f'{strategy_id} Signal'
                )
                
                # Additional direction markers
                buy_points = signal_df[signal_df['direction'] == 'buy']
                sell_points = signal_df[signal_df['direction'] == 'sell']
                
                if not buy_points.empty:
                    axes[2].scatter(
                        buy_points['timestamp'],
                        buy_points['signal'],
                        color='green',
                        marker='^',
                        s=100,
                        label=f'{strategy_id} Buy'
                    )
                
                if not sell_points.empty:
                    axes[2].scatter(
                        sell_points['timestamp'],
                        sell_points['signal'],
                        color='red',
                        marker='v',
                        s=100,
                        label=f'{strategy_id} Sell'
                    )
        
        # Format the plot
        axes[0].set_title(f"{symbol} Position vs Market Regime")
        axes[0].set_ylabel("Position Value ($)")
        axes[0].grid(True)
        
        axes[1].set_ylabel("Regime Confidence")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True)
        
        axes[2].set_ylabel("Signal Strength")
        axes[2].set_xlabel("Date")
        axes[2].grid(True)
        
        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
        
        plt.tight_layout()
        
        # Save the chart
        filename = f"{symbol}_position_vs_regime_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        
        logger.info(f"Generated position vs regime chart for {symbol}: {filepath}")
        return filepath
    
    def generate_weight_evolution_chart(self, days: int = 30) -> str:
        """
        Generate a chart showing snowball weight evolution alongside P&L
        
        Args:
            days: Number of days of history to show
            
        Returns:
            Path to the generated chart file
        """
        start_date = datetime.now() - timedelta(days=days)
        
        weight_data = [w for w in self.weight_history if w['timestamp'] >= start_date]
        performance_data = [p for p in self.performance_history if p['timestamp'] >= start_date]
        
        if not weight_data or not performance_data:
            logger.warning("Insufficient data to generate weight evolution chart")
            return None
        
        # Process weight data to create a DataFrame
        weight_records = []
        for record in weight_data:
            timestamp = record['timestamp']
            weights = record['weights']
            
            for strategy_id, weight in weights.items():
                weight_records.append({
                    'timestamp': timestamp,
                    'strategy_id': strategy_id,
                    'weight': weight
                })
        
        weight_df = pd.DataFrame(weight_records)
        performance_df = pd.DataFrame(performance_data)
        
        # Create pivot table for strategy weights
        pivot_df = weight_df.pivot(index='timestamp', columns='strategy_id', values='weight')
        pivot_df = pivot_df.reset_index()
        
        # Setup the plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Strategy Weight Evolution
        strategies = pivot_df.columns[1:]  # Skip timestamp column
        bottom = np.zeros(len(pivot_df))
        
        for strategy in strategies:
            axes[0].fill_between(
                pivot_df['timestamp'],
                bottom,
                bottom + pivot_df[strategy].fillna(0),
                label=strategy
            )
            bottom += pivot_df[strategy].fillna(0)
        
        # Plot 2: Equity Curve
        performance_df.plot(
            x='timestamp',
            y='equity',
            ax=axes[1],
            color='green',
            label='Portfolio Equity'
        )
        
        # Format the plot
        axes[0].set_title("Strategy Weight Evolution")
        axes[0].set_ylabel("Weight Allocation")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True)
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        axes[1].set_title("Portfolio Performance")
        axes[1].set_ylabel("Equity ($)")
        axes[1].grid(True)
        
        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
        
        plt.tight_layout()
        
        # Save the chart
        filename = f"weight_evolution_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        
        logger.info(f"Generated weight evolution chart: {filepath}")
        return filepath
    
    def generate_performance_dashboard(self) -> str:
        """
        Generate a comprehensive performance dashboard with multiple metrics
        
        Returns:
            Path to the generated dashboard file
        """
        if not self.performance_history:
            logger.warning("No performance data available for dashboard")
            return None
        
        # Create performance DataFrame
        performance_df = pd.DataFrame(self.performance_history)
        
        # Setup the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Equity Curve
        performance_df.plot(
            x='timestamp',
            y='equity',
            ax=axes[0, 0],
            color='blue',
            label='Portfolio Equity'
        )
        axes[0, 0].set_title("Portfolio Equity")
        axes[0, 0].set_ylabel("Equity ($)")
        axes[0, 0].grid(True)
        
        # Plot 2: Drawdown
        performance_df.plot(
            x='timestamp',
            y='max_drawdown',
            ax=axes[0, 1],
            color='red',
            label='Max Drawdown'
        )
        axes[0, 1].set_title("Maximum Drawdown")
        axes[0, 1].set_ylabel("Drawdown (%)")
        axes[0, 1].grid(True)
        
        # Plot 3: Sharpe Ratio
        performance_df.plot(
            x='timestamp',
            y='sharpe_ratio',
            ax=axes[1, 0],
            color='purple',
            label='Sharpe Ratio'
        )
        axes[1, 0].set_title("Sharpe Ratio Evolution")
        axes[1, 0].set_ylabel("Sharpe Ratio")
        axes[1, 0].grid(True)
        
        # Plot 4: Win Rate
        performance_df.plot(
            x='timestamp',
            y='win_rate',
            ax=axes[1, 1],
            color='green',
            label='Win Rate'
        )
        axes[1, 1].set_title("Win Rate Evolution")
        axes[1, 1].set_ylabel("Win Rate (%)")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True)
        
        # Format the plot
        plt.tight_layout()
        
        # Save the dashboard
        filename = f"performance_dashboard_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        
        logger.info(f"Generated performance dashboard: {filepath}")
        return filepath
    
    def save_data(self):
        """Save collected data to files for later analysis"""
        # Save position history
        if self.position_history:
            position_file = os.path.join(self.output_dir, 'position_history.json')
            self._save_to_json(self.position_history, position_file)
        
        # Save regime history
        if self.regime_history:
            regime_file = os.path.join(self.output_dir, 'regime_history.json')
            self._save_to_json(self.regime_history, regime_file)
        
        # Save weight history
        if self.weight_history:
            weight_file = os.path.join(self.output_dir, 'weight_history.json')
            self._save_to_json(self.weight_history, weight_file)
        
        # Save performance history
        if self.performance_history:
            performance_file = os.path.join(self.output_dir, 'performance_history.json')
            self._save_to_json(self.performance_history, performance_file)
        
        # Save signal history
        if self.signal_history:
            signal_file = os.path.join(self.output_dir, 'signal_history.json')
            self._save_to_json(self.signal_history, signal_file)
    
    def _save_to_json(self, data, filepath):
        """Helper method to save data to JSON file"""
        try:
            # Convert any datetime objects to strings
            serializable_data = self._make_serializable(data)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
    
    def _make_serializable(self, data):
        """Make data JSON-serializable by converting datetime objects to strings"""
        if isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = self._make_serializable(value)
            return result
        
        else:
            return data

# Singleton instance
_visualizer_instance = None

def get_visualizer() -> StrategySignalVisualizer:
    """Get the global visualizer instance"""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = StrategySignalVisualizer()
    return _visualizer_instance

# Usage example (if run as script)
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create visualizer instance
    visualizer = StrategySignalVisualizer()
    
    # Generate sample data
    print("Visualizer created. Would need a controller with history data to generate charts.")
    print(f"Output directory: {visualizer.output_dir}")
