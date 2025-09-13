#!/usr/bin/env python
"""
Paper Trading Dashboard

This module provides visualization components for the paper trading system,
displaying performance metrics, position history, regime changes, and other
key indicators for the adaptive trading strategy.
"""

import os
import sys
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Configure paths - needed to handle relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import trading system components
from trading_bot.dashboard.strategy_signal_visualization import get_visualizer
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.execution.adaptive_paper_integration import AdaptivePaperTrading, get_paper_trading_instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'paper_dashboard.log'))
    ]
)

logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class PaperTradingDashboard:
    """
    Dashboard for paper trading performance visualization
    
    This class provides methods to generate various visualizations for 
    paper trading performance, including equity curves, trades, 
    regime changes, and strategy weights.
    """
    
    def __init__(self, results_dir: str = 'results/paper_trading'):
        """
        Initialize the paper trading dashboard
        
        Args:
            results_dir: Directory where results are stored
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.visualizer = get_visualizer()
        self.paper_trading = None
        self.controller = None
        
        # Performance metrics
        self.equity_history = None
        self.trade_history = None
        self.regime_history = None
        self.weight_history = None
        
        # Time range for analysis
        self.start_date = None
        self.end_date = None
        
    def load_data(self, session_id: Optional[str] = None):
        """
        Load trading data from results directory
        
        Args:
            session_id: Specific session ID to load, or most recent if None
        """
        # Find most recent session if not specified
        if session_id is None:
            sessions = self._get_available_sessions()
            if not sessions:
                logger.warning("No trading sessions found")
                return False
            
            # Get most recent session
            session_id = sessions[-1]
        
        logger.info(f"Loading data for session {session_id}")
        
        # Load equity history
        equity_file = os.path.join(self.results_dir, f"{session_id}_equity.csv")
        if os.path.exists(equity_file):
            self.equity_history = pd.read_csv(equity_file, parse_dates=['timestamp'])
            self.start_date = self.equity_history['timestamp'].min()
            self.end_date = self.equity_history['timestamp'].max()
        
        # Load trade history
        trades_file = os.path.join(self.results_dir, f"{session_id}_trades.csv")
        if os.path.exists(trades_file):
            self.trade_history = pd.read_csv(trades_file, parse_dates=['entry_time', 'exit_time'])
        
        # Load regime history
        regime_file = os.path.join(self.results_dir, f"{session_id}_regimes.csv")
        if os.path.exists(regime_file):
            self.regime_history = pd.read_csv(regime_file, parse_dates=['timestamp'])
        
        # Load weight history
        weights_file = os.path.join(self.results_dir, f"{session_id}_weights.csv")
        if os.path.exists(weights_file):
            self.weight_history = pd.read_csv(weights_file, parse_dates=['timestamp'])
        
        return True
    
    def load_live_data(self):
        """Load data from currently running paper trading system"""
        if not self.paper_trading:
            self.paper_trading = get_paper_trading_instance()
            
        if not self.controller:
            self.controller = self.paper_trading.get_controller()
            
        if not self.controller:
            logger.warning("No active controller found")
            return False
        
        # Get data from paper trading instance
        self.equity_history = self.paper_trading.get_equity_history()
        self.trade_history = self.paper_trading.get_trade_history()
        
        # Get data from controller
        self.regime_history = self.controller.get_regime_history()
        self.weight_history = self.controller.get_weight_history()
        
        # Set time range
        if self.equity_history is not None and not self.equity_history.empty:
            self.start_date = self.equity_history['timestamp'].min()
            self.end_date = self.equity_history['timestamp'].max()
        
        return True
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate performance summary metrics
        
        Returns:
            Dictionary with key performance metrics
        """
        if self.equity_history is None or self.equity_history.empty:
            logger.warning("No equity data available for performance summary")
            return {}
        
        # Initialize metrics
        metrics = {}
        
        # Basic metrics
        initial_equity = self.equity_history['equity'].iloc[0]
        final_equity = self.equity_history['equity'].iloc[-1]
        metrics['starting_equity'] = initial_equity
        metrics['ending_equity'] = final_equity
        metrics['total_return'] = (final_equity / initial_equity - 1) * 100  # percentage
        
        # Trading period
        metrics['start_date'] = self.start_date
        metrics['end_date'] = self.end_date
        trading_days = (self.end_date - self.start_date).days
        metrics['trading_days'] = trading_days if trading_days > 0 else 1
        
        # Annualized return
        if trading_days > 0:
            ann_factor = 252 / trading_days
            metrics['annualized_return'] = ((final_equity / initial_equity) ** ann_factor - 1) * 100
        else:
            metrics['annualized_return'] = 0
        
        # Drawdown analysis
        self.equity_history['peak'] = self.equity_history['equity'].cummax()
        self.equity_history['drawdown'] = (self.equity_history['equity'] / self.equity_history['peak'] - 1) * 100
        metrics['max_drawdown'] = self.equity_history['drawdown'].min()
        
        # Volatility (annualized)
        if len(self.equity_history) > 1:
            self.equity_history['return'] = self.equity_history['equity'].pct_change()
            daily_volatility = self.equity_history['return'].std()
            metrics['daily_volatility'] = daily_volatility * 100  # percentage
            metrics['annualized_volatility'] = daily_volatility * np.sqrt(252) * 100
        else:
            metrics['daily_volatility'] = 0
            metrics['annualized_volatility'] = 0
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        if metrics['annualized_volatility'] != 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Trade statistics
        if self.trade_history is not None and not self.trade_history.empty:
            metrics['total_trades'] = len(self.trade_history)
            
            # Win rate
            winning_trades = self.trade_history[self.trade_history['pnl'] > 0]
            metrics['winning_trades'] = len(winning_trades)
            metrics['win_rate'] = len(winning_trades) / len(self.trade_history) * 100
            
            # Average P&L
            metrics['avg_pnl'] = self.trade_history['pnl'].mean()
            
            # Profit factor
            gross_profit = self.trade_history[self.trade_history['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(self.trade_history[self.trade_history['pnl'] < 0]['pnl'].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Average trade duration
            self.trade_history['duration'] = (self.trade_history['exit_time'] - self.trade_history['entry_time'])
            metrics['avg_duration'] = self.trade_history['duration'].mean()
            
        else:
            metrics['total_trades'] = 0
            metrics['winning_trades'] = 0
            metrics['win_rate'] = 0
            metrics['avg_pnl'] = 0
            metrics['profit_factor'] = 0
            metrics['avg_duration'] = 0
        
        # Regime statistics
        if self.regime_history is not None and not self.regime_history.empty:
            regime_counts = self.regime_history['regime'].value_counts()
            metrics['regime_distribution'] = regime_counts.to_dict()
            metrics['regime_changes'] = (self.regime_history['regime'] != self.regime_history['regime'].shift(1)).sum()
        else:
            metrics['regime_distribution'] = {}
            metrics['regime_changes'] = 0
        
        return metrics
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate equity curve visualization
        
        Args:
            save_path: Path to save the figure, or None to return it
            
        Returns:
            Matplotlib figure object
        """
        if self.equity_history is None or self.equity_history.empty:
            logger.warning("No equity data available for equity curve")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No equity data available", ha='center', va='center')
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(self.equity_history['timestamp'], self.equity_history['equity'], 
                linewidth=2, label='Equity')
        
        # Plot drawdowns
        if 'drawdown' in self.equity_history.columns:
            # Create twin axis for drawdown
            ax2 = ax.twinx()
            ax2.fill_between(self.equity_history['timestamp'], 
                            self.equity_history['drawdown'], 
                            0, 
                            color='red', 
                            alpha=0.3, 
                            label='Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim(self.equity_history['drawdown'].min() * 1.5, 5)  # Some buffer
        
        # Add markers for trades if available
        if self.trade_history is not None and not self.trade_history.empty:
            # Winning trades in green
            winning_trades = self.trade_history[self.trade_history['pnl'] > 0]
            if not winning_trades.empty:
                ax.scatter(winning_trades['exit_time'], 
                          winning_trades['exit_price'], 
                          marker='^', 
                          color='green', 
                          s=50, 
                          label='Winning Trade')
            
            # Losing trades in red
            losing_trades = self.trade_history[self.trade_history['pnl'] <= 0]
            if not losing_trades.empty:
                ax.scatter(losing_trades['exit_time'], 
                          losing_trades['exit_price'], 
                          marker='v', 
                          color='red', 
                          s=50, 
                          label='Losing Trade')
        
        # Add markers for regime changes if available
        if self.regime_history is not None and not self.regime_history.empty:
            # Get points where regime changes
            regime_changes = self.regime_history[self.regime_history['regime'] != self.regime_history['regime'].shift(1)]
            
            if not regime_changes.empty:
                # Find corresponding equity values by timestamp matching
                for idx, row in regime_changes.iterrows():
                    # Find nearest equity timestamp
                    nearest_idx = (self.equity_history['timestamp'] - row['timestamp']).abs().argmin()
                    equity_val = self.equity_history['equity'].iloc[nearest_idx]
                    
                    # Plot vertical line for regime change
                    ax.axvline(x=row['timestamp'], color='purple', linestyle='--', alpha=0.7)
                    
                    # Add text label for regime
                    ax.text(row['timestamp'], equity_val * 1.02, 
                           f"R{row['regime']}", 
                           horizontalalignment='center', 
                           fontsize=9)
        
        # Configure plot
        ax.set_title('Paper Trading Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.grid(True)
        ax.legend(loc='upper left')
        
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_weight_evolution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate strategy weight evolution visualization
        
        Args:
            save_path: Path to save the figure, or None to return it
            
        Returns:
            Matplotlib figure object
        """
        if self.weight_history is None or self.weight_history.empty:
            logger.warning("No weight data available for weight evolution")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No weight data available", ha='center', va='center')
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get strategy columns (all except timestamp)
        strategy_cols = [col for col in self.weight_history.columns if col != 'timestamp']
        
        # Plot each strategy's weight
        for strategy in strategy_cols:
            ax.plot(self.weight_history['timestamp'], 
                   self.weight_history[strategy], 
                   linewidth=2, 
                   label=strategy)
        
        # Add regime change markers if available
        if self.regime_history is not None and not self.regime_history.empty:
            # Get points where regime changes
            regime_changes = self.regime_history[self.regime_history['regime'] != self.regime_history['regime'].shift(1)]
            
            if not regime_changes.empty:
                for idx, row in regime_changes.iterrows():
                    # Plot vertical line for regime change
                    ax.axvline(x=row['timestamp'], color='purple', linestyle='--', alpha=0.7)
                    
                    # Add text label for regime
                    max_weight = self.weight_history[strategy_cols].max().max()
                    ax.text(row['timestamp'], max_weight * 1.05, 
                           f"Regime {row['regime']}", 
                           horizontalalignment='center', 
                           fontsize=9)
        
        # Configure plot
        ax.set_title('Strategy Weight Evolution')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.grid(True)
        ax.legend(loc='upper left')
        
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_trade_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate trade summary visualization
        
        Args:
            save_path: Path to save the figure, or None to return it
            
        Returns:
            Matplotlib figure object
        """
        if self.trade_history is None or self.trade_history.empty:
            logger.warning("No trade data available for trade summary")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trade data available", ha='center', va='center')
            return fig
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. P&L Distribution
        axs[0, 0].hist(self.trade_history['pnl'], bins=20, color='navy', alpha=0.7)
        axs[0, 0].axvline(x=0, color='red', linestyle='--')
        axs[0, 0].set_title('P&L Distribution')
        axs[0, 0].set_xlabel('P&L ($)')
        axs[0, 0].set_ylabel('Frequency')
        
        # 2. Cumulative P&L
        self.trade_history['cumulative_pnl'] = self.trade_history['pnl'].cumsum()
        axs[0, 1].plot(range(len(self.trade_history)), 
                       self.trade_history['cumulative_pnl'], 
                       color='green', 
                       linewidth=2)
        axs[0, 1].set_title('Cumulative P&L')
        axs[0, 1].set_xlabel('Trade #')
        axs[0, 1].set_ylabel('Cumulative P&L ($)')
        
        # 3. Trade Duration Distribution
        if 'duration' in self.trade_history.columns:
            # Convert to hours for better visualization
            trade_duration_hours = self.trade_history['duration'].dt.total_seconds() / 3600
            axs[1, 0].hist(trade_duration_hours, bins=20, color='purple', alpha=0.7)
            axs[1, 0].set_title('Trade Duration Distribution')
            axs[1, 0].set_xlabel('Duration (hours)')
            axs[1, 0].set_ylabel('Frequency')
        
        # 4. Win/Loss by Symbol
        if 'symbol' in self.trade_history.columns:
            # Calculate win rate by symbol
            symbol_results = {}
            for symbol in self.trade_history['symbol'].unique():
                symbol_trades = self.trade_history[self.trade_history['symbol'] == symbol]
                wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                losses = len(symbol_trades[symbol_trades['pnl'] <= 0])
                win_rate = wins / len(symbol_trades) * 100 if len(symbol_trades) > 0 else 0
                symbol_results[symbol] = {
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate
                }
            
            # Plot results
            symbols = list(symbol_results.keys())
            win_rates = [symbol_results[s]['win_rate'] for s in symbols]
            
            bars = axs[1, 1].bar(symbols, win_rates, color='teal', alpha=0.7)
            axs[1, 1].set_title('Win Rate by Symbol')
            axs[1, 1].set_xlabel('Symbol')
            axs[1, 1].set_ylabel('Win Rate (%)')
            axs[1, 1].axhline(y=50, color='red', linestyle='--')
            axs[1, 1].set_ylim(0, 100)
            
            # Add trade counts as text
            for i, bar in enumerate(bars):
                height = bar.get_height()
                total_trades = symbol_results[symbols[i]]['wins'] + symbol_results[symbols[i]]['losses']
                axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                              f'{total_trades}', ha='center', va='bottom', rotation=0)
        
        fig.suptitle('Trade Summary Analysis', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # Save if requested
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_fill_quality_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate trade execution/fill quality visualization
        
        Args:
            save_path: Path to save the figure, or None to return it
            
        Returns:
            Matplotlib figure object
        """
        if self.trade_history is None or self.trade_history.empty:
            logger.warning("No trade data available for fill quality analysis")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trade data available", ha='center', va='center')
            return fig
        
        # Check if slippage data is available
        slippage_available = all(col in self.trade_history.columns 
                               for col in ['intended_entry_price', 'entry_price',
                                          'intended_exit_price', 'exit_price'])
        
        if not slippage_available:
            logger.warning("Slippage data not available for fill quality analysis")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Slippage data not available", ha='center', va='center')
            return fig
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate slippage
        self.trade_history['entry_slippage_pct'] = (self.trade_history['entry_price'] - 
                                                   self.trade_history['intended_entry_price']) / self.trade_history['intended_entry_price'] * 100
        self.trade_history['exit_slippage_pct'] = (self.trade_history['exit_price'] - 
                                                  self.trade_history['intended_exit_price']) / self.trade_history['intended_exit_price'] * 100
        
        # For short positions, invert the slippage calculation
        if 'direction' in self.trade_history.columns:
            short_positions = self.trade_history['direction'] == 'short'
            self.trade_history.loc[short_positions, 'entry_slippage_pct'] *= -1
            self.trade_history.loc[short_positions, 'exit_slippage_pct'] *= -1
        
        # 1. Entry Slippage Distribution
        axs[0, 0].hist(self.trade_history['entry_slippage_pct'], bins=20, color='blue', alpha=0.7)
        axs[0, 0].axvline(x=0, color='red', linestyle='--')
        axs[0, 0].set_title('Entry Slippage Distribution')
        axs[0, 0].set_xlabel('Entry Slippage (%)')
        axs[0, 0].set_ylabel('Frequency')
        
        # 2. Exit Slippage Distribution
        axs[0, 1].hist(self.trade_history['exit_slippage_pct'], bins=20, color='green', alpha=0.7)
        axs[0, 1].axvline(x=0, color='red', linestyle='--')
        axs[0, 1].set_title('Exit Slippage Distribution')
        axs[0, 1].set_xlabel('Exit Slippage (%)')
        axs[0, 1].set_ylabel('Frequency')
        
        # 3. Slippage vs. Time of Day (if timestamp available)
        if 'entry_time' in self.trade_history.columns:
            entry_hour = self.trade_history['entry_time'].dt.hour + self.trade_history['entry_time'].dt.minute / 60
            axs[1, 0].scatter(entry_hour, self.trade_history['entry_slippage_pct'], alpha=0.5, color='purple')
            axs[1, 0].set_title('Entry Slippage vs. Time of Day')
            axs[1, 0].set_xlabel('Hour of Day')
            axs[1, 0].set_ylabel('Entry Slippage (%)')
            axs[1, 0].axhline(y=0, color='red', linestyle='--')
            axs[1, 0].set_xlim(9, 16)  # Trading hours
        
        # 4. Slippage Impact on P&L
        total_slippage = self.trade_history['entry_slippage_pct'] + self.trade_history['exit_slippage_pct']
        axs[1, 1].scatter(total_slippage, self.trade_history['pnl'], alpha=0.5, color='orange')
        axs[1, 1].set_title('Slippage Impact on P&L')
        axs[1, 1].set_xlabel('Total Slippage (%)')
        axs[1, 1].set_ylabel('P&L ($)')
        axs[1, 1].axhline(y=0, color='red', linestyle='--')
        axs[1, 1].axvline(x=0, color='red', linestyle='--')
        
        fig.suptitle('Fill Quality Analysis', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # Save if requested
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def generate_dashboard(self, output_dir: Optional[str] = None):
        """
        Generate a complete dashboard with all visualizations
        
        Args:
            output_dir: Directory to save dashboard files
        """
        if output_dir is None:
            output_dir = 'dashboard/paper_trading'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate performance summary
        metrics = self.generate_performance_summary()
        with open(os.path.join(output_dir, 'performance_summary.json'), 'w') as f:
            # Convert non-serializable objects to strings
            for key, value in metrics.items():
                if isinstance(value, (datetime, timedelta, pd.Timestamp)):
                    metrics[key] = str(value)
            json.dump(metrics, f, indent=4)
        
        # Generate visualizations
        self.plot_equity_curve(os.path.join(output_dir, 'equity_curve.png'))
        self.plot_weight_evolution(os.path.join(output_dir, 'weight_evolution.png'))
        self.plot_trade_summary(os.path.join(output_dir, 'trade_summary.png'))
        self.plot_fill_quality_analysis(os.path.join(output_dir, 'fill_quality.png'))
        
        logger.info(f"Dashboard generated in {output_dir}")
        
        # Return path to dashboard
        return os.path.abspath(output_dir)
    
    def _get_available_sessions(self) -> List[str]:
        """
        Get list of available trading sessions
        
        Returns:
            List of session IDs, sorted by date
        """
        if not os.path.exists(self.results_dir):
            return []
        
        # Get all equity files
        files = [f for f in os.listdir(self.results_dir) if f.endswith('_equity.csv')]
        
        # Extract session IDs
        sessions = [f.replace('_equity.csv', '') for f in files]
        
        # Sort by date (assuming session ID starts with date in format YYYYMMDD)
        sessions.sort()
        
        return sessions

def get_dashboard_instance():
    """Get or create a dashboard instance"""
    return PaperTradingDashboard()

def generate_latest_dashboard():
    """Generate dashboard for the most recent session"""
    dashboard = get_dashboard_instance()
    
    # Try to load live data first
    if not dashboard.load_live_data():
        # If no live data, load most recent session
        dashboard.load_data()
    
    # Generate dashboard
    dashboard_path = dashboard.generate_dashboard()
    
    return dashboard_path

if __name__ == "__main__":
    # Generate dashboard for most recent session
    dashboard_path = generate_latest_dashboard()
    print(f"Dashboard generated at: {dashboard_path}")
