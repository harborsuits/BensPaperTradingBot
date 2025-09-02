"""
Options Performance Analyzer

This module extends the performance analysis framework with options-specific metrics.
It integrates with the existing PerformanceAnalyzer class to provide detailed analysis
of options trading strategies, Greek exposures, and IV-based metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import os
import json
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class OptionsPerformanceAnalyzer:
    """
    Specialized analyzer for options trading performance metrics.
    
    This class extends the base performance analysis with options-specific 
    metrics such as Greek exposures, IV percentiles, and strategy-specific 
    performance tracking.
    """
    
    def __init__(self, performance_analyzer, options_risk_manager=None, multi_asset_adapter=None, db_connector=None):
        """
        Initialize the options performance analyzer.
        
        Args:
            performance_analyzer: Reference to the main performance analyzer
            options_risk_manager: Reference to the options risk management system
            multi_asset_adapter: Reference to the multi-asset adapter for cross-asset analysis
            db_connector: Database connector for storing and retrieving trade data
        """
        self.performance_analyzer = performance_analyzer
        self.options_risk_manager = options_risk_manager
        self.multi_asset_adapter = multi_asset_adapter
        self.db_connector = db_connector
        self.trades_history = []
        self.greeks_history = []
        self.daily_greeks_snapshots = []
        self.iv_metrics_history = []
        
        # Initialize database tables if connector provided
        if self.db_connector:
            self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables for options performance data if they don't exist"""
        if not self.db_connector:
            return
            
        try:
            # Create options trades table
            self.db_connector.execute("""
                CREATE TABLE IF NOT EXISTS options_trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol TEXT,
                    underlying TEXT,
                    option_type TEXT,
                    strike REAL,
                    expiration DATE,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    side TEXT,
                    strategy TEXT,
                    iv_percentile REAL,
                    iv_rank REAL,
                    pnl REAL,
                    exit_timestamp TIMESTAMP,
                    days_held INTEGER,
                    dte_entry INTEGER,
                    dte_exit INTEGER,
                    greeks_json TEXT,
                    execution_quality_json TEXT
                )
            """)
            
            # Create greeks snapshots table
            self.db_connector.execute("""
                CREATE TABLE IF NOT EXISTS greeks_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    account_value REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    rho REAL,
                    delta_dollars REAL,
                    gamma_dollars REAL,
                    theta_dollars REAL,
                    vega_dollars REAL,
                    rho_dollars REAL
                )
            """)
            
            logger.info("Options performance database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def add_options_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Record an options trade with associated Greeks and IV data
        
        Args:
            trade_data: Dictionary containing trade details including:
                - trade_id: Unique identifier for the trade
                - timestamp: When the trade was executed
                - symbol: Option symbol
                - underlying: Underlying symbol
                - option_type: CALL or PUT
                - strike: Strike price
                - expiration: Expiration date
                - entry_price: Entry price per contract
                - exit_price: Exit price per contract (None if position still open)
                - quantity: Number of contracts
                - side: LONG or SHORT
                - strategy: Strategy name (e.g., VERTICAL_SPREAD, IRON_CONDOR)
                - iv_percentile: IV percentile at entry
                - iv_rank: IV rank at entry
                - greeks: Dictionary with delta, gamma, theta, vega, rho values
                - pnl: Realized profit/loss (None if position still open)
                - exit_timestamp: When the trade was closed (None if position still open)
                - days_held: Number of days position was held
                - dte_entry: Days to expiration at entry
                - dte_exit: Days to expiration at exit (None if position still open)
        """
        self.trades_history.append(trade_data)
        
        # Record Greeks snapshot at trade time if we have the options risk manager
        if self.options_risk_manager:
            greeks_snapshot = {
                'timestamp': trade_data['timestamp'],
                'trade_id': trade_data['trade_id'],
                'portfolio_delta': self.options_risk_manager.get_portfolio_greeks().delta,
                'portfolio_gamma': self.options_risk_manager.get_portfolio_greeks().gamma,
                'portfolio_theta': self.options_risk_manager.get_portfolio_greeks().theta,
                'portfolio_vega': self.options_risk_manager.get_portfolio_greeks().vega,
                'portfolio_rho': self.options_risk_manager.get_portfolio_greeks().rho
            }
            self.greeks_history.append(greeks_snapshot)
        
        # Store in database if available
        if self.db_connector:
            try:
                # Convert greeks to JSON
                greeks_json = json.dumps(trade_data.get('greeks', {}))
                exec_quality_json = json.dumps(trade_data.get('execution_quality', {}))
                
                self.db_connector.execute("""
                    INSERT OR REPLACE INTO options_trades (
                        trade_id, timestamp, symbol, underlying, option_type, strike, expiration,
                        entry_price, exit_price, quantity, side, strategy, iv_percentile, iv_rank,
                        pnl, exit_timestamp, days_held, dte_entry, dte_exit, greeks_json, execution_quality_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('trade_id'), trade_data.get('timestamp'), trade_data.get('symbol'),
                    trade_data.get('underlying'), trade_data.get('option_type'), trade_data.get('strike'),
                    trade_data.get('expiration'), trade_data.get('entry_price'), trade_data.get('exit_price'),
                    trade_data.get('quantity'), trade_data.get('side'), trade_data.get('strategy'),
                    trade_data.get('iv_percentile'), trade_data.get('iv_rank'), trade_data.get('pnl'),
                    trade_data.get('exit_timestamp'), trade_data.get('days_held'), trade_data.get('dte_entry'),
                    trade_data.get('dte_exit'), greeks_json, exec_quality_json
                ))
                logger.debug(f"Saved options trade {trade_data.get('trade_id')} to database")
            except Exception as e:
                logger.error(f"Error saving trade to database: {str(e)}")
    
    def add_daily_greeks_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Add a daily snapshot of portfolio Greeks
        
        Args:
            snapshot: Dictionary containing:
                - timestamp: Time of snapshot
                - account_value: Total account value
                - delta: Portfolio delta
                - gamma: Portfolio gamma
                - theta: Portfolio theta
                - vega: Portfolio vega
                - rho: Portfolio rho
                - delta_dollars: Delta exposure in dollars
                - theta_dollars: Theta in dollars (daily decay)
                - vega_dollars: Vega in dollars (per 1% IV change)
        """
        self.daily_greeks_snapshots.append(snapshot)
    
    def analyze_iv_metrics(self) -> Dict[str, Any]:
        """
        Analyze IV percentile and rank at trade entry and exit
        
        Returns:
            Dictionary of IV metrics analysis
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Extract IV percentiles from trades
        iv_entries = [trade.get('iv_percentile', None) for trade in self.trades_history 
                    if trade.get('iv_percentile') is not None]
        
        iv_exits = [trade.get('iv_percentile_exit', None) for trade in self.trades_history 
                   if trade.get('iv_percentile_exit') is not None]
        
        if not iv_entries:
            return {"error": "No IV data available"}
        
        # Calculate IV percentile metrics at entry
        entry_metrics = {
            'mean': np.mean(iv_entries),
            'median': np.median(iv_entries),
            'min': np.min(iv_entries),
            'max': np.max(iv_entries),
            'quartiles': np.percentile(iv_entries, [25, 50, 75]),
            'histogram': np.histogram(iv_entries, bins=10, range=(0, 1))[0].tolist()
        }
        
        # Calculate success rates by IV percentile buckets
        iv_buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bucket_names = ['Very Low (0-20%)', 'Low (20-40%)', 'Medium (40-60%)', 
                        'High (60-80%)', 'Very High (80-100%)']
        
        iv_success_rates = []
        iv_profit_factors = []
        
        for i in range(len(iv_buckets) - 1):
            lower = iv_buckets[i]
            upper = iv_buckets[i+1]
            
            # Filter trades within this IV percentile bucket
            bucket_trades = [t for t in self.trades_history 
                            if t.get('iv_percentile') is not None 
                            and lower <= t['iv_percentile'] < upper
                            and t.get('pnl') is not None]
            
            if bucket_trades:
                # Calculate success rate
                profitable_trades = [t for t in bucket_trades if t['pnl'] > 0]
                success_rate = len(profitable_trades) / len(bucket_trades)
                
                # Calculate profit factor
                winning_sum = sum(t['pnl'] for t in bucket_trades if t['pnl'] > 0)
                losing_sum = abs(sum(t['pnl'] for t in bucket_trades if t['pnl'] < 0))
                profit_factor = winning_sum / losing_sum if losing_sum > 0 else float('inf')
                
                iv_success_rates.append({
                    'bucket': bucket_names[i],
                    'success_rate': success_rate,
                    'trade_count': len(bucket_trades),
                    'profit_factor': profit_factor
                })
            else:
                iv_success_rates.append({
                    'bucket': bucket_names[i],
                    'success_rate': 0,
                    'trade_count': 0,
                    'profit_factor': 0
                })
        
        # Calculate directional bias of trades based on IV
        iv_direction_bias = []
        for i in range(len(iv_buckets) - 1):
            lower = iv_buckets[i]
            upper = iv_buckets[i+1]
            
            bucket_trades = [t for t in self.trades_history 
                            if t.get('iv_percentile') is not None 
                            and lower <= t['iv_percentile'] < upper]
            
            if bucket_trades:
                # Count long vs short trades
                long_calls = len([t for t in bucket_trades 
                                if t.get('option_type') == 'CALL' and t.get('side') == 'LONG'])
                short_calls = len([t for t in bucket_trades 
                                 if t.get('option_type') == 'CALL' and t.get('side') == 'SHORT'])
                long_puts = len([t for t in bucket_trades 
                               if t.get('option_type') == 'PUT' and t.get('side') == 'LONG'])
                short_puts = len([t for t in bucket_trades 
                                if t.get('option_type') == 'PUT' and t.get('side') == 'SHORT'])
                
                iv_direction_bias.append({
                    'bucket': bucket_names[i],
                    'long_calls': long_calls,
                    'short_calls': short_calls,
                    'long_puts': long_puts,
                    'short_puts': short_puts,
                    'long_ratio': (long_calls + long_puts) / len(bucket_trades) if len(bucket_trades) > 0 else 0,
                    'short_ratio': (short_calls + short_puts) / len(bucket_trades) if len(bucket_trades) > 0 else 0,
                })
            else:
                iv_direction_bias.append({
                    'bucket': bucket_names[i],
                    'long_calls': 0,
                    'short_calls': 0,
                    'long_puts': 0,
                    'short_puts': 0,
                    'long_ratio': 0,
                    'short_ratio': 0,
                })
        
        # Combine all metrics
        return {
            'entry_metrics': entry_metrics,
            'success_by_iv': iv_success_rates,
            'direction_bias_by_iv': iv_direction_bias
        }
    
    def analyze_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by strategy type
        
        Returns:
            Dictionary with performance metrics for each strategy
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Group trades by strategy
        strategy_groups = defaultdict(list)
        
        for trade in self.trades_history:
            if 'strategy' not in trade:
                continue
            
            strategy = trade['strategy']
            strategy_groups[strategy].append(trade)
        
        results = {}
        for strategy, trades in strategy_groups.items():
            # Filter for closed trades with PnL data
            closed_trades = [t for t in trades if t.get('pnl') is not None]
            
            if not closed_trades:
                continue
                
            profit_trades = [t for t in closed_trades if t['pnl'] > 0]
            loss_trades = [t for t in closed_trades if t['pnl'] <= 0]
            
            # Calculate win rate and average profit/loss
            win_rate = len(profit_trades) / len(closed_trades) if closed_trades else 0
            avg_profit = np.mean([t['pnl'] for t in profit_trades]) if profit_trades else 0
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            
            # Calculate profit factor
            total_profit = sum(t['pnl'] for t in profit_trades)
            total_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio (if we have daily returns)
            daily_returns = []
            if all('daily_return' in t for t in closed_trades):
                daily_returns = [t['daily_return'] for t in closed_trades]
            
            sharpe = 0
            if daily_returns:
                avg_return = np.mean(daily_returns)
                std_dev = np.std(daily_returns)
                sharpe = (avg_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
            
            # Calculate average holding period
            avg_days_held = np.mean([t.get('days_held', 0) for t in closed_trades])
            
            # Store results
            results[strategy] = {
                'trade_count': len(closed_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': sum(t['pnl'] for t in closed_trades),
                'sharpe': sharpe,
                'avg_days_held': avg_days_held
            }
            
        return results
    
    def analyze_dte_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by days to expiration at entry
        
        Returns:
            Dictionary with performance metrics by DTE bracket
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Define DTE brackets
        dte_brackets = {
            '0-7': (0, 7),
            '8-14': (8, 14),
            '15-30': (15, 30),
            '31-60': (31, 60),
            '61+': (61, float('inf'))
        }
        
        # Group trades by DTE bracket
        dte_groups = defaultdict(list)
        
        for trade in self.trades_history:
            if 'dte_entry' not in trade or trade.get('pnl') is None:
                continue
                
            dte = trade['dte_entry']
            
            for bracket_name, (min_dte, max_dte) in dte_brackets.items():
                if min_dte <= dte <= max_dte:
                    dte_groups[bracket_name].append(trade)
                    break
        
        results = {}
        for bracket_name, trades in dte_groups.items():
            if not trades:
                continue
                
            profit_trades = [t for t in trades if t['pnl'] > 0]
            loss_trades = [t for t in trades if t['pnl'] <= 0]
            
            # Calculate metrics
            win_rate = len(profit_trades) / len(trades) if trades else 0
            avg_profit = np.mean([t['pnl'] for t in profit_trades]) if profit_trades else 0
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            
            # Calculate profit factor
            total_profit = sum(t['pnl'] for t in profit_trades)
            total_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Store results
            results[bracket_name] = {
                'trade_count': len(trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': sum(t['pnl'] for t in trades)
            }
            
        return results
    
    def analyze_greeks_exposure(self) -> Dict[str, Any]:
        """
        Analyze historical Greeks exposure over time
        
        Returns:
            Dictionary with Greeks exposure metrics
        """
        if not self.daily_greeks_snapshots:
            return {"error": "No Greeks data available"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.daily_greeks_snapshots)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Calculate rolling metrics (7-day and 30-day)
        rolling_metrics = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            if greek in df.columns:
                # Calculate absolute average
                rolling_metrics[f'{greek}_7d_avg'] = df[greek].rolling('7D').mean()
                rolling_metrics[f'{greek}_30d_avg'] = df[greek].rolling('30D').mean()
                
                # Calculate absolute max
                rolling_metrics[f'{greek}_7d_max'] = df[greek].rolling('7D').max()
                rolling_metrics[f'{greek}_30d_max'] = df[greek].rolling('30D').max()
                
                # Calculate absolute min
                rolling_metrics[f'{greek}_7d_min'] = df[greek].rolling('7D').min()
                rolling_metrics[f'{greek}_30d_min'] = df[greek].rolling('30D').min()
        
        # Calculate dollar-weighted metrics if available
        dollar_metrics = {}
        
        if all(col in df.columns for col in ['delta_dollars', 'theta_dollars', 'vega_dollars']):
            # Calculate as percentage of account value
            if 'account_value' in df.columns:
                df['delta_pct'] = df['delta_dollars'] / df['account_value'] * 100
                df['theta_pct'] = df['theta_dollars'] / df['account_value'] * 100
                df['vega_pct'] = df['vega_dollars'] / df['account_value'] * 100
                
                dollar_metrics['max_delta_pct'] = df['delta_pct'].max()
                dollar_metrics['min_delta_pct'] = df['delta_pct'].min()
                dollar_metrics['avg_delta_pct'] = df['delta_pct'].mean()
                
                dollar_metrics['max_theta_pct'] = df['theta_pct'].max()
                dollar_metrics['min_theta_pct'] = df['theta_pct'].min()
                dollar_metrics['avg_theta_pct'] = df['theta_pct'].mean()
                
                dollar_metrics['max_vega_pct'] = df['vega_pct'].max()
                dollar_metrics['min_vega_pct'] = df['vega_pct'].min()
                dollar_metrics['avg_vega_pct'] = df['vega_pct'].mean()
        
        # Calculate best and worst days for each Greek
        best_worst = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            if greek in df.columns:
                greek_dollars = f'{greek}_dollars'
                
                if greek_dollars in df.columns:
                    best_worst[f'best_{greek}_day'] = df[greek_dollars].idxmax()
                    best_worst[f'worst_{greek}_day'] = df[greek_dollars].idxmin()
                    best_worst[f'best_{greek}_value'] = df[greek_dollars].max()
                    best_worst[f'worst_{greek}_value'] = df[greek_dollars].min()
        
        # Combine all metrics
        return {
            'summary_stats': {
                'delta': {
                    'min': df['delta'].min() if 'delta' in df.columns else None,
                    'max': df['delta'].max() if 'delta' in df.columns else None,
                    'mean': df['delta'].mean() if 'delta' in df.columns else None,
                    'std': df['delta'].std() if 'delta' in df.columns else None
                },
                'gamma': {
                    'min': df['gamma'].min() if 'gamma' in df.columns else None,
                    'max': df['gamma'].max() if 'gamma' in df.columns else None,
                    'mean': df['gamma'].mean() if 'gamma' in df.columns else None,
                    'std': df['gamma'].std() if 'gamma' in df.columns else None
                },
                'theta': {
                    'min': df['theta'].min() if 'theta' in df.columns else None,
                    'max': df['theta'].max() if 'theta' in df.columns else None,
                    'mean': df['theta'].mean() if 'theta' in df.columns else None,
                    'std': df['theta'].std() if 'theta' in df.columns else None
                },
                'vega': {
                    'min': df['vega'].min() if 'vega' in df.columns else None,
                    'max': df['vega'].max() if 'vega' in df.columns else None,
                    'mean': df['vega'].mean() if 'vega' in df.columns else None,
                    'std': df['vega'].std() if 'vega' in df.columns else None
                }
            },
            'rolling_metrics': rolling_metrics,
            'dollar_metrics': dollar_metrics,
            'best_worst_days': best_worst,
            'recent_greeks': df.iloc[-1].to_dict() if not df.empty else {}
        }
    
    def plot_greeks_over_time(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots of Greeks exposure over time
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        if not self.daily_greeks_snapshots:
            print("No Greeks data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.daily_greeks_snapshots)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot Delta
        if 'delta' in df.columns:
            axes[0].plot(df.index, df['delta'], label='Delta', color='blue')
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0].set_title('Portfolio Delta Over Time')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot Gamma
        if 'gamma' in df.columns:
            axes[1].plot(df.index, df['gamma'], label='Gamma', color='green')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_title('Portfolio Gamma Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Plot Theta
        if 'theta' in df.columns:
            axes[2].plot(df.index, df['theta'], label='Theta', color='red')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_title('Portfolio Theta Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Plot Vega
        if 'vega' in df.columns:
            axes[3].plot(df.index, df['vega'], label='Vega', color='purple')
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[3].set_title('Portfolio Vega Over Time')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_iv_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots of IV percentile distribution at trade entry
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        if not self.trades_history:
            print("No trade data available for plotting")
            return
        
        # Extract IV percentiles
        iv_entries = [trade.get('iv_percentile', None) for trade in self.trades_history 
                    if trade.get('iv_percentile') is not None]
        
        if not iv_entries:
            print("No IV data available for plotting")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        sns.histplot(iv_entries, bins=20, kde=True)
        
        # Add vertical lines for quartiles
        quartiles = np.percentile(iv_entries, [25, 50, 75])
        plt.axvline(quartiles[0], color='r', linestyle='--', alpha=0.5, label='25th Percentile')
        plt.axvline(quartiles[1], color='g', linestyle='--', alpha=0.5, label='Median')
        plt.axvline(quartiles[2], color='b', linestyle='--', alpha=0.5, label='75th Percentile')
        
        # Add mean
        plt.axvline(np.mean(iv_entries), color='purple', linestyle='-', alpha=0.5, label='Mean')
        
        plt.title('Distribution of IV Percentile at Trade Entry')
        plt.xlabel('IV Percentile')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_win_rate_by_iv(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots of win rate by IV percentile
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        iv_analysis = self.analyze_iv_metrics()
        
        if 'error' in iv_analysis:
            print(iv_analysis['error'])
            return
        
        if 'success_by_iv' not in iv_analysis:
            print("No success rate by IV data available")
            return
        
        success_by_iv = iv_analysis['success_by_iv']
        
        # Extract data
        buckets = [item['bucket'] for item in success_by_iv]
        success_rates = [item['success_rate'] for item in success_by_iv]
        trade_counts = [item['trade_count'] for item in success_by_iv]
        profit_factors = [item['profit_factor'] for item in success_by_iv]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot win rate
        bars = ax1.bar(buckets, success_rates, color='skyblue')
        
        # Add trade count as text on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'n={trade_counts[i]}',
                   ha='center', va='bottom', rotation=0)
        
        ax1.set_title('Win Rate by IV Percentile')
        ax1.set_ylabel('Win Rate')
        ax1.set_ylim(0, 1.1)  # Make room for the count labels
        ax1.grid(True, alpha=0.3)
        
        # Plot profit factor
        ax2.bar(buckets, profit_factors, color='salmon')
        ax2.set_title('Profit Factor by IV Percentile')
        ax2.set_xlabel('IV Percentile Bucket')
        ax2.set_ylabel('Profit Factor')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def analyze_underlying_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by underlying asset
        
        Returns:
            Dictionary with performance metrics for each underlying
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Group trades by underlying
        underlying_groups = defaultdict(list)
        
        for trade in self.trades_history:
            if 'underlying' not in trade:
                continue
                
            underlying = trade['underlying']
            underlying_groups[underlying].append(trade)
        
        results = {}
        for underlying, trades in underlying_groups.items():
            # Filter for closed trades with PnL data
            closed_trades = [t for t in trades if t.get('pnl') is not None]
            
            if not closed_trades:
                continue
                
            profit_trades = [t for t in closed_trades if t['pnl'] > 0]
            loss_trades = [t for t in closed_trades if t['pnl'] <= 0]
            
            # Calculate metrics
            win_rate = len(profit_trades) / len(closed_trades) if closed_trades else 0
            avg_profit = np.mean([t['pnl'] for t in profit_trades]) if profit_trades else 0
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
            
            # Calculate profit factor
            total_profit = sum(t['pnl'] for t in profit_trades)
            total_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Store results
            results[underlying] = {
                'trade_count': len(closed_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': sum(t['pnl'] for t in closed_trades)
            }
            
        return results
    
    def analyze_time_decay(self) -> Dict[str, Any]:
        """
        Analyze theta decay and its impact on performance
        
        Returns:
            Dictionary with time decay analysis
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Filter trades with theta values and PnL
        theta_trades = [t for t in self.trades_history 
                      if 'greeks' in t and 'theta' in t.get('greeks', {}) 
                      and t.get('pnl') is not None
                      and t.get('days_held') is not None]
        
        if not theta_trades:
            return {"error": "No trades with theta data available"}
        
        # Calculate realized daily decay vs. theoretical
        realized_vs_theoretical = []
        
        for trade in theta_trades:
            theta = trade['greeks']['theta']
            days_held = trade['days_held']
            pnl = trade['pnl']
            side = trade['side']
            quantity = trade['quantity']
            
            # Adjust theta based on position side
            position_theta = theta * quantity * (1 if side == 'LONG' else -1)
            
            # Theoretical decay
            theoretical_decay = position_theta * days_held
            
            # Realized PnL
            realized = pnl
            
            realized_vs_theoretical.append({
                'trade_id': trade.get('trade_id', ''),
                'theoretical_decay': theoretical_decay,
                'realized_pnl': realized,
                'days_held': days_held,
                'theta_capture': realized / theoretical_decay if theoretical_decay != 0 else 0
            })
        
        # Calculate average theta capture
        avg_theta_capture = np.mean([t['theta_capture'] for t in realized_vs_theoretical 
                                  if t['theoretical_decay'] != 0])
        
        # Group by strategy
        strategy_theta_performance = defaultdict(list)
        
        for trade in theta_trades:
            if 'strategy' in trade:
                strategy = trade['strategy']
                theta = trade['greeks']['theta']
                days_held = trade['days_held']
                pnl = trade['pnl']
                side = trade['side']
                quantity = trade['quantity']
                
                position_theta = theta * quantity * (1 if side == 'LONG' else -1)
                theoretical_decay = position_theta * days_held
                
                strategy_theta_performance[strategy].append({
                    'theoretical_decay': theoretical_decay,
                    'realized_pnl': pnl,
                    'theta_capture': pnl / theoretical_decay if theoretical_decay != 0 else 0
                })
        
        # Calculate average theta capture by strategy
        strategy_avg_capture = {}
        
        for strategy, decays in strategy_theta_performance.items():
            if decays:
                strategy_avg_capture[strategy] = np.mean([d['theta_capture'] for d in decays 
                                                        if d['theoretical_decay'] != 0])
        
        return {
            'avg_theta_capture': avg_theta_capture,
            'realized_vs_theoretical': realized_vs_theoretical,
            'strategy_theta_capture': strategy_avg_capture
        }
    
    def analyze_vega_exposure(self) -> Dict[str, Any]:
        """
        Analyze vega exposure and its impact on performance
        
        Returns:
            Dictionary with vega exposure analysis
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Filter trades with vega values and PnL
        vega_trades = [t for t in self.trades_history 
                      if 'greeks' in t and 'vega' in t.get('greeks', {}) 
                      and t.get('pnl') is not None
                      and t.get('iv_percentile') is not None
                      and t.get('iv_percentile_exit') is not None]
        
        if not vega_trades:
            return {"error": "No trades with vega data available"}
        
        # Calculate IV change impact
        iv_change_impact = []
        
        for trade in vega_trades:
            vega = trade['greeks']['vega']
            iv_entry = trade['iv_percentile']
            iv_exit = trade['iv_percentile_exit']
            iv_change = iv_exit - iv_entry
            pnl = trade['pnl']
            side = trade['side']
            quantity = trade['quantity']
            
            # Adjust vega based on position side
            position_vega = vega * quantity * (1 if side == 'LONG' else -1)
            
            # Theoretical impact of IV change (simple approximation)
            # Note: This is a rough approximation, real options would need more precise modeling
            theoretical_impact = position_vega * iv_change * 100  # Scale factor
            
            iv_change_impact.append({
                'trade_id': trade.get('trade_id', ''),
                'iv_entry': iv_entry,
                'iv_exit': iv_exit,
                'iv_change': iv_change,
                'position_vega': position_vega,
                'theoretical_impact': theoretical_impact,
                'realized_pnl': pnl,
                'vega_contribution': theoretical_impact / pnl if pnl != 0 else 0
            })
        
        # Calculate correlation between IV change and PnL
        iv_changes = [t['iv_change'] for t in iv_change_impact]
        pnls = [t['realized_pnl'] for t in iv_change_impact]
        
        correlation = np.corrcoef(iv_changes, pnls)[0, 1] if len(iv_changes) > 1 else 0
        
        # Group by strategy
        strategy_vega_performance = defaultdict(list)
        
        for trade in vega_trades:
            if 'strategy' in trade:
                strategy = trade['strategy']
                iv_entry = trade['iv_percentile']
                iv_exit = trade['iv_percentile_exit']
                iv_change = iv_exit - iv_entry
                
                strategy_vega_performance[strategy].append({
                    'iv_change': iv_change,
                    'pnl': trade['pnl']
                })
        
        # Calculate correlation by strategy
        strategy_correlations = {}
        
        for strategy, changes in strategy_vega_performance.items():
            if len(changes) > 1:
                strategy_iv_changes = [c['iv_change'] for c in changes]
                strategy_pnls = [c['pnl'] for c in changes]
                
                strategy_correlations[strategy] = np.corrcoef(strategy_iv_changes, strategy_pnls)[0, 1]
        
        return {
            'iv_pnl_correlation': correlation,
            'iv_change_impact': iv_change_impact,
            'strategy_correlations': strategy_correlations
        }
    
    def generate_options_performance_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive options performance report
        
        Args:
            save_path: Directory to save plot images (optional)
            
        Returns:
            Dictionary with all performance metrics
        """
        # Gather all metrics
        iv_metrics = self.analyze_iv_metrics()
        strategy_performance = self.analyze_strategy_performance()
        dte_performance = self.analyze_dte_performance()
        greeks_exposure = self.analyze_greeks_exposure()
        underlying_performance = self.analyze_underlying_performance()
        time_decay_analysis = self.analyze_time_decay()
        vega_exposure_analysis = self.analyze_vega_exposure()
        
        # Generate plots if save_path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            self.plot_greeks_over_time(f"{save_path}/greeks_over_time.png")
            self.plot_iv_distribution(f"{save_path}/iv_distribution.png")
            self.plot_win_rate_by_iv(f"{save_path}/win_rate_by_iv.png")
        
        # Calculate overall metrics
        closed_trades = [t for t in self.trades_history if t.get('pnl') is not None]
        profitable_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        
        overall_win_rate = len(profitable_trades) / len(closed_trades) if closed_trades else 0
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        
        # Calculate average metrics by option type
        call_trades = [t for t in closed_trades if t.get('option_type') == 'CALL']
        put_trades = [t for t in closed_trades if t.get('option_type') == 'PUT']
        
        call_win_rate = (len([t for t in call_trades if t.get('pnl', 0) > 0]) / 
                       len(call_trades) if call_trades else 0)
        put_win_rate = (len([t for t in put_trades if t.get('pnl', 0) > 0]) / 
                      len(put_trades) if put_trades else 0)
        
        # Aggregate all metrics into a comprehensive report
        report = {
            'overview': {
                'trade_count': len(self.trades_history),
                'closed_trade_count': len(closed_trades),
                'total_pnl': total_pnl,
                'overall_win_rate': overall_win_rate,
                'call_win_rate': call_win_rate,
                'put_win_rate': put_win_rate,
                'avg_trade_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
                'profitable_trades': len(profitable_trades),
                'unprofitable_trades': len(closed_trades) - len(profitable_trades)
            },
            'iv_metrics': iv_metrics,
            'strategy_performance': strategy_performance,
            'dte_performance': dte_performance,
            'greeks_exposure': greeks_exposure,
            'underlying_performance': underlying_performance,
            'time_decay_analysis': time_decay_analysis,
            'vega_exposure_analysis': vega_exposure_analysis
        }
        
        # Log a summary
        logger.info(f"Generated options performance report with {len(closed_trades)} closed trades")
        logger.info(f"Overall win rate: {overall_win_rate:.2%}, Total P&L: ${total_pnl:.2f}")
        
        return report
    
    def calculate_options_drawdown(self, include_open_positions: bool = False) -> Dict[str, Any]:
        """
        Calculate options-specific drawdown metrics, which are particularly important for short options positions
        
        Args:
            include_open_positions: Whether to include open positions in the calculation
            
        Returns:
            Dictionary with drawdown metrics
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Get closed trades or include open ones based on parameter
        if include_open_positions:
            relevant_trades = self.trades_history
        else:
            relevant_trades = [t for t in self.trades_history if t.get('pnl') is not None]
        
        if not relevant_trades:
            return {"error": "No relevant trades for drawdown calculation"}
        
        # Sort trades by timestamp
        sorted_trades = sorted(relevant_trades, key=lambda x: x.get('timestamp', ''))
        
        # Calculate daily P&L per strategy and overall
        daily_pnl = defaultdict(lambda: defaultdict(float))
        strategy_cumulative = defaultdict(list)
        overall_cumulative = []
        overall_underwater = []
        strategy_underwater = defaultdict(list)
        dates = []
        
        # Track peak equity
        peak_equity_overall = 0
        peak_equity_strategy = defaultdict(float)
        
        # Track maximum drawdowns
        max_dd_overall = 0
        max_dd_strategy = defaultdict(float)
        max_dd_date_overall = None
        max_dd_date_strategy = defaultdict(lambda: None)
        
        # Group by date and calculate daily P&L
        for trade in sorted_trades:
            date_str = trade.get('timestamp', '').split('T')[0] if isinstance(trade.get('timestamp', ''), str) else ''
            if not date_str:
                continue
                
            pnl = trade.get('pnl', 0)
            strategy = trade.get('strategy', 'unknown')
            
            daily_pnl[date_str][strategy] += pnl
            daily_pnl[date_str]['overall'] += pnl
        
        # Convert to sorted dates
        sorted_dates = sorted(daily_pnl.keys())
        
        # Calculate cumulative P&L and underwater equity
        current_equity_overall = 0
        current_equity_strategy = defaultdict(float)
        
        for date in sorted_dates:
            # Update current equity
            current_equity_overall += daily_pnl[date]['overall']
            
            # Update peak equity overall
            if current_equity_overall > peak_equity_overall:
                peak_equity_overall = current_equity_overall
            
            # Calculate drawdown
            dd_overall = peak_equity_overall - current_equity_overall
            dd_pct_overall = dd_overall / peak_equity_overall if peak_equity_overall > 0 else 0
            
            # Track maximum drawdown
            if dd_overall > max_dd_overall:
                max_dd_overall = dd_overall
                max_dd_date_overall = date
            
            # Add to overall lists
            dates.append(date)
            overall_cumulative.append(current_equity_overall)
            overall_underwater.append(-dd_pct_overall * 100)  # Negative for underwater chart
            
            # Process per strategy
            for strategy in set([t.get('strategy', 'unknown') for t in relevant_trades]):
                # Update current equity for this strategy
                current_equity_strategy[strategy] += daily_pnl[date].get(strategy, 0)
                
                # Update peak equity for this strategy
                if current_equity_strategy[strategy] > peak_equity_strategy[strategy]:
                    peak_equity_strategy[strategy] = current_equity_strategy[strategy]
                
                # Calculate drawdown for this strategy
                dd_strategy = peak_equity_strategy[strategy] - current_equity_strategy[strategy]
                dd_pct_strategy = dd_strategy / peak_equity_strategy[strategy] if peak_equity_strategy[strategy] > 0 else 0
                
                # Track maximum drawdown for this strategy
                if dd_strategy > max_dd_strategy[strategy]:
                    max_dd_strategy[strategy] = dd_strategy
                    max_dd_date_strategy[strategy] = date
                
                # Add to strategy lists
                if not strategy_cumulative.get(strategy):
                    strategy_cumulative[strategy] = [0] * len(dates)
                    strategy_underwater[strategy] = [0] * len(dates)
                
                # Ensure lists are the correct length by appending if needed
                while len(strategy_cumulative[strategy]) < len(dates):
                    strategy_cumulative[strategy].append(strategy_cumulative[strategy][-1])
                    strategy_underwater[strategy].append(strategy_underwater[strategy][-1])
                
                # Update the last element with current value
                strategy_cumulative[strategy][-1] = current_equity_strategy[strategy]
                strategy_underwater[strategy][-1] = -dd_pct_strategy * 100  # Negative for underwater chart
        
        # Calculate additional metrics
        recovery_periods = []
        
        # Identify drawdown periods and calculate recovery time
        in_drawdown = False
        drawdown_start = None
        drawdown_peak = None
        drawdown_value = 0
        
        for i, eq in enumerate(overall_cumulative):
            if i == 0:
                continue
                
            if eq < overall_cumulative[i-1] and not in_drawdown:
                # Entering drawdown
                in_drawdown = True
                drawdown_start = dates[i]
                drawdown_peak = overall_cumulative[i-1]
            
            if in_drawdown:
                # Update max drawdown during this period
                current_dd = drawdown_peak - eq
                if current_dd > drawdown_value:
                    drawdown_value = current_dd
                
                # Check if recovered
                if eq >= drawdown_peak:
                    in_drawdown = False
                    recovery_end = dates[i]
                    # Calculate recovery time in days
                    start_dt = datetime.strptime(drawdown_start, '%Y-%m-%d')
                    end_dt = datetime.strptime(recovery_end, '%Y-%m-%d')
                    recovery_time = (end_dt - start_dt).days
                    
                    recovery_periods.append({
                        'start_date': drawdown_start,
                        'end_date': recovery_end,
                        'days': recovery_time,
                        'drawdown_value': drawdown_value,
                        'drawdown_pct': (drawdown_value / drawdown_peak) * 100 if drawdown_peak > 0 else 0
                    })
                    
                    # Reset tracking variables
                    drawdown_value = 0
        
        # Special metrics for short options positions
        short_options_metrics = {}
        
        # Get all short option trades
        short_option_trades = [t for t in relevant_trades if t.get('side') == 'SHORT']
        
        if short_option_trades:
            # Calculate maximum theoretical risk for short options
            theoretical_risk = 0
            
            for trade in short_option_trades:
                option_type = trade.get('option_type')
                trade_type = trade.get('strategy', '').lower()
                quantity = trade.get('quantity', 0)
                strike = trade.get('strike', 0)
                
                # Max loss depends on option type and trade structure
                if 'credit_spread' in trade_type or 'iron_condor' in trade_type:
                    # For credit spreads, max loss is width of spread - credit received
                    max_loss_per_contract = trade.get('spread_width', 0) - trade.get('entry_price', 0)
                elif option_type == 'CALL' and 'naked' in trade_type:
                    # For naked calls, theoretical max loss is unlimited, use a large number for calculation
                    max_loss_per_contract = 10 * strike  # Arbitrary large multiplier
                elif option_type == 'PUT' and 'naked' in trade_type:
                    # For naked puts, max loss is strike price - premium
                    max_loss_per_contract = strike - trade.get('entry_price', 0)
                else:
                    # For other short positions, use a conservative estimate
                    max_loss_per_contract = 2 * trade.get('entry_price', 0)
                
                theoretical_risk += max_loss_per_contract * quantity * 100  # * 100 for contract multiplier
            
            # Calculate realized vs theoretical drawdown ratio
            realized_drawdown_ratio = max_dd_overall / theoretical_risk if theoretical_risk > 0 else 0
            
            short_options_metrics = {
                'theoretical_risk': theoretical_risk,
                'realized_drawdown_ratio': realized_drawdown_ratio,
                'worst_case_utilization_pct': (max_dd_overall / theoretical_risk) * 100 if theoretical_risk > 0 else 0
            }
        
        return {
            'max_drawdown_pct': (max_dd_overall / peak_equity_overall) * 100 if peak_equity_overall > 0 else 0,
            'max_drawdown_value': max_dd_overall,
            'max_drawdown_date': max_dd_date_overall,
            'strategy_max_drawdowns': {
                strategy: {
                    'max_drawdown_pct': (max_dd_strategy[strategy] / peak_equity_strategy[strategy]) * 100 
                        if peak_equity_strategy[strategy] > 0 else 0,
                    'max_drawdown_value': max_dd_strategy[strategy],
                    'max_drawdown_date': max_dd_date_strategy[strategy]
                } for strategy in max_dd_strategy
            },
            'recovery_periods': recovery_periods,
            'avg_recovery_days': np.mean([r['days'] for r in recovery_periods]) if recovery_periods else 0,
            'short_options_metrics': short_options_metrics,
            'time_series': {
                'dates': dates,
                'equity_curve': overall_cumulative,
                'underwater_curve': overall_underwater,
                'strategy_equity': strategy_cumulative,
                'strategy_underwater': strategy_underwater
            }
        }
    
    def calculate_greeks_at_risk(self, confidence_level: float = 0.95, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Calculate "Greeks at Risk" (GaR) - similar to Value at Risk but for Greeks exposure
        
        Args:
            confidence_level: Confidence level for VaR calculation (default: 95%)
            lookback_days: Number of days to look back for historical simulation
            
        Returns:
            Dictionary with Greeks at Risk metrics
        """
        if not self.daily_greeks_snapshots:
            return {"error": "No Greeks data available"}
            
        # Convert snapshots to DataFrame
        df = pd.DataFrame(self.daily_greeks_snapshots)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_df = df[df.index > cutoff_date]
        
        if historical_df.empty:
            return {"error": "Not enough historical data for GaR calculation"}
            
        # Calculate daily changes in Greeks
        greek_columns = ['delta', 'gamma', 'theta', 'vega', 'rho']
        
        # Check which Greeks are available
        available_greeks = [g for g in greek_columns if g in historical_df.columns]
        
        if not available_greeks:
            return {"error": "No Greek data found in snapshots"}
            
        # Calculate daily changes for each Greek
        changes_df = historical_df[available_greeks].diff().dropna()
        
        if changes_df.empty:
            return {"error": "Cannot calculate changes in Greeks"}
            
        # Calculate GaR for each Greek
        gar_results = {}
        
        for greek in available_greeks:
            # Get daily changes
            daily_changes = changes_df[greek].values
            
            # Calculate historical VaR
            var_index = int(len(daily_changes) * (1 - confidence_level))
            if var_index < 0 or var_index >= len(daily_changes):
                gar_results[f"{greek}_gar"] = None
                continue
                
            sorted_changes = np.sort(daily_changes)
            historical_var = abs(sorted_changes[var_index])
            
            # Store GaR
            gar_results[f"{greek}_gar"] = historical_var
            
            # Calculate parametric VaR as alternative
            mean_change = np.mean(daily_changes)
            std_change = np.std(daily_changes)
            z_score = stats.norm.ppf(1 - confidence_level)
            parametric_var = abs(mean_change + z_score * std_change)
            
            gar_results[f"{greek}_parametric_gar"] = parametric_var
            
            # Calculate as percentage of current exposure if account value available
            if 'account_value' in historical_df.columns and greek in historical_df.columns:
                current_value = historical_df[greek].iloc[-1]
                current_account = historical_df['account_value'].iloc[-1]
                
                if current_account > 0:
                    gar_results[f"{greek}_gar_pct"] = (historical_var / current_value) * 100 if current_value != 0 else 0
                    gar_results[f"{greek}_gar_pct_of_account"] = (historical_var / current_account) * 100
        
        # Calculate combined Greeks at Risk - normalize and sum
        if all(f"{greek}_gar" in gar_results and gar_results[f"{greek}_gar"] is not None for greek in available_greeks):
            # Normalize each GaR by dividing by average absolute Greek value
            normalized_gar = {}
            
            for greek in available_greeks:
                avg_abs_value = historical_df[greek].abs().mean()
                if avg_abs_value > 0:
                    normalized_gar[greek] = gar_results[f"{greek}_gar"] / avg_abs_value
                else:
                    normalized_gar[greek] = 0
            
            # Sum normalized GaRs
            combined_gar = sum(normalized_gar.values())
            gar_results["combined_normalized_gar"] = combined_gar
        
        # Current Greeks exposure
        current_exposure = {greek: historical_df[greek].iloc[-1] if not historical_df[greek].empty else None 
                          for greek in available_greeks}
        
        return {
            "greeks_at_risk": gar_results,
            "current_exposure": current_exposure,
            "confidence_level": confidence_level,
            "lookback_days": lookback_days,
            "calculation_date": datetime.now().strftime('%Y-%m-%d')
        }
    
    def run_stress_test(self, 
                        price_changes: List[float] = [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20],
                        volatility_changes: List[float] = [-0.50, -0.30, -0.10, 0.10, 0.30, 0.50],
                        time_horizons: List[int] = [1, 5, 10, 30]) -> Dict[str, Any]:
        """
        Run stress tests for extreme market moves, especially important for options
        
        Args:
            price_changes: List of price changes to test (as decimal percentages)
            volatility_changes: List of volatility changes to test (as decimal percentages)
            time_horizons: List of days forward to project
            
        Returns:
            Dictionary with stress test results
        """
        if not self.options_risk_manager:
            return {"error": "Options risk manager required for stress testing"}
            
        # Get current positions
        current_positions = self.options_risk_manager.get_current_positions()
        
        if not current_positions:
            return {"error": "No open positions for stress testing"}
            
        # Generate stress test matrix
        stress_matrix = {}
        
        for symbol, position in current_positions.items():
            # Get current price of underlying
            underlying = position.get('underlying')
            current_price = self.options_risk_manager.get_underlying_price(underlying)
            current_iv = position.get('implied_volatility', 0.30)  # Default to 30% if not available
            
            # Skip if we can't get price
            if not current_price:
                continue
                
            # Initialize symbol in stress matrix
            stress_matrix[symbol] = {
                'price_changes': {},
                'volatility_changes': {},
                'time_decay': {},
                'combined_scenarios': []
            }
            
            # Price change scenarios
            for pct_change in price_changes:
                new_price = current_price * (1 + pct_change)
                
                # Calculate new Greeks and value
                new_values = self.options_risk_manager.calculate_position_greeks(
                    symbol=symbol,
                    underlying_price=new_price,
                    volatility=current_iv,
                    days_to_expiration=position.get('days_to_expiration', 30)
                )
                
                # Calculate P&L
                current_value = position.get('current_value', 0)
                theo_value = new_values.get('position_value', 0)
                pnl = theo_value - current_value
                
                stress_matrix[symbol]['price_changes'][pct_change] = {
                    'new_price': new_price,
                    'new_value': theo_value,
                    'pnl': pnl,
                    'pnl_pct': (pnl / current_value) * 100 if current_value != 0 else 0,
                    'greeks': {
                        'delta': new_values.get('delta', 0),
                        'gamma': new_values.get('gamma', 0),
                        'theta': new_values.get('theta', 0),
                        'vega': new_values.get('vega', 0)
                    }
                }
            
            # Volatility change scenarios
            for vol_change in volatility_changes:
                new_iv = current_iv * (1 + vol_change)
                
                # Calculate new Greeks and value
                new_values = self.options_risk_manager.calculate_position_greeks(
                    symbol=symbol,
                    underlying_price=current_price,
                    volatility=new_iv,
                    days_to_expiration=position.get('days_to_expiration', 30)
                )
                
                # Calculate P&L
                current_value = position.get('current_value', 0)
                theo_value = new_values.get('position_value', 0)
                pnl = theo_value - current_value
                
                stress_matrix[symbol]['volatility_changes'][vol_change] = {
                    'new_iv': new_iv,
                    'new_value': theo_value,
                    'pnl': pnl,
                    'pnl_pct': (pnl / current_value) * 100 if current_value != 0 else 0,
                    'greeks': {
                        'delta': new_values.get('delta', 0),
                        'gamma': new_values.get('gamma', 0),
                        'theta': new_values.get('theta', 0),
                        'vega': new_values.get('vega', 0)
                    }
                }
            
            # Time decay scenarios
            for days in time_horizons:
                new_dte = position.get('days_to_expiration', 30) - days
                
                if new_dte <= 0:
                    # Option expired
                    stress_matrix[symbol]['time_decay'][days] = {
                        'value_at_expiration': self._calculate_expiration_value(position, current_price),
                        'status': 'expired'
                    }
                    continue
                
                # Calculate new Greeks and value
                new_values = self.options_risk_manager.calculate_position_greeks(
                    symbol=symbol,
                    underlying_price=current_price,
                    volatility=current_iv,
                    days_to_expiration=new_dte
                )
                
                # Calculate P&L
                current_value = position.get('current_value', 0)
                theo_value = new_values.get('position_value', 0)
                pnl = theo_value - current_value
                
                stress_matrix[symbol]['time_decay'][days] = {
                    'new_dte': new_dte,
                    'new_value': theo_value,
                    'pnl': pnl,
                    'pnl_pct': (pnl / current_value) * 100 if current_value != 0 else 0,
                    'daily_theta': pnl / days if days > 0 else 0,
                    'greeks': {
                        'delta': new_values.get('delta', 0),
                        'gamma': new_values.get('gamma', 0),
                        'theta': new_values.get('theta', 0),
                        'vega': new_values.get('vega', 0)
                    }
                }
            
            # Combined extreme scenarios
            extreme_scenarios = [
                # Severe down move with volatility spike
                {"scenario": "crash", "price_change": -0.15, "vol_change": 0.50, "days": 1},
                # Large down move
                {"scenario": "correction", "price_change": -0.10, "vol_change": 0.30, "days": 5},
                # Large up move with volatility spike
                {"scenario": "short_squeeze", "price_change": 0.15, "vol_change": 0.30, "days": 1},
                # Slow bleed down
                {"scenario": "slow_decline", "price_change": -0.05, "vol_change": 0.10, "days": 10},
                # Volatility collapse
                {"scenario": "vol_collapse", "price_change": 0.05, "vol_change": -0.30, "days": 5}
            ]
            
            for scenario in extreme_scenarios:
                # Get scenario parameters
                price_change = scenario["price_change"]
                vol_change = scenario["vol_change"]
                days = scenario["days"]
                
                # Calculate new values
                new_price = current_price * (1 + price_change)
                new_iv = current_iv * (1 + vol_change)
                new_dte = position.get('days_to_expiration', 30) - days
                
                if new_dte <= 0:
                    # Option expired in this scenario
                    scenario_result = {
                        'scenario': scenario["scenario"],
                        'price_change': price_change,
                        'vol_change': vol_change,
                        'days': days,
                        'value_at_expiration': self._calculate_expiration_value(position, new_price),
                        'status': 'expired'
                    }
                else:
                    # Calculate new Greeks and value
                    new_values = self.options_risk_manager.calculate_position_greeks(
                        symbol=symbol,
                        underlying_price=new_price,
                        volatility=new_iv,
                        days_to_expiration=new_dte
                    )
                    
                    # Calculate P&L
                    current_value = position.get('current_value', 0)
                    theo_value = new_values.get('position_value', 0)
                    pnl = theo_value - current_value
                    
                    scenario_result = {
                        'scenario': scenario["scenario"],
                        'price_change': price_change,
                        'vol_change': vol_change,
                        'days': days,
                        'new_price': new_price,
                        'new_iv': new_iv,
                        'new_dte': new_dte,
                        'new_value': theo_value,
                        'pnl': pnl,
                        'pnl_pct': (pnl / current_value) * 100 if current_value != 0 else 0,
                        'greeks': {
                            'delta': new_values.get('delta', 0),
                            'gamma': new_values.get('gamma', 0),
                            'theta': new_values.get('theta', 0),
                            'vega': new_values.get('vega', 0)
                        }
                    }
                
                stress_matrix[symbol]['combined_scenarios'].append(scenario_result)
        
        # Calculate portfolio level stress test results
        portfolio_results = {
            'price_changes': {},
            'volatility_changes': {},
            'time_decay': {},
            'combined_scenarios': {}
        }
        
        # Aggregate price change impacts
        for pct_change in price_changes:
            total_pnl = sum(stress_matrix[symbol]['price_changes'][pct_change]['pnl'] 
                            for symbol in stress_matrix if pct_change in stress_matrix[symbol]['price_changes'])
            
            portfolio_results['price_changes'][pct_change] = {
                'total_pnl': total_pnl,
                'pnl_pct_of_portfolio': self._calculate_pct_of_portfolio(total_pnl)
            }
        
        # Aggregate volatility change impacts
        for vol_change in volatility_changes:
            total_pnl = sum(stress_matrix[symbol]['volatility_changes'][vol_change]['pnl'] 
                            for symbol in stress_matrix if vol_change in stress_matrix[symbol]['volatility_changes'])
            
            portfolio_results['volatility_changes'][vol_change] = {
                'total_pnl': total_pnl,
                'pnl_pct_of_portfolio': self._calculate_pct_of_portfolio(total_pnl)
            }
        
        # Aggregate time decay impacts
        for days in time_horizons:
            valid_symbols = [
                symbol for symbol in stress_matrix 
                if days in stress_matrix[symbol]['time_decay'] and 'pnl' in stress_matrix[symbol]['time_decay'][days]
            ]
            
            if valid_symbols:
                total_pnl = sum(stress_matrix[symbol]['time_decay'][days]['pnl'] for symbol in valid_symbols)
                
                portfolio_results['time_decay'][days] = {
                    'total_pnl': total_pnl,
                    'pnl_pct_of_portfolio': self._calculate_pct_of_portfolio(total_pnl),
                    'daily_theta': total_pnl / days if days > 0 else 0
                }
        
        # Aggregate combined scenario impacts
        for scenario in extreme_scenarios:
            scenario_name = scenario["scenario"]
            
            # Get all results for this scenario
            scenario_results = []
            for symbol in stress_matrix:
                for result in stress_matrix[symbol]['combined_scenarios']:
                    if result['scenario'] == scenario_name:
                        scenario_results.append(result)
            
            if scenario_results:
                valid_results = [r for r in scenario_results if 'pnl' in r]
                total_pnl = sum(r['pnl'] for r in valid_results) if valid_results else 0
                
                portfolio_results['combined_scenarios'][scenario_name] = {
                    'total_pnl': total_pnl,
                    'pnl_pct_of_portfolio': self._calculate_pct_of_portfolio(total_pnl),
                    'details': scenario
                }
        
        return {
            'position_level_results': stress_matrix,
            'portfolio_level_results': portfolio_results,
            'test_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _calculate_pct_of_portfolio(self, value: float) -> float:
        """Helper to calculate percentage of portfolio value"""
        # Get current portfolio value
        portfolio_value = 0
        if self.options_risk_manager:
            portfolio_value = self.options_risk_manager.get_portfolio_value()
        
        if portfolio_value > 0:
            return (value / portfolio_value) * 100
        return 0
    
    def _calculate_expiration_value(self, position: Dict[str, Any], underlying_price: float) -> float:
        """Calculate option value at expiration given underlying price"""
        option_type = position.get('option_type', 'CALL')
        strike = position.get('strike', 0)
        quantity = position.get('quantity', 0)
        side = position.get('side', 'LONG')
        
        # Calculate intrinsic value
        if option_type == 'CALL':
            intrinsic = max(0, underlying_price - strike)
        else:  # PUT
            intrinsic = max(0, strike - underlying_price)
        
        # Adjust for side
        if side == 'LONG':
            return intrinsic * quantity * 100  # * 100 for contract multiplier
        else:  # SHORT
            return -intrinsic * quantity * 100  # * 100 for contract multiplier
    
    def analyze_performance_by_market_regime(self, regime_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze options performance across different market regimes
        
        Args:
            regime_data: Series with regime labels indexed by date
            
        Returns:
            Dictionary with performance metrics by regime
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
            
        if regime_data.empty:
            return {"error": "No regime data provided"}
        
        # Extract trades with date and PnL
        trades_with_date = []
        for trade in self.trades_history:
            if not trade.get('pnl') or not trade.get('timestamp'):
                continue
            
            # Extract date from timestamp
            if isinstance(trade['timestamp'], str):
                trade_date = trade['timestamp'].split('T')[0]
            elif isinstance(trade['timestamp'], datetime):
                trade_date = trade['timestamp'].strftime('%Y-%m-%d')
            else:
                continue
                
            trades_with_date.append({
                'date': trade_date,
                'pnl': trade['pnl'],
                'strategy': trade.get('strategy', 'unknown'),
                'option_type': trade.get('option_type', 'unknown'),
                'side': trade.get('side', 'unknown'),
                'underlying': trade.get('underlying', 'unknown')
            })
        
        if not trades_with_date:
            return {"error": "No trades with valid dates and PnL"}
        
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades_with_date)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df.set_index('date', inplace=True)
        
        # Add regime data
        # Ensure regime_data index is datetime
        if not isinstance(regime_data.index, pd.DatetimeIndex):
            regime_data.index = pd.to_datetime(regime_data.index)
        
        # Merge trades with regime data
        merged_df = trades_df.copy()
        merged_df['regime'] = None
        
        # Assign regime to each trade
        for date, row in merged_df.iterrows():
            # Find the closest regime date that is less than or equal to the trade date
            regime_dates = regime_data.index[regime_data.index <= date]
            if len(regime_dates) > 0:
                closest_date = regime_dates[-1]
                merged_df.at[date, 'regime'] = regime_data.loc[closest_date]
        
        # Remove trades without regime
        merged_df = merged_df.dropna(subset=['regime'])
        
        if merged_df.empty:
            return {"error": "No trades with matching regime data"}
        
        # Analyze performance by regime
        regime_performance = {}
        
        for regime in merged_df['regime'].unique():
            regime_trades = merged_df[merged_df['regime'] == regime]
            
            # Calculate overall metrics
            total_pnl = regime_trades['pnl'].sum()
            avg_pnl = regime_trades['pnl'].mean()
            win_rate = len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades)
            losing_pnl = regime_trades[regime_trades['pnl'] < 0]['pnl'].sum()
            winning_pnl = regime_trades[regime_trades['pnl'] > 0]['pnl'].sum()
            profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
            
            # Calculate strategy-specific metrics
            strategy_metrics = {}
            for strategy in regime_trades['strategy'].unique():
                strategy_trades = regime_trades[regime_trades['strategy'] == strategy]
                
                if len(strategy_trades) < 2:  # Need at least 2 trades for meaningful metrics
                    continue
                
                strategy_metrics[strategy] = {
                    'trade_count': len(strategy_trades),
                    'total_pnl': strategy_trades['pnl'].sum(),
                    'avg_pnl': strategy_trades['pnl'].mean(),
                    'win_rate': len(strategy_trades[strategy_trades['pnl'] > 0]) / len(strategy_trades),
                    'best_trade': strategy_trades['pnl'].max(),
                    'worst_trade': strategy_trades['pnl'].min()
                }
            
            # Calculate directional metrics
            long_calls = regime_trades[(regime_trades['option_type'] == 'CALL') & (regime_trades['side'] == 'LONG')]
            short_calls = regime_trades[(regime_trades['option_type'] == 'CALL') & (regime_trades['side'] == 'SHORT')]
            long_puts = regime_trades[(regime_trades['option_type'] == 'PUT') & (regime_trades['side'] == 'LONG')]
            short_puts = regime_trades[(regime_trades['option_type'] == 'PUT') & (regime_trades['side'] == 'SHORT')]
            
            directional_metrics = {
                'long_calls': {
                    'trade_count': len(long_calls),
                    'total_pnl': long_calls['pnl'].sum() if not long_calls.empty else 0,
                    'win_rate': len(long_calls[long_calls['pnl'] > 0]) / len(long_calls) if not long_calls.empty else 0
                },
                'short_calls': {
                    'trade_count': len(short_calls),
                    'total_pnl': short_calls['pnl'].sum() if not short_calls.empty else 0,
                    'win_rate': len(short_calls[short_calls['pnl'] > 0]) / len(short_calls) if not short_calls.empty else 0
                },
                'long_puts': {
                    'trade_count': len(long_puts),
                    'total_pnl': long_puts['pnl'].sum() if not long_puts.empty else 0,
                    'win_rate': len(long_puts[long_puts['pnl'] > 0]) / len(long_puts) if not long_puts.empty else 0
                },
                'short_puts': {
                    'trade_count': len(short_puts),
                    'total_pnl': short_puts['pnl'].sum() if not short_puts.empty else 0,
                    'win_rate': len(short_puts[short_puts['pnl'] > 0]) / len(short_puts) if not short_puts.empty else 0
                }
            }
            
            # Store all regime metrics
            regime_performance[regime] = {
                'trade_count': len(regime_trades),
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'strategies': strategy_metrics,
                'directional': directional_metrics
            }
        
        # Calculate relative performance (normalized to overall)
        overall_pnl_per_trade = merged_df['pnl'].mean()
        overall_win_rate = len(merged_df[merged_df['pnl'] > 0]) / len(merged_df)
        
        relative_performance = {}
        for regime, metrics in regime_performance.items():
            relative_performance[regime] = {
                'pnl_ratio': metrics['avg_pnl'] / overall_pnl_per_trade if overall_pnl_per_trade != 0 else 0,
                'win_rate_ratio': metrics['win_rate'] / overall_win_rate if overall_win_rate != 0 else 0
            }
        
        return {
            'regime_performance': regime_performance,
            'relative_performance': relative_performance,
            'trade_count_by_regime': {regime: metrics['trade_count'] for regime, metrics in regime_performance.items()},
            'total_trades_analyzed': len(merged_df)
        }
    
    def analyze_volatility_regimes(self, vix_data: pd.Series = None) -> Dict[str, Any]:
        """
        Analyze performance across different volatility regimes
        
        Args:
            vix_data: Optional Series with VIX index values indexed by date
                     If not provided, will use IV percentiles from trades
            
        Returns:
            Dictionary with performance metrics by volatility regime
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Define volatility regimes based on VIX levels or IV percentiles
        if vix_data is not None and not vix_data.empty:
            # Use VIX data to define regimes
            regime_source = "VIX"
            
            # Convert index to datetime if needed
            if not isinstance(vix_data.index, pd.DatetimeIndex):
                vix_data.index = pd.to_datetime(vix_data.index)
            
            # Define VIX regime thresholds
            vix_regimes = {
                'very_low': (0, 12),
                'low': (12, 16),
                'normal': (16, 22),
                'high': (22, 30),
                'very_high': (30, 100)
            }
            
            # Create regime labels
            vix_regime_labels = pd.Series(index=vix_data.index, dtype=object)
            for date, vix in vix_data.items():
                for regime_name, (lower, upper) in vix_regimes.items():
                    if lower <= vix < upper:
                        vix_regime_labels.loc[date] = regime_name
                        break
            
            vol_regime_data = vix_regime_labels
        else:
            # Use IV percentiles from trades to define regimes
            regime_source = "IV Percentile"
            
            # Extract trades with date and IV percentile
            trades_with_iv = []
            for trade in self.trades_history:
                if trade.get('iv_percentile') is None or not trade.get('timestamp'):
                    continue
                
                # Extract date from timestamp
                if isinstance(trade['timestamp'], str):
                    trade_date = trade['timestamp'].split('T')[0]
                elif isinstance(trade['timestamp'], datetime):
                    trade_date = trade['timestamp'].strftime('%Y-%m-%d')
                else:
                    continue
                    
                trades_with_iv.append({
                    'date': trade_date,
                    'iv_percentile': trade['iv_percentile']
                })
            
            if not trades_with_iv:
                return {"error": "No trades with IV percentile data"}
            
            # Convert to DataFrame
            iv_df = pd.DataFrame(trades_with_iv)
            iv_df['date'] = pd.to_datetime(iv_df['date'])
            iv_df.set_index('date', inplace=True)
            
            # Define IV percentile regime thresholds
            iv_regimes = {
                'very_low': (0, 0.2),
                'low': (0.2, 0.4),
                'normal': (0.4, 0.6),
                'high': (0.6, 0.8),
                'very_high': (0.8, 1.0)
            }
            
            # Create regime labels
            iv_regime_labels = pd.Series(index=iv_df.index, dtype=object)
            for date, row in iv_df.iterrows():
                for regime_name, (lower, upper) in iv_regimes.items():
                    if lower <= row['iv_percentile'] < upper:
                        iv_regime_labels.loc[date] = regime_name
                        break
            
            vol_regime_data = iv_regime_labels
        
        # Now analyze performance by volatility regime
        result = self.analyze_performance_by_market_regime(vol_regime_data)
        result['regime_source'] = regime_source
        return result
    
    def correlate_regime_with_greeks_exposure(self, regime_data: pd.Series) -> Dict[str, Any]:
        """
        Correlate market regime with Greeks exposure
        
        Args:
            regime_data: Series with regime labels indexed by date
            
        Returns:
            Dictionary with correlation metrics
        """
        if not self.daily_greeks_snapshots:
            return {"error": "No Greeks data available"}
            
        if regime_data.empty:
            return {"error": "No regime data provided"}
        
        # Convert snapshots to DataFrame
        df = pd.DataFrame(self.daily_greeks_snapshots)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Add regime data
        # Ensure regime_data index is datetime
        if not isinstance(regime_data.index, pd.DatetimeIndex):
            regime_data.index = pd.to_datetime(regime_data.index)
        
        # Merge snapshots with regime data
        merged_df = df.copy()
        merged_df['regime'] = None
        
        # Assign regime to each snapshot
        for date, row in merged_df.iterrows():
            # Find the closest regime date that is less than or equal to the snapshot date
            regime_dates = regime_data.index[regime_data.index <= date]
            if len(regime_dates) > 0:
                closest_date = regime_dates[-1]
                merged_df.at[date, 'regime'] = regime_data.loc[closest_date]
        
        # Remove snapshots without regime
        merged_df = merged_df.dropna(subset=['regime'])
        
        if merged_df.empty:
            return {"error": "No Greeks snapshots with matching regime data"}
        
        # Analyze Greeks by regime
        greek_columns = ['delta', 'gamma', 'theta', 'vega', 'rho']
        available_greeks = [g for g in greek_columns if g in merged_df.columns]
        
        if not available_greeks:
            return {"error": "No Greek data found in snapshots"}
        
        # Calculate average Greeks by regime
        regime_greeks = {}
        
        for regime in merged_df['regime'].unique():
            regime_snapshots = merged_df[merged_df['regime'] == regime]
            
            regime_greeks[regime] = {
                greek: {
                    'avg': regime_snapshots[greek].mean() if greek in regime_snapshots.columns else None,
                    'min': regime_snapshots[greek].min() if greek in regime_snapshots.columns else None,
                    'max': regime_snapshots[greek].max() if greek in regime_snapshots.columns else None,
                    'std': regime_snapshots[greek].std() if greek in regime_snapshots.columns else None
                } for greek in available_greeks
            }
        
        # Calculate optimal Greeks by regime based on performance
        # This requires trade data to be correlated with Greeks and regime
        optimal_greeks = self._calculate_optimal_greeks_by_regime(regime_data)
        
        return {
            'regime_greeks': regime_greeks,
            'optimal_greeks': optimal_greeks,
            'regimes_analyzed': list(regime_greeks.keys()),
            'greeks_analyzed': available_greeks
        }
    
    def _calculate_optimal_greeks_by_regime(self, regime_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate optimal Greeks exposure by regime based on historical performance
        
        Args:
            regime_data: Series with regime labels indexed by date
            
        Returns:
            Dictionary with optimal Greeks by regime
        """
        if not self.trades_history or not self.greeks_history:
            return {}
        
        # Extract trades with date, PnL, and Greeks
        trades_with_data = []
        for trade in self.trades_history:
            if not trade.get('pnl') or not trade.get('timestamp') or not trade.get('trade_id'):
                continue
            
            # Find matching Greeks snapshot
            matching_snapshot = None
            for snapshot in self.greeks_history:
                if snapshot.get('trade_id') == trade.get('trade_id'):
                    matching_snapshot = snapshot
                    break
            
            if not matching_snapshot:
                continue
            
            # Extract date from timestamp
            if isinstance(trade['timestamp'], str):
                trade_date = trade['timestamp'].split('T')[0]
            elif isinstance(trade['timestamp'], datetime):
                trade_date = trade['timestamp'].strftime('%Y-%m-%d')
            else:
                continue
                
            trades_with_data.append({
                'date': trade_date,
                'pnl': trade['pnl'],
                'portfolio_delta': matching_snapshot.get('portfolio_delta'),
                'portfolio_gamma': matching_snapshot.get('portfolio_gamma'),
                'portfolio_theta': matching_snapshot.get('portfolio_theta'),
                'portfolio_vega': matching_snapshot.get('portfolio_vega'),
                'portfolio_rho': matching_snapshot.get('portfolio_rho')
            })
        
        if not trades_with_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades_with_data)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df.set_index('date', inplace=True)
        
        # Add regime data
        # Ensure regime_data index is datetime
        if not isinstance(regime_data.index, pd.DatetimeIndex):
            regime_data.index = pd.to_datetime(regime_data.index)
        
        # Merge trades with regime data
        merged_df = trades_df.copy()
        merged_df['regime'] = None
        
        # Assign regime to each trade
        for date, row in merged_df.iterrows():
            # Find the closest regime date that is less than or equal to the trade date
            regime_dates = regime_data.index[regime_data.index <= date]
            if len(regime_dates) > 0:
                closest_date = regime_dates[-1]
                merged_df.at[date, 'regime'] = regime_data.loc[closest_date]
        
        # Remove trades without regime
        merged_df = merged_df.dropna(subset=['regime'])
        
        if merged_df.empty:
            return {}
        
        # Define Greek columns
        greek_columns = ['portfolio_delta', 'portfolio_gamma', 'portfolio_theta', 'portfolio_vega', 'portfolio_rho']
        available_greeks = [g for g in greek_columns if g in merged_df.columns]
        
        # Find optimal Greeks by regime
        optimal_greeks = {}
        
        for regime in merged_df['regime'].unique():
            regime_trades = merged_df[merged_df['regime'] == regime]
            
            if len(regime_trades) < 5:  # Need enough trades for meaningful analysis
                continue
            
            # Sort trades by PnL
            top_trades = regime_trades.sort_values('pnl', ascending=False).head(max(3, int(len(regime_trades) * 0.2)))
            
            # Calculate average Greeks of top performing trades
            optimal_greeks[regime] = {
                greek.replace('portfolio_', ''): {
                    'optimal_value': top_trades[greek].mean() if greek in top_trades.columns else None,
                    'range_min': top_trades[greek].quantile(0.25) if greek in top_trades.columns else None,
                    'range_max': top_trades[greek].quantile(0.75) if greek in top_trades.columns else None
                } for greek in available_greeks
            }
        
        return optimal_greeks
    
    def generate_volatility_surface_analysis(self, vix_data: pd.Series = None) -> Dict[str, Any]:
        """
        Analyze performance across the volatility surface (strike vs expiration)
        
        Args:
            vix_data: Optional Series with VIX index values indexed by date
            
        Returns:
            Dictionary with performance metrics across the volatility surface
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Extract trades with strike, expiration, and PnL
        trades_with_data = []
        for trade in self.trades_history:
            if not trade.get('pnl') or not trade.get('strike') or not trade.get('dte_entry'):
                continue
                
            trades_with_data.append({
                'pnl': trade.get('pnl', 0),
                'strike': trade.get('strike', 0),
                'dte': trade.get('dte_entry', 0),
                'option_type': trade.get('option_type', 'unknown'),
                'side': trade.get('side', 'unknown'),
                'moneyness': trade.get('moneyness', 0),  # Strike / Underlying price ratio
                'iv_percentile': trade.get('iv_percentile', 0),
                'timestamp': trade.get('timestamp', ''),
                'underlying': trade.get('underlying', 'unknown')
            })
        
        if not trades_with_data:
            return {"error": "No trades with strike, expiration, and PnL data"}
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades_with_data)
        
        # Calculate moneyness if not provided
        if all(m == 0 for m in trades_df['moneyness']):
            logger.warning("Moneyness data not available in trades, analysis will be limited")
        
        # Define moneyness buckets
        trades_df['moneyness_bucket'] = pd.cut(
            trades_df['moneyness'], 
            bins=[0, 0.85, 0.95, 1.05, 1.15, float('inf')],
            labels=['deep_itm', 'itm', 'atm', 'otm', 'deep_otm']
        )
        
        # Define DTE buckets
        trades_df['dte_bucket'] = pd.cut(
            trades_df['dte'], 
            bins=[0, 7, 14, 30, 60, float('inf')],
            labels=['0-7', '8-14', '15-30', '31-60', '61+']
        )
        
        # Calculate IV regimes if we have VIX data
        if vix_data is not None and not vix_data.empty:
            # Convert timestamp to datetime
            trades_df['date'] = pd.to_datetime(trades_df['timestamp'])
            
            # Ensure VIX index is datetime
            if not isinstance(vix_data.index, pd.DatetimeIndex):
                vix_data.index = pd.to_datetime(vix_data.index)
            
            # Add VIX data
            trades_df['vix'] = None
            
            # Assign VIX to each trade
            for i, row in trades_df.iterrows():
                if pd.isna(row['date']):
                    continue
                    
                # Find the closest VIX date that is less than or equal to the trade date
                vix_dates = vix_data.index[vix_data.index <= row['date']]
                if len(vix_dates) > 0:
                    closest_date = vix_dates[-1]
                    trades_df.at[i, 'vix'] = vix_data.loc[closest_date]
            
            # Define VIX buckets
            trades_df['vix_bucket'] = pd.cut(
                trades_df['vix'].astype(float), 
                bins=[0, 12, 16, 22, 30, float('inf')],
                labels=['very_low', 'low', 'normal', 'high', 'very_high']
            )
        else:
            # Use IV percentile as proxy
            trades_df['vix_bucket'] = pd.cut(
                trades_df['iv_percentile'], 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['very_low', 'low', 'normal', 'high', 'very_high']
            )
        
        # Calculate performance metrics across the surface
        surface_metrics = {}
        
        # 1. Performance by moneyness and DTE
        moneyness_dte_matrix = {}
        
        for moneyness in trades_df['moneyness_bucket'].dropna().unique():
            moneyness_dte_matrix[moneyness] = {}
            
            for dte in trades_df['dte_bucket'].dropna().unique():
                subset = trades_df[(trades_df['moneyness_bucket'] == moneyness) & 
                                 (trades_df['dte_bucket'] == dte)]
                
                if len(subset) >= 3:  # Need enough trades for meaningful metrics
                    moneyness_dte_matrix[moneyness][dte] = {
                        'trade_count': len(subset),
                        'avg_pnl': subset['pnl'].mean(),
                        'win_rate': len(subset[subset['pnl'] > 0]) / len(subset),
                        'best_trade': subset['pnl'].max(),
                        'worst_trade': subset['pnl'].min()
                    }
        
        surface_metrics['moneyness_dte_matrix'] = moneyness_dte_matrix
        
        # 2. Performance by moneyness and volatility
        moneyness_vol_matrix = {}
        
        for moneyness in trades_df['moneyness_bucket'].dropna().unique():
            moneyness_vol_matrix[moneyness] = {}
            
            for vol in trades_df['vix_bucket'].dropna().unique():
                subset = trades_df[(trades_df['moneyness_bucket'] == moneyness) & 
                                 (trades_df['vix_bucket'] == vol)]
                
                if len(subset) >= 3:  # Need enough trades for meaningful metrics
                    moneyness_vol_matrix[moneyness][vol] = {
                        'trade_count': len(subset),
                        'avg_pnl': subset['pnl'].mean(),
                        'win_rate': len(subset[subset['pnl'] > 0]) / len(subset),
                        'best_trade': subset['pnl'].max(),
                        'worst_trade': subset['pnl'].min()
                    }
        
        surface_metrics['moneyness_vol_matrix'] = moneyness_vol_matrix
        
        # 3. Performance by DTE and volatility
        dte_vol_matrix = {}
        
        for dte in trades_df['dte_bucket'].dropna().unique():
            dte_vol_matrix[dte] = {}
            
            for vol in trades_df['vix_bucket'].dropna().unique():
                subset = trades_df[(trades_df['dte_bucket'] == dte) & 
                                 (trades_df['vix_bucket'] == vol)]
                
                if len(subset) >= 3:  # Need enough trades for meaningful metrics
                    dte_vol_matrix[dte][vol] = {
                        'trade_count': len(subset),
                        'avg_pnl': subset['pnl'].mean(),
                        'win_rate': len(subset[subset['pnl'] > 0]) / len(subset),
                        'best_trade': subset['pnl'].max(),
                        'worst_trade': subset['pnl'].min()
                    }
        
        surface_metrics['dte_vol_matrix'] = dte_vol_matrix
        
        # 4. Best and worst combinations
        all_combinations = []
        
        for moneyness in trades_df['moneyness_bucket'].dropna().unique():
            for dte in trades_df['dte_bucket'].dropna().unique():
                for vol in trades_df['vix_bucket'].dropna().unique():
                    subset = trades_df[(trades_df['moneyness_bucket'] == moneyness) & 
                                     (trades_df['dte_bucket'] == dte) &
                                     (trades_df['vix_bucket'] == vol)]
                    
                    if len(subset) >= 3:  # Need enough trades for meaningful metrics
                        all_combinations.append({
                            'moneyness': moneyness,
                            'dte': dte,
                            'volatility': vol,
                            'trade_count': len(subset),
                            'avg_pnl': subset['pnl'].mean(),
                            'win_rate': len(subset[subset['pnl'] > 0]) / len(subset),
                            'total_pnl': subset['pnl'].sum()
                        })
        
        # Sort by average PnL
        all_combinations.sort(key=lambda x: x['avg_pnl'], reverse=True)
        
        surface_metrics['best_combinations'] = all_combinations[:5] if len(all_combinations) >= 5 else all_combinations
        surface_metrics['worst_combinations'] = all_combinations[-5:] if len(all_combinations) >= 5 else []
        
        return {
            'surface_metrics': surface_metrics,
            'trade_count': len(trades_df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def analyze_execution_quality(self) -> Dict[str, Any]:
        """
        Analyze execution quality for options trades by tracking bid-ask spreads and fill quality
        
        Returns:
            Dictionary with execution quality metrics
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Filter trades with execution quality data
        trades_with_execution = [
            t for t in self.trades_history 
            if 'execution_quality' in t and isinstance(t['execution_quality'], dict)
        ]
        
        if not trades_with_execution:
            return {"error": "No trades with execution quality data"}
        
        # Extract key metrics
        metrics = []
        for trade in trades_with_execution:
            exec_data = trade['execution_quality']
            
            # Skip if missing critical data
            if 'quoted_price' not in exec_data or 'executed_price' not in exec_data:
                continue
            
            quoted_price = exec_data['quoted_price']
            executed_price = exec_data['executed_price']
            
            # Calculate slippage
            slippage = executed_price - quoted_price if trade['side'] == 'LONG' else quoted_price - executed_price
            slippage_pct = (slippage / quoted_price) * 100 if quoted_price != 0 else 0
            
            # Calculate bid-ask metrics
            bid_ask_spread = exec_data.get('ask', 0) - exec_data.get('bid', 0)
            spread_pct = (bid_ask_spread / ((exec_data.get('ask', 0) + exec_data.get('bid', 0)) / 2)) * 100 if exec_data.get('ask', 0) + exec_data.get('bid', 0) > 0 else 0
            
            # Calculate fill quality (where in the spread did we get filled)
            fill_quality = 0
            if bid_ask_spread > 0 and 'bid' in exec_data and 'ask' in exec_data:
                if trade['side'] == 'LONG':
                    # For buys, lower is better (closer to bid)
                    fill_quality = (executed_price - exec_data['bid']) / bid_ask_spread
                else:
                    # For sells, higher is better (closer to ask)
                    fill_quality = (exec_data['ask'] - executed_price) / bid_ask_spread
            
            # Get trade details
            metrics.append({
                'trade_id': trade.get('trade_id', ''),
                'timestamp': trade.get('timestamp', ''),
                'symbol': trade.get('symbol', ''),
                'strategy': trade.get('strategy', 'unknown'),
                'side': trade.get('side', ''),
                'option_type': trade.get('option_type', ''),
                'quoted_price': quoted_price,
                'executed_price': executed_price,
                'slippage': slippage,
                'slippage_pct': slippage_pct,
                'bid_ask_spread': bid_ask_spread,
                'spread_pct': spread_pct,
                'fill_quality': fill_quality,
                'execution_time_ms': exec_data.get('execution_time_ms', 0),
                'volume': exec_data.get('volume', 0),
                'open_interest': exec_data.get('open_interest', 0)
            })
        
        if not metrics:
            return {"error": "No trades with sufficient execution data"}
        
        # Convert to DataFrame for analysis
        exec_df = pd.DataFrame(metrics)
        
        # Calculate overall metrics
        overall_metrics = {
            'avg_slippage_pct': exec_df['slippage_pct'].mean(),
            'avg_spread_pct': exec_df['spread_pct'].mean(),
            'avg_fill_quality': exec_df['fill_quality'].mean(),
            'avg_execution_time_ms': exec_df['execution_time_ms'].mean(),
            'best_fill': exec_df['fill_quality'].min(),
            'worst_fill': exec_df['fill_quality'].max(),
            'trade_count': len(exec_df)
        }
        
        # Calculate metrics by strategy
        strategy_metrics = {}
        for strategy in exec_df['strategy'].unique():
            strategy_df = exec_df[exec_df['strategy'] == strategy]
            
            strategy_metrics[strategy] = {
                'avg_slippage_pct': strategy_df['slippage_pct'].mean(),
                'avg_spread_pct': strategy_df['spread_pct'].mean(),
                'avg_fill_quality': strategy_df['fill_quality'].mean(),
                'avg_execution_time_ms': strategy_df['execution_time_ms'].mean(),
                'trade_count': len(strategy_df)
            }
        
        # Calculate metrics by option type
        option_type_metrics = {}
        for option_type in exec_df['option_type'].unique():
            if not option_type:
                continue
                
            type_df = exec_df[exec_df['option_type'] == option_type]
            
            option_type_metrics[option_type] = {
                'avg_slippage_pct': type_df['slippage_pct'].mean(),
                'avg_spread_pct': type_df['spread_pct'].mean(),
                'avg_fill_quality': type_df['fill_quality'].mean(),
                'avg_execution_time_ms': type_df['execution_time_ms'].mean(),
                'trade_count': len(type_df)
            }
        
        # Calculate metrics by side
        side_metrics = {}
        for side in exec_df['side'].unique():
            if not side:
                continue
                
            side_df = exec_df[exec_df['side'] == side]
            
            side_metrics[side] = {
                'avg_slippage_pct': side_df['slippage_pct'].mean(),
                'avg_spread_pct': side_df['spread_pct'].mean(),
                'avg_fill_quality': side_df['fill_quality'].mean(),
                'avg_execution_time_ms': side_df['execution_time_ms'].mean(),
                'trade_count': len(side_df)
            }
        
        # Calculate impact on P&L
        pnl_impact = self._calculate_execution_pnl_impact(exec_df)
        
        return {
            'overall_metrics': overall_metrics,
            'strategy_metrics': strategy_metrics,
            'option_type_metrics': option_type_metrics,
            'side_metrics': side_metrics,
            'pnl_impact': pnl_impact,
            'trades_analyzed': len(exec_df)
        }
    
    def _calculate_execution_pnl_impact(self, exec_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate the impact of execution quality on P&L
        
        Args:
            exec_df: DataFrame with execution metrics
            
        Returns:
            Dictionary with P&L impact metrics
        """
        # Match execution data with trade results
        trades_with_both = []
        
        for _, row in exec_df.iterrows():
            trade_id = row['trade_id']
            
            # Find matching trade
            matching_trade = None
            for trade in self.trades_history:
                if trade.get('trade_id') == trade_id and trade.get('pnl') is not None:
                    matching_trade = trade
                    break
            
            if matching_trade:
                trades_with_both.append({
                    'trade_id': trade_id,
                    'pnl': matching_trade['pnl'],
                    'slippage': row['slippage'],
                    'slippage_pct': row['slippage_pct'],
                    'fill_quality': row['fill_quality'],
                    'spread_pct': row['spread_pct']
                })
        
        if not trades_with_both:
            return {}
        
        # Convert to DataFrame
        impact_df = pd.DataFrame(trades_with_both)
        
        # Calculate correlation between execution metrics and P&L
        slippage_corr = impact_df['slippage'].corr(impact_df['pnl'])
        fill_quality_corr = impact_df['fill_quality'].corr(impact_df['pnl'])
        spread_corr = impact_df['spread_pct'].corr(impact_df['pnl'])
        
        # Estimate total P&L impact of slippage
        total_slippage_cost = impact_df['slippage'].sum()
        avg_slippage_per_trade = impact_df['slippage'].mean()
        slippage_pct_of_pnl = (total_slippage_cost / impact_df['pnl'].sum()) * 100 if impact_df['pnl'].sum() != 0 else 0
        
        # Calculate theoretical P&L without slippage
        theoretical_pnl = impact_df['pnl'].sum() + total_slippage_cost
        pnl_improvement_pct = (theoretical_pnl / impact_df['pnl'].sum() - 1) * 100 if impact_df['pnl'].sum() != 0 else 0
        
        return {
            'slippage_correlation': slippage_corr,
            'fill_quality_correlation': fill_quality_corr,
            'spread_correlation': spread_corr,
            'total_slippage_cost': total_slippage_cost,
            'avg_slippage_per_trade': avg_slippage_per_trade,
            'slippage_pct_of_pnl': slippage_pct_of_pnl,
            'theoretical_pnl_without_slippage': theoretical_pnl,
            'potential_pnl_improvement_pct': pnl_improvement_pct
        }
    
    def portfolio_contribution_analysis(self) -> Dict[str, Any]:
        """
        Analyze contribution of different options strategies and positions to overall performance and risk
        
        Returns:
            Dictionary with contribution analysis
        """
        if not self.trades_history:
            return {"error": "No trade data available"}
        
        # Filter trades with PnL
        closed_trades = [t for t in self.trades_history if t.get('pnl') is not None]
        
        if not closed_trades:
            return {"error": "No closed trades with PnL data"}
        
        # Create DataFrame for analysis
        trades_df = pd.DataFrame([
            {
                'trade_id': t.get('trade_id', ''),
                'timestamp': t.get('timestamp', ''),
                'symbol': t.get('symbol', ''),
                'underlying': t.get('underlying', ''),
                'strategy': t.get('strategy', 'unknown'),
                'option_type': t.get('option_type', ''),
                'side': t.get('side', ''),
                'pnl': t.get('pnl', 0),
                'days_held': t.get('days_held', 0),
                'max_risk': t.get('max_risk', 0)
            } for t in closed_trades
        ])
        
        # Calculate total PnL
        total_pnl = trades_df['pnl'].sum()
        
        # Contribution by strategy
        strategy_contribution = {}
        for strategy in trades_df['strategy'].unique():
            strategy_df = trades_df[trades_df['strategy'] == strategy]
            
            strategy_pnl = strategy_df['pnl'].sum()
            contribution_pct = (strategy_pnl / total_pnl) * 100 if total_pnl != 0 else 0
            win_rate = len(strategy_df[strategy_df['pnl'] > 0]) / len(strategy_df)
            
            # Calculate risk efficiency (PnL per unit of risk)
            if 'max_risk' in strategy_df.columns and strategy_df['max_risk'].sum() > 0:
                risk_efficiency = strategy_pnl / strategy_df['max_risk'].sum()
            else:
                risk_efficiency = None
            
            # Average holding period
            avg_holding_period = strategy_df['days_held'].mean()
            
            # Annualized return
            if avg_holding_period > 0:
                ann_factor = 365 / avg_holding_period
                ann_return = ((1 + (strategy_pnl / strategy_df['max_risk'].sum())) ** ann_factor - 1) * 100 if strategy_df['max_risk'].sum() > 0 else None
            else:
                ann_return = None
            
            strategy_contribution[strategy] = {
                'pnl': strategy_pnl,
                'contribution_pct': contribution_pct,
                'trade_count': len(strategy_df),
                'win_rate': win_rate,
                'avg_pnl_per_trade': strategy_pnl / len(strategy_df),
                'risk_efficiency': risk_efficiency,
                'avg_holding_period': avg_holding_period,
                'annualized_return': ann_return
            }
        
        # Contribution by underlying
        underlying_contribution = {}
        for underlying in trades_df['underlying'].unique():
            if not underlying:
                continue
                
            underlying_df = trades_df[trades_df['underlying'] == underlying]
            
            underlying_pnl = underlying_df['pnl'].sum()
            contribution_pct = (underlying_pnl / total_pnl) * 100 if total_pnl != 0 else 0
            
            underlying_contribution[underlying] = {
                'pnl': underlying_pnl,
                'contribution_pct': contribution_pct,
                'trade_count': len(underlying_df),
                'win_rate': len(underlying_df[underlying_df['pnl'] > 0]) / len(underlying_df)
            }
        
        # Contribution by option type
        option_type_contribution = {}
        for option_type in trades_df['option_type'].unique():
            if not option_type:
                continue
                
            type_df = trades_df[trades_df['option_type'] == option_type]
            
            type_pnl = type_df['pnl'].sum()
            contribution_pct = (type_pnl / total_pnl) * 100 if total_pnl != 0 else 0
            
            option_type_contribution[option_type] = {
                'pnl': type_pnl,
                'contribution_pct': contribution_pct,
                'trade_count': len(type_df),
                'win_rate': len(type_df[type_df['pnl'] > 0]) / len(type_df)
            }
        
        # Contribution by side
        side_contribution = {}
        for side in trades_df['side'].unique():
            if not side:
                continue
                
            side_df = trades_df[trades_df['side'] == side]
            
            side_pnl = side_df['pnl'].sum()
            contribution_pct = (side_pnl / total_pnl) * 100 if total_pnl != 0 else 0
            
            side_contribution[side] = {
                'pnl': side_pnl,
                'contribution_pct': contribution_pct,
                'trade_count': len(side_df),
                'win_rate': len(side_df[side_df['pnl'] > 0]) / len(side_df)
            }
        
        # Calculate Herfindahl Index for diversification
        if total_pnl != 0:
            strategy_weights = [(s['pnl'] / total_pnl) ** 2 for s in strategy_contribution.values()]
            strategy_herfindahl = sum(strategy_weights)
            
            underlying_weights = [(u['pnl'] / total_pnl) ** 2 for u in underlying_contribution.values()]
            underlying_herfindahl = sum(underlying_weights)
        else:
            strategy_herfindahl = None
            underlying_herfindahl = None
        
        # Calculate effective number of positions
        effective_n_strategies = 1 / strategy_herfindahl if strategy_herfindahl else None
        effective_n_underlyings = 1 / underlying_herfindahl if underlying_herfindahl else None
        
        # Calculate risk contribution if we have Greeks data
        risk_contribution = self._calculate_risk_contribution() if self.daily_greeks_snapshots else {}
        
        return {
            'total_pnl': total_pnl,
            'total_trades': len(trades_df),
            'strategy_contribution': strategy_contribution,
            'underlying_contribution': underlying_contribution,
            'option_type_contribution': option_type_contribution,
            'side_contribution': side_contribution,
            'diversification': {
                'strategy_herfindahl': strategy_herfindahl,
                'underlying_herfindahl': underlying_herfindahl,
                'effective_n_strategies': effective_n_strategies,
                'effective_n_underlyings': effective_n_underlyings
            },
            'risk_contribution': risk_contribution
        }
    
    def _calculate_risk_contribution(self) -> Dict[str, Any]:
        """
        Calculate risk contribution based on Greeks exposure
        
        Returns:
            Dictionary with risk contribution metrics
        """
        if not self.daily_greeks_snapshots:
            return {}
        
        # Get most recent Greeks snapshot
        latest_snapshot = max(self.daily_greeks_snapshots, key=lambda x: x.get('timestamp', ''))
        
        # Check if we have position-level Greeks in the snapshot
        if 'positions' not in latest_snapshot:
            return {}
        
        positions = latest_snapshot['positions']
        
        # Calculate total absolute exposure for each Greek
        total_delta_abs = sum(abs(p.get('delta', 0)) for p in positions)
        total_gamma_abs = sum(abs(p.get('gamma', 0)) for p in positions)
        total_theta_abs = sum(abs(p.get('theta', 0)) for p in positions)
        total_vega_abs = sum(abs(p.get('vega', 0)) for p in positions)
        
        # Calculate risk contribution by position
        position_risk = {}
        
        for position in positions:
            symbol = position.get('symbol', 'unknown')
            
            delta_contrib = (abs(position.get('delta', 0)) / total_delta_abs) * 100 if total_delta_abs > 0 else 0
            gamma_contrib = (abs(position.get('gamma', 0)) / total_gamma_abs) * 100 if total_gamma_abs > 0 else 0
            theta_contrib = (abs(position.get('theta', 0)) / total_theta_abs) * 100 if total_theta_abs > 0 else 0
            vega_contrib = (abs(position.get('vega', 0)) / total_vega_abs) * 100 if total_vega_abs > 0 else 0
            
            # Calculate weighted average contribution
            total_contrib = (delta_contrib + gamma_contrib + theta_contrib + vega_contrib) / 4
            
            position_risk[symbol] = {
                'delta_contrib': delta_contrib,
                'gamma_contrib': gamma_contrib,
                'theta_contrib': theta_contrib,
                'vega_contrib': vega_contrib,
                'total_contrib': total_contrib
            }
        
        # Calculate risk contribution by strategy
        strategy_risk = {}
        
        for position in positions:
            symbol = position.get('symbol', 'unknown')
            strategy = position.get('strategy', 'unknown')
            
            if strategy not in strategy_risk:
                strategy_risk[strategy] = {
                    'delta': 0,
                    'gamma': 0,
                    'theta': 0,
                    'vega': 0
                }
            
            strategy_risk[strategy]['delta'] += position.get('delta', 0)
            strategy_risk[strategy]['gamma'] += position.get('gamma', 0)
            strategy_risk[strategy]['theta'] += position.get('theta', 0)
            strategy_risk[strategy]['vega'] += position.get('vega', 0)
        
        # Calculate percentage contributions
        for strategy in strategy_risk:
            strategy_risk[strategy]['delta_contrib'] = (abs(strategy_risk[strategy]['delta']) / total_delta_abs) * 100 if total_delta_abs > 0 else 0
            strategy_risk[strategy]['gamma_contrib'] = (abs(strategy_risk[strategy]['gamma']) / total_gamma_abs) * 100 if total_gamma_abs > 0 else 0
            strategy_risk[strategy]['theta_contrib'] = (abs(strategy_risk[strategy]['theta']) / total_theta_abs) * 100 if total_theta_abs > 0 else 0
            strategy_risk[strategy]['vega_contrib'] = (abs(strategy_risk[strategy]['vega']) / total_vega_abs) * 100 if total_vega_abs > 0 else 0
            
            # Calculate weighted average contribution
            strategy_risk[strategy]['total_contrib'] = (
                strategy_risk[strategy]['delta_contrib'] +
                strategy_risk[strategy]['gamma_contrib'] +
                strategy_risk[strategy]['theta_contrib'] +
                strategy_risk[strategy]['vega_contrib']
            ) / 4
        
        return {
            'position_risk_contribution': position_risk,
            'strategy_risk_contribution': strategy_risk,
            'snapshot_timestamp': latest_snapshot.get('timestamp', '')
        }