#!/usr/bin/env python
"""
Nightly Recap System

This module implements a comprehensive nightly performance analysis, benchmarking,
and feedback system that:
1. Compares daily P&L against benchmarks (SPY, VIX)
2. Analyzes performance metrics over rolling windows
3. Flags strategies with deteriorating performance
4. Generates actionable insights and recommendations
5. Sends a daily summary report via email
6. Optionally triggers optimization jobs automatically
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import yfinance as yf
import traceback
import argparse
import subprocess

# Configure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import trading system components
from trading_bot.dashboard.paper_trading_dashboard import PaperTradingDashboard
from trading_bot.execution.adaptive_paper_integration import get_paper_trading_instance
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'nightly_recap.log'))
    ]
)

logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class NightlyRecap:
    """
    Nightly performance recap and feedback system that analyzes trading
    performance, generates insights, and provides optimization suggestions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the nightly recap system
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        
        # Create output directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('reports/nightly', exist_ok=True)
        
        # Set default thresholds if not in config
        if 'thresholds' not in self.config:
            self.config['thresholds'] = {
                'sharpe_ratio': 0.5,
                'win_rate': 45.0,  # percentage
                'max_drawdown': -10.0,  # percentage
                'rolling_windows': [5, 10, 20, 60]  # days
            }
        
        # Initialize components
        self.dashboard = PaperTradingDashboard()
        self.paper_trading = get_paper_trading_instance()
        self.controller = None
        
        # Performance data
        self.today_results = None
        self.benchmark_data = None
        self.strategy_metrics = None
        self.alerts = []
        self.suggestions = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'email': {
                'enabled': True,
                'server': 'smtp.gmail.com',
                'port': 587,
                'username': '',
                'password': '',
                'recipients': []
            },
            'benchmarks': ['SPY', 'VIX'],
            'thresholds': {
                'sharpe_ratio': 0.5,
                'win_rate': 45.0,  # percentage
                'max_drawdown': -10.0,  # percentage
                'rolling_windows': [5, 10, 20, 60]  # days
            },
            'optimization': {
                'auto_optimize': False,
                'optimization_threshold': -20.0  # percentage deterioration
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                
                logger.info(f"Loaded configuration from {config_path}")
                return config
            
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.warning("Using default configuration")
                return default_config
        else:
            logger.info("Using default configuration")
            return default_config
    
    def run_nightly_recap(self, force_date: Optional[str] = None):
        """
        Run the nightly recap process
        
        Args:
            force_date: Force a specific date (YYYY-MM-DD) for testing
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting nightly recap at {start_time}")
            
            # 1. Load trading data
            logger.info("Loading trading data...")
            self._load_trading_data()
            
            # 2. Fetch benchmark data
            logger.info("Fetching benchmark data...")
            self._fetch_benchmark_data(force_date)
            
            # 3. Calculate performance metrics
            logger.info("Calculating performance metrics...")
            self._calculate_performance_metrics()
            
            # 4. Analyze strategy performance
            logger.info("Analyzing strategy performance...")
            self._analyze_strategy_performance()
            
            # 5. Generate insights and suggestions
            logger.info("Generating insights and suggestions...")
            self._generate_insights()
            
            # 6. Create performance report
            logger.info("Creating performance report...")
            report_path = self._create_performance_report()
            
            # 7. Send email report
            if self.config['email']['enabled'] and self.config['email']['recipients']:
                logger.info("Sending email report...")
                self._send_email_report(report_path)
            
            # 8. Auto-optimize if configured
            if (self.config['optimization']['auto_optimize'] and 
                any(alert['action_required'] for alert in self.alerts)):
                logger.info("Triggering optimization jobs...")
                self._trigger_optimization_jobs()
            
            end_time = datetime.now()
            logger.info(f"Nightly recap completed in {end_time - start_time}")
            
            return True, "Nightly recap completed successfully"
            
        except Exception as e:
            error_msg = f"Error in nightly recap: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg
    
    def _load_trading_data(self):
        """
        Load trading data from paper trading system
        """
        # Try to get controller from paper trading
        if self.paper_trading:
            self.controller = self.paper_trading.get_controller()
        
        # Load data into dashboard
        success = self.dashboard.load_live_data()
        if not success:
            # If live data not available, load most recent session
            success = self.dashboard.load_data()
            
        if not success:
            raise Exception("Failed to load trading data")
        
        # Get equity and trade history
        self.equity_history = self.dashboard.equity_history
        self.trade_history = self.dashboard.trade_history
        
        # Calculate today's results
        self._calculate_today_results()
    
    def _calculate_today_results(self):
        """
        Calculate today's trading results
        """
        if self.equity_history is None or self.equity_history.empty:
            logger.warning("No equity data available")
            self.today_results = {
                'date': datetime.now().date(),
                'starting_equity': 0,
                'ending_equity': 0,
                'daily_pnl': 0,
                'daily_return': 0,
                'trades': 0,
                'winning_trades': 0,
                'win_rate': 0
            }
            return
        
        # Get today's date or most recent date in equity history
        today = datetime.now().date()
        if 'timestamp' in self.equity_history.columns:
            date_column = 'timestamp'
        else:
            # Find date column
            date_columns = [col for col in self.equity_history.columns 
                          if pd.api.types.is_datetime64_any_dtype(self.equity_history[col])]
            if date_columns:
                date_column = date_columns[0]
            else:
                logger.warning("No date column found in equity history")
                self.today_results = None
                return
        
        # Get most recent date
        most_recent_date = self.equity_history[date_column].max().date()
        
        # Filter for today's data
        today_equity = self.equity_history[self.equity_history[date_column].dt.date == most_recent_date]
        
        # If no data for today, use the most recent day
        if today_equity.empty:
            logger.warning(f"No equity data found for {today}, using most recent date")
            most_recent_dates = self.equity_history[date_column].dt.date.unique()
            most_recent_dates.sort()
            if len(most_recent_dates) > 0:
                most_recent_date = most_recent_dates[-1]
                today_equity = self.equity_history[self.equity_history[date_column].dt.date == most_recent_date]
        
        # Calculate daily metrics
        if not today_equity.empty:
            starting_equity = today_equity['equity'].iloc[0]
            ending_equity = today_equity['equity'].iloc[-1]
            daily_pnl = ending_equity - starting_equity
            daily_return = daily_pnl / starting_equity * 100
            
            # Get today's trades
            if self.trade_history is not None and not self.trade_history.empty:
                if 'exit_time' in self.trade_history.columns:
                    today_trades = self.trade_history[self.trade_history['exit_time'].dt.date == most_recent_date]
                    trades_count = len(today_trades)
                    winning_trades = len(today_trades[today_trades['pnl'] > 0])
                    win_rate = winning_trades / trades_count * 100 if trades_count > 0 else 0
                else:
                    trades_count = 0
                    winning_trades = 0
                    win_rate = 0
            else:
                trades_count = 0
                winning_trades = 0
                win_rate = 0
            
            self.today_results = {
                'date': most_recent_date,
                'starting_equity': starting_equity,
                'ending_equity': ending_equity,
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'trades': trades_count,
                'winning_trades': winning_trades,
                'win_rate': win_rate
            }
        else:
            logger.warning("No equity data available for today")
            self.today_results = None
    
    def _fetch_benchmark_data(self, force_date: Optional[str] = None):
        """
        Fetch benchmark data for comparison
        
        Args:
            force_date: Force a specific date (YYYY-MM-DD) for testing
        """
        benchmarks = self.config.get('benchmarks', ['SPY', 'VIX'])
        
        # Determine date range
        end_date = datetime.now().date()
        if force_date:
            try:
                end_date = datetime.strptime(force_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Invalid date format: {force_date}, using today's date")
        
        # Start date should be 30 days before end date to get enough context
        start_date = end_date - timedelta(days=30)
        
        try:
            # Fetch benchmark data
            benchmark_data = {}
            for symbol in benchmarks:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                
                if not hist.empty:
                    # Calculate daily returns
                    hist['Return'] = hist['Close'].pct_change() * 100
                    benchmark_data[symbol] = hist
                else:
                    logger.warning(f"No data available for {symbol}")
            
            self.benchmark_data = benchmark_data
            
            # Calculate today's benchmark performance
            self._calculate_benchmark_performance(end_date)
            
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {e}")
            self.benchmark_data = {}
    
    def _calculate_benchmark_performance(self, target_date: datetime.date):
        """
        Calculate benchmark performance for comparison
        
        Args:
            target_date: Target date for performance calculation
        """
        benchmark_performance = {}
        
        for symbol, data in self.benchmark_data.items():
            # Get data for target date
            target_data = data[data.index.date == target_date]
            
            # If no data for target date, get the most recent date
            if target_data.empty:
                available_dates = data.index.date
                available_dates = [d for d in available_dates if d <= target_date]
                if available_dates:
                    most_recent_date = max(available_dates)
                    target_data = data[data.index.date == most_recent_date]
            
            if not target_data.empty:
                daily_return = target_data['Return'].iloc[-1]
                close_price = target_data['Close'].iloc[-1]
                
                benchmark_performance[symbol] = {
                    'date': target_data.index[0].date(),
                    'close': close_price,
                    'daily_return': daily_return
                }
            else:
                logger.warning(f"No data available for {symbol} on or before {target_date}")
        
        self.benchmark_performance = benchmark_performance
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for all strategies
        """
        # Get data from controller or dashboard
        if self.controller:
            weight_history = self.controller.get_weight_history()
        else:
            weight_history = self.dashboard.weight_history
        
        if weight_history is None or weight_history.empty:
            logger.warning("No weight history available")
            self.strategy_metrics = {}
            return
        
        # Get strategy names (all columns except timestamp)
        strategy_names = [col for col in weight_history.columns if col != 'timestamp']
        
        # Initialize metrics dictionary
        self.strategy_metrics = {}
        
        # Get trade history by strategy
        trade_history_by_strategy = {}
        if self.trade_history is not None and not self.trade_history.empty:
            if 'strategy' in self.trade_history.columns:
                for strategy in strategy_names:
                    strategy_trades = self.trade_history[self.trade_history['strategy'] == strategy]
                    if not strategy_trades.empty:
                        trade_history_by_strategy[strategy] = strategy_trades
        
        # Calculate metrics for each strategy
        for strategy in strategy_names:
            # Get trades for this strategy
            strategy_trades = trade_history_by_strategy.get(strategy, pd.DataFrame())
            
            # Calculate metrics
            metrics = {}
            
            # Basic metrics
            if not strategy_trades.empty:
                metrics['trades_total'] = len(strategy_trades)
                metrics['trades_winning'] = len(strategy_trades[strategy_trades['pnl'] > 0])
                metrics['win_rate'] = metrics['trades_winning'] / metrics['trades_total'] * 100
                metrics['avg_pnl'] = strategy_trades['pnl'].mean()
                metrics['total_pnl'] = strategy_trades['pnl'].sum()
                
                # Calculate metrics for recent periods
                for window in self.config['thresholds']['rolling_windows']:
                    cutoff_date = datetime.now().date() - timedelta(days=window)
                    recent_trades = strategy_trades[strategy_trades['exit_time'].dt.date >= cutoff_date]
                    
                    if not recent_trades.empty:
                        metrics[f'trades_total_{window}d'] = len(recent_trades)
                        metrics[f'trades_winning_{window}d'] = len(recent_trades[recent_trades['pnl'] > 0])
                        metrics[f'win_rate_{window}d'] = metrics[f'trades_winning_{window}d'] / metrics[f'trades_total_{window}d'] * 100
                        metrics[f'avg_pnl_{window}d'] = recent_trades['pnl'].mean()
                        metrics[f'total_pnl_{window}d'] = recent_trades['pnl'].sum()
                    else:
                        metrics[f'trades_total_{window}d'] = 0
                        metrics[f'trades_winning_{window}d'] = 0
                        metrics[f'win_rate_{window}d'] = 0
                        metrics[f'avg_pnl_{window}d'] = 0
                        metrics[f'total_pnl_{window}d'] = 0
                
                # Calculate drawdown
                if 'equity' in strategy_trades.columns:
                    strategy_trades['peak'] = strategy_trades['equity'].cummax()
                    strategy_trades['drawdown'] = (strategy_trades['equity'] / strategy_trades['peak'] - 1) * 100
                    metrics['max_drawdown'] = strategy_trades['drawdown'].min()
                    
                    # Calculate drawdown for recent periods
                    for window in self.config['thresholds']['rolling_windows']:
                        cutoff_date = datetime.now().date() - timedelta(days=window)
                        recent_trades = strategy_trades[strategy_trades['exit_time'].dt.date >= cutoff_date]
                        
                        if not recent_trades.empty and 'equity' in recent_trades.columns:
                            recent_trades['peak'] = recent_trades['equity'].cummax()
                            recent_trades['drawdown'] = (recent_trades['equity'] / recent_trades['peak'] - 1) * 100
                            metrics[f'max_drawdown_{window}d'] = recent_trades['drawdown'].min()
                        else:
                            metrics[f'max_drawdown_{window}d'] = 0
                
                # Calculate Sharpe ratio
                if 'return' in strategy_trades.columns:
                    returns = strategy_trades['return'].values
                    if len(returns) > 1:
                        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
                        
                        # Calculate Sharpe for recent periods
                        for window in self.config['thresholds']['rolling_windows']:
                            cutoff_date = datetime.now().date() - timedelta(days=window)
                            recent_trades = strategy_trades[strategy_trades['exit_time'].dt.date >= cutoff_date]
                            
                            if not recent_trades.empty and 'return' in recent_trades.columns:
                                recent_returns = recent_trades['return'].values
                                if len(recent_returns) > 1:
                                    metrics[f'sharpe_ratio_{window}d'] = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                                else:
                                    metrics[f'sharpe_ratio_{window}d'] = 0
                            else:
                                metrics[f'sharpe_ratio_{window}d'] = 0
                    else:
                        metrics['sharpe_ratio'] = 0
                else:
                    # Approximate Sharpe from daily PnL if necessary
                    daily_pnl = strategy_trades.groupby(strategy_trades['exit_time'].dt.date)['pnl'].sum()
                    if len(daily_pnl) > 1:
                        metrics['sharpe_ratio'] = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                    else:
                        metrics['sharpe_ratio'] = 0
            else:
                # No trades for this strategy
                metrics['trades_total'] = 0
                metrics['trades_winning'] = 0
                metrics['win_rate'] = 0
                metrics['avg_pnl'] = 0
                metrics['total_pnl'] = 0
                metrics['max_drawdown'] = 0
                metrics['sharpe_ratio'] = 0
                
                for window in self.config['thresholds']['rolling_windows']:
                    metrics[f'trades_total_{window}d'] = 0
                    metrics[f'trades_winning_{window}d'] = 0
                    metrics[f'win_rate_{window}d'] = 0
                    metrics[f'avg_pnl_{window}d'] = 0
                    metrics[f'total_pnl_{window}d'] = 0
                    metrics[f'max_drawdown_{window}d'] = 0
                    metrics[f'sharpe_ratio_{window}d'] = 0
            
            # Get current weight
            current_weight = weight_history[strategy].iloc[-1] if not weight_history.empty else 0
            metrics['current_weight'] = current_weight
            
            # Store metrics
            self.strategy_metrics[strategy] = metrics
