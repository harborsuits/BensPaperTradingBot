"""
Performance Tracker

This module tracks and calculates performance metrics for strategies and positions,
providing data for the dynamic allocation and position sizing systems.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
import json
import os

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and calculates performance metrics for trading strategies.
    
    The PerformanceTracker maintains performance history for each strategy
    and calculates key metrics like Sharpe ratio, win rate, profit factor,
    and drawdown. These metrics are used by the Snowball allocation system
    and position sizer to optimize capital deployment.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the performance tracker.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Performance metric calculation parameters
        self.sharpe_window = self.config.get('sharpe_window', 30)  # Days for Sharpe ratio
        self.performance_windows = self.config.get('performance_windows', [7, 30, 90])  # Days
        self.min_trades_required = self.config.get('min_trades_required', 10)  # Min trades for reliable metrics
        self.risk_free_rate = self.config.get('risk_free_rate', 0.0)  # Risk-free rate for Sharpe (annualized)
        
        # Data storage
        self.trade_history = {}  # strategy_id -> list of trade dicts
        self.daily_returns = {}  # strategy_id -> DataFrame
        self.performance_metrics = {}  # strategy_id -> metrics dict
        
        # Data saving/loading
        self.data_dir = self.config.get('data_dir', './data/performance')
        self.auto_save = self.config.get('auto_save', True)
        self.save_interval = self.config.get('save_interval', 24*60*60)  # 24 hours
        self.last_save_time = datetime.now()
        
        # Ensure data directory exists
        if self.auto_save and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        # Try to load existing data
        self._load_data()
        
        logger.info(f"Initialized PerformanceTracker with {len(self.trade_history)} " 
                    f"strategies and sharpe_window={self.sharpe_window}")
    
    def record_trade(self, 
                    strategy_id: str, 
                    trade_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Record a new trade and update performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            trade_data: Trade information including:
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp
                - symbol: Trading symbol
                - direction: 'long' or 'short'
                - entry_price: Entry price
                - exit_price: Exit price
                - quantity: Position size
                - pnl: Profit/loss amount
                - pnl_pct: Percentage return
                - fees: Trading fees
                - slippage: Execution slippage
                - tags: Optional list of trade tags
                
        Returns:
            Dict with updated performance metrics for this strategy
        """
        # Initialize if first trade for this strategy
        if strategy_id not in self.trade_history:
            self.trade_history[strategy_id] = []
            
        # Add timestamp if not provided
        if 'entry_time' not in trade_data:
            trade_data['entry_time'] = datetime.now().isoformat()
        if 'exit_time' not in trade_data:
            trade_data['exit_time'] = datetime.now().isoformat()
            
        # Add trade to history
        self.trade_history[strategy_id].append(trade_data)
        
        # Update daily returns
        self._update_daily_returns(strategy_id)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(strategy_id)
        self.performance_metrics[strategy_id] = metrics
        
        # Auto-save if enabled
        if self.auto_save:
            time_since_save = (datetime.now() - self.last_save_time).total_seconds()
            if time_since_save > self.save_interval:
                self._save_data()
                
        return metrics
    
    def update_daily_equity(self,
                          strategy_id: str,
                          date: Optional[str] = None,
                          equity: float = None) -> None:
        """
        Update daily equity value for a strategy (for strategies without individual trades).
        
        Args:
            strategy_id: Strategy identifier
            date: Date string (YYYY-MM-DD), defaults to today
            equity: Equity value for this strategy
        """
        if equity is None:
            return
            
        date_str = date or datetime.now().strftime('%Y-%m-%d')
        
        # Initialize if first equity update for this strategy
        if strategy_id not in self.daily_returns:
            self.daily_returns[strategy_id] = pd.DataFrame(
                columns=['date', 'equity', 'returns']
            )
            
        df = self.daily_returns[strategy_id]
        
        # Check if we already have this date
        if date_str in df['date'].values:
            # Update existing entry
            idx = df.index[df['date'] == date_str][0]
            prev_idx = idx - 1 if idx > 0 else None
            
            df.at[idx, 'equity'] = equity
            
            # Calculate return if we have previous data
            if prev_idx is not None:
                prev_equity = df.at[prev_idx, 'equity']
                if prev_equity > 0:
                    daily_return = (equity / prev_equity) - 1
                    df.at[idx, 'returns'] = daily_return
        else:
            # Add new entry
            new_row = {'date': date_str, 'equity': equity, 'returns': 0.0}
            
            # Calculate return if we have previous data
            if len(df) > 0:
                prev_equity = df.iloc[-1]['equity']
                if prev_equity > 0:
                    new_row['returns'] = (equity / prev_equity) - 1
                    
            # Append row
            df = df.append(new_row, ignore_index=True)
            self.daily_returns[strategy_id] = df
            
        # Update metrics
        self._calculate_metrics(strategy_id)
        
        # Auto-save if enabled
        if self.auto_save:
            time_since_save = (datetime.now() - self.last_save_time).total_seconds()
            if time_since_save > self.save_interval:
                self._save_data()
    
    def get_metrics(self, 
                   strategy_id: str, 
                   window_days: Optional[int] = None) -> Dict[str, float]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            window_days: Optional lookback window in days
            
        Returns:
            Dict with performance metrics
        """
        if strategy_id not in self.performance_metrics:
            return {}
            
        metrics = self.performance_metrics[strategy_id]
        
        # If specific window requested and available
        if window_days is not None:
            window_key = f'{window_days}d'
            if window_key in metrics:
                return metrics[window_key]
                
        return metrics
    
    def get_all_metrics(self, window_days: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all strategies.
        
        Args:
            window_days: Optional lookback window in days
            
        Returns:
            Dict mapping strategy IDs to performance metrics
        """
        all_metrics = {}
        
        for strategy_id, metrics in self.performance_metrics.items():
            if window_days is not None:
                window_key = f'{window_days}d'
                if window_key in metrics:
                    all_metrics[strategy_id] = metrics[window_key]
                else:
                    all_metrics[strategy_id] = metrics
            else:
                all_metrics[strategy_id] = metrics
                
        return all_metrics
    
    def get_trade_history(self, 
                        strategy_id: str,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trade history for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries, most recent first
        """
        if strategy_id not in self.trade_history:
            return []
            
        # Sort by exit time descending and limit
        trades = sorted(
            self.trade_history[strategy_id],
            key=lambda x: x.get('exit_time', ''),
            reverse=True
        )
        
        return trades[:limit]
    
    def _update_daily_returns(self, strategy_id: str) -> None:
        """
        Update daily returns from trade history.
        
        Args:
            strategy_id: Strategy identifier
        """
        if not self.trade_history.get(strategy_id):
            return
            
        # Group trades by day
        daily_pnl = {}
        
        for trade in self.trade_history[strategy_id]:
            # Use exit time to determine day
            exit_time_str = trade.get('exit_time')
            if not exit_time_str:
                continue
                
            try:
                exit_time = datetime.fromisoformat(exit_time_str)
                date_str = exit_time.strftime('%Y-%m-%d')
                
                if date_str not in daily_pnl:
                    daily_pnl[date_str] = {'pnl': 0.0, 'trades': 0}
                    
                pnl_amount = trade.get('pnl', 0.0)
                daily_pnl[date_str]['pnl'] += pnl_amount
                daily_pnl[date_str]['trades'] += 1
                
            except (ValueError, TypeError):
                logger.warning(f"Invalid exit_time in trade: {exit_time_str}")
                continue
                
        # Convert to DataFrame
        if not daily_pnl:
            return
            
        dates = []
        pnls = []
        
        for date_str, data in sorted(daily_pnl.items()):
            dates.append(date_str)
            pnls.append(data['pnl'])
            
        df = pd.DataFrame({
            'date': dates,
            'pnl': pnls
        })
        
        # Build cumulative equity curve assuming starting equity of 10000
        starting_equity = 10000.0  # Arbitrary starting value
        df['equity'] = starting_equity + df['pnl'].cumsum()
        
        # Calculate returns
        df['returns'] = df['pnl'] / df['equity'].shift(1)
        df.loc[0, 'returns'] = df.loc[0, 'pnl'] / starting_equity
        
        self.daily_returns[strategy_id] = df
    
    def _calculate_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict with performance metrics
        """
        metrics = {
            'strategy_id': strategy_id,
            'last_updated': datetime.now().isoformat()
        }
        
        # Get trade history and daily returns
        trades = self.trade_history.get(strategy_id, [])
        returns_df = self.daily_returns.get(strategy_id)
        
        # Trade count statistics
        metrics['total_trades'] = len(trades)
        
        if len(trades) == 0:
            return metrics
            
        # Basic trade statistics
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        
        metrics['win_count'] = wins
        metrics['loss_count'] = losses
        metrics['win_rate'] = wins / len(trades) if len(trades) > 0 else 0
        
        # Profit metrics
        total_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        total_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        metrics['total_profit'] = total_profit
        metrics['total_loss'] = total_loss
        metrics['net_profit'] = total_profit - total_loss
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average trade metrics
        metrics['avg_profit'] = total_profit / wins if wins > 0 else 0
        metrics['avg_loss'] = total_loss / losses if losses > 0 else 0
        metrics['avg_trade'] = metrics['net_profit'] / len(trades)
        
        # Calculate metrics for different time windows
        if returns_df is not None and len(returns_df) > 0:
            # Overall metrics
            metrics['start_date'] = returns_df['date'].iloc[0]
            metrics['end_date'] = returns_df['date'].iloc[-1]
            metrics['starting_equity'] = returns_df['equity'].iloc[0]
            metrics['ending_equity'] = returns_df['equity'].iloc[-1]
            metrics['total_return'] = (returns_df['equity'].iloc[-1] / returns_df['equity'].iloc[0]) - 1
            
            # Returns analysis
            returns = returns_df['returns'].fillna(0).values
            if len(returns) > 0:
                metrics['daily_return_mean'] = np.mean(returns)
                metrics['daily_return_std'] = np.std(returns)
                
                # Annualized metrics (assuming 252 trading days per year)
                metrics['annualized_return'] = (1 + metrics['daily_return_mean']) ** 252 - 1
                metrics['annualized_volatility'] = metrics['daily_return_std'] * np.sqrt(252)
                
                # Sharpe ratio
                excess_return = metrics['daily_return_mean'] - (self.risk_free_rate / 252)
                if metrics['daily_return_std'] > 0:
                    metrics['sharpe_ratio'] = (excess_return / metrics['daily_return_std']) * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0
                    
                # Maximum drawdown
                equity_curve = returns_df['equity'].values
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (peak - equity_curve) / peak
                metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
                
                # Calmar ratio
                if metrics['max_drawdown'] > 0:
                    metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
                else:
                    metrics['calmar_ratio'] = float('inf')
                    
                # Sortino ratio (downside risk only)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        metrics['sortino_ratio'] = (excess_return / downside_std) * np.sqrt(252)
                    else:
                        metrics['sortino_ratio'] = float('inf')
                else:
                    metrics['sortino_ratio'] = float('inf')
                
            # Calculate window-specific metrics
            for window in self.performance_windows:
                if len(returns_df) >= window:
                    window_df = returns_df.iloc[-window:]
                    window_metrics = {}
                    
                    # Basic return
                    window_metrics['return'] = (window_df['equity'].iloc[-1] / window_df['equity'].iloc[0]) - 1
                    
                    # Returns analysis
                    window_returns = window_df['returns'].fillna(0).values
                    
                    if len(window_returns) > 0:
                        window_metrics['daily_return_mean'] = np.mean(window_returns)
                        window_metrics['daily_return_std'] = np.std(window_returns)
                        
                        # Annualized metrics
                        window_metrics['annualized_return'] = (1 + window_metrics['daily_return_mean']) ** 252 - 1
                        window_metrics['annualized_volatility'] = window_metrics['daily_return_std'] * np.sqrt(252)
                        
                        # Sharpe ratio
                        excess_return = window_metrics['daily_return_mean'] - (self.risk_free_rate / 252)
                        if window_metrics['daily_return_std'] > 0:
                            window_metrics['sharpe_ratio'] = (excess_return / window_metrics['daily_return_std']) * np.sqrt(252)
                        else:
                            window_metrics['sharpe_ratio'] = 0
                            
                        # Maximum drawdown
                        equity_curve = window_df['equity'].values
                        peak = np.maximum.accumulate(equity_curve)
                        drawdown = (peak - equity_curve) / peak
                        window_metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
                        
                    metrics[f'{window}d'] = window_metrics
        
        return metrics
    
    def _save_data(self) -> bool:
        """
        Save performance data to disk.
        
        Returns:
            bool: True if successful
        """
        try:
            # Save trade history
            trade_file = os.path.join(self.data_dir, 'trade_history.json')
            with open(trade_file, 'w') as f:
                json.dump(self.trade_history, f)
                
            # Save metrics
            metrics_file = os.path.join(self.data_dir, 'performance_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f)
                
            self.last_save_time = datetime.now()
            logger.info(f"Saved performance data for {len(self.trade_history)} strategies")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
            return False
    
    def _load_data(self) -> bool:
        """
        Load performance data from disk.
        
        Returns:
            bool: True if successful
        """
        try:
            # Load trade history
            trade_file = os.path.join(self.data_dir, 'trade_history.json')
            if os.path.exists(trade_file):
                with open(trade_file, 'r') as f:
                    self.trade_history = json.load(f)
                    
            # Load metrics
            metrics_file = os.path.join(self.data_dir, 'performance_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
                    
            # Rebuild daily returns from trade history
            for strategy_id in self.trade_history:
                self._update_daily_returns(strategy_id)
                
            logger.info(f"Loaded performance data for {len(self.trade_history)} strategies")
            return True
            
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            return False
