#!/usr/bin/env python3
"""
Performance Analyzer Module

This module provides the PerformanceAnalyzer class for tracking and evaluating
trading strategy performance metrics.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PerformanceAnalyzer")

class PerformanceAnalyzer:
    """
    Comprehensive strategy performance analyzer.
    
    This class provides functionality to:
    1. Analyze trade-level metrics
    2. Track performance across market regimes
    3. Measure drawdowns with detailed breakdown
    4. Perform attribution analysis
    """
    
    def __init__(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        equity_curve: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        market_regime_data: Optional[pd.DataFrame] = None,
        journal_path: Optional[str] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the performance analyzer.
        
        Args:
            trade_data: DataFrame containing trade records with fields:
                        - entry_time
                        - exit_time
                        - symbol
                        - strategy
                        - entry_price
                        - exit_price
                        - quantity
                        - direction (1 for long, -1 for short)
                        - pnl
                        - pnl_pct
                        - tags (optional)
            equity_curve: Series containing portfolio equity values indexed by datetime
            benchmark_returns: Optional Series with benchmark returns for comparison
            market_regime_data: DataFrame with market regime classifications
            journal_path: Optional path to trade journal exports
            risk_free_rate: Annual risk-free rate (decimal)
        """
        self.trade_data = trade_data
        self.equity_curve = equity_curve
        self.benchmark_returns = benchmark_returns
        self.market_regime_data = market_regime_data
        self.journal_path = journal_path
        self.risk_free_rate = risk_free_rate
        
        # Calculated metrics
        self.performance_metrics = {}
        self.regime_performance = {}
        self.attribution_results = {}
        self.drawdown_analysis = {}
        
        # If data is provided at initialization, process it
        if trade_data is not None:
            self.process_trade_data()
            
        if equity_curve is not None:
            self.calculate_equity_metrics()
            
        if market_regime_data is not None and trade_data is not None:
            self.analyze_regime_performance()
            
    def load_trade_data(self, file_path: str) -> None:
        """
        Load trade data from a CSV or JSON file.
        
        Args:
            file_path: Path to the trade data file
        """
        try:
            if file_path.endswith('.csv'):
                self.trade_data = pd.read_csv(file_path, parse_dates=['entry_time', 'exit_time'])
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.trade_data = pd.DataFrame(data)
                self.trade_data['entry_time'] = pd.to_datetime(self.trade_data['entry_time'])
                self.trade_data['exit_time'] = pd.to_datetime(self.trade_data['exit_time'])
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
                
            self.process_trade_data()
            logger.info(f"Loaded {len(self.trade_data)} trades from {file_path}")
        except Exception as e:
            logger.error(f"Error loading trade data: {str(e)}")
            raise
    
    def load_equity_curve(self, file_path: str) -> None:
        """
        Load equity curve data from a file.
        
        Args:
            file_path: Path to the equity curve data file (CSV)
        """
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if 'equity' in df.columns:
                self.equity_curve = df['equity']
            else:
                self.equity_curve = df.iloc[:, 0]  # Assume first column is equity
                
            self.calculate_equity_metrics()
            logger.info(f"Loaded equity curve with {len(self.equity_curve)} data points")
        except Exception as e:
            logger.error(f"Error loading equity curve: {str(e)}")
            raise
    
    def process_trade_data(self) -> None:
        """Process trade data and calculate basic statistics."""
        if self.trade_data is None or len(self.trade_data) == 0:
            logger.warning("No trade data available to process")
            return
            
        # Ensure required columns exist
        required_cols = ['entry_time', 'exit_time', 'symbol', 'entry_price', 
                         'exit_price', 'quantity', 'direction']
        missing = [col for col in required_cols if col not in self.trade_data.columns]
        if missing:
            raise ValueError(f"Missing required columns in trade data: {missing}")
            
        # Calculate PnL if not provided
        if 'pnl' not in self.trade_data.columns:
            self.trade_data['pnl'] = (
                self.trade_data['exit_price'] - self.trade_data['entry_price']
            ) * self.trade_data['quantity'] * self.trade_data['direction']
            
        if 'pnl_pct' not in self.trade_data.columns:
            self.trade_data['pnl_pct'] = (
                (self.trade_data['exit_price'] - self.trade_data['entry_price']) / 
                self.trade_data['entry_price'] * self.trade_data['direction']
            )
            
        # Calculate trade duration
        self.trade_data['duration'] = (
            self.trade_data['exit_time'] - self.trade_data['entry_time']
        )
        
        # Ensure strategy column exists
        if 'strategy' not in self.trade_data.columns:
            self.trade_data['strategy'] = 'unknown'
            
        # Calculate basic trade statistics
        self._calculate_trade_metrics()
    
    def _calculate_trade_metrics(self) -> None:
        """Calculate basic trade performance metrics."""
        data = self.trade_data
        
        # Basic metrics
        total_trades = len(data)
        winning_trades = len(data[data['pnl'] > 0])
        losing_trades = len(data[data['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = data['pnl'].sum()
        avg_win = data.loc[data['pnl'] > 0, 'pnl'].mean() if winning_trades > 0 else 0
        avg_loss = data.loc[data['pnl'] <= 0, 'pnl'].mean() if losing_trades > 0 else 0
        avg_pnl = data['pnl'].mean()
        median_pnl = data['pnl'].median()
        
        # Risk-adjusted metrics
        profit_factor = (
            abs(data.loc[data['pnl'] > 0, 'pnl'].sum()) / 
            abs(data.loc[data['pnl'] <= 0, 'pnl'].sum())
        ) if abs(data.loc[data['pnl'] <= 0, 'pnl'].sum()) > 0 else float('inf')
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Expected value per trade
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Duration metrics
        avg_duration = data['duration'].mean()
        median_duration = data['duration'].median()
        
        # Standard deviation and Sharpe ratio of trade returns
        std_returns = data['pnl_pct'].std()
        mean_return = data['pnl_pct'].mean()
        sharpe_ratio = (
            (mean_return - self.risk_free_rate / 252) / std_returns * np.sqrt(252)
        ) if std_returns > 0 else 0
        
        # Store metrics
        self.performance_metrics['trade_metrics'] = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            'avg_duration': avg_duration,
            'median_duration': median_duration,
            'return_std': std_returns,
            'sharpe_ratio': sharpe_ratio
        }
        
        # Calculate per-strategy metrics
        self._calculate_per_strategy_metrics()
        
        # Calculate per-asset metrics
        self._calculate_per_asset_metrics()
        
        # Calculate metrics by time period
        self._calculate_time_period_metrics()
        
    def _calculate_per_strategy_metrics(self) -> None:
        """Calculate performance metrics broken down by strategy."""
        if 'strategy' not in self.trade_data.columns:
            return
            
        strategies = self.trade_data['strategy'].unique()
        strategy_metrics = {}
        
        for strategy in strategies:
            strategy_data = self.trade_data[self.trade_data['strategy'] == strategy]
            
            total_trades = len(strategy_data)
            winning_trades = len(strategy_data[strategy_data['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            strategy_metrics[strategy] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': strategy_data['pnl'].sum(),
                'avg_pnl': strategy_data['pnl'].mean(),
                'profit_factor': (
                    abs(strategy_data.loc[strategy_data['pnl'] > 0, 'pnl'].sum()) / 
                    abs(strategy_data.loc[strategy_data['pnl'] <= 0, 'pnl'].sum())
                ) if abs(strategy_data.loc[strategy_data['pnl'] <= 0, 'pnl'].sum()) > 0 else float('inf')
            }
            
        self.performance_metrics['strategy_metrics'] = strategy_metrics
    
    def _calculate_per_asset_metrics(self) -> None:
        """Calculate performance metrics broken down by asset/symbol."""
        if 'symbol' not in self.trade_data.columns:
            return
            
        symbols = self.trade_data['symbol'].unique()
        symbol_metrics = {}
        
        for symbol in symbols:
            symbol_data = self.trade_data[self.trade_data['symbol'] == symbol]
            
            total_trades = len(symbol_data)
            winning_trades = len(symbol_data[symbol_data['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            symbol_metrics[symbol] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': symbol_data['pnl'].sum(),
                'avg_pnl': symbol_data['pnl'].mean(),
                'profit_factor': (
                    abs(symbol_data.loc[symbol_data['pnl'] > 0, 'pnl'].sum()) / 
                    abs(symbol_data.loc[symbol_data['pnl'] <= 0, 'pnl'].sum())
                ) if abs(symbol_data.loc[symbol_data['pnl'] <= 0, 'pnl'].sum()) > 0 else float('inf')
            }
            
        self.performance_metrics['asset_metrics'] = symbol_metrics
    
    def _calculate_time_period_metrics(self) -> None:
        """Calculate performance metrics by time period (monthly, weekly)."""
        data = self.trade_data.copy()
        
        # Add period columns
        data['exit_month'] = data['exit_time'].dt.to_period('M')
        data['exit_week'] = data['exit_time'].dt.to_period('W')
        
        # Monthly performance
        monthly_perf = data.groupby('exit_month').agg({
            'pnl': 'sum',
            'entry_time': 'count',
            'strategy': 'nunique'
        }).rename(columns={'entry_time': 'trade_count', 'strategy': 'strategy_count'})
        
        monthly_perf['win_rate'] = data.groupby('exit_month').apply(
            lambda x: len(x[x['pnl'] > 0]) / len(x) if len(x) > 0 else 0
        )
        
        # Weekly performance
        weekly_perf = data.groupby('exit_week').agg({
            'pnl': 'sum',
            'entry_time': 'count',
            'strategy': 'nunique'
        }).rename(columns={'entry_time': 'trade_count', 'strategy': 'strategy_count'})
        
        weekly_perf['win_rate'] = data.groupby('exit_week').apply(
            lambda x: len(x[x['pnl'] > 0]) / len(x) if len(x) > 0 else 0
        )
        
        self.performance_metrics['monthly_performance'] = monthly_perf
        self.performance_metrics['weekly_performance'] = weekly_perf
    
    def calculate_equity_metrics(self) -> None:
        """Calculate performance metrics from equity curve."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            logger.warning("No equity curve data available")
            return
            
        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        
        # Annualized metrics (assuming daily data)
        period_length = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = period_length / 365
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility and risk metrics
        daily_std = returns.std()
        ann_volatility = daily_std * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_return = returns - daily_rf
        sharpe = (excess_return.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino = (
            (ann_return - self.risk_free_rate) / downside_deviation
        ) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = abs(ann_return / max_drawdown) if max_drawdown != 0 else float('inf')
        
        # Store equity metrics
        self.performance_metrics['equity_metrics'] = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'period_days': period_length
        }
        
        # Calculate drawdown stats
        self._analyze_drawdowns(returns)
        
    def _analyze_drawdowns(self, returns: pd.Series) -> None:
        """
        Analyze drawdowns in detail.
        
        Args:
            returns: Daily returns series
        """
        # Calculate equity curve if we only have returns
        if self.equity_curve is None:
            equity = (1 + returns).cumprod()
        else:
            equity = self.equity_curve.copy()
            
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown percentage
        drawdown_pct = (equity / running_max) - 1
        
        # Find drawdown periods
        is_drawdown = drawdown_pct < 0
        
        # Find start of each drawdown period
        drawdown_started = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ended = (~is_drawdown & is_drawdown.shift(1).fillna(False))
        
        # Get start and end indices
        drawdown_start_indices = drawdown_started[drawdown_started].index
        drawdown_end_indices = drawdown_ended[drawdown_ended].index
        
        # Handle case where we're still in a drawdown at the end
        if len(drawdown_start_indices) > len(drawdown_end_indices):
            drawdown_end_indices = drawdown_end_indices.append(pd.Index([equity.index[-1]]))
            
        # Create drawdown details
        drawdowns = []
        
        for i, (start_idx, end_idx) in enumerate(zip(drawdown_start_indices, drawdown_end_indices)):
            # Find the bottom of this drawdown
            underwater = drawdown_pct.loc[start_idx:end_idx]
            bottom_idx = underwater.idxmin()
            max_dd = underwater.min()
            
            # Calculate recovery time
            recovery_time = (end_idx - bottom_idx).days if end_idx != equity.index[-1] else np.nan
            drawdown_length = (end_idx - start_idx).days
            
            drawdowns.append({
                'start_date': start_idx,
                'bottom_date': bottom_idx,
                'end_date': end_idx,
                'max_drawdown': max_dd,
                'recovery_days': recovery_time,
                'underwater_days': drawdown_length
            })
            
        # Convert to DataFrame
        self.drawdown_analysis['drawdowns'] = pd.DataFrame(drawdowns)
        
        # Calculate summary statistics
        if len(drawdowns) > 0:
            self.drawdown_analysis['summary'] = {
                'num_drawdowns': len(drawdowns),
                'avg_drawdown': self.drawdown_analysis['drawdowns']['max_drawdown'].mean(),
                'median_drawdown': self.drawdown_analysis['drawdowns']['max_drawdown'].median(),
                'avg_recovery_days': self.drawdown_analysis['drawdowns']['recovery_days'].mean(),
                'avg_underwater_days': self.drawdown_analysis['drawdowns']['underwater_days'].mean(),
            }
        else:
            self.drawdown_analysis['summary'] = {
                'num_drawdowns': 0,
                'avg_drawdown': 0,
                'median_drawdown': 0,
                'avg_recovery_days': 0,
                'avg_underwater_days': 0,
            }
            
    def analyze_regime_performance(self) -> None:
        """Analyze performance metrics specific to different market regimes."""
        if self.market_regime_data is None or self.trade_data is None:
            logger.warning("Missing market regime data or trade data")
            return
            
        # Ensure we have regimes mapped to each trade
        trade_data = self.trade_data.copy()
        
        # Add regime to trades based on exit time
        regime_data = self.market_regime_data.copy()
        
        # Make sure indexes are datetime
        if not isinstance(regime_data.index, pd.DatetimeIndex):
            if 'date' in regime_data.columns:
                regime_data = regime_data.set_index('date')
            else:
                logger.error("Cannot identify date column in regime data")
                return
                
        if 'regime' not in regime_data.columns:
            logger.error("No 'regime' column found in market regime data")
            return
            
        # Map regimes to trades
        trade_dates = trade_data['exit_time'].dt.floor('D')
        
        # Find the regime for each trade date
        regimes = []
        for date in trade_dates:
            # Find closest date in regime data that's not after the trade date
            regime_idx = regime_data.index[regime_data.index <= date]
            if len(regime_idx) > 0:
                closest_date = regime_idx[-1]
                regimes.append(regime_data.loc[closest_date, 'regime'])
            else:
                regimes.append(None)
                
        trade_data['regime'] = regimes
        
        # Only analyze trades with valid regimes
        trade_data = trade_data[trade_data['regime'].notnull()]
        
        # Calculate metrics per regime
        regime_metrics = {}
        
        for regime in trade_data['regime'].unique():
            regime_trades = trade_data[trade_data['regime'] == regime]
            
            total_trades = len(regime_trades)
            winning_trades = len(regime_trades[regime_trades['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            regime_metrics[regime] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': regime_trades['pnl'].sum(),
                'avg_pnl': regime_trades['pnl'].mean(),
                'profit_factor': (
                    abs(regime_trades.loc[regime_trades['pnl'] > 0, 'pnl'].sum()) / 
                    abs(regime_trades.loc[regime_trades['pnl'] <= 0, 'pnl'].sum())
                ) if abs(regime_trades.loc[regime_trades['pnl'] <= 0, 'pnl'].sum()) > 0 else float('inf')
            }
            
        self.regime_performance['regime_metrics'] = regime_metrics
        
        # Analyze strategy performance per regime
        if 'strategy' in trade_data.columns:
            strategy_regime_metrics = {}
            
            for strategy in trade_data['strategy'].unique():
                strategy_metrics = {}
                
                for regime in trade_data['regime'].unique():
                    strategy_regime_trades = trade_data[
                        (trade_data['strategy'] == strategy) & 
                        (trade_data['regime'] == regime)
                    ]
                    
                    total_trades = len(strategy_regime_trades)
                    if total_trades == 0:
                        continue
                        
                    winning_trades = len(strategy_regime_trades[strategy_regime_trades['pnl'] > 0])
                    win_rate = winning_trades / total_trades
                    
                    strategy_metrics[regime] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': win_rate,
                        'total_pnl': strategy_regime_trades['pnl'].sum(),
                        'avg_pnl': strategy_regime_trades['pnl'].mean()
                    }
                    
                strategy_regime_metrics[strategy] = strategy_metrics
                
            self.regime_performance['strategy_regime_metrics'] = strategy_regime_metrics
    
    def perform_attribution_analysis(
        self,
        factors: Dict[str, pd.Series] = None,
        custom_factors: Dict[str, pd.Series] = None
    ) -> Dict[str, float]:
        """
        Perform attribution analysis to determine performance drivers.
        
        Args:
            factors: Dictionary of factor returns series (e.g., market, size, value)
            custom_factors: Dictionary of custom factor returns 
            
        Returns:
            Dictionary of attribution results
        """
        if self.equity_curve is None:
            logger.warning("Equity curve required for attribution analysis")
            return {}
            
        # Calculate strategy returns
        strategy_returns = self.equity_curve.pct_change().dropna()
        
        # If no factors provided, use some basic ones
        if factors is None and self.benchmark_returns is not None:
            factors = {'Market': self.benchmark_returns}
            
        # Merge factors and strategy returns
        if factors:
            combined_data = pd.DataFrame({'Strategy': strategy_returns})
            
            for name, series in factors.items():
                aligned_series = series.reindex(combined_data.index)
                combined_data[name] = aligned_series
                
            # Add custom factors if provided
            if custom_factors:
                for name, series in custom_factors.items():
                    aligned_series = series.reindex(combined_data.index)
                    combined_data[name] = aligned_series
                    
            # Drop rows with NaN values
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 30:
                logger.warning("Insufficient data points for robust attribution analysis")
                return {}
                
            # Perform regression analysis
            X = combined_data.drop('Strategy', axis=1)
            y = combined_data['Strategy']
            
            # Add constant term
            X = sm.add_constant(X)
            
            # OLS regression
            model = sm.OLS(y, X).fit()
            
            # Extract results
            alpha = model.params['const'] * 252  # Annualized
            exposures = model.params.drop('const')
            t_values = model.tvalues.drop('const')
            p_values = model.pvalues.drop('const')
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            
            # Calculate contribution to return
            contributions = {}
            for factor in exposures.index:
                contributions[factor] = exposures[factor] * X[factor].mean() * 252  # Annualized
                
            # Store results
            self.attribution_results = {
                'alpha': alpha,
                'factor_exposures': exposures.to_dict(),
                't_values': t_values.to_dict(),
                'p_values': p_values.to_dict(),
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'return_contributions': contributions
            }
            
            return self.attribution_results
        else:
            logger.warning("No factors provided for attribution analysis")
            return {}
    
    def plot_equity_curve(
        self,
        include_benchmark: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot equity curve with drawdowns.
        
        Args:
            include_benchmark: Whether to include benchmark returns
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.equity_curve is None:
            logger.warning("Equity curve data required for plotting")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        equity_normalized = self.equity_curve / self.equity_curve.iloc[0]
        ax.plot(equity_normalized.index, equity_normalized, label='Strategy', linewidth=2)
        
        # Add benchmark if available
        if include_benchmark and self.benchmark_returns is not None:
            benchmark_curve = (1 + self.benchmark_returns).cumprod()
            benchmark_normalized = benchmark_curve / benchmark_curve.iloc[0]
            
            # Align dates with equity curve
            common_dates = equity_normalized.index.intersection(benchmark_normalized.index)
            if len(common_dates) > 0:
                ax.plot(
                    benchmark_normalized.loc[common_dates].index,
                    benchmark_normalized.loc[common_dates],
                    label='Benchmark',
                    linestyle='--',
                    alpha=0.7
                )
        
        # Highlight drawdowns
        if self.drawdown_analysis and 'drawdowns' in self.drawdown_analysis:
            drawdowns = self.drawdown_analysis['drawdowns']
            
            for _, dd in drawdowns.iterrows():
                if dd['max_drawdown'] < -0.05:  # Only show significant drawdowns
                    ax.axvspan(
                        dd['start_date'],
                        dd['end_date'],
                        alpha=0.2,
                        color='red',
                        label='_nolegend_'
                    )
        
        ax.set_title('Equity Curve')
        ax.set_ylabel('Normalized Value')
        ax.grid(True)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
    
    def plot_drawdown_chart(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot underwater equity curve (drawdowns).
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.equity_curve is None:
            logger.warning("Equity curve data required for plotting drawdowns")
            return None
            
        returns = self.equity_curve.pct_change().dropna()
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(drawdowns.index, 0, drawdowns * 100, color='red', alpha=0.3)
        ax.plot(drawdowns.index, drawdowns * 100, color='red', linewidth=1)
        
        ax.set_title('Drawdown Chart')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        ax.set_ylim(bottom=min(drawdowns.min() * 100 * 1.1, -5))  # Give some extra space
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot monthly returns as a heatmap.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.equity_curve is None:
            logger.warning("Equity curve data required for monthly returns heatmap")
            return None
            
        # Calculate daily returns
        daily_returns = self.equity_curve.pct_change().dropna()
        
        # Convert to monthly returns
        monthly_returns = daily_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create month-year matrix
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).mean().unstack() * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            monthly_pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            linewidths=1,
            ax=ax,
            cbar_kws={'label': 'Monthly Return (%)'}
        )
        
        # Set labels
        ax.set_title('Monthly Returns (%)')
        ax.set_ylabel('Year')
        ax.set_xlabel('Month')
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
    
    def plot_regime_performance(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance by market regime.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.regime_performance or 'regime_metrics' not in self.regime_performance:
            logger.warning("Regime performance data required for plotting")
            return None
            
        regime_metrics = self.regime_performance['regime_metrics']
        
        # Extract key metrics for plotting
        regimes = list(regime_metrics.keys())
        win_rates = [regime_metrics[r]['win_rate'] for r in regimes]
        avg_pnls = [regime_metrics[r]['avg_pnl'] for r in regimes]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot win rates by regime
        ax1.bar(regimes, win_rates, color='skyblue')
        ax1.set_title('Win Rate by Market Regime')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Win Rate')
        ax1.grid(axis='y')
        
        # Plot average PnL by regime
        bars = ax2.bar(regimes, avg_pnls)
        ax2.set_title('Average PnL by Market Regime')
        ax2.set_ylabel('Average PnL')
        ax2.grid(axis='y')
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if avg_pnls[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with all performance metrics
        """
        report = {
            'trade_analysis': self.performance_metrics.get('trade_metrics', {}),
            'equity_analysis': self.performance_metrics.get('equity_metrics', {}),
            'drawdown_analysis': self.drawdown_analysis.get('summary', {}),
            'strategy_analysis': self.performance_metrics.get('strategy_metrics', {}),
            'asset_analysis': self.performance_metrics.get('asset_metrics', {}),
            'regime_analysis': self.regime_performance.get('regime_metrics', {}),
            'attribution_analysis': self.attribution_results
        }
        
        return report
    
    def export_report(self, file_path: str) -> None:
        """
        Export performance report to file.
        
        Args:
            file_path: Path to save the report (JSON)
        """
        report = self.generate_performance_report()
        
        # Convert any non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, pd.Period):
                return str(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Recursive function to handle nested dictionaries
        def convert_dict(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = convert_dict(v)
                elif isinstance(v, list):
                    result[k] = [convert_to_serializable(item) for item in v]
                else:
                    result[k] = convert_to_serializable(v)
            return result
        
        serializable_report = convert_dict(report)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(serializable_report, f, indent=4)
            logger.info(f"Performance report exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            raise

# For attribution analysis
try:
    import statsmodels.api as sm
except ImportError:
    logger.warning("Statsmodels not installed, attribution analysis will be limited")
    sm = None 