"""
Plotting utilities for backtesting visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def plot_equity_curve(equity_data: pd.Series, 
                     benchmark_data: Optional[pd.Series] = None,
                     title: str = "Portfolio Performance",
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot equity curve with optional benchmark comparison.
    
    Args:
        equity_data: Series of portfolio values indexed by date
        benchmark_data: Optional benchmark series indexed by date
        title: Plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot portfolio equity curve
    equity_data.plot(ax=ax, linewidth=2, color='#0066cc', label='Portfolio')
    
    # Plot benchmark if provided
    if benchmark_data is not None:
        benchmark_data.plot(ax=ax, linewidth=1.5, color='#999999', 
                           linestyle='--', label='Benchmark', alpha=0.7)
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Save figure if path provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save equity curve plot: {e}")
    
    return fig

def plot_drawdown(drawdown_data: pd.Series,
                 title: str = "Portfolio Drawdown",
                 figsize: Tuple[int, int] = (10, 4),
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot drawdown chart over time.
    
    Args:
        drawdown_data: Series of drawdown percentages indexed by date
        title: Plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdown
    ax.fill_between(drawdown_data.index, 0, drawdown_data.values * 100,
                   color='#e63946', alpha=0.5)
    ax.plot(drawdown_data.index, drawdown_data.values * 100,
           color='#e63946', linewidth=1)
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Invert y-axis so drawdowns go down
    ax.invert_yaxis()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Save figure if path provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved drawdown plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save drawdown plot: {e}")
    
    return fig

def plot_monthly_returns(returns_data: pd.Series,
                        title: str = "Monthly Returns",
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot monthly returns as a heatmap calendar.
    
    Args:
        returns_data: Series of daily returns indexed by date
        title: Plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Resample to monthly returns
    monthly_returns = returns_data.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Create monthly returns by year
    monthly_returns_table = monthly_returns.groupby(
        [lambda x: x.year, lambda x: x.month]
    ).first().unstack()
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a heatmap
    cmap = plt.cm.RdYlGn  # Red for negative, green for positive
    sns.heatmap(monthly_returns_table * 100, 
               cmap=cmap, 
               annot=True, 
               fmt=".2f", 
               center=0, 
               linewidths=.5, 
               cbar_kws={'label': 'Return (%)'},
               ax=ax)
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    # Set the x-axis labels to month names
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Save figure if path provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved monthly returns plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save monthly returns plot: {e}")
    
    return fig

def plot_strategy_rotation(equity_curve: pd.Series,
                          rotation_points: Dict[str, List[str]],
                          title: str = "Strategy Rotation Performance",
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot equity curve with strategy rotation points highlighted.
    
    Args:
        equity_curve: Series of portfolio values indexed by date
        rotation_points: Dictionary mapping dates to lists of active strategies
        title: Plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    equity_curve.plot(ax=ax, linewidth=2, color='#0066cc', label='Portfolio')
    
    # Add vertical lines for strategy rotation points
    colors = ['#e63946', '#2a9d8f', '#f4a261', '#6a0dad', '#118ab2']
    
    # Sort rotation points by date
    sorted_dates = sorted(rotation_points.keys())
    
    for i, date_str in enumerate(sorted_dates):
        date = pd.to_datetime(date_str)
        color = colors[i % len(colors)]
        
        if date in equity_curve.index:
            ax.axvline(x=date, color=color, linestyle='--', alpha=0.7)
            
            # Add annotation for strategies
            strategies = rotation_points[date_str]
            y_pos = equity_curve[equity_curve.index >= date].iloc[0]
            ax.annotate(', '.join(strategies), 
                       xy=(date, y_pos),
                       xytext=(10, 10),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       rotation=90,
                       fontsize=8,
                       color=color,
                       backgroundcolor='white',
                       alpha=0.8)
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Save figure if path provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved strategy rotation plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save strategy rotation plot: {e}")
    
    return fig

def plot_strategy_comparison(strategy_returns: Dict[str, pd.Series],
                            title: str = "Strategy Comparison",
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple strategy returns.
    
    Args:
        strategy_returns: Dict mapping strategy names to their return series
        title: Plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert returns to cumulative performance
    cumulative_returns = {}
    for strategy, returns in strategy_returns.items():
        cumulative_returns[strategy] = (1 + returns).cumprod()
    
    # Create dataframe from dict
    df = pd.DataFrame(cumulative_returns)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(ax=ax, linewidth=2)
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Save figure if path provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved strategy comparison plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save strategy comparison plot: {e}")
    
    return fig

def plot_rolling_metrics(returns: pd.Series,
                        window: int = 30,
                        metrics: List[str] = ['return', 'volatility', 'sharpe'],
                        title: str = "Rolling Performance Metrics",
                        figsize: Tuple[int, int] = (12, 10),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot rolling performance metrics over time.
    
    Args:
        returns: Series of daily returns indexed by date
        window: Rolling window in days
        metrics: List of metrics to calculate ('return', 'volatility', 'sharpe')
        title: Plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Calculate rolling metrics
    rolling_data = {}
    
    if 'return' in metrics:
        rolling_data['Rolling Return (%)'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
    
    if 'volatility' in metrics:
        rolling_data['Rolling Volatility (%)'] = returns.rolling(window).std() * np.sqrt(252) * 100
    
    if 'sharpe' in metrics:
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_data['Rolling Sharpe'] = rolling_return / rolling_vol
    
    # Create dataframe
    df = pd.DataFrame(rolling_data)
    
    # Create plot with subplots
    fig, axes = plt.subplots(len(rolling_data), 1, figsize=figsize, sharex=True)
    
    # If only one metric, axes will not be an array
    if len(rolling_data) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric, values) in enumerate(df.items()):
        values.plot(ax=axes[i], linewidth=2)
        axes[i].set_title(metric, fontsize=12)
        axes[i].grid(alpha=0.3)
    
    # Format plot
    fig.suptitle(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure if path provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved rolling metrics plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save rolling metrics plot: {e}")
    
    return fig
